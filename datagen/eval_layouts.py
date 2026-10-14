"""Re-arranged EVALUATION layouts of the 21 evaluation objects (sim/scene_objects.py): new positions,
yaws and stacks, seeded with seeds disjoint from the training data (datagen/config_v1.json seed
20261014 hashes; these use 900000 + layout index). Used for evaluation only, never for training.

numpy only; datagen/eval_scene_capture.py applies a layout to the simulated scene.

    poses = rearranged_layout(3)   # {name: {"xy": (x, y), "yaw_deg": d, "on": name | None}}
"""

from __future__ import annotations

import numpy as np

from control.robot import table_camera_pose
from sim.scene_objects import OBJECTS

LAYOUT_SEED_BASE = 900000
R_MIN, R_MAX = 0.30, 1.15  # m from the base (outside the base, inside the table camera's view)
GAP = 0.02  # m between footprints
STACK_P = 0.15
W, H, HFOV = 1280, 720, 69.4


def _in_view(xy, z=0.03, margin=40) -> bool:
    T = table_camera_pose()
    c = T[:3, :3].T @ (np.array([xy[0], xy[1], z]) - T[:3, 3])
    if c[2] <= 0.05:
        return False
    f = W / 2 / np.tan(np.radians(HFOV / 2))
    u, v = f * c[0] / c[2] + W / 2, f * c[1] / c[2] + H / 2
    return margin <= u <= W - margin and margin <= v <= H - margin


def _radius(name: str) -> float:
    sx, sy = OBJECTS[name].size[:2]
    return 0.5 * float(np.hypot(sx, sy))


def rearranged_layout(index: int) -> dict[str, dict]:
    rng = np.random.default_rng(LAYOUT_SEED_BASE + index)
    names = list(OBJECTS)
    order = [names[i] for i in rng.permutation(len(names))]
    placed: dict[str, dict] = {}
    for name in order:
        o = OBJECTS[name]
        supports = [n for n, p in placed.items() if p["on"] is None and OBJECTS[n].shape == "box"
                    and min(OBJECTS[n].size[:2]) >= max(o.size[:2]) and OBJECTS[n].height <= 0.06]
        if o.shape == "box" and supports and rng.random() < STACK_P:
            s = supports[int(rng.integers(len(supports)))]
            placed[name] = {"xy": placed[s]["xy"], "yaw_deg": float(rng.uniform(-45, 45)), "on": s}
            continue
        for _ in range(500):
            r, a = rng.uniform(R_MIN, R_MAX), rng.uniform(-np.pi, np.pi)
            xy = (float(r * np.cos(a)), float(r * np.sin(a)))
            if not _in_view(xy):
                continue
            if all(np.hypot(xy[0] - p["xy"][0], xy[1] - p["xy"][1]) >= _radius(name) + _radius(n) + GAP
                   for n, p in placed.items() if p["on"] is None):
                placed[name] = {"xy": xy, "yaw_deg": float(rng.uniform(-90, 90)), "on": None}
                break
        else:
            raise RuntimeError(f"layout {index}: no free spot for {name}")
    return placed


def base_height(layout: dict[str, dict], name: str) -> float:
    """Height of the surface the object stands on in this layout."""
    on = layout[name]["on"]
    return 0.0 if on is None else base_height(layout, on) + OBJECTS[on].height
