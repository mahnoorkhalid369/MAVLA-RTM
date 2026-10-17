"""Design-time check of a test scene (evaluation, no simulator): fraction of each object's top-face sample
points that the TABLE camera sees, by ray casting against every other object's bounding box. Written next to
the scenario as ground-truth visibility notes.

    python3 evaluation/scenario_design.py ap_occlusion
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.robot import table_camera_pose  # noqa: E402

ASSETS = json.loads((ROOT / "data/assets/asset_dims.json").read_text())


def boxes(spec):
    out = {}
    for o in spec["objects"]:
        if o["shape"] == "ycb":
            sx, sy, sz = ASSETS[o["asset"]]["size"]
            size = (sx, sz, sy)
        else:
            size = o["size"]
        out[o["name"]] = (np.array([o["xy"][0], o["xy"][1], size[2] / 2]), np.array(size) / 2, np.radians(o.get("yaw_deg", 0)))
    return out


def hits(p0, p1, c, h, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    a, b = R.T @ (p0 - c), R.T @ (p1 - c)
    t0, t1, d = 0.0, 1.0, b - a
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(a[i]) > h[i]:
                return False
            continue
        u, v = (-h[i] - a[i]) / d[i], (h[i] - a[i]) / d[i]
        t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        if t0 > t1:
            return False
    return t1 > 0 and t0 < 0.999


def main(name):
    spec = json.loads((ROOT / f"scenarios/scenes/{name}.json").read_text())
    cam = table_camera_pose()[:3, 3]
    B = boxes(spec)
    vis = {}
    for n, (c, h, yaw) in B.items():
        pts = [c + [x * h[0] * 0.8, y * h[1] * 0.8, h[2] + 0.001] for x in (-1, 0, 1) for y in (-1, 0, 1)]
        seen = [not any(hits(p, cam, c2, h2, y2) for m, (c2, h2, y2) in B.items() if m != n) for p in pts]
        vis[n] = round(sum(seen) / len(seen), 2)
    rep = {"scenario": name, "table_camera_top_face_visible_fraction": vis,
           "distance_from_base_m": {n: round(float(np.hypot(*c[:2])), 2) for n, (c, _, _) in B.items()}}
    out = ROOT / f"out/active/scenarios/{name}_design.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
