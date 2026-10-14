#!/usr/bin/env python3
"""Grasp-feasibility map for the VT6L + 2F-85: can a complete pick be planned, as a function of the
object's position, yaw and height and the approach direction?

"Feasible" = the whole pick plans: hover, straight descent, close, lift, every motion collision-checked
along its path (robot, gripper, held object vs floor and itself; no other obstacles) and the pads get
enough contact. Each failure is classified by the first stage that failed.
Approach directions: top-down only (side grasps are not implemented in control/skills.py yet).

Outputs out/feasibility/grasp_map_<time>.json (every sample) and .png (feasible share per position).
Regenerate after any change to the robot model, the gripper calibration or the skills.

    /usr/bin/python3 evaluation/grasp_feasibility_map.py            # ~8 min on 16 workers
    /usr/bin/python3 evaluation/grasp_feasibility_map.py --quick    # coarse grid, ~1 min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_planner = None


def _init() -> None:
    global _planner
    from control.collision import CollisionChecker
    from control.kinematics import Kinematics
    from control.skills import SkillPlanner

    _planner = SkillPlanner(Kinematics(), collision=CollisionChecker())


def classify(message: str) -> str:
    for key, label in (("move above", "hover unreachable"), ("descend", "descent"), ("lift", "lift"),
                       ("collision", "collision"), ("pads would touch", "pad contact"), ("across the fingers", "too wide")):
        if key in message:
            return label
    return "other"


def evaluate(sample: tuple) -> dict:
    from control.motion import PlanningError
    from control.robot import ARM_HOME

    r, bearing, yaw, height, width = sample
    a = np.radians(bearing)
    x, y = -r * np.sin(a), r * np.cos(a)  # bearing 0 = straight ahead (+y), + = to the robot's left
    try:
        _planner.pick(ARM_HOME, np.array([x, y, height]), width=width, yaw_deg=yaw, height=height)
        result = "feasible"
    except PlanningError as exc:
        result = classify(str(exc))
    return {"r": r, "bearing": bearing, "x": round(x, 4), "y": round(y, 4), "yaw": yaw, "height": height,
            "width": width, "approach": "top-down", "result": result}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.quick:
        radii, bearings, yaws, heights = np.arange(0.30, 1.01, 0.05), np.arange(-180, 180, 30), [0.0, 45.0], [0.06]
    else:
        radii, bearings, yaws, heights = np.arange(0.30, 1.001, 0.025), np.arange(-180, 180, 15), [0.0, 30.0, 45.0, 60.0], [0.035, 0.06]
    samples = [(round(float(r), 3), float(b), y, h, min(h, 0.06)) for h in heights for y in yaws for b in bearings for r in radii]
    print(f"{len(samples)} samples on {args.workers} workers...", flush=True)
    t0 = time.monotonic()
    with Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(evaluate, samples, chunksize=8)
    secs = time.monotonic() - t0
    counts = Counter(r["result"] for r in rows)
    print(f"done in {secs:.0f} s: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))

    out = ROOT / "out" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / time.strftime("grasp_map_%Y%m%d_%H%M%S")
    meta = {"robot": "assets/vt6l/vt6l.urdf", "approach": ["top-down"], "start_pose": "ARM_HOME",
            "obstacles": "none (floor + self only)", "seconds": round(secs)}
    stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "samples": rows}, indent=1))

    # radial summary per height: feasible share over bearings and yaws
    for h in heights:
        print(f"\nheight {h * 1000:.0f} mm: share of (bearing, yaw) samples feasible, by distance")
        for r in radii:
            sel = [s for s in rows if s["height"] == h and s["r"] == round(float(r), 3)]
            share = np.mean([s["result"] == "feasible" for s in sel])
            print(f"  {r:.3f} m  {'#' * int(round(share * 40)):40s} {share * 100:5.1f} %")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(heights), figsize=(6 * len(heights), 5.5), squeeze=False)
    for ax, h in zip(axes[0], heights):
        sel = [s for s in rows if s["height"] == h]
        keys = sorted({(s["x"], s["y"]) for s in sel})
        share = [np.mean([s["result"] == "feasible" for s in sel if (s["x"], s["y"]) == k]) for k in keys]
        sc = ax.scatter([k[0] for k in keys], [k[1] for k in keys], c=share, cmap="viridis", vmin=0, vmax=1, s=14)
        ax.plot(0, 0, "k^", ms=8)
        ax.set_aspect("equal")
        ax.set_title(f"top-down pick, object {h * 1000:.0f} mm tall\nshare of yaws {yaws} feasible")
        ax.set_xlabel("x (m, robot's right)")
        ax.set_ylabel("y (m, forward)")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    print(f"\nwrote {stem.with_suffix('.json').relative_to(ROOT)} and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
