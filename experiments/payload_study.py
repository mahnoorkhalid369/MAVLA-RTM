#!/usr/bin/env python3
"""Payload repeatability study (simulation): the same pick and place repeated at several speeds with a
light and a heavy block, recording joint tracking error, settling, object slip, gripper contact
stability and placement error.

Caveat: the VT6L model's inertias and joint effort limits are NOT manufacturer values (see
assets/vt6l/vt6l.urdf), so this measures the behaviour of the current simulated dynamics, not the
real robot's payload performance.

Each block shuttles between two floor spots; start from a fresh simulation.
    ./launch/run.sh --headless   (or python -m sim.run_live --headless), fresh
    source /opt/ros/jazzy/setup.bash && /usr/bin/python3 experiments/payload_study.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.known_pose_pick_place import run_pick_place  # noqa: E402
from sim.scene_objects import OBJECTS  # noqa: E402

SPEEDS = (0.5, 1.0, 2.0)
REPEATS = 3
# block -> (spot A = its start, spot B) in mm
SHUTTLES = {"red_cube": ((0, 580), (270, 620)), "steel_cube_heavy": ((-250, -500), (-450, -150))}


def main() -> int:
    from execution.ros_executor import RosExecutor

    ex = RosExecutor()
    if not ex.wait_ready(need_truth=True):
        print("no simulation on this ROS_DOMAIN_ID")
        return 2
    rows = []
    quiet = lambda *_: None  # noqa: E731
    for name, (a, b) in SHUTTLES.items():
        at_b = False
        for speed in SPEEDS:
            for rep in range(REPEATS):
                dest = a if at_b else b
                rep_ = run_pick_place(ex, name, None, dest, "truth", None, speed, go_home=False, say=quiet)
                at_b = not at_b if rep_["passed"] else at_b
                c = rep_["checks"]
                steps = [s for ph in rep_["phases"] for s in ph.get("steps", [])]
                track = max((s["max_tracking_deg"] for s in steps), default=float("nan"))
                settle = max((s["settled_deg"] for s in steps), default=float("nan"))
                row = {"object": name, "mass_kg": round(OBJECTS[name].mass, 3), "speed": speed, "rep": rep + 1,
                       "passed": rep_["passed"], "failure": rep_.get("failure", ""),
                       "max_tracking_deg": round(track, 3), "max_settled_deg": round(settle, 3),
                       "slip_lift_mm": c.get("object_secured", {}).get("slip_during_lift_mm"),
                       "slip_carry_mm": c.get("grasp_verified", {}).get("slip_during_carry_mm"),
                       "finger_drift_deg": c.get("grasp_verified", {}).get("finger_drift_deg"),
                       "place_xy_mm": c.get("placement", {}).get("xy_error_mm"),
                       "place_z_mm": c.get("placement", {}).get("z_error_mm")}
                rows.append(row)
                print(f"{name:17s} x{speed:<3} #{rep + 1}: {'ok  ' if row['passed'] else 'FAIL'} track {row['max_tracking_deg']:5.2f} deg, "
                      f"settle {row['max_settled_deg']:5.3f} deg, slip lift {row['slip_lift_mm']} / carry {row['slip_carry_mm']} mm, "
                      f"finger drift {row['finger_drift_deg']} deg, place {row['place_xy_mm']} mm {row['failure']}", flush=True)

    print(f"\n{'object':17s} {'speed':5s} {'ok':4s} {'track deg (max)':16s} {'settle deg':11s} {'slip lift mm':13s} {'slip carry':11s} {'place xy mm':11s}")
    for name in SHUTTLES:
        for speed in SPEEDS:
            sel = [r for r in rows if r["object"] == name and r["speed"] == speed]
            ok = sum(r["passed"] for r in sel)
            stat = lambda k: f"{np.nanmean([r[k] for r in sel if r[k] is not None]):.2f}±{np.nanstd([r[k] for r in sel if r[k] is not None]):.2f}" if any(r[k] is not None for r in sel) else "-"  # noqa: E731
            print(f"{name:17s} x{speed:<4} {ok}/{len(sel)}  {stat('max_tracking_deg'):16s} {stat('max_settled_deg'):11s} "
                  f"{stat('slip_lift_mm'):13s} {stat('slip_carry_mm'):11s} {stat('place_xy_mm'):11s}")
    out = ROOT / "out" / "experiments" / time.strftime("payload_%Y%m%d_%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"caveat": "simulated dynamics with unverified VT6L inertia / effort limits", "runs": rows}, indent=2))
    print(f"\nreport: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
