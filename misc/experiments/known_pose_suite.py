#!/usr/bin/env python3
"""The pick-and-place scenario suite (Milestone 4), for Baseline A (ground-truth poses) and
Baseline B (camera poses), with the same cases so the two can be compared directly.

Start from a FRESH simulation (objects at their initial poses): the cases move objects, and later
destinations assume earlier objects have left their spots.

    ./launch/run.sh --headless   (or python -m sim.run_live --headless), fresh
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 experiments/known_pose_suite.py                    # Baseline A
    /usr/bin/python3 experiments/known_pose_suite.py --poses camera     # Baseline B (needs camera images)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.known_pose_pick_place import run_pick_place  # noqa: E402

# (object, onto, to_mm, expected pass?, what it probes, injected error mm)
CASES = [
    ("red_cube", "green_cube", None, True, "stack on a cube", None),
    ("brown_cube_rotated", None, (350, 320), True, "45 deg rotation: grip yaw", None),
    ("yellow_cube", None, (300, 450), True, "floor spot", None),
    ("orange_cube", None, (-250, 420), True, "small 35 mm cube", None),
    ("lime_cube_stacked", None, (-400, 150), True, "take the top of a stack", None),
    ("navy_cube_far", None, (0, 700), True, "far reach edge (0.89 m)", None),
    ("olive_cube_near_base", None, (450, 0), True, "near edge (0.43 m)", None),
    ("tan_box_wide", None, (500, -60), True, "75 mm wide: 12 mm spare", None),
    ("purple_cylinder", None, (500, 250), True, "round object", None),
    ("steel_cube_heavy", None, (-450, -150), True, "3 kg", None),
    ("pink_cube", None, (450, 150), False, "injected 70 mm pose error: must be caught", (0, 70)),
]


def main() -> int:
    from execution.ros_executor import RosExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", choices=["truth", "camera"], default="truth")
    args = ap.parse_args()
    estimate = None
    if args.poses == "camera":
        from perception.classical_detector import TablePoseEstimator

        estimate = TablePoseEstimator()
    ex = RosExecutor()
    if not ex.wait_ready(need_truth=True):
        print("no simulation on this ROS_DOMAIN_ID")
        return 2
    rows = []
    for obj, onto, to, expected, probe, inject in CASES:
        print(f"\n=== {obj} -> {onto or to}  ({probe})", flush=True)
        t0 = time.monotonic()
        rep = run_pick_place(ex, obj, onto, to, args.poses, estimate, inject_error_mm=inject, go_home=False)
        c = rep["checks"]
        rows.append({"object": obj, "probe": probe, "expected": expected, "passed": bool(rep["passed"]),
                     "grasp_state": rep["grasp_state"], "failure_class": rep.get("failure_class", ""),
                     "failure": rep.get("failure", ""), "perception": rep.get("perception", {}).get(obj),
                     "placement": c.get("placement", {}), "secured": c.get("object_secured", {}),
                     "verified": c.get("grasp_verified", {}), "seconds": round(time.monotonic() - t0)})
    ok = sum(r["passed"] == r["expected"] for r in rows)
    print(f"\n{'object':22s} {'expect':6s} {'result':6s} {'grasp state':18s} {'perception xy/yaw':18s} {'place xy mm':11s} {'slip mm':8s} failure")
    for r in rows:
        p = r["perception"]
        perc = f"{p['xy_error_mm']} mm/{p['yaw_error_deg']} deg" if p else "-"
        slip = r["verified"].get("slip_during_carry_mm", "-")
        flag = "" if r["passed"] == r["expected"] else "  <-- UNEXPECTED"
        print(f"{r['object']:22s} {'pass' if r['expected'] else 'fail':6s} {'pass' if r['passed'] else 'fail':6s} "
              f"{r['grasp_state']:18s} {perc:18s} {str(r['placement'].get('xy_error_mm', '-')):11s} {str(slip):8s} "
              f"{(r['failure_class'] + ': ' + r['failure']) if r['failure'] else ''}{flag}")
    print(f"\n{ok}/{len(rows)} cases behaved as expected ({args.poses} poses)")
    out = ROOT / "out" / "experiments" / time.strftime(f"suite_{args.poses}_%Y%m%d_%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"poses": args.poses, "cases": rows, "as_expected": ok, "total": len(rows)}, indent=2))
    print(f"summary: {out.relative_to(ROOT)}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
