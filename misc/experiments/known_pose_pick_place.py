#!/usr/bin/env python3
"""Deterministic pick and place, no language model (Milestone 4), with measured outcomes.

Pose sources:
  truth   Baseline A: the object poses come from the simulator's ground truth. Separates manipulation
          from perception: a failure here is kinematics, planning, collisions or gripper physics.
  camera  Baseline B: poses from the table camera's RGB-D (perception/classical_detector.py).
          Ground truth is then used ONLY to score the perception and the result.

Grasp states (each needs evidence; none is assumed from "the command returned"):
  contact suspected  the fingers stopped before the commanded position (could also be interference)
  object secured     after the lift the object rose with the gripper and sits at the TCP (ground truth)
  grasp verified     after carrying to the destination it is still at the TCP (no drop, slip measured)
Placement: after release + 3 s settling: position, height, tilt and drift against the intended pose.

    ./launch/run.sh --headless   (or python -m sim.run_live --headless)
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 experiments/known_pose_pick_place.py --object red_cube --onto green_cube
    /usr/bin/python3 experiments/known_pose_pick_place.py --object yellow_cube --to 300 450 --speed 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from control.collision import Box, CollisionChecker  # noqa: E402
from control.kinematics import Kinematics  # noqa: E402
from control.motion import PlanningError  # noqa: E402
from control.robot import ARM_SPEED, GRIPPER_OPEN, TCP_SPEED, gripper_gap  # noqa: E402
from control.skills import GripperStep, SkillParams, SkillPlanner  # noqa: E402
from sim.scene_objects import OBJECTS, obstacle_boxes  # noqa: E402

PLACE_XY_TOL, PLACE_Z_TOL, TILT_TOL_DEG, DRIFT_TOL = 0.010, 0.005, 5.0, 0.001  # m, m, deg, m
CONTACT_DEG = 1.5  # deg: fingers stopping this far before the close target = contact suspected
LIFT_FRACTION = 0.8  # secured: the object rose at least this share of the TCP's rise ...
FOLLOW_TOL = 0.015  # ... and its centre is within this of the TCP horizontally (m)
SLIP_TOL = 0.010  # verified: it moved less than this relative to the TCP during the carry (m)


def yaw_tilt(q_wxyz) -> tuple[float, float]:
    w, x, y, z = q_wxyz
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))  # body z vs world z
    return float(yaw), float(tilt)


def wrap(deg: float, period: float) -> float:
    return (deg + period / 2) % period - period / 2


def symmetry_of(name: str) -> float:
    o = OBJECTS[name]
    return 90.0 if o.shape != "box" or abs(o.size[0] - o.size[1]) < 0.008 else 180.0


def grasp_for(name: str, yaw: float) -> tuple[float, float, float, float]:
    """(gripper yaw, width across the fingers, length, symmetry) for an object at a yaw (deg).
    Fingers close along (-sin g, cos g): g = yaw closes across the object's local y (width = size y)."""
    o = OBJECTS[name]
    sx, sy = o.size[:2]
    if o.shape != "box":
        return 0.0, sx, sx, 90.0
    if abs(sx - sy) < 0.008:
        return wrap(yaw, 90.0), min(sx, sy), max(sx, sy), 90.0
    return (yaw, sy, sx, 180.0) if sy <= sx else (yaw + 90.0, sx, sy, 180.0)


def boxes_from(poses: dict[str, tuple[np.ndarray, float]]) -> list[Box]:
    """Obstacle boxes at the given (believed) poses {name: (centre, yaw deg)} + fixed props."""
    boxes = {b.name: b for b in obstacle_boxes()}
    for name, (centre, yaw) in poses.items():
        if name in boxes:
            a = np.radians(yaw)
            boxes[name] = Box(name, np.asarray(centre, float),
                              np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]), boxes[name].half)
    return list(boxes.values())


def truth_poses(truth: dict) -> dict[str, tuple[np.ndarray, float]]:
    return {n: (np.array(t["position"]), yaw_tilt(t["quat_wxyz"])[0]) for n, t in truth.items()}


def run_pick_place(ex, obj_name: str, onto: str | None = None, to_mm=None, pose_source: str = "truth",
                   estimate_poses=None, speed_scale: float = 1.0, inject_error_mm=None, go_home: bool = True,
                   say=print) -> dict:
    """One pick and place on a running simulation; returns a report dict ("passed", "failure", checks...).

    estimate_poses(ex) -> {name: (centre, yaw deg)} is required for pose_source="camera"."""
    report: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "object": obj_name, "pose_source": pose_source,
                    "speed_scale": speed_scale, "grasp_state": "none", "checks": {}, "phases": []}

    def done(ok: bool, why: str = "", failure_class: str = "") -> dict:
        report.update(passed=ok, failure=why, failure_class=failure_class, safety_events=list(ex.safety_events))
        say(f"[pick-place] {'PASSED' if ok else 'FAILED (' + failure_class + '): ' + why}")
        return report

    truth = ex.ground_truth()
    obj = OBJECTS[obj_name]
    if pose_source == "truth":
        believed = truth_poses(truth)
    else:  # ground truth only scores the estimate here
        believed = estimate_poses(ex)
        perception = {}
        for name, (c, yaw) in believed.items():
            if name in truth:
                t_c, (t_yaw, _) = np.array(truth[name]["position"]), yaw_tilt(truth[name]["quat_wxyz"])
                perception[name] = {"xy_error_mm": round(float(np.linalg.norm(c[:2] - t_c[:2])) * 1000, 1),
                                    "z_error_mm": round(float(c[2] - t_c[2]) * 1000, 1),
                                    "yaw_error_deg": round(abs(wrap(yaw - t_yaw, symmetry_of(name))), 1)}
        report["perception"] = perception
        if obj_name not in believed:
            return done(False, f"perception did not find {obj_name}", "perception")
    if inject_error_mm:
        c, yaw = believed[obj_name]
        believed[obj_name] = (c + [inject_error_mm[0] / 1000, inject_error_mm[1] / 1000, 0.0], yaw)
        report["injected_error_mm"] = list(inject_error_mm)
        say(f"[pick-place] FAILURE INJECTION: {obj_name}'s position shifted by {inject_error_mm} mm")

    centre, yaw = believed[obj_name]
    top = centre + [0, 0, obj.height / 2]
    g_yaw, width, length, sym = grasp_for(obj_name, yaw)
    if to_mm is not None:
        support_top, support = np.array([to_mm[0] / 1000, to_mm[1] / 1000, 0.0]), None
    else:
        if onto not in believed:
            return done(False, f"destination {onto} not found", "perception")
        support_top, support = believed[onto][0] + [0, 0, OBJECTS[onto].height / 2], onto
    target_centre = support_top + [0, 0, obj.height / 2]
    say(f"[pick-place] {obj_name} at ({centre[0] * 1000:.0f}, {centre[1] * 1000:.0f}) mm, yaw {yaw:+.1f} deg -> grip yaw "
        f"{g_yaw:+.1f} deg, {width * 1000:.0f} mm; to {'on ' + support if support else 'the floor'} at "
        f"({support_top[0] * 1000:.0f}, {support_top[1] * 1000:.0f}, {support_top[2] * 1000:.0f}) mm; speed x{speed_scale}")

    # ------------------------------------------------------------- plan everything first
    params = SkillParams(joint_speed=ARM_SPEED * speed_scale, tcp_speed=TCP_SPEED * speed_scale)
    skills = SkillPlanner(Kinematics(), params, collision=CollisionChecker(boxes_from(believed)))
    kin = skills.kin
    t0 = time.monotonic()
    try:
        skills.check_width(width)
        hover, q, g_used = skills.hover(ex.q(), top, g_yaw, sym)
        approach, lift, q, held = skills.grasp(q, top, g_used, obj_name, width, length, obj.on, obj.height)
        place, q = skills.place(q, held, support_top)
        home, _ = skills.home(q)
    except PlanningError as exc:
        return done(False, str(exc), "planning")
    carry, rest = place[:1], place[1:]
    report["planning_seconds"] = round(time.monotonic() - t0, 2)
    report["grasp"] = {"gripper_yaw_deg": round(g_used, 1), "width_mm": round(width * 1000, 1)}

    def run(label: str, steps) -> bool:
        res = ex.execute(steps)
        report["phases"].append({"phase": label, "ok": res.ok, "log": res.log, "steps": res.steps, "failure": res.reason})
        for line in res.log:
            say(f"    [{label}] {line}")
        return res.ok

    def obj_rel_tcp() -> np.ndarray:
        o = np.array(ex.ground_truth(ex.sim_time)[obj_name]["position"])
        return o - kin.fk(ex.q())[:3, 3]

    # ------------------------------------------------------------- contact suspected
    if not run("approach", hover + approach[:1]) or not run("close", approach[1:]):
        return done(False, report["phases"][-1]["failure"], "execution")
    finger, target = ex.gripper(), approach[1].position
    short = float(np.degrees(target - finger))
    contact = short >= CONTACT_DEG
    report["checks"]["contact_suspected"] = {"finger_deg": round(float(np.degrees(finger)), 2),
                                             "target_deg": round(float(np.degrees(target)), 2),
                                             "stopped_short_deg": round(short, 2), "pad_gap_mm": round(gripper_gap(finger) * 1000, 1),
                                             "ok": bool(contact)}
    say(f"[pick-place] close: stopped {short:.1f} deg before the target -> "
        f"{'contact suspected' if contact else 'reached the target: nothing between the pads'}")
    if not contact:
        run("recover", [GripperStep(GRIPPER_OPEN, "open gripper", 0.5)] + lift)
        return done(False, "the fingers closed on nothing", "grasp")
    report["grasp_state"] = "contact suspected"

    # ------------------------------------------------------------- object secured (lift)
    before, tcp_before = np.array(ex.ground_truth(ex.sim_time)[obj_name]["position"]), kin.fk(ex.q())[:3, 3]
    if not run("lift", lift):
        return done(False, report["phases"][-1]["failure"], "execution")
    rel_lift = obj_rel_tcp()
    after, tcp = np.array(ex.ground_truth(ex.sim_time)[obj_name]["position"]), kin.fk(ex.q())[:3, 3]
    rise, tcp_rise, offset = float(after[2] - before[2]), float(tcp[2] - tcp_before[2]), float(np.linalg.norm(rel_lift[:2]))
    secured = rise >= LIFT_FRACTION * tcp_rise and offset < FOLLOW_TOL
    report["checks"]["object_secured"] = {"object_rise_mm": round(rise * 1000, 1), "tcp_rise_mm": round(tcp_rise * 1000, 1),
                                          "slip_during_lift_mm": round((tcp_rise - rise) * 1000, 1),
                                          "xy_offset_from_tcp_mm": round(offset * 1000, 1), "ok": bool(secured)}
    say(f"[pick-place] lift: object rose {rise * 1000:.0f} of {tcp_rise * 1000:.0f} mm, {offset * 1000:.1f} mm from the TCP -> "
        f"{'object secured' if secured else 'NOT HELD'}")
    if not secured:
        return done(False, "the object did not rise with the gripper", "grasp")
    report["grasp_state"] = "object secured"

    # ------------------------------------------------------------- grasp verified (after the carry)
    finger_lift = ex.gripper()
    if not run("carry", carry):
        return done(False, report["phases"][-1]["failure"], "execution")
    rel_carry = obj_rel_tcp()
    slip = float(np.linalg.norm(rel_carry - rel_lift))
    verified = slip < SLIP_TOL
    report["checks"]["grasp_verified"] = {"slip_during_carry_mm": round(slip * 1000, 1),
                                          "finger_drift_deg": round(float(np.degrees(ex.gripper() - finger_lift)), 2),
                                          "ok": bool(verified)}
    say(f"[pick-place] carry: object moved {slip * 1000:.1f} mm relative to the TCP -> {'grasp verified' if verified else 'SLIPPED / DROPPED'}")
    if not verified:
        return done(False, "the object slipped or dropped during the carry", "grasp")
    report["grasp_state"] = "grasp verified"

    # ------------------------------------------------------------- place, settle, verify
    if not run("place", rest):
        return done(False, report["phases"][-1]["failure"], "execution")
    ex.wait_sim(2.0)
    p1 = np.array(ex.ground_truth(ex.sim_time)[obj_name]["position"])
    ex.wait_sim(1.0)
    final = ex.ground_truth(ex.sim_time)[obj_name]
    p2 = np.array(final["position"])
    _, tilt = yaw_tilt(final["quat_wxyz"])
    xy_err, z_err, drift = float(np.linalg.norm(p2[:2] - target_centre[:2])), float(p2[2] - target_centre[2]), float(np.linalg.norm(p2 - p1))
    placed = xy_err <= PLACE_XY_TOL and abs(z_err) <= PLACE_Z_TOL and tilt <= TILT_TOL_DEG and drift <= DRIFT_TOL
    # also against where it should be in the WORLD (true support pose), not only the believed one
    true_target = (np.array(truth[onto]["position"]) + [0, 0, (OBJECTS[onto].height + obj.height) / 2]) if onto else target_centre
    report["checks"]["placement"] = {"xy_error_mm": round(xy_err * 1000, 1), "z_error_mm": round(z_err * 1000, 1),
                                     "tilt_deg": round(tilt, 2), "drift_mm_over_1s": round(drift * 1000, 2),
                                     "xy_error_vs_true_destination_mm": round(float(np.linalg.norm(p2[:2] - true_target[:2])) * 1000, 1),
                                     "final_position_mm": [round(v * 1000, 1) for v in p2], "ok": bool(placed)}
    say(f"[pick-place] placed: xy {xy_err * 1000:.1f} mm, height {z_err * 1000:+.1f} mm, tilt {tilt:.1f} deg, drift "
        f"{drift * 1000:.2f} mm/s -> {'OK' if placed else 'NOT AT THE DESTINATION'}")
    if go_home:
        run("home", home)
    return done(placed, "" if placed else "placement outside tolerance", "" if placed else "placement")


def main() -> int:
    from execution.ros_executor import RosExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="red_cube")
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument("--onto", default="green_cube", help="put it on top of this object")
    dest.add_argument("--to", nargs=2, type=float, metavar=("X_MM", "Y_MM"), help="put it on the floor here")
    ap.add_argument("--speed", type=float, default=1.0, help="scale the joint and TCP speeds")
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--inject-error-mm", nargs=2, type=float, metavar=("DX", "DY"), default=None,
                    help="FAILURE INJECTION: shift the object's believed position, as a perception error would")
    args = ap.parse_args()

    ex = RosExecutor()
    if not ex.wait_ready(need_truth=True):
        print("[pick-place] no /joint_states, /clock or ground truth: is sim.run_live running on this ROS_DOMAIN_ID?")
        return 2
    report = run_pick_place(ex, args.object, None if args.to else args.onto, args.to, "truth", None, args.speed,
                            args.inject_error_mm, not args.no_home)
    out = ROOT / "out" / "experiments" / time.strftime("known_pose_%Y%m%d_%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[pick-place] report {out.relative_to(ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
