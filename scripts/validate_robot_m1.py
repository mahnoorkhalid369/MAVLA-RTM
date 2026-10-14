#!/usr/bin/env python3
"""Milestone 1: does the robot in Isaac Sim match our kinematic model (control/kinematics.py)?

Runs against a RUNNING simulation (sim.run_live) through its ROS 2 interface only, so it uses the
simulator's own physics state, not our code's idea of it:
  A. /joint_states carries the six arm joints (+ finger_joint) under the URDF's names.
  B. Fixed frames: the table camera's TF pose vs control/robot.py table_camera_pose(); one TF tree.
  C. At every test pose: our FK of the MEASURED joint angles vs the simulator's TF pose of each link
     (j1..j6), the wrist camera (camera_link, camera_color_optical_frame) and the gripper base.
     Translation AND rotation; acceptance 1 mm / 0.1 deg (software consistency, not robot accuracy).
     Poses: home, six mixed poses (non-zero J1, J4, J6) and a +20 deg step on each joint alone from a
     reference pose: a wrong axis sign shows up as a large error on exactly that step.
  D. Wrist camera follows J5, not J6: a J6-only step must not move it; a J5-only step must.
  E. Tracking: each pose settles within 0.5 deg (a false collision would block or deflect motion).
  F. Joint limit: a J2 command 5 deg beyond +65 deg must stop at the limit (simulator enforces it).
Writes out/validation/m1_<time>.json; exit code 0 only if everything passed.

Not covered: arm-to-floor / arm-to-object contact (self-collision is disabled in the articulation
by sim/import_vt6l.py); agreement here shows implementation consistency, not that Epson's real
kinematic parameters are right (that needs the controller or CAD).

    ./launch/run.sh --headless          # or: python -m sim.run_live --headless
    source /opt/ros/jazzy/setup.bash && /usr/bin/python3 scripts/validate_robot_m1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from control.kinematics import Kinematics  # noqa: E402
from control.robot import ARM_HOME, ARM_JOINTS, GRIPPER_JOINT, table_camera_pose  # noqa: E402

POS_TOL_MM, ROT_TOL_DEG = 1.0, 0.1  # FK vs simulator
TRACK_TOL_DEG = 0.5
MOVE_SPEED = np.radians(30.0)
MIN_CLEARANCE = 0.15  # m: TCP and wrist stay this high above the floor in every test pose
GRIPPER_FRAME = "World_robotiq_2f85_Robotiq_2F_85_base_link"  # 2F-85 base, fixed to j6_link
GRIPPER_MOUNT_YAW_DEG = 90.0  # sim/scene.py: gripper turned 90 deg about the flange axis
REF_DEG = [10, -20, 30, 20, -40, 30]  # reference pose for the single-joint steps
MIXED_DEG = [
    [0, -5, -50, 45, -5, 0], [45, -20, -30, 0, 30, 90], [-60, -10, -40, -45, -20, -90],
    [90, -30, -20, 90, 45, 180], [-120, -15, -45, 0, -30, 45], [150, -25, 10, -150, 60, -270],
]
FRAMES = [f"j{i}_link" for i in range(1, 7)] + ["camera_link", "camera_color_optical_frame"]


def quat_to_matrix(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """(translation mm, rotation deg) between two 4x4 poses."""
    dp = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000)
    c = (np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1) / 2
    return dp, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


class Sim:
    def __init__(self) -> None:
        self.node = rclpy.create_node("m1_validator")
        self.joints: dict[str, float] = {}
        self.node.create_subscription(JointState, "joint_states", lambda m: self.joints.update(zip(m.name, m.position)), 10)
        self.cmd = self.node.create_publisher(JointState, "joint_command", 10)
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self.node)

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def q(self) -> np.ndarray:
        return np.array([self.joints[j] for j in ARM_JOINTS])

    def move(self, q_goal: np.ndarray, timeout: float = 40.0) -> tuple[bool, float, float]:
        """Command q_goal; (settled, seconds, max joint error deg after settling)."""
        self.cmd.publish(JointState(name=list(ARM_JOINTS), position=[float(v) for v in q_goal],
                                    velocity=[float(MOVE_SPEED)] * len(ARM_JOINTS)))
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            self.spin(0.1)
            if np.degrees(np.max(np.abs(self.q() - q_goal))) < TRACK_TOL_DEG:
                self.spin(1.0)  # settle, then let /tf catch up
                return True, time.monotonic() - start, float(np.degrees(np.max(np.abs(self.q() - q_goal))))
        return False, timeout, float(np.degrees(np.max(np.abs(self.q() - q_goal))))

    def pose(self, frame: str, root: str = "world") -> np.ndarray | None:
        try:
            t = self.tf.lookup_transform(root, frame, Time()).transform
        except Exception:  # noqa: BLE001 - tf2 raises several lookup exception types
            return None
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
        T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
        return T


def main() -> int:
    rclpy.init()
    sim = Sim()
    sim.spin(3.0)
    report: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tolerance": {"mm": POS_TOL_MM, "deg": ROT_TOL_DEG}}
    failures: list[str] = []

    # A. names
    expected = ARM_JOINTS + [GRIPPER_JOINT]
    missing = [j for j in expected if j not in sim.joints]
    report["joint_names"] = {"seen": sorted(sim.joints), "missing": missing}
    print(f"[A] joints on /joint_states: {'OK' if not missing else 'MISSING ' + str(missing)}")
    if missing:
        print("    is the simulation running on this ROS_DOMAIN_ID?")
        return 1

    # B. fixed frames, one tree
    T_cam = sim.pose("table_camera_optical_frame")
    if T_cam is None:
        failures.append("table camera frame not reachable from 'world' (separate TF trees?)")
        print("[B] table camera: NOT in the robot's TF tree")
    else:
        dp, dr = pose_error(T_cam, table_camera_pose())
        ok = dp < POS_TOL_MM and dr < ROT_TOL_DEG
        report["table_camera"] = {"mm": round(dp, 3), "deg": round(dr, 4), "ok": ok}
        print(f"[B] table camera TF vs table_camera_pose(): {dp:.3f} mm, {dr:.4f} deg -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append("table camera extrinsics")

    # C/D/E. poses
    kin = {f: Kinematics(tip=f) for f in FRAMES}
    flange, tcp, wrist = Kinematics(tip="j6_link"), Kinematics(), Kinematics(tip="j5_link")
    poses = [("home", [round(float(v), 3) for v in np.degrees(ARM_HOME)])] + [(f"mixed {i + 1}", d) for i, d in enumerate(MIXED_DEG)]
    poses.append(("reference", REF_DEG))
    for j in range(6):
        step = list(REF_DEG)
        step[j] += 20
        poses.append((f"J{j + 1} +20", step))
    worst: dict[str, list[float]] = {}
    rows, cam_at = [], {}
    for name, deg in poses:
        q_goal = np.radians(deg)
        if np.any(q_goal < flange.lower) or np.any(q_goal > flange.upper):
            failures.append(f"{name}: test pose outside the joint limits")
            continue
        low = min(tcp.fk(q_goal)[2, 3], wrist.fk(q_goal)[2, 3])
        if low < MIN_CLEARANCE:
            failures.append(f"{name}: test pose only {low * 1000:.0f} mm above the floor (skipped)")
            continue
        settled, secs, track = sim.move(q_goal)
        q = sim.q()
        row = {"pose": name, "deg": deg, "settled": settled, "seconds": round(secs, 1), "track_err_deg": round(track, 3), "frames": {}}
        if not settled:
            failures.append(f"{name}: did not settle (max error {track:.2f} deg)")
        checks = {f: kin[f].fk(q) for f in FRAMES}
        checks[GRIPPER_FRAME] = flange.fk(q) @ rot_z(GRIPPER_MOUNT_YAW_DEG)
        for frame, T_model in checks.items():
            T_sim = sim.pose(frame)
            if T_sim is None:
                row["frames"][frame] = "missing on /tf"
                failures.append(f"{name}: {frame} missing on /tf")
                continue
            dp, dr = pose_error(T_model, T_sim)
            row["frames"][frame] = {"mm": round(dp, 3), "deg": round(dr, 4)}
            w = worst.setdefault(frame, [0.0, 0.0])
            w[0], w[1] = max(w[0], dp), max(w[1], dr)
            if dp >= POS_TOL_MM or dr >= ROT_TOL_DEG:
                failures.append(f"{name}: {frame} off by {dp:.2f} mm / {dr:.3f} deg")
        cam_at[name] = sim.pose("camera_color_optical_frame")
        bad = [f for f, e in row["frames"].items() if isinstance(e, dict) and (e["mm"] >= POS_TOL_MM or e["deg"] >= ROT_TOL_DEG)]
        print(f"[C] {name:10s} {str(deg):34s} {'settled' if settled else 'NOT settled'} {secs:4.1f} s, "
              f"track {track:.3f} deg; FK vs sim: {'OK' if not bad else 'MISMATCH ' + ', '.join(bad)}")
        rows.append(row)
    report["poses"] = rows
    report["worst_per_frame"] = {f: {"mm": round(w[0], 3), "deg": round(w[1], 4)} for f, w in worst.items()}
    print("[C] worst error per frame:")
    for f, w in worst.items():
        print(f"      {f:44s} {w[0]:7.3f} mm  {w[1]:7.4f} deg")

    # D. wrist camera follows J5, not J6
    def cam_shift(a: str, b: str) -> tuple[float, float] | None:
        if cam_at.get(a) is None or cam_at.get(b) is None:
            return None
        return pose_error(cam_at[a], cam_at[b])
    j6, j5 = cam_shift("reference", "J6 +20"), cam_shift("reference", "J5 +20")
    ok_d = j6 is not None and j5 is not None and j6[0] < 0.5 and j6[1] < 0.1 and j5[1] > 10.0
    report["wrist_camera_link"] = {"j6_step_shift": j6, "j5_step_shift": j5, "ok": ok_d}
    print(f"[D] wrist camera: J6 step moves it {j6 and f'{j6[0]:.2f} mm / {j6[1]:.2f} deg'}, "
          f"J5 step {j5 and f'{j5[0]:.1f} mm / {j5[1]:.1f} deg'} -> {'follows J5 only: OK' if ok_d else 'WRONG LINK'}")
    if not ok_d:
        failures.append("wrist camera does not follow J5 only")

    # F. joint limit enforced by the simulator
    beyond = np.radians(REF_DEG)
    beyond[1] = flange.upper[1] + np.radians(5.0)
    sim.move(beyond, timeout=8.0)
    j2 = float(np.degrees(sim.q()[1]))
    ok_f = j2 <= np.degrees(flange.upper[1]) + 0.5
    report["j2_limit"] = {"commanded_deg": float(np.degrees(beyond[1])), "reached_deg": round(j2, 3), "ok": ok_f}
    print(f"[F] J2 commanded {np.degrees(beyond[1]):.1f} deg (limit {np.degrees(flange.upper[1]):.1f}): "
          f"reached {j2:.2f} deg -> {'limit enforced' if ok_f else 'LIMIT NOT ENFORCED'}")
    if not ok_f:
        failures.append("J2 limit not enforced by the simulator")

    sim.move(np.asarray(ARM_HOME, float))
    report["failures"] = failures
    report["passed"] = not failures
    out = ROOT / "out" / "validation" / time.strftime("m1_%Y%m%d_%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"[M1] {'PASSED' if not failures else 'FAILED (' + str(len(failures)) + ')'}; report: {out.relative_to(ROOT)}")
    for f in failures:
        print(f"     - {f}")
    sim.node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
