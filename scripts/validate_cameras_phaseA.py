#!/usr/bin/env python3
"""Milestone 4, Phase A: are the RGB-D observations valid, and how good is the classical perception?

Against a RUNNING simulation with camera rendering (sim.run_live; needs a working GPU driver):
  A1. both cameras: RGB, depth and camera_info arrive; sizes agree; RGB/depth timestamps agree with
      each other and with /clock; intrinsics match the D435 model (1280x720, 69.4 deg HFOV, centred)
  A2. extrinsics: table camera TF vs control/robot.py table_camera_pose(); wrist camera TF vs our FK
      of the measured joints (camera_color_optical_frame)
  A3. depth: valid fraction, and the floor reads z = 0 when back-projected (sample of floor pixels)
  A4. classical detector (perception/classical_detector.py) on one table-camera frame: every
      catalogue object found / missed / wrongly named, and its position and yaw error against the
      simulator ground truth (ground truth is used for SCORING only)
Writes out/validation/phaseA_<time>.json.

    ./launch/run.sh --headless   (or python -m sim.run_live --headless), after the reboot
    source /opt/ros/jazzy/setup.bash && /usr/bin/python3 scripts/validate_cameras_phaseA.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from control.kinematics import Kinematics  # noqa: E402
from control.robot import table_camera_pose  # noqa: E402
from experiments.known_pose_pick_place import symmetry_of, wrap, yaw_tilt  # noqa: E402
from perception.classical_detector import ClassicalDetector  # noqa: E402
from sim.scene_objects import OBJECTS  # noqa: E402

EXPECTED = {"width": 1280, "height": 720, "fx": 1280 / 2 / np.tan(np.radians(69.4 / 2))}
POS_TOL_MM, ROT_TOL_DEG = 1.0, 0.1


def quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    from execution.ros_executor import RosExecutor

    ex = RosExecutor()
    if not ex.wait_ready(need_truth=True):
        print("[A] no simulation on this ROS_DOMAIN_ID")
        return 2
    msgs: dict = {}
    for cam, prefix in (("table", "table_camera"), ("wrist", "camera")):
        for kind, typ, topic in (("rgb", Image, "color/image_raw"), ("depth", Image, "depth/image_rect_raw"), ("info", CameraInfo, "color/camera_info")):
            ex.node.create_subscription(typ, f"{prefix}/{topic}", lambda m, k=(cam, kind): msgs.__setitem__(k, m), qos_profile_sensor_data)
    tf = Buffer()
    TransformListener(tf, ex.node)
    t_req = ex.sim_time
    end = time.monotonic() + 20
    while time.monotonic() < end and len(msgs) < 6:
        ex.spin(0.1)
    ex.spin(1.0)
    report: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cameras": {}}
    failures = []
    stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9  # noqa: E731
    frames = {}
    for cam in ("table", "wrist"):
        if not all((cam, k) in msgs for k in ("rgb", "depth", "info")):
            failures.append(f"{cam}: no images (is camera rendering working?)")
            print(f"[A1] {cam}: NO IMAGES")
            continue
        rgb_m, d_m, info = msgs[(cam, "rgb")], msgs[(cam, "depth")], msgs[(cam, "info")]
        K = (info.k[0], info.k[4], info.k[2], info.k[5])
        sizes_ok = (rgb_m.height, rgb_m.width) == (d_m.height, d_m.width) == (info.height, info.width) == (EXPECTED["height"], EXPECTED["width"])
        sync = abs(stamp(rgb_m) - stamp(d_m))
        fresh = stamp(d_m) >= t_req - 0.5
        intr_ok = abs(K[0] - EXPECTED["fx"]) < 2 and abs(K[1] - K[0]) < 1 and abs(K[2] - 640) < 2 and abs(K[3] - 360) < 2
        depth = np.frombuffer(d_m.data, np.float32).reshape(d_m.height, d_m.width)
        rgb = np.frombuffer(rgb_m.data, np.uint8).reshape(rgb_m.height, rgb_m.width, -1)[..., :3]
        valid = float(np.mean(np.isfinite(depth) & (depth > 0)))
        frame_id = "table_camera_optical_frame" if cam == "table" else "camera_color_optical_frame"
        if cam == "table":
            T_model = table_camera_pose()
        else:
            T_model = Kinematics(tip="camera_color_optical_frame").fk(ex.q())
        try:
            t = tf.lookup_transform("world", frame_id, Time()).transform
            T_tf = np.eye(4)
            T_tf[:3, :3] = quat_to_R(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
            T_tf[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
            dp = float(np.linalg.norm(T_tf[:3, 3] - T_model[:3, 3]) * 1000)
            dr = float(np.degrees(np.arccos(np.clip((np.trace(T_tf[:3, :3].T @ T_model[:3, :3]) - 1) / 2, -1, 1))))
        except Exception as exc:  # noqa: BLE001
            dp, dr = float("nan"), float("nan")
            print(f"[A2] {cam}: TF lookup failed: {exc}")
        extr_ok = dp < POS_TOL_MM and dr < ROT_TOL_DEG
        # A3: back-project pixels that should see the floor (depth beyond 1 cm of any object) -> z ~ 0
        h, w = depth.shape
        v, u = np.mgrid[0:h:8, 0:w:8]
        z = depth[v, u].astype(float)
        ok = np.isfinite(z) & (z > 0)
        cam_pts = np.stack([(u[ok] - K[2]) / K[0] * z[ok], (v[ok] - K[3]) / K[1] * z[ok], z[ok]], axis=1)
        world = cam_pts @ T_model[:3, :3].T + T_model[:3, 3]
        floor = world[np.abs(world[:, 2]) < 0.01]
        floor_z = float(np.median(floor[:, 2]) * 1000) if len(floor) else float("nan")
        entry = {"sizes_ok": sizes_ok, "rgb_depth_sync_s": round(sync, 4), "fresh": fresh, "intrinsics": [round(k, 2) for k in K],
                 "intrinsics_ok": intr_ok, "depth_valid_fraction": round(valid, 3), "extrinsics_vs_tf_mm": round(dp, 3),
                 "extrinsics_vs_tf_deg": round(dr, 4), "extrinsics_ok": extr_ok, "floor_z_median_mm": round(floor_z, 2),
                 "floor_points": int(len(floor))}
        report["cameras"][cam] = entry
        frames[cam] = (rgb, depth, K, T_model)
        print(f"[A1] {cam}: sizes {'OK' if sizes_ok else 'BAD'}, rgb/depth sync {sync * 1000:.1f} ms, {'fresh' if fresh else 'STALE'}, "
              f"intrinsics {'OK' if intr_ok else 'BAD'} {np.round(K, 1)}, depth valid {valid * 100:.0f} %")
        print(f"[A2] {cam}: model vs TF {dp:.3f} mm / {dr:.4f} deg -> {'OK' if extr_ok else 'MISMATCH'}")
        print(f"[A3] {cam}: floor reads z = {floor_z:+.2f} mm (median of {len(floor)} sampled floor points)")
        if not (sizes_ok and sync < 0.1 and fresh and intr_ok and extr_ok and abs(floor_z) < 3):
            failures.append(f"{cam}: frame checks")

    # A4: classical perception scored against ground truth
    if "table" in frames:
        rgb, depth, K, T = frames["table"]
        det = ClassicalDetector()
        found = det.detect(depth, rgb, K, T)
        poses = det.identify(found)
        truth = ex.ground_truth()
        rows = {}
        for name in OBJECTS:
            t_c = np.array(truth[name]["position"])
            t_yaw, _ = yaw_tilt(truth[name]["quat_wxyz"])
            if name not in poses:
                rows[name] = {"found": False}
                continue
            c, yaw = poses[name]
            rows[name] = {"found": True, "xy_error_mm": round(float(np.linalg.norm(c[:2] - t_c[:2])) * 1000, 1),
                          "z_error_mm": round(float(c[2] - t_c[2]) * 1000, 1),
                          "yaw_error_deg": round(abs(wrap(yaw - t_yaw, symmetry_of(name))), 1)}
        report["perception"] = {"detections": len(found), "objects": rows,
                                "unassigned_detections": [f"{d.colour} {np.round(d.centre_xy * 1000)}" for d in found
                                                          if not any(np.allclose(d.centre_xy, poses[n][0][:2]) for n in poses)]}
        print(f"\n[A4] classical detector: {len(found)} detections, {len(poses)} identified")
        for name, r in rows.items():
            print(f"     {name:22s} " + (f"xy {r['xy_error_mm']:5.1f} mm, z {r['z_error_mm']:+5.1f} mm, yaw {r['yaw_error_deg']:4.1f} deg"
                                         if r["found"] else "NOT FOUND (occluded, out of view, or misnamed)"))
    report["failures"] = failures
    out = ROOT / "out" / "validation" / time.strftime("phaseA_%Y%m%d_%H%M%S.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[A] {'frame checks PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}; report {out.relative_to(ROOT)}")
    ex.node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
