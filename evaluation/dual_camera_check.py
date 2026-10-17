"""Checkpoint 2 (evaluation): table + wrist RGB-D observations in one world frame, checked against simulator
ground truth (EVALUATION ONLY: the agent code never reads it). No robot motion.

For each object measured by the wrist camera: its world centre from the wrist camera (FK of the URDF camera
frame), the associated table-camera measurement, and the true centre; plus a pixel-level calibration check
(true centres reprojected into each camera image must fall inside the matching mask).

    (sim running, ROS sourced, ROS_DOMAIN_ID=42)
    .venv-seg/bin/python -m evaluation.dual_camera_check --out out/active/cp2_dual_camera.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Truth:
    def __init__(self, node):
        from std_msgs.msg import String
        self.latest = None
        node.create_subscription(String, "sim/ground_truth/objects", lambda m: setattr(self, "latest", json.loads(m.data)), 10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from perception.detector import YoloDetector
    from agent.ros_bridge import RosBridge
    from agent.tools import Tools
    from perception.world_model import WorldModel
    from sim.scene_objects import OBJECTS

    bridge = RosBridge()
    truth = Truth(bridge.node)  # evaluation subscriber on the same node (the agent's Tools never see it)
    assert bridge.ready(300), "simulator not ready"
    end = time.monotonic() + 30
    while truth.latest is None and time.monotonic() < end:
        time.sleep(0.2)
    det = YoloDetector(str(ROOT / "out/seg/runs/yolo11s_seg_r1/weights/best.pt"))
    out = Path(a.out)
    (out.parent / "cp2_images").mkdir(parents=True, exist_ok=True)
    t = Tools(bridge, det, None, None, lambda *x, **k: None, out.parent / "cp2_images")
    (out.parent / "cp2_images" / "images").mkdir(exist_ok=True)
    t.run_dir = out.parent / "cp2_images"
    rep = {"cameras": {}}
    obs = {}
    for cam in ("table", "wrist"):
        t0 = time.monotonic()
        frame_t0 = bridge.sim_time
        obs[cam] = t.perceive(cam)
        rep["cameras"][cam] = {"seconds": round(time.monotonic() - t0, 2), "objects": len(obs[cam].marks),
                               "requested_at_sim_s": round(frame_t0, 2), "frame_id": obs[cam].frame_id,
                               "pose_world": t._camera_pose(bridge.snapshot(cam)).round(4).tolist()}
    gt = {n: np.array(v["position"]) for n, v in truth.latest["objects"].items()}
    gt_top = {n: gt[n][2] + OBJECTS[n].size[2] / 2 for n in gt}
    world = WorldModel()
    table_ids = {k: world.observe(g, None) for k, g in obs["table"].measured.items()}
    rows = []
    for k, g in obs["wrist"].measured.items():
        c = np.array(g["centre"])
        name = min(gt, key=lambda n: np.hypot(*(gt[n][:2] - c[:2])))  # evaluation only
        tid = world.match(g)  # cross-camera association: the same geometric gates as within one camera
        tg = world.objects[tid].geometry if tid else None
        tk = next((m for m, i in table_ids.items() if i == tid), None)
        rows.append({"wrist_mark": k, "truth_name": name,
                     "wrist_xy_err_mm": round(float(np.hypot(*(c[:2] - gt[name][:2]))) * 1000, 1),
                     "wrist_top_err_mm": round((g["top_z"] - gt_top[name]) * 1000, 1),
                     "table_mark": tk, "associated": tid is not None,
                     "association_correct": None if tg is None else bool(min(gt, key=lambda n: np.hypot(*(gt[n][:2] - np.array(tg["centre"][:2])))) == name),
                     "table_xy_err_mm": None if tg is None else round(float(np.hypot(*(np.array(tg["centre"][:2]) - gt[name][:2]))) * 1000, 1),
                     "wrist_vs_table_xy_mm": None if tg is None else round(float(np.hypot(*(np.array(tg["centre"][:2]) - c[:2]))) * 1000, 1),
                     "wrist_distance_m": round(float(np.linalg.norm(c - np.array(rep["cameras"]["wrist"]["pose_world"])[:3, 3])), 2)})
    # pixel-level calibration check: true centres reprojected into each image must land in the matching mask
    reproj = {}
    for cam in ("table", "wrist"):
        o, T = obs[cam], np.array(rep["cameras"][cam]["pose_world"])
        fx, fy, cx, cy = bridge.snapshot(cam).K
        hits, n = 0, 0
        for name, p in gt.items():
            top = np.array([p[0], p[1], gt_top[name] - 0.002])
            pc = T[:3, :3].T @ (top - T[:3, 3])
            if pc[2] <= 0.1:
                continue
            u, v = int(round(fx * pc[0] / pc[2] + cx)), int(round(fy * pc[1] / pc[2] + cy))
            if not (0 <= u < 1280 and 0 <= v < 720):
                continue
            inside = [m for m, mask in o.marks.items() if mask[v, u]]
            if not inside:
                continue  # not detected / occluded: not a calibration sample
            n += 1
            hits += 1 if any(np.hypot(*(np.array(o.measured[m]["centre"][:2]) - p[:2])) < 0.03 for m in inside) else 0
        reproj[cam] = {"samples": n, "true_top_centre_in_matching_mask": hits}
    rep["reprojection"] = reproj
    rep["wrist_objects"] = rows
    errs = [r["wrist_xy_err_mm"] for r in rows if r["wrist_xy_err_mm"] < 50]
    rep["summary"] = {"wrist_objects": len(rows), "associated": sum(r["associated"] for r in rows),
                      "association_correct": sum(bool(r["association_correct"]) for r in rows),
                      "wrist_xy_err_mm_median": float(np.median(errs)) if errs else None,
                      "table_xy_err_mm_median": float(np.median([r["table_xy_err_mm"] for r in rows if r["table_xy_err_mm"] is not None] or [np.nan]))}
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("cameras", "reprojection", "summary")}, indent=1, default=float))
    for r in rows:
        print(r)
    bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
