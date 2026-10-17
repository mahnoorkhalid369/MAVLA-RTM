"""Segmentation generalization (EVALUATION ONLY): the four approaches of evaluation/seg_diagnosis.py, with their
parameters FIXED as committed (3d3cc3b), on frames captured by evaluation/lab_capture_frames.py. Ground truth = the
simulator's instance segmentation of each view + scenario metadata (scoring only).

Per frame and approach:
  containers visible   ground-truth containers with >= 50 visible pixels
  recall               visible containers matched by an instance (IoU >= 0.2 with the VISIBLE GT mask AND measured centre
                       within 30 mm of the true centre)
  precision            matched / (matched + unmatched instances whose pixels are mostly on a container: merges, partial
                       pieces, duplicates)
  IoU, xy error, top error, full-height error (measured height vs the container's true height)
  fixture / decor / robot / background instances: instances whose pixels are mostly on that kind of thing
  seconds              approach runtime (YOLO time reported separately)

    .venv-seg/bin/python evaluation/seg_generalization.py out/lab_seg_frames
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluation.seg_diagnosis as sd  # noqa: E402

DEV_FRAMES = {("easy_s42", "table")}  # B2 was designed on this frame (3d3cc3b): not an independent test frame


def gt_masks(view: Path):
    ins = cv2.imread(str(view / "instance.png"), cv2.IMREAD_UNCHANGED)
    lab = json.loads((view / "instance.json").read_text())
    kinds = {}  # instance id -> (kind, name)
    for i, path in lab.items():
        p = str(path)
        if "/World/Objects/" in p:
            kinds[int(i)] = ("object", p.split("/World/Objects/")[1].split("/")[0])
        elif "/Fixtures/" in p:
            kinds[int(i)] = ("fixture", p.split("/Fixtures/")[1].split("/")[0])
        elif "/Decor/" in p:
            kinds[int(i)] = ("decor", p.split("/Decor/")[1].split("/")[0])
        elif "vt6l" in p or "robotiq" in p.lower():
            kinds[int(i)] = ("robot", "robot")
    return ins, kinds


def evaluate(view: Path, meta: dict, detector) -> dict:
    from perception.geometry import RobotVolume
    from control.collision import CollisionChecker
    from control.kinematics import Kinematics
    from control.robot import gripper_gap
    cam = json.loads((view / "camera.json").read_text())
    K, T, q = tuple(cam["K"]), np.array(cam["T_world_cam"]), np.array(cam["q"])
    rgb = cv2.cvtColor(cv2.imread(str(view / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(view / "depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    ins, kinds = gt_masks(view)
    containers = {o["id"]: o for o in meta["objects"] if o["category"] in ("sample_tube", "sample_vial")}
    settled = meta["settled"]
    gt_vis = {}
    for i, (kind, name) in kinds.items():
        if kind == "object" and name in containers:
            gt_vis[name] = gt_vis.get(name, np.zeros(ins.shape, bool)) | (ins == i)
    gt_vis = {k: m for k, m in gt_vis.items() if m.sum() >= 50}
    robot = RobotVolume(CollisionChecker(), q, gripper_gap(0.0), None, Kinematics(tip="tool0").fk(q))
    t = time.monotonic()
    raw = [d.mask for d in sorted(detector.detect(rgb), key=lambda d: -d.score)]
    yolo_s = round(time.monotonic() - t, 2)
    W = sd.world_points(depth, K, T)
    res = {"containers_total": len(containers), "containers_visible": len(gt_vis), "yolo_raw": len(raw), "yolo_seconds": yolo_s}
    for name, fn in (("A_baseline", lambda: sd.approach_a(raw, depth, K, T, robot)),
                     ("B_protrusion_split", lambda: sd.approach_b(raw, depth, K, T, robot, W)),
                     ("B2_protrusion_split_world_xy", lambda: sd.approach_b2(raw, depth, K, T, robot, W)),
                     ("C_height_map", lambda: sd.approach_c(depth, K, T, robot, W))):
        t = time.monotonic()
        inst = fn()
        secs = round(time.monotonic() - t, 2)
        matched, used = {}, set()
        for gid, gm in gt_vis.items():
            x, y = settled[gid]["position"][:2]
            best = None
            for k, (m, g) in enumerate(inst):
                if k in used:
                    continue
                j = sd.iou(m, gm)
                dxy = float(np.hypot(g["centre"][0] - x, g["centre"][1] - y))
                if j >= sd.MATCH_IOU and dxy <= sd.MATCH_XY and (best is None or j > best[1]):
                    best = (k, j, dxy, g)
            if best:
                k, j, dxy, g = best
                used.add(k)
                h_true = containers[gid]["dimensions"]["height"]
                top_true = settled[gid]["position"][2] + h_true
                matched[gid] = {"iou": round(j, 2), "xy_mm": round(dxy * 1000, 1), "top_mm": round((g["top_z"] - top_true) * 1000, 1),
                                "height_err_mm": round((g["height"] - h_true) * 1000, 1)}
        counts = {"container_extra": 0, "fixture": 0, "decor": 0, "robot": 0, "other_object": 0, "background": 0}
        for k, (m, g) in enumerate(inst):
            if k in used:
                continue
            ids, n = np.unique(ins[m], return_counts=True)
            top = ids[np.argmax(n)] if len(ids) else 0
            kind, nm = kinds.get(int(top), ("background", None))
            if kind == "object":
                counts["container_extra" if nm in containers else "other_object"] += 1
            else:
                counts[kind] = counts.get(kind, 0) + 1
        tp = len(matched)
        res[name] = {"seconds": secs, "instances": len(inst), "matched": tp,
                     "recall": round(tp / len(gt_vis), 3) if gt_vis else None,
                     "precision": round(tp / (tp + counts["container_extra"]), 3) if tp + counts["container_extra"] else None,
                     "mean_iou": round(float(np.mean([v["iou"] for v in matched.values()])), 2) if matched else None,
                     "max_xy_mm": max((v["xy_mm"] for v in matched.values()), default=None),
                     "mean_abs_top_mm": round(float(np.mean([abs(v["top_mm"]) for v in matched.values()])), 1) if matched else None,
                     "mean_height_err_mm": round(float(np.mean([v["height_err_mm"] for v in matched.values()])), 1) if matched else None,
                     "false_positives": counts, "missed": sorted(set(gt_vis) - set(matched)), "matched_detail": matched}
        if name == "B2_protrusion_split_world_xy":  # failure example image
            vis = rgb.copy()
            for m, _ in inst:
                vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
            for gid, gm in gt_vis.items():
                c, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, c, -1, (0, 255, 0) if gid in matched else (255, 0, 0), 2)
            cv2.imwrite(str(view / "B2_overlay.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return res


def main(root: str) -> int:
    from perception.detector import YoloDetector
    from config.models import DETECTOR_CONF, DETECTOR_IMGSZ, YOLO_WEIGHTS
    detector = YoloDetector(str(YOLO_WEIGHTS), DETECTOR_IMGSZ, DETECTOR_CONF)
    rootp = Path(root)
    out = {"params": {"PROTRUDE_m": sd.PROTRUDE, "MIN_PX": sd.MIN_PX, "CELL_m": sd.CELL, "OPEN_m": sd.OPEN,
                      "MATCH_XY": sd.MATCH_XY, "MATCH_IOU": sd.MATCH_IOU}, "frames": {}}
    for cfg in sorted(p for p in rootp.iterdir() if p.is_dir()):
        meta = json.loads((cfg / "metadata.json").read_text())
        for view in ("table", "wrist_home", "wrist_rack"):
            v = cfg / view
            if not (v / "camera.json").exists():
                continue
            r = evaluate(v, meta, detector)
            r["dev_frame"] = (cfg.name, view) in DEV_FRAMES
            out["frames"][f"{cfg.name}/{view}"] = r
            print(f"[gen] {cfg.name}/{view}: visible {r['containers_visible']} | " + " | ".join(
                f"{a[:2]} R {r[a]['recall']} P {r[a]['precision']}" for a in ("A_baseline", "B_protrusion_split",
                                                                              "B2_protrusion_split_world_xy", "C_height_map")),
                  flush=True)
    (rootp / "generalization_results.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
