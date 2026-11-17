#!/usr/bin/env python3
"""Fine-tuned YOLO-seg: recall and precision at mask IoU >= 0.5 per object type, on val, test_novel
and the captured evaluation scene(s), each reported separately; checks the demo pass criteria.

Matching: greedy one-to-one by IoU (highest first, IoU >= 0.5). Recall per type = matched GT / GT.
Precision overall = matched detections / detections; per type, an unmatched detection is charged to
the type of the ground-truth object it overlaps most (or to "background" if none).
Object types in the evaluation scene come from the simulator's scene list (EVALUATION ONLY).

    .venv-seg/bin/python seg/eval_rounds.py --weights out/seg/runs/yolo11s_seg/weights/best.pt --round 1
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import SegSplit  # noqa: E402
from seg.evaluate import iou_matrix  # noqa: E402

CRITERIA = {"recall_box": 0.95, "recall_cylinder": 0.95, "precision": 0.90}  # on today's scene (fixed by the brief)


def family(a: dict) -> str:
    if "family" in a:
        return a["family"]
    from sim.scene_objects import OBJECTS
    return OBJECTS[a["name"]].shape  # evaluation scene: box | cylinder | sphere


def greedy_match(M: np.ndarray, thr: float = 0.5) -> list[tuple[int, int]]:
    pairs, used_d, used_g = [], set(), set()
    if M.size == 0:
        return pairs
    for flat in np.argsort(-M, axis=None):
        d, g = np.unravel_index(flat, M.shape)
        if M[d, g] < thr:
            break
        if d in used_d or g in used_g:
            continue
        pairs.append((int(d), int(g)))
        used_d.add(d)
        used_g.add(g)
    return pairs


def overlay(rgb, masks, hit_gt, gt, path) -> None:
    out = rgb.copy()
    for m in masks:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 255, 0), 2)
    for g, h in zip(gt, hit_gt):
        if not h:  # missed ground truth in red
            cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, (255, 0, 0), 2)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def evaluate_set(model, split: SegSplit, ids, imgsz: int, conf: float, device: str, overlay_dir: Path | None, n_overlays: int):
    gt_n, tp_gt = collections.Counter(), collections.Counter()
    det_tp, det_fp = collections.Counter(), collections.Counter()
    missed = []
    times = []
    for j, i in enumerate(ids):
        rgb = split.rgb(int(i))
        gt, anns = split.masks(int(i))
        t = time.perf_counter()
        r = model.predict(rgb[..., ::-1], imgsz=imgsz, conf=conf, retina_masks=True, verbose=False, device=device, max_det=100)[0]
        times.append(time.perf_counter() - t)
        masks = (r.masks.data.cpu().numpy() > 0.5) if r.masks is not None else np.zeros((0, *rgb.shape[:2]), bool)
        M = iou_matrix(list(masks), gt)
        pairs = greedy_match(M)
        hit = np.zeros(len(anns), bool)
        matched_d = set()
        for d, g in pairs:
            hit[g] = True
            matched_d.add(d)
            det_tp[family(anns[g])] += 1
        for d in range(len(masks)):
            if d in matched_d:
                continue
            if M.size and M[d].max() > 0:
                det_fp[family(anns[int(np.argmax(M[d]))])] += 1
            else:
                det_fp["background"] += 1
        for a, h in zip(anns, hit):
            gt_n[family(a)] += 1
            tp_gt[family(a)] += int(h)
            if not h:
                missed.append({"image": split.images[int(i)]["file_name"], "type": family(a), "area_px": int(a["area"]),
                               "name": a.get("name", a.get("asset"))})
        if overlay_dir is not None and j < n_overlays:
            overlay(rgb, masks, hit, gt, overlay_dir / f"{Path(split.images[int(i)]['file_name']).stem}.jpg")
    n_det = sum(det_tp.values()) + sum(det_fp.values())
    fams = sorted(set(gt_n) | set(det_fp))
    return {
        "images": len(ids), "objects": sum(gt_n.values()), "detections": n_det,
        "recall_all": round(sum(tp_gt.values()) / max(sum(gt_n.values()), 1), 4),
        "precision_all": round(sum(det_tp.values()) / max(n_det, 1), 4),
        "recall_by_type": {f: {"hit": tp_gt[f], "n": gt_n[f], "recall": round(tp_gt[f] / gt_n[f], 4)} for f in fams if gt_n[f]},
        "precision_by_type": {f: round(det_tp[f] / max(det_tp[f] + det_fp[f], 1), 4) for f in fams},
        "false_positives_background": det_fp["background"],
        "missed": missed[:200], "seconds_per_image": round(float(np.median(times)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default=str(ROOT / "data/perception/synthetic_v1"))
    ap.add_argument("--scenes", nargs="+", default=["today"])
    ap.add_argument("--out", default=str(ROOT / "out/seg/eval"))
    a = ap.parse_args()
    from ultralytics import YOLO, settings

    assert settings["sync"] is False
    model = YOLO(a.weights)
    out = Path(a.out) / f"round{a.round}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"weights": a.weights, "round": a.round, "imgsz": a.imgsz, "conf": a.conf, "iou": 0.5, "sets": {}}
    sets = [("val", SegSplit(Path(a.data), "val")), ("test_novel", SegSplit(Path(a.data), "test_novel"))]
    sets += [(f"scene_{s}", SegSplit(ROOT / "data/perception/eval_layouts", s)) for s in a.scenes]
    for name, split in sets:
        od = out / f"overlays_{name}"
        od.mkdir(exist_ok=True)
        res = evaluate_set(model, split, range(len(split)), a.imgsz, a.conf, a.device, od, 12 if name.startswith(("val", "test")) else 999)
        report["sets"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "missed"}, indent=1), flush=True)
    t = report["sets"].get("scene_today")
    if t:
        rb = t["recall_by_type"].get("box", {}).get("recall", 0.0)
        rc = t["recall_by_type"].get("cylinder", {}).get("recall", 0.0)
        report["criteria"] = {"recall_box": rb, "recall_cylinder": rc, "precision": t["precision_all"], "thresholds": CRITERIA,
                              "pass": rb >= CRITERIA["recall_box"] and rc >= CRITERIA["recall_cylinder"] and
                              t["precision_all"] >= CRITERIA["precision"]}
        print("criteria", report["criteria"], flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
