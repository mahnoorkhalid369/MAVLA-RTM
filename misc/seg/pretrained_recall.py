#!/usr/bin/env python3
"""How well does an OFF-THE-SHELF (COCO-pretrained) segmentation model find our objects?
Class-agnostic: every predicted mask counts as "object", whatever COCO class it is given (the VLM
does the naming). Recall at mask IoU >= 0.5 per object family, at two confidence thresholds, on
already-rendered images with ground-truth masks. CPU only by default (the GPU stays with the sim).

    .venv-seg/bin/python seg/pretrained_recall.py --model yolo11s-seg.pt --images 100
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import SegSplit  # noqa: E402
from seg.evaluate import iou_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-images", type=int, default=80)
    ap.add_argument("--novel", default=str(ROOT / "data" / "perception" / "synthetic_v1_pilot"))
    ap.add_argument("--data", default=str(ROOT / "data" / "perception" / "synthetic_v1"))
    ap.add_argument("--out", default=str(ROOT / "out" / "seg" / "pretrained_recall.json"))
    args = ap.parse_args()
    from ultralytics import YOLO, settings

    assert settings["sync"] is False
    model = YOLO(args.model)
    sets = []
    train = SegSplit(Path(args.data), "train")
    idx = np.random.default_rng(0).choice(len(train), args.train_images, replace=False)
    sets.append(("train (seen distribution, unseen by this model)", train, idx))
    novel = SegSplit(Path(args.novel), "test_novel")
    sets.append(("test_novel (pilot)", novel, np.arange(len(novel))))
    report = {"model": args.model, "imgsz": args.imgsz, "device": args.device, "iou": 0.5, "sets": {}}
    for label, split, ids in sets:
        stats = {thr: collections.Counter() for thr in (0.25, 0.05)}
        totals = collections.Counter()
        classes = collections.Counter()
        views = collections.Counter()
        times = []
        for i in ids:
            rgb = split.rgb(int(i))
            gt, anns = split.masks(int(i))
            t = time.perf_counter()
            r = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=0.05, retina_masks=True, verbose=False,
                              device=args.device, max_det=100)[0]
            times.append(time.perf_counter() - t)
            masks = (r.masks.data.cpu().numpy() > 0.5) if r.masks is not None else np.zeros((0, *rgb.shape[:2]), bool)
            conf = r.boxes.conf.cpu().numpy() if r.masks is not None else np.zeros(0)
            cls = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()] if r.masks is not None else []
            for c in cls:
                classes[c] += 1
            for a in anns:
                totals[a["family"]] += 1
            views[split.images[int(i)]["view"]] += 1
            for thr in stats:
                keep = [m for m, s in zip(masks, conf) if s >= thr]
                M = iou_matrix(keep, gt)
                hit = M.max(axis=0) >= 0.5 if M.size else np.zeros(len(gt), bool)
                for a, h in zip(anns, hit):
                    stats[thr][a["family"]] += int(h)
                stats[thr]["_detections"] += len(keep)
                stats[thr]["_detections_matching_any"] += int((M.max(axis=1) >= 0.5).sum()) if M.size else 0
        res = {"images": len(ids), "views": dict(views), "objects": sum(totals.values()),
               "seconds_per_image": round(float(np.median(times)), 2), "top_coco_classes": classes.most_common(8)}
        for thr, c in stats.items():
            res[f"conf>={thr}"] = {
                "recall_all": round(sum(c[f] for f in totals) / max(sum(totals.values()), 1), 3),
                "recall_by_family": {f: f"{c[f]}/{n} = {c[f] / n:.2f}" for f, n in sorted(totals.items())},
                "precision": round(c["_detections_matching_any"] / max(c["_detections"], 1), 3),
                "detections": c["_detections"]}
        report["sets"][label] = res
        print(label, json.dumps(res, indent=1), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
