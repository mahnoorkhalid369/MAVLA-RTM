#!/usr/bin/env python3
"""Evaluate class-agnostic instance segmentation models on the same images, the same way.

Both models are wrapped as `predict(rgb) -> [(mask HxW bool, score)]`. Metrics per split:
  COCO mask AP, AP50, AP75, AR@100 and AP for small / medium / large objects (pycocotools);
  recall / precision at mask IoU >= 0.5 (score >= 0.5, greedy one-to-one matching by IoU);
  mean IoU of the matched masks; latency per image (batch 1, after warm-up); peak GPU memory.
Splits are reported separately (val = training distribution, test_novel = shapes never trained on).

    .venv-seg/bin/python seg/evaluate.py --data data/perception/synthetic_v1 \\
        --maskrcnn out/seg/maskrcnn/last.pt --yolo out/seg/runs/yolo11s_seg/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import SegSplit  # noqa: E402

SCORE_THR, IOU_THR = 0.5, 0.5


class MaskRCNNPredictor:
    name = "maskrcnn_r50_fpn_v2 (torchvision, BSD-3)"

    def __init__(self, ckpt: Path) -> None:
        from seg.train_maskrcnn import build_model

        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.eval().cuda()

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        x = torch.as_tensor(rgb).permute(2, 0, 1).float().cuda() / 255.0
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model([x])[0]
        masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
        return list(zip(masks, out["scores"].float().cpu().numpy()))


class YoloPredictor:
    name = "YOLO11s-seg (Ultralytics, AGPL-3.0)"

    def __init__(self, weights: Path, imgsz: int = 1024) -> None:
        from ultralytics import YOLO

        self.model, self.imgsz = YOLO(str(weights)), imgsz

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        r = self.model.predict(rgb[..., ::-1], imgsz=self.imgsz, conf=0.05, retina_masks=True, verbose=False, max_det=100)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        return list(zip(masks, r.boxes.conf.cpu().numpy()))


def iou_matrix(pred: list[np.ndarray], gt: np.ndarray) -> np.ndarray:
    if not pred or not len(gt):
        return np.zeros((len(pred), len(gt)))
    P = np.stack(pred).reshape(len(pred), -1).astype(np.float32)
    G = gt.reshape(len(gt), -1).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def evaluate(predictor, split: SegSplit, limit: int | None = None) -> dict:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    n = len(split) if limit is None else min(limit, len(split))
    dets, lat, tp, fp, fn, ious = [], [], 0, 0, 0, []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n):
        rgb = split.rgb(i)
        gt, _ = split.masks(i)
        if i == 0:
            predictor(rgb)  # warm-up
        torch.cuda.synchronize()
        t = time.perf_counter()
        preds = predictor(rgb)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
        image_id = split.images[i]["id"]
        for m, s in preds:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            dets.append({"image_id": image_id, "category_id": 1, "segmentation": rle, "score": float(s)})
        keep = [m for m, s in preds if s >= SCORE_THR]
        M = iou_matrix(keep, gt)
        matched_p, matched_g = set(), set()
        for k in np.argsort(-M, axis=None):
            p, g = np.unravel_index(k, M.shape)
            if M[p, g] < IOU_THR:
                break
            if p in matched_p or g in matched_g:
                continue
            matched_p.add(p)
            matched_g.add(g)
            ious.append(M[p, g])
        tp += len(matched_g)
        fp += len(keep) - len(matched_p)
        fn += len(gt) - len(matched_g)
    coco_gt = COCO(str(split.coco_path))
    img_ids = [im["id"] for im in split.images[:n]]
    res = {"images": n, "recall@0.5": round(tp / max(tp + fn, 1), 4), "precision@0.5": round(tp / max(tp + fp, 1), 4),
           "mean_iou_matched": round(float(np.mean(ious)) if ious else 0.0, 4),
           "latency_ms_median": round(1000 * float(np.median(lat)), 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "tp": tp, "fp": fp, "fn": fn}
    if dets:
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.imgIds = img_ids
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        st = ev.stats
        res.update({"mask_AP": round(st[0], 4), "mask_AP50": round(st[1], 4), "mask_AP75": round(st[2], 4),
                    "AP_small": round(st[3], 4), "AP_medium": round(st[4], 4), "AP_large": round(st[5], 4), "AR@100": round(st[8], 4)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "perception" / "synthetic_v1"))
    ap.add_argument("--maskrcnn", default=None)
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--yolo-imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "out" / "seg" / "comparison.json"))
    args = ap.parse_args()
    models = []
    if args.maskrcnn:
        models.append(lambda: MaskRCNNPredictor(Path(args.maskrcnn)))
    if args.yolo:
        models.append(lambda: YoloPredictor(Path(args.yolo), args.yolo_imgsz))
    report = {}
    for make in models:
        pred = make()
        report[pred.name] = {s: evaluate(pred, SegSplit(Path(args.data), s), args.limit) for s in ("val", "test_novel")}
        print(pred.name, json.dumps(report[pred.name], indent=1), flush=True)
        del pred
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
