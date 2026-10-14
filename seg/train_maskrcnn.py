#!/usr/bin/env python3
"""Train the PERMISSIVE class-agnostic instance-segmentation model: torchvision Mask R-CNN
(ResNet-50-FPN v2, BSD-3-Clause code; COCO-pretrained weights from torchvision), one class "object",
on the synthetic dataset. Run with the simulation stopped (12 GB GPU).

    .venv-seg/bin/python seg/train_maskrcnn.py --data data/perception/synthetic_v1 --hours 2.5
Output: out/seg/maskrcnn/{last.pt, train_log.jsonl}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from seg.data import SegSplit, TorchvisionDataset  # noqa: E402


def build_model(pretrained: bool = True):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    m = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, min_size=720, max_size=1280,
                                 box_detections_per_img=100)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "perception" / "synthetic_v1"))
    ap.add_argument("--hours", type=float, default=2.5, help="wall-clock training budget")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "out" / "seg" / "maskrcnn"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ds = TorchvisionDataset(SegSplit(Path(args.data), "train"), train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=lambda b: tuple(zip(*b)), persistent_workers=True)
    dev = torch.device("cuda")
    model = build_model().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    budget = args.hours * 3600
    steps_per_epoch = len(dl)
    t0, step, epoch = time.monotonic(), 0, 0
    log = (out / "train_log.jsonl").open("w")
    warmup = 500
    done = False
    while not done:
        model.train()
        for images, targets in dl:
            elapsed = time.monotonic() - t0
            if elapsed > budget:
                done = True
                break
            frac = elapsed / budget  # cosine decay over the time budget, linear warm-up
            lr = args.lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + torch.cos(torch.tensor(frac * 3.14159)).item())
            for g in opt.param_groups:
                g["lr"] = lr
            images = [im.to(dev, non_blocking=True) for im in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(images, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch, "elapsed_s": round(elapsed), "lr": lr, "loss": float(loss),
                       **{k: float(v) for k, v in losses.items()}, "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[maskrcnn] e{epoch} step {step}/{steps_per_epoch * (epoch + 1)} {elapsed / 60:.0f} min loss {float(loss):.3f} lr {lr:.4f}", flush=True)
            step += 1
        epoch += 1
        torch.save({"model": model.state_dict(), "epoch": epoch, "step": step}, out / "last.pt")
    torch.save({"model": model.state_dict(), "epoch": epoch, "step": step}, out / "last.pt")
    print(f"[maskrcnn] done: {epoch} epochs ({step} steps) in {(time.monotonic() - t0) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
