#!/usr/bin/env python3
"""Train Ultralytics YOLO-seg (AGPL-3.0; COCO-pretrained weights from Ultralytics), one class
"object", on the synthetic dataset's YOLO-seg labels (datagen/verify_dataset.py writes them and
dataset.yaml). Same wall-clock budget as seg/train_maskrcnn.py. Run with the simulation stopped.
Analytics sync is disabled in the Ultralytics settings (sync=False).

    .venv-seg/bin/python seg/train_yolo.py --data data/perception/synthetic_v1 --hours 2.5
Output: out/seg/runs/<name>/weights/{best,last}.pt + results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "perception" / "synthetic_v1"))
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="yolo11s_seg")
    args = ap.parse_args()
    from ultralytics import YOLO, settings

    assert settings["sync"] is False, "Ultralytics analytics sync must be off (yolo settings sync=False)"
    model = YOLO(str(ROOT / "models" / "ultralytics" / args.model) if (ROOT / "models" / "ultralytics" / args.model).exists() else args.model)
    model.train(data=str(Path(args.data) / "dataset.yaml"), imgsz=args.imgsz, batch=args.batch, time=args.hours,
                epochs=200, patience=1000, workers=6, project=str(ROOT / "out" / "seg" / "runs"), name=args.name,
                exist_ok=True, single_cls=True, seed=0, deterministic=False, cos_lr=True, plots=True, val=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
