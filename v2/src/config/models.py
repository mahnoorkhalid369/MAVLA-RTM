"""Model files and detector settings (Gemma's settings are in vlm/config.py)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/<package>/)

YOLO_WEIGHTS = ROOT / "out/seg/runs/yolo11s_seg_r1/weights/best.pt"  # YOLO11s-seg, class-agnostic instance masks
DETECTOR_IMGSZ = 1024
DETECTOR_CONF = 0.25
