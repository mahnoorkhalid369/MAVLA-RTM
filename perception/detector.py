"""Detector interface: class-agnostic instance masks. In-process YOLO-seg now; the same interface can
later be served by a separate inference process without changing the agents.

    det = YoloDetector("out/seg/runs/yolo11s_seg_r1/weights/best.pt")
    masks = det.detect(rgb)        # [Detection(mask HxW bool, score)]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    mask: np.ndarray  # H x W bool
    score: float


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    """Ultralytics YOLO-seg (AGPL-3.0: accepted for the demo only, DECISIONS.md 2026-10-14)."""

    def __init__(self, weights: str, imgsz: int = 1024, conf: float = 0.25, device: str = "cuda:0") -> None:
        from ultralytics import YOLO, settings

        assert settings["sync"] is False, "Ultralytics analytics sync must be off"
        self.model, self.imgsz, self.conf, self.device = YOLO(weights), imgsz, conf, device

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        r = self.model.predict(rgb[..., ::-1], imgsz=self.imgsz, conf=self.conf, retina_masks=True, verbose=False,
                               device=self.device, max_det=100)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        return [Detection(m, float(s)) for m, s in zip(masks, r.boxes.conf.cpu().numpy())]
