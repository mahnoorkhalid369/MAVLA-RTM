"""Synthetic instance-segmentation data (datagen/ output) for training and evaluation.

Reads the COCO master file and the per-image instance-id PNGs (exact masks; uid = object id). Only
objects with a COCO annotation (>= the dataset's minimum visible pixels) are instances.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class SegSplit:
    def __init__(self, root: Path, split: str) -> None:
        self.dir = Path(root) / split
        coco = json.loads((self.dir / "annotations.json").read_text())
        self.images = coco["images"]
        self.by_image: dict[int, list[dict]] = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            self.by_image[a["image_id"]].append(a)
        self.coco_path = self.dir / "annotations.json"

    def __len__(self) -> int:
        return len(self.images)

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.dir / self.images[i]["file_name"])), cv2.COLOR_BGR2RGB)

    def masks(self, i: int) -> tuple[np.ndarray, list[dict]]:
        """(N x H x W bool masks, their annotations) for image i."""
        im = self.images[i]
        ids = cv2.imread(str(self.dir / "masks" / f"{Path(im['file_name']).stem}.png"), cv2.IMREAD_UNCHANGED)
        anns = self.by_image[im["id"]]
        if not anns:
            return np.zeros((0, *ids.shape), bool), []
        return np.stack([ids == a["object_uid"] for a in anns]), anns


class TorchvisionDataset:
    """torch Dataset for torchvision detection models: (image tensor, target dict)."""

    def __init__(self, split: SegSplit, train: bool) -> None:
        self.split, self.train = split, train

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int):
        import torch

        rgb = self.split.rgb(i)
        masks, anns = self.split.masks(i)
        if self.train and np.random.rand() < 0.5:  # horizontal flip
            rgb, masks = rgb[:, ::-1].copy(), masks[:, :, ::-1].copy()
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {
            "boxes": torch.as_tensor(np.array(boxes, np.float32).reshape(-1, 4)),
            "labels": torch.ones(len(anns), dtype=torch.int64),
            "masks": torch.as_tensor(masks.astype(np.uint8)),
            "image_id": torch.tensor(self.split.images[i]["id"]),
        }
        return torch.as_tensor(rgb).permute(2, 0, 1).float() / 255.0, target
