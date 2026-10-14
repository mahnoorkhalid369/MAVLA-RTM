"""Set-of-mark images: numbered mask outlines for the VLM, and object crops for confirmation."""

from __future__ import annotations

import cv2
import numpy as np


def _label_point(mask: np.ndarray) -> tuple[int, int]:
    """A point inside the mask far from its border (distance transform maximum)."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(u), int(v)


def draw_marks(rgb: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
    """RGB copy with each mask outlined and its number written next to it (white on black)."""
    out = rgb.copy()
    for k, m in masks.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 0), 2)
    for k, m in masks.items():
        u, v = _label_point(m)
        ys, xs = np.nonzero(m)
        v = max(int(ys.min()) - 6, 18)  # just above the object, so the label does not hide it
        text = str(k)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        u0 = int(np.clip(u - w // 2, 0, out.shape[1] - w - 4))
        cv2.rectangle(out, (u0 - 2, v - h - 4), (u0 + w + 2, v + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (u0, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def crop(rgb: np.ndarray, mask: np.ndarray, margin: float = 0.6, min_side: int = 160) -> np.ndarray:
    """Crop around a mask with context (margin x its size each side), outline drawn, upscaled if small."""
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    pad = int(max(h, w) * margin) + 8
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    out = rgb[y0:y1, x0:x1].copy()
    cs, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 0), 1)
    s = max(1.0, min_side / max(out.shape[:2]))
    if s > 1.0:
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return out
