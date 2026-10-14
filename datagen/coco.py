"""COCO instance-segmentation annotations from instance-id masks, and COCO -> YOLO-seg conversion.
numpy + OpenCV only (no Isaac Sim), so it is unit-tested offline.

Masks are MODAL (visible pixels only). One category: {"id": 1, "name": "object"}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CATEGORIES = [{"id": 1, "name": "object", "supercategory": "object"}]


def mask_polygons(mask: np.ndarray, min_area: float = 4.0) -> list[list[float]]:
    """Outer polygons of a binary mask as flat [x1, y1, x2, y2, ...] lists (COCO 'segmentation')."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        polys.append([float(v) for v in c.reshape(-1, 2).ravel()])
    return polys


def annotations_from_ids(ids: np.ndarray, image_id: int, next_ann_id: int, uid_info: dict[int, dict],
                         min_pixels: int = 30) -> tuple[list[dict], int]:
    """COCO annotations for every object uid present in an instance-id image (0 = background).
    uid_info[uid] carries extra per-object fields (family, asset, held, stacked) kept in the annotation."""
    anns = []
    for uid in np.unique(ids):
        if uid == 0 or int(uid) not in uid_info:
            continue
        m = ids == uid
        area = int(m.sum())
        if area < min_pixels:
            continue
        ys, xs = np.nonzero(m)
        polys = mask_polygons(m)
        if not polys:
            continue
        anns.append({"id": next_ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0, "area": area,
                     "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                     "segmentation": polys, "object_uid": int(uid), **uid_info[int(uid)]})
        next_ann_id += 1
    return anns, next_ann_id


def coco_to_yolo(coco_json: Path, out_dir: Path, images_dir: Path | None = None) -> int:
    """Write YOLO-seg labels (one .txt per image: 'class x1 y1 x2 y2 ...' normalised) from a COCO file.
    Multi-part masks become one line per part. Returns the number of label lines."""
    coco = json.loads(Path(coco_json).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    lines_total = 0
    for img in coco["images"]:
        w, h = img["width"], img["height"]
        lines = []
        for a in by_image.get(img["id"], []):
            for poly in a["segmentation"]:
                xy = np.asarray(poly, float).reshape(-1, 2) / [w, h]
                lines.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        (out_dir / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        lines_total += len(lines)
    return lines_total
