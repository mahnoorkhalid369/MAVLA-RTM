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


def merge_parts(polys: list[np.ndarray]) -> np.ndarray:
    """One polygon from several parts of one mask: each part is joined to the polygon so far by a
    zero-width bridge between their closest points (as Ultralytics' COCO converter does), so a YOLO
    instance stays ONE object even when an occluder splits its visible mask."""
    merged = np.asarray(polys[0], float)
    for q in polys[1:]:
        q = np.asarray(q, float)
        d = np.linalg.norm(merged[:, None, :] - q[None, :, :], axis=-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        q_loop = np.concatenate([q[j:], q[:j + 1]])  # walk the whole part and come back to its start
        merged = np.concatenate([merged[:i + 1], q_loop, merged[i:]])
    return merged


def coco_to_yolo(coco_json: Path, out_dir: Path, images_dir: Path | None = None) -> int:
    """Write YOLO-seg labels (one .txt per image: 'class x1 y1 x2 y2 ...' normalised) from a COCO file.
    One line per instance: a multi-part mask is merged into one polygon (merge_parts). Returns the
    number of label lines."""
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
            parts = [np.asarray(poly, float).reshape(-1, 2) for poly in a["segmentation"] if len(poly) >= 6]
            if not parts:
                continue
            xy = merge_parts(parts) / [w, h]
            lines.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        (out_dir / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        lines_total += len(lines)
    return lines_total
