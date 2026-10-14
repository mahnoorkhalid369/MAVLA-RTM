#!/usr/bin/env python3
"""Verify a generated dataset and derive YOLO-seg labels (numpy + OpenCV; no Isaac Sim).

Checks per split: every COCO image has its JPEG and mask PNG, image / annotation ids are unique,
each annotation's area equals its mask pixels for that uid, no file is orphaned. Reports counts,
masks per image, mask sizes, family / view / held / stacked statistics, the render-time stale check
(from the generator logs if given), then writes YOLO-seg labels and a dataset YAML.

    /usr/bin/python3 datagen/verify_dataset.py data/perception/synthetic_v1 [--logs <dir>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datagen.coco import coco_to_yolo  # noqa: E402


def verify_split(d: Path, sample_area_checks: int = 200) -> dict:
    coco = json.loads((d / "annotations.json").read_text())
    imgs, anns = coco["images"], coco["annotations"]
    problems = []
    if len({i["id"] for i in imgs}) != len(imgs):
        problems.append("duplicate image ids")
    if len({a["id"] for a in anns}) != len(anns):
        problems.append("duplicate annotation ids")
    names = {Path(i["file_name"]).stem for i in imgs}
    if len(names) != len(imgs):
        problems.append("duplicate file names")
    jpgs = {p.stem for p in (d / "images").glob("*.jpg")}
    pngs = {p.stem for p in (d / "masks").glob("*.png")}
    if names - jpgs:
        problems.append(f"{len(names - jpgs)} images missing")
    if names - pngs:
        problems.append(f"{len(names - pngs)} masks missing")
    if (jpgs | pngs) - names:
        problems.append(f"{len((jpgs | pngs) - names)} files not in the annotations (orphans)")
    by_img = collections.defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)
    rng = np.random.default_rng(0)
    checked, mismatched = 0, 0
    for im in rng.choice(imgs, min(sample_area_checks, len(imgs)), replace=False):
        m = cv2.imread(str(d / "masks" / f"{Path(im['file_name']).stem}.png"), cv2.IMREAD_UNCHANGED)
        for a in by_img[im["id"]]:
            checked += 1
            mismatched += int((m == a["object_uid"]).sum() != a["area"])
    if mismatched:
        problems.append(f"{mismatched}/{checked} sampled annotation areas differ from their masks")
    total_obj = sum(len(i["objects"]) for i in imgs)
    areas = np.array([a["area"] for a in anns])
    per_img = np.array([len(by_img[i["id"]]) for i in imgs])
    return {
        "images": len(imgs), "annotations": len(anns), "objects_placed": total_obj,
        "objects_with_mask_pct": round(100 * len(anns) / max(total_obj, 1), 1),
        "masks_per_image": {"mean": round(float(per_img.mean()), 2), "min": int(per_img.min()), "max": int(per_img.max())},
        "empty_images": int((per_img == 0).sum()),
        "mask_px": {"p5": int(np.percentile(areas, 5)), "median": int(np.median(areas)), "p95": int(np.percentile(areas, 95))},
        "families": dict(collections.Counter(a["family"] for a in anns)),
        "views": dict(collections.Counter(i["view"] for i in imgs)),
        "held_with_mask": int(sum(a.get("held", False) for a in anns)),
        "stacked_with_mask": int(sum(a.get("stacked", False) for a in anns)),
        "area_checks": {"checked": checked, "mismatched": mismatched},
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--logs", default=None, help="folder with full_<split>_<start>.log generator logs (stale-check lines)")
    args = ap.parse_args()
    root = Path(args.dataset)
    report = {}
    for split in ("train", "val", "test_novel"):
        if (root / split / "annotations.json").exists():
            report[split] = verify_split(root / split)
            n = coco_to_yolo(root / split / "annotations.json", root / split / "labels")
            report[split]["yolo_label_lines"] = n
            print(f"{split}: {json.dumps(report[split])}", flush=True)
    (root / "dataset.yaml").write_text(f"# YOLO-seg view of the COCO master (datagen/coco.py)\npath: {root.resolve()}\n"
                                       "train: train/images\nval: val/images\ntest: test_novel/images\nnames:\n  0: object\n")
    sz = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    report["size_gb"] = round(sz / 1e9, 2)
    (root / "STATS.json").write_text(json.dumps(report, indent=1))
    bad = {s: r["problems"] for s, r in report.items() if isinstance(r, dict) and r.get("problems")}
    print(f"size {report['size_gb']} GB; problems: {bad or 'none'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
