"""Visible source evidence, independent of the expected object's bounding box or name."""

import cv2
import numpy as np

from config.perception import ABOVE_SUPPORT, GROUND_Z, MASK_ERODE_PX
from perception.geometry import in_fixtures


def foreground(mask, frame, pose, *, support_z=None):
    """Remove invalid depth, configured fixtures and independently measured support planes.

    Keep ALL other foreground, including neighbouring objects. Clipping to the expected volume
    would manufacture a positive association for a merged mask. Pixels and points remain aligned.
    Robot pixels have already been excluded by perceive().
    """
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((2 * MASK_ERODE_PX + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(eroded)
    z = frame.depth[ys, xs].astype(float)
    valid = np.isfinite(z) & (z > .05) & (z < 6.)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    fx, fy, cx, cy = frame.K
    points = np.column_stack([(xs-cx)/fx*z, (ys-cy)/fy*z, z]) @ pose[:3, :3].T + pose[:3, 3]
    keep = (points[:, 2] > GROUND_Z + ABOVE_SUPPORT) & ~in_fixtures(points)
    if support_z is not None and support_z > GROUND_Z + ABOVE_SUPPORT:
        # Only a thin plane is excluded: other foreground below an elevated support remains evidence.
        keep &= np.abs(points[:, 2] - support_z) > ABOVE_SUPPORT
    clean = np.zeros_like(mask, dtype=bool)
    clean[ys[keep], xs[keep]] = True
    return clean, points[keep], {"valid_points": len(points), "foreground_points": int(keep.sum()),
                                "foreground_fraction": float(keep.mean()) if len(points) else 0.}


def crop_quality(mask, rgb, *, minimum_pixels=100):
    ys, xs = np.nonzero(mask)
    h, w = rgb.shape[:2]
    clipped = bool(len(xs) and (xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w-3 or ys.max() >= h-3))
    reason = "source silhouette is clipped" if clipped else "too few source pixels" if len(xs) < minimum_pixels else ""
    return {"ok": not reason, "reason": reason, "pixels": len(xs), "clipped": clipped}
