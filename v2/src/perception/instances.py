"""Remove measured robot pixels and consolidate redundant object/part masks before numbering."""

import cv2
import numpy as np


def clean_mask(mask, frame, pose, robot):
    ys, xs = np.nonzero(mask)
    z = frame.depth[ys, xs].astype(float)
    valid = np.isfinite(z) & (z > .05) & (z < 6)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    fx, fy, cx, cy = frame.K
    p = np.column_stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z]) @ pose[:3, :3].T + pose[:3, 3]
    excluded = robot.contains(p) if len(p) else np.zeros(0, bool)
    clean = np.zeros_like(mask, dtype=bool)
    keep = ~excluded
    clean[ys[keep], xs[keep]] = True
    return clean, float(excluded.sum() / max(len(z), 1))


def parent_of_parts(masks, measured):
    """Whole masks take precedence over contained labels; adjacent bottle caps join their measured body.

    The cap rule requires a tall round body, a smaller short round top, support-height agreement,
    and touching image evidence. Ordinary stacked boxes and neighbouring bottles remain distinct.
    """
    parents = {}
    for k in sorted(masks, key=lambda n: int(masks[n].sum())):
        g, child = measured[k], masks[k]
        options = []
        for j, parent in masks.items():
            if j == k or parent.sum() < child.sum() * 1.5:
                continue
            h = measured[j]
            contained = (np.logical_and(child, parent).sum() / max(child.sum(), 1) > .85
                         and g['top_z'] <= h['top_z'] + .006)
            cap = (h['yaw_deg'] is None and g['yaw_deg'] is None
                   and h['height'] > max(h['footprint']) * 1.3
                   and max(g['footprint']) < min(h['footprint']) * .72
                   and g['height'] < .035
                   and abs(g.get('visible_bottom_z', g['support_z']) - h['top_z']) < .012
                   and np.linalg.norm(np.array(g['centre'][:2]) - h['centre'][:2]) < .018
                   and np.logical_and(child, cv2.dilate(parent.astype(np.uint8), np.ones((17, 17), np.uint8))).any())
            if contained or cap:
                options.append(j)
        if len(options) == 1:
            parents[k] = options[0]
    # Never form a cycle, and keep the largest complete mask as the root.
    for k, j in list(parents.items()):
        seen = {k}
        while j in parents and j not in seen:
            seen.add(j)
            j = parents[j]
        parents[k] = j
    return parents


def colour_signature(rgb, mask):
    pixels = rgb[mask]
    if not len(pixels):
        return None
    return np.concatenate([np.histogram(pixels[:, k], bins=8, range=(0, 256))[0] / len(pixels)
                           for k in range(3)]).tolist()
