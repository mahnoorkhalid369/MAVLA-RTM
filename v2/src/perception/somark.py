"""Set-of-mark images: numbered mask outlines for the VLM, and object crops for confirmation."""

from __future__ import annotations

import cv2
import numpy as np


def _label_point(mask: np.ndarray) -> tuple[int, int]:
    """A point inside the mask far from its border (distance transform maximum)."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(u), int(v)


def layout(masks: dict[int, np.ndarray]) -> list[tuple]:
    """(number, outline contours, label position) per mask: computed once, drawn on any image of the same size."""
    out, occupied = [], []
    if not masks:
        return out
    height, width = next(iter(masks.values())).shape
    foreground = np.logical_or.reduce(list(masks.values()))
    for k, m in sorted(masks.items()):
        if not m.any():
            continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        u, _ = _label_point(m)
        ys = np.nonzero(m)[0]
        (tw, th), _ = cv2.getTextSize(f'#{k}', cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
        bw, bh = tw + 8, th + 10
        preferred = (u - bw // 2, int(ys.min()) - bh - 5)
        candidates = [(preferred[0] + dx, preferred[1] + dy)
                      for dy in (0, -bh - 5, bh + 5, -2 * bh - 10, 2 * bh + 10)
                      for dx in (0, -bw - 5, bw + 5, -2 * bw - 10, 2 * bw + 10)]
        candidates += [(x, y) for y in range(2, height - bh, bh + 6) for x in range(2, width - bw, bw + 6)]
        options = []
        for x, y in candidates:
            x, y = int(np.clip(x, 2, max(2, width - bw - 2))), int(np.clip(y, 2, max(2, height - bh - 2)))
            rect = (x, y, x + bw, y + bh)
            if any(x < r[2] + 4 and x + bw + 4 > r[0] and y < r[3] + 4 and y + bh + 4 > r[1] for r in occupied):
                continue
            score = foreground[y:y + bh, x:x + bw].sum() * 1000 + np.hypot(x - preferred[0], y - preferred[1])
            options.append((score, rect))
        if not options:
            raise ValueError('number labels do not fit without overlap; use the object crop board')
        _, rect = min(options, key=lambda item: item[0])
        occupied.append(rect)
        out.append((k, cs, (rect[0] + 4 + tw // 2, rect[1] + 4 + th)))
    return out


def draw_layout(rgb: np.ndarray, lay: list[tuple]) -> np.ndarray:
    """RGB copy with thin black outlines (a colour no object is named by) and white-on-black numbers."""
    out = rgb.copy()
    for _, cs, _ in lay:
        cv2.drawContours(out, cs, -1, (0, 0, 0), 2)
    for k, _, (u, v) in lay:
        text = f'#{k}'
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        u0 = int(np.clip(u - w // 2, 0, out.shape[1] - w - 4))
        cv2.rectangle(out, (u0 - 2, v - h - 4), (u0 + w + 2, v + 3), (0, 0, 0), -1)
        # Connect a displaced number to its own outline, so label spacing cannot change its referent.
        contours = next(cs for number, cs, _ in lay if number == k)
        points = np.concatenate(contours).reshape(-1, 2)
        anchor = points[np.argmin(np.linalg.norm(points - [u, v], axis=1))]
        cv2.line(out, (u, v + 4), tuple(map(int, anchor)), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.rectangle(out, (u0 - 2, v - h - 4), (u0 + w + 2, v + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (u0, v), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_marks(rgb: np.ndarray, masks: dict[int, np.ndarray]) -> np.ndarray:
    """The image Gemma grounds on (and the dashboard shows): each mask outlined in black, its number above it."""
    return draw_layout(rgb, layout(masks))


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


def draw_names(som: np.ndarray, masks: dict[int, np.ndarray], names: dict[int, str]) -> np.ndarray:
    """For the report: the set-of-mark image with the VLM's name for each mark under the object."""
    out = som.copy()
    for k, m in masks.items():
        ys, xs = np.nonzero(m)
        text = f"{k}: {names.get(k, '?')}"[:40]
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        u0 = int(np.clip(xs.mean() - w // 2, 0, out.shape[1] - w - 4))
        v = int(min(ys.max() + h + 6, out.shape[0] - 4))
        cv2.rectangle(out, (u0 - 2, v - h - 3), (u0 + w + 2, v + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (u0, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def isolate(rgb: np.ndarray, mask: np.ndarray, margin: float = 0.6, min_side: int = 224, dim: float = 0.3) -> np.ndarray:
    """Crop around a mask with context; everything outside the mask is dimmed and NOTHING is drawn on
    the object, so its own colours reach the VLM unchanged (an outline colour bleeds into small objects
    once the image is downscaled). Upscaled if small."""
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    pad = int(max(h, w) * margin) + 8
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
    out = rgb[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1].astype(bool)
    out[~m] *= dim
    out = out.astype(np.uint8)
    s = max(1.0, min_side / max(out.shape[:2]))
    if s > 1.0:
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return out


def montage(crops: dict[int, np.ndarray], height: int = 224) -> np.ndarray:
    """Isolated crops side by side at one height, each with its number in a header strip ABOVE it
    (nothing drawn on the objects)."""
    tiles = []
    for k, c in crops.items():
        c = cv2.resize(c, (max(1, int(c.shape[1] * height / c.shape[0])), height), interpolation=cv2.INTER_AREA)
        head = np.zeros((36, c.shape[1], 3), np.uint8)
        cv2.putText(head, str(k), (max(4, c.shape[1] // 2 - 10), 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([head, c]))
        tiles.append(np.full((height + 36, 12, 3), 255, np.uint8))  # white gap between objects
    return np.hstack(tiles[:-1])


def grid(crops: dict[int, np.ndarray], tile: int = 150, cols: int = 6) -> np.ndarray:
    """All isolated crops in one picture: square tiles, each with its number in a header ABOVE it
    (nothing drawn on the objects). For one batched VLM call instead of one call per object."""
    keys = sorted(crops)
    rows = (len(keys) + cols - 1) // cols
    head = 30
    out = np.full((rows * (tile + head), cols * tile, 3), 255, np.uint8)
    for i, k in enumerate(keys):
        c = crops[k]
        s = tile / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * (tile + head), (i % cols) * tile
        out[y:y + head, x:x + tile] = 0
        cv2.putText(out, f'#{k}', (x + 8, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        oy, ox = y + head + (tile - c.shape[0]) // 2, x + (tile - c.shape[1]) // 2
        out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
        cv2.rectangle(out, (x, y), (x + tile - 1, y + head + tile - 1), (255, 255, 255), 2)
    return out
