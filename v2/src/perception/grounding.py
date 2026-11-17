"""Ground a 2-D box from the model into a 3-D object pose using the depth image.

No ROS, no model. Inputs are plain arrays:
  box      (ymin, xmin, ymax, xmax), 0-1000 normalised to the image (the model's convention)
  depth    HxW float32, metres along the optical axis (0 / inf / nan = no return)
  K        (fx, fy, cx, cy) pinhole intrinsics of the depth image
  T_world_camera  4x4 pose of the optical frame (z forward, x right, y down) in the world
  rgb      HxWx3 uint8 colour image (optional: measures the object's colour)

The model's boxes are often loose or shifted, so the box is only a hint: the depth points around it
(box grown by 30 %) that stand above the table are split into separate objects ("blobs"), and the blob
nearest the box centre is the object. Its top face gives the position, size, yaw (theta) and colour.

    est = locate(box, depth, K, T_world_camera, rgb)
    est.position    # centre of the top face (world, m); for an empty area: the surface point
    est.yaw_deg     # rotation about vertical (-45..45 deg), None if round
    est.colour      # measured colour name, e.g. "orange"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

GROUND_Z = 0.0  # table / floor height in the world (m)
ABOVE_GROUND = 0.01  # points higher than this count as "object"
MAX_OBJECT_Z = 0.25  # m: higher points are the robot arm, not a test object (tallest stack ~0.15 m)
SEARCH_GROW = 1.3  # look in a box 30 % bigger than the model's (its boxes are loose / shifted)
CELL = 0.005  # m: grid for splitting points into separate objects
MIN_BLOB_POINTS = 30

# Basic colour names and their usual sRGB values; the measured colour is the nearest one (in Lab)
COLOUR_NAMES = {
    "red": (200, 30, 30), "orange": (240, 150, 15), "yellow": (240, 215, 30), "lime": (150, 215, 30),
    "green": (40, 180, 60), "teal": (0, 140, 140), "blue": (30, 70, 215), "navy": (15, 30, 100),
    "purple": (125, 50, 190), "magenta": (215, 30, 170), "pink": (250, 120, 180), "brown": (115, 70, 30),
    "tan": (205, 165, 115), "beige": (240, 225, 190), "olive": (115, 115, 30), "gold": (215, 165, 40),
    "black": (15, 15, 15), "gray": (128, 128, 128), "silver": (190, 190, 200), "white": (240, 240, 240),
}


class GroundingError(ValueError):
    """No usable depth inside the box."""


@dataclass
class ObjectEstimate:
    position: np.ndarray  # (3,) world, m: centre of the top face (object) or surface point (area)
    top_z: float  # m
    size_xy: tuple[float, float]  # footprint extent along world x and y (m)
    is_object: bool  # False: the box covers an empty surface area
    pixel: tuple[int, int]  # (u, v) box centre in the image
    points: int  # depth points used
    yaw_deg: float | None = None  # rotation about vertical, -45..45 deg (None: round, or no object)
    footprint: tuple[float, float] = (0.0, 0.0)  # top-face length x width along its own axes (m)
    colour: str | None = None  # measured colour name
    offset_mm: float = 0.0  # how far the object's centre is from the model's box centre
    rgb: tuple[int, int, int] | None = None  # measured median colour (sRGB 0-255) the name comes from
    # How to grip it from above: gripper yaw (deg; fingers close along (-sin, cos) of it), the object's
    # size across the fingers (m), and its symmetry (90: square / round, any quarter turn grips the same)
    grasp_yaw_deg: float = 0.0
    grasp_width: float = 0.0
    grasp_symmetry_deg: float = 180.0

    def pose_text(self) -> str:
        """'x 123 mm, y 456 mm, theta 12 deg' for logs."""
        x, y = self.position[0] * 1000, self.position[1] * 1000
        theta = "n/a (round)" if self.yaw_deg is None else f"{self.yaw_deg:+.0f} deg"
        return f"x {x:.0f} mm, y {y:.0f} mm, theta {theta}"


def box_pixels(box: tuple[int, int, int, int], width: int, height: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    """(u0, v0, u1, v1) pixel rectangle of a 0-1000 box, scaled about its centre."""
    y0, x0, y1, x1 = (v / 1000.0 for v in box)
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
    u0, u1 = int(np.floor((cx - hw) * width)), int(np.ceil((cx + hw) * width))
    v0, v1 = int(np.floor((cy - hh) * height)), int(np.ceil((cy + hh) * height))
    return max(u0, 0), max(v0, 0), min(u1, width), min(v1, height)


def footprint_yaw(xy: np.ndarray) -> tuple[float | None, tuple[float, float]]:
    """Yaw (deg, -45..45) and (length, width) of the smallest rotated rectangle around 2-D points.

    Turn the points in 1-deg steps and measure the box around them; the angle with the smallest
    box is the object's orientation. A round footprint gives nearly the same box at every angle,
    so it returns yaw None.
    """
    xy = xy - xy.mean(axis=0)
    areas, dims = [], []
    for a in np.radians(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        ext = np.percentile(r, 98, axis=0) - np.percentile(r, 2, axis=0)
        areas.append(ext[0] * ext[1])
        dims.append(ext)
    areas = np.array(areas)
    best = int(np.argmin(areas))
    length, width = sorted(map(float, dims[best]), reverse=True)
    if areas.max() / max(areas.min(), 1e-12) < 1.12:  # same box at every angle: round
        return None, (length, width)
    yaw = float(best) if best < 45 else float(best) - 90.0  # rectangles repeat every 90 deg
    return yaw, (length, width)


# Colour calibration: how named colours actually look through the table camera under the scene's
# light (brighter and paler than the textbook values above; measured with locate() on the test objects).
# Each name can have several references; the nearest one wins.
CALIBRATED_COLOURS = [
    ("pink", (245, 184, 229)), ("brown", (182, 114, 38)), ("olive", (168, 168, 28)), ("beige", (243, 239, 233)),
    ("tan", (236, 223, 190)), ("navy", (9, 24, 146)), ("white", (221, 223, 227)), ("silver", (210, 210, 221)),
    ("gray", (130, 136, 148)), ("gold", (215, 189, 43)), ("yellow", (238, 233, 29)), ("orange", (237, 157, 16)),
    ("lime", (206, 239, 29)),
]


GRIP_BAND = 0.02  # m: the pads (27 mm wide) close within this distance of the grasp centre, along their faces


def grasp_pose(xy: np.ndarray, yaw: float | None, footprint: tuple[float, float],
               centre: np.ndarray | None = None) -> tuple[float, float, float]:
    """(gripper yaw deg, width across the fingers m, symmetry deg) for a top-face footprint.

    The fingers should close across the NARROW side: at gripper yaw a they close along (-sin a, cos a),
    so measure the face along its two edge directions and turn the gripper 90 deg if needed.
    The width is measured where the pads close: within GRIP_BAND of the grasp centre (default: the points' mean)
    along the pads. For a box that is its width; for scissors or a clamp it is the shaft, not the finger rings."""
    if yaw is None:  # round: any direction
        return 0.0, float(min(footprint)), 90.0
    a = np.radians(yaw)
    centred = xy - (xy.mean(axis=0) if centre is None else np.asarray(centre)[:2])
    u, v = centred @ np.array([np.cos(a), np.sin(a)]), centred @ np.array([-np.sin(a), np.cos(a)])
    ext = [float(np.percentile(p, 98) - np.percentile(p, 2)) for p in (u, v)]
    symmetry = 90.0 if abs(ext[0] - ext[1]) < 0.008 else 180.0  # square within 8 mm
    if ext[1] > ext[0] + 0.008:  # narrow side lies along (cos a, sin a): turn the fingers to it
        return float(yaw + 90.0 if yaw <= 0 else yaw - 90.0), _width_at_pads(u, v), symmetry
    return float(yaw), _width_at_pads(v, u), symmetry


def _width_at_pads(across: np.ndarray, along: np.ndarray) -> float:
    """Extent across the fingers of the points the pads close on (all points if too few lie there)."""
    near = np.abs(along) <= GRIP_BAND
    a = across[near] if near.sum() >= 10 else across
    return float(np.percentile(a, 98) - np.percentile(a, 2))


def colour_name(rgb: np.ndarray) -> str:
    """Nearest basic colour name to an sRGB colour (0-255), compared in CIE Lab (perceptual)."""
    refs = list(COLOUR_NAMES.items()) + CALIBRATED_COLOURS
    names = [n for n, _ in refs]
    lab = _to_lab(np.array([c for _, c in refs], float))
    target = _to_lab(np.asarray(rgb, float)[None, :])
    return names[int(np.argmin(np.linalg.norm(lab - target, axis=1)))]


def _to_lab(rgb255: np.ndarray) -> np.ndarray:
    c = rgb255 / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _back_project(box, scale, depth, K, T_world_camera):
    """World points + their pixel (u, v) for the valid depth pixels in a (scaled) box."""
    h, w = depth.shape[:2]
    fx, fy, cx, cy = K
    u0, v0, u1, v1 = box_pixels(box, w, h, scale)
    if u1 - u0 < 2 or v1 - v0 < 2:
        raise GroundingError("box is too small")
    d = depth[v0:v1, u0:u1].astype(np.float64)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    valid = np.isfinite(d) & (d > 0.05) & (d < 5.0)
    if valid.sum() < 10:
        raise GroundingError("no depth readings inside the box")
    z = d[valid]
    cam = np.stack([(uu[valid] - cx) / fx * z, (vv[valid] - cy) / fy * z, z], axis=1)
    world = cam @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    return world, uu[valid], vv[valid]


def _blobs(xy: np.ndarray) -> list[np.ndarray]:
    """Split 2-D points into touching groups (one per object) on a CELL grid; returns index arrays."""
    cells = np.floor(xy / CELL).astype(int)
    cells -= cells.min(axis=0)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, cells)):
        grid.setdefault(key, []).append(i)
    seen, groups = set(), []
    for start in grid:
        if start in seen:
            continue
        queue, members = deque([start]), []
        seen.add(start)
        while queue:
            cx, cy = queue.popleft()
            members.extend(grid[(cx, cy)])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        groups.append(np.array(members))
    return groups


def locate(box, depth: np.ndarray, K, T_world_camera: np.ndarray, rgb: np.ndarray | None = None,
           max_z: float | None = None) -> ObjectEstimate:
    """max_z: ignore points higher than this (default MAX_OBJECT_Z); e.g. just above the expected top
    when the gripper's own fingers are in the image."""
    h, w = depth.shape[:2]
    u0, v0, u1, v1 = box_pixels(box, w, h)
    centre_px = ((u0 + u1) // 2, (v0 + v1) // 2)
    world, uu, vv = _back_project(box, SEARCH_GROW, depth, K, T_world_camera)

    # above the floor, below the arm: a side view sees the arm's links right next to the objects
    above = (world[:, 2] > GROUND_Z + ABOVE_GROUND) & (world[:, 2] < (GROUND_Z + MAX_OBJECT_Z if max_z is None else max_z))
    inside = (uu >= u0) & (uu < u1) & (vv >= v0) & (vv < v1)
    if (above & inside).sum() < MIN_BLOB_POINTS:  # mostly surface: an empty area to place on
        surface = np.median(world[inside] if inside.any() else world, axis=0)
        return ObjectEstimate(surface, float(surface[2]), (0.0, 0.0), False, centre_px, len(world))

    # The object = the blob of above-table points closest to the centre of the model's box
    pts, pu, pv = world[above], uu[above], vv[above]
    groups = [g for g in _blobs(pts[:, :2]) if len(g) >= MIN_BLOB_POINTS] or [np.arange(len(pts))]
    box_centre = np.median(world[inside & above][:, :2], axis=0)
    obj_idx = min(groups, key=lambda g: np.linalg.norm(np.median(pts[g, :2], axis=0) - box_centre))
    obj, ou, ov = pts[obj_idx], pu[obj_idx], pv[obj_idx]

    top_z = float(np.percentile(obj[:, 2], 95))
    on_top = obj[:, 2] > top_z - 0.012  # points on the top face
    face = obj[on_top]
    xy = np.median(face[:, :2], axis=0)
    size = tuple(float(v) for v in np.percentile(face[:, :2], 98, axis=0) - np.percentile(face[:, :2], 2, axis=0))
    yaw, footprint = footprint_yaw(face[:, :2])
    colour, measured = None, None
    if rgb is not None:
        # whole visible object (top + sides): a brightly lit top face alone looks paler / yellower
        measured = np.median(rgb[ov, ou, :3].astype(float), axis=0)
        colour = colour_name(measured)
    g_yaw, g_width, g_sym = grasp_pose(face[:, :2], yaw, footprint, centre=xy)
    return ObjectEstimate(
        np.array([xy[0], xy[1], top_z]), top_z, size, True, centre_px, len(obj), yaw, footprint, colour,
        float(np.linalg.norm(xy - box_centre) * 1000), None if measured is None else tuple(int(c) for c in measured),
        g_yaw, g_width, g_sym,
    )
