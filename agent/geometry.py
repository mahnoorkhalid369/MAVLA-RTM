"""Measured geometry of one detected object from depth inside its mask (numpy only).

Back-projects the (eroded) mask's depth pixels to the world, removes points that belong to the
robot (its collision model at the current joint state, inflated; the held object's volume is kept)
or to configured fixtures, drops the floor at the mask edges, and measures: top height, top-face
centre (min-area rectangle), footprint, yaw, symmetry, grasp yaw / width, height and support height.
Reuses the geometry functions of perception/grounding.py (no naming, no catalogue).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from agent.config import (ABOVE_SUPPORT, FIXTURE_INFLATE, FIXTURES, GROUND_Z, ROBOT_BASE_EXCLUSION, MASK_ERODE_PX, MIN_POINTS, ROBOT_INFLATE,
                          SUPPORT_SNAP, TOP_LAYER, UP_NORMAL)
from perception.grounding import footprint_yaw, grasp_pose


@dataclass
class Measured:
    centre: list[float]  # (x, y, z) of the object's centre, world (m)
    top_z: float
    height: float
    support_z: float  # height of what it stands on (floor = 0)
    yaw_deg: float | None  # top-face rectangle yaw, None if round
    extents: list[float]  # top face along its own x and y axes (at yaw) (m)
    footprint: list[float]  # (long, short) (m)
    grasp_yaw_deg: float
    grasp_width: float
    symmetry_deg: float
    tilt_deg: float | None  # top-face normal vs vertical; None if no top face is visible
    points: int
    mask_pixels: int
    excluded_robot: int = 0
    excluded_fixture: int = 0
    visible_bottom_z: float = 0.0  # lowest visible point (before snapping the support)

    def to_dict(self) -> dict:
        return asdict(self)


class RobotVolume:
    """Point-in-robot test from the collision model's spheres at a joint state (+ held object kept)."""

    def __init__(self, checker, q, opening: float, held_box=None, tool_pose=None) -> None:
        c, r, idx, names = checker.spheres(q, opening, held_box)
        robot = np.array([names[i] != "held" for i in idx])
        self.centres, self.radii = c[robot], r[robot] + ROBOT_INFLATE
        self.held = None
        if held_box is not None and tool_pose is not None:  # held object's box in the world
            T_held, size = held_box
            self.held = (tool_pose @ T_held, np.asarray(size) / 2)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        inside = np.zeros(len(pts), bool)
        for k in range(0, len(self.centres), 32):
            d = np.linalg.norm(pts[:, None, :] - self.centres[None, k:k + 32], axis=-1)
            inside |= np.any(d <= self.radii[None, k:k + 32], axis=1)
        if self.held is not None:  # the held object is NOT robot: keep its points
            T, half = self.held
            local = (pts - T[:3, 3]) @ T[:3, :3]
            inside &= ~np.all(np.abs(local) <= half + 0.002, axis=1)
        return inside


def in_fixtures(pts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(pts), bool)
    for f in FIXTURES + [ROBOT_BASE_EXCLUSION]:
        out |= np.all(np.abs(pts - np.asarray(f.centre)) <= np.asarray(f.half) + FIXTURE_INFLATE, axis=1)
    return out


def back_project(mask: np.ndarray, depth: np.ndarray, K, T: np.ndarray, erode: int = MASK_ERODE_PX,
                 normals: bool = False):
    """World points of the (eroded) mask's valid depth pixels; with normals=True also |normal z| per
    point (surface normal from neighbouring pixels: top faces ~1, vertical sides ~0)."""
    m = mask.astype(np.uint8)
    if erode > 0:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return (np.zeros((0, 3)), np.zeros(0)) if normals else np.zeros((0, 3))
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, depth.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, depth.shape[1])
    v, u = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < 6.0)
    fx, fy, cx, cy = K
    zz = np.where(ok, z, np.nan)
    P = np.stack([(u - cx) / fx * zz, (v - cy) / fy * zz, zz], axis=-1) @ T[:3, :3].T + T[:3, 3]
    sel = (m[y0:y1, x0:x1] > 0) & ok
    pts = P[sel]
    if not normals:
        return pts
    du, dv = np.gradient(P, axis=1), np.gradient(P, axis=0)
    n = np.cross(du, dv)
    nz = np.abs(n[..., 2]) / np.maximum(np.linalg.norm(n, axis=-1), 1e-12)
    return pts, np.nan_to_num(nz[sel], nan=0.0)


def _rect_frame(xy: np.ndarray, yaw: float | None) -> tuple[np.ndarray, np.ndarray]:
    """(centre, extents along the rectangle's axes) of 2-D points at a yaw (robust percentiles)."""
    a = np.radians(yaw or 0.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = xy @ R
    lo, hi = np.percentile(local, 2, axis=0), np.percentile(local, 98, axis=0)
    return ((lo + hi) / 2) @ R.T, hi - lo


def measure(mask: np.ndarray, depth: np.ndarray, K, T: np.ndarray, robot: RobotVolume | None) -> tuple[Measured | None, str]:
    """(measurement, reason) — None with a reason when the mask cannot be measured as an object."""
    pts, nz = back_project(mask, depth, K, T, normals=True)
    n_mask = int(mask.sum())
    if len(pts) < MIN_POINTS:
        return None, f"too few depth points ({len(pts)})"
    rob = robot.contains(pts) if robot is not None else np.zeros(len(pts), bool)
    fix = in_fixtures(pts)
    if (rob | fix).mean() > 0.5:
        return None, f"mostly robot ({rob.mean():.0%}) / fixture ({fix.mean():.0%})"
    keep = ~(rob | fix)
    pts, nz = pts[keep], nz[keep]
    above = pts[:, 2] > GROUND_Z + ABOVE_SUPPORT
    pts, nz = pts[above], nz[above]
    if len(pts) < MIN_POINTS:
        return None, "no points above the support"
    top_z = float(np.percentile(pts[:, 2], 98))
    layer = pts[:, 2] > top_z - TOP_LAYER
    face = pts[layer & (nz > UP_NORMAL)]  # the upward-facing top surface (side faces excluded)
    top_visible = len(face) >= 10
    # outline from the top face when it is visible (the upper side-face points lean towards the camera)
    xy = face[:, :2] if top_visible else pts[layer][:, :2]
    yaw, fp = footprint_yaw(xy)
    centre_xy, ext = _rect_frame(xy, yaw)
    g_yaw, g_width, sym = grasp_pose(xy, yaw, fp)
    # the eroded mask edge lost MASK_ERODE_PX pixels on each side: add that width back (pixel size at the face)
    cam_dist = float(np.median(np.linalg.norm(pts[layer] - T[:3, 3], axis=1)))
    pad = 2 * MASK_ERODE_PX * cam_dist / K[0]
    ext, fp, g_width = ext + pad, (fp[0] + pad, fp[1] + pad), g_width + pad
    if yaw is None:  # round: a percentile range under-measures a disc; use the radius around the centre
        d = 2 * float(np.percentile(np.linalg.norm(xy - centre_xy, axis=1), 99)) + pad
        ext, fp, g_width = np.array([d, d]), (d, d), d
    bottom = float(np.percentile(pts[:, 2], 2))
    support_z = GROUND_Z if bottom < GROUND_Z + SUPPORT_SNAP else bottom  # refined by snap_supports
    height = top_z - support_z
    tilt = None  # unknown when the top face is hidden (never guessed)
    if top_visible:  # plane fit of the upward-facing points
        normal = np.linalg.svd(face - face.mean(axis=0), full_matrices=False)[2][-1]
        tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    return Measured([float(centre_xy[0]), float(centre_xy[1]), float(support_z + height / 2)], top_z, float(height),
                    float(support_z), yaw, [float(ext[0]), float(ext[1])], [float(max(fp)), float(min(fp))], float(g_yaw),
                    float(g_width), float(sym), tilt, int(len(pts)), n_mask, int(rob.sum()), int(fix.sum()), bottom), "ok"


def snap_supports(objs: list[dict]) -> None:
    """Set each object's support to the highest real surface below its lowest visible point: the
    floor or the measured top of another object under its centre (within SUPPORT_SNAP). In place."""
    for g in objs:
        b = g["visible_bottom_z"]
        surfaces = [GROUND_Z] + [o["top_z"] for o in objs if o is not g and o["top_z"] < g["top_z"] - 0.01 and
                                 np.hypot(o["centre"][0] - g["centre"][0], o["centre"][1] - g["centre"][1]) < max(o["footprint"]) / 2]
        below = [s for s in surfaces if s <= b + 0.003]
        if below and b - max(below) <= SUPPORT_SNAP:
            g["support_z"] = float(max(below))
        g["height"] = g["top_z"] - g["support_z"]
        g["centre"][2] = g["support_z"] + g["height"] / 2
