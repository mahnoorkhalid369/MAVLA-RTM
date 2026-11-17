"""Measured geometry of one detected object from depth inside its mask (numpy only).

Back-projects the (eroded) mask's depth pixels to the world, removes points that belong to the
robot (its collision model at the current joint state, inflated; the held object's volume is kept)
or to configured fixtures, drops the floor at the mask edges, and measures: top height, top-face
centre (min-area rectangle), footprint, yaw, symmetry, grasp yaw / width, height and support height.
Reuses the geometry functions of perception/grounding.py (no naming, no catalogue).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from config.perception import (ABOVE_SUPPORT, FIXTURE_INFLATE, FIXTURES, GROUND_Z, ROBOT_BASE_EXCLUSION, MASK_ERODE_PX, MIN_ABOVE_FRACTION, MIN_POINTS, ROBOT_INFLATE,
                               SUPPORT_SNAP, TOP_LAYER, UP_NORMAL)
from perception.grounding import footprint_yaw, grasp_pose
from control.kinematics import URDF_PATH, _rpy_to_matrix


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
    grasp_centre: list[float] | None = None  # solid central grasp region; distinct from the object's bounding centre
    top_planar: bool = True
    top_plane_fraction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Robot parts that are VISUAL ONLY in the URDF (no collision shape) but seen by the cameras:
# (link, box centre in the link frame, half sizes). The wrist-camera bracket (URDF camera_link visual
# "bracket", 20 x 40 x 50 mm at z -37.5 mm) was measured as a floating object (2026-10-14).
VISUAL_ONLY_PARTS = [("camera_link", (0.0, 0.0, -0.0375), (0.010, 0.020, 0.025))]


@lru_cache(maxsize=1)
def visual_primitives():
    parts = []
    for link in ET.parse(URDF_PATH).getroot().findall('link'):
        for visual in link.findall('visual'):
            origin, geometry = visual.find('origin'), visual.find('geometry')
            if geometry is None:
                continue
            pose = np.eye(4)
            if origin is not None:
                pose[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                pose[:3, :3] = _rpy_to_matrix(*[float(v) for v in origin.get('rpy', '0 0 0').split()])
            box, cylinder = geometry.find('box'), geometry.find('cylinder')
            if box is not None:
                parts.append((link.get('name'), pose, 'box', np.array([float(v) for v in box.get('size').split()]) / 2))
            elif cylinder is not None:
                parts.append((link.get('name'), pose, 'cylinder', (float(cylinder.get('radius')), float(cylinder.get('length')) / 2)))
    return parts


class RobotVolume:
    """Point-in-robot test from the collision model's spheres at a joint state, plus visual-only robot parts
    (VISUAL_ONLY_PARTS, posed by forward kinematics), inflated; the held object's volume is kept."""

    def __init__(self, checker, q, opening: float, held_box=None, tool_pose=None) -> None:
        c, r, idx, names = checker.spheres(q, opening, held_box)
        robot = np.array([names[i] != "held" for i in idx])
        self.centres, self.radii = c[robot], r[robot] + ROBOT_INFLATE
        self.boxes = []
        self.cylinders = []
        poses = checker.link_poses(q)
        for link, origin, kind, dimensions in list(getattr(checker, "primitives", [])) + visual_primitives():
            if link not in poses:
                continue
            T = poses[link] @ origin
            if kind == "box":
                self.boxes.append((T, np.array(dimensions) + ROBOT_INFLATE))
            else:
                self.cylinders.append((T, dimensions[0] + ROBOT_INFLATE, dimensions[1] + ROBOT_INFLATE))
        for link, centre, half in VISUAL_ONLY_PARTS:
            if link in poses:
                T = poses[link] @ np.vstack([np.column_stack([np.eye(3), centre]), [0, 0, 0, 1]])
                self.boxes.append((T, np.asarray(half) + ROBOT_INFLATE))
        self.held = None
        if held_box is not None and tool_pose is not None:  # held object's box in the world
            T_held, size = held_box
            self.held = (tool_pose @ T_held, np.asarray(size) / 2)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        inside = np.zeros(len(pts), bool)
        for k in range(0, len(self.centres), 32):
            d = np.linalg.norm(pts[:, None, :] - self.centres[None, k:k + 32], axis=-1)
            inside |= np.any(d <= self.radii[None, k:k + 32], axis=1)
        for T, half in self.boxes:
            local = (pts - T[:3, 3]) @ T[:3, :3]
            inside |= np.all(np.abs(local) <= half, axis=1)
        for T, radius, half_height in self.cylinders:
            local = (pts - T[:3, 3]) @ T[:3, :3]
            inside |= (np.hypot(local[:, 0], local[:, 1]) <= radius) & (np.abs(local[:, 2]) <= half_height)
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


def central_grasp(xy: np.ndarray, centre: np.ndarray, yaw: float | None) -> np.ndarray:
    """Centre of a measured solid cross-section near the footprint's longitudinal midpoint.

    A curved object's bounding centre may be empty. Keep a regular object's centre
    when depth supports it; otherwise use the central cross-section, with a measured
    point fallback if that cross-section is disconnected.
    """
    if np.min(np.linalg.norm(xy - centre, axis=1)) <= 0.003:
        return centre.copy()
    a = np.radians(yaw or 0.)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = (xy - centre) @ R
    axis = int(np.argmax(np.ptp(local, axis=0)))
    band = np.abs(local[:, axis]) <= 0.010
    points = local[band] if band.sum() >= 10 else local
    candidate = (np.percentile(points, 2, axis=0) + np.percentile(points, 98, axis=0)) / 2
    nearest = points[np.argmin(np.linalg.norm(points - candidate, axis=1))]
    if np.linalg.norm(nearest - candidate) > 0.003:
        candidate = nearest
    return centre + candidate @ R.T


def round_section_grasp(pts, top_z, centre):
    """Fit visible circular arcs at pad heights; a front half-cylinder is not a thin graspable slab.

    Several independent height slices must agree on the axis. Flat carton sides and filled top faces
    fail the radial-residual test. The largest fitted pad-height diameter supplies the closing width.
    """
    fits = []
    for depth in np.arange(.020, .065, .004):
        xy = pts[(pts[:, 2] < top_z - depth) & (pts[:, 2] > top_z - depth - .004), :2]
        if len(xy) < 30:
            continue
        local = xy - centre
        A = np.column_stack([2 * local, np.ones(len(local))])
        if np.linalg.matrix_rank(A) < 3:
            continue
        sol = np.linalg.lstsq(A, np.sum(local ** 2, axis=1), rcond=None)[0]
        radius = np.sqrt(max(0., sol[2] + sol[0] ** 2 + sol[1] ** 2))
        residual = np.abs(np.linalg.norm(local - sol[:2], axis=1) - radius)
        if (.008 < radius < .050 and np.linalg.norm(sol[:2]) < .025
                and np.percentile(residual, 90) < .0015):
            fits.append((sol[:2] + centre, radius))
    if len(fits) < 3:
        return None
    centres = np.array([c for c, _ in fits])
    axis = np.median(centres, axis=0)
    good = np.linalg.norm(centres - axis, axis=1) < .005
    if good.sum() < 3:
        return None
    return axis, 2 * max(r for ok, (_, r) in zip(good, fits) if ok)


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
    if len(pts) and above.mean() < MIN_ABOVE_FRACTION and above.sum() >= MIN_POINTS:
        return None, f"mostly the support surface ({1 - above.mean():.0%} of its points)"
    pts, nz = pts[above], nz[above]
    if len(pts) < MIN_POINTS:
        return None, "no points above the support"
    top_z = float(np.percentile(pts[:, 2], 98))
    layer = pts[:, 2] > top_z - TOP_LAYER
    face = pts[layer & (nz > UP_NORMAL)]  # the upward-facing top surface (side faces excluded)
    top_visible = len(face) >= 10
    # outline from the top face when it is visible (the upper side-face points lean towards the camera)
    # Full visible body bounds, rather than the small upward-facing patch of a curved apple or bottle cap.
    xy = pts[:, :2]
    yaw, fp = footprint_yaw(xy)
    centre_xy, ext = _rect_frame(xy, yaw)
    surface_xy = face[:, :2] if top_visible else pts[layer, :2]
    round_fit = round_section_grasp(pts, top_z, centre_xy) if max(fp) / max(min(fp), .001) < 1.5 else None
    hull_area = cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))
    rectangular = yaw is not None and hull_area / max(float(np.prod(ext)), 1e-8) > .88
    if round_fit is not None:
        centre_xy, g_width = round_fit
        yaw, g_yaw, sym = None, 0., 90.
        grasp_xy = centre_xy.copy()
    elif rectangular:
        grasp_xy = central_grasp(surface_xy, centre_xy, yaw)
        g_yaw = yaw + 90 if ext[0] < ext[1] else yaw
        g_width, sym = float(min(ext)), 90. if abs(ext[0] - ext[1]) < .008 else 180.
    else:
        surface_yaw, surface_fp = footprint_yaw(surface_xy)
        grasp_xy = central_grasp(surface_xy, centre_xy, surface_yaw)
        g_yaw, g_width, sym = grasp_pose(surface_xy, surface_yaw, surface_fp, centre=grasp_xy)
    # the eroded mask edge lost MASK_ERODE_PX pixels on each side: add that width back (pixel size at the face)
    cam_dist = float(np.median(np.linalg.norm(pts[layer] - T[:3, 3], axis=1)))
    pad = 2 * MASK_ERODE_PX * cam_dist / K[0]
    ext, fp, g_width = ext + pad, (fp[0] + pad, fp[1] + pad), g_width + (0. if round_fit is not None else pad)
    if yaw is not None and sym == 90.0:  # square footprint: both extents measure the same width; average
        w = float(np.mean(ext))         # them (the oblique view under-reads one axis and over-reads the other)
        ext, fp = np.array([w, w]), (w, w)
    if yaw is None:  # round: a percentile range under-measures a disc; use the radius around the centre
        d = 2 * float(np.percentile(np.linalg.norm(xy - centre_xy, axis=1), 99)) + pad
        ext, fp = np.array([d, d]), (d, d)
    bottom = float(np.percentile(pts[:, 2], 2))
    support_z = GROUND_Z if bottom < GROUND_Z + SUPPORT_SNAP else bottom  # refined by snap_supports
    height = top_z - support_z
    tilt = None  # unknown when the top face is hidden (never guessed)
    planar = False
    fraction = 0.
    if top_visible:  # plane fit of the upward-facing points
        fit = face
        for _ in range(3):
            centre = fit.mean(axis=0)
            normal = np.linalg.svd(fit - centre, full_matrices=False)[2][-1]
            residuals = np.abs((face - centre) @ normal)
            fit = face[residuals <= max(.0005, np.percentile(residuals, 80))]
        tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
        inliers = residuals < .0015
        fraction = float(inliers.mean())
        values = nz[layer & (nz > UP_NORMAL)][inliers]
        spread = np.ptp(np.percentile(values, [20, 80])) if len(values) else 1.
        planar = bool(fraction >= .80 and spread < .006)
    return Measured([float(centre_xy[0]), float(centre_xy[1]), float(support_z + height / 2)], top_z, float(height),
                    float(support_z), yaw, [float(ext[0]), float(ext[1])], [float(max(fp)), float(min(fp))], float(g_yaw),
                    float(g_width), float(sym), tilt, int(len(pts)), n_mask, int(rob.sum()), int(fix.sum()), bottom,
                    [float(grasp_xy[0]), float(grasp_xy[1])], planar, fraction), "ok"


def snap_supports(objs: list[dict], surfaces: np.ndarray | None = None, surface_exclusions=None) -> None:
    """Set each object's support to the highest real surface below its lowest visible point: the
    floor or the measured top of another object under its centre (within SUPPORT_SNAP). In place."""
    for index, g in enumerate(objs):
        b = g["visible_bottom_z"]
        candidates = [GROUND_Z] + [o["top_z"] for o in objs if o is not g and o["top_z"] < g["top_z"] - 0.01 and
                                 np.hypot(o["centre"][0] - g["centre"][0], o["centre"][1] - g["centre"][1]) < max(o["footprint"]) / 2]
        below = [s for s in candidates if s <= b + 0.003]
        g['support_confident'] = False
        if surfaces is not None:
            cloud = surfaces if surface_exclusions is None else surfaces[~surface_exclusions[index]]
            radius = max(g['footprint']) / 2 + .045
            p = cloud[(np.linalg.norm(cloud[:, :2] - g['centre'][:2], axis=1) < radius)
                      & (cloud[:, 2] <= b + .003) & (cloud[:, 2] < g['top_z'] - .010)]
            if len(p):
                bins = np.rint(p[:, 2] / .003).astype(int)
                levels, counts = np.unique(bins, return_counts=True)
                for i in np.argsort(-levels):
                    patch = p[bins == levels[i]]
                    if len(patch) >= 12 and min(np.ptp(patch[:, :2], axis=0)) >= min(g['footprint']) * .6:
                        g['support_z'] = float(np.median(patch[:, 2]))
                        g['support_confident'] = True
                        break
        if not g['support_confident'] and below and b - max(below) <= SUPPORT_SNAP:
            g["support_z"] = float(max(below))
            g['support_confident'] = True
        elif not g['support_confident']:
            # A hidden lower half is not a floating support. Keep conservative full-height bounds,
            # and require another view before a grasp can rely on this support hypothesis.
            g['support_z'] = GROUND_Z
        g["height"] = g["top_z"] - g["support_z"]
        g["centre"][2] = g["support_z"] + g["height"] / 2
