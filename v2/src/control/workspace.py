"""Where the VT6L + Robotiq 2F-85 can pick a block from the table: the measured pick envelope, and the floor area outside
it (shaded in the simulator by sim/scene.py, so it is visible where the arm cannot pick).

Measured 2026-10-25 with the planner itself (control/skills.py: top-down, else tilted up to SkillParams.max_tilt_deg = 40
deg; every motion IK-, joint-limit-, collision- and Epson-zone-checked) for a 50 mm block standing on the table. Why it has
this shape (Epson VT series manual Rev.6, 2.4 and 5.3):
  - inner hole: the wrist centre cannot come closer than ~0.29-0.35 m to the J1 axis, and the hand reaches 214 mm below it;
  - outer edge: the wrist centre reaches 0.92 m; pointing (nearly) down costs reach;
  - behind the base (bearing 180): J1 stops at +-170 deg, so the arm never faces a 20 deg slice; only a tilted hand reaches
    into it, and only 0.46-0.74 m out;
  - towards the back (|J1| > 110) the forearm may not dip below J3 -40 deg (Epson's forbidden combinations), so the inner
    edge moves out there.
Pure numpy (no Isaac Sim): tests/unit/test_workspace.py.
"""

from __future__ import annotations

import numpy as np

# bearing (deg; 0 = +y forward from the base, +90 = robot's right, +-180 = behind) -> pick radii (m) from the J1 axis:
# (inner, outer). Measured at the listed bearings; mirrored where only one side was measured (the arm is symmetric about
# the y axis); interpolated in between.
PICK_ENVELOPE: dict[float, tuple[float, float]] = {
    0.0: (0.24, 1.02),
    45.0: (0.26, 0.98), -45.0: (0.26, 0.98),
    90.0: (0.24, 1.02), -90.0: (0.28, 1.02),
    135.0: (0.36, 0.98), -135.0: (0.36, 0.98),
    165.0: (0.36, 1.02), -165.0: (0.36, 1.02),
    170.0: (0.42, 1.02), -170.0: (0.42, 1.02),
    175.0: (0.42, 0.86), -175.0: (0.42, 0.86),
    180.0: (0.46, 0.74),  # straight behind: only tilted, J1 at its +-170 limit
}
SHADE_TO_M = 1.40  # the shaded floor reaches this far from the base (the decor stays beyond 1.15 m)


def envelope(bearing_deg: float) -> tuple[float, float]:
    """(inner, outer) pick radii (m) at this bearing, interpolated between the measured bearings."""
    b = (bearing_deg + 180.0) % 360.0 - 180.0
    keys = sorted(PICK_ENVELOPE)  # -175 ... 180; -180 is the same bearing as 180
    xs = [keys[-1] - 360.0] + keys
    ys = [PICK_ENVELOPE[keys[-1]]] + [PICK_ENVELOPE[k] for k in keys]
    return (float(np.interp(b, xs, [y[0] for y in ys])), float(np.interp(b, xs, [y[1] for y in ys])))


def bearing_of(x: float, y: float) -> float:
    return float(np.degrees(np.arctan2(x, y)))


def pickable(x: float, y: float) -> bool:
    """Inside the measured pick envelope (robot frame, m)."""
    inner, outer = envelope(bearing_of(x, y))
    return inner <= float(np.hypot(x, y)) <= outer


def _clip(poly: list, xmin: float, xmax: float, ymin: float, ymax: float) -> list:
    """Sutherland-Hodgman: a convex polygon [(x, y)] clipped to an axis-aligned rectangle."""
    for inside, cut in ((lambda p: p[0] >= xmin, lambda p, q: (xmin, p[1] + (q[1] - p[1]) * (xmin - p[0]) / (q[0] - p[0]))),
                        (lambda p: p[0] <= xmax, lambda p, q: (xmax, p[1] + (q[1] - p[1]) * (xmax - p[0]) / (q[0] - p[0]))),
                        (lambda p: p[1] >= ymin, lambda p, q: (p[0] + (q[0] - p[0]) * (ymin - p[1]) / (q[1] - p[1]), ymin)),
                        (lambda p: p[1] <= ymax, lambda p, q: (p[0] + (q[0] - p[0]) * (ymax - p[1]) / (q[1] - p[1]), ymax))):
        out = []
        for i, q in enumerate(poly):
            p = poly[i - 1]
            if inside(q):
                if not inside(p):
                    out.append(cut(p, q))
                out.append(q)
            elif inside(p):
                out.append(cut(p, q))
        poly = out
        if not poly:
            break
    return poly


def no_go_clipped(xmin: float, xmax: float, ymin: float, ymax: float, z: float = 0.0004, step_deg: float = 2.0):
    """no_go_mesh clipped to a rectangle (a table top): the same smooth edges, nothing beyond the table."""
    pts, counts, idx = no_go_mesh(step_deg=step_deg, shade_to=float(np.hypot(max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax)))) + 0.01, z=z)
    points, out_counts, out_idx = [], [], []
    for quad in np.array(idx).reshape(-1, 4):
        poly = _clip([tuple(pts[k][:2]) for k in quad], xmin, xmax, ymin, ymax)
        if len(poly) >= 3:
            k = len(points)
            points += [(x, y, z) for x, y in poly]
            out_counts.append(len(poly))
            out_idx.extend(range(k, k + len(poly)))
    return np.array(points, float), out_counts, out_idx


def no_go_mesh(step_deg: float = 2.0, shade_to: float = SHADE_TO_M, z: float = 0.0004):
    """Quads covering the floor outside the pick envelope, out to shade_to: (points (n, 3), face vertex counts,
    face vertex indices) for a USD mesh. Each bearing slice gets the inner hole and the band beyond the outer edge."""
    points, counts, idx = [], [], []

    def quad(b0, b1, r00, r01, r10, r11):  # radii (from, to) at b0 and at b1: neighbouring slices share their edges
        a0, a1 = np.radians(b0), np.radians(b1)
        k = len(points)
        for a, r in ((a0, r00), (a1, r10), (a1, r11), (a0, r01)):
            points.append((r * np.sin(a), r * np.cos(a), z))
        counts.append(4)
        idx.extend([k, k + 1, k + 2, k + 3])

    for b0 in np.arange(-180.0, 180.0, step_deg):
        b1 = b0 + step_deg
        (i0, o0), (i1, o1) = envelope(b0), envelope(b1)
        quad(b0, b1, 0.0, i0, 0.0, i1)
        quad(b0, b1, o0, shade_to, o1, shade_to)
    return np.array(points, float), counts, idx
