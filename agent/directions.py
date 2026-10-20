"""Directions as vectors in an explicit frame, and where the wrist camera actually points (measured).

Gemma states a direction as a vector [x, y, z] in a named frame (the "look" operation); this module turns it into a
unit vector in the robot base frame and measures the wrist camera's optical axis from joint feedback (forward
kinematics of the measured joint state). Pure geometry: no words, no direction names, no scene knowledge.

    robot frame   the robot's base: x = the robot's right, y = forward, z = up (the world frame of the agent)
    camera frame  the fixed (table) camera's picture: x = image right on the floor, y = the camera's viewing
                  direction on the floor, z = up
"""

from __future__ import annotations

import numpy as np

FRAMES = ("robot", "camera")
VIEW_TOLERANCE_DEG = 30.0  # a sideways direction: the looked-at floor point within this bearing of it
VERTICAL_TOLERANCE_DEG = 15.0  # a (mostly) vertical direction: the optical axis within this angle of it
VERTICAL = 0.3  # a direction whose horizontal part is shorter than this (unit vector) counts as vertical
CAMERA_TIP = "camera_color_optical_frame"  # the wrist camera frame (URDF); its +z is the optical axis


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def direction_vector(direction, frame: str = "robot") -> np.ndarray:
    """Unit vector in robot base coordinates of a direction [x, y, z] given in `frame`."""
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r} (one of {FRAMES})")
    try:
        v = np.asarray(direction, float).reshape(-1)
    except (TypeError, ValueError):
        v = np.zeros(0)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-6:
        raise ValueError(f"a direction is three numbers [x, y, z], not all zero (got {direction!r})")
    if frame == "camera":
        from control.robot import table_camera_pose
        R = table_camera_pose()[:3, :3]
        right, forward = _unit(R[:2, 0]), _unit(R[:2, 2])  # image right; the viewing direction on the floor
        v = np.array([v[0] * right[0] + v[1] * forward[0], v[0] * right[1] + v[1] * forward[1], v[2]])
    return _unit(v)


def is_vertical(d: np.ndarray) -> bool:
    return float(np.linalg.norm(d[:2])) < VERTICAL


def camera_pose(q) -> np.ndarray:
    """4x4 pose of the wrist camera for a joint state (forward kinematics; J6 does not move the camera)."""
    from control.kinematics import Kinematics
    kin = Kinematics(tip=CAMERA_TIP)
    return kin.fk(np.asarray(q, float)[:len(kin.lower)])


def view_error(q, direction, frame: str = "robot") -> dict:
    """How far the wrist camera (joint state q) is from looking along `direction` ([x, y, z] in `frame`).
    A sideways direction: where the optical axis meets the floor, as a bearing from the base, against the direction's
    horizontal part (what the camera looks at lies on that side of the robot; a camera looking down beside itself
    does not count). An axis that does not meet the floor uses its heading. A vertical direction: the full axis."""
    T = camera_pose(q)
    axis, pos = T[:3, 2], T[:3, 3]
    want = direction_vector(direction, frame)
    seen = None
    if is_vertical(want):
        err, tol = float(np.degrees(np.arccos(np.clip(axis @ want, -1.0, 1.0)))), VERTICAL_TOLERANCE_DEG
    else:
        if axis[2] < -0.1:  # meets the floor (z = 0)
            seen = pos + (-pos[2] / axis[2]) * axis
            h = seen[:2]
        else:
            h = axis[:2]
        tol = VIEW_TOLERANCE_DEG
        if np.linalg.norm(h) < 0.15:  # looking at the base itself / straight up: no side
            err = 90.0
        else:
            err = float(np.degrees(np.arccos(np.clip(_unit(h) @ _unit(want[:2]), -1.0, 1.0))))
    return {"direction": [round(float(v), 3) for v in np.asarray(direction, float)], "frame": frame,
            "direction_robot": [round(float(v), 3) for v in want], "error_deg": round(err, 1),
            "axis": [round(float(v), 3) for v in axis], "camera_position_mm": [round(float(v) * 1000) for v in pos],
            "looks_at_mm": None if seen is None else [round(float(v) * 1000) for v in seen[:2]],
            "pointing": bool(err <= tol), "tolerance_deg": tol}
