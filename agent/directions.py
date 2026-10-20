"""Directions in an explicit frame, and where the wrist camera actually points (measured).

Gemma names a direction and a frame (the "look" capability); this module turns them into a unit vector and measures
the wrist camera's optical axis from joint feedback (forward kinematics of the measured joint state). Pure geometry:
no words, no scene knowledge.

    robot frame   the robot's base: x = the robot's right, y = forward, z = up (the world frame of the agent)
    camera frame  as seen in the fixed (table) camera's picture: right = image right, forward = the camera's
                  viewing direction on the floor
    down          straight down in either frame
"""

from __future__ import annotations

import numpy as np

DIRECTIONS = ("left", "right", "forward", "backward", "down")
FRAMES = ("robot", "camera")
VIEW_TOLERANCE_DEG = 30.0  # a horizontal direction: the looked-at floor point within this bearing of it
DOWN_TOLERANCE_DEG = 15.0  # 'down': the optical axis within this angle of straight down (HOME's camera is 30 deg off)
CAMERA_TIP = "camera_color_optical_frame"  # the wrist camera frame (URDF); its +z is the optical axis


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def direction_vector(direction: str, frame: str = "robot") -> np.ndarray:
    """Unit vector (world / robot base coordinates) of a named direction in the given frame."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r} (one of {DIRECTIONS})")
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r} (one of {FRAMES})")
    if direction == "down":
        return np.array([0.0, 0.0, -1.0])
    if frame == "camera":
        from control.robot import table_camera_pose
        R = table_camera_pose()[:3, :3]
        right, forward = _unit(R[:2, 0]), _unit(R[:2, 2])  # image right; the viewing direction on the floor
    else:
        right, forward = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    h = {"left": -right, "right": right, "forward": forward, "backward": -forward}[direction]
    return np.array([h[0], h[1], 0.0])


def camera_pose(q) -> np.ndarray:
    """4x4 pose of the wrist camera for a joint state (forward kinematics; J6 does not move the camera)."""
    from control.kinematics import Kinematics
    kin = Kinematics(tip=CAMERA_TIP)
    return kin.fk(np.asarray(q, float)[:len(kin.lower)])


def view_error(q, direction: str, frame: str = "robot") -> dict:
    """How far the wrist camera (joint state q) is from looking `direction`.
    Horizontal directions: where the optical axis meets the floor, as a bearing from the base, against the direction
    (what the camera looks at lies on that side of the robot; a camera that looks down beside itself does not count
    as looking sideways). An axis that does not meet the floor in front of it uses its heading. 'down' compares the
    full axis with straight down."""
    T = camera_pose(q)
    axis, pos = T[:3, 2], T[:3, 3]
    want = direction_vector(direction, frame)
    seen = None
    if direction == "down":
        err = float(np.degrees(np.arccos(np.clip(axis @ want, -1.0, 1.0))))
    else:
        if axis[2] < -0.1:  # meets the floor (z = 0)
            seen = pos + (-pos[2] / axis[2]) * axis
            h = seen[:2]
        else:
            h = axis[:2]
        if np.linalg.norm(h) < 0.15:  # looking at the base itself / straight up: no side
            err = 90.0
        else:
            err = float(np.degrees(np.arccos(np.clip(_unit(h) @ want[:2], -1.0, 1.0))))
    tol = DOWN_TOLERANCE_DEG if direction == "down" else VIEW_TOLERANCE_DEG
    return {"direction": direction, "frame": frame, "error_deg": round(err, 1),
            "axis": [round(float(v), 3) for v in axis], "camera_position_mm": [round(float(v) * 1000) for v in pos],
            "looks_at_mm": None if seen is None else [round(float(v) * 1000) for v in seen[:2]],
            "pointing": bool(err <= tol), "tolerance_deg": tol}
