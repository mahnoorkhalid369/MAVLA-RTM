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
VIEW_TOLERANCE_DEG = 30.0  # the measured view counts as pointing that way within this angle
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
    """How far the wrist camera's optical axis (from joint state q) is from pointing `direction`.
    Horizontal directions compare the axis's heading on the floor (the camera may look down while it looks left);
    'down' compares the full axis with straight down."""
    T = camera_pose(q)
    axis = T[:3, 2]
    want = direction_vector(direction, frame)
    if direction == "down":
        err = float(np.degrees(np.arccos(np.clip(axis @ want, -1.0, 1.0))))
    else:
        h = axis[:2]
        if np.linalg.norm(h) < 0.2:  # looking (almost) straight down: no heading
            err = 90.0
        else:
            err = float(np.degrees(np.arccos(np.clip(_unit(h) @ want[:2], -1.0, 1.0))))
    return {"direction": direction, "frame": frame, "error_deg": round(err, 1),
            "axis": [round(float(v), 3) for v in axis], "camera_position_mm": [round(float(v) * 1000) for v in T[:3, 3]],
            "pointing": bool(err <= VIEW_TOLERANCE_DEG), "tolerance_deg": VIEW_TOLERANCE_DEG}
