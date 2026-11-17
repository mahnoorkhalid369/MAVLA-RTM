"""Robot constants shared by the simulator, the experiments and the ROS 2 nodes.

Plain Python + numpy: no Isaac Sim or ROS imports, so every part of the project can use it.
Joint limits come from the URDF via control.kinematics.
"""

from __future__ import annotations

import os

import numpy as np

ARM_JOINTS = [f"j{i}_joint" for i in range(1, 7)]
ARM_HOME = np.deg2rad([0.0, 0.0, 0.0, 0.0, -90.0, 0.0])  # tool pointing down at the blocks

# Robotiq 2F-85 ("mimic" variant): one driven joint, 0 = open 85 mm ... 47 deg = closed
GRIPPER_JOINT = "finger_joint"
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = np.deg2rad(45.0)  # drive saturates on contact, limit is 47 deg

# 2F-85 calibration, measured in the simulator (2026-10-14) from the pad poses on /tf and the asset's
# pad faces: finger_joint angle -> gap between the (parallel) pads, and how far the fingertips have
# moved AWAY from the flange compared with fully open (the linkage swings the pads down as they close).
GRIPPER_CAL_DEG = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
GRIPPER_CAL_GAP = np.array([87.0, 79.1, 70.8, 62.0, 52.8, 43.3, 33.6, 23.7, 13.8, 3.8]) / 1000  # m
GRIPPER_CAL_TIP_DROP = np.array([0.0, 3.1, 5.8, 8.1, 10.1, 11.6, 12.8, 13.5, 13.7, 13.6]) / 1000  # m
FINGERTIP_BELOW_TCP_OPEN = 0.0147  # m: pad tips beyond the TCP with the gripper open (asset geometry)


def gripper_gap(finger_rad: float) -> float:
    """Gap between the pads (m) at a finger_joint angle (rad)."""
    return float(np.interp(np.degrees(finger_rad), GRIPPER_CAL_DEG, GRIPPER_CAL_GAP))


def finger_angle_for_gap(gap: float) -> float:
    """finger_joint angle (rad) at which the pads are `gap` (m) apart."""
    return float(np.radians(np.interp(gap, GRIPPER_CAL_GAP[::-1], GRIPPER_CAL_DEG[::-1])))


def fingertip_below_tcp(gap: float) -> float:
    """How far (m) the fingertips reach beyond the TCP when the pads are `gap` apart."""
    drop = np.interp(gap, GRIPPER_CAL_GAP[::-1], GRIPPER_CAL_TIP_DROP[::-1])
    return float(FINGERTIP_BELOW_TCP_OPEN + drop)


# Default speeds for commanded moves (the simulator ramps targets at these rates)
ARM_SPEED = np.deg2rad(30.0)  # rad/s
GRIPPER_SPEED = np.deg2rad(90.0)  # rad/s
TCP_SPEED = 0.08  # m/s, peak gripper speed on straight-line moves

# Where the wrist camera sees most of the table without the open gripper in the way
# (3 of the 5 test blocks fully visible; found by projecting the blocks through FK + intrinsics)
OBSERVE_POSE = np.deg2rad([0.0, -5.0, -50.0, 45.0, -5.0, 0.0])

# Camera views for exploring, combined with any J1 heading (the camera's clear upper image area
# looks at the floor this far from the base; found with FK + the camera intrinsics)
VIEW_POSES = {
    "near": np.deg2rad([0.0, -5.0, -60.0, 45.0, -10.0, 0.0]),  # camera axis meets the floor ~0.37 m out
    "mid": OBSERVE_POSE,  # ~0.55 m
    "far": np.deg2rad([0.0, -20.0, -25.0, 45.0, -10.0, 0.0]),  # ~0.78 m
}

# Fixed table camera (eye-to-hand): a D435 on a stand at the robot's right (+x), 1.5 m from the
# base, 1.8 m high, looking down ~50 deg at the work area. From there every test object is in view
# and none is hidden behind the robot's base (the moving arm can still block some, as in a real cell).
TABLE_CAMERA_POSITION = (1.5, 0.25, 1.8)  # m, world
TABLE_CAMERA_TARGET = (0.0, 0.25, 0.0)  # point on the floor at the image centre


def _scenario_camera(value: str | None):
    """A scenario with its own work cell mounts the table camera elsewhere (scenarios/<name>.py TABLE_CAMERA). The
    launchers export it as TABLE_CAMERA="ex,ey,ez,tx,ty,tz" (m) to the simulator and the agent alike, before anything
    imports this module; unset or empty = the frozen cell's camera above."""
    if not value:
        return None
    v = [float(c) for c in value.split(",")]
    if len(v) != 6:
        raise ValueError(f"TABLE_CAMERA={value!r}: six numbers ex,ey,ez,tx,ty,tz (m)")
    return tuple(v[:3]), tuple(v[3:])


# a work cell with its own camera frame has neither the frozen camera stand nor the microphone (scenarios/build.py hides them)
SCENARIO_CELL = bool(_scenario_camera(os.environ.get("TABLE_CAMERA")))
if SCENARIO_CELL:
    TABLE_CAMERA_POSITION, TABLE_CAMERA_TARGET = _scenario_camera(os.environ["TABLE_CAMERA"])


def table_camera_pose() -> np.ndarray:
    """4x4 pose of the table camera's ROS optical frame (z forward, x right, y down) in the world."""
    eye = np.asarray(TABLE_CAMERA_POSITION, float)
    z = np.asarray(TABLE_CAMERA_TARGET, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])  # image right: horizontal
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.column_stack([x, np.cross(z, x), z]), eye
    return T
