"""Perception configuration: depth-measurement thresholds and the fixed fixtures that are never objects.

Used by agent/geometry.py (measurement and exclusion) and agent/tools (planning obstacles).
"""

from __future__ import annotations

from dataclasses import dataclass

from control.robot import TABLE_CAMERA_POSITION

GROUND_Z = 0.0  # m: the support surface (floor) of the work cell
ABOVE_SUPPORT = 0.003  # m: depth points closer than this to the floor inside a mask edge are floor
SUPPORT_SNAP = 0.02  # m: the lowest VISIBLE point of an object sits up to this far above what it stands on
#   (mask-edge erosion ~5 mm at 2 m, the 2nd percentile, neighbours hiding the bottom edge): snap to the
#   highest real surface (floor or a measured top) within this distance below it
UP_NORMAL = 0.9  # |normal z| above this (< ~25 deg from vertical) = an upward-facing surface pixel
TOP_LAYER = 0.012  # m: points this close below the top count as the top face
MASK_ERODE_PX = 2  # pixels removed at mask edges (depth bleeding)
MIN_POINTS = 30  # depth points needed to measure an object
ROBOT_INFLATE = 0.01  # m: robot collision spheres are inflated by this for exclusion
FIXTURE_INFLATE = 0.01

# Floor position (m) of the microphone prop of the default scene: the same value as the frozen
# sim/scene_objects.MIC_POSITION (copied so the agent does not import simulator modules).
MIC_POSITION = (-0.45, 0.35, 0.0)


@dataclass(frozen=True)
class FixtureBox:
    name: str
    centre: tuple[float, float, float]
    half: tuple[float, float, float]


def _fixtures() -> list[FixtureBox]:
    x, y, z = TABLE_CAMERA_POSITION
    mx, my, _ = MIC_POSITION
    return [FixtureBox("camera_stand_pole", (x, y, z / 2), (0.015, 0.015, z / 2)),
            FixtureBox("camera_stand_foot", (x, y, 0.01), (0.15, 0.15, 0.01)),
            FixtureBox("microphone_base", (mx, my, 0.006), (0.055, 0.055, 0.006)),
            FixtureBox("microphone_stand", (mx, my, 0.16), (0.023, 0.023, 0.15))]


FIXTURES = _fixtures()

# Robot base column (fixed robot geometry, configuration): the simplified collision spheres are smaller
# than the rendered base + J1 housing, so depth points inside this box are robot, never an object.
# Used for perception exclusion ONLY (not a planning obstacle: the arm itself is inside it). The base
# plate is 0.240 x 0.285 m (URDF); the closest workspace position is 0.34 m from the base axis.
ROBOT_BASE_EXCLUSION = FixtureBox("robot_base_column", (0.0, 0.0, 0.25), (0.15, 0.16, 0.25))
