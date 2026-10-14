"""Fixed knowledge of the agent: physical and safety configuration ONLY (no object knowledge).

Everything about objects is discovered (masks), measured (depth) and described (VLM) at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from control.robot import TABLE_CAMERA_POSITION
from sim.scene_objects import MIC_POSITION  # physical position of the microphone prop (configuration)

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


@dataclass
class Tolerances:
    """Placement success (the same numbers as Baseline A); never loosened to pass a test."""
    xy_m: float = 0.010
    z_m: float = 0.005
    tilt_deg: float = 5.0
    contact_deg: float = 1.5  # fingers stopping this far before the close target = contact
    min_support_width_m: float = 0.030  # "on": the support's top must be at least this wide


@dataclass
class Budgets:
    """Recovery budgets per failure class (approved)."""
    retries: dict = field(default_factory=lambda: {"perception": 2, "planning": 2, "grasp": 1, "execution": 0,
                                                    "placement": 1, "understanding": 0})
    max_questions: int = 2
    question_timeout_s: float = 120.0


@dataclass
class Relations:
    """Placement relations (robot frame: +x = robot's right, +y = forward from the base)."""
    gap_m: float = 0.02  # between footprints for next_to / directional relations
    search_step_m: float = 0.01
    search_max_m: float = 0.10
    search_angles_deg: tuple = (0.0, 15.0, -15.0, 30.0, -30.0)
    relation_check_max_gap_m: float = 0.15  # relational REFERENCE check: target within this of the anchor


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)
