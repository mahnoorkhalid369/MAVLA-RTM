"""The test field: graded objects around the arm (pure data + numpy, no Isaac Sim, so tests can use it).

sim/scene.py builds them in the simulator; tests/test_scene_reachability.py checks every object's
difficulty tier against the VT6L's actual reach (top-down pick planned with control/skills.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneObject:
    tier: str  # "easy" | "medium" | "hard" | "impossible" (what the test is meant to probe)
    shape: str  # "box" | "cylinder" | "sphere"
    xy: tuple[float, float]  # centre on the table (m); the arm faces +Y at J1 = 0, +X is its right
    size: tuple[float, float, float]  # box: x, y, z edges; cylinder: (d, d, height); sphere: (d, d, d)
    color: tuple[float, float, float]  # sRGB 0-1 (as on a screen)
    yaw_deg: float = 0.0  # rotation about vertical
    mass_kg: float | None = None  # default: wood density x volume
    on: str | None = None  # stacked on top of this object

    @property
    def height(self) -> float:
        return self.size[2]

    @property
    def mass(self) -> float:
        if self.mass_kg is not None:
            return self.mass_kg
        sx, sy, sz = self.size
        volume = {"box": sx * sy * sz, "cylinder": np.pi * (sx / 2) ** 2 * sz, "sphere": np.pi / 6 * sx**3}[self.shape]
        return WOOD_DENSITY * volume

    @property
    def base_z(self) -> float:
        """Height of the surface it rests on."""
        return 0.0 if self.on is None else OBJECTS[self.on].base_z + OBJECTS[self.on].height


WOOD_DENSITY = 463.0  # kg/m^3: a 60 mm cube weighs 0.1 kg
C = 0.06  # medium cube edge (m)

# Test field: graded difficulty, all around the arm, at different distances.
# Reach facts for the VT6L model (grasp-feasibility map, evaluation/grasp_feasibility_map.py,
# 2026-10-14): top-down picks plan for every grip yaw from 0.45 to 0.90 m from the base in all
# directions except directly behind (J1 stops at +-170 deg); 0.375-0.45 m depends on the bearing and
# the yaw. tests/test_scene_reachability.py and tests/test_collision.py re-check every tier.
# Gripper: 2F-85 opens to 85 mm (use <= 80 mm). Payload: ~4.9 kg after gripper + camera.
OBJECTS = {
    # --- easy: medium cubes, square to the gripper, in front, 0.5-0.73 m
    "red_cube": SceneObject("easy", "box", (0.00, 0.58), (C, C, C), (0.80, 0.10, 0.10)),  # pick_place.py
    "green_cube": SceneObject("easy", "box", (0.15, 0.50), (0.05, 0.05, 0.05), (0.10, 0.70, 0.20)),
    "blue_cube": SceneObject("easy", "box", (-0.15, 0.50), (0.05, 0.05, 0.05), (0.10, 0.25, 0.85)),
    "yellow_cube": SceneObject("easy", "box", (0.10, 0.72), (C, C, C), (0.95, 0.85, 0.10)),
    # --- medium: small, at the sides, round shapes
    "orange_cube": SceneObject("medium", "box", (-0.12, 0.70), (0.035, 0.035, 0.035), (0.90, 0.40, 0.05)),
    "pink_cube": SceneObject("medium", "box", (0.62, 0.10), (0.04, 0.04, 0.04), (1.00, 0.45, 0.70)),
    "black_cube": SceneObject("medium", "box", (-0.60, 0.05), (0.045, 0.045, 0.045), (0.05, 0.05, 0.05)),
    "purple_cylinder": SceneObject("medium", "cylinder", (0.45, 0.45), (0.05, 0.05, 0.07), (0.50, 0.20, 0.75)),
    "gray_cylinder": SceneObject("medium", "cylinder", (-0.32, 0.58), (0.04, 0.04, 0.05), (0.50, 0.50, 0.50)),
    # --- hard: things the current skills do not handle well
    "brown_cube_rotated": SceneObject(  # 45 deg: its 85 mm diagonal is the full gripper opening
        "hard", "box", (0.50, -0.25), (C, C, C), (0.45, 0.28, 0.12), yaw_deg=45.0),
    "teal_sphere": SceneObject("hard", "sphere", (-0.40, -0.35), (0.05, 0.05, 0.05), (0.00, 0.55, 0.55)),  # rolls
    "white_tall_cylinder": SceneObject(  # thin and tall: tips over
        "hard", "cylinder", (0.25, -0.55), (0.025, 0.025, 0.12), (0.95, 0.95, 0.95)),
    "lime_cube_stacked": SceneObject(
        "hard", "box", (-0.15, 0.50), (0.04, 0.04, 0.04), (0.55, 0.85, 0.10), on="blue_cube"),
    "navy_cube_far": SceneObject("hard", "box", (0.00, 0.89), (0.045, 0.045, 0.045), (0.05, 0.10, 0.40)),  # 10 mm inside the 0.90 m far edge
    "olive_cube_near_base": SceneObject(  # 0.43 m: in the partial zone (feasible at yaw 0 straight ahead)
        "hard", "box", (0.00, 0.43), (0.045, 0.045, 0.045), (0.45, 0.45, 0.10)),
    "tan_box_wide": SceneObject("hard", "box", (0.72, -0.08), (0.075, 0.075, 0.05), (0.80, 0.65, 0.45)),  # 75 mm
    "steel_cube_heavy": SceneObject(
        "hard", "box", (-0.25, -0.50), (C, C, C), (0.35, 0.38, 0.42), mass_kg=3.0),
    # --- impossible: the plan should be refused
    "gold_cube_far": SceneObject("impossible", "box", (0.00, 1.15), (0.05, 0.05, 0.05), (0.85, 0.65, 0.15)),
    "silver_cube_behind": SceneObject(  # J1 would need 180 deg
        "impossible", "box", (0.00, -0.62), (0.05, 0.05, 0.05), (0.75, 0.75, 0.80)),
    "beige_cube_too_close": SceneObject(  # 0.34 m at the side: arm cannot fold that far
        "impossible", "box", (0.30, 0.15), (0.045, 0.045, 0.045), (0.95, 0.88, 0.75)),
    "magenta_box_too_wide": SceneObject(  # 100 mm: wider than the gripper opens
        "impossible", "box", (-0.68, -0.12), (0.10, 0.10, 0.05), (0.85, 0.10, 0.65)),
}
BLOCKS = OBJECTS  # older name


# Fixed props (also built by sim/scene.py), needed by the collision checker
MIC_POSITION = (-0.45, 0.35, 0.0)  # floor position (m), front-left of the arm, clear of the blocks
MIC_STEM_HEIGHT = 0.22


def obstacle_boxes(objects: dict | None = None, with_fixtures: bool = True) -> list:
    """Collision boxes (control.collision.Box) for the test objects at their placed poses and the
    fixed props (table-camera stand, microphone). Ground truth: for planning experiments with known
    poses and for evaluation; the agent builds its boxes from perception instead."""
    from control.collision import Box  # local import: keeps this module importable without control/
    from control.robot import TABLE_CAMERA_POSITION

    boxes = []
    for name, o in (objects or OBJECTS).items():
        a = np.radians(o.yaw_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        centre = np.array([o.xy[0], o.xy[1], o.base_z + o.height / 2])
        boxes.append(Box(name, centre, R, np.asarray(o.size, float) / 2))  # cylinders / spheres: bounding box
    if with_fixtures:
        x, y, z = TABLE_CAMERA_POSITION
        boxes += [
            Box("camera_stand_pole", np.array([x, y, z / 2]), np.eye(3), np.array([0.015, 0.015, z / 2])),
            Box("camera_stand_foot", np.array([x, y, 0.01]), np.eye(3), np.array([0.15, 0.15, 0.01])),
            Box("microphone_base", np.array([MIC_POSITION[0], MIC_POSITION[1], 0.006]), np.eye(3), np.array([0.055, 0.055, 0.006])),
            Box("microphone_stand", np.array([MIC_POSITION[0], MIC_POSITION[1], 0.16]), np.eye(3), np.array([0.023, 0.023, 0.15])),
        ]
    return boxes
