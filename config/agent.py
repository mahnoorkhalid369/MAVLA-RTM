"""Agent task configuration: placement success tolerances, recovery budgets and placement relations."""

from __future__ import annotations

from dataclasses import dataclass, field


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
