"""Whether planned motions are sent to the robot. Safety does not depend on it: it can only prevent motion.

    real     planned motions are executed (execution/ros_executor.py, checked again by the simulator's safety layer)
    dry_run  planned motions are logged and NOT executed; the step fails ("dry run: ..."), so nothing assumes it moved
    blocked  planned motions are NOT executed; the step fails with blocked_reason
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MOTION_MODES = ("real", "dry_run", "blocked")


@dataclass
class ExecutionControls:
    motion: str = "real"
    blocked_reason: str = "motion blocked by the execution controls"
    on_event: Callable | None = None  # optional callback(event, **fields) for an experiment's own log

    def __post_init__(self) -> None:
        if self.motion not in MOTION_MODES:
            raise ValueError(f"motion must be one of {MOTION_MODES}, not {self.motion!r}")

    def notify(self, event: str, **fields) -> None:
        if self.on_event is not None:
            self.on_event(event, **fields)
