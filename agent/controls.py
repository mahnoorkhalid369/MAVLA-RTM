"""Execution controls of one agent process: explicit switches for experiments, passed to the constructors
(Tools, GoalSupervisor, Session, agent.run.main) instead of replacing production functions at runtime.
ExecutionControls() is normal operation.

    motion           "real"     planned motions are executed (frozen RosExecutor + the simulator's safety layer)
                     "dry_run"  planned motions are logged (event tool=execute_dry_run) and NOT executed; the
                                motion step fails ("dry run: ..."), so nothing downstream assumes the robot moved
                     "blocked"  planned motions are NOT executed; the step fails with blocked_reason
    disabled         capability names shown to Gemma as "(not available yet)" and refused with outcome
                     "unsupported" if a plan names them anyway
    max_replans      Gemma decision budget per instruction: how many times the supervisor may ask Gemma what to do
                     next after the first plan (a failure, an uncertain result, a goal not shown); 0 = never
    disabled_reason  completes "<capability> is disabled <reason>" in the refusal
    blocked_reason   the failure reason of a blocked motion
    on_event         optional callback(event, **fields) for a harness log: capability_refused, motion_refused,
                     motion_allowed, motion_result

`motion` may change while the process runs (e.g. by a directive line, agent.ask.LineReader.on_directive); Tools
reads it at every execute. Safety is unaffected by every setting: the controls can only prevent motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MOTION_MODES = ("real", "dry_run", "blocked")


@dataclass
class ExecutionControls:
    motion: str = "real"
    disabled: frozenset = frozenset()
    max_replans: int = 3
    disabled_reason: str = "by the execution controls"
    blocked_reason: str = "motion blocked by the execution controls"
    on_event: Callable | None = None

    def __post_init__(self) -> None:
        if self.motion not in MOTION_MODES:
            raise ValueError(f"motion must be one of {MOTION_MODES}, not {self.motion!r}")
        self.disabled = frozenset(self.disabled)

    def notify(self, event: str, **fields) -> None:
        if self.on_event is not None:
            self.on_event(event, **fields)
