"""Supervisor <-> worker contracts, the worker registry and the task ledger.

The supervisor delegates every subtask as a WorkerRequest and gets a WorkerResult back; workers never start
other workers or change the goal (their `suggestion` is advisory). Workers are logical roles in this process
(one shared Gemma / YOLO / ROS bridge), each owning a set of operations implemented over the existing,
validated code (agent/capabilities.py executors, PickPlaceWorkflow, the viewpoint planner, Tools checks).

    reg = WorkerRegistry(log)
    req = reg.request(state, capability="pick", args={...}, objective="hold the dark brown block", ...)
    res = reg.dispatch(req, ctx)          # -> WorkerResult (status, data, evidence, world changes, metrics)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

STATUSES = ("pending", "running", "completed", "failed", "blocked", "cancelled", "insufficient_info")

# step outcome (agent/world_state.OUTCOMES) -> worker status
STATUS_OF = {"completed": "completed", "insufficient_info": "insufficient_info", "ambiguous": "blocked",
             "unsupported": "blocked", "infeasible": "failed", "unsafe": "failed", "failed": "failed",
             "cancelled": "cancelled"}


@dataclass
class WorkerRequest:
    task_id: str
    goal_id: str
    worker: str
    capability: str  # the worker operation
    objective: str  # what this subtask must achieve (for the record and the viewer)
    args: dict
    object_refs: list = field(default_factory=list)  # world ids or reference texts involved
    state_refs: list = field(default_factory=list)  # e.g. observation frame ids the subtask relies on
    preconditions: list = field(default_factory=list)
    expected_effect: str = ""
    safety: str = "read_only"  # read_only | motion
    budget: dict = field(default_factory=dict)  # {"retries": n, "seconds": s}
    depends_on: list = field(default_factory=list)  # task ids


@dataclass
class WorkerResult:
    task_id: str
    worker: str
    status: str  # STATUSES
    outcome: str  # agent/world_state.OUTCOMES (finer classification)
    summary: str
    data: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)  # references: {"kind": "frame"|"verification"|"vlm", ...}
    world_changes: list = field(default_factory=list)
    confidence: str = "measured"  # measured | model (unverified) | none
    failure_reason: str = ""
    suggestion: str = ""  # advisory only
    metrics: dict = field(default_factory=dict)


@dataclass
class WorkerSpec:
    name: str
    responsibility: str
    operations: dict  # capability -> executor(ctx, args, req) -> StepResult
    can_move_robot: bool
    status: str = "available"  # available | running | blocked | failed
    current_task: str | None = None
    last_result: dict | None = None


class WorkerRegistry:
    """Which workers exist, what they do, their live status; dispatch = run one operation, wrap the result."""

    def __init__(self, log: Callable, executors: dict[str, tuple[str, Callable]], specs: dict) -> None:
        """executors: capability -> (worker name, fn(ctx, args, req) -> StepResult); specs: capability -> CapabilitySpec."""
        self.log, self.specs = log, specs
        self.workers: dict[str, WorkerSpec] = {}
        for cap, (wname, fn) in executors.items():
            w = self.workers.setdefault(wname, WorkerSpec(wname, RESPONSIBILITY.get(wname, ""), {}, False))
            w.operations[cap] = fn
            w.can_move_robot |= getattr(specs.get(cap), "safety", "read_only") == "motion"
        self._ids = itertools.count(1)

    def worker_of(self, capability: str) -> str:
        return next(n for n, w in self.workers.items() if capability in w.operations)

    def request(self, goal_id: str, capability: str, args: dict, objective: str = "", object_refs=None,
                state_refs=None, depends_on=None) -> WorkerRequest:
        spec = self.specs.get(capability)
        return WorkerRequest(f"t{next(self._ids)}", goal_id, self.worker_of(capability), capability, objective or capability,
                             dict(args or {}), list(object_refs or []), list(state_refs or []),
                             [spec.preconditions] if spec else [], spec.effects if spec else "",
                             spec.safety if spec else "read_only", {}, list(depends_on or []))

    def dispatch(self, req: WorkerRequest, ctx) -> WorkerResult:
        w = self.workers[req.worker]
        w.status, w.current_task = "running", req.task_id
        self.log("delegate", **{k: v for k, v in asdict(req).items() if k not in ("preconditions",)})
        t0 = time.monotonic()
        vlm_before = getattr(ctx, "vlm_calls", lambda: 0)()
        try:
            r = w.operations[req.capability](ctx, req.args, req)
        except Exception as exc:  # a worker crash is a failed subtask, never a success
            from agent.capabilities import StepResult
            r = StepResult("failed", f"{req.worker} worker error: {exc!r}")
        status = STATUS_OF.get(r.outcome, "failed")
        res = WorkerResult(req.task_id, req.worker, status, r.outcome, r.summary, dict(r.data or {}),
                           evidence=list((r.data or {}).get("evidence", [])),
                           world_changes=list((r.data or {}).get("world_changes", [])),
                           confidence=(r.data or {}).get("confidence", "measured" if status == "completed" else "none"),
                           failure_reason="" if status == "completed" else r.summary,
                           suggestion=(r.data or {}).get("suggestion", ""),
                           metrics={"seconds": round(time.monotonic() - t0, 2),
                                    "gemma_calls": getattr(ctx, "vlm_calls", lambda: 0)() - vlm_before})
        w.status = {"completed": "available", "blocked": "blocked", "failed": "failed"}.get(status, "available")
        w.current_task, w.last_result = None, {"task_id": req.task_id, "status": status, "summary": r.summary[:200]}
        self.log("worker_result", **{k: v for k, v in asdict(res).items() if k != "data"},
                 data_keys=sorted(res.data))
        return res

    def snapshot(self) -> dict:
        return {n: {"responsibility": w.responsibility, "operations": sorted(w.operations), "can_move_robot": w.can_move_robot,
                    "status": w.status, "current_task": w.current_task, "last_result": w.last_result}
                for n, w in self.workers.items()}


RESPONSIBILITY = {
    "perception": "observe with the cameras, measure objects, describe / ground references, answer from evidence",
    "exploration": "choose and reach wrist-camera viewpoints to reduce uncertainty (inspect, look behind)",
    "manipulation": "grasp, transport + release, composite pick-and-place, return to a safe pose",
    "planning": "decompose a goal into subtasks over the measured scene",
    "verification": "check evidence and the goal's final-state conditions",
}
