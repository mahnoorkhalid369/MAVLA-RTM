"""Supervisor <-> worker contracts and the worker registry.

The supervisor delegates every subtask as a WorkerRequest and gets a WorkerResult back. Workers never start other
workers, never change the goal and never declare the user's task complete: they report what they did, what they
observed, the evidence, their uncertainty and (advisory) what they would need next. Only the supervisor decides.
Workers are logical roles in this process (one shared Gemma / YOLO / ROS bridge), each owning a set of
operations implemented by agent/capabilities/ over the tools (agent/tools/):

    perception     observe, describe_scene, answer, locate
    inspection     inspect, explore, look (wrist-camera viewpoints; moves the arm, never objects)
    manipulation   pick, place, pick_place, return_home
    planning       decompose (Gemma over the measured objects)
    verification   verify (goal / sub-goal conditions over measured state; internal, never planned by Gemma)

    reg = WorkerRegistry(log, executors, specs)
    req = reg.request("put a on b", "pick_place", args, objective="pick_place a on b", context={...}, constraints={...})
    res = reg.dispatch(req, ctx, authority)   # -> WorkerResult (status, outputs, evidence, uncertainty, needs, tools)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

STATUSES = ("pending", "running", "completed", "failed", "blocked", "cancelled", "insufficient_info")

# step outcome (agent/task_state.OUTCOMES) -> worker status
STATUS_OF = {"completed": "completed", "insufficient_info": "insufficient_info", "ambiguous": "blocked",
             "unsupported": "blocked", "infeasible": "failed", "unsafe": "failed", "failed": "failed",
             "cancelled": "cancelled", "unauthorized": "blocked"}

@dataclass
class WorkerRequest:
    task_id: str
    goal_id: str  # short id of the user's goal (the instruction's first words)
    worker: str  # the receiver
    capability: str  # the requested operation
    objective: str  # the current sub-goal, in words
    args: dict  # the operation's inputs
    object_refs: list = field(default_factory=list)  # world ids or reference texts involved
    state_refs: list = field(default_factory=list)  # e.g. observation frame ids the subtask relies on
    preconditions: list = field(default_factory=list)
    expected_effect: str = ""
    safety: str = "read_only"  # read_only | motion
    budget: dict = field(default_factory=dict)  # {"retries": n, "seconds": s}
    depends_on: list = field(default_factory=list)  # task ids
    sender: str = "supervisor"
    receiver: str = ""  # = worker
    goal: str = ""  # the user's original goal (full instruction)
    context: dict = field(default_factory=dict)  # relevant task / robot state (held object, bindings, last frame, ...)
    constraints: dict = field(default_factory=dict)  # allowed effects, safety class, motion mode, location constraint
    expected: list = field(default_factory=list)  # conditions the sub-goal should make true (agent/goals.py predicates)


@dataclass
class WorkerResult:
    task_id: str
    worker: str
    status: str  # STATUSES
    outcome: str  # agent/task_state.OUTCOMES (finer classification)
    summary: str
    data: dict = field(default_factory=dict)  # the worker's outputs / observations
    evidence: list = field(default_factory=list)  # references: {"kind": "frame"|"verification"|"vlm", ...}
    world_changes: list = field(default_factory=list)
    confidence: str = "measured"  # measured | model (unverified) | none
    failure_reason: str = ""
    suggestion: str = ""  # advisory only
    metrics: dict = field(default_factory=dict)
    sender: str = ""  # = worker
    receiver: str = "supervisor"
    tools: list = field(default_factory=list)  # tools and Gemma roles the worker invoked, in order
    uncertainty: list = field(default_factory=list)  # what the worker is not sure about
    needs: str = ""  # what the worker itself says it needs (advisory; the supervisor / Gemma decides)


@dataclass
class WorkerSpec:
    name: str
    responsibility: str
    operations: dict  # capability -> executor(ctx, args, req) -> StepResult
    can_move_robot: bool
    status: str = "available"  # available | running | blocked | failed
    current_task: str | None = None
    last_result: dict | None = None


def assess(capability: str, outcome: str, data: dict, confidence: str) -> tuple[list, str]:
    """Uncertainty of a worker result, stated as facts from its outcome and outputs, and what the worker itself
    says it needs (data["needs"], if any). Python does not suggest which operation should come next: Gemma decides."""
    unc = []
    if confidence == "model":
        unc.append("model-generated output (Gemma), not measured")
    if data.get("certain") is False:
        unc.append("the answer is not certain from the current view")
    if data.get("check_closer_world"):
        unc.append(f"Gemma named objects a closer look could settle: {data['check_closer_world']}")
    if outcome == "ambiguous":
        unc.append("several objects match")
    elif outcome == "insufficient_info":
        unc.append("the evidence does not settle the sub-goal")
    if data.get("unknown_conditions"):
        unc.append("not decidable from the evidence: " + ", ".join(data["unknown_conditions"]))
    return unc, str(data.get("needs") or "")


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

    def worker_of(self, capability: str) -> str | None:
        """The worker that implements an operation; None if no worker does (the operation is unavailable)."""
        return next((n for n, w in self.workers.items() if capability in w.operations), None)

    def request(self, goal_id: str, capability: str, args: dict, objective: str = "", object_refs=None,
                state_refs=None, depends_on=None, goal: str = "", context: dict | None = None,
                constraints: dict | None = None, expected: list | None = None) -> WorkerRequest:
        spec = self.specs.get(capability)
        worker = self.worker_of(capability)
        return WorkerRequest(task_id=f"t{next(self._ids)}", goal_id=goal_id, worker=worker, capability=capability,
                             objective=objective or capability, args=dict(args or {}), object_refs=list(object_refs or []),
                             state_refs=list(state_refs or []), preconditions=[spec.preconditions] if spec else [],
                             expected_effect=spec.effects if spec else "", safety=spec.safety if spec else "read_only",
                             budget={}, depends_on=list(depends_on or []), receiver=worker, goal=goal,
                             context=dict(context or {}), constraints=dict(constraints or {}), expected=list(expected or []))

    def dispatch(self, req: WorkerRequest, ctx, authority=None) -> WorkerResult:
        """Run one operation. Authorization comes first: a capability whose effect class (CapabilitySpec.effect)
        is not granted by the request's authority is refused here, before its executor runs, whatever plan
        asked for it. authority None = least privilege."""
        w = self.workers[req.worker]
        spec = self.specs.get(req.capability)
        effect = getattr(spec, "effect", "object_change")  # unknown -> most restricted
        grants = tuple(getattr(authority, "grants", None) or ("none", "sensing"))
        if effect not in grants:
            why = (f"{req.capability} would change objects ({effect}); the request ("
                   f"{getattr(authority, 'kind', 'unknown')}) does not ask for that, so it is not authorized")
            self.log("authorization", task_id=req.task_id, capability=req.capability, effect=effect, grants=list(grants),
                     allowed=False, reason=why)
            res = WorkerResult(req.task_id, req.worker, "blocked", "unauthorized", why, {}, confidence="none", failure_reason=why,
                               sender=req.worker)
            self.log("worker_result", **{k: v for k, v in asdict(res).items() if k != "data"}, data_keys=[])
            return res
        w.status, w.current_task = "running", req.task_id
        self.log("delegate", **{k: v for k, v in asdict(req).items() if k not in ("preconditions",)})
        t0 = time.monotonic()
        vlm_before = getattr(ctx, "vlm_calls", lambda: 0)()
        mark = self.log.mark() if hasattr(self.log, "mark") else None
        try:
            r = w.operations[req.capability](ctx, req.args, req)
        except Exception as exc:  # a worker crash is a failed subtask, never a success
            from agent.capabilities import StepResult
            r = StepResult("failed", f"{req.worker} worker error: {exc!r}")
        status = STATUS_OF.get(r.outcome, "failed")
        data = dict(r.data or {})
        confidence = data.get("confidence", "measured" if status == "completed" else "none")
        uncertainty, needs = assess(req.capability, r.outcome, data, confidence)
        res = WorkerResult(req.task_id, req.worker, status, r.outcome, r.summary, data,
                           evidence=list(data.get("evidence", [])),
                           world_changes=list(data.get("world_changes", [])),
                           confidence=confidence,
                           failure_reason="" if status == "completed" else r.summary,
                           suggestion=data.get("suggestion", ""),
                           metrics={"seconds": round(time.monotonic() - t0, 2),
                                    "gemma_calls": getattr(ctx, "vlm_calls", lambda: 0)() - vlm_before},
                           sender=req.worker, tools=self.log.since(mark) if mark is not None else [],
                           uncertainty=uncertainty, needs=needs)
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
    "perception": "acquire observations from the cameras, detect and measure objects, keep the world model, describe / "
                  "ground references, answer from the measured objects; reports observations and uncertainty",
    "inspection": "choose and reach wrist-camera viewpoints for closer or hidden views (motion through the authorized, "
                  "collision-checked executor); returns refined observations",
    "manipulation": "interpret a requested physical action, obtain grasp / motion plans, execute them through the "
                    "existing executor, report execution results and failures",
    "planning": "decompose a multi-object goal into subtasks over the measured scene (Gemma)",
    "verification": "compare measured state with a goal or sub-goal's conditions; return evidence of success, failure "
                    "or uncertainty (never declares the user's task complete)",
}
