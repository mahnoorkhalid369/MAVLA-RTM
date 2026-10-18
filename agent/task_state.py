"""Shared task state of the goal supervisor: one inspectable object per instruction, written to
<run_dir>/task_state.json after every step.

It keeps the kinds of information apart:
  facts        what sensors measured (object geometry from depth, detector scores)
  hypotheses   what Gemma said (object descriptions, interpretations, answers): never physical ground truth
  assumptions  what the supervisor assumed without measuring it (e.g. which object "it" means)
  plan         the planned capability steps, their status and each worker's structured result
  verified     outcomes confirmed by independent physical checks (contact, camera placement check, HOME joints)
  subgoal_checks / goal_check   the verification worker's evidence per sub-goal / for the final goal
Simulator ground truth never enters this ledger.

Ownership: the supervisor owns this ledger (one per instruction). The world model, the held object and the
session memory are owned by the session's Tools (agent/tools/); `robot` here is a snapshot of them, refreshed after
every supervisor state, for the record and for Gemma's re-planning context.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Outcome classes for a step or a whole goal (requested distinction; "completed" = verified success)
OUTCOMES = ("completed", "unsupported", "insufficient_info", "ambiguous", "infeasible", "unsafe", "cancelled", "failed",
            "unauthorized")  # unauthorized: the user's request does not permit this effect (agent/authority.py)


@dataclass
class Step:
    index: int
    capability: str
    args: dict
    status: str = "pending"  # pending | running | done | failed | skipped | replaced
    outcome: str | None = None  # one of OUTCOMES once finished
    summary: str = ""
    seconds: float = 0.0
    worker: str = ""  # assigned worker (agent.workers)
    task_id: str = ""  # WorkerRequest.task_id once delegated
    depends_on: list = field(default_factory=list)  # indices of steps that must be done first
    result: dict = field(default_factory=dict)  # WorkerResult (without bulky data)
    origin: str = "planner"  # planner | completion rule | subtask policy | decomposition | replan


@dataclass
class TaskState:
    instruction: str
    goal: str = ""  # the planner's one-line interpretation (hypothesis)
    success_condition: str = ""  # the planner's statement of what "done" means (hypothesis)
    plan: list[Step] = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # {"objects": [...], "frame_id", "taken_at", "view_blocked", ...}
    hypotheses: dict = field(default_factory=dict)  # {"descriptions": {world_id: {...}}, "answers": [...]}
    verified: list[dict] = field(default_factory=list)  # independent verification records
    actions: list[dict] = field(default_factory=list)  # capability calls with outcome and duration
    questions: list[dict] = field(default_factory=list)
    replans: int = 0
    bindings: dict = field(default_factory=dict)  # reference text / "#id" -> world object id (from worker results)
    conditions: list = field(default_factory=list)  # desired final state (agent.goals predicates)
    goal_check: list = field(default_factory=list)  # evaluation of the conditions (satisfied, confidence, evidence)
    attempted: list = field(default_factory=list)  # (capability, args) already tried and failed (no identical retry)
    authority: dict = field(default_factory=dict)  # RequestAuthority of this instruction (from the user's words only)
    refused: list = field(default_factory=list)  # steps refused for lack of authority (never executed)
    workers: dict = field(default_factory=dict)  # registry snapshot at the end
    robot: dict = field(default_factory=dict)  # snapshot: held object, last referent, motion count, motion mode
    sensors: list = field(default_factory=list)  # available sensors (from the tools)
    capabilities: list = field(default_factory=list)  # capabilities available to the planner for this instruction
    uncertainties: list = field(default_factory=list)  # [{task_id, capability, items}] from worker results
    assumptions: list = field(default_factory=list)  # supervisor assumptions (not measured), e.g. pronoun bindings
    subgoal_checks: list = field(default_factory=list)  # verification worker results per completed sub-goal
    consults: list = field(default_factory=list)  # Gemma re-planning decisions: [{reason, accepted, why}]
    status: str = "running"  # running | completed | failed | cancelled | unsupported
    outcome: str | None = None
    message: str = ""
    started: float = field(default_factory=time.time)

    def current(self) -> Step | None:
        return next((s for s in self.plan if s.status in ("pending", "running")), None)

    def remaining(self, after: Step) -> list[Step]:
        return [s for s in self.plan if s.index > after.index and s.status == "pending"]

    def record_observation(self, obs) -> None:
        """Facts from an agent.tools.Observation (measured) and its descriptions (Gemma hypotheses)."""
        objs = []
        for k in obs.marks:
            g = obs.measured.get(k, {})
            objs.append({"mark": k, "world_id": obs.object_ids.get(k), "centre_m": [round(v, 4) for v in g.get("centre", [])],
                         "footprint_m": [round(v, 4) for v in g.get("footprint", [])], "height_m": round(g.get("height", 0.0), 4),
                         "top_visible": g.get("tilt_deg") is not None, "detector_score": obs.scores.get(k)})
        self.facts = {"frame_id": obs.frame_id, "camera": obs.camera, "objects": objs, "cleared_view": obs.cleared_view,
                      "excluded_masks": len(obs.excluded)}
        self.hypotheses.setdefault("descriptions", {})
        for k, d in obs.descriptions.items():
            self.hypotheses["descriptions"][str(obs.object_ids.get(k, f"mark{k}"))] = {**d, "source": "gemma", "frame_id": obs.frame_id}

    def progress_text(self, limit: int = 900) -> str:
        """Where the task stands, for Gemma's next decision: done / failed / remaining steps with their results,
        the held object and the last goal check. Built from this ledger only."""
        lines = []
        for s in self.plan:
            if s.status in ("done", "failed", "replaced", "refused"):
                said = s.summary.split(" | verification: ")  # the measured evidence matters more than the worker's words
                text = f"verification: {said[1]}" if len(said) > 1 else said[0]
                lines.append(f"{s.index}. {s.capability} {_args_text(s.args)} -> {s.outcome or s.status}: {text[:160]}")
            elif s.status == "pending":
                lines.append(f"{s.index}. {s.capability} {_args_text(s.args)} -> not started")
        held = self.robot.get("held_object")
        lines.append(f"Robot: {'holding #' + str(held) if held is not None else 'gripper empty'}.")
        if self.goal_check:
            lines.append("Goal check: " + "; ".join(
                f"{c['condition'].get('pred')}({', '.join(map(str, c['condition'].get('args', [])))})="
                f"{ {True: 'met', False: 'NOT met', None: 'unknown'}[c['satisfied']] }" for c in self.goal_check))
        text = "Progress so far:\n" + "\n".join(lines)
        return text if len(text) <= limit else text[:limit - 3] + "..."

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, run_dir) -> None:
        p = Path(run_dir) / "task_state.json"
        p.write_text(json.dumps(self.to_dict(), indent=1, default=str))


def _args_text(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, dict):
            v = v.get("text")
        if v not in (None, "", []):
            parts.append(f"{k}={v}")
    return "(" + ", ".join(parts) + ")" if parts else ""
