"""Supervisor: the central coordinator (LangGraph). For every instruction it opens a fresh task memory
(agent/task_state.py), fixes the ORIGINAL goal (the user's words + Gemma's measurable success criteria; a plan is only
a strategy and never changes them), and runs one subtask at a time:

    Gemma chooses the worker and the operation -> delegate -> the worker executes with its trusted tools
    -> WorkerResult -> verification of the subtask's own effect -> task-memory update -> evaluation of the original goal
    -> verified: COMPLETED   |   otherwise Gemma reads the task memory and decides
       (continue / replan: another worker, changed information, another approach / ask_user / finish / stop)

    INTERPRETING -> DISPATCHING -> DELEGATING -> REVIEWING -> DECIDING -> DISPATCHING ... -> VERIFYING_GOAL
    terminal: COMPLETED | FAILED | CANCELLED | UNSUPPORTED

Deterministic, never overridden by Gemma: argument / worker interfaces, authorization, the unsafe and cancelled stops,
measured verification, the decision budget and the no-unchanged-retry guard. Tool success (the operation ran),
subtask success (its effect measured) and task success (the original goal verified) are kept apart: only the last
one completes the task. Workers never start other workers; every motion goes through the collision-checked planners
+ RosExecutor + the simulator's safety layer.
"""

from __future__ import annotations

import copy
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent import authority as authority_mod
from agent import capabilities as caps
from agent.controls import ExecutionControls
from config.agent import Budgets
from agent.eventlog import EventLog
from agent.goals import GROUNDABLE, derive_conditions
from agent.planning import _ref_texts
from agent.vlm_roles import VLMFailure
from agent.workers import WorkerRegistry
from agent.task_state import Step, TaskState

STATES = ["INTERPRETING", "DISPATCHING", "DELEGATING", "REVIEWING", "DECIDING", "VERIFYING_GOAL"]
TERMINAL = ["COMPLETED", "FAILED", "CANCELLED", "UNSUPPORTED"]
MAX_STEPS_EXECUTED = 14  # hard bound on worker delegations per instruction
STOP_STATUS = {"unsupported": "UNSUPPORTED", "cancelled": "CANCELLED", "ambiguous": "CANCELLED"}  # else FAILED


class GoalState(TypedDict, total=False):
    instruction: str
    next: str
    timeline: list


class GoalSupervisor:
    def __init__(self, tools, log: EventLog, run_dir: Path, responder, budgets: Budgets | None = None,
                 controls: ExecutionControls | None = None) -> None:
        self.tools, self.log, self.run_dir = tools, log, Path(run_dir)
        self.budgets = budgets or Budgets()
        self.controls = controls or ExecutionControls()  # disabled capabilities, replanning limit (agent/controls.py)
        self.specs = caps.available(self.controls.disabled)  # capability specs as this supervisor offers them
        self.responder = responder
        self.ts: TaskState | None = None
        self.authority = authority_mod.least_privilege("not classified yet")
        self.ctx: caps.Context | None = None
        self.executed = 0
        self.current: Step | None = None
        self.last_result = None
        self.event, self.event_outcome = "", None  # the latest event that needs Gemma's decision
        self.event_step: Step | None = None  # ... and the step that caused it (None: the final goal check)
        self.reg = WorkerRegistry(log, self._executors(), {**self.specs, **caps.INTERNAL})
        g = StateGraph(GoalState)
        for s in STATES:
            g.add_node(s, self._wrap(s, getattr(self, "_" + s.lower())))
        for s in TERMINAL:
            g.add_node(s, self._wrap(s, lambda st: {}))
            g.add_edge(s, END)
        g.set_entry_point("INTERPRETING")
        for s in STATES:
            g.add_conditional_edges(s, lambda st: st["next"], {n: n for n in STATES + TERMINAL})
        self.conn = sqlite3.connect(str(self.run_dir / "checkpoints.sqlite"), check_same_thread=False)
        self.graph = g.compile(checkpointer=SqliteSaver(self.conn))

    # ----------------------------------------------------------------------------------- workers
    def _executors(self) -> dict:
        """The implemented operations (agent/capabilities EXECUTORS); a disabled one is refused as unsupported."""
        executors = dict(caps.EXECUTORS)
        for name in self.controls.disabled & set(executors):  # disabled: refused as unsupported, never run
            executors[name] = (executors[name][0], self._refuse_disabled(name))
        return executors

    def _refuse_disabled(self, name: str):
        def refuse(c, a, r):
            self.controls.notify("capability_refused", capability=name)
            return caps.StepResult("unsupported", f"{name} is disabled {self.controls.disabled_reason}")
        return refuse

    # ---------------------------------------------------------------------------------- plumbing
    def _wrap(self, name, fn):
        def node(state: GoalState) -> dict:
            self.log("state", state=name, level="goal")
            t = time.monotonic()
            out = fn(state)
            if self.ts is not None:
                self._record_robot()
                self.ts.workers = self.reg.snapshot()
                self.ts.save(self.run_dir)
            return {**out, "timeline": list(state.get("timeline", [])) + [{"state": name, "seconds": round(time.monotonic() - t, 2)}]}
        return node

    def _record_robot(self) -> None:
        """Snapshot of the robot state the session's Tools own (the owner stays Tools; this is the record)."""
        t = self.tools
        self.ts.robot = {"held_object": t.held_object, "last_referent": t.last_referent,
                         "motion_count": getattr(t, "motion_count", None),
                         "motion_mode": getattr(getattr(t, "controls", None), "motion", None)}

    def _log_plan(self, why: str) -> None:
        """The current plan, after the supervisor changed it (trace: plan_update)."""
        self.log("plan_update", reason=why, plan=[{"index": s.index, "capability": s.capability, "status": s.status,
                                                    "origin": s.origin} for s in self.ts.plan])

    def _delegate(self, capability: str, args: dict, objective: str, expected: list | None = None,
                  depends_on: list | None = None, step: Step | None = None):
        """One structured delegation: request (goal, sub-goal, context, constraints, expected result) -> registry
        (authorization first) -> worker -> structured result. The supervisor processes the result itself."""
        refs = _ref_texts(args)
        spec = {**self.specs, **caps.INTERNAL}.get(capability)
        held = self.tools.held_object
        context = {"held_object": held, "last_referent": self.tools.last_referent,
                   "bindings": dict(self.ts.bindings), "frame": self.ctx.obs.frame_id if self.ctx.obs is not None else None,
                   "done": [f"{s.index}. {s.capability}" for s in self.ts.plan if s.status == "done"]}
        if step is not None:  # where the step stands in the plan: executors use the later steps' declared classes
            later = []
            for s in self.ts.remaining(step):
                sp = self.specs.get(s.capability)
                later.append({"capability": s.capability, "safety": getattr(sp, "safety", None),
                              "effect": getattr(sp, "effect", None),
                              "authorized": sp is not None and self.authority.allows(sp.effect)})
            context.update(step_index=step.index, later_steps=later)
        constraints = {"allowed_effects": list(self.authority.grants), "effect": getattr(spec, "effect", None),
                       "safety": getattr(spec, "safety", None),
                       "motion": getattr(getattr(self.tools, "controls", None), "motion", None)}
        if args.get("relation") and args.get("anchor"):
            constraints["location"] = f"{args['relation']} {args['anchor']}"
        req = self.reg.request(self.ts.instruction[:40], capability, args, objective=objective, object_refs=refs,
                               state_refs=[self.ctx.obs.frame_id] if self.ctx.obs is not None else [],
                               depends_on=depends_on, goal=self.ts.instruction, context=context,
                               constraints=constraints, expected=expected or [])
        return req, self.reg.dispatch(req, self.ctx, self.authority)

    def _measured_subgoal(self, step: Step) -> list[dict]:
        """The conditions a completed step should have made true, when its operation declares a measurable effect
        (spec.verify_after). Answers and reports are model / text outputs: checked only in the final goal check."""
        spec = self.specs.get(step.capability)
        if spec is None or not spec.verify_after:
            return []
        return derive_conditions([(step.capability, step.args)])

    def _finish(self, status: str, outcome: str, message: str) -> dict:
        self.ts.status, self.ts.outcome, self.ts.message = status.lower(), outcome, message
        if status != "COMPLETED":
            for s in self.ts.plan:
                if s.status == "pending":
                    s.status = "skipped"
        return {"next": status}

    def _context(self) -> str:
        t = self.tools
        if t.held_object is not None:
            o = t.world.objects.get(t.held_object)
            d = (o.description or {}) if o is not None else {}
            return f"The robot is HOLDING object #{t.held_object} ({d.get('colour', '')} {d.get('name', '')})."
        if t.last_referent is not None:
            return f"The robot's gripper is empty. The last instruction handled object #{t.last_referent}."
        return ""

    def _add_steps(self, steps, origin: str, after: Step | None = None, success=None, reason: str = "",
                   seconds: float | None = None) -> list[Step]:
        """steps: [(operation, args, worker)] (Gemma's choice of worker) -> ledger Steps inserted after `after` (or
        appended). A step without a worker keeps "" (it is reported, never assigned by Python)."""
        nxt = max([s.index for s in self.ts.plan], default=0) + 1
        new = []
        for i, st in enumerate(steps):
            c, a, w = (tuple(st) + ("",))[:3]
            new.append(Step(nxt + i, c, copy.deepcopy(a), worker=caps.worker_name(w), origin=origin))
        for i, st in enumerate(new):  # sequential dependency: each step needs the previous one
            st.depends_on = [new[i - 1].index] if i else []
        if after is None:
            self.ts.plan.extend(new)
        else:
            rest = self.ts.remaining(after)
            keep = [s for s in self.ts.plan if s not in rest]
            self.ts.plan = keep + new + rest
        self.log("goal_plan", goal=self.ts.goal, origin=origin, reason=reason, success=success or [], seconds=seconds,
                 steps=[{"index": s.index, "capability": s.capability, "args": s.args, "worker": s.worker} for s in new])
        return new

    # ----------------------------------------------------------------------------- task memory
    def _signature(self) -> tuple:
        """What counts as new evidence since a failure: motion, a new camera frame, an answer from the user."""
        return (getattr(self.tools, "motion_count", 0), len(self.ts.frames),
                sum(1 for q in self.ts.questions if q.get("answer")))

    def _evaluate_goal(self, scope: str = "progress") -> tuple[list, list, list]:
        """The verification worker evaluates the ORIGINAL goal's criteria on measured state -> (rows, false, unknown);
        the memory's unmet list is updated. Gemma's answers count as 'model' (reported, never called measured)."""
        conds = list(self.ts.conditions)
        self._ground_criteria(conds)
        _, v = self._delegate("verify", {"conditions": conds, "scope": "goal"}, "verify the user's original goal",
                              expected=conds)
        rows = v.data.get("results", [])
        self.ts.goal_check = rows
        false = [r for r in rows if r["satisfied"] is False]
        unknown = [r for r in rows if r["satisfied"] is None and r["confidence"] != "model"]
        self.ts.unmet = [f"{_cond(r['condition'])}: {r['evidence']}" for r in false + unknown]
        self.log("goal_check", results=rows, scope=scope)
        return rows, false, unknown

    def _ground_criteria(self, conds: list) -> None:
        """The original criteria name objects in the user's words; a plan may act on measured ids (#3). An object
        reference no worker has bound yet is grounded once per observation by perception (Gemma's resolve over the
        measured objects) and kept as a binding of this task; an ambiguous or failed grounding leaves it unbound
        (the criterion stays unknown, never assumed)."""
        obs = self.ctx.obs
        if obs is None:
            return
        for c in conds:
            if c.get("pred") not in GROUNDABLE:  # located(x) means a subtask found x: grounding must not satisfy it
                continue
            for ref in c.get("args") or []:
                if not isinstance(ref, str) or ref.startswith("#") or ref in self.ts.bindings:
                    continue
                key = (ref, getattr(obs, "frame_id", None))
                if key in self.grounded:
                    continue
                self.grounded.add(key)
                try:
                    r = self.tools.resolve(obs, ref)
                except VLMFailure:
                    continue
                wid = obs.object_ids.get(r.id) if r.id is not None else None
                if wid is not None:
                    self.ts.bindings[ref] = wid
                    self.log("binding", text=ref, world_id=wid, reason=f"grounded for the goal check (frame {obs.frame_id})")

    def _goal_met(self, false, unknown) -> bool:
        return bool(self.ts.conditions) and not false and not unknown

    def _remember(self, step: Step, res, verification: str = "") -> None:
        """Task memory update after a worker result: the exchange (attempted / happened / evidence / uncertainty /
        failure / state change), and a failure record with the state signature (no unchanged retry)."""
        ev = verification or "; ".join(f"{e.get('kind')} {e.get('id', e.get('check', ''))}".strip()
                                       for e in (getattr(res, "evidence", None) or [])[:3])
        held_before = (self.ts.exchanges[-1]["held_after"] if self.ts.exchanges else self.ts.robot.get("held_object"))
        e = {"step": step.index, "worker": step.worker or "-", "operation": step.capability, "args": _args_line(step.args),
             "attempted": step.capability + (f" ({_args_line(step.args)})" if step.args else ""),
             "tool": getattr(res, "status", "not run") if res is not None else "not run",
             "outcome": step.outcome, "subtask": step.subtask or "not achieved", "happened": step.summary,
             "evidence": ev, "uncertainty": list(getattr(res, "uncertainty", []) or []),
             "failure": getattr(res, "failure_reason", "") if res is not None else step.summary,
             "held_before": held_before, "held_after": self.tools.held_object,
             "motion_count": getattr(self.tools, "motion_count", 0)}
        self.ts.exchanges.append(e)
        if step.subtask != "achieved" and step.outcome != "completed":
            sig = self._signature()
            self.ts.failures.append({"step": step.index, "worker": e["worker"], "operation": step.capability,
                                     "args": e["args"], "outcome": step.outcome, "evidence": step.summary[:200],
                                     "signature": list(sig)})
            self.ts.attempted.append({"capability": step.capability, "args": step.args, "outcome": step.outcome,
                                      "signature": list(sig)})
        self.log("task_memory", step=step.index, worker=e["worker"], operation=step.capability, tool=e["tool"],
                 subtask=e["subtask"], held=self.tools.held_object, failures=len(self.ts.failures))

    # ----------------------------------------------------------------------------------- states
    def _interpreting(self, state) -> dict:
        # authority first, from the request alone; no plan (Gemma's or decomposed) can change it afterwards
        self.authority = authority_mod.classify(state["instruction"], self.tools.roles)
        self.ts.authority = self.authority.to_dict()
        self.log("authority", request_kind=self.authority.kind, grants=list(self.authority.grants),
                 evidence=self.authority.evidence, source=self.authority.source, note=self.authority.note)
        workers = {s.worker for s in self.specs.values() if s.implemented}
        self.log("planning", request_kind=self.authority.kind, workers=len(workers),
                 operations=sum(1 for s in self.specs.values() if s.implemented), context=self._context())
        t_plan = time.monotonic()
        try:
            plan = self.tools.roles.plan_goal(state["instruction"], caps.planner_menu(self.specs), context=self._context())
        except VLMFailure as exc:
            return self._finish("FAILED", "failed", f"I could not turn the instruction into steps ({exc}). "
                                + self._capabilities_line())
        self.ts.goal = plan.goal
        if not plan.steps:
            return self._finish("UNSUPPORTED", "unsupported", f"I cannot do that: {plan.reason or 'no steps planned'}. "
                                + self._capabilities_line())
        # the ORIGINAL goal: Gemma's success criteria (+ the answer, for a question), fixed for the whole task
        criteria = [{**c.model_dump(exclude_none=True), "source": "gemma"} for c in plan.success]
        if self.authority.kind == "question":  # the goal of a question is its answer (a report alone is not one)
            criteria = [{"pred": "answered", "args": [state["instruction"]], "source": "request"}] + [
                c for c in criteria if c["pred"] != "answered"]
        self.ts.conditions = criteria
        self.ts.original_goal = {"instruction": state["instruction"], "statement": plan.goal, "criteria": list(criteria),
                                 "source": "gemma" + (" + request" if self.authority.kind == "question" else "")}
        self.ts.unmet = [_cond(c) for c in criteria]
        self._add_steps([(st.capability, st.args, st.worker) for st in plan.steps], "planner", success=criteria,
                        reason=plan.reason, seconds=round(time.monotonic() - t_plan, 1))
        self.log("goal_conditions", goal=plan.goal, conditions=self.ts.conditions, origin="original goal")
        return {"next": "DISPATCHING"}

    def _capabilities_line(self) -> str:
        return "Operations implemented now: " + ", ".join(n for n, sp in self.specs.items() if sp.implemented) + "."

    def _dispatching(self, state) -> dict:
        step = self.ts.current()
        if step is None:
            return {"next": "VERIFYING_GOAL"}
        if self.executed >= MAX_STEPS_EXECUTED:
            return self._finish("FAILED", "failed", f"stopped after {MAX_STEPS_EXECUTED} subtasks without reaching the goal")
        spec = self.specs.get(step.capability)
        why = None
        if spec is None or not self.reg.worker_of(step.capability):  # Gemma named an operation nobody implements
            why = (f"{step.capability!r} is not implemented: no worker can run it, so nothing was executed "
                   f"({self._capabilities_line()})")
        elif step.worker != spec.worker:  # Gemma's worker choice does not implement the operation (interface check)
            why = (f"the {step.worker or '(unnamed)'} worker does not implement {step.capability!r}, so nothing was "
                   f"executed")
        if why:
            step.status, step.outcome, step.summary = "failed", "unsupported", why
            self.log("decision", index=step.index, capability=step.capability, worker=step.worker or None, reason=why)
            self.last_result = None
            return {"next": "REVIEWING"}
        if spec.needs_view and not self.tools.is_fresh(self.ctx.obs):  # precondition: a measured view (obstacles)
            self.log("step", event="auto_observe", index=step.index, reason="no fresh observation")
            _, r = self._delegate("observe", {"clear_view": False}, f"fresh observation for step {step.index} ({step.capability})")
            if r.outcome != "completed":
                step.status, step.outcome, step.summary = "failed", r.outcome, r.summary
                self.last_result = r
                return {"next": "REVIEWING"}
        self.current = step
        self.log("decision", index=step.index, capability=step.capability, worker=step.worker,
                 reason=f"Gemma's plan ({step.origin}): the {step.worker} worker for {step.capability}")
        return {"next": "DELEGATING"}

    def _delegating(self, state) -> dict:
        step = self.current
        step.status = "running"
        refs = _ref_texts(step.args)
        spec = self.specs.get(step.capability)
        self.log("step", event="start", index=step.index, capability=step.capability, args=step.args,
                 safety=getattr(spec, "safety", "read_only"), worker=self.reg.worker_of(step.capability),
                 task_id=f"step{step.index}")
        t0 = time.monotonic()
        req, res = self._delegate(step.capability, step.args, f"{step.capability} {', '.join(refs)}".strip(),
                                  expected=[c for c in derive_conditions([(step.capability, step.args)])],
                                  depends_on=[f"step{d}" for d in step.depends_on], step=step)
        step.task_id, step.worker = req.task_id, req.worker
        if res.uncertainty:
            self.ts.uncertainties.append({"task_id": res.task_id, "capability": step.capability, "items": res.uncertainty})
        step.status = "done" if res.status == "completed" else "failed"
        step.outcome, step.summary, step.seconds = res.outcome, res.summary, round(time.monotonic() - t0, 1)
        step.result = {k: v for k, v in asdict(res).items() if k != "data"}
        self.executed += 1
        self.ts.bindings.update({k: v for k, v in (res.data.get("bindings") or {}).items() if v is not None})
        self.ts.actions.append({"index": step.index, "capability": step.capability, "worker": res.worker,
                                "status": res.status, "outcome": res.outcome, "seconds": step.seconds})
        self.log("step", event="end", index=step.index, capability=step.capability, outcome=res.outcome,
                 status=res.status, worker=res.worker, summary=res.summary[:300], seconds=step.seconds)
        self.last_result = res
        return {"next": "REVIEWING"}

    def _review(self, step: Step, decision: str, why: str) -> None:
        """The supervisor's explicit processing of one worker result (trace: review)."""
        res = self.last_result
        self.log("review", index=step.index, capability=step.capability, outcome=step.outcome,
                 needs=getattr(res, "needs", ""), uncertainty=getattr(res, "uncertainty", []), decision=decision, reason=why)

    def _reviewing(self, state) -> dict:
        """delegate -> execute -> WorkerResult -> HERE: verify the subtask's own effect, apply the deterministic stops
        (unsafe, cancelled), update the task memory, evaluate the ORIGINAL goal; then Gemma decides (DECIDING), or the
        task is verified (VERIFYING_GOAL). Tool success, subtask success and task success are kept apart."""
        step = next(s for s in reversed(self.ts.plan) if s.status in ("done", "failed"))
        res = self.last_result
        data = res.data if res is not None else {}
        verification = ""
        if step.outcome == "completed":
            conds = self._measured_subgoal(step)
            if conds:
                _, v = self._delegate("verify", {"conditions": conds, "scope": "subgoal"},
                                      f"verify step {step.index} ({step.capability})", expected=conds)
                self.ts.subgoal_checks.append({"step": step.index, "task_id": v.task_id, "outcome": v.outcome,
                                               "summary": v.summary, "results": v.data.get("results", [])})
                verification = v.summary
                if v.outcome == "failed":  # the worker reported success, the measured state says otherwise
                    step.status, step.outcome, step.subtask = "failed", "failed", "not achieved"
                    step.summary = f"{step.summary} | verification: the sub-goal is {v.summary}"
                elif v.outcome == "completed":
                    step.subtask = "achieved"
                else:
                    step.subtask = "unknown"
                    self.ts.uncertainties.append({"task_id": v.task_id, "capability": "verify", "items": v.uncertainty})
            else:
                step.subtask = "done (no measurable effect)" if data.get("certain") is not False else "uncertain"
        else:
            step.subtask = "not achieved"
        if step.outcome == "unauthorized":  # authorization (deterministic): refused before execution
            step.status = "refused"
            self.ts.refused.append({"capability": step.capability, "args": step.args, "origin": step.origin})
            self._log_plan("refused step")
        self._remember(step, res, verification)
        if step.outcome == "unsafe":  # safety (deterministic): no new motion is planned
            self._review(step, "stop (unsafe)", step.summary)
            return self._finish("FAILED", "unsafe", f"Step {step.index} ({step.capability}) unsafe: {step.summary}"
                                + self._done_so_far() + " I did not report success.")
        if step.outcome == "cancelled":  # the user decided
            self._review(step, "stop (cancelled by the user)", step.summary)
            return self._finish("CANCELLED", "cancelled", step.summary + self._done_so_far())
        if step.outcome == "completed" and data.get("steps"):  # a worker (decompose: Gemma) returned steps to insert
            steps = [tuple(x) for x in data.get("steps", [])]
            allowed = [x for x in steps if self.specs.get(x[0]) is None or self.authority.allows(self.specs[x[0]].effect)]
            refused = [x for x in steps if x not in allowed]
            if refused:  # a decomposition cannot add effects the user did not ask for
                self.log("authorization", origin="decomposition", allowed=False, refused=[x[0] for x in refused],
                         reason=f"the request ({self.authority.kind}) does not authorize changing objects")
                self.ts.refused += [{"capability": x[0], "args": x[1], "origin": "decomposition"} for x in refused]
            self._add_steps(allowed, "decomposition", after=step)
            extra = [{**c, "source": "decomposition"} for c in data.get("conditions", [])] if not refused else []
            if extra:  # added to the original goal (logged), never replacing it
                self.ts.conditions = self.ts.conditions + extra
                self.log("goal_conditions", goal=self.ts.goal, conditions=self.ts.conditions,
                         origin="original goal + decomposition criteria")
            self._log_plan("decomposition")
        rows, false, unknown = self._evaluate_goal()
        if self._goal_met(false, unknown):
            self._review(step, "original goal verified", f"tool {getattr(res, 'status', '-')}, subtask {step.subtask}")
            return {"next": "VERIFYING_GOAL"}
        unsettled = list(getattr(res, "uncertainty", []) or [])
        note = (f" | your last decision was continue, so the steps you gave with it ({', '.join(self.ignored_steps)}) "
                "were not used; use replan to change the steps") if self.ignored_steps else ""
        self.ignored_steps = []
        self.event = (f"step {step.index} ({step.worker or '?'} worker: {step.capability} {_args_line(step.args)}): "
                      f"tool {getattr(res, 'status', 'not run')}, subtask {step.subtask}: {step.summary[:300]}"
                      + (f" | uncertain: {'; '.join(unsettled)}" if unsettled else "") + note)
        self.event_outcome, self.event_step = step.outcome, step
        self._review(step, "ask Gemma", f"tool {getattr(res, 'status', 'not run')}, subtask {step.subtask}, "
                                        f"{len(self.ts.unmet)} original criteria not yet shown")
        return {"next": "DECIDING"}

    def _failed_operations(self) -> dict[str, str]:
        """Operations that failed in this task (task memory), with how often and the last reason, for the menu."""
        out: dict[str, list] = {}
        for f in self.ts.failures:
            out.setdefault(f["operation"], []).append(f.get("evidence", ""))
        return {op: f"{len(ev)} time(s), last: {ev[-1][:120]}" for op, ev in out.items()}

    def _deciding(self, state) -> dict:
        """Gemma reads the task memory and decides the next subtask. Changes of course (replan / ask / finish / stop)
        are bounded by ExecutionControls.max_replans; "continue" is bounded by the delegation limit. A re-plan whose
        first subtask already failed with no new evidence since is sent back to Gemma."""
        event, outcome = self.event, self.event_outcome
        pending = [s for s in self.ts.plan if s.status == "pending"]
        self.log("consult", reason=event, replans=self.ts.replans, budget=self.controls.max_replans)
        try:
            d = self.tools.roles.decide(self.ts.instruction, self.ts.memory_text(), event,
                                        caps.planner_menu(self.specs, self._failed_operations()),
                                        has_pending=bool(pending), context=self._context())
        except VLMFailure as exc:
            self.ts.consults.append({"reason": event, "action": None, "accepted": False, "why": str(exc)})
            return self._stop(outcome, f"{event}. (Gemma could not decide: {exc}.)")
        self.log("gemma_decision", action=d.action, reason=d.reason, question=d.question,
                 steps=[{"worker": st.worker, "capability": st.capability, "args": st.args} for st in d.steps])
        record = {"reason": event, "action": d.action, "accepted": True, "why": d.reason,
                  "steps": [st.capability for st in d.steps]}
        self.ts.consults.append(record)
        if d.action == "continue":  # keeps the planned steps; steps sent with it are not used (Gemma is told)
            if d.steps:
                self.ignored_steps = [st.capability for st in d.steps]
                self.log("replan", accepted=False, reason=f"continue keeps the plan: the steps given with it "
                                                         f"({', '.join(self.ignored_steps)}) were not used")
            return {"next": "DISPATCHING"}
        if self.ts.replans >= self.controls.max_replans:  # a change of course beyond the budget
            record["accepted"] = False
            return self._stop(outcome, f"{event}. Gemma chose {d.action}, but no further changes are allowed "
                                       f"(budget {self.controls.max_replans} used).")
        self.ts.replans += 1
        if d.action == "finish":
            for s in pending:
                s.status = "skipped"
            self._log_plan("Gemma: finish")
            return {"next": "VERIFYING_GOAL"}
        if d.action == "stop":
            return self._stop(outcome, f"{event}. Gemma stopped: {d.reason}")
        if d.action == "ask_user":
            ask = getattr(self.responder, "ask", None)
            answer = ask(d.question) if ask is not None else None
            self.ts.questions.append({"question": d.question, "answer": answer, "asked_by": "gemma"})
            self.log("ask_user", answer="text" if answer else "none", raw=answer or "", question=d.question,
                     source="terminal" if type(self.responder).__name__ == "TerminalResponder" else "scripted")
            if answer is None:
                return self._finish("CANCELLED", "cancelled", f"I asked: {d.question} No answer, so I stopped."
                                    + self._done_so_far())
            self.event, self.event_outcome = f"the user answered {d.question!r}: {answer!r}", outcome
            return {"next": "DECIDING"}
        # replan: the first new subtask must not repeat a failed one with no new evidence since
        first = (d.steps[0].capability, d.steps[0].args)
        sig = list(self._signature())
        repeats = any(first == (a["capability"], a["args"]) and a.get("signature") == sig for a in self.ts.attempted)
        if repeats:
            record["accepted"] = False
            self.log("replan", accepted=False, reason="repeats a failed subtask with no new evidence")
            self.event = f"{event} | your last plan repeated a failed subtask although nothing changed; choose differently"
            return {"next": "DECIDING"}
        self.log("replan", accepted=True)
        if self.event_step is not None and (self.event_step.status == "failed" or (
                self.event_step.status == "done" and self.event_step.subtask in ("uncertain", "unknown"))):
            self.event_step.status = "replaced"  # the failed / doubtful subtask Gemma re-planned around
        for s in pending:
            s.status = "skipped"
        self._add_steps([(st.capability, st.args, st.worker) for st in d.steps], "replan", reason=d.reason)
        self._log_plan("re-planned by Gemma (the original goal is unchanged)")
        return {"next": "DISPATCHING"}

    def _stop(self, outcome: str | None, message: str) -> dict:
        status = STOP_STATUS.get(outcome or "", "FAILED")
        tail = "" if status == "UNSUPPORTED" else " I did not report success."
        return self._finish(status, outcome or "failed", message + self._done_so_far() + (tail if status == "FAILED" else ""))

    def _verifying_goal(self, state) -> dict:
        """Task success: the ORIGINAL goal's criteria on measured state (Gemma's answers count only as 'model').
        Reached when the goal was shown after a subtask, the plan ran out, or Gemma said finish. Not shown: Gemma
        decides again (within the budget), otherwise the task fails with the unmet criteria."""
        steps = [s for s in self.ts.plan if s.status not in ("skipped", "replaced", "refused")]
        rows, false, unknown = self._evaluate_goal("goal")
        summary = self._result_message(steps)
        if not self._goal_met(false, unknown):
            gap = "; ".join(self.ts.unmet) or "the goal has no measurable success criteria"
            if self.ts.replans < self.controls.max_replans and self.executed < MAX_STEPS_EXECUTED:
                self.event, self.event_outcome, self.event_step = f"the original goal is not shown yet: {gap}", "failed", None
                self.log("review", index=None, capability="verify", outcome="failed", needs="",
                         uncertainty=[], decision="ask Gemma", reason=f"original goal not shown: {gap}")
                return {"next": "DECIDING"}
            kind = "failed" if false else "insufficient_info"
            return self._finish("FAILED", kind, f"The original goal is not shown: {gap}.\n{summary}\nI did not report success.")
        model = [r for r in rows if r["confidence"] == "model"]
        refused = (f" I did not {', '.join(sorted({r['capability'] for r in self.ts.refused}))}: your request did not ask me to "
                   "change any objects." if self.ts.refused else "")
        shown = "; ".join(f"{_cond(r['condition'])} [{r['confidence']}]" for r in rows)
        self.log("review", index=None, capability="verify", outcome="completed", needs="", uncertainty=[],
                 decision="task verified", reason=shown)
        return self._finish("COMPLETED", "completed", summary + (" (answer from Gemma, not independently verified)" if model else "")
                            + refused + f"\n(goal verified: {shown})")

    @staticmethod
    def _result_message(steps) -> str:
        """The last completed step's result first, then (in brackets) up to two earlier ones."""
        done = [s for s in steps if s.summary and s.status == "done"] or [s for s in steps if s.summary]
        if not done:
            return ""
        msg = done[-1].summary
        if len(done) > 1:
            msg += "\n(" + "; ".join(s.summary for s in done[-3:-1]) + ")"
        return msg

    def _done_so_far(self) -> str:
        done = [f"{s.index}. {s.capability}" for s in self.ts.plan if s.status == "done"]
        return f" Completed before that: {', '.join(done)}." if done else ""

    # --------------------------------------------------------------------------------------- run
    def run(self, instruction: str, thread_id: str = "run") -> dict:
        t0 = time.monotonic()
        self.ts = TaskState(instruction)
        self.ts.sensors = list(getattr(self.tools, "SENSORS", ()))
        self.ts.capabilities = [n for n, s in self.specs.items() if s.implemented]
        self.event, self.event_outcome, self.event_step = "", None, None  # fresh task memory for this instruction
        self.grounded = set()  # (reference, frame) grounded for the goal check in this task
        self.ignored_steps = []  # steps Gemma gave with a "continue" (not used; it is told in the next event)
        self.ctx = caps.Context(self.tools, self.log, self.run_dir, self.responder, self.budgets, self.ts)
        self.ctx.vlm_calls = lambda: getattr(self.tools.roles, "calls", 0)
        self.log("instruction", text=instruction, held=self.tools.held_object, last_referent=self.tools.last_referent)
        final = self.graph.invoke({"instruction": instruction, "timeline": []},
                                  {"configurable": {"thread_id": thread_id}, "recursion_limit": 200})
        status = final["timeline"][-1]["state"]
        self.ts.workers = self.reg.snapshot()
        self.ts.finalized = True  # this task memory is closed; the next instruction starts a new one
        self.ts.save(self.run_dir)
        report = {"instruction": instruction, "status": status, "outcome": self.ts.outcome, "message": self.ts.message,
                  "seconds": round(time.monotonic() - t0, 1), "goal": self.ts.goal, "conditions": self.ts.conditions,
                  "goal_check": self.ts.goal_check, "bindings": self.ts.bindings,
                  "plan": [{"index": s.index, "capability": s.capability, "args": s.args, "worker": s.worker, "status": s.status,
                            "outcome": s.outcome, "seconds": s.seconds, "origin": s.origin} for s in self.ts.plan],
                  "verifications": self.ts.verified, "subgoal_checks": self.ts.subgoal_checks, "consults": self.ts.consults,
                  "uncertainties": self.ts.uncertainties, "assumptions": self.ts.assumptions,
                  "questions": self.ts.questions, "replans": self.ts.replans,
                  "holding": self.tools.held_object, "timeline": final["timeline"], "authority": self.ts.authority,
                  "refused": self.ts.refused, "original_goal": self.ts.original_goal, "unmet": self.ts.unmet,
                  "exchanges": self.ts.exchanges, "failures": self.ts.failures,
                  "failure": None if status == "COMPLETED" else {"cls": self.ts.outcome, "reason": self.ts.message}}
        self.log("report", status=status, message=self.ts.message)
        return report


def _args_line(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        v = v.get("text") if isinstance(v, dict) else v
        if v not in (None, "", []):
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _cond(c: dict) -> str:
    tail = f" along {c['direction']} ({c.get('frame', 'robot')})" if c.get("direction") is not None else ""
    return f"{c.get('pred')}({', '.join(map(str, c.get('args', [])))}){tail}"
