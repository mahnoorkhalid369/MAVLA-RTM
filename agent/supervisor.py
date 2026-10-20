"""Supervisor: the central coordinator (LangGraph). It owns the user's goal and the task ledger. Gemma is the
decision-maker: it classifies the request, writes the first plan and, whenever a result needs a decision, chooses
the next step (continue / replan / ask_user / finish / stop). The supervisor delegates one subtask at a time to a
worker as a typed request, reviews each typed result, applies the deterministic checks (schema validation,
authorization, measured verification, safety stops, budgets) and completes only when the goal is verified.

    INTERPRETING -> DISPATCHING -> DELEGATING -> REVIEWING -> DISPATCHING ... -> VERIFYING_GOAL
                                                     |-> DECIDING (Gemma; bounded) -> DISPATCHING | VERIFYING_GOAL
    VERIFYING_GOAL --goal not shown, budget left--> DECIDING
    terminal: COMPLETED | FAILED | CANCELLED | UNSUPPORTED

Workers (agent/workers.py; logical roles in this process, one shared Gemma / YOLO / ROS bridge):
  perception    observe, describe_scene, answer, locate
  inspection    inspect, explore, look (wrist-camera viewpoints, perception/viewpoints.py)
  manipulation  pick, place, pick_place, return_home (the validated PickPlaceWorkflow in three modes)
  planning      decompose (Gemma over the measured objects)
  verification  goal / sub-goal conditions over measured state (agent/goals.py) + the skills' own physical checks
Safety: Gemma names only registry capabilities; every motion goes through SkillPlanner + CollisionChecker
+ RosExecutor + the simulator's safety layer; workers never start other workers.
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
from agent.goals import derive_conditions
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
        self.failed_motion = 0
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

    def _add_steps(self, steps, origin: str, after: Step | None = None) -> list[Step]:
        """steps: [(capability, args)] -> ledger Steps inserted after `after` (or appended)."""
        nxt = max([s.index for s in self.ts.plan], default=0) + 1
        new = [Step(nxt + i, c, copy.deepcopy(a), worker=self.reg.worker_of(c) or "", origin=origin) for i, (c, a) in enumerate(steps)]
        for i, st in enumerate(new):  # sequential dependency: each step needs the previous one
            st.depends_on = [new[i - 1].index] if i else []
        if after is None:
            self.ts.plan.extend(new)
        else:
            rest = self.ts.remaining(after)
            keep = [s for s in self.ts.plan if s not in rest]
            self.ts.plan = keep + new + rest
        self.log("goal_plan", goal=self.ts.goal, origin=origin, reason="",
                 steps=[{"index": s.index, "capability": s.capability, "args": s.args, "worker": s.worker} for s in new])
        return new

    def _refresh_conditions(self) -> None:
        request = [c for c in self.ts.conditions if c.get("source") == "request"]  # from the user's request: always kept
        explicit = [c for c in self.ts.conditions if c.get("source") == "explicit"]
        # the goal = the conditions of the steps in the current plan (Gemma's latest plan); refused steps were never
        # authorized; replaced / skipped steps are no longer part of it
        live = [(s.capability, s.args) for s in sorted(self.ts.plan, key=lambda x: x.index)
                if s.status not in ("skipped", "replaced", "refused")]
        derived = [{**c, "source": "derived"} for c in derive_conditions(live)]
        if any(c["pred"] == "answered" for c in request):  # the request's own question already stands for the answer
            derived = [c for c in derived if c["pred"] != "answered"]
        self.ts.conditions = request + (explicit or derived)

    # ----------------------------------------------------------------------------------- states
    def _interpreting(self, state) -> dict:
        # authority first, from the request alone; no plan (Gemma's or decomposed) can change it afterwards
        self.authority = authority_mod.classify(state["instruction"], self.tools.roles)
        self.ts.authority = self.authority.to_dict()
        self.log("authority", request_kind=self.authority.kind, grants=list(self.authority.grants),
                 evidence=self.authority.evidence, source=self.authority.source, note=self.authority.note)
        try:
            plan = self.tools.roles.plan_goal(state["instruction"], caps.planner_menu(self.specs), context=self._context())
        except VLMFailure as exc:
            return self._finish("FAILED", "failed", f"I could not turn the instruction into steps ({exc}). "
                                + self._capabilities_line())
        self.ts.goal, self.ts.success_condition = plan.goal, plan.success
        if not plan.steps:
            return self._finish("UNSUPPORTED", "unsupported", f"I cannot do that: {plan.reason or 'no steps planned'}. "
                                + self._capabilities_line())
        self._add_steps([(st.capability, st.args) for st in plan.steps], "planner")
        if self.authority.kind == "question":  # the goal of a question is its answer (a report alone is not one)
            self.ts.conditions = [{"pred": "answered", "args": [state["instruction"]], "source": "request"}]
        self._refresh_conditions()
        self.log("goal_conditions", goal=plan.goal, conditions=self.ts.conditions)
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
        if spec is None or not self.reg.worker_of(step.capability):  # Gemma named an operation nobody implements
            step.status, step.outcome = "failed", "unsupported"
            step.summary = (f"{step.capability!r} is not implemented: no worker can run it, so nothing was executed "
                            f"({self._capabilities_line()})")
            self.log("decision", index=step.index, capability=step.capability, worker=None, reason="not implemented")
            self.last_result = None
            return {"next": "REVIEWING"}
        if spec.needs_view and not self.tools.is_fresh(self.ctx.obs):
            self.log("step", event="auto_observe", index=step.index, reason="no fresh observation")
            _, r = self._delegate("observe", {"clear_view": False}, f"fresh observation for step {step.index} ({step.capability})")
            if r.outcome != "completed":
                step.status, step.outcome, step.summary = "failed", r.outcome, r.summary
                self.last_result = None
                return {"next": "REVIEWING"}
        self.current = step
        self.log("decision", index=step.index, capability=step.capability, worker=self.reg.worker_of(step.capability),
                 reason=f"next pending step of the plan (from the {step.origin}); {step.capability} is the "
                        f"{self.reg.worker_of(step.capability)} worker's operation")
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
        """Deterministic part of the review: verify the sub-goal, apply authorization and safety stops, insert a
        decomposition. A routine success continues without Gemma; anything else goes to Gemma (DECIDING)."""
        step = next(s for s in reversed(self.ts.plan) if s.status in ("done", "failed"))
        res = self.last_result
        data = res.data if res is not None else {}
        unsettled = list(getattr(res, "uncertainty", []) or [])
        doubtful = (bool(getattr(res, "needs", "")) or data.get("certain") is False  # the worker itself is not sure
                    or bool(data.get("unknown_conditions")))
        if step.outcome == "completed" and (conds := self._measured_subgoal(step)):
            _, v = self._delegate("verify", {"conditions": conds, "scope": "subgoal"},
                                  f"verify step {step.index} ({step.capability})", expected=conds)
            self.ts.subgoal_checks.append({"step": step.index, "task_id": v.task_id, "outcome": v.outcome,
                                           "summary": v.summary, "results": v.data.get("results", [])})
            if v.outcome == "failed":  # the worker reported success, the measured state says otherwise
                step.status, step.outcome = "failed", "failed"
                step.summary = f"{step.summary} | verification: the sub-goal is {v.summary}"
            elif v.outcome != "completed":
                doubtful = True
                unsettled += v.uncertainty or [v.summary]
                self.ts.uncertainties.append({"task_id": v.task_id, "capability": "verify", "items": v.uncertainty})
        if step.outcome == "unauthorized":  # authorization (deterministic): refused before execution; go on
            step.status = "refused"
            self.ts.refused.append({"capability": step.capability, "args": step.args, "origin": step.origin})
            for s in self.ts.remaining(step):
                if not self.authority.allows(self.specs[s.capability].effect):
                    s.status = "refused"
                    self.ts.refused.append({"capability": s.capability, "args": s.args, "origin": s.origin})
            self._refresh_conditions()
            self._review(step, "refuse and continue", "the request does not authorize this effect")
            self._log_plan("refused steps")
            return {"next": "DISPATCHING"}
        if step.outcome == "unsafe":  # safety (deterministic): no new motion is planned
            self._review(step, "stop (unsafe)", step.summary)
            return self._finish("FAILED", "unsafe", f"Step {step.index} ({step.capability}) unsafe: {step.summary}"
                                + self._done_so_far() + " I did not report success.")
        if step.outcome == "cancelled":  # the user decided
            self._review(step, "stop (cancelled by the user)", step.summary)
            return self._finish("CANCELLED", "cancelled", step.summary + self._done_so_far())
        if step.outcome == "completed" and data.get("steps"):  # a worker (decompose: Gemma) returned steps to insert
            steps = data.get("steps", [])
            allowed = [(c, a) for c, a in steps if self.authority.allows(self.specs[c].effect)]
            refused = [(c, a) for c, a in steps if (c, a) not in allowed]
            if refused:  # a decomposition cannot add effects the user did not ask for
                self.log("authorization", origin="decomposition", allowed=False, refused=[c for c, _ in refused],
                         reason=f"the request ({self.authority.kind}) does not authorize changing objects")
                self.ts.refused += [{"capability": c, "args": a, "origin": "decomposition"} for c, a in refused]
            self._add_steps(allowed, "decomposition", after=step)
            explicit = [{**c, "source": "explicit"} for c in data.get("conditions", [])] if not refused else []
            if explicit:
                self.ts.conditions = explicit
            else:
                self._refresh_conditions()
            self.log("goal_conditions", goal=self.ts.goal, conditions=self.ts.conditions, origin="decomposition")
            self._review(step, "insert the decomposed steps", f"{len(allowed)} step(s) added, {len(refused)} refused")
            self._log_plan("decomposition")
            return {"next": "DISPATCHING"}
        if step.outcome == "completed" and not doubtful:
            self._review(step, "continue", "sub-goal achieved" + (" (verified)" if self._measured_subgoal(step) else ""))
            return {"next": "DISPATCHING"}
        if step.outcome != "completed":
            self.ts.attempted.append({"capability": step.capability, "args": step.args, "outcome": step.outcome})
            self.failed_motion = getattr(self.tools, "motion_count", 0)
        needs = getattr(res, "needs", "")
        self.event = (f"step {step.index} ({step.capability} {_args_line(step.args)}) by the {step.worker} worker: "
                      f"{step.outcome}: {step.summary[:300]}"
                      + (f" | uncertain: {'; '.join(unsettled)}" if unsettled else "")
                      + (f" | the worker suggests: {needs}" if needs else ""))
        self.event_outcome, self.event_step = step.outcome, step
        self._review(step, "ask Gemma", "the result needs a decision: " + (step.outcome if step.outcome != "completed"
                                                                         else "uncertain result"))
        return {"next": "DECIDING"}

    def _deciding(self, state) -> dict:
        """Gemma decides the next step from the goal, the progress (task ledger) and the latest event; the supervisor
        validates and executes the decision. Bounded by ExecutionControls.max_replans (Gemma decisions after the first
        plan); a re-plan that repeats a failed step in an unchanged situation is sent back to Gemma."""
        event, outcome = self.event, self.event_outcome
        if self.ts.replans >= self.controls.max_replans:
            return self._stop(outcome, f"{event}. No further decisions allowed (budget {self.controls.max_replans} used).")
        self.ts.replans += 1
        pending = [s for s in self.ts.plan if s.status == "pending"]
        self.log("consult", reason=event, replans=self.ts.replans, budget=self.controls.max_replans)
        try:
            d = self.tools.roles.decide(self.ts.instruction, self.ts.progress_text(), event, caps.planner_menu(self.specs),
                                        has_pending=bool(pending), context=self._context())
        except VLMFailure as exc:
            self.ts.consults.append({"reason": event, "action": None, "accepted": False, "why": str(exc)})
            return self._stop(outcome, f"{event}. (Gemma could not decide: {exc}.)")
        self.log("gemma_decision", action=d.action, reason=d.reason, question=d.question,
                 steps=[{"capability": st.capability, "args": st.args} for st in d.steps])
        record = {"reason": event, "action": d.action, "accepted": True, "why": d.reason,
                  "steps": [st.capability for st in d.steps]}
        self.ts.consults.append(record)
        if d.action == "continue":
            return {"next": "DISPATCHING"}
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
        # replan
        # the new plan starts with a step that already failed, and nothing has moved since: the same result follows
        unchanged = getattr(self.tools, "motion_count", 0) == self.failed_motion
        first = (d.steps[0].capability, d.steps[0].args)
        repeats = unchanged and any(first == (a["capability"], a["args"]) for a in self.ts.attempted)
        if repeats:
            record["accepted"] = False
            self.log("replan", accepted=False, reason="repeats a failed step in an unchanged situation")
            self.event = f"{event} | your last plan repeated a failed step although nothing changed; choose differently"
            return {"next": "DECIDING"}
        self.log("replan", accepted=True)
        if self.event_step is not None and self.event_step.status in ("done", "failed"):
            self.event_step.status = "replaced"  # the failed / doubtful step Gemma re-planned around
        for s in pending:
            s.status = "skipped"
        self._add_steps([(st.capability, st.args) for st in d.steps], "replan")
        self._refresh_conditions()
        self._log_plan("re-planned by Gemma")
        return {"next": "DISPATCHING"}

    def _stop(self, outcome: str | None, message: str) -> dict:
        status = STOP_STATUS.get(outcome or "", "FAILED")
        tail = "" if status == "UNSUPPORTED" else " I did not report success."
        return self._finish(status, outcome or "failed", message + self._done_so_far() + (tail if status == "FAILED" else ""))

    def _verifying_goal(self, state) -> dict:
        """The original objective is the final authority: the verification worker evaluates every goal condition over
        measured state (Gemma answers count only as 'model', never as verified). If the goal is not shown, Gemma
        decides what to do next (within the budget)."""
        steps = [s for s in self.ts.plan if s.status not in ("skipped", "replaced", "refused")]
        _, v = self._delegate("verify", {"conditions": list(self.ts.conditions), "scope": "goal"}, "verify the user's goal",
                              expected=list(self.ts.conditions))
        rows = v.data.get("results", [])
        self.ts.goal_check = rows
        self.log("goal_check", results=self.ts.goal_check)
        false = [r for r in rows if r["satisfied"] is False]
        unknown = [r for r in rows if r["satisfied"] is None and r["confidence"] != "model"]
        failed_steps = [s for s in steps if s.status == "failed"]
        summary = self._result_message(steps)

        def fmt(rs):
            return "; ".join(f"{r['condition']['pred']}({', '.join(map(str, r['condition'].get('args', [])))}): {r['evidence']}" for r in rs)
        nothing = not self.ts.conditions  # no step sets a checkable effect: nothing can show the goal was reached
        if false or unknown or nothing or (failed_steps and not rows):
            gap = (fmt(false + unknown) or (f"step {failed_steps[0].index} ({failed_steps[0].capability}) failed" if failed_steps
                   else "no step of the plan sets an effect that can be verified"))
            if self.ts.replans < self.controls.max_replans and self.executed < MAX_STEPS_EXECUTED:
                self.event, self.event_outcome = f"every step ran, but the goal is not shown: {gap}", "failed"
                self.event_step = None
                self.failed_motion = getattr(self.tools, "motion_count", 0)
                self.log("review", index=None, capability="verify", outcome=v.outcome, needs="replan",
                         uncertainty=v.uncertainty, decision="ask Gemma", reason=f"goal not shown: {gap}")
                return {"next": "DECIDING"}
            if false:
                return self._finish("FAILED", "failed", f"The steps ran, but the goal is not met: {fmt(false)}.\n{summary}\n"
                                    "I did not report success.")
            if unknown:
                return self._finish("FAILED", "insufficient_info", f"The steps ran, but the evidence does not show the goal: "
                                    f"{fmt(unknown)}.\n{summary}")
            if nothing:
                return self._finish("FAILED", "insufficient_info", "The steps ran, but none of them sets an effect I can "
                                    f"verify, so I cannot say the goal was reached.\n{summary}")
            return self._finish("FAILED", failed_steps[0].outcome or "failed", f"Step {failed_steps[0].index} "
                                f"({failed_steps[0].capability}) {failed_steps[0].outcome}: {failed_steps[0].summary}"
                                + self._done_so_far() + " I did not report success.")
        model = [r for r in rows if r["confidence"] == "model"]
        refused = (f" I did not {', '.join(sorted({r['capability'] for r in self.ts.refused}))}: your request did not ask me to "
                   "change any objects." if self.ts.refused else "")
        return self._finish("COMPLETED", "completed", summary + (" (answer from Gemma, not independently verified)" if model else "")
                            + refused)

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
        self.event, self.event_outcome, self.event_step = "", None, None
        self.ctx = caps.Context(self.tools, self.log, self.run_dir, self.responder, self.budgets, self.ts)
        self.ctx.vlm_calls = lambda: getattr(self.tools.roles, "calls", 0)
        self.log("instruction", text=instruction, held=self.tools.held_object, last_referent=self.tools.last_referent)
        final = self.graph.invoke({"instruction": instruction, "timeline": []},
                                  {"configurable": {"thread_id": thread_id}, "recursion_limit": 200})
        status = final["timeline"][-1]["state"]
        self.ts.workers = self.reg.snapshot()
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
                  "refused": self.ts.refused,
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
