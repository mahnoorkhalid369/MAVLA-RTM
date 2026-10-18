"""Supervisor: the central coordinator (LangGraph). It owns the user's goal and the task ledger, interprets the
goal into a desired final state plus a first plan, delegates each subtask to a worker as a typed request,
reviews each typed result (evidence, uncertainty, advisory suggestions), adapts the plan, and completes only
when the desired final state is verified over measured state.

    INTERPRETING -> DISPATCHING -> DELEGATING -> REVIEWING -> DISPATCHING ... -> VERIFYING_GOAL
                                                     |-> REPLANNING (bounded, objective-preserving) -> DISPATCHING
    terminal: COMPLETED | FAILED | CANCELLED | UNSUPPORTED

Workers (agent/workers.py; logical roles in this process, one shared Gemma / YOLO / ROS bridge):
  planning      goal -> plan (Roles.plan_goal), multi-object decomposition over measured objects (decompose)
  perception    observe, describe_scene, answer, locate
  exploration   inspect, explore (wrist-camera viewpoints, perception/viewpoints.py)
  manipulation  pick, place, pick_place, return_home (the validated PickPlaceWorkflow in three modes)
  verification  goal conditions over measured state (agent/goals.py) + the skills' own physical checks
Safety: the planner names only registry capabilities; every motion goes through SkillPlanner + CollisionChecker
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
from agent.planning import _ref_texts, achieves, complete_plan
from agent.vlm_roles import VLMFailure
from agent.workers import WorkerRegistry
from agent.task_state import Step, TaskState

STATES = ["INTERPRETING", "DISPATCHING", "DELEGATING", "REVIEWING", "REPLANNING", "VERIFYING_GOAL"]
TERMINAL = ["COMPLETED", "FAILED", "CANCELLED", "UNSUPPORTED"]
NEEDS_OBSERVATION = {"describe_scene", "answer", "locate", "inspect", "explore", "decompose"}
MOTION = {"pick", "place", "pick_place", "return_home", "inspect", "explore"}
MANIPULATION = {"pick", "place", "pick_place"}
MAX_STEPS_EXECUTED = 14  # hard bound on worker delegations per instruction
MAX_INSPECTIONS_FOR_ANSWER = 2
RECOVERABLE = {"insufficient_info", "infeasible", "failed"}  # may be replanned once; unsafe / cancelled never
PRONOUNS = {"it", "that", "this", "this one", "that one", "the object", "the held object", "#held"}


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
        self.goal_gap: str | None = None  # set when the final goal check fails and Gemma is asked again
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
        def later_manip(req):  # only manipulation this request authorizes counts (no parking for refused steps)
            return any(s.capability in MANIPULATION and self.authority.allows(self.specs[s.capability].effect)
                       for s in self.ts.remaining(self.current))

        def last_motion(req):
            return not any(s.capability in MOTION for s in self.ts.remaining(self.current))
        executors = {
            "observe": ("perception", lambda c, a, r: caps.run_observe(c, a, clear_view=a["clear_view"] if "clear_view" in a
                                                                   else later_manip(r))),
            "describe_scene": ("perception", lambda c, a, r: caps.run_describe_scene(c, a)),
            "answer": ("perception", lambda c, a, r: caps.run_answer(c, a)),
            "locate": ("perception", lambda c, a, r: caps.run_locate(c, a)),
            "inspect": ("inspection", lambda c, a, r: caps.run_inspect(c, a)),
            "explore": ("inspection", lambda c, a, r: caps.run_explore(c, a)),
            "pick": ("manipulation", lambda c, a, r: caps.run_pick(c, a, self.current.index)),
            "place": ("manipulation", lambda c, a, r: caps.run_place(c, a, last_motion(r), self.current.index)),
            "pick_place": ("manipulation", lambda c, a, r: caps.run_pick_place(c, a, last_motion(r), self.current.index)),
            "return_home": ("manipulation", lambda c, a, r: caps.run_return_home(c, a)),
            "decompose": ("planning", lambda c, a, r: caps.run_decompose(c, a)),
            "verify": ("verification", lambda c, a, r: caps.run_verify(c, a)),  # internal: never in the planner's menu
        }
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
                  depends_on: list | None = None):
        """One structured delegation: request (goal, sub-goal, context, constraints, expected result) -> registry
        (authorization first) -> worker -> structured result. The supervisor processes the result itself."""
        refs = _ref_texts(args)
        spec = {**self.specs, **caps.INTERNAL}.get(capability)
        held = self.tools.held_object
        context = {"held_object": held, "last_referent": self.tools.last_referent,
                   "bindings": dict(self.ts.bindings), "frame": self.ctx.obs.frame_id if self.ctx.obs is not None else None,
                   "done": [f"{s.index}. {s.capability}" for s in self.ts.plan if s.status == "done"]}
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

    @staticmethod
    def _measured_subgoal(step: Step) -> list[dict]:
        """The conditions a completed step should have made true, when they are measurable (physical or located /
        observed objects). Answers and reports are model / text outputs: checked only in the final goal check."""
        if step.capability not in ("pick", "place", "pick_place", "return_home", "locate", "inspect"):
            return []
        return derive_conditions([(step.capability, step.args)])

    def _finish(self, status: str, outcome: str, message: str) -> dict:
        self.ts.status, self.ts.outcome, self.ts.message = status.lower(), outcome, message
        return {"next": status}

    def _context(self) -> str:
        t = self.tools
        if t.held_object is not None:
            o = t.world.objects.get(t.held_object)
            d = (o.description or {}) if o is not None else {}
            return f"The robot is HOLDING an object ({d.get('colour', '')} {d.get('name', '')})."
        if t.last_referent is not None:
            return "The robot's gripper is empty. The last instruction handled one object (\"it\" may mean it)."
        return ""

    def _add_steps(self, steps, origin: str, after: Step | None = None) -> list[Step]:
        """steps: [(capability, args)] -> ledger Steps inserted after `after` (or appended)."""
        nxt = max([s.index for s in self.ts.plan], default=0) + 1
        new = [Step(nxt + i, c, copy.deepcopy(a), worker=self.reg.worker_of(c), origin=origin) for i, (c, a) in enumerate(steps)]
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
        live = [(s.capability, s.args) for s in self.ts.plan if s.status not in ("skipped", "replaced", "refused")
                and s.capability not in ("observe", "decompose")]
        derived = [{**c, "source": "derived"} for c in derive_conditions(live)]
        if any(c["pred"] == "answered" for c in request):  # the request's own question already stands for the answer
            derived = [c for c in derived if c["pred"] != "answered"]
        self.ts.conditions = request + (explicit or derived)

    # ------------------------------------------------------------------------- state-aware binding
    def _bind_refs(self, step: Step) -> str | None:
        """Pronouns -> the held / last-handled object (#id); returns a reason if a reference cannot be bound."""
        t = self.tools
        for k in ("object", "reference"):
            v = step.args.get(k)
            if isinstance(v, dict) and (v.get("text") or "").strip().lower() in PRONOUNS:
                who = t.held_object if t.held_object is not None else t.last_referent
                if who is None:
                    return f"it is unclear what {v['text']!r} refers to (nothing held, nothing handled before)"
                why = "held object" if t.held_object is not None else "last handled object"
                self.log("binding", text=v["text"], world_id=who, reason=why)
                self.ts.assumptions.append({"step": step.index, "assumed": f"{v['text']!r} means #{who}", "because": why})
                v["text"] = f"#{who}"
        return None

    def _adapt_to_robot_state(self, step: Step) -> None:
        """Deterministic, logged adjustment from the measured robot state: never regrasp a held object."""
        held = self.tools.held_object
        obj = (step.args.get("object") or {}).get("text") if isinstance(step.args.get("object"), dict) else None
        if held is not None and obj == f"#{held}" and step.capability == "pick_place":
            step.capability, step.worker = "place", "manipulation"
            self.log("plan_adjusted", index=step.index, change="pick_place -> place", reason=f"#{held} is already held")
            self._refresh_conditions()

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
            return self._finish("FAILED", "failed", f"I could not turn the instruction into steps I can do ({exc}). "
                                + self._capabilities_line())
        applied = complete_plan(plan, state["instruction"], self.authority.kind)
        if applied:
            self.log("plan_completed", rules=applied)
        self.ts.goal, self.ts.success_condition = plan.goal, plan.success
        if not plan.steps:
            return self._finish("UNSUPPORTED", "unsupported", f"I cannot do that with my current capabilities: "
                                f"{plan.reason or 'no matching capability'}. " + self._capabilities_line())
        self._add_steps([(st.capability, st.args) for st in plan.steps], "planner")
        if self.authority.kind == "question":  # the goal of a question is its answer (a report alone is not one)
            self.ts.conditions = [{"pred": "answered", "args": [state["instruction"]], "source": "request"}]
        self._refresh_conditions()
        self.log("goal_conditions", goal=plan.goal, conditions=self.ts.conditions)
        return {"next": "DISPATCHING"}

    @staticmethod
    def _capabilities_line() -> str:
        return ("I can observe and describe the table, answer questions about it, find or inspect a described object "
                "(also behind others), pick up an object and hold it, place a held object, move one object onto / next to "
                "another, plan multi-object moves over the measured objects, and return HOME.")

    def _dispatching(self, state) -> dict:
        step = self.ts.current()
        if step is None:
            return {"next": "VERIFYING_GOAL"}
        if self.executed >= MAX_STEPS_EXECUTED:
            return self._finish("FAILED", "failed", f"stopped after {MAX_STEPS_EXECUTED} subtasks without reaching the goal")
        before = copy.deepcopy(step.args)
        why = self._bind_refs(step)
        if step.args != before:
            self._refresh_conditions()  # derived conditions now name the bound object (#id), not the pronoun
        if why:
            step.status, step.outcome, step.summary = "failed", "insufficient_info", why
            return self._finish("CANCELLED", "insufficient_info", why + " — please name the object.")
        self._adapt_to_robot_state(step)
        if step.capability in NEEDS_OBSERVATION and not self.tools.is_fresh(self.ctx.obs):
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
                                  depends_on=[f"step{d}" for d in step.depends_on])
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
        """Critic of one result: verify the sub-goal, then adapt the plan from the evidence (the supervisor decides;
        workers only advise through `needs` / `suggestion`)."""
        step = next(s for s in reversed(self.ts.plan) if s.status in ("done", "failed"))
        res = self.last_result
        data = res.data if res is not None else {}
        if step.outcome == "completed" and (conds := self._measured_subgoal(step)):
            _, v = self._delegate("verify", {"conditions": conds, "scope": "subgoal"},
                                  f"verify step {step.index} ({step.capability})", expected=conds)
            self.ts.subgoal_checks.append({"step": step.index, "task_id": v.task_id, "outcome": v.outcome,
                                           "summary": v.summary, "results": v.data.get("results", [])})
            if v.outcome == "failed":  # the worker reported success, the measured state says otherwise
                step.status, step.outcome = "failed", "failed"
                step.summary = f"{step.summary} | verification: the sub-goal is {v.summary}"
                self._review(step, "sub-goal not met", v.summary)
            elif v.outcome != "completed":
                self.ts.uncertainties.append({"task_id": v.task_id, "capability": "verify", "items": v.uncertainty})
        if step.outcome == "completed":
            if step.capability == "decompose":  # planning worker returned concrete steps + final conditions
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
            if (step.capability == "answer" and data.get("certain") is False and data.get("check_closer_world")
                    and not any(s.capability == "inspect" for s in self.ts.plan) and self.specs["inspect"].implemented):
                ids = data["check_closer_world"][:MAX_INSPECTIONS_FOR_ANSWER]
                step.status = "replaced"
                self.log("subtask", reason=f"answer uncertain from the table view: closer look at "
                                           f"{['#%d' % i for i in ids]}, then answer again")
                self._add_steps([("inspect", {"target": f"#{i}"}) for i in ids] + [("answer", dict(step.args))],
                                "subtask policy", after=step)
                self._review(step, "closer look, then answer again", "the worker is not certain and names objects to check")
                self._log_plan("subtask: closer look")
                return {"next": "DISPATCHING"}
            self._review(step, "continue", "sub-goal achieved" + (" (verified)" if self._measured_subgoal(step) else ""))
            return {"next": "DISPATCHING"}
        if step.outcome == "unauthorized":  # refused before execution: drop it (and later steps of the same kind), go on
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
        for s in self.ts.remaining(step):  # nothing after a failed step runs on stale assumptions (may be re-added)
            s.status = "skipped"
        # information-gathering subtask: a target not visible to the table camera -> explore once, then locate again
        if (step.capability == "locate" and step.outcome == "insufficient_info" and self.specs["explore"].implemented
                and not any(s.capability == "explore" for s in self.ts.plan) and self.executed < MAX_STEPS_EXECUTED - 2):
            step.status = "replaced"
            for s in self.ts.plan:
                if s.status == "skipped" and s.index > step.index:
                    s.status = "pending"
            self.log("subtask", reason=f"{step.args.get('target')!r} not visible to the table camera: explore hidden "
                                       "regions, then locate again")
            self._add_steps([("explore", {"target": step.args.get("target"), "near": None}), ("locate", dict(step.args))],
                            "subtask policy", after=step)
            self._review(step, "explore, then locate again", "not visible from the current view")
            self._log_plan("subtask: explore")
            return {"next": "DISPATCHING"}
        if step.outcome in ("cancelled", "ambiguous", "unsupported"):
            self._review(step, f"stop ({step.outcome})", step.summary)
        if step.outcome == "cancelled":
            return self._finish("CANCELLED", "cancelled", step.summary + self._done_so_far())
        if step.outcome == "ambiguous":
            return self._finish("CANCELLED", "ambiguous", step.summary + self._done_so_far())
        if step.outcome == "unsupported":
            return self._finish("UNSUPPORTED", "unsupported", f"Step {step.index} ({step.capability}) is not possible: "
                                f"{step.summary}" + self._done_so_far())
        self.ts.attempted.append({"capability": step.capability, "args": step.args, "outcome": step.outcome})
        deterministic = step.outcome == "infeasible"  # e.g. out of reach: re-planning the same move cannot help
        if step.outcome in RECOVERABLE and not deterministic and self.ts.replans < self.controls.max_replans and self.executed < MAX_STEPS_EXECUTED:
            self.failed_motion = getattr(self.tools, "motion_count", 0)
            self._review(step, "ask Gemma to re-plan", f"recoverable {step.outcome}; re-planning budget "
                                                       f"{self.ts.replans}/{self.controls.max_replans} used")
            return {"next": "REPLANNING"}
        self._review(step, f"stop ({step.outcome})", "not recoverable" if step.outcome not in RECOVERABLE or deterministic
                     else "re-planning budget used")
        return self._finish("FAILED", step.outcome, f"Step {step.index} ({step.capability}) {step.outcome}: {step.summary}"
                            + self._done_so_far() + " I did not report success.")

    def _replanning(self, state) -> dict:
        """Gemma decides again, from the goal, the progress so far and the new evidence: after a recoverable failure,
        or when every step ran but the goal check is not satisfied (self.goal_gap)."""
        self.ts.replans += 1
        gap, self.goal_gap = self.goal_gap, None
        failed = None if gap else next(s for s in reversed(self.ts.plan) if s.status == "failed")
        tried = "; ".join(f"{a['capability']} {a['args']} -> {a['outcome']}" for a in self.ts.attempted)[:400]
        if gap:
            note = (f"{state['instruction']}\n(The steps so far did not achieve the goal: {gap}. Plan only the remaining "
                    "steps from the current situation.)")
        else:
            note = f"{state['instruction']}\n(Already tried, do not repeat: {tried}. Plan again from the current situation.)"
        context = "\n".join(x for x in (self._context(), self.ts.progress_text()) if x)
        self.log("consult", reason=f"goal not met: {gap}" if gap else f"step {failed.index} ({failed.capability}) {failed.outcome}",
                 replans=self.ts.replans, budget=self.controls.max_replans)
        try:
            plan = self.tools.roles.plan_goal(note, caps.planner_menu(self.specs), context=context)
        except VLMFailure as exc:
            self.ts.consults.append({"reason": "goal" if gap else "failure", "accepted": False, "why": str(exc)})
            if gap:
                return self._finish("FAILED", "failed", f"The steps ran, but the goal is not met: {gap}. (Re-planning failed: "
                                    f"{exc}.)" + self._done_so_far() + " I did not report success.")
            return self._finish("FAILED", failed.outcome, f"{failed.summary} (replanning failed: {exc})" + " I did not report success.")
        unchanged = getattr(self.tools, "motion_count", 0) == self.failed_motion
        done = [(s.capability, s.args) for s in self.ts.plan if s.status == "done"]
        if not plan.steps:
            why = f"Gemma found no further steps ({plan.reason or 'no reason given'})"
        elif failed is not None and not any(achieves(st.capability, failed.capability) for st in plan.steps):
            why = f"the new plan no longer pursues the failed step ({failed.capability})"
        elif unchanged and any((st.capability, st.args) == (a["capability"], a["args"]) for st in plan.steps for a in self.ts.attempted):
            why = "the new plan repeats an attempted step in an unchanged situation"
        elif gap and unchanged and all((st.capability, st.args) in done for st in plan.steps):
            why = "the new plan only repeats completed steps in an unchanged situation"
        else:
            why = None
        self.ts.consults.append({"reason": "goal" if gap else "failure", "accepted": why is None, "why": why or plan.reason,
                                 "steps": [st.capability for st in plan.steps]})
        if why:
            self.log("replan", accepted=False, reason=why)
            if gap:
                return self._finish("FAILED", "failed", f"The steps ran, but the goal is not met: {gap}. Re-planning offered "
                                    "nothing different." + self._done_so_far() + " I did not report success.")
            return self._finish("FAILED", failed.outcome, f"Step {failed.index} ({failed.capability}) {failed.outcome}: "
                                f"{failed.summary}. Replanning offered nothing different." + self._done_so_far())
        self.log("replan", accepted=True)
        if failed is not None:
            failed.status = "replaced"
        self._add_steps([(st.capability, st.args) for st in plan.steps], "replan")
        self._refresh_conditions()
        self._log_plan("re-planned by Gemma (" + ("goal not met" if gap else "after a failure") + ")")
        return {"next": "DISPATCHING"}

    def _verifying_goal(self, state) -> dict:
        """The original objective is the final authority: the verification worker evaluates every desired-final-state
        condition over measured state (Gemma answers count only as 'model', never as verified). If the goal is not
        shown and re-planning budget remains, Gemma decides the remaining steps (closed loop)."""
        steps = [s for s in self.ts.plan if s.status not in ("skipped", "replaced", "refused")]
        if not steps or not all(s.status == "done" and s.outcome == "completed" for s in steps):
            return self._finish("FAILED", "failed", "not every step was completed" + self._done_so_far())
        for r in (s for s in self.ts.plan if s.status == "replaced"):  # a replaced step's objective must be achieved later
            if r.origin != "subtask policy" and not any(s.index > r.index and achieves(s.capability, r.capability) for s in steps):
                return self._finish("FAILED", r.outcome or "failed", f"Step {r.index} ({r.capability}) {r.outcome}: {r.summary}; "
                                    "the replanned steps did not achieve it." + self._done_so_far() + " I did not report success.")
        _, v = self._delegate("verify", {"conditions": list(self.ts.conditions), "scope": "goal"}, "verify the user's goal",
                              expected=list(self.ts.conditions))
        rows = v.data.get("results", [])
        self.ts.goal_check = rows
        self.log("goal_check", results=self.ts.goal_check)
        false = [r for r in rows if r["satisfied"] is False]
        unknown = [r for r in rows if r["satisfied"] is None and r["confidence"] != "model"]
        summary = self._result_message(steps)

        def fmt(rs):
            return "; ".join(f"{r['condition']['pred']}({', '.join(map(str, r['condition'].get('args', [])))}): {r['evidence']}" for r in rs)
        if (false or unknown) and self.ts.replans < self.controls.max_replans and self.executed < MAX_STEPS_EXECUTED:
            self.goal_gap = fmt(false + unknown)
            self.failed_motion = getattr(self.tools, "motion_count", 0)
            self.log("review", index=None, capability="verify", outcome=v.outcome, needs="replan", uncertainty=v.uncertainty,
                     decision="ask Gemma for the remaining steps", reason=f"goal not shown: {self.goal_gap}")
            return {"next": "REPLANNING"}
        if false:
            return self._finish("FAILED", "failed", f"The steps ran, but the goal is not met: {fmt(false)}.\n{summary}\n"
                                "I did not report success.")
        if unknown:
            return self._finish("FAILED", "insufficient_info", f"The steps ran, but the evidence does not show the goal: "
                                f"{fmt(unknown)}.\n{summary}")
        model = [r for r in rows if r["confidence"] == "model"]
        refused = (f" I did not {', '.join(sorted({r['capability'] for r in self.ts.refused}))}: your request did not ask me to "
                   "change any objects." if self.ts.refused else "")
        return self._finish("COMPLETED", "completed", summary + (" (answer from Gemma, not independently verified)" if model else "")
                            + refused)

    @staticmethod
    def _result_message(steps) -> str:
        supporting = {"observe", "explore", "decompose"}
        result = [s for s in steps if s.capability not in supporting and s.summary]
        support = [s for s in steps if s.capability in supporting and s.summary]
        msg = "\n".join(s.summary for s in result) or "\n".join(s.summary for s in support)
        if result and support:
            msg += "\n(" + "; ".join(s.summary for s in support[-2:]) + ")"
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
        self.goal_gap = None
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
