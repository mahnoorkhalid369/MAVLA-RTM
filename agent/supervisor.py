"""Goal supervisor: the central coordinator (LangGraph). It interprets the user's goal into a capability plan,
delegates each step to the responsible agent, checks the step's evidence, replans once if a step fails for a
recoverable reason, and completes only when every step has been verified.

    INTERPRETING -> DISPATCHING -> {OBSERVING_SCENE | REPORTING_SCENE | ANSWERING | MANIPULATING |
                                    RETURNING_SAFE | UNAVAILABLE} -> VERIFYING_STEP -> DISPATCHING ...
                 -> VERIFYING_GOAL -> COMPLETED | FAILED | CANCELLED | UNSUPPORTED
                    VERIFYING_STEP -> REPLANNING (at most MAX_REPLANS) -> DISPATCHING

Agents (logical responsibilities, one shared Gemma / YOLO instance):
  planning      Roles.plan_goal (Gemma, capability-constrained JSON, deterministic validation)
  perception    capabilities.run_observe / run_describe_scene / run_answer / run_locate (YOLO + depth + Gemma)
  manipulation  capabilities.run_pick_place (the validated PickPlaceWorkflow skill) / run_return_home
  verification  inside the skills (contact, camera placement, HOME joints) + VERIFYING_STEP / VERIFYING_GOAL here
  exploration   declared, not implemented (reported as unsupported)
Safety: the planner can only name capabilities from agent/capabilities.py; every motion still goes through
SkillPlanner + CollisionChecker + RosExecutor + the simulator's safety layer.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent import capabilities as caps
from agent.config import Budgets
from agent.eventlog import EventLog
from agent.vlm_roles import VLMFailure
from agent.world_state import Step, TaskState

STATES = ["INTERPRETING", "DISPATCHING", "OBSERVING_SCENE", "REPORTING_SCENE", "ANSWERING", "LOCATING", "MANIPULATING",
          "RETURNING_SAFE", "UNAVAILABLE", "VERIFYING_STEP", "REPLANNING", "VERIFYING_GOAL"]
TERMINAL = ["COMPLETED", "FAILED", "CANCELLED", "UNSUPPORTED"]
NODE_OF = {"observe": "OBSERVING_SCENE", "describe_scene": "REPORTING_SCENE", "answer": "ANSWERING", "locate": "LOCATING",
           "pick_place": "MANIPULATING", "return_home": "RETURNING_SAFE"}
NEEDS_OBSERVATION = {"describe_scene", "answer", "locate"}
MAX_REPLANS = 1
MAX_STEPS_EXECUTED = 10  # hard bound on capability executions per instruction (plans + replans)
RECOVERABLE = {"insufficient_info", "infeasible", "failed"}  # may be replanned once; unsafe / cancelled never
# capabilities that can achieve a failed step's objective in a replan (the step itself always can)
ALTERNATIVES = {"locate": {"explore"}, "answer": {"describe_scene"}, "describe_scene": {"answer"}}


def achieves(step_cap: str, failed_cap: str) -> bool:
    return step_cap == failed_cap or step_cap in ALTERNATIVES.get(failed_cap, set())


CONSUMERS = {"describe_scene", "answer", "locate", "pick_place", "explore"}  # steps that use an observation


def complete_plan(plan, instruction: str) -> list[str]:
    """Deterministic, logged completions of the planner's output (generic rules, not command templates):
    an observation nobody uses is reported to the user; a question gets an answer step. Returns the rules applied."""
    from agent.vlm_roles import PlanStep
    applied = []
    if instruction.strip().endswith("?") and not {"answer", "locate", "describe_scene"} & {s.do for s in plan.steps}:
        plan.steps.append(PlanStep(do="answer", question=instruction.strip()))
        applied.append("question without an answer step -> answer")
    if "observe" in {s.do for s in plan.steps} and not CONSUMERS & {s.do for s in plan.steps}:
        plan.steps.append(PlanStep(do="describe_scene"))
        applied.append("observation without a consumer -> describe_scene")
    return applied


class GoalState(TypedDict, total=False):
    instruction: str
    next: str
    timeline: list


class GoalSupervisor:
    def __init__(self, tools, log: EventLog, run_dir: Path, responder, budgets: Budgets | None = None) -> None:
        self.tools, self.log, self.run_dir = tools, log, Path(run_dir)
        self.budgets = budgets or Budgets()
        self.responder = responder
        self.ts: TaskState | None = None
        self.ctx: caps.Context | None = None
        self.executed = 0
        self.failed_motion = 0
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

    # ------------------------------------------------------------------------------------ plumbing
    def _wrap(self, name, fn):
        def node(state: GoalState) -> dict:
            self.log("state", state=name, level="goal")
            t = time.monotonic()
            out = fn(state)
            if self.ts is not None:
                self.ts.save(self.run_dir)
            return {**out, "timeline": list(state.get("timeline", [])) + [{"state": name, "seconds": round(time.monotonic() - t, 2)}]}
        return node

    def _finish(self, status: str, outcome: str, message: str) -> dict:
        self.ts.status, self.ts.outcome, self.ts.message = status.lower(), outcome, message
        return {"next": status}

    def _set_plan(self, plan, first_index: int = 1) -> None:
        self.ts.goal, self.ts.success_condition = plan.goal, plan.success
        new = [Step(first_index + i, st.capability, st.args) for i, st in enumerate(plan.steps)]
        self.ts.plan = [s for s in self.ts.plan if s.status != "pending"] + new
        self.log("goal_plan", goal=plan.goal, success=plan.success, reason=plan.reason,
                 steps=[{"index": s.index, "capability": s.capability, "args": s.args} for s in new])

    # -------------------------------------------------------------------------------------- states
    def _interpreting(self, state) -> dict:
        try:
            plan = self.tools.roles.plan_goal(state["instruction"], caps.planner_menu())
        except VLMFailure as exc:
            return self._finish("FAILED", "failed", f"I could not turn the instruction into steps I can do ({exc}). I can "
                                "observe and describe the table, answer questions about it, find a described object, move "
                                "objects onto / next to others and return HOME.")
        applied = complete_plan(plan, state["instruction"])
        if applied:
            self.log("plan_completed", rules=applied)
        self._set_plan(plan)
        if not plan.steps:
            return self._finish("UNSUPPORTED", "unsupported", f"I cannot do that with my current capabilities: "
                                f"{plan.reason or 'no matching capability'}. I can observe and describe the table, answer "
                                "questions about it, move objects onto / next to others and return HOME.")
        return {"next": "DISPATCHING"}

    def _dispatching(self, state) -> dict:
        step = self.ts.current()
        if step is None:
            return {"next": "VERIFYING_GOAL"}
        if self.executed >= MAX_STEPS_EXECUTED:
            return self._finish("FAILED", "failed", f"stopped after {MAX_STEPS_EXECUTED} steps without reaching the goal")
        spec = caps.SPECS[step.capability]
        if not spec.implemented:
            return {"next": "UNAVAILABLE"}
        step.status = "running"
        self.log("step", event="start", index=step.index, capability=step.capability, args=step.args, safety=spec.safety)
        # perception precondition: describe / answer need a fresh observation of the scene
        if step.capability in NEEDS_OBSERVATION and not self.tools.is_fresh(self.ctx.obs):
            self.log("step", event="auto_observe", index=step.index, reason="no fresh observation")
            r = caps.run_observe(self.ctx, {}, clear_view=False)
            if r.outcome != "completed":
                return self._step_done(step, r)
        return {"next": NODE_OF[step.capability]}

    def _step_done(self, step: Step, r: caps.StepResult, t0: float | None = None) -> dict:
        step.status = "done" if r.outcome == "completed" else "failed"
        step.outcome, step.summary = r.outcome, r.summary
        step.seconds = caps.elapsed(t0) if t0 else 0.0
        self.executed += 1
        self.ts.actions.append({"index": step.index, "capability": step.capability, "outcome": r.outcome,
                                "seconds": step.seconds})
        self.log("step", event="end", index=step.index, capability=step.capability, outcome=r.outcome,
                 summary=r.summary[:300], seconds=step.seconds)
        return {"next": "VERIFYING_STEP"}

    def _later_motion(self, step: Step) -> bool:
        return any(s.capability in caps.MOTION for s in self.ts.remaining(step))

    def _observing_scene(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        # park the arm out of the view only if a motion step follows (that step then reuses this observation)
        return self._step_done(step, caps.run_observe(self.ctx, step.args, clear_view=self._later_motion(step)), t0)

    def _reporting_scene(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        return self._step_done(step, caps.run_describe_scene(self.ctx, step.args), t0)

    def _answering(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        return self._step_done(step, caps.run_answer(self.ctx, step.args), t0)

    def _locating(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        return self._step_done(step, caps.run_locate(self.ctx, step.args), t0)

    def _manipulating(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        final_home = not self._later_motion(step)  # HOME once per instruction, after the last motion step
        return self._step_done(step, caps.run_pick_place(self.ctx, step.args, final_home, step.index), t0)

    def _returning_safe(self, state) -> dict:
        step, t0 = self.ts.current(), time.monotonic()
        return self._step_done(step, caps.run_return_home(self.ctx, step.args), t0)

    def _unavailable(self, state) -> dict:
        step = self.ts.current()
        step.status = "running"
        return self._step_done(step, caps.run_unimplemented(self.ctx, step.capability, step.args))

    def _verifying_step(self, state) -> dict:
        step = next(s for s in reversed(self.ts.plan) if s.status in ("done", "failed"))
        if step.outcome == "completed":
            return {"next": "DISPATCHING"}
        for s in self.ts.remaining(step):  # nothing after a failed step runs on stale assumptions
            s.status = "skipped"
        if step.outcome == "cancelled":
            return self._finish("CANCELLED", "cancelled", step.summary)
        if step.outcome == "ambiguous":  # reported, not guessed (pick_place asks the user itself)
            return self._finish("CANCELLED", "ambiguous", step.summary + self._done_so_far())
        if step.outcome == "unsupported":
            return self._finish("UNSUPPORTED", "unsupported", f"Step {step.index} ({step.capability}) is not possible yet: "
                                f"{step.summary}" + self._done_so_far())
        if step.outcome in RECOVERABLE and self.ts.replans < MAX_REPLANS and self.executed < MAX_STEPS_EXECUTED:
            self.failed_motion = getattr(self.tools, "motion_count", 0)
            return {"next": "REPLANNING"}
        return self._finish("FAILED", step.outcome, f"Step {step.index} ({step.capability}) {step.outcome}: {step.summary}"
                            + self._done_so_far() + " I did not report success.")

    def _replanning(self, state) -> dict:
        self.ts.replans += 1
        failed = next(s for s in reversed(self.ts.plan) if s.status == "failed")
        failed_motion = self.failed_motion
        note = (f"{state['instruction']}\n(Previous attempt: step '{failed.capability}' with {failed.args} ended "
                f"'{failed.outcome}': {failed.summary[:200]}. Plan again from the current situation.)")
        try:
            plan = self.tools.roles.plan_goal(note, caps.planner_menu())
        except VLMFailure as exc:
            return self._finish("FAILED", failed.outcome, f"{failed.summary} (replanning failed: {exc})")
        unchanged = getattr(self.tools, "motion_count", 0) == failed_motion
        if not any(achieves(st.capability, failed.capability) for st in plan.steps):
            why = f"the new plan no longer pursues the failed step ({failed.capability})"
        elif unchanged and any(st.capability == failed.capability and st.args == failed.args for st in plan.steps):
            why = "the new plan repeats the failed step in an unchanged situation"
        else:
            why = None
        if why:
            self.log("replan", accepted=False, reason=why)
            return self._finish("FAILED", failed.outcome, f"Step {failed.index} ({failed.capability}) {failed.outcome}: "
                                f"{failed.summary}. Replanning offered nothing different." + self._done_so_far())
        self.log("replan", accepted=True)
        failed.status = "replaced"  # superseded by the new plan; kept in the record
        self._set_plan(plan, first_index=max(s.index for s in self.ts.plan) + 1)
        return {"next": "DISPATCHING"}

    def _verifying_goal(self, state) -> dict:
        steps = [s for s in self.ts.plan if s.status not in ("skipped", "replaced")]
        ok = steps and all(s.status == "done" and s.outcome == "completed" for s in steps)
        if not ok:
            return self._finish("FAILED", "failed", "not every step was verified" + self._done_so_far())
        for r in (s for s in self.ts.plan if s.status == "replaced"):  # a replaced step's objective must be achieved later
            if not any(s.index > r.index and achieves(s.capability, r.capability) for s in steps):
                return self._finish("FAILED", r.outcome or "failed", f"Step {r.index} ({r.capability}) {r.outcome}: {r.summary}; "
                                    "the replanned steps did not achieve it." + self._done_so_far() + " I did not report success.")
        evidence = [v.get("check") for v in self.ts.verified if v.get("pass")]
        self.log("goal_verified", steps=len(steps), evidence=evidence)
        result = [s for s in steps if s.capability != "observe" and s.summary]  # what the user asked for first
        support = [s for s in steps if s.capability == "observe" and s.summary]
        msg = "\n".join(s.summary for s in result) or "\n".join(s.summary for s in support)
        if result and support:
            msg += f"\n({support[-1].summary})"
        return self._finish("COMPLETED", "completed", msg)

    def _done_so_far(self) -> str:
        done = [f"{s.index}. {s.capability}" for s in self.ts.plan if s.status == "done"]
        return f" Completed before that: {', '.join(done)}." if done else ""

    # ---------------------------------------------------------------------------------------- run
    def run(self, instruction: str, thread_id: str = "run") -> dict:
        t0 = time.monotonic()
        self.ts = TaskState(instruction)
        self.ctx = caps.Context(self.tools, self.log, self.run_dir, self.responder, self.budgets, self.ts)
        self.log("instruction", text=instruction)
        final = self.graph.invoke({"instruction": instruction, "timeline": []},
                                  {"configurable": {"thread_id": thread_id}, "recursion_limit": 120})
        status = final["timeline"][-1]["state"]
        self.ts.save(self.run_dir)
        report = {"instruction": instruction, "status": status, "outcome": self.ts.outcome, "message": self.ts.message,
                  "seconds": round(time.monotonic() - t0, 1), "goal": self.ts.goal,
                  "plan": [{"index": s.index, "capability": s.capability, "args": s.args, "status": s.status,
                            "outcome": s.outcome, "seconds": s.seconds} for s in self.ts.plan],
                  "verifications": self.ts.verified, "questions": self.ts.questions, "replans": self.ts.replans,
                  "timeline": final["timeline"], "failure": None if status == "COMPLETED" else {"cls": self.ts.outcome,
                                                                                               "reason": self.ts.message}}
        self.log("report", status=status, message=self.ts.message)
        return report
