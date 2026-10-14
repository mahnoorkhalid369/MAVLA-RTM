"""Supervisor: a LangGraph state machine over the agent tools (ARCHITECTURE.md rev 3, demo slice).

IDLE -> UNDERSTANDING -> OBSERVING -> LOCALIZING -> PLANNING_GRASP -> APPROACHING -> GRASPING ->
VERIFYING_GRASP -> TRANSPORTING -> PLACING -> RETREATING -> VERIFYING_PLACEMENT -> RETURNING_HOME ->
VERIFYING_HOME -> COMPLETED
                                          any -> RECOVERING (budgeted) | FAILED | CANCELLED

The LLM only parses, names, resolves and confirms; motions come from SkillPlanner (collision-checked)
and go through RosExecutor and the simulator's safety layer. "COMPLETED" needs verification passed.
The graph state holds plain data only (checkpointed to SQLite); plans and images live in `ctx`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.ask import Answer, Candidate, Question
from agent.config import Budgets
from agent.vlm_roles import VLMFailure
from control.motion import PlanningError

STATES = ["UNDERSTANDING", "OBSERVING", "LOCALIZING", "PLANNING_GRASP", "APPROACHING", "GRASPING",
          "VERIFYING_GRASP", "TRANSPORTING", "PLACING", "RETREATING", "VERIFYING_PLACEMENT", "RETURNING_HOME",
          "VERIFYING_HOME", "RECOVERING"]
TERMINAL = ["COMPLETED", "FAILED", "CANCELLED"]


class EventLog:
    """JSON-lines event log: one object per line with time, kind and fields."""

    def __init__(self, path: Path, echo: bool = False) -> None:
        """echo: also print a readable line per event to the terminal (agent.events.format_event)."""
        self.path, self.t0, self.echo = Path(path), time.monotonic(), echo
        self.f = open(self.path, "a", buffering=1)

    def __call__(self, kind: str, **fields) -> None:
        rec = {"t": round(time.monotonic() - self.t0, 3), "wall": time.strftime("%H:%M:%S"), "kind": kind, **fields}
        line = json.dumps(rec, default=_jsonable)
        self.f.write(line + "\n")
        if self.echo:
            from agent.events import format_event
            ev = format_event(json.loads(line))
            if ev is not None:
                print(str(ev), flush=True)

    def close(self) -> None:
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


class AgentState(TypedDict, total=False):
    instruction: str
    next: str
    task: dict
    object_id: int | None
    reference_id: int | None
    retries: dict
    recover: dict  # {"cls": ..., "reason": ...} while RECOVERING
    questions: list
    timeline: list  # [{"state", "t"}]
    verifications: list
    failure: dict | None
    plan: dict | None


class Supervisor:
    def __init__(self, tools, log: EventLog, run_dir: Path, responder, budgets: Budgets | None = None) -> None:
        """responder: agent.ask.TerminalResponder (a person) or ScriptedResponder (unattended runs).
        Required: there is no silent default answer."""
        self.tools, self.log, self.run_dir, self.responder = tools, log, Path(run_dir), responder
        self.budgets = budgets or Budgets()
        self.ctx: dict = {}  # non-serialisable working data: observation, plan
        g = StateGraph(AgentState)
        for s in STATES:
            g.add_node(s, self._wrap(s, getattr(self, "_" + s.lower())))
        for s in TERMINAL:
            g.add_node(s, self._wrap(s, lambda st: {}))
            g.add_edge(s, END)
        g.set_entry_point("UNDERSTANDING")
        for s in STATES:
            g.add_conditional_edges(s, lambda st: st["next"], {n: n for n in STATES + TERMINAL})
        self.conn = sqlite3.connect(str(self.run_dir / "checkpoints.sqlite"), check_same_thread=False)
        self.graph = g.compile(checkpointer=SqliteSaver(self.conn))

    # ----------------------------------------------------------------- plumbing
    def _wrap(self, name, fn):
        def node(state: AgentState) -> dict:
            self.log("state", state=name)
            t = time.monotonic()
            out = fn(state)
            timeline = list(state.get("timeline", [])) + [{"state": name, "seconds": round(time.monotonic() - t, 2)}]
            return {**out, "timeline": timeline, "questions": list(self.ctx.get("questions", []))}
        return node

    def _fail(self, state, cls: str, reason: str, **extra) -> dict:
        """Budgeted recovery: RECOVERING if the class has retries left, else FAILED."""
        used = dict(state.get("retries", {}))
        budget = self.budgets.retries.get(cls, 0)
        self.log("failure", cls=cls, reason=reason, used=used.get(cls, 0), budget=budget, **extra)
        if used.get(cls, 0) < budget:
            used[cls] = used.get(cls, 0) + 1
            return {"next": "RECOVERING", "retries": used, "recover": {"cls": cls, "reason": reason}}
        return {"next": "FAILED", "retries": used, "failure": {"cls": cls, "reason": reason, **extra}}

    def _candidates(self, obs, marks) -> list[Candidate]:
        out = []
        for k in marks:
            g, d = obs.measured[k], obs.descriptions.get(k, {}) or {}
            desc = " ".join(str(d.get(f, "")) for f in ("colour", "name")).strip() or "(no description)"
            out.append(Candidate(k, desc, (g["centre"][0] * 1000, g["centre"][1] * 1000), float(np.hypot(*g["centre"][:2])),
                                 (g["footprint"][0] * 1000, g["footprint"][1] * 1000, g["height"] * 1000)))
        return out

    def _question(self, kind: str, text: str, role: str, obs, marks) -> Answer:
        """Ask via the configured responder (a person, or a script in unattended runs) and log it."""
        image = None
        if hasattr(self.tools, "candidate_image"):
            image = self.tools.candidate_image(obs, list(marks))
        q = Question(kind, text, role, self._candidates(obs, marks), image)
        self.log("question", question=text, question_kind=kind, role=role, candidates=[c.mark for c in q.candidates], image=image)
        a = self.responder(q)
        rec = {"question": text, "question_kind": kind, "role": role, "candidates": [c.mark for c in q.candidates],
               "answer": a.kind, "mark": a.mark, "raw": a.raw, "source": a.source, "rejected": a.attempts}
        self.ctx.setdefault("questions", []).append(rec)
        self.log("ask_user", **rec)
        return a

    def _cancelled(self, a: Answer, question: str) -> dict:
        why = {"cancel": f"you answered {a.raw!r}", "eof": "end of input (nobody could answer)",
               "timeout": f"no answer within {getattr(self.responder, 'timeout_s', 0):.0f} s"}[a.kind]
        return {"next": "CANCELLED", "failure": {"cls": "asked", "reason": f"{question} -> {why}", "answer_source": a.source}}

    def _run(self, steps, label, state, next_state) -> dict:
        res = self.tools.execute(steps, label)
        if not res.ok:
            return self._fail(state, "execution", f"{label}: {res.failed_step}: {res.reason}",
                              safety_events=list(self.tools.ex.safety_events[-3:]))
        return {"next": next_state}

    # -------------------------------------------------------------------- states
    def _understanding(self, state) -> dict:
        try:
            task = self.tools.roles.parse(state["instruction"])
        except VLMFailure as exc:
            return self._fail(state, "understanding", str(exc))
        self.log("task", **task.model_dump())
        if task.task != "pick_place" or task.object is None or task.reference is None or task.relation is None:
            return {"task": task.model_dump(), "next": "FAILED",
                    "failure": {"cls": "unsupported", "reason": f"I can only move one object onto or next to another "
                                f"({task.reason or task.task}); please rephrase the instruction"}}
        return {"next": "OBSERVING", "task": task.model_dump()}

    def _observing(self, state) -> dict:
        p = self.tools.park_for_view(None)  # whole workspace: least-blocking pose
        if not p["ok"]:
            return self._fail(state, "perception", "no collision-free pose that leaves the table camera's view clear")
        try:
            obs = self.tools.perceive("table")
            self.tools.describe(obs)
        except (VLMFailure, TimeoutError) as exc:
            return self._fail(state, "perception", str(exc))
        self.tools.update_world(obs)
        self.ctx["obs"] = obs
        self.log("world", objects=self.tools.world.summary(),
                 names={k: obs.descriptions.get(k, {}).get("name") for k in obs.marks})
        if not obs.marks:
            return self._fail(state, "perception", "no objects measured on the table")
        return {"next": "LOCALIZING"}

    def _resolve_one(self, state, obs, ref: dict, role: str):
        """-> (mark, None) or (None, update dict). Several matches -> the user chooses (never the model)."""
        try:
            r = self.tools.resolve(obs, ref["text"])
        except VLMFailure as exc:
            return None, self._fail(state, "perception", str(exc))
        self.log("resolution", role=role, text=ref["text"], **r.model_dump())
        user_chosen = False
        mark = r.id
        if r.id is None:
            if not r.candidates:
                return None, self._fail(state, "perception", f"nothing matches {ref['text']!r}", role=role)
            question = f"Several objects could be {ref['text']!r}. Which one do you mean?"
            if len(self.ctx.get("questions", [])) >= self.budgets.max_questions:
                return None, {"next": "CANCELLED", "failure": {"cls": "asked", "reason": f"{question} -> question budget "
                                                                f"({self.budgets.max_questions}) used up"}}
            a = self._question("choose", question, role, obs, r.candidates)
            if a.kind != "choice":
                return None, self._cancelled(a, question)
            mark, user_chosen = a.mark, True
        if ref.get("relation") and ref.get("anchor"):
            try:
                an = self.tools.resolve(obs, ref["anchor"])
            except VLMFailure as exc:
                return None, self._fail(state, "perception", str(exc))
            if an.id is None or not self.tools.relation_holds(obs.measured[mark], obs.measured[an.id], ref["relation"],
                                                               state["task"].get("frame", "robot")):
                return None, {"next": "FAILED", "failure": {"cls": "perception", "reason": f"could not verify that "
                              f"{ref['text']!r} is {ref['relation']} {ref['anchor']!r} from the measured positions"}}
        if user_chosen:  # the person's choice is the answer; the model does not re-judge it
            self.log("confirmation", role=role, mark=mark, text=ref["text"], match="user_selected", reason="chosen by the user")
            return mark, None
        try:
            c = self.tools.confirm(obs, mark, ref["text"])
        except VLMFailure as exc:
            return None, self._fail(state, "perception", str(exc))
        self.log("confirmation", role=role, mark=mark, text=ref["text"], **c.model_dump())
        if c.match == "no":
            return None, self._fail(state, "perception", f"confirm_object said no for {ref['text']!r} (mark {mark})", role=role)
        if c.match == "unsure":
            question = f"Is object {mark} the {ref['text']!r}? I am not sure."
            others = [k for k in (self.tools.lexical_candidates(obs, ref["text"]) if hasattr(self.tools, "lexical_candidates") else [])
                      if k != mark]
            a = self._question("confirm", question, role, obs, [mark] + others)
            if a.kind == "yes":
                return mark, None
            if a.kind == "choice":
                return a.mark, None
            if a.kind == "no":
                return None, {"next": "FAILED", "failure": {"cls": "asked", "reason": f"you said object {mark} is not "
                              f"{ref['text']!r} and chose no other object"}}
            return None, self._cancelled(a, question)
        return mark, None

    def _localizing(self, state) -> dict:
        obs, task = self.ctx["obs"], state["task"]
        m_obj, upd = self._resolve_one(state, obs, task["object"], "object")
        if upd:
            return upd
        m_ref, upd = self._resolve_one(state, obs, task["reference"], "reference")
        if upd:
            return upd
        if m_obj == m_ref:
            return {"next": "FAILED", "failure": {"cls": "understanding", "reason": f"the object and the destination "
                    f"are the same object ({m_obj}); please rephrase"}}
        oid, rid = obs.object_ids[m_obj], obs.object_ids[m_ref]
        self.log("localized", object_mark=m_obj, object_id=oid, reference_mark=m_ref, reference_id=rid,
                 object=self.tools.world.objects[oid].geometry, reference=self.tools.world.objects[rid].geometry)
        return {"next": "PLANNING_GRASP", "object_id": oid, "reference_id": rid}

    def _planning_grasp(self, state) -> dict:
        task = state["task"]
        try:
            plan = self.tools.plan_pick_place(state["object_id"], task["relation"], state["reference_id"],
                                              task.get("frame", "robot"), self.tools.bridge.q())
        except PlanningError as exc:
            return self._fail(state, "planning", str(exc))
        self.ctx["plan"] = plan
        summary = {"grasp": plan.grasp, "target_centre_mm": [round(v * 1000, 1) for v in plan.target_centre],
                   "target_top_mm": round(plan.target_top * 1000, 1), "relation": plan.relation,
                   "held": {"height_mm": round(plan.held.height * 1000, 1), "grip_depth_mm": round(plan.held.grip_depth * 1000, 1),
                            "width_mm": round(plan.held.width * 1000, 1), "length_mm": round(plan.held.length * 1000, 1)}}
        self.log("plan", **summary)
        return {"next": "APPROACHING", "plan": summary}

    def _approaching(self, state) -> dict:
        p = self.ctx["plan"]
        return self._run(p.hover + p.approach[:1], "approach", state, "GRASPING")

    def _grasping(self, state) -> dict:
        p = self.ctx["plan"]
        return self._run(p.approach[1:], "close", state, "VERIFYING_GRASP")

    def _verifying_grasp(self, state) -> dict:
        p = self.ctx["plan"]
        v = self.tools.verify_contact(p.approach[1].position)
        vs = list(state.get("verifications", [])) + [v]
        if not v["pass"]:
            out = self._fail(state, "grasp", "the fingers closed without stalling on the object (no contact)")
            return {**out, "verifications": vs}
        self.tools.held_box = p.held.collision_body()
        self.tools.world.holding = p.object_id
        self.tools.world.objects[p.object_id].status = "held"
        return {"next": "TRANSPORTING", "verifications": vs}

    def _transporting(self, state) -> dict:
        p = self.ctx["plan"]
        out = self._run(p.lift + p.place[:1], "lift and carry", state, "PLACING")
        if out["next"] != "PLACING":
            return out
        v = self.tools.verify_contact(p.approach[1].position)  # still holding after the carry?
        v["check"] = "contact_after_carry"
        vs = list(state.get("verifications", [])) + [v]
        if not v["pass"]:
            return {**self._fail(state, "execution", "object lost during transport (fingers no longer stalled)"), "verifications": vs}
        return {**out, "verifications": vs}

    def _placing(self, state) -> dict:
        p = self.ctx["plan"]
        return self._run(p.place[1:-1], "lower and release", state, "RETREATING")

    def _retreating(self, state) -> dict:
        p = self.ctx["plan"]  # the planner's own collision-checked straight retreat above the released object
        return self._run(p.place[-1:], "retreat", state, "VERIFYING_PLACEMENT")

    def _verifying_placement(self, state) -> dict:
        p = self.ctx["plan"]
        self.tools.assume_placed(p)
        park = self.tools.park_for_view(p.target_centre[:2])
        if not park["ok"]:
            return self._fail(state, "placement", "no pose with a clear camera view of the placed object")
        try:
            v = self.tools.verify_placement(p)
        except (VLMFailure, TimeoutError) as exc:
            return self._fail(state, "placement", f"re-observation failed: {exc}")
        vs = list(state.get("verifications", [])) + [v]
        if v["pass"]:
            return {"next": "RETURNING_HOME", "verifications": vs}
        return {**self._fail(state, "placement", v.get("reason") or "placement outside tolerance (xy 10 mm, z 5 mm, tilt 5 deg)"),
                "verifications": vs}

    def _returning_home(self, state) -> dict:
        """Once per instruction, after a VERIFIED placement only (failures never move the arm home)."""
        r = self.tools.return_home()
        if r["ok"]:
            return {"next": "VERIFYING_HOME"}
        if r.get("result") is not None:  # it moved and the executor / safety layer reported a failure
            return self._fail(state, "execution", f"return HOME: {r['reason']}",
                              safety_events=list(self.tools.ex.safety_events[-3:]))
        return {"next": "FAILED", "failure": {"cls": "home", "reason": f"HOME not attempted: {r['reason']}"}}

    def _verifying_home(self, state) -> dict:
        v = self.tools.verify_home()
        vs = list(state.get("verifications", [])) + [v]
        if v["pass"]:
            return {"next": "COMPLETED", "verifications": vs}
        return {"next": "FAILED", "verifications": vs, "failure": {"cls": "home", "reason": f"HOME not reached: max joint "
                f"error {v['max_error_deg']} deg > {v['tolerance_deg']} deg"}}

    def _recovering(self, state) -> dict:
        r = state["recover"]
        self.log("recovery", **r, retries=state["retries"])
        cls = r["cls"]
        if cls == "grasp":  # open, lift back up (collision-checked lift from the plan), re-observe everything
            p = self.ctx["plan"]
            from control.robot import GRIPPER_OPEN
            from control.skills import GripperStep
            res = self.tools.execute([GripperStep(GRIPPER_OPEN, "open gripper", 0.8)] + p.lift, "grasp recovery")
            if not res.ok:
                return self._fail(state, "execution", f"grasp recovery: {res.reason}")
            return {"next": "OBSERVING"}
        if cls == "placement":  # look again once (whole-scene re-perception), same tolerances
            return {"next": "VERIFYING_PLACEMENT"}
        if cls in ("perception", "planning"):
            return {"next": "OBSERVING"}
        return {"next": "FAILED", "failure": {"cls": cls, "reason": r["reason"]}}

    # ------------------------------------------------------------------- run
    def run(self, instruction: str, thread_id: str = "run") -> dict:
        t0 = time.monotonic()
        self.log("instruction", text=instruction)
        init: AgentState = {"instruction": instruction, "retries": {}, "questions": [], "timeline": [],
                            "verifications": [], "failure": None, "plan": None}
        final = self.graph.invoke(init, {"configurable": {"thread_id": thread_id}, "recursion_limit": 80})
        status = final["timeline"][-1]["state"]
        report = {"instruction": instruction, "status": status, "seconds": round(time.monotonic() - t0, 1),
                  "task": final.get("task"), "object_id": final.get("object_id"), "reference_id": final.get("reference_id"),
                  "plan": final.get("plan"), "verifications": final.get("verifications"), "failure": final.get("failure"),
                  "questions": final.get("questions"), "retries": final.get("retries"), "timeline": final["timeline"],
                  "world": self.tools.world.to_dict()}
        report["message"] = honest_message(report)
        self.log("report", status=status, message=report["message"])
        return report


def honest_message(r: dict) -> str:
    s = r["status"]
    vs = r.get("verifications") or []
    place = [x for x in vs if x.get("check") == "placement" and x.get("pass")]
    home = [x for x in vs if x.get("check") == "home"]
    placed = (f"placed {place[-1]['xy_error_mm']} mm from the planned spot, top {place[-1]['z_error_mm']:+} mm, "
              f"tilt {place[-1]['tilt_deg']} deg") if place else ""
    if s == "COMPLETED":
        return (f"Done (verified by the table camera): {placed}; back at HOME (max joint error "
                f"{home[-1]['max_error_deg']} deg).")
    if s == "CANCELLED":
        return f"Cancelled: {r['failure']['reason']}."
    f = r.get("failure") or {}
    pre = f"Placement verified ({placed}), but " if place else ""
    return f"{pre}Failed ({f.get('cls', '?')}): {f.get('reason', 'unknown')}. I did not report success."
