"""Pick-and-place SKILL: the validated LangGraph chain, run by the goal supervisor (agent/supervisor.py) for
one already-interpreted pick-and-place step.

OBSERVING -> LOCALIZING -> PLANNING_GRASP -> APPROACHING -> GRASPING -> VERIFYING_GRASP -> TRANSPORTING ->
PLACING -> RETREATING -> VERIFYING_PLACEMENT -> [RETURNING_HOME -> VERIFYING_HOME] -> COMPLETED
                                          any -> RECOVERING (budgeted) | FAILED | CANCELLED

Unchanged from the validated demo (2026-10-14) except: interpretation moved to the planner (no UNDERSTANDING
state here), a fresh cleared-view observation from the supervisor can be reused (no second capture), and the
final HOME phase is skipped when another manipulation step follows (final_home=False). The LLM only names,
resolves and confirms; motions come from SkillPlanner (collision-checked) through RosExecutor and the
simulator's safety layer. "COMPLETED" needs verification passed.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.ask import Answer, Candidate, Question
from agent.eventlog import EventLog
from config.agent import Budgets
from agent.vlm_roles import VLMFailure
from config import layout
from control.motion import PlanningError

STATES = ["OBSERVING", "LOCALIZING", "PLANNING_GRASP", "APPROACHING", "GRASPING",
          "VERIFYING_GRASP", "TRANSPORTING", "PLACING", "RETREATING", "VERIFYING_PLACEMENT", "RETURNING_HOME",
          "VERIFYING_HOME", "RECOVERING"]
TERMINAL = ["COMPLETED", "FAILED", "CANCELLED"]


class AgentState(TypedDict, total=False):
    instruction: str
    next: str
    task: dict
    object_id: int | None
    reference_id: int | None
    reference_region: str | None  # a fixed region of the calibrated layout (config/layout.py) as the destination
    retries: dict
    recover: dict  # {"cls": ..., "reason": ...} while RECOVERING
    questions: list
    timeline: list  # [{"state", "t"}]
    verifications: list
    failure: dict | None
    plan: dict | None


class PickPlaceWorkflow:
    MODES = ("pick_place", "pick", "place")

    def __init__(self, tools, log: EventLog, run_dir: Path, responder, budgets: Budgets | None = None,
                 final_home: bool = True, mode: str = "pick_place") -> None:
        """mode: "pick_place" (validated composite), "pick" (grasp + lift + retention check, ends HOLDING, no HOME),
        "place" (the object already held: resolve the destination only, carry, release, verify, HOME)."""
        assert mode in self.MODES
        self.mode = mode
        """responder: agent.ask.TerminalResponder (a person) or ScriptedResponder (unattended runs).
        Required: there is no silent default answer."""
        self.tools, self.log, self.run_dir, self.responder = tools, log, Path(run_dir), responder
        self.final_home = final_home
        self.budgets = budgets or Budgets()
        self.ctx: dict = {}  # non-serialisable working data: observation, plan
        g = StateGraph(AgentState)
        for s in STATES:
            g.add_node(s, self._wrap(s, getattr(self, "_" + s.lower())))
        for s in TERMINAL:
            g.add_node(s, self._wrap(s, lambda st: {}))
            g.add_edge(s, END)
        g.set_entry_point("OBSERVING")
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
        q = Question(kind, text, role, self._candidates(obs, marks), image, visible=sorted(obs.marks))
        self.log("question", question=text, question_kind=kind, role=role, candidates=[c.mark for c in q.candidates],
                 visible=q.visible, frame_id=obs.frame_id, image=image)
        a = self.responder(q)
        rec = {"question": text, "question_kind": kind, "role": role, "candidates": [c.mark for c in q.candidates],
               "answer": a.kind, "mark": a.mark, "raw": a.raw, "source": a.source, "rejected": a.attempts}
        self.ctx.setdefault("questions", []).append(rec)
        self.log("ask_user", **rec)
        return a

    def _cancelled(self, a: Answer, question: str) -> dict:
        if a.kind == "text":  # the person answered in words: the step ends and the supervisor (Gemma) reads them
            return {"next": "FAILED", "failure": {"cls": "answered", "reason": f"{question} -> the user answered in "
                    f"words: {a.raw!r}", "answer_source": a.source}}
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
    def _observing(self, state) -> dict:
        """Observe from the current pose. The existing clear-view park is a perception-recovery fallback only: it runs
        when the previous localization found no object for the required reference (none matched, or the referenced
        id is not in the view) AND the arm blocks part of the camera's view now (otherwise moving cannot help). A found
        but rejected candidate, or generic workspace occlusion alone, never moves the arm."""
        given = self.ctx.pop("given_obs", None)
        if given is not None and self.tools.is_fresh(given):
            obs = given  # the supervisor's observation: same frame, robot not moved since
            self.log("tool", tool="observation_reused", frame_id=obs.frame_id)
            self.ctx["obs"] = obs
            if not obs.marks:
                return self._fail(state, "perception", "no objects measured on the table")
            return {"next": "LOCALIZING"}
        if self.ctx.pop("localization_failed", False):
            blocked, total = self.tools.view_blockage()
            self.log("tool", tool="clear_view_decision", reason="the required object was not localized",
                     view_blocked=f"{blocked}/{total}", park=blocked > 0)
            if blocked > 0:
                p = self.tools.park_for_view(None)  # existing clear-view park: whole workspace, least-blocking pose
                if not p["ok"]:
                    return self._fail(state, "perception", "no collision-free pose that leaves the table camera's view clear")
        try:
            obs = self.tools.perceive("table")
            self.tools.describe(obs)  # every object (a one-image shortlist was benchmarked and rejected: 4/8 wrong)
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
        """-> (mark, None) or (None, update dict). Several matches -> the user chooses (never the model).
        "#<id>" refers to a world object directly (from the supervisor's state, no Gemma call); a reference the
        user already clarified in this session is reused while that object is still measured."""
        import re as _re
        m = _re.fullmatch(r"#(\d+)", ref["text"].strip())
        rejected = self.ctx.setdefault("rejected", {}).get(ref["text"], set())  # world ids Gemma's crop check rejected
        # identity: a world id already chosen for these words (by id, the user, this skill's earlier attempt, or this
        # task's binding) is kept while the object is measured in the current frame (data association by the world
        # model); marks are only this frame's labels
        bound = self.ctx.setdefault("bound", {}).get(ref["text"])
        wid, why = ((int(m.group(1)), "reference by id") if m else
                    (self.tools.resolved_refs.get(ref["text"].strip().lower()), "clarified by the user earlier"))
        if wid is None and bound is not None:
            wid, why = bound, "chosen earlier in this step, kept by data association"
        if wid is None and ref.get("world_id") is not None:
            wid, why = ref["world_id"], "bound earlier in this task, kept by data association"
        if wid is not None and not m and wid in rejected:
            wid = None
        if wid is not None:
            mark = next((k for k, v in obs.object_ids.items() if v == wid), None)
            if mark is not None:
                self.log("resolution", role=role, text=ref["text"], id=mark, ambiguous=False, candidates=[mark],
                         world_id=wid, frame_id=obs.frame_id, reason=f"world object {wid} ({why}; mark {mark} in {obs.frame_id})")
                self.ctx["bound"][ref["text"]] = wid
                return mark, None
            if m:
                self.ctx["not_found"] = True  # the referenced object is not in this view
                return None, self._fail(state, "perception", f"object {wid} is not measured in the current view", role=role)
            self.log("identity", text=ref["text"], world_id=wid, frame_id=obs.frame_id,
                     reason="not measured in this frame: matching again")
        try:
            # a destination is never the object being put down: the held one (seen lifted in the gripper) or the one
            # this pick_place moves (live 2026-10-26: the held cube was a "red tray" candidate). A physical fact, not
            # a judgement of the words.
            moving = {w for w in (self.tools.held_object, self.ctx.get("moving_id")) if w is not None} if role == "reference" else set()
            exclude = {k for k, v in obs.object_ids.items() if v in rejected or v in moving}
            r = self.tools.resolve(obs, ref["text"], exclude=exclude) if exclude else self.tools.resolve(obs, ref["text"])
        except VLMFailure as exc:
            return None, self._fail(state, "perception", str(exc))
        self.log("resolution", role=role, text=ref["text"], **r.model_dump())
        user_chosen = False
        mark = r.id
        if r.id is None:
            if not r.candidates:
                self.ctx["not_found"] = True  # no measured object matches: it may be hidden from this view
                return None, self._fail(state, "perception", f"nothing matches {ref['text']!r}", role=role)
            question = f"Several objects could be {ref['text']!r}. Which one do you mean?"
            if len(self.ctx.get("questions", [])) >= self.budgets.max_questions:
                return None, {"next": "CANCELLED", "failure": {"cls": "asked", "reason": f"{question} -> question budget "
                                                                f"({self.budgets.max_questions}) used up"}}
            a = self._question("choose", question, role, obs, r.candidates)
            if a.kind != "choice":
                return None, self._cancelled(a, question)
            if a.mark not in obs.marks or obs.object_ids.get(a.mark) is None:  # never a stale or unmeasured number
                return None, self._fail(state, "asked", f"object {a.mark} is not in the current observation", role=role)
            mark, user_chosen = a.mark, True
            if obs.object_ids.get(mark) is not None:  # remember the clarification for retries / later steps
                self.tools.resolved_refs[ref["text"].strip().lower()] = obs.object_ids[mark]
        anchor_region = layout.region(ref.get("anchor")) if ref.get("relation") else None
        if anchor_region is not None:  # "in" a fixed region: the measured footprint against its calibrated outline
            g = obs.measured[mark]
            ok = ref["relation"] == "in" and self.tools.in_region(g, anchor_region)["inside"]
            self.log("location_check", text=ref["text"], mark=mark, relation=ref["relation"], anchor=anchor_region.id,
                     anchor_mark=None, holds=ok)
            if not ok:
                return None, {"next": "FAILED", "failure": {"cls": "perception", "reason": f"could not verify that "
                              f"{ref['text']!r} is {ref['relation']} {anchor_region.id} ({anchor_region.name}) from the "
                              f"measured position"}}
        elif ref.get("relation") and ref.get("anchor"):
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
            self.ctx["bound"][ref["text"]] = obs.object_ids[mark]
            return mark, None
        # the crop shows what the object looks like, not where it is or what it relates to: it is checked against
        # Gemma's appearance words only (live 2026-11-04: "... at x +429, y +496 mm" said "no" to the right object)
        words = ref["text"]
        try:
            spec = self.tools.reference_spec(ref["text"]) if hasattr(self.tools, "reference_spec") else None
            words = (spec.appearance or "").strip() or ref["text"] if spec is not None else ref["text"]
        except VLMFailure:
            pass
        try:
            c = self.tools.confirm(obs, mark, words)
        except VLMFailure as exc:
            return None, self._fail(state, "perception", str(exc))
        self.log("confirmation", role=role, mark=mark, text=ref["text"], checked_words=words, **c.model_dump())
        if c.match == "no":  # not this one: later attempts in this task leave it out (by world id, stable across frames)
            if obs.object_ids.get(mark) is not None:
                self.ctx.setdefault("rejected", {}).setdefault(ref["text"], set()).add(obs.object_ids[mark])
            return None, self._fail(state, "perception", f"confirm_object said no for {ref['text']!r} (mark {mark})", role=role)
        if c.match == "unsure":
            question = f"Is object {mark} the {ref['text']!r}? I am not sure."
            others = [k for k in r.candidates if k != mark]  # Gemma's own other candidates, if it gave any
            a = self._question("confirm", question, role, obs, [mark] + others)
            if a.kind == "choice":  # the asked object's number confirms it; another visible number chooses that one
                if a.mark not in obs.marks or obs.object_ids.get(a.mark) is None:
                    return None, self._fail(state, "asked", f"object {a.mark} is not in the current observation", role=role)
                self.ctx["bound"][ref["text"]] = obs.object_ids[a.mark]
                return a.mark, None
            return None, self._cancelled(a, question)
        if obs.object_ids.get(mark) is not None:
            self.ctx["bound"][ref["text"]] = obs.object_ids[mark]
        return mark, None

    def _localizing(self, state) -> dict:
        self.ctx["not_found"] = False
        upd = self._localize(state)
        if self.ctx.pop("not_found", False) and (upd.get("recover") or upd.get("failure") or {}).get("cls") == "perception":
            self.ctx["localization_failed"] = True  # not in view: the next observation may use the clear-view park
        return upd  # a candidate that was found but rejected (crop check "no") is not a visibility problem

    def _localize(self, state) -> dict:
        obs, task = self.ctx["obs"], state["task"]
        region = layout.region((task.get("reference") or {}).get("text")) if self.mode != "pick" else None
        if self.mode == "place" and region is not None:  # a fixed region: its calibrated geometry, no matching
            self._log_region(obs, None, self.tools.held_object, region, self.tools.held_geom)
            return {"next": "PLANNING_GRASP", "object_id": self.tools.held_object, "reference_id": None,
                    "reference_region": region.id}
        if self.mode == "place":  # the object is the held one; only the destination is resolved
            m_ref, upd = self._resolve_one(state, obs, task["reference"], "reference")
            if upd:
                return upd
            rid = obs.object_ids[m_ref]
            self.log("localized", frame_id=obs.frame_id, object_mark=None, object_id=self.tools.held_object,
                     reference_mark=m_ref, reference_id=rid, object=self.tools.held_geom,
                     reference=self.tools.world.objects[rid].geometry)
            return {"next": "PLANNING_GRASP", "object_id": self.tools.held_object, "reference_id": rid}
        m_obj, upd = self._resolve_one(state, obs, task["object"], "object")
        if upd:
            return upd
        self.ctx["moving_id"] = obs.object_ids.get(m_obj)  # never this step's destination
        if self.mode == "pick":
            oid = obs.object_ids[m_obj]
            self.log("localized", frame_id=obs.frame_id, object_mark=m_obj, object_id=oid, reference_mark=None,
                     reference_id=None, object=self.tools.world.objects[oid].geometry, reference={})
            return {"next": "PLANNING_GRASP", "object_id": oid, "reference_id": None}
        if region is not None:
            oid = obs.object_ids[m_obj]
            self._log_region(obs, m_obj, oid, region, self.tools.world.objects[oid].geometry)
            return {"next": "PLANNING_GRASP", "object_id": oid, "reference_id": None, "reference_region": region.id}
        m_ref, upd = self._resolve_one(state, obs, task["reference"], "reference")
        if upd:
            return upd
        if m_obj == m_ref:
            return {"next": "FAILED", "failure": {"cls": "understanding", "reason": f"the object and the destination "
                    f"are the same object ({m_obj}); please rephrase"}}
        oid, rid = obs.object_ids[m_obj], obs.object_ids[m_ref]
        self.log("localized", frame_id=obs.frame_id, object_mark=m_obj, object_id=oid, reference_mark=m_ref, reference_id=rid,
                 object=self.tools.world.objects[oid].geometry, reference=self.tools.world.objects[rid].geometry)
        return {"next": "PLANNING_GRASP", "object_id": oid, "reference_id": rid}

    def _log_region(self, obs, mark, oid, region, geom) -> None:
        self.log("localized", frame_id=obs.frame_id, object_mark=mark, object_id=oid, reference_mark=None, reference_id=None,
                 reference_region=region.id, region_name=region.name, object=geom or {},
                 reference={"centre": list(region.centre), "size": list(region.size)})

    def _unreachable_reason(self, state, error: str) -> str | None:
        """For an out-of-reach grasp: one clear statement when no gripper orientation the arm can take picks the
        object up (Tools.grasp_reach, kinematics over the whole joint range); None to keep the planner's own message
        (e.g. the object is reachable and the destination is not)."""
        oid = state.get("object_id")
        if "out of reach" not in error or oid is None or self.mode == "place" or not hasattr(self.tools, "grasp_reach"):
            return None
        r = self.tools.grasp_reach(oid)
        if r["from_above"]:
            return None
        b = r["bearing_deg"]  # robot frame: 0 = straight ahead, + = to the right
        side = "ahead" if abs(b) <= 45 else "behind the base" if abs(b) >= 135 else ("to the right" if b > 0 else "to the left")
        marks = {w: m for m, w in (getattr(self.ctx.get("obs"), "object_ids", None) or {}).items()}
        shown = f" (number {marks[oid]} in the camera picture)" if oid in marks else ""
        where = (f"object #{oid}{shown} at x {r['centre'][0] * 1000:+.0f}, y {r['centre'][1] * 1000:+.0f} mm "
                 f"({r['distance_m']:.2f} m from the base, {side})")
        if not r["any_orientation"]:
            return f"No possible gripper orientation can pick up {where}: it is outside the arm's range."
        from control.skills import SkillParams
        limit = SkillParams().max_tilt_deg
        if r["tilt_deg"] is not None and r["tilt_deg"] > limit:
            return (f"No possible gripper orientation can pick up {where}: the gripper reaches it only tilted "
                    f"{r['tilt_deg']} deg from vertical, beyond the {limit:.0f} deg limit for tilted grasps.")
        return (f"Neither a top-down nor a tilted grasp (up to {limit:.0f} deg) can pick up {where}: the hand reaches that "
                f"point only tilted {r['tilt_deg']} deg or more, and no grasp with the pads on two of its faces passed "
                "the reach and collision checks of the approach, grasp and lift.")

    def _planning_grasp(self, state) -> dict:
        task = state["task"]
        try:
            q = self.tools.bridge.q()
            region = {"region": state["reference_region"]} if state.get("reference_region") else {}  # a fixed region
            if self.mode == "pick":
                plan = self.tools.plan_pick(state["object_id"], q)
            elif self.mode == "place":
                plan = self.tools.plan_place_held(task["relation"], state["reference_id"], task.get("frame", "robot"), q, **region)
            else:
                plan = self.tools.plan_pick_place(state["object_id"], task["relation"], state["reference_id"],
                                                  task.get("frame", "robot"), q, **region)
        except PlanningError as exc:
            if self.tools.deterministic(str(exc)):  # re-observing cannot change it: no retry, report it as such
                reason = self._unreachable_reason(state, str(exc)) or str(exc)
                self.log("failure", cls="infeasible", reason=reason, used=0, budget=0)
                return {"next": "FAILED", "failure": {"cls": "infeasible", "reason": reason}}
            return self._fail(state, "planning", str(exc))
        self.ctx["plan"] = plan
        mm = lambda v: round(float(v) * 1000, 1)  # noqa: E731  plain floats: the summary is checkpointed (msgpack)
        summary = {"grasp": {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                             for k, v in plan.grasp.items()},
                   "target_centre_mm": [mm(v) for v in plan.target_centre], "target_top_mm": mm(plan.target_top),
                   "relation": plan.relation,
                   "held": {"height_mm": mm(plan.held.height), "grip_depth_mm": mm(plan.held.grip_depth),
                            "width_mm": mm(plan.held.width), "length_mm": mm(plan.held.length)}}
        self.log("plan", **summary, mode=self.mode)
        return {"next": "TRANSPORTING" if self.mode == "place" else "APPROACHING", "plan": summary}

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
        borderline = v.get("stopped_short_deg", 0.0) >= self.tools.tol.empty_deg if hasattr(self.tools, "tol") else False
        if not v["pass"] and not borderline:
            out = self._fail(state, "grasp", "the fingers closed without stalling on the object (no contact)")
            return {**out, "verifications": vs}
        if not v["pass"]:  # between "clearly stalled" and "closed on nothing" (thin objects): the check after the lift,
            # with the wrist camera looking between the fingers, decides (live 2026-10-26: 1.26 deg, the block was held)
            self.log("verification", check="contact_borderline", stopped_short_deg=v.get("stopped_short_deg"),
                     decision="lift, then the retention check (fingers + wrist camera) decides")
        self.tools.set_held(p, p.approach[1].position)
        return {"next": "TRANSPORTING", "verifications": vs}

    def _transporting(self, state) -> dict:
        p = self.ctx["plan"]
        if self.mode == "pick":  # lift only, then check retention: the goal state is "holding"
            out = self._run(p.lift, "lift", state, "COMPLETED")
            if out["next"] != "COMPLETED":
                return out
            v = self._held(p, "retention_after_lift")
            vs = list(state.get("verifications", [])) + [v]
            if not v["pass"]:
                self.tools.unconfirmed_grasp = (p, self.tools.held_close_target)  # re-checked at the next instruction
                self.tools.clear_held()
                return {**self._fail(state, "execution", "object lost while lifting (fingers no longer stalled)"), "verifications": vs}
            self.tools.last_referent = p.object_id
            return {**out, "verifications": vs}
        steps = p.place[:1] if self.mode == "place" else p.lift + p.place[:1]
        out = self._run(steps, "carry" if self.mode == "place" else "lift and carry", state, "PLACING")
        if out["next"] != "PLACING":
            return out
        v = self._held(p, "contact_after_carry")  # still holding after the carry?
        vs = list(state.get("verifications", [])) + [v]
        if not v["pass"]:
            return {**self._fail(state, "execution", "object lost during transport (fingers no longer stalled)"), "verifications": vs}
        return {**out, "verifications": vs}

    def _held(self, p, check: str) -> dict:
        """Retention: fingers, and the table camera when the fingers are borderline (Tools.confirm_held)."""
        if hasattr(self.tools, "confirm_held"):
            return self.tools.confirm_held(self.tools.held_close_target, p.object_id, check)
        v = self.tools.verify_contact(self.tools.held_close_target)
        v["check"] = check
        return v

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
        if v["pass"]:  # HOME once per instruction: only after the last manipulation step
            self.tools.clear_held()
            self.tools.last_referent = p.object_id
            return {"next": "RETURNING_HOME" if self.final_home else "COMPLETED", "verifications": vs}
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
    def run(self, instruction: str, task: dict, observation=None, thread_id: str = "run") -> dict:
        """task: the planner's pick-and-place step {"object": {...}, "relation", "reference": {...}, "frame"}."""
        t0 = time.monotonic()
        self.log("skill", skill="pick_place", event="start", task=task, final_home=self.final_home)
        if observation is not None:
            self.ctx["given_obs"] = observation
        init: AgentState = {"instruction": instruction, "task": task, "retries": {}, "questions": [], "timeline": [],
                            "verifications": [], "failure": None, "plan": None}
        final = self.graph.invoke(init, {"configurable": {"thread_id": thread_id}, "recursion_limit": 80})
        status = final["timeline"][-1]["state"]
        report = {"instruction": instruction, "status": status, "seconds": round(time.monotonic() - t0, 1),
                  "task": final.get("task"), "object_id": final.get("object_id"), "reference_id": final.get("reference_id"),
                  "reference_region": final.get("reference_region"),
                  "plan": final.get("plan"), "verifications": final.get("verifications"), "failure": final.get("failure"),
                  "questions": final.get("questions"), "retries": final.get("retries"), "timeline": final["timeline"],
                  "home": self.final_home}
        report["message"] = honest_message(report)
        self.log("skill", skill="pick_place", event="end", status=status, message=report["message"])
        return report


def honest_message(r: dict) -> str:
    s = r["status"]
    vs = r.get("verifications") or []
    place = [x for x in vs if x.get("check") == "placement" and x.get("pass")]
    home = [x for x in vs if x.get("check") == "home"]
    placed = (f"placed {place[-1]['xy_error_mm']} mm from the planned spot, top {place[-1]['z_error_mm']:+} mm, "
              f"tilt {place[-1]['tilt_deg']} deg") if place else ""
    if s == "COMPLETED":
        if not place:  # a pick: nothing was placed, the grasp is what was checked
            return f"Done: holding object #{r.get('object_id')} (finger contact confirmed after the lift)."
        back = f"; back at HOME (max joint error {home[-1]['max_error_deg']} deg)" if home else ""
        return f"Done (verified by the table camera): {placed}{back}."
    if s == "CANCELLED":
        return f"Cancelled: {r['failure']['reason']}."
    f = r.get("failure") or {}
    pre = f"Placement verified ({placed}), but " if place else ""
    return f"{pre}Failed ({f.get('cls', '?')}): {f.get('reason', 'unknown')}. I did not report success."
