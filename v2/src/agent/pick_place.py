"""Measured one-object transfer, with no language-model collision overrides.

PREPARING (empty-gripper HOME) -> OBSERVING -> LOCALIZING -> PLANNING_GRASP ->
APPROACHING (hover only) -> INSPECTING_GRASP (wrist RGB-D) -> DESCENDING ->
GRASPING -> VERIFYING_GRASP -> TRANSPORTING (lift + retention) -> CARRYING ->
PLACING -> RELEASING -> RETREATING -> VERIFYING_PLACEMENT -> RETURNING_HOME -> VERIFYING_HOME.

VERIFYING_PLACEMENT: the wrist camera looks down at the released object from a collision-checked viewpoint and
Gemma judges the picture in words (judge_placement, set by the manipulation worker); unsure -> one look from
farther. Without a judge (offline tests) the table camera re-measures the placement instead (verify_placement).

The full approach, lift, carrying waypoint, placement and empty-hand return are
planned before pickup and replanned after wrist refinement. A missing/ambiguous
wrist view stops before descent. Obstacles are preserved; collision failure is
returned to task orchestration. Pick-only ends holding after lift verification;
place-only starts with the recorded held object. All motion uses Tools.execute.

A top-view source identity rejection gets one collision-checked side view of the SAME source.
On acceptance, return to hover and refresh top RGB-D geometry before planning/descent. An unresolved
side check aborts without grasping and attempts a collision-checked, sensor-verified HOME return.
"""

from __future__ import annotations

import time
from agent.timing import span, skill_state
from agent.performance import options
from agent.contracts import sensor_epoch

from config.agent import Budgets
from control.motion import PlanningError
from control.robot import GRIPPER_OPEN
from control.skills import GripperStep, SkillParams


class PickPlace:
    def __init__(self, tools, log, budgets: Budgets | None = None) -> None:
        self.tools, self.log, self.budgets = tools, log, budgets or Budgets()
        self.judge_source = None  # worker-owned bounded semantic fallback; tools retain the geometric gates

    def run(self, mode: str, object_id: int | None, relation: str | None = None, reference_id: int | None = None,
            final_home: bool = True, observation=None, region: str | None = None) -> dict:
        assert mode in ("pick", "place", "pick_place")
        self.mode, self.final_home, self.given_obs = mode, final_home, observation
        self.object_id, self.relation, self.reference_id, self.region = object_id, relation, reference_id, region
        self.retries, self.verifications, self.failure, self.timeline = {}, [], None, []
        self.obs, self.plan, self.localization_failed = None, None, False
        self.source_side_used, self.source_identity = False, None
        self.started_at = time.monotonic()
        self.effects = {"grasp_attempted": False, "released": False, "holding_verified": False,
                        "placement_verified": False}
        self.log("skill", event="start", mode=mode, object_id=object_id, relation=relation, reference_id=reference_id,
                 region=region)
        return self._drive("PREPARING")

    def _drive(self, state):
        while state not in ("COMPLETED", "FAILED"):
            self.last_stage = state
            self.log("state", state=state)
            ts = time.monotonic()
            token = skill_state.set(state)
            try:
                with span(self.log, "skill", state):
                    nxt = getattr(self, "_" + state.lower())()
            finally:
                skill_state.reset(token)
            self.timeline.append({"state": state, "seconds": round(time.monotonic() - ts, 2)})
            state = nxt
        report = {"status": state, "mode": self.mode, "object_id": self.object_id, "reference_id": self.reference_id,
                  "relation": self.relation, "seconds": round(time.monotonic() - self.started_at, 1), "verifications": self.verifications,
                  "failure": self.failure, "retries": self.retries, "timeline": self.timeline,
                  "stage": self.last_stage, "effects": {**self.effects, "held_object": self.tools.held_object}}
        report["message"] = message(report)
        self.log("skill", event="end", status=state, message=report["message"])
        return report

    def resume(self, state):
        """Only the measured recovery handler selects this stage, never a model-generated command."""
        self.failure = None
        return self._drive(state)

    def _effect(self, name):
        self.log("physical_effect", effect=name, object_id=self.object_id)
        observer = getattr(self, "on_effect", None)
        if observer is not None:
            observer(name, self.object_id)

    # ------------------------------------------------------------------ helpers
    def _fail(self, cls: str, reason: str) -> str:
        used, budget = self.retries.get(cls, 0), self.budgets.retries.get(cls, 0)
        self.log("failure", cls=cls, reason=reason, used=used, budget=budget)
        if used < budget:
            self.retries[cls] = used + 1
            self.recover = {"cls": cls, "reason": reason}
            return "RECOVERING"
        self.failure = {"cls": cls, "reason": reason}
        return "FAILED"

    def _stop(self, cls: str, reason: str) -> str:  # no retry can change it
        self.log("failure", cls=cls, reason=reason, used=0, budget=0)
        self.failure = {"cls": cls, "reason": reason}
        return "FAILED"

    def _run(self, steps, label: str, next_state: str) -> str:
        res = self.tools.execute(steps, label)
        if not res.ok:
            return self._fail("execution", f"{label}: {res.failed_step}: {res.reason}")
        return next_state

    def _held(self, check: str) -> dict:
        v = self.tools.confirm_held(self.tools.held_close_target, self.plan.object_id, check)
        self.verifications.append(v)
        return v

    # ------------------------------------------------------------------- states
    def _preparing(self) -> str:
        if self.mode == "place":
            return "OBSERVING"
        result = self.tools.prepare_pick()
        return "OBSERVING" if result["ok"] else self._stop("home", result.get("reason", "initial HOME unavailable"))

    def _observing(self) -> str:
        t = self.tools
        if self.given_obs is not None and t.is_fresh(self.given_obs):
            self.obs, self.given_obs = self.given_obs, None
            return "LOCALIZING"
        self.given_obs = None
        if self.localization_failed:  # the object was not in view: park the arm out of the camera's view if it blocks it
            self.localization_failed = False
            blocked, _ = t.view_blockage()
            if blocked > 0 and not t.park_for_view(None)["ok"]:
                return self._fail("perception", "no collision-free pose that leaves the table camera's view clear")
        self.obs = t.perceive("table")
        t.update_world(self.obs)
        if not self.obs.marks:
            return self._fail("perception", "no objects measured on the table")
        return "LOCALIZING"

    def _localizing(self) -> str:
        t, seen = self.tools, set(self.obs.object_ids.values())
        if self.mode == "place":
            self.object_id = t.held_object
        elif self.object_id not in seen:
            self.localization_failed = True
            return self._fail("perception", f"the source (object #{self.object_id}) is not measured in the current view")
        if self.mode != "pick" and self.region is None:
            if self.reference_id not in seen:
                self.localization_failed = True
                return self._fail("perception", f"the destination (object #{self.reference_id}) is not measured in the current view")
            if self.reference_id == self.object_id:
                return self._stop("understanding", f"the object and the destination are the same object (#{self.object_id})")
        self.log("localized", frame_id=self.obs.frame_id, object_id=self.object_id, reference_id=self.reference_id)
        return "PLANNING_GRASP"

    def _planning_grasp(self) -> str:
        t = self.tools
        try:
            self.plan = self._plan(t)
        except PlanningError as exc:
            if "collision" in str(exc):
                return self._stop("blocked", str(exc))
            if t.deterministic(str(exc)):
                return self._stop("infeasible", self._unreachable(str(exc)) or str(exc))
            return self._fail("planning", str(exc))
        p = self.plan
        self.log("plan", mode=self.mode, grasp=p.grasp, relation=p.relation,
                 target_centre_mm=[round(float(v) * 1000, 1) for v in p.target_centre],
                 return_home_preplanned=bool(p.home))
        return "TRANSPORTING" if self.mode == "place" else "APPROACHING"

    def _plan(self, t):
        q = t.bridge.q()
        place = {"region": self.region} if self.region else {}  # a calibrated place instead of an object
        if self.mode == "pick":
            return t.plan_pick(self.object_id, q)
        if self.mode == "place":
            return t.plan_place_held(self.relation, self.reference_id, "robot", q, **place)
        return t.plan_pick_place(self.object_id, self.relation, self.reference_id, "robot", q, **place)

    def _approaching(self) -> str:
        return self._run(self.plan.hover, "pre-grasp inspection pose", "INSPECTING_GRASP")

    def _inspecting_grasp(self) -> str:
        begin = getattr(self.judge_source, "begin_inspection", None)
        if begin is not None:
            begin()
        evidence = self.tools.inspect_grasp(self.object_id)
        self.verifications.append({"check": "wrist_grasp", "pass": evidence["ok"], **evidence})
        if not evidence["ok"]:
            if evidence.get("identity_mismatch"):
                if self.source_side_used:
                    return self._abort_source_identity("source identity remains unconfirmed after the side-view budget")
                self.inspection_q = self.tools.bridge.q().copy()
                self.inspection_epoch = sensor_epoch(self.tools)
                return "INSPECTING_SOURCE_SIDE"
            return self._stop("inspection", evidence.get("reason", "wrist inspection failed"))
        return self._replan_after_inspection()

    def _inspecting_source_side(self):
        self.source_side_used = True
        try:
            kwargs = {"judge_identity": self.judge_source} if self.judge_source is not None else {}
            evidence = self.tools.inspect_source_side(self.object_id, **kwargs)
        except Exception as exc:
            evidence = {"ok": False, "reason": f"side inspection unavailable: {exc}"}
        self.verifications.append({"check": "source_side_identity", "pass": evidence["ok"],
                                   **{k: v for k, v in evidence.items() if k != "identity_evidence"}})
        if not evidence["ok"]:
            return self._abort_source_identity(evidence.get("reason", "side-view source identity unconfirmed"))
        self.source_identity = evidence["identity_evidence"]
        return "RETURNING_TO_INSPECTION"

    def _returning_to_inspection(self):
        try:
            result = self.tools.return_to_source_inspection(self.inspection_q)
        except Exception as exc:
            result = {"ok": False, "reason": str(exc)}
        if not result["ok"]:
            return self._abort_source_identity("cannot return to the grasp inspection pose: " + result["reason"])
        return "RECHECKING_GRASP_GEOMETRY"

    def _rechecking_grasp_geometry(self):
        try:
            evidence = self.tools.inspect_grasp(self.object_id, identity_evidence=self.source_identity)
        except Exception as exc:
            evidence = {"ok": False, "reason": f"fresh top geometry unavailable: {exc}"}
        self.verifications.append({"check": "wrist_grasp_after_side", "pass": evidence["ok"], **evidence})
        if not evidence["ok"]:
            return self._abort_source_identity("top-view grasp geometry unconfirmed after side inspection: "
                                               + evidence.get("reason", "inspection failed"))
        return self._replan_after_inspection()

    def _abort_source_identity(self, reason):
        self.failure = {"cls": "source_identity", "reason": reason}
        self.log("failure", **self.failure, used=0, budget=0)
        return "ABORTING_SOURCE_HOME"

    def _aborting_source_home(self):
        # Failure remains failure even if HOME succeeds. Never open/close on an unconfirmed source.
        if sensor_epoch(self.tools) != self.inspection_epoch:
            self.failure = {"cls": "sensor_epoch", "reason": "simulator reset during source inspection; "
                            "HOME not attempted with invalid object geometry; restart the session"}
            return "FAILED"
        try:
            result = self.tools.return_home()
            verification = self.tools.verify_home() if result["ok"] else None
            if verification is not None:
                self.verifications.append(verification)
            home_ok = bool(verification and verification["pass"])
            reason = result.get("reason") or "HOME joint check failed"
        except Exception as exc:
            home_ok, reason = False, str(exc)
        self.failure["reason"] += "; returned HOME (joint feedback passed)" if home_ok else "; HOME not verified: " + reason
        self.log("source_inspection_abort", object_id=self.object_id, home_verified=home_ok,
                 reason=self.failure["reason"])
        return "FAILED"

    def _replan_after_inspection(self):
        # Recompute the entire remaining journey using the refined wrist geometry,
        # actual joint feedback and local depth obstacles before any descent.
        try:
            self.plan = self._plan(self.tools)
        except PlanningError as exc:
            return self._stop("blocked" if "collision" in str(exc) else "planning", str(exc))
        return "DESCENDING"

    def _descending(self) -> str:
        return self._run(self.plan.hover + self.plan.approach[:1], "refined approach", "GRASPING")

    def _grasping(self) -> str:
        self.effects["grasp_attempted"] = True
        return self._run(self.plan.approach[1:], "close", "VERIFYING_GRASP")

    def _verifying_grasp(self) -> str:
        t, p = self.tools, self.plan
        v = t.verify_contact(p.approach[1].position)
        self.verifications.append(v)
        if not v["pass"] and v.get("stopped_short_deg", 0.0) < t.tol.empty_deg:
            return self._fail("grasp", "the fingers closed without stalling on the object (no contact)")
        # a borderline contact (thin objects) is decided by the retention check after the lift
        t.set_held(p, p.approach[1].position)
        return "TRANSPORTING"

    def _transporting(self) -> str:
        t, p = self.tools, self.plan
        if self.mode != "place":
            nxt = self._run(p.lift, "lift", "VERIFYING_RETENTION")
            if nxt != "VERIFYING_RETENTION":
                return nxt
        if not self._held("retention_before_carry")["pass"]:
            t.unconfirmed_grasp = (p, t.held_close_target)
            t.clear_held()
            return self._stop("retention", "grasp retention failed before transport; recheck held state and source pose before retrying")
        if self.mode == "pick":
            self.effects["holding_verified"] = True
            self._effect("holding_verified")
            t.last_referent = p.object_id
            return "COMPLETED"
        self.effects["holding_verified"] = True
        self._effect("holding_verified")
        return "CARRYING"

    def _carrying(self) -> str:
        nxt = self._run(self.plan.place[:1], "carry via validated waypoint", "PLACING")
        if nxt != "PLACING":
            return nxt
        if not self._held("contact_after_carry")["pass"]:
            self.tools.unconfirmed_grasp = (self.plan, self.tools.held_close_target)
            self.tools.clear_held()
            return self._stop("retention", "grasp retention failed during transport; object pose must be remeasured")
        if self.reference_id is not None:  # above the destination: is it what the catalogue calls it?
            identity = self.tools.inspect_destination(self.reference_id)
            if identity is not None and not identity["ok"]:
                return self._stop("identity", f"up close the destination #{self.reference_id} looks like a "
                                              f"{identity['top']}, not the {identity['expected']}")
        return "PLACING"

    def _placing(self) -> str:
        return self._run(self.plan.place[1:2], "lower", "RELEASING")

    def _releasing(self) -> str:
        nxt = self._run(self.plan.place[2:-1], "release", "RETREATING")
        if nxt == "RETREATING":
            self.effects.update(released=True, holding_verified=False)
            self.tools.assume_placed(self.plan)
            self._effect("released")
        return nxt

    def _retreating(self) -> str:
        return self._run(self.plan.place[-1:], "retreat", "VERIFYING_PLACEMENT")

    def _verifying_placement(self) -> str:
        t, p = self.tools, self.plan
        judge = getattr(self, "judge_placement", None)  # Gemma on the wrist picture (set by the manipulation worker)
        if judge is not None:
            return self._placement_by_image(judge)
        if not t.park_for_view(p.target_centre[:2])["ok"]:
            return self._fail("placement", "no pose with a clear camera view of the placed object")
        v = t.verify_placement(p)
        self.verifications.append(v)
        if not v["pass"]:
            return self._fail("placement", v.get("reason") or "placement outside tolerance (xy 10 mm, z 5 mm, tilt 5 deg)")
        self.effects["placement_verified"] = True
        self._effect("placement_verified")
        t.clear_held()
        t.last_referent = p.object_id
        return "RETURNING_HOME" if self.final_home else "COMPLETED"

    def _placement_by_image(self, judge) -> str:
        """The wrist camera looks down at the released object and Gemma says, in words, whether it is where the
        request put it. yes -> verified; no -> failed with what Gemma sees; unsure -> one more look from farther."""
        t, p = self.tools, self.plan
        for farther in (False, True):
            view = t.look_at_placement(p, farther=farther)
            if not view["ok"]:
                if farther:
                    break
                continue
            view.update(object_id=p.object_id, reference_id=p.reference_id, relation=p.relation)
            # Opt-in overlap: a captured image is immutable; only the already-released,
            # empty-hand, collision-checked HOME motion runs on the executor thread.
            self.home_during_judgement = None
            evidence_revision = getattr(t, "scene_revision", None)
            evidence_epoch = view.get("epoch", sensor_epoch(t))
            if options(t).overlap_home and self.final_home and t.held_object is None:
                from concurrent.futures import ThreadPoolExecutor
                from contextvars import copy_context
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(copy_context().run, t.return_home)
                    word, text = judge(view)
                    self.home_during_judgement = future.result()
            else:
                word, text = judge(view)
            if evidence_epoch != sensor_epoch(t):
                return self._stop("sensor_epoch", "simulator reset after the placement picture; restart the session")
            v = {"check": "placement", "by": "wrist image", "pass": word == "yes", "answer": f"{word}: {text}",
                 "frame_id": view["frame_id"], "image": view["path"], "farther": farther}
            self.verifications.append(v)
            self.log("verification", **v)
            if word == "yes":
                self.placement_evidence = {"view": view, "text": text, "word": word,
                                           "scene_revision": evidence_revision,
                                           "epoch": evidence_epoch}
                self.effects["placement_verified"] = True
                self._effect("placement_verified")
                t.last_referent = p.object_id
                return "RETURNING_HOME" if self.final_home else "COMPLETED"
            if word == "no":
                return self._stop("placement", f"the wrist camera shows it is not placed as asked: {text}")
        last = next((v["answer"] for v in reversed(self.verifications) if v.get("by") == "wrist image"), None)
        return self._stop("placement", "placement unconfirmed: " + (last or f"no wrist view of the placement ({view['reason']})"))

    def _returning_home(self) -> str:
        r = getattr(self, "home_during_judgement", None)
        if r is None:
            r = self.tools.return_home()
        if r["ok"]:
            return "VERIFYING_HOME"
        if r.get("result") is not None:  # it moved and the executor / safety layer reported a failure
            return self._fail("execution", f"return HOME: {r['reason']}")
        return self._stop("home", f"HOME not attempted: {r['reason']}")

    def _verifying_home(self) -> str:
        v = self.tools.verify_home()
        self.verifications.append(v)
        if v["pass"]:
            return "COMPLETED"
        return self._stop("home", f"HOME not reached: max joint error {v['max_error_deg']} deg > {v['tolerance_deg']} deg")

    def _recovering(self) -> str:
        cls = self.recover["cls"]
        self.log("recovery", **self.recover, retries=self.retries)
        if cls == "grasp":  # open, lift back up, look again
            res = self.tools.execute([GripperStep(GRIPPER_OPEN, "open gripper", 0.8)] + self.plan.lift, "grasp recovery")
            return "OBSERVING" if res.ok else self._fail("execution", f"grasp recovery: {res.reason}")
        if cls == "placement":  # look once more
            return "VERIFYING_PLACEMENT"
        if cls in ("perception", "planning"):
            return "OBSERVING"
        self.failure = dict(self.recover)
        return "FAILED"

    def _unreachable(self, error: str) -> str | None:
        if "out of reach" not in error or self.mode == "place":
            return None
        r = self.tools.grasp_reach(self.object_id)
        if r["from_above"]:
            return None
        where = (f"object #{self.object_id} at x {r['centre'][0] * 1000:+.0f}, y {r['centre'][1] * 1000:+.0f} mm "
                 f"({r['distance_m']:.2f} m from the base)")
        if not r["any_orientation"]:
            return f"No gripper orientation can pick up {where}: it is outside the arm's range."
        limit = SkillParams().max_tilt_deg
        return (f"Neither a top-down nor a tilted grasp (up to {limit:.0f} deg) can pick up {where}"
                + (f": the hand reaches it only tilted {r['tilt_deg']} deg" if r["tilt_deg"] is not None else "") + ".")


def message(r: dict) -> str:
    vs = r.get("verifications") or []
    place = [v for v in vs if v.get("check") == "placement" and v.get("pass")]
    home = [v for v in vs if v.get("check") == "home"]
    placed = (f"placed; the wrist camera shows it ({place[-1]['answer'].partition(':')[2].strip()})"
              if place and place[-1].get("by") == "wrist image" else
              "placed; confirmed by depth at the planned spot" if place and place[-1].get("xy_error_mm") is None else
              f"placed {place[-1]['xy_error_mm']} mm from the planned spot, top {place[-1]['z_error_mm']:+} mm, "
              f"tilt {place[-1]['tilt_deg']} deg" if place else "")
    if r["status"] == "COMPLETED":
        if not place:
            return f"holding #{r['object_id']} (finger contact confirmed after the lift)"
        back = f"; back at HOME (max joint error {home[-1]['max_error_deg']} deg)" if home else ""
        return f"{placed}{back}"
    f = r.get("failure") or {}
    return ("placement verified, but " if place else "") + f"{f.get('cls', '?')}: {f.get('reason', 'unknown')}"
