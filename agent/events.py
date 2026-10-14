"""Readable one-line summaries of the agent's real events (events.jsonl), for the terminal and the demo
viewer. Pure Python (no numpy / pydantic): the viewer imports it from the system Python.

Every line is derived from an event the agent actually logged; nothing is inferred or invented. Events
that carry no information for an operator (e.g. each valid describe call) return None.

    line = format_event(event)   # -> Line(wall, component, text, level) or None
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("info", "warn", "error", "success", "question")


@dataclass
class Line:
    wall: str
    component: str
    text: str
    level: str = "info"

    def __str__(self) -> str:
        return f"[{self.wall}] [{self.component}]".ljust(26) + f" {self.text}"


STATE_TEXT = {
    "UNDERSTANDING": "understanding the instruction", "OBSERVING": "observing the table",
    "LOCALIZING": "finding the object and the destination", "PLANNING_GRASP": "planning grasp and placement",
    "APPROACHING": "approaching the object", "GRASPING": "closing the gripper", "VERIFYING_GRASP": "checking finger contact",
    "TRANSPORTING": "lifting and carrying", "PLACING": "lowering and releasing", "RETREATING": "retreating from the object",
    "VERIFYING_PLACEMENT": "checking the placement with the table camera", "RETURNING_HOME": "returning HOME",
    "VERIFYING_HOME": "checking the HOME joint positions", "RECOVERING": "recovering", "COMPLETED": "task complete",
    "FAILED": "task failed", "CANCELLED": "task cancelled", "UNSUPPORTED": "not possible with the current capabilities",
    "INTERPRETING": "planning the goal (capabilities only)", "DISPATCHING": "choosing the next step",
    "OBSERVING_SCENE": "perception: observing the table", "REPORTING_SCENE": "perception: reporting the scene",
    "ANSWERING": "perception: answering from the measured objects", "LOCATING": "perception: finding the described object", "MANIPULATING": "manipulation: pick-and-place skill",
    "RETURNING_SAFE": "manipulation: returning HOME", "INSPECTING": "exploration: close look with the wrist camera",
    "EXPLORING": "exploration: looking into hidden regions", "UNAVAILABLE": "capability not available",
    "VERIFYING_STEP": "verification: checking the step's evidence", "REPLANNING": "planning again after a failure",
    "VERIFYING_GOAL": "verification: checking every step of the goal",
}


def _args(a: dict) -> str:
    if "object" in a:
        return f"{(a.get('object') or {}).get('text')!r} {a.get('relation')} {(a.get('reference') or {}).get('text')!r}"
    return ", ".join(f"{k}={v!r}" for k, v in a.items())


def _mm(v) -> str:
    return "(" + ", ".join(f"{x * 1000:.0f}" for x in v) + ") mm"


def format_event(e: dict) -> Line | None:  # noqa: C901 (a flat dispatch on the event kind)
    k, w = e.get("kind"), e.get("wall", "--:--:--")
    if k == "instruction":
        return Line(w, "SUPERVISOR", f"task received: {e.get('text')!r}")
    if k == "state":
        s = e.get("state")
        lvl = {"COMPLETED": "success", "FAILED": "error", "CANCELLED": "warn", "RECOVERING": "warn",
               "UNSUPPORTED": "warn", "REPLANNING": "warn"}.get(s, "info")
        who = "SUPERVISOR" if e.get("level") == "goal" else "SKILL"
        return Line(w, who, f"state {s}: {STATE_TEXT.get(s, '')}", lvl)
    if k == "goal_plan":
        steps = "; ".join(f"{x['index']}. {x['capability']}" + (f" {_args(x['args'])}" if x.get("args") else "") for x in e.get("steps", []))
        return Line(w, "PLANNER", f"goal: {e.get('goal')} | plan: {steps or '(none)'}" + (f" | {e.get('reason')}" if not e.get("steps") else ""),
                    "info" if e.get("steps") else "warn")
    if k == "step":
        ev = e.get("event")
        if ev == "start":
            return Line(w, "SUPERVISOR", f"step {e.get('index')}: delegate {e.get('capability')} ({e.get('safety')})")
        if ev == "auto_observe":
            return Line(w, "SUPERVISOR", f"step {e.get('index')}: needs a fresh observation first")
        if ev == "end":
            o = e.get("outcome")
            return Line(w, "SUPERVISOR", f"step {e.get('index')} {e.get('capability')}: {o} in {e.get('seconds')} s",
                        "success" if o == "completed" else ("warn" if o in ("cancelled", "ambiguous", "unsupported") else "error"))
    if k == "skill":
        if e.get("event") == "start":
            return Line(w, "MANIPULATION", f"pick-and-place skill started (final HOME: {e.get('final_home')})")
        return None
    if k == "scene_report":
        return Line(w, "PERCEPTION", f"scene report: {e.get('objects')} objects (frame {e.get('frame_id')})")
    if k == "answer":
        return Line(w, "GEMMA", f"answer: {e.get('answer')} (objects {e.get('object_ids')}; unverified)")
    if k == "exploration":
        pl = e.get("plan") or {}
        if pl.get("ok"):
            b = pl["best"]
            return Line(w, "EXPLORATION", f"{e.get('purpose')} {e.get('target')!r}: viewpoint {b['distance_m']} m away "
                                          f"(elev {b['elevation_deg']} deg, az {b['azimuth_deg']} deg, expected {b['expected_gain']}x pixels), "
                                          f"{pl['evaluated']} candidates, {e.get('seconds')} s; path collision-checked")
        return Line(w, "EXPLORATION", f"{e.get('purpose')} {e.get('target')!r}: {pl.get('reason')} {pl.get('rejected')}", "warn")
    if k == "inspection":
        return Line(w, "PERCEPTION", f"wrist view of {e.get('target')!r}: {_mm(e.get('wrist_centre') or [])} "
                                     f"(table {_mm(e.get('table_centre') or [])}); association {(e.get('association') or {}).get('status')}")
    if k == "explore_result":
        return Line(w, "EXPLORATION", f"{e.get('target')!r}: {e.get('viewpoints')} viewpoint(s), {e.get('new_objects')} newly seen object(s), "
                                      + ("found" if e.get("found") else "not found"), "success" if e.get("found") else "warn")
    if k == "subtask":
        return Line(w, "SUPERVISOR", f"information-gathering subtask: {e.get('reason')}", "warn")
    if k == "plan_completed":
        return Line(w, "SUPERVISOR", "plan completed by rule: " + "; ".join(e.get("rules") or []))
    if k == "located":
        return Line(w, "PERCEPTION", f"{e.get('text')!r}: object {e.get('mark')} at {_mm(e.get('centre') or [])}, crop check {e.get('confirm')}")
    if k == "replan":
        return Line(w, "PLANNER", "replanned" if e.get("accepted") else f"replan rejected: {e.get('reason')}", "warn")
    if k == "goal_verified":
        return Line(w, "VERIFY", f"goal: all {e.get('steps')} steps verified (evidence: {', '.join(e.get('evidence') or []) or 'perception'})", "success")
    if k == "task":
        if e.get("task") != "pick_place":
            return Line(w, "GEMMA", f"parsed as {e.get('task')!r}: {e.get('reason', '')}", "warn")
        obj, ref = (e.get("object") or {}).get("text"), (e.get("reference") or {}).get("text")
        return Line(w, "GEMMA", f"parsed: PICK {obj!r} -> PLACE {e.get('relation')} {ref!r} (frame {e.get('frame')})")
    if k == "vlm":
        if e.get("valid"):
            return None  # valid answers are reported by the event that uses them
        return Line(w, "GEMMA", f"{e.get('role')}: invalid answer (attempt {e.get('attempt')}): {e.get('error')}", "warn")
    if k == "tool":
        t = e.get("tool")
        if t == "perceive":
            ex = len(e.get("excluded") or [])
            return Line(w, "PERCEPTION", f"{e.get('camera')} camera, frame {e.get('frame_id')}: YOLO {e.get('detections')} masks, "
                                         f"{e.get('objects')} objects measured" + (f", {ex} excluded (robot/fixtures/too small)" if ex else ""))
        if t == "lexical_check":
            if not e.get("compare"):
                return None
            return Line(w, "GEMMA", f"{e.get('text')!r}: resolver chose {e.get('resolver_id')}, descriptions fit {e.get('lexical')}: comparing")
        if t == "compare":
            ok = e.get("agree")
            return Line(w, "GEMMA", f"compare {e.get('ids')}: visual choice {e.get('id')}, most specific description "
                                    f"{e.get('most_specific')} -> " + ("accepted" if ok else "cues disagree: ask the user"),
                        "info" if ok else "warn")
        if t == "execute_start":
            return Line(w, "EXECUTION", f"{e.get('label')}: " + " -> ".join(e.get("steps") or []))
        if t == "execute":
            if e.get("ok"):
                return Line(w, "EXECUTION", f"{e.get('label')}: done in {e.get('seconds')} s")
            return Line(w, "EXECUTION", f"{e.get('label')}: FAILED at {e.get('failed_step')!r}: {e.get('reason')}", "error")
        if t == "park_for_view" and not e.get("ok"):
            return Line(w, "MANIPULATION", "no collision-free pose with a clear camera view", "error")
        if t == "observation_reused":
            return Line(w, "PERCEPTION", f"reusing observation {e.get('frame_id')} (robot not moved since)")
        if t == "describe_reused":
            return None  # summarised by the "described N objects" line
        if t == "shortlist":
            if not e.get("ok"):
                return Line(w, "GEMMA", f"shortlist failed ({e.get('reason')}): describing every object", "warn")
            lists = "; ".join(f"{k!r}: {v}" for k, v in (e.get("lists") or {}).items())
            extra = f" (empty for {e.get('fallback_all')}: describing every object)" if e.get("fallback_all") else ""
            return Line(w, "GEMMA", f"one call over {e.get('objects')} objects shortlisted {lists} in {e.get('seconds')} s{extra}")
        if t == "return_home":
            if e.get("ok") and e.get("note"):
                return Line(w, "SAFETY", f"HOME: {e.get('note')}; no further motion")
            if e.get("ok"):
                return Line(w, "SAFETY", f"HOME path collision-checked: free ({e.get('planned_s')} s planned)")
            return Line(w, "SAFETY", f"HOME not attempted: {e.get('reason')}", "error")
        return None
    if k == "world":
        if e.get("event") == "disturbed":
            return Line(w, "PERCEPTION", f"objects moved since the last look: {[m['id'] for m in e.get('moved', [])]}", "warn")
        names = e.get("names") or {}
        return Line(w, "GEMMA", f"described {len(names)} objects: " + ", ".join(f"{k}={v}" for k, v in list(names.items())[:8])
                    + (" ..." if len(names) > 8 else ""))
    if k == "resolution":
        if e.get("id") is None:
            return Line(w, "GEMMA", f"{e.get('role')} {e.get('text')!r}: ambiguous, candidates {e.get('candidates')}", "warn")
        return Line(w, "GEMMA", f"{e.get('role')} {e.get('text')!r} -> object {e.get('id')} ({e.get('reason', '')[:60]})")
    if k == "confirmation":
        m = e.get("match")
        if m == "user_selected":
            return Line(w, "USER", f"{e.get('role')}: object {e.get('mark')} chosen by the user (not re-judged by Gemma)")
        return Line(w, "GEMMA", f"confirm object {e.get('mark')} is {e.get('text')!r}: {m}", "info" if m == "yes" else "warn")
    if k == "question":
        return Line(w, "CLARIFY", f"waiting for the user: {e.get('question')} candidates {e.get('candidates')}", "question")
    if k == "ask_user":
        a = e.get("answer")
        who = "USER" if e.get("source") == "terminal" else "SCRIPT"
        if a == "choice":
            return Line(w, who, f"selected object {e.get('mark')} (typed {e.get('raw')!r})", "question")
        return Line(w, who, f"answer: {a} ({e.get('raw')!r})", "warn")
    if k == "localized":
        o, r = e.get("object") or {}, e.get("reference") or {}
        return Line(w, "GROUNDING", f"target = object {e.get('object_mark')} at {_mm(o.get('centre', []))}, "
                                    f"destination = object {e.get('reference_mark')} at {_mm(r.get('centre', []))}")
    if k == "plan":
        g = e.get("grasp") or {}
        return Line(w, "MANIPULATION", f"grasp yaw {g.get('yaw_deg')} deg, width {g.get('width_mm')} mm, height {g.get('height_mm')} mm; "
                                       f"place at {e.get('target_centre_mm')} mm; every motion collision-checked")
    if k == "verification":
        c = e.get("check")
        ok = e.get("pass")
        lvl = "success" if ok else "error"
        if c in ("contact", "contact_after_carry"):
            what = "grasp" if c == "contact" else "still holding after the carry"
            return Line(w, "VERIFY", f"{what}: fingers stopped {e.get('stopped_short_deg')} deg short (need >= 1.5): "
                                     + ("confirmed" if ok else "NO CONTACT"), lvl)
        if c == "placement":
            if "xy_error_mm" not in e:
                return Line(w, "VERIFY", f"placement: {e.get('reason')}", lvl)
            return Line(w, "VERIFY", f"placement: xy {e.get('xy_error_mm')} mm, z {e.get('z_error_mm')} mm, tilt {e.get('tilt_deg')} deg "
                                     "(limits 10 / 5 / 5): " + ("confirmed" if ok else "OUT OF TOLERANCE"), lvl)
        if c == "viewpoint":
            return Line(w, "VERIFY", f"viewpoint reached: joints {e.get('max_joint_error_deg')} deg, camera {e.get('camera_pos_error_mm')} mm, "
                                     f"axis {e.get('axis_error_deg')} deg", lvl)
        if c == "home":
            return Line(w, "VERIFY", f"HOME: max joint error {e.get('max_error_deg')} deg (limit {e.get('tolerance_deg')}), "
                                     f"TCP {e.get('tcp_error_mm')} mm: " + ("reached" if ok else "NOT REACHED"), lvl)
    if k == "failure":
        return Line(w, "SUPERVISOR", f"{e.get('cls')} failure: {e.get('reason')} (retries used {e.get('used')}/{e.get('budget')})", "warn")
    if k == "recovery":
        return Line(w, "SUPERVISOR", f"recovering from {e.get('cls')} (retries {e.get('retries')})", "warn")
    if k == "report":
        lvl = {"COMPLETED": "success", "CANCELLED": "warn", "UNSUPPORTED": "warn"}.get(e.get("status"), "error")
        return Line(w, "SUPERVISOR", f"{e.get('status')}: {e.get('message')}", lvl)
    if k == "error":
        return Line(w, "SUPERVISOR", f"internal error: {e.get('error')}", "error")
    return None
