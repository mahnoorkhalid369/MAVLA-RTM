"""Capabilities: the only things the planner can ask for, and how each one runs.

The planner (Gemma) composes these into a plan; it cannot name anything else (the plan schema is an enum of
the names below), cannot pass code, and never produces joint targets. Each capability declares its inputs,
preconditions, safety class, effects, failure conditions and verification. Executors call the existing,
validated tools: motions still go through SkillPlanner + CollisionChecker + RosExecutor + the safety layer.

Safety classes: "read_only" (cameras / models only, no robot command), "motion" (robot commands, each one
collision-checked and verified as before).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from agent.vlm_roles import VLMFailure


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    summary: str  # shown to the planner
    args: str  # JSON shape shown to the planner
    safety: str  # read_only | motion
    preconditions: str
    effects: str
    failures: str
    verification: str
    implemented: bool = True
    effect: str = "none"  # none | sensing | object_change (agent/authority.py): what running it may change


@dataclass
class StepResult:
    outcome: str  # agent.world_state.OUTCOMES
    summary: str
    data: dict = field(default_factory=dict)


_SPECS = [
    CapabilitySpec("observe", "look at the table now and measure every object", '{"do": "observe"}',
                   "read_only", "cameras streaming", "fresh observation in the task state (facts + Gemma descriptions)",
                   "no frame; no objects measured", "at least one object measured, or reported as none"),
    CapabilitySpec("describe_scene", "tell the user which objects are on the table and where", '{"do": "describe_scene"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "a grounded scene report",
                   "no fresh observation", "the report lists only measured objects; names marked as Gemma's"),
    CapabilitySpec("answer", "answer a question about the objects on the table", '{"do": "answer", "question": "<the question>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "an answer (a hypothesis)",
                   "no observation; invalid model answer", "refers only to measured objects; marked unverified"),
    CapabilitySpec("locate", "find one described object on the table and say where it is", '{"do": "locate", "object": "<words>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "object id and measured position",
                   "not visible / not confirmed / ambiguous", "same resolution, cross-check and confirmation as pick_place"),
    CapabilitySpec("pick", "grasp ONE object and hold it (no destination)", '{"do": "pick", "object": "<words>"}', "motion",
                   "gripper empty; object resolvable and graspable", "the object is held (lifted)",
                   "not found / ambiguous / too wide / out of reach / no contact / lost while lifting",
                   "finger contact at the grasp and retention after the lift (finger stall)"),
    CapabilitySpec("place", "put the object the robot is HOLDING onto or next to another object",
                   '{"do": "place", "object": "it", "relation": "on" | "next_to" | "left_of" | "right_of" | "in_front_of" | '
                   '"behind", "target": "<destination object>"}', "motion",
                   "an object is held; destination resolvable", "the held object stands at the destination; arm at HOME",
                   "nothing held / destination not found / no free reachable spot / collision / placement out of tolerance",
                   "contact during the carry, camera placement check, HOME joints"),
    CapabilitySpec("decompose", "plan a multi-object goal over the measured objects", '{"do": "decompose", "goal": "<the goal>"}',
                   "read_only", "a fresh observation (taken automatically)", "concrete steps on #ids + final conditions",
                   "invalid plan / not achievable with the available steps", "references must be measured objects"),
    CapabilitySpec("pick_place", "move ONE object onto or next to another object",
                   '{"do": "pick_place", "object": "<object to move>", "relation": "on" | "next_to" | "left_of" | '
                   '"right_of" | "in_front_of" | "behind", "target": "<destination object>"}',
                   "motion", "gripper empty; object and destination resolvable (asks the user if ambiguous)",
                   "the object stands at the destination; arm at HOME after the last motion step",
                   "not found / ambiguous / out of reach / collision / no contact / placement out of tolerance / safety reject",
                   "finger contact, camera placement check (xy 10 mm, z 5 mm, tilt 5 deg), HOME joints (1.5 deg)"),
    CapabilitySpec("return_home", "move the arm to its safe home pose", '{"do": "return_home"}', "motion",
                   "nothing held; gripper open", "arm at HOME", "holding; path collides; execution failure",
                   "HOME joints within 1.5 deg (joint feedback)"),
    CapabilitySpec("inspect", "move the wrist camera closer to look at one visible object in detail",
                   '{"do": "inspect", "object": "<words>"}', "motion",
                   "object visible and confirmed; nothing held", "close-up position / size / description (wrist camera); arm back at HOME",
                   "object not found / no safe viewpoint / move failed", "viewpoint reached (joint feedback), wrist measurement associated"),
    CapabilitySpec("explore", "look for something the table camera cannot see (e.g. behind another object)",
                   '{"do": "explore", "object": "<words>", "near": "<words for an object it may be behind, or null>"}', "motion",
                   "nothing held", "wrist observations of hidden regions in the world model; arm back at HOME",
                   "no hidden region / no safe viewpoint / not found", "finding is decided by locate, not by moving"),
]
EFFECT = {"pick": "object_change", "place": "object_change", "pick_place": "object_change", "inspect": "sensing",
          "explore": "sensing", "return_home": "sensing"}  # everything else: none (read-only)
SPECS: dict[str, CapabilitySpec] = {s.name: replace(s, effect=EFFECT.get(s.name, "none")) for s in _SPECS}

MOTION = {n for n, s in SPECS.items() if s.safety == "motion"}


def planner_menu() -> str:
    """The capability list as the planner sees it."""
    return "\n".join(f"- {s.name}: {s.summary}. args {s.args}" + ("" if s.implemented else " (not available yet)")
                     for s in SPECS.values())


# ----------------------------------------------------------------------------------------- executors
@dataclass
class Context:
    """What executors may use. One instance per instruction, owned by the supervisor."""
    tools: object
    log: Callable
    run_dir: object
    responder: object
    budgets: object
    state: object  # agent.world_state.TaskState
    obs: object = None  # latest agent.tools.Observation (table camera)
    extra_obs: list = field(default_factory=list)  # wrist observations taken during this instruction


def _failure_outcome(cls: str | None, reason: str) -> str:
    """PickPlaceWorkflow failure class -> supervisor outcome. The supervisor replans only "failed" and
    "insufficient_info"; "infeasible", "unsafe", "cancelled" and "unsupported" end the goal without new motion.
      infeasible  deterministic planning error (Tools.deterministic: out of reach, too wide, no free spot, ...):
                  the same action can never succeed, so replanning it would only cost a Gemma call
      planning    a non-deterministic planning error left after the workflow's own re-observation retries (e.g. a
                  collision with the current layout): recoverable, an alternative action may work -> "failed"
      home        the manipulation ran but HOME was not attempted / not reached: stop; replanning would command
                  new manipulation from an unverified pose -> "unsafe"
      execution   "unsafe" when the safety layer rejected a trajectory, otherwise "failed" (e.g. controller timeout,
                  object lost in transport)
      grasp, placement, perception, understanding -> "failed" (recoverable), except perception that found nothing
      asked       the person cancelled, did not answer, or rejected the candidate -> "cancelled"
    """
    if cls in ("asked",):
        return "cancelled"
    if cls == "infeasible":
        return "infeasible"
    if cls == "home":
        return "unsafe"
    if cls == "execution" and "safety" in (reason or "").lower():
        return "unsafe"
    if cls == "perception" and ("nothing matches" in reason or "no objects" in reason):
        return "insufficient_info"
    if cls == "unsupported":
        return "unsupported"
    return "failed"


def run_observe(ctx: Context, args: dict, clear_view: bool) -> StepResult:
    """Observe. clear_view=True parks the arm out of the camera's view first (only when a motion step follows,
    so the manipulation can reuse this observation); otherwise no robot motion at all."""
    t = ctx.tools
    if clear_view:
        p = t.park_for_view(None)
        if not p["ok"]:
            return StepResult("unsafe", "no collision-free pose that leaves the camera's view clear")
    try:
        obs = t.perceive("table")
        t.describe(obs)
    except (VLMFailure, TimeoutError) as exc:
        return StepResult("failed", f"observation failed: {exc}")
    t.update_world(obs)
    ctx.obs = obs
    ctx.state.record_observation(obs)
    blocked, total = t.view_blockage()
    ctx.state.facts["view_blocked"] = {"blocked_points": blocked, "workspace_points": total}
    if not obs.marks:
        return StepResult("insufficient_info", "no objects measured on the table")
    return StepResult("completed", f"{len(obs.marks)} objects measured (frame {obs.frame_id}); the arm hides "
                                   f"{blocked} of {total} workspace points from the camera")


def scene_report(obs, view_blocked: dict | None) -> str:
    lines = []
    for k in sorted(obs.marks, key=lambda k: (obs.measured[k]["centre"][1], obs.measured[k]["centre"][0])):
        g, d = obs.measured[k], obs.descriptions.get(k) or {}
        name = " ".join(x for x in (d.get("colour", ""), d.get("name", "")) if x).strip() or "unnamed object"
        lines.append(f"  {k:>2}. {name} at x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm "
                     f"({np.hypot(*g['centre'][:2]):.2f} m from the robot), about {g['footprint'][0] * 1000:.0f} x "
                     f"{g['footprint'][1] * 1000:.0f} mm, {g['height'] * 1000:.0f} mm tall"
                     + ("" if g.get("tilt_deg") is not None else " (top hidden: size uncertain)"))
    vb = view_blocked or {}
    tail = (f"\n  The arm hides {vb['blocked_points']} of {vb['workspace_points']} workspace points from the camera; "
            "objects there may be missing.") if vb.get("blocked_points") else ""
    return (f"I see {len(obs.marks)} objects on the table (positions and sizes measured by the table camera's depth; "
            f"names are Gemma's descriptions, not verified):\n" + "\n".join(lines) + tail)


def _ensure_described(ctx: Context) -> None:
    missing = [k for k in ctx.obs.marks if k not in ctx.obs.descriptions]
    if missing:  # e.g. the verification frame after a manipulation (session memory makes this cheap)
        ctx.tools.describe(ctx.obs, missing)
        ctx.state.record_observation(ctx.obs)


def run_describe_scene(ctx: Context, args: dict) -> StepResult:
    _ensure_described(ctx)
    report = scene_report(ctx.obs, ctx.state.facts.get("view_blocked"))
    ctx.log("scene_report", frame_id=ctx.obs.frame_id, objects=len(ctx.obs.marks), text=report)
    return StepResult("completed", report)


def world_facts(ctx: Context) -> tuple[str, list[int]]:
    """Measured objects as '#<world id>: description; position; size' (close-up descriptions when available)."""
    lines, ids = [], []
    marks = {v: k for k, v in ctx.obs.object_ids.items()}
    for oid, o in ctx.tools.world.objects.items():
        if o.status not in ("on_surface", "occluded", "held") or oid not in marks and o.status != "held":
            continue
        g = o.geometry
        d = o.descriptions_by_camera.get("wrist") or o.description or {}
        where = "IN THE GRIPPER" if o.status == "held" else f"x {g['centre'][0] * 1000:+.0f}, y {g['centre'][1] * 1000:+.0f}"
        lines.append(f"#{oid}: {d.get('colour', '')} {d.get('name', '')} ({d.get('shape', '')}); {where}; "
                     f"footprint {g['footprint'][0] * 1000:.0f} x {g['footprint'][1] * 1000:.0f}, height {g['height'] * 1000:.0f}")
        ids.append(oid)
    return "\n".join(lines), ids


def facts_text(obs) -> str:
    out = []
    for k in obs.marks:
        g, d = obs.measured[k], obs.descriptions.get(k) or {}
        out.append(f"{k}: {d.get('colour', '')} {d.get('name', '')} ({d.get('shape', '')}) at x {g['centre'][0] * 1000:+.0f} mm, "
                   f"y {g['centre'][1] * 1000:+.0f} mm, top {g['top_z'] * 1000:.0f} mm, footprint {g['footprint'][0] * 1000:.0f} x "
                   f"{g['footprint'][1] * 1000:.0f} mm")
    return "\n".join(out)


def run_answer(ctx: Context, args: dict) -> StepResult:
    q = (args or {}).get("question", "").strip() or ctx.state.instruction
    _ensure_described(ctx)
    try:
        a = ctx.tools.roles.answer(q, facts_text(ctx.obs), list(ctx.obs.marks))
    except VLMFailure as exc:
        return StepResult("failed", f"could not answer: {exc}")
    ctx.state.hypotheses.setdefault("answers", []).append({"question": q, **a.model_dump(), "source": "gemma"})
    ctx.log("answer", question=q, **a.model_dump())
    refs = f" (objects {a.object_ids})" if a.object_ids else ""
    wid = {k: ctx.obs.object_ids.get(k) for k in set(a.object_ids) | set(a.check_closer)}
    note = "" if a.certain else " — not certain from the table view"
    return StepResult("completed", f"{a.answer}{refs}{note} [Gemma's answer from the measured object list; not verified]",
                      {"certain": a.certain, "check_closer_world": [wid[k] for k in a.check_closer if wid.get(k)],
                       "object_world_ids": [wid[k] for k in a.object_ids if wid.get(k)], "confidence": "model"})


def run_locate(ctx: Context, args: dict) -> StepResult:
    """Find one described object in the current observation: the pick-and-place resolution path (Gemma
    resolution, lexical cross-check, comparison) plus confirm_object on its crop. Read-only."""
    text = (args or {}).get("target", "").strip()
    _ensure_described(ctx)
    for w in ctx.extra_obs:  # wrist observations from this instruction (exploration / inspection)
        found = _find_in(ctx, w, text)
        if found is not None:
            return found
    try:
        r = ctx.tools.resolve(ctx.obs, text)
    except VLMFailure as exc:
        return StepResult("failed", f"could not search for {text!r}: {exc}")
    if r.id is None:
        # Gemma lists every object when nothing fits: if no description contains the words either, it is not visible
        if not ctx.tools.lexical_candidates(ctx.obs, text):
            seen = f" or in {len(ctx.extra_obs)} wrist-camera view(s) of hidden regions" if ctx.extra_obs else ""
            return StepResult("insufficient_info", f"I do not see {text!r} in the table camera's view{seen} (no measured "
                                                   "object is described that way or confirmed on its crop)")
        if r.candidates:
            lst = ", ".join(f"{k} at x {ctx.obs.measured[k]['centre'][0] * 1000:+.0f}, y {ctx.obs.measured[k]['centre'][1] * 1000:+.0f} mm"
                            for k in r.candidates)
            return StepResult("ambiguous", f"several objects could be {text!r}: {lst}")
        return StepResult("insufficient_info", f"I do not see {text!r} from the current view")
    where = _location_check(ctx, r.id, args)  # measured location constraint first: appearance cannot override it
    if where is not None:
        return where
    try:
        c = ctx.tools.confirm(ctx.obs, r.id, text)
    except VLMFailure as exc:
        return StepResult("failed", f"could not confirm {text!r}: {exc}")
    g = ctx.obs.measured[r.id]
    ctx.log("located", text=text, mark=r.id, confirm=c.match, centre=g["centre"],
            relation=(args or {}).get("relation"), anchor=(args or {}).get("anchor"))
    if c.match != "yes":
        return StepResult("insufficient_info", f"object {r.id} might be {text!r} but the check on its crop said {c.match}")
    return StepResult("completed", f"{text!r} is object {r.id} at x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm "
                                   f"({np.hypot(*g['centre'][:2]):.2f} m from the robot), {g['height'] * 1000:.0f} mm tall "
                                   "[position measured; identity from Gemma, confirmed on its crop]",
                      {"mark": r.id, "bindings": {text: ctx.obs.object_ids.get(r.id)},
                       "evidence": [{"kind": "frame", "id": ctx.obs.frame_id, "mark": r.id}]})


def _location_check(ctx: Context, mark: int, args: dict) -> StepResult | None:
    """The request said WHERE the object is (args relation + anchor, e.g. in the buffer tray): the candidate must
    satisfy it on the measured geometry (Tools.relation_holds). None = satisfied or no constraint."""
    rel, anchor = (args or {}).get("relation"), (args or {}).get("anchor")
    if not (rel and anchor):
        return None
    text = (args or {}).get("target", "")
    try:
        a = ctx.tools.resolve(ctx.obs, anchor)
    except VLMFailure as exc:
        return StepResult("failed", f"could not find {anchor!r} to check where {text!r} is: {exc}")
    if a.id is None or a.id == mark:
        return StepResult("insufficient_info", f"I do not see {anchor!r}, so I cannot check that {text!r} is {rel} it")
    ok = ctx.tools.relation_holds(ctx.obs.measured[mark], ctx.obs.measured[a.id], rel)
    g, ga = ctx.obs.measured[mark]["centre"], ctx.obs.measured[a.id]["centre"]
    ctx.log("location_check", text=text, mark=mark, relation=rel, anchor=anchor, anchor_mark=a.id, holds=bool(ok),
            centre=g, anchor_centre=ga)
    if ok:
        return None
    return StepResult("insufficient_info", f"object {mark} may look like {text!r}, but it is not {rel} {anchor!r} "
                      f"(measured at x {g[0] * 1000:+.0f}, y {g[1] * 1000:+.0f} mm; {anchor!r} is object {a.id} at "
                      f"x {ga[0] * 1000:+.0f}, y {ga[1] * 1000:+.0f} mm), so I do not use it")


def _find_in(ctx: Context, wobs, text: str) -> StepResult | None:
    """A described object among a wrist observation's NEW objects (not seen by the table camera), confirmed on
    its crop; None if not there."""
    new = [k for k in wobs.marks if getattr(wobs, "assoc", {}).get(k, {}).get("status") == "new"]
    if not new:
        return None
    ctx.tools.describe(wobs, new)
    words = ctx.tools.words(text) - GENERIC_WORDS
    if not words:  # "the object / something behind ...": the single newly seen COMPLETE object answers it
        whole = [k for k in new if wobs.measured[k].get("tilt_deg") is not None]  # top face measured (not a side face)
        if len(whole) != 1:
            return None
        k = whole[0]
        g, d = wobs.measured[k], wobs.descriptions.get(k) or {}
        ctx.log("located", text=text, mark=k, confirm="generic reference", centre=g["centre"], camera="wrist")
        return StepResult("completed", f"Behind it, the wrist camera found an object the table camera cannot see (frame {wobs.frame_id}): "
                                       f"at x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm, {g['height'] * 1000:.0f} mm "
                                       f"tall, described as {d.get('colour', '')} {d.get('name', '')} [position measured; description Gemma's]",
                          {"mark": k, "camera": "wrist", "bindings": {text: wobs.assoc.get(k, {}).get("world_id")},
                           "evidence": [{"kind": "frame", "id": wobs.frame_id, "mark": k}]})
    else:
        cands = [k for k in ctx.tools.lexical_candidates(wobs, text) if k in new]
        if not cands:  # descriptions use other words (e.g. "green shape" for a cylinder): check the crops of
            # newly seen objects sharing at least one word with the reference, at most 3
            def overlap(k):
                d = wobs.descriptions.get(k) or {}
                return len(words & ctx.tools.words(" ".join(str(d.get(f, "")) for f in ("name", "colour", "shape"))))
            cands = [k for k in sorted(new, key=overlap, reverse=True) if overlap(k) > 0][:3]
    k, c = None, None
    for cand in cands:
        c = ctx.tools.confirm(wobs, cand, text)
        if c.match == "yes":
            k = cand
            break
    if k is None:
        return None
    g = wobs.measured[k]
    ctx.log("located", text=text, mark=k, confirm=c.match, centre=g["centre"], camera="wrist")
    return StepResult("completed", f"{text!r} found with the wrist camera (frame {wobs.frame_id}, not visible to the table camera) at "
                                   f"x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm, {g['height'] * 1000:.0f} mm tall "
                                   "[position measured; identity from Gemma, confirmed on its crop]",
                      {"mark": k, "camera": "wrist", "bindings": {text: wobs.assoc.get(k, {}).get("world_id")},
                       "evidence": [{"kind": "frame", "id": wobs.frame_id, "mark": k}]})


def _target_mark(ctx: Context, text: str, args: dict | None = None) -> tuple[int | None, StepResult | None]:
    """Resolve + confirm a described object in the current (table) observation, like locate (with its location
    constraint, if the request gave one)."""
    r = run_locate(ctx, {"target": text, "relation": (args or {}).get("relation"), "anchor": (args or {}).get("anchor")})
    if r.outcome != "completed":
        return None, r
    return r.data["mark"], None


def _wrist_observation(ctx: Context, target_world: np.ndarray, target_id: int | None) -> dict:
    """Fresh wrist frame -> every wrist object folded into the world model (association + fusion rule);
    the wrist mark whose world centre is nearest the look-at point is the inspected one."""
    t = ctx.tools
    wobs = t.perceive("wrist")
    ctx.extra_obs.append(wobs)
    assoc = {}
    for k, g in wobs.measured.items():
        st, oid = t.world.add_observation(g, "wrist", wobs.frame_id, k)
        assoc[k] = {"status": st, "world_id": oid}
    wobs.assoc = assoc
    near = sorted(wobs.measured, key=lambda k: np.hypot(*(np.array(wobs.measured[k]["centre"][:2]) - target_world[:2])))
    mark = near[0] if near and np.hypot(*(np.array(wobs.measured[near[0]]["centre"][:2]) - target_world[:2])) < 0.06 else None
    desc = None
    if mark is not None:
        crop = __import__("agent.somark", fromlist=["isolate"]).isolate(wobs.rgb, wobs.marks[mark])
        desc = t.roles.describe_one(crop).model_dump()
        wobs.descriptions[mark] = desc
        if assoc[mark]["world_id"] is not None:
            t.world.objects[assoc[mark]["world_id"]].descriptions_by_camera["wrist"] = desc
    return {"obs": wobs, "assoc": assoc, "mark": mark, "description": desc}


def run_inspect(ctx: Context, args: dict) -> StepResult:
    """Closer look at one described, visible object with the wrist camera. Read-only for the scene; moves
    the arm (collision-checked) and returns HOME (verified) at the end."""
    from agent.exploration import ViewpointPlanner
    from control.kinematics import Kinematics
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text = (args or {}).get("target", "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not inspecting while holding an object")
    mark, fail = _target_mark(ctx, text, args)
    if fail is not None:
        return fail
    g = ctx.obs.measured[mark]
    oid = ctx.obs.object_ids.get(mark)
    target = np.array([g["centre"][0], g["centre"][1], g["top_z"]])
    q0 = t.bridge.q()
    vp = ViewpointPlanner()
    tp = time.monotonic()
    plan = vp.plan(target, q0, t.obstacles(), target_box_name=f"obj_{oid}" if oid else None)
    plan_s = round(time.monotonic() - tp, 2)
    ctx.log("exploration", purpose="inspect", target=text, mark=mark, plan=plan.summary(), seconds=plan_s)
    if not plan.ok:
        return StepResult("infeasible", f"no safe inspection viewpoint for {text!r}: {plan.reason}", {"plan": plan.summary()})
    b = plan.best
    res = t.execute([MoveStep(joint_move(Kinematics(), q0, b.q, ARM_SPEED), f"move wrist camera to inspect {text}")], "inspect view")
    if not res.ok:
        return StepResult("unsafe" if "safety" in (res.reason or "").lower() else "failed", f"inspection move failed: {res.reason}")
    t.bridge.wait_still()
    reach = vp.reached(t.bridge.q(), b)
    ctx.log("verification", check="viewpoint", **reach)
    if not reach["pass"]:
        return StepResult("failed", f"viewpoint not reached (max joint error {reach['max_joint_error_deg']} deg)")
    w = _wrist_observation(ctx, target, oid)
    home = run_return_home(ctx, {})
    wm = w["mark"]
    data = {"plan": plan.summary(), "plan_seconds": plan_s, "reach": reach, "wrist_frame": w["obs"].frame_id,
            "association": w["assoc"].get(wm) if wm else None, "home": home.outcome, "seconds": round(time.monotonic() - t0, 1)}
    if wm is None:
        return StepResult("insufficient_info", f"moved the wrist camera to {text!r} but did not measure it there "
                          f"(table measurement kept); {home.summary}", data)
    wg = w["obs"].measured[wm]
    d = w["description"] or {}
    data.update({"table_centre": g["centre"], "wrist_centre": wg["centre"], "wrist_footprint": wg["footprint"],
                 "wrist_height": wg["height"], "wrist_description": d})
    ctx.log("inspection", target=text, mark=mark, wrist_mark=wm, table_centre=g["centre"], wrist_centre=wg["centre"],
            description=d, association=data["association"])
    shift = np.hypot(*(np.array(wg["centre"][:2]) - np.array(g["centre"][:2]))) * 1000
    used = (data["association"] or {}).get("status") == "matched" and t.world.objects[data["association"]["world_id"]].observations[-1]["used_for_geometry"]
    data["bindings"] = {text: oid}
    data["evidence"] = [{"kind": "frame", "id": w["obs"].frame_id, "mark": wm}]
    return StepResult("completed",
                      f"Inspected {text!r} from {b.distance:.2f} m with the wrist camera: at x {wg['centre'][0] * 1000:+.0f} mm, "
                      f"y {wg['centre'][1] * 1000:+.0f} mm ({shift:.0f} mm from the table camera's estimate), about "
                      f"{wg['footprint'][0] * 1000:.0f} x {wg['footprint'][1] * 1000:.0f} mm, {wg['height'] * 1000:.0f} mm tall; "
                      f"close-up: {d.get('colour', '')} {d.get('name', '')}, shape {d.get('shape', '?')!r} [Gemma, from the wrist view]. "
                      f"{'Wrist measurement adopted' if used else 'Table measurement kept (wrist view partial or inconsistent)'}; {home.summary}.",
                      data)


MAX_EXPLORE_VIEWPOINTS = 2
GENERIC_WORDS = {"object", "thing", "something", "item", "anything", "stuff"}  # English: no category meaning


def run_explore(ctx: Context, args: dict) -> StepResult:
    """Look into regions the table camera cannot see: the occlusion shadows behind measured objects (behind the
    object named in `near` if given, else the largest shadows), at most MAX_EXPLORE_VIEWPOINTS collision-checked
    wrist viewpoints, then HOME. Moving the camera is not finding: the result says what was newly seen; the
    supervisor decides the goal with locate."""
    from agent.exploration import ViewpointPlanner, shadow_points
    from control.kinematics import Kinematics
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text, near = (args or {}).get("target", "").strip(), ((args or {}).get("near") or "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not exploring while holding an object")
    objs = [{"id": o.id, "geometry": o.geometry} for o in t.world.objects.values() if o.status in ("on_surface", "occluded")]
    if near:  # behind the named object; if several objects fit the name, behind each (bounded)
        r = ctx.tools.resolve(ctx.obs, near)
        marks = [r.id] if r.id is not None else list(r.candidates)
        if not marks:
            return StepResult("insufficient_info", f"cannot find {near!r} to look behind")
        ids = {ctx.obs.object_ids.get(m) for m in marks}
        objs = [o for o in objs if o["id"] in ids]
    regions = shadow_points(objs, n_max=MAX_EXPLORE_VIEWPOINTS)
    if not regions:
        return StepResult("insufficient_info", "no hidden region to look into")
    vp, views, new_total, tried = ViewpointPlanner(), 0, 0, []
    found = None
    for point, behind_id in regions:
        q0 = t.bridge.q()
        tp = time.monotonic()
        from agent.exploration import REGION_HEIGHT, REGION_RADIUS
        plan = vp.plan(point, q0, t.obstacles(), region_radius=REGION_RADIUS, region_height=REGION_HEIGHT,
                       elevations=(70.0, 55.0, 40.0))  # steeper views see further behind an occluder
        ctx.log("exploration", purpose="explore", target=text, region=[round(v, 3) for v in point], behind=behind_id,
                plan=plan.summary(), seconds=round(time.monotonic() - tp, 2))
        tried.append({"region": point.tolist(), "behind": behind_id, "plan": plan.summary()})
        if not plan.ok:
            continue
        res = t.execute([MoveStep(joint_move(Kinematics(), q0, plan.best.q, ARM_SPEED), f"move wrist camera to look behind object {behind_id}")], "explore view")
        if not res.ok:
            run_return_home(ctx, {})
            return StepResult("unsafe" if "safety" in (res.reason or "").lower() else "failed", f"exploration move failed: {res.reason}")
        t.bridge.wait_still()
        reach = vp.reached(t.bridge.q(), plan.best)
        ctx.log("verification", check="viewpoint", **reach)
        if not reach["pass"]:
            break
        views += 1
        w = _wrist_observation(ctx, point, None)
        new = [k for k, a in w["obs"].assoc.items() if a["status"] == "new"]
        new_total += len(new)
        found = _find_in(ctx, w["obs"], text) if text else None
        if found is not None:
            break
    home = run_return_home(ctx, {}) if views else None
    ctx.log("explore_result", target=text, viewpoints=views, new_objects=new_total, found=found is not None,
            seconds=round(time.monotonic() - t0, 1))
    data = {"viewpoints": views, "new_objects": new_total, "regions": tried, "found": found is not None}
    if views == 0:
        reasons = "; ".join(r["plan"]["reason"] for r in tried)
        return StepResult("infeasible", f"no safe viewpoint into the hidden region(s): {reasons}", data)
    tail = f" {home.summary}." if home else ""
    return StepResult("completed", f"Looked into {views} hidden region(s) with the wrist camera: {new_total} object(s) not "
                                   f"visible to the table camera" + (f"; {found.summary}" if found else f"; no {text!r} among them") + tail,
                      {**data, "found_summary": found.summary if found else None})


def _skill(ctx: Context, mode: str, task: dict, final_home: bool, step_no: int) -> StepResult:
    from agent.pick_place import PickPlaceWorkflow
    given = ctx.obs if ctx.tools.is_fresh(ctx.obs, need_clear_view=True) else None
    wf = PickPlaceWorkflow(ctx.tools, ctx.log, ctx.run_dir, ctx.responder, ctx.budgets, final_home=final_home, mode=mode)
    r = wf.run(ctx.state.instruction, task, observation=given, thread_id=f"{ctx.state.started:.0f}_step{step_no}_{mode}")
    ctx.state.verified.extend(r.get("verifications") or [])
    ctx.state.questions.extend(r.get("questions") or [])
    ctx.obs = wf.ctx.get("obs", ctx.obs)
    b = {}
    if r.get("object_id") is not None and (task.get("object") or {}).get("text"):
        b[task["object"]["text"]] = r["object_id"]
    if r.get("reference_id") is not None and (task.get("reference") or {}).get("text"):
        b[task["reference"]["text"]] = r["reference_id"]
    data = {"report": r, "bindings": b, "evidence": [{"kind": "verification", "check": v.get("check"), "pass": v.get("pass")}
                                                     for v in r.get("verifications") or []],
            "world_changes": [f"holding #{ctx.tools.held_object}"] if mode == "pick" and r["status"] == "COMPLETED" else []}
    if r["status"] == "COMPLETED":
        return StepResult("completed", r["message"], data)
    f = r.get("failure") or {}
    return StepResult(_failure_outcome(f.get("cls"), f.get("reason", "")), r["message"], data)


def run_pick(ctx: Context, args: dict, step_no: int) -> StepResult:
    t, text = ctx.tools, (args.get("object") or {}).get("text", "")
    if t.held_object is not None:
        if text == f"#{t.held_object}":
            return StepResult("completed", f"already holding #{t.held_object} (no new grasp)", {"bindings": {text: t.held_object}})
        return StepResult("unsafe", f"the gripper already holds #{t.held_object}; place it before picking another object")
    return _skill(ctx, "pick", {"object": args["object"], "relation": None, "reference": None, "frame": "robot"}, False, step_no)


def run_place(ctx: Context, args: dict, final_home: bool, step_no: int) -> StepResult:
    t = ctx.tools
    named = (args.get("object") or {}).get("text", "") if isinstance(args.get("object"), dict) else ""
    if t.held_object is None:
        o = t.world.objects.get(int(named[1:])) if named.startswith("#") and named[1:].isdigit() else None
        if o is not None and o.status == "missing":  # it was held, the fingers say it no longer is
            return StepResult("ambiguous", f"{named} is no longer in the gripper and its position is unknown. Nothing was "
                              "moved. Should I look for it, or pick it up again?")
        return StepResult("unsupported", "nothing is held: there is nothing to place (pick it up first, or ask for pick-and-place)")
    if named.startswith("#") and named[1:].isdigit() and int(named[1:]) != t.held_object:
        return StepResult("ambiguous", f"you asked to place {named}, but the gripper holds #{t.held_object}; which one do you mean?")
    held = t.held_object
    ev = t.revalidate_held()  # the memory says "held": the fingers must say so too, before any motion
    if ev["held"] != "verified":
        return StepResult("ambiguous", f"I no longer hold #{held}: the fingers are not closed on anything (finger "
                          f"{ev['finger_deg']} deg). Nothing was moved. Should I look for it, or pick it up again?",
                          {"evidence": [{"kind": "verification", "check": "held_state", "pass": False}]})
    task = {"object": {"text": f"#{t.held_object}"}, "relation": args["relation"], "reference": args["reference"],
            "frame": args.get("frame") or "robot"}
    return _skill(ctx, "place", task, final_home, step_no)


def run_decompose(ctx: Context, args: dict) -> StepResult:
    """Planning worker: the goal over the measured objects -> concrete steps (#ids) + final conditions."""
    _ensure_described(ctx)
    facts, ids = world_facts(ctx)
    held = f"The gripper holds #{ctx.tools.held_object}." if ctx.tools.held_object else "The gripper is empty."
    try:
        sp = ctx.tools.roles.decompose(args["goal"], facts, ids, context=held)
    except VLMFailure as exc:
        return StepResult("failed", f"could not plan the goal over the measured objects: {exc}")
    if not sp.steps:
        return StepResult("infeasible", f"the goal cannot be planned with the available skills: {sp.reason}")
    steps = [(st.capability, st.args) for st in sp.steps]
    conds = [c.model_dump(exclude_none=True) for c in sp.goal]
    ctx.log("decomposed", goal=args["goal"], steps=[{"capability": c, "args": a} for c, a in steps], conditions=conds,
            reason=sp.reason)
    return StepResult("completed", f"planned {len(steps)} step(s) over the measured objects: {sp.reason}",
                      {"steps": steps, "conditions": conds, "confidence": "model"})


def run_pick_place(ctx: Context, args: dict, final_home: bool, step_no: int) -> StepResult:
    if ctx.tools.held_object is not None:
        return StepResult("unsafe", f"the gripper already holds #{ctx.tools.held_object}; place it before picking another object")
    task = {"object": args["object"], "relation": args["relation"], "reference": args["reference"],
            "frame": args.get("frame") or "robot"}
    return _skill(ctx, "pick_place", task, final_home, step_no)


def run_return_home(ctx: Context, args: dict) -> StepResult:
    r = ctx.tools.return_home()
    if not r["ok"]:
        return StepResult("unsafe" if r.get("result") is None else "failed", f"HOME not reached: {r['reason']}")
    v = ctx.tools.verify_home()
    ctx.state.verified.append(v)
    if v["pass"]:
        return StepResult("completed", f"at HOME (max joint error {v['max_error_deg']} deg)")
    return StepResult("failed", f"HOME not reached: max joint error {v['max_error_deg']} deg > {v['tolerance_deg']} deg")

