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
from dataclasses import dataclass, field
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


@dataclass
class StepResult:
    outcome: str  # agent.world_state.OUTCOMES
    summary: str
    data: dict = field(default_factory=dict)


SPECS: dict[str, CapabilitySpec] = {s.name: s for s in [
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
]}

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
    if cls in ("asked",):
        return "cancelled"
    if cls == "planning":
        return "infeasible"
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
    return StepResult("completed", f"{a.answer}{refs} [Gemma's answer from the measured object list; not verified]")


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
            return StepResult("insufficient_info", f"I do not see {text!r} from the current view (no measured object "
                                                   "is described that way; searching other viewpoints is not available yet)")
        if r.candidates:
            lst = ", ".join(f"{k} at x {ctx.obs.measured[k]['centre'][0] * 1000:+.0f}, y {ctx.obs.measured[k]['centre'][1] * 1000:+.0f} mm"
                            for k in r.candidates)
            return StepResult("ambiguous", f"several objects could be {text!r}: {lst}")
        return StepResult("insufficient_info", f"I do not see {text!r} from the current view")
    try:
        c = ctx.tools.confirm(ctx.obs, r.id, text)
    except VLMFailure as exc:
        return StepResult("failed", f"could not confirm {text!r}: {exc}")
    g = ctx.obs.measured[r.id]
    ctx.log("located", text=text, mark=r.id, confirm=c.match, centre=g["centre"])
    if c.match != "yes":
        return StepResult("insufficient_info", f"object {r.id} might be {text!r} but the check on its crop said {c.match}")
    return StepResult("completed", f"{text!r} is object {r.id} at x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm "
                                   f"({np.hypot(*g['centre'][:2]):.2f} m from the robot), {g['height'] * 1000:.0f} mm tall "
                                   "[position measured; identity from Gemma, confirmed on its crop]", {"mark": r.id})


def _find_in(ctx: Context, wobs, text: str) -> StepResult | None:
    """A described object among a wrist observation's NEW objects (not seen by the table camera), confirmed on
    its crop; None if not there."""
    new = [k for k in wobs.marks if getattr(wobs, "assoc", {}).get(k, {}).get("status") == "new"]
    if not new:
        return None
    ctx.tools.describe(wobs, new)
    lex = [k for k in ctx.tools.lexical_candidates(wobs, text) if k in new]
    if len(lex) != 1:
        return None
    k = lex[0]
    c = ctx.tools.confirm(wobs, k, text)
    if c.match != "yes":
        return None
    g = wobs.measured[k]
    ctx.log("located", text=text, mark=k, confirm=c.match, centre=g["centre"], camera="wrist")
    return StepResult("completed", f"{text!r} found with the wrist camera (frame {wobs.frame_id}, not visible to the table camera) at "
                                   f"x {g['centre'][0] * 1000:+.0f} mm, y {g['centre'][1] * 1000:+.0f} mm, {g['height'] * 1000:.0f} mm tall "
                                   "[position measured; identity from Gemma, confirmed on its crop]", {"mark": k, "camera": "wrist"})


def _target_mark(ctx: Context, text: str) -> tuple[int | None, StepResult | None]:
    """Resolve + confirm a described object in the current (table) observation, like locate."""
    r = run_locate(ctx, {"target": text})
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
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text = (args or {}).get("target", "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not inspecting while holding an object")
    mark, fail = _target_mark(ctx, text)
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
    res = t.execute([MoveStep(joint_move(Kinematics_tcp(), q0, b.q, ARM_SPEED), f"move wrist camera to inspect {text}")], "inspect view")
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
    return StepResult("completed",
                      f"Inspected {text!r} from {b.distance:.2f} m with the wrist camera: at x {wg['centre'][0] * 1000:+.0f} mm, "
                      f"y {wg['centre'][1] * 1000:+.0f} mm ({shift:.0f} mm from the table camera's estimate), about "
                      f"{wg['footprint'][0] * 1000:.0f} x {wg['footprint'][1] * 1000:.0f} mm, {wg['height'] * 1000:.0f} mm tall; "
                      f"close-up: {d.get('colour', '')} {d.get('name', '')}, shape {d.get('shape', '?')!r} [Gemma, from the wrist view]. "
                      f"{'Wrist measurement adopted' if used else 'Table measurement kept (wrist view partial or inconsistent)'}; {home.summary}.",
                      data)


MAX_EXPLORE_VIEWPOINTS = 2


def run_explore(ctx: Context, args: dict) -> StepResult:
    """Look into regions the table camera cannot see: the occlusion shadows behind measured objects (behind the
    object named in `near` if given, else the largest shadows), at most MAX_EXPLORE_VIEWPOINTS collision-checked
    wrist viewpoints, then HOME. Moving the camera is not finding: the result says what was newly seen; the
    supervisor decides the goal with locate."""
    from agent.exploration import ViewpointPlanner, shadow_points
    from control.motion import joint_move
    from control.robot import ARM_SPEED
    from control.skills import MoveStep
    text, near = (args or {}).get("target", "").strip(), ((args or {}).get("near") or "").strip()
    t, t0 = ctx.tools, time.monotonic()
    if t.world.holding is not None:
        return StepResult("unsafe", "not exploring while holding an object")
    objs = [{"id": o.id, "geometry": o.geometry} for o in t.world.objects.values() if o.status in ("on_surface", "occluded")]
    if near:
        m, fail = _target_mark(ctx, near)
        if fail is not None:
            return StepResult(fail.outcome, f"cannot find {near!r} to look behind: {fail.summary}")
        oid = ctx.obs.object_ids.get(m)
        objs = [o for o in objs if o["id"] == oid]
    regions = shadow_points(objs, n_max=MAX_EXPLORE_VIEWPOINTS if not near else 1)
    if not regions:
        return StepResult("insufficient_info", "no hidden region to look into")
    vp, views, new_total, tried = ViewpointPlanner(), 0, 0, []
    found = None
    for point, behind_id in regions:
        q0 = t.bridge.q()
        tp = time.monotonic()
        plan = vp.plan(point, q0, t.obstacles())
        ctx.log("exploration", purpose="explore", target=text, region=[round(v, 3) for v in point], behind=behind_id,
                plan=plan.summary(), seconds=round(time.monotonic() - tp, 2))
        tried.append({"region": point.tolist(), "behind": behind_id, "plan": plan.summary()})
        if not plan.ok:
            continue
        res = t.execute([MoveStep(joint_move(Kinematics_tcp(), q0, plan.best.q, ARM_SPEED), f"move wrist camera to look behind object {behind_id}")], "explore view")
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


def Kinematics_tcp():
    from control.kinematics import Kinematics
    return Kinematics()


def run_pick_place(ctx: Context, args: dict, final_home: bool, step_no: int) -> StepResult:
    from agent.pick_place import PickPlaceWorkflow
    task = {"object": args["object"], "relation": args["relation"], "reference": args["reference"],
            "frame": args.get("frame") or "robot"}
    given = ctx.obs if ctx.tools.is_fresh(ctx.obs, need_clear_view=True) else None
    wf = PickPlaceWorkflow(ctx.tools, ctx.log, ctx.run_dir, ctx.responder, ctx.budgets, final_home=final_home)
    r = wf.run(ctx.state.instruction, task, observation=given, thread_id=f"{ctx.state.started:.0f}_step{step_no}")
    ctx.state.verified.extend(r.get("verifications") or [])
    ctx.state.questions.extend(r.get("questions") or [])
    ctx.obs = wf.ctx.get("obs", ctx.obs)  # latest observation (stale after the motion: is_fresh says so)
    if r["status"] == "COMPLETED":
        return StepResult("completed", r["message"], {"report": r})
    f = r.get("failure") or {}
    return StepResult(_failure_outcome(f.get("cls"), f.get("reason", "")), r["message"], {"report": r})


def run_return_home(ctx: Context, args: dict) -> StepResult:
    r = ctx.tools.return_home()
    if not r["ok"]:
        return StepResult("unsafe" if r.get("result") is None else "failed", f"HOME not reached: {r['reason']}")
    v = ctx.tools.verify_home()
    ctx.state.verified.append(v)
    if v["pass"]:
        return StepResult("completed", f"at HOME (max joint error {v['max_error_deg']} deg)")
    return StepResult("failed", f"HOME not reached: max joint error {v['max_error_deg']} deg > {v['tolerance_deg']} deg")


def run_unimplemented(ctx: Context, name: str, args: dict) -> StepResult:
    why = {"explore": "searching other viewpoints is not implemented yet: perception uses the fixed table camera "
                      "only (the wrist camera is not used for perception), and the arm only moves aside to clear that view"}
    return StepResult("unsupported", why.get(name, f"'{name}' is not available"))


def elapsed(t0: float) -> float:
    return round(time.monotonic() - t0, 1)
