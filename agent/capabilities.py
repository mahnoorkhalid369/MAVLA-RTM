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
    CapabilitySpec("explore", "look for something that is not visible from the current view", '{"do": "explore", "object": "<words>"}',
                   "motion", "-", "-", "not implemented", "-", implemented=False),
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
    obs: object = None  # latest agent.tools.Observation


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
    try:
        r = ctx.tools.resolve(ctx.obs, text)
    except VLMFailure as exc:
        return StepResult("failed", f"could not search for {text!r}: {exc}")
    if r.id is None:
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
                                   "[position measured; identity from Gemma, confirmed on its crop]")


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
