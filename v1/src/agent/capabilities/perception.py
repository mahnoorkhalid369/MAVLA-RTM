"""Perception capabilities (read-only, no robot motion except parking the arm out of the camera's view for a
following manipulation): observe, describe_scene, answer, locate (+ the measured location check)."""

from __future__ import annotations

import numpy as np

from agent.capabilities.specs import Context, StepResult
from agent.vlm_roles import VLMFailure


MAX_WRIST_CONFIRMS = 4  # newly seen wrist objects checked against a reference (one Gemma crop check each)


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
    ctx.state.reports.append({"frame_id": ctx.obs.frame_id, "objects": len(ctx.obs.marks)})  # what the user was told
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
    wid = {k: ctx.obs.object_ids.get(k) for k in set(a.object_ids) | set(a.check_closer)}
    note = "" if a.certain else " — not certain from the table view"
    return StepResult("completed", f"{a.answer}{refs}{note} [Gemma's answer from the measured object list; not verified]",
                      {"certain": a.certain, "check_closer_world": [wid[k] for k in a.check_closer if wid.get(k)],
                       "object_world_ids": [wid[k] for k in a.object_ids if wid.get(k)], "confidence": "model"})


def run_locate(ctx: Context, args: dict) -> StepResult:
    """Find one described object in the current observation: Gemma's resolution over the descriptions, then
    Gemma's check on the object's crop. Read-only. Not resolved: Gemma's candidates (ambiguous) or none (not seen)."""
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
        if r.candidates:
            lst = ", ".join(f"{k} at x {ctx.obs.measured[k]['centre'][0] * 1000:+.0f}, y {ctx.obs.measured[k]['centre'][1] * 1000:+.0f} mm"
                            for k in r.candidates)
            return StepResult("ambiguous", f"several objects could be {text!r}: {lst}")
        seen = f" or in {len(ctx.extra_obs)} wrist-camera view(s)" if ctx.extra_obs else ""
        return StepResult("insufficient_info", f"I do not see {text!r} in the table camera's view{seen} (Gemma matched "
                                               f"no measured object{': ' + r.reason if r.reason else ''})")
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
    """A described object among a wrist observation's NEW objects (not seen by the table camera): Gemma checks each
    one's crop against the user's words (at most MAX_WRIST_CONFIRMS); None if none is confirmed."""
    new = [k for k in wobs.marks if getattr(wobs, "assoc", {}).get(k, {}).get("status") == "new"]
    if not new:
        return None
    ctx.tools.describe(wobs, new)
    cands = new[:MAX_WRIST_CONFIRMS]  # Gemma checks each newly seen object's crop against the user's words
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
