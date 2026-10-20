"""Planning capability: decompose a multi-object goal over the measured objects into concrete steps (#ids) and
final conditions (Gemma; the supervisor filters the steps by the request's authority)."""

from __future__ import annotations

from agent.capabilities.perception import _ensure_described
from agent.capabilities.specs import Context, StepResult, planner_menu
from agent.vlm_roles import VLMFailure


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


def run_decompose(ctx: Context, args: dict) -> StepResult:
    """Planning worker: the goal over the measured objects -> concrete steps (#ids) + final conditions."""
    _ensure_described(ctx)
    facts, ids = world_facts(ctx)
    held = f"The gripper holds #{ctx.tools.held_object}." if ctx.tools.held_object else "The gripper is empty."
    try:
        sp = ctx.tools.roles.decompose(args["goal"], facts, ids, context=held, menu=planner_menu())
    except VLMFailure as exc:
        return StepResult("failed", f"could not plan the goal over the measured objects: {exc}")
    if not sp.steps:
        return StepResult("infeasible", f"the goal cannot be planned over the measured objects: {sp.reason}")
    steps = [(st.capability, st.args) for st in sp.steps]
    conds = [c.model_dump(exclude_none=True) for c in sp.goal]
    ctx.log("decomposed", goal=args["goal"], steps=[{"capability": c, "args": a} for c, a in steps], conditions=conds,
            reason=sp.reason)
    return StepResult("completed", f"planned {len(steps)} step(s) over the measured objects: {sp.reason}",
                      {"steps": steps, "conditions": conds, "confidence": "model"})
