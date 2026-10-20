"""Manipulation capabilities (change objects): pick, place, pick_place through the validated PickPlaceWorkflow,
and return_home. The workflow's failure classes map to supervisor outcomes (_failure_outcome)."""

from __future__ import annotations

from agent.capabilities.specs import Context, StepResult


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


def _names_held(ctx: Context, text: str) -> bool:
    """The reference names the held object: its "#id", or words this task already bound to it (task memory)."""
    held = ctx.tools.held_object
    return held is not None and (text == f"#{held}" or (ctx.state is not None and ctx.state.bindings.get(text) == held))


def run_pick(ctx: Context, args: dict, step_no: int) -> StepResult:
    t, text = ctx.tools, (args.get("object") or {}).get("text") or ""
    if t.held_object is not None:
        if _names_held(ctx, text):
            return StepResult("completed", f"already holding #{t.held_object} (no new grasp)", {"bindings": {text: t.held_object}})
        return StepResult("unsafe", f"the gripper already holds #{t.held_object}; place it before picking another object")
    return _skill(ctx, "pick", {"object": args["object"], "relation": None, "reference": None, "frame": "robot"}, False, step_no)


def run_place(ctx: Context, args: dict, final_home: bool, step_no: int) -> StepResult:
    t = ctx.tools
    named = ((args.get("object") or {}).get("text") or "") if isinstance(args.get("object"), dict) else ""
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
    r = _skill(ctx, "place", task, final_home, step_no)
    r.data.setdefault("bindings", {})["#held"] = held  # the object placed (a place step may leave it unnamed)
    return r


def run_pick_place(ctx: Context, args: dict, final_home: bool, step_no: int) -> StepResult:
    held = ctx.tools.held_object
    if held is not None:
        if _names_held(ctx, (args.get("object") or {}).get("text") or ""):  # a precondition fact (no motion); Gemma decides
            return StepResult("infeasible", f"#{held} is already in the gripper; pick_place starts with a grasp, so "
                                            "nothing was executed")
        return StepResult("unsafe", f"the gripper already holds #{held}; place it before picking another object")
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
