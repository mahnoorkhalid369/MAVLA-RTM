"""Plan interpretation rules: deterministic, logged completions of the planner's output and the rule that says
whether a step still pursues a failed objective. Pure functions over the plan (no robot, no model)."""

from __future__ import annotations

ALTERNATIVES = {"answer": {"describe_scene"}, "describe_scene": {"answer"}, "pick_place": {"place"}}


def achieves(step_cap: str, failed_cap: str) -> bool:
    return step_cap == failed_cap or step_cap in ALTERNATIVES.get(failed_cap, set())


CONSUMERS = {"describe_scene", "answer", "locate", "inspect", "pick_place", "pick", "place", "explore", "decompose"}


def complete_plan(plan, instruction: str, kind: str | None = None) -> list[str]:
    """Deterministic, logged completions of the planner's output (generic rules, not command templates).
    kind: the request kind from agent/authority.py ("question" needs an answering step: a scene report is not one)."""
    from agent.vlm_roles import PlanStep
    applied = []
    answering = {"answer", "locate"} if kind == "question" else {"answer", "locate", "describe_scene"}
    if (kind == "question" or instruction.strip().endswith("?")) and not answering & {s.do for s in plan.steps}:
        plan.steps.append(PlanStep(do="answer", question=instruction.strip()))
        applied.append("question without an answer step -> answer")
    for i, st in enumerate(list(plan.steps)):  # exploring is never finding: an explore must be followed by locate
        if st.do == "explore" and not any(s.do == "locate" and s.object == st.object for s in plan.steps[i + 1:]):
            plan.steps.append(PlanStep(do="locate", object=st.object))
            applied.append(f"explore without locate -> locate {st.object!r}")
    if "observe" in {s.do for s in plan.steps} and not CONSUMERS & {s.do for s in plan.steps}:
        plan.steps.append(PlanStep(do="describe_scene"))
        applied.append("observation without a consumer -> describe_scene")
    return applied


def _ref_texts(args: dict) -> list[str]:
    out = []
    for k in ("object", "reference"):
        v = args.get(k)
        if isinstance(v, dict) and v.get("text"):
            out.append(v["text"])
    if args.get("target"):
        out.append(args["target"])
    return out
