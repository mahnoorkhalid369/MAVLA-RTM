"""Verification capability (the verification worker's operation): compare measured state with a goal's or a
sub-goal's conditions (agent/goals.py predicates) and return evidence of success, failure or uncertainty. It never
declares the user's task complete: the supervisor decides from its result. Internal: Gemma cannot plan it."""

from __future__ import annotations

from agent.capabilities.specs import Context, StepResult
from agent.goals import GoalEvaluator


def run_verify(ctx: Context, args: dict) -> StepResult:
    """args: {"conditions": [{"pred", "args", ...}], "scope": "subgoal" | "goal"}. Outcome: completed (every
    condition satisfied or only model-confidence), failed (some condition false), insufficient_info (some
    condition not decidable from the evidence)."""
    conds = list((args or {}).get("conditions") or [])
    ev = GoalEvaluator(ctx.tools, ctx.state)
    results = [ev.evaluate(c, ctx.state.bindings) for c in conds]
    rows = [{"condition": r.condition, "satisfied": r.satisfied, "confidence": r.confidence, "evidence": r.evidence,
             "bound": r.bound} for r in results]

    def name(r):
        return f"{r.condition.get('pred')}({', '.join(map(str, r.condition.get('args', [])))})"
    false = [r for r in results if r.satisfied is False]
    unknown = [r for r in results if r.satisfied is None and r.confidence != "model"]
    model = [r for r in results if r.confidence == "model"]
    data = {"results": rows, "scope": (args or {}).get("scope", "subgoal"),
            "evidence": [{"kind": "condition", "condition": name(r), "satisfied": r.satisfied, "confidence": r.confidence}
                         for r in results],
            "unknown_conditions": [name(r) for r in unknown],
            "confidence": "model" if model and not false and not unknown else "measured"}
    if false:
        return StepResult("failed", "not met: " + "; ".join(f"{name(r)}: {r.evidence}" for r in false), data)
    if unknown:
        return StepResult("insufficient_info", "not decidable from the evidence: "
                          + "; ".join(f"{name(r)}: {r.evidence}" for r in unknown), data)
    return StepResult("completed", ("met: " + ", ".join(name(r) for r in results)) if results else "no conditions to check", data)
