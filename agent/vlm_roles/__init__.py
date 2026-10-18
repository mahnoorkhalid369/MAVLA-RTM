"""Gemma roles: every question the agent asks the vision-language model, on one shared model, with strict JSON
schemas (pydantic), a deterministic check, one retry and a typed failure (VLMFailure).

    gemma.py    GemmaVLM (Gemma 4 E2B via vlm/model.py), RecordedVLM (tests), the VLM protocol
    prompts.py  every prompt text, verbatim (no object knowledge: placeholders <A>, <B> only)
    schemas.py  the answer schemas (PlanStep, GoalPlan, Resolution, Confirmation, ...) and VLMFailure
    roles.py    Roles: classify_request, plan_goal, decompose, answer, describe(_one/_many), shortlist, resolve,
                compare, confirm, decide (the closed-loop decision), parse; structural step validation (check_steps)

    vlm = GemmaVLM()                    # or RecordedVLM(answers) in tests
    roles = Roles(vlm, log=event_log)
    plan = roles.plan_goal(instruction, capabilities.planner_menu())

Every name is importable from agent.vlm_roles, as before the split.
"""

from agent.vlm_roles.gemma import VLM, GemmaVLM, RecordedVLM
from agent.vlm_roles.prompts import (ANSWER_PROMPT, COMPARE_PROMPT, CONFIRM_PROMPT, DECIDE_SYSTEM, DECOMPOSE_PROMPT,
                                     DESCRIBE_ONE_PROMPT, DESCRIBE_PROMPT, PARSE_SYSTEM, PLAN_SYSTEM, REQUEST_SYSTEM,
                                     RESOLVE_PROMPT, SHORTLIST_PROMPT)
from agent.vlm_roles.roles import Roles, _json, check_decision, check_steps
from agent.vlm_roles.schemas import (PLACEHOLDERS, RELATIONS, Condition, Confirmation, Decision, Description, Descriptions, GoalPlan,
                                     ObjectDescription, PlanStep, Ref, RequestKind, Resolution, SceneAnswer, Shortlist,
                                     SubPlan, TaskSpec, VLMFailure)

__all__ = ["VLM", "GemmaVLM", "RecordedVLM", "ANSWER_PROMPT", "COMPARE_PROMPT", "CONFIRM_PROMPT", "DECOMPOSE_PROMPT",
           "DESCRIBE_ONE_PROMPT", "DESCRIBE_PROMPT", "PARSE_SYSTEM", "PLAN_SYSTEM", "REQUEST_SYSTEM", "RESOLVE_PROMPT",
           "SHORTLIST_PROMPT", "DECIDE_SYSTEM", "Roles", "_json", "check_steps", "check_decision", "Decision",
           "PLACEHOLDERS", "RELATIONS", "Condition", "Confirmation", "Description", "Descriptions", "GoalPlan",
           "ObjectDescription", "PlanStep", "Ref", "RequestKind", "Resolution", "SceneAnswer", "Shortlist", "SubPlan",
           "TaskSpec", "VLMFailure"]
