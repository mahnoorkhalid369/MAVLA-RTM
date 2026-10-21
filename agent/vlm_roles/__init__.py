"""Gemma roles: every question the agent asks the vision-language model, on one shared model, with strict JSON
schemas (pydantic), a deterministic check, one retry and a typed failure (VLMFailure).

    gemma.py    GemmaVLM (Gemma 4 E2B via vlm/model.py), RecordedVLM (tests), the VLM protocol
    prompts.py  every prompt text, verbatim (no object knowledge: placeholders <A>, <B> only); IDENTITY_SYSTEM is
                put in front of the embodied roles' prompts by Roles._call (EMBODIED_ROLES)
    schemas.py  the answer schemas (PlanStep, GoalPlan, Resolution, Confirmation, ...) and VLMFailure
    roles.py    Roles: classify_request, plan_goal, decompose, answer, describe(_one/_many), shortlist, resolve,
                confirm, decide (the closed-loop decision); structural step validation (check_steps)

    vlm = GemmaVLM()                    # or RecordedVLM(answers) in tests
    roles = Roles(vlm, log=event_log)
    plan = roles.plan_goal(instruction, capabilities.planner_menu())

Every name is importable from agent.vlm_roles, as before the split.
"""

from agent.vlm_roles.gemma import VLM, GemmaVLM, RecordedVLM
from agent.vlm_roles.prompts import (ANSWER_PROMPT, CONFIRM_PROMPT, DECIDE_SYSTEM, DECOMPOSE_PROMPT, DESCRIBE_ONE_PROMPT,
                                     DESCRIBE_PROMPT, IDENTITY_SYSTEM, PLAN_SYSTEM, REQUEST_SYSTEM, RESOLVE_PROMPT,
                                     SHORTLIST_PROMPT, REFERENCE_PROMPT, APPEARANCE_PROMPT,
                                     APPEARANCE_ONE_PROMPT)
from agent.vlm_roles.roles import EMBODIED_ROLES, Roles, _json, check_decision, check_steps
from agent.vlm_roles.schemas import (AppearanceMatch, ReferenceSpec, SideSpec, SizeSpec, Condition, Confirmation, Decision, Description, Descriptions, GoalPlan,
                                     ObjectDescription, PlanStep, RequestKind, Resolution, SceneAnswer, Shortlist, SubPlan,
                                     VLMFailure)

__all__ = ["VLM", "GemmaVLM", "RecordedVLM", "ANSWER_PROMPT", "CONFIRM_PROMPT", "DECIDE_SYSTEM", "DECOMPOSE_PROMPT",
           "DESCRIBE_ONE_PROMPT", "DESCRIBE_PROMPT", "IDENTITY_SYSTEM", "PLAN_SYSTEM", "REQUEST_SYSTEM", "RESOLVE_PROMPT",
           "SHORTLIST_PROMPT", "EMBODIED_ROLES", "Roles", "_json", "check_decision", "check_steps", "Condition",
           "Confirmation", "Decision", "Description", "Descriptions", "GoalPlan", "ObjectDescription", "PlanStep",
           "RequestKind", "Resolution", "SceneAnswer", "Shortlist", "SubPlan", "VLMFailure", "REFERENCE_PROMPT",
           "APPEARANCE_PROMPT", "APPEARANCE_ONE_PROMPT", "AppearanceMatch", "ReferenceSpec", "SideSpec", "SizeSpec"]
