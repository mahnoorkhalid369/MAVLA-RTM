"""Strict JSON schemas (pydantic) for every Gemma answer, and the typed failure after one invalid retry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

class PlanStep(BaseModel):
    """One step Gemma plans: an operation name and its own flat arguments. The name is open (not a list of allowed
    tasks); an implemented operation checks its arguments through its interface (agent/capabilities/specs.py
    adapt_args, ValueError -> Gemma's one retry); an operation without an implementation stays in the plan and is
    reported unavailable when reached."""
    model_config = ConfigDict(extra="allow")
    do: str
    worker: str | None = None  # the worker Gemma chooses for this subtask (checked against what each worker implements)

    @model_validator(mode="before")
    @classmethod
    def flat(cls, v):  # JSON shape only: arguments given flat or nested under "arguments" / "args" are the same
        if isinstance(v, dict):
            for key in ("arguments", "args"):
                if isinstance(v.get(key), dict):
                    v = {**{k: x for k, x in v.items() if k != key}, **{k: x for k, x in v[key].items() if k != "do"}}
        return v

    @field_validator("do", mode="before")
    @classmethod
    def name(cls, v):
        v = str(v or "").strip()
        if not v:
            raise ValueError("every step needs an operation name in \"do\"")
        return v

    @property
    def capability(self) -> str:
        return self.do

    @property
    def raw(self) -> dict:
        return dict(self.model_extra or {})

    @property
    def args(self) -> dict:
        from agent.capabilities.specs import adapt_args  # the operation's interface (lazy: no import cycle)
        return adapt_args(self.do, self.raw)


class Decision(BaseModel):
    """Gemma's next decision in the closed loop (Roles.decide), executed by the supervisor after validation."""
    action: Literal["continue", "replan", "ask_user", "finish", "stop"]
    steps: list[PlanStep] = []  # replan: the remaining steps (they replace the pending ones)
    question: str | None = None  # ask_user: one short question for the user
    reason: str = ""


class Condition(BaseModel):
    """A measurable condition (agent/goals.py predicates); extra keys (e.g. "direction", "frame", "camera") are kept."""
    model_config = ConfigDict(extra="allow")
    pred: str
    args: list[str] = []
    by: Literal["size", "height", "width"] | None = None
    axis: Literal["x", "y"] | None = None


class SubPlan(BaseModel):
    steps: list[PlanStep] = []
    goal: list[Condition] = []
    reason: str = ""


class GoalPlan(BaseModel):
    """Gemma's reading of the request: the goal in words, its measurable success criteria (the original goal, fixed for
    the whole task; a plan is only a strategy) and the first steps."""
    goal: str = ""
    success: list[Condition] = []
    steps: list[PlanStep] = []
    reason: str = ""

    @field_validator("success", mode="before")
    @classmethod
    def criteria(cls, v):  # JSON shape only: one condition object or none written as text
        if isinstance(v, dict):
            return [v]
        return v if isinstance(v, list) else []


class SceneAnswer(BaseModel):
    answer: str
    object_ids: list[int] = []
    certain: bool = True  # False: the descriptions do not settle it
    check_closer: list[int] = []  # objects a closer look could settle


class Description(BaseModel):
    id: int
    name: str
    colour: str = ""
    shape: str = ""
    material: str = ""


class ObjectDescription(BaseModel):
    name: str
    colour: str = ""
    shape: str = ""
    material: str = ""


class Descriptions(BaseModel):
    objects: list[Description]


class Shortlist(BaseModel):
    lists: dict[str, list[int]]


class Resolution(BaseModel):
    id: int | None
    ambiguous: bool = False
    candidates: list[int] = []
    reason: str = ""


class Confirmation(BaseModel):
    match: Literal["yes", "no", "unsure"]
    reason: str = ""

    @field_validator("match", mode="before")
    @classmethod
    def lower(cls, v):
        return str(v).strip().lower()


class RequestKind(BaseModel):
    kind: Literal["question", "description", "action"]
    evidence: str = ""


class VLMFailure(RuntimeError):
    pass
