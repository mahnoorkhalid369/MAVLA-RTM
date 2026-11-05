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
    def flat(cls, v):  # JSON shape only: arguments given flat or nested under "arguments" / "argument" / "args" are the same
        if isinstance(v, dict):
            for key in ("arguments", "argument", "args"):
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


class Bind(BaseModel):
    """Gemma settles which measured object the user's words in the goal mean: text -> "#<id>" (one of the candidates
    it was shown; Roles.decide checks that). The supervisor keeps it as the task's binding of those words."""
    text: str
    id: str


class Decision(BaseModel):
    """Gemma's next decision in the closed loop (Roles.decide), executed by the supervisor after validation."""
    action: Literal["continue", "replan", "ask_user", "finish", "stop", "new_goal"]
    steps: list[PlanStep] = []  # replan: the remaining steps (they replace the pending ones)
    question: str | None = None  # ask_user: one short question for the user
    request: str | None = None  # new_goal: the user's new request (their words): it replaces the goal
    answer: str | None = None  # finish: Gemma's reply to the user in its own words (a question, a description)
    binds: list[Bind] = []  # optional: goal words -> the measured candidate Gemma chose for them (BIND lines)
    reason: str = ""


class Clarification(BaseModel):
    """Gemma's reading of a request before any plan (Roles.clarify): clear (restated completely) or one question."""
    status: Literal["clear", "ask"]
    question: str | None = None
    request: str = ""
    binds: list[Bind] = []  # clear: the user's words for an object -> the measured #id the conversation settled
    reason: str = ""

    @field_validator("request", "reason", mode="before")
    @classmethod
    def _null_is_empty(cls, v):  # an "ask" leaves the request open: Gemma writes null (live 2026-10-26, twice)
        return "" if v is None else v


def _notation(text: str):
    """The predicate notation the prompts use, "name(arg, arg)" -> (name, [args]); None if the text is not in it.
    A syntax parse (quotes stripped), not an interpretation of words."""
    import re
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*", str(text))
    if m is None:
        return None
    args = [a.strip().strip("'\"").strip() for a in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", m.group(2))]
    return m.group(1), [a for a in args if a]


class Condition(BaseModel):
    """A measurable condition (agent/goals.py predicates); extra keys (e.g. "direction", "frame", "camera") are kept.
    Given in the prompts' notation ("on(<x>, <y>)", as a string or as "pred"), it is read structurally."""
    model_config = ConfigDict(extra="allow")
    pred: str
    args: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def notation(cls, v):
        if isinstance(v, str):
            parsed = _notation(v)
            v = {"pred": parsed[0], "args": parsed[1]} if parsed else {"pred": v.strip()}
        if isinstance(v, dict):
            v = dict(v)
            parsed = _notation(v.get("pred", ""))
            if parsed:
                v["pred"], v["args"] = parsed[0], v.get("args") or parsed[1]
            if isinstance(v.get("args"), dict):  # JSON shape: the condition's own keys nested under "args"
                nested = v.pop("args")
                v = {**{k: x for k, x in nested.items() if k != "args"}, **v, "args": nested.get("args") or []}
            if isinstance(v.get("args"), (str, int, float)):
                v["args"] = [str(v["args"])]
            args = v.get("args")
            if (v.get("pred") == "viewing" and v.get("direction") is None and isinstance(args, list) and len(args) == 3
                    and all(isinstance(x, (int, float)) for x in args)):  # viewing's vector given as its arguments
                v["direction"], v["args"] = list(args), []
            elif isinstance(args, list):
                v["args"] = [x if isinstance(x, str) else str(x) for x in args]
            if isinstance(v.get("args"), list):  # a bare number names a measured object, the same as "#<n>"
                v["args"] = [f"#{x.strip()}" if isinstance(x, str) and x.strip().isdigit() else x for x in v["args"]]
        return v
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


class SizeSpec(BaseModel):
    """A relative-size constraint, evaluated on measured geometry (never by eye)."""
    measure: Literal["volume", "height", "footprint", "width"] = "volume"
    order: Literal["largest", "smallest"]


class SideSpec(BaseModel):
    """A side of the robot (or of the fixed camera's picture), as a direction vector in an explicit frame."""
    direction: list[float]
    frame: Literal["robot", "camera"] = "robot"


class ReferenceSpec(BaseModel):
    """The user's words for one object, split by how each part is judged: appearance by Gemma on the images, size
    and side on measured geometry."""
    appearance: str = ""
    size: SizeSpec | None = None
    side: SideSpec | None = None
    other: str = ""  # any other words that pick the object out (e.g. a relation to another object)

    @field_validator("appearance", "other", mode="before")
    @classmethod
    def _words(cls, v):  # Gemma sometimes gives the words as a list, or null for none
        if v is None:
            return ""
        return " ".join(str(x) for x in v) if isinstance(v, list) else v


class AppearanceMatch(BaseModel):
    matches: list[int] = []  # numbers whose appearance fits
    unsure: list[int] = []  # numbers that cannot be judged from the picture


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
