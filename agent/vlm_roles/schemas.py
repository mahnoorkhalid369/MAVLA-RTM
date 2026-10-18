"""Strict JSON schemas (pydantic) for every Gemma answer, and the typed failure after one invalid retry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

RELATIONS = ("on", "next_to", "left_of", "right_of", "in_front_of", "behind")


class Ref(BaseModel):
    text: str  # the object's description as said in the instruction, e.g. "red cube"
    relation: Literal["next_to", "left_of", "right_of", "in_front_of", "behind", "on"] | None = None
    anchor: str | None = None  # for relational references: "the cube next to the blue cylinder"


class TaskSpec(BaseModel):
    task: Literal["pick_place", "pick", "unsupported"]
    object: Ref | None = None
    relation: Literal["on", "next_to", "left_of", "right_of", "in_front_of", "behind"] | None = None
    reference: Ref | None = None
    frame: Literal["robot", "camera"] = "robot"
    reason: str = ""


PLACEHOLDERS = {"", "null", "none", "nothing", "n/a", "na", "?", "unknown", "no", "nowhere"}


class PlanStep(BaseModel):
    """One flat step (the 2B model drops braces in nested JSON; flat keys are converted deterministically)."""
    do: Literal["observe", "describe_scene", "answer", "locate", "inspect", "explore", "pick", "place", "pick_place",
                "return_home", "decompose"]
    object: str | None = None  # pick / pick_place: object to move; locate / inspect / explore: what to find
    relation: Literal["on", "next_to", "left_of", "right_of", "in_front_of", "behind"] | None = None
    target: str | None = None  # place / pick_place: destination object
    frame: Literal["robot", "camera"] = "robot"
    object_relation: Literal["next_to", "left_of", "right_of", "in_front_of", "behind", "on", "in"] | None = None
    object_anchor: str | None = None  # "the <A> that is next to the <B>": object_relation next_to, object_anchor <B>;
    # "the <A> in the <B>": object_relation in (inside <B>'s footprint), object_anchor <B>
    question: str | None = None  # answer
    near: str | None = None  # explore: an object the thing may be hidden behind
    goal: str | None = None  # decompose: the goal to plan over the measured scene

    @field_validator("object", "target", "object_anchor", "question", "near", "goal", mode="before")
    @classmethod
    def no_placeholders(cls, v):  # "null" / "none" written as text means "not given": never an object name
        return None if v is None or str(v).strip().lower() in PLACEHOLDERS else str(v).strip()

    @property
    def capability(self) -> str:
        return self.do

    @property
    def args(self) -> dict:
        if self.do == "pick_place":
            return {"object": {"text": self.object, "relation": self.object_relation, "anchor": self.object_anchor},
                    "relation": self.relation, "reference": {"text": self.target, "relation": None, "anchor": None},
                    "frame": self.frame}
        if self.do == "pick":
            return {"object": {"text": self.object, "relation": self.object_relation, "anchor": self.object_anchor}}
        if self.do == "place":
            return {"object": {"text": self.object or "it", "relation": None, "anchor": None}, "relation": self.relation,
                    "reference": {"text": self.target, "relation": None, "anchor": None}, "frame": self.frame}
        if self.do in ("locate", "inspect"):  # the location constraint travels with the target (checked on measurements)
            return {"target": self.object, "relation": self.object_relation, "anchor": self.object_anchor}
        if self.do == "explore":
            return {"target": self.object, "near": self.near}
        if self.do == "answer":
            return {"question": self.question}
        if self.do == "decompose":
            return {"goal": self.goal}
        return {}


class Decision(BaseModel):
    """Gemma's next decision in the closed loop (Roles.decide), executed by the supervisor after validation."""
    action: Literal["continue", "replan", "ask_user", "finish", "stop"]
    steps: list[PlanStep] = []  # replan: the remaining steps (they replace the pending ones)
    question: str | None = None  # ask_user: one short question for the user
    reason: str = ""


class Condition(BaseModel):
    pred: str
    args: list[str] = []
    by: Literal["size", "height", "width"] | None = None
    axis: Literal["x", "y"] | None = None


class SubPlan(BaseModel):
    steps: list[PlanStep] = []
    goal: list[Condition] = []
    reason: str = ""


class GoalPlan(BaseModel):
    goal: str = ""
    steps: list[PlanStep] = []
    success: str = ""
    reason: str = ""


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
