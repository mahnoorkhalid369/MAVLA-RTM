"""VLM roles on one shared model, with strict JSON schemas (pydantic), one retry, typed failures.

Roles: parse (instruction -> task), describe (numbered image -> descriptions), resolve (reference
text -> mask id or "ambiguous"), confirm (object crop + reference attributes -> match / no / unsure).
No colour names, shape lists or object lists appear in any prompt: the model uses its own words.

    vlm = GemmaVLM()                    # or RecordedVLM(answers) in tests
    roles = Roles(vlm, log=event_log)
    task = roles.parse(instruction)
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

RELATIONS = ("on", "next_to", "left_of", "right_of", "in_front_of", "behind")


class VLM(Protocol):
    def generate(self, text: str, image: np.ndarray | None = None, system: str | None = None,
                 max_new_tokens: int | None = None, thinking: bool | None = None) -> str: ...


class GemmaVLM:
    """The project's Gemma 4 E2B (vlm/model.py), loaded once."""

    def __init__(self) -> None:
        from vlm.config import VLMConfig
        from vlm.model import VisionLanguageModel

        self.model = VisionLanguageModel(VLMConfig())

    def generate(self, text, image=None, system=None, max_new_tokens=None, thinking=None) -> str:
        return self.model.generate(text, image=image, system=system, max_new_tokens=max_new_tokens, thinking=thinking)


class RecordedVLM:
    """Replays recorded answers in order (tests: no GPU)."""

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.calls = list(answers), []

    def generate(self, text, image=None, system=None, max_new_tokens=None, thinking=None) -> str:
        self.calls.append(text)
        return self.answers.pop(0)


# ------------------------------------------------------------------------------------------ schemas
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


class Description(BaseModel):
    id: int
    name: str
    colour: str = ""
    shape: str = ""
    material: str = ""


class Descriptions(BaseModel):
    objects: list[Description]


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


class VLMFailure(RuntimeError):
    pass


def _json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in the answer")
    s = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[s:end + 1])


PARSE_SYSTEM = """You turn a robot instruction into JSON. The robot can pick one object and place it.
Copy the words the user uses for each object (do not invent or translate them).
task: "pick_place" when the user says where the object should go (on / next to / left of / right of /
in front of / behind another object); "pick" only when the user asks to pick something up and gives
NO destination; "unsupported" for anything else.
Answer with ONE JSON object only:
{"task": "pick_place" | "pick" | "unsupported",
 "object": {"text": "<words for the object to move>", "relation": null, "anchor": null},
 "relation": "on" | "next_to" | "left_of" | "right_of" | "in_front_of" | "behind" | null,
 "reference": {"text": "<words for the destination object>", "relation": null, "anchor": null} | null,
 "frame": "robot" | "camera",
 "reason": "<short>"}
"relation" and "reference" (top level) say WHERE to put the object. Keep object.relation and
object.anchor null, EXCEPT when the user picks out the object to move by where it is now (e.g. "the
<A> that is next to the <B>": object text "<A>", object relation "next_to", object anchor "<B>").
frame is "camera" only if the user explicitly says so; otherwise "robot".
Example: "put the <A> on the <B>" -> {"task": "pick_place", "object": {"text": "<A>", "relation": null,
"anchor": null}, "relation": "on", "reference": {"text": "<B>", "relation": null, "anchor": null},
"frame": "robot", "reason": "move <A> onto <B>"}"""

DESCRIBE_PROMPT = """Each object in the image is outlined and numbered. The numbers are: {ids}.
For EVERY number, describe the object inside that outline as you see it. Answer with ONE JSON object only:
{{"objects": [{{"id": <number>, "name": "<short name>", "colour": "<colour>", "shape": "<shape>", "material": "<material>"}}, ...]}}
Include every number exactly once and no other numbers."""

RESOLVE_PROMPT = """The objects in the image are outlined and numbered. Their descriptions:
{descriptions}
Which ONE numbered object does the user mean by "{text}"?
If exactly one object fits, give its number. If several fit equally well, or none fits, set "id" to null,
"ambiguous" to true and list the numbers that could fit. Answer with ONE JSON object only:
{{"id": <number or null>, "ambiguous": <true|false>, "candidates": [<numbers>], "reason": "<short>"}}"""

CONFIRM_PROMPT = """Look at the outlined object in this picture. Is it "{text}"?
Answer with ONE JSON object only: {{"match": "yes" | "no" | "unsure", "reason": "<short>"}}"""


class Roles:
    def __init__(self, vlm: VLM, log=None, thinking: bool = False) -> None:
        self.vlm, self.log, self.thinking = vlm, log, thinking

    def _call(self, role: str, schema, text: str, image=None, system=None, check=None, max_new_tokens: int = 400):
        error = None
        for attempt in (1, 2):  # one retry with the validation error
            prompt = text if error is None else f"{text}\n\nYour previous answer was not valid ({error}). Answer again with only the JSON."
            t0 = time.monotonic()
            raw = self.vlm.generate(prompt, image=image, system=system, max_new_tokens=max_new_tokens, thinking=self.thinking)
            dt = time.monotonic() - t0
            try:
                obj = schema.model_validate(_json(raw))
                if check is not None:
                    check(obj)
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, prompt=prompt, answer=raw, valid=True, seconds=round(dt, 2))
                return obj
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                error = str(exc).splitlines()[0][:200]
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, prompt=prompt, answer=raw, valid=False, error=error, seconds=round(dt, 2))
        raise VLMFailure(f"{role}: invalid answer after one retry ({error})")

    def parse(self, instruction: str) -> TaskSpec:
        def check(t: TaskSpec):  # structural consistency only (no object knowledge)
            if t.task == "pick" and t.relation is not None and t.reference is not None:
                raise ValueError('a destination is given, so task must be "pick_place"')
            if t.task == "pick_place" and (t.object is None or t.relation is None or t.reference is None):
                raise ValueError('"pick_place" needs object, relation and reference')
            if t.object is not None and t.reference is not None and t.object.anchor and \
                    t.object.anchor.strip().lower() == t.reference.text.strip().lower() and t.object.relation == t.relation:
                raise ValueError("object.relation / object.anchor repeat the destination; keep them null "
                                 "unless the user describes where the object to move is now")
        return self._call("parse", TaskSpec, f"Instruction: {instruction}", system=PARSE_SYSTEM, check=check)

    def describe(self, som_image: np.ndarray, ids: list[int]) -> Descriptions:
        def check(d: Descriptions):
            got = sorted(o.id for o in d.objects)
            if got != sorted(ids):
                raise ValueError(f"ids {got} do not match the drawn ids {sorted(ids)}")
        return self._call("describe", Descriptions, DESCRIBE_PROMPT.format(ids=", ".join(map(str, ids))), image=som_image, check=check,
                          max_new_tokens=80 * len(ids) + 300)

    def resolve(self, som_image: np.ndarray, descriptions: list[dict], text: str, ids: list[int]) -> Resolution:
        def check(r: Resolution):
            if r.id is not None and r.id not in ids:
                raise ValueError(f"id {r.id} is not one of the drawn ids")
            if r.id is None and not r.ambiguous:
                raise ValueError("id is null but ambiguous is false")
        desc = "\n".join(f"{d['id']}: {d.get('name', '')} ({d.get('colour', '')}, {d.get('shape', '')}, {d.get('material', '')})"
                         for d in descriptions)
        return self._call("resolve", Resolution, RESOLVE_PROMPT.format(descriptions=desc, text=text), image=som_image, check=check)

    def confirm(self, crop_image: np.ndarray, text: str) -> Confirmation:
        return self._call("confirm", Confirmation, CONFIRM_PROMPT.format(text=text), image=crop_image)
