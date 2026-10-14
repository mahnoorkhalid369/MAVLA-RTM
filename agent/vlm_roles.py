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

    def _generate_batch_disabled(self, text: str, images: list, max_new_tokens: int) -> list[str]:
        """DISABLED (benchmarked 2026-10-14): batched multimodal generation with left padding returned
        invalid / pad-only rows for 7 of 9 crops and was not faster than single calls. Kept for reference;
        describe_many falls back to one call per crop because GemmaVLM has no `generate_batch`."""
        import torch

        m = self.model
        msgs = [[{"role": "user", "content": [{"type": "image", "image": m._image(img)}, {"type": "text", "text": text}]}]
                for img in images]
        tok = m.processor.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "left"
        try:
            inputs = m.processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                                     add_generation_prompt=True, enable_thinking=False,
                                                     processor_kwargs={"padding": True}).to(m.model.device)
            with torch.inference_mode():
                out = m.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            tok.padding_side = old_side
        n = inputs["input_ids"].shape[-1]
        answers = []
        pad = tok.pad_token_id
        for i, row in enumerate(out):
            raw = m.processor.decode(row[n:], skip_special_tokens=False)
            prompt = inputs["input_ids"][i]
            prompt = prompt[prompt != pad] if pad is not None else prompt  # this row's prompt without left padding
            parsed = m.processor.parse_response(raw, prefix=prompt[None])
            answers.append(str(parsed.get("content") if isinstance(parsed, dict) else parsed).strip())
        return answers


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


class PickPlaceArgs(BaseModel):
    object: Ref
    relation: Literal["on", "next_to", "left_of", "right_of", "in_front_of", "behind"]
    reference: Ref
    frame: Literal["robot", "camera"] = "robot"


class PlanStep(BaseModel):
    capability: Literal["observe", "describe_scene", "answer", "pick_place", "return_home", "explore"]
    args: dict = {}


class GoalPlan(BaseModel):
    goal: str = ""
    steps: list[PlanStep] = []
    success: str = ""
    reason: str = ""


class SceneAnswer(BaseModel):
    answer: str
    object_ids: list[int] = []


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

PLAN_SYSTEM = """You plan for a robot arm that has a camera looking at a table with objects on it.
Turn the user's instruction into a short list of steps, using ONLY these capabilities:
{menu}
Rules: copy the user's own words for objects; one pick_place step per object to move, in the order the user
wants; use "observe" before "describe_scene" or "answer"; no steps that the user did not ask for. If none of
the capabilities can do what the user wants, return an empty step list and explain why in "reason".
For pick_place, keep object.relation and object.anchor null unless the user picks out the object to move by
where it is now (e.g. "the <A> that is next to the <B>"). frame is "camera" only if the user says so.
Answer with ONE JSON object only:
{{"goal": "<one line>", "steps": [{{"capability": "<name>", "args": {{...}}}}], "success": "<what must be true at the end>", "reason": "<short>"}}
Examples:
"put the <A> on the <B>" -> {{"goal": "move <A> onto <B>", "steps": [{{"capability": "pick_place", "args": {{"object": {{"text": "<A>", "relation": null, "anchor": null}}, "relation": "on", "reference": {{"text": "<B>", "relation": null, "anchor": null}}, "frame": "robot"}}}}], "success": "<A> stands on <B>", "reason": "one move"}}
"what is on the table?" -> {{"goal": "report the scene", "steps": [{{"capability": "observe", "args": {{}}}}, {{"capability": "describe_scene", "args": {{}}}}], "success": "the user knows what is on the table", "reason": "observation only"}}"""

ANSWER_PROMPT = """These objects were measured on the table by the robot's camera (number: description, position):
{facts}
Answer the user's question using only these objects. Answer with ONE JSON object only:
{{"answer": "<short answer>", "object_ids": [<numbers of the objects your answer refers to>]}}
Question: {question}"""

DESCRIBE_PROMPT = """Each object in the image is outlined and numbered. The numbers are: {ids}.
For EVERY number, describe the object inside that outline as you see it. Answer with ONE JSON object only:
{{"objects": [{{"id": <number>, "name": "<short name>", "colour": "<colour>", "shape": "<shape>", "material": "<material>"}}, ...]}}
Include every number exactly once and no other numbers."""

SHORTLIST_PROMPT = """The picture shows {n} objects, each in its own tile with its number written above it.
For each phrase below, list EVERY numbered object that could be what the phrase describes. Include
doubtful ones; leave out only objects that clearly do not fit. A list may be empty.
{phrases}
Answer with ONE JSON object only: {{"lists": {{{keys}}}}}"""

DESCRIBE_ONE_PROMPT = """This picture shows one object at full brightness; everything around it is dimmed.
Describe that object as you see it. Answer with ONE JSON object only:
{"name": "<short name>", "colour": "<main colour>", "shape": "<shape>", "material": "<material>"}"""

RESOLVE_PROMPT = """The objects in the image are outlined and numbered. Their descriptions:
{descriptions}
Which ONE numbered object does the user mean by "{text}"?
If exactly one object fits, give its number. If several fit equally well, or none fits, set "id" to null,
"ambiguous" to true and list the numbers that could fit in "candidates". "id" is always ONE number or
null, never a list. Answer with ONE JSON object only:
{{"id": <number or null>, "ambiguous": <true|false>, "candidates": [<numbers>], "reason": "<short>"}}"""

COMPARE_PROMPT = """The picture shows {n} objects side by side; the number of each object is written above it.
Which ONE of them is "{text}"? Compare them with each other. If one fits clearly better than the others,
give its number. If several fit equally well or none fits, set "id" to null, "ambiguous" to true and
list the numbers that could fit in "candidates". "id" is always ONE number or null, never a list.
Answer with ONE JSON object only:
{{"id": <number or null>, "ambiguous": <true|false>, "candidates": [<numbers>], "reason": "<short>"}}"""

CONFIRM_PROMPT = """Look at the object at full brightness in this picture (everything around it is dimmed). Is it "{text}"?
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
                if isinstance(exc, ValidationError):  # say WHICH field is wrong and why (the retry needs it)
                    error = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'answer'}: {e['msg']}" for e in exc.errors())[:300]
                else:
                    error = str(exc).splitlines()[0][:300]
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

    def plan_goal(self, instruction: str, menu: str, max_steps: int = 6) -> GoalPlan:
        """Goal -> capability plan (one call; replaces `parse`). Validated deterministically: known capability
        names (schema enum), argument shapes per capability, step count, the parse consistency rules."""
        def check(p: GoalPlan):
            if len(p.steps) > max_steps:
                raise ValueError(f"at most {max_steps} steps")
            for i, st in enumerate(p.steps):
                if st.capability == "pick_place":
                    a = PickPlaceArgs.model_validate(st.args)
                    if a.object.anchor and a.object.anchor.strip().lower() == a.reference.text.strip().lower() \
                            and a.object.relation == a.relation:
                        raise ValueError("object.relation / object.anchor repeat the destination; keep them null "
                                         "unless the user describes where the object to move is now")
                    st.args = a.model_dump()
                elif st.capability == "answer" and not str(st.args.get("question", "")).strip():
                    raise ValueError(f"step {i + 1}: answer needs args.question")
        return self._call("plan", GoalPlan, f"Instruction: {instruction}", system=PLAN_SYSTEM.format(menu=menu),
                          check=check, max_new_tokens=600)

    def answer(self, question: str, facts: str, ids: list[int]) -> SceneAnswer:
        def check(a: SceneAnswer):
            bad = sorted(set(a.object_ids) - set(ids))
            if bad:
                raise ValueError(f"object_ids {bad} are not in the list")
        return self._call("answer", SceneAnswer, ANSWER_PROMPT.format(facts=facts, question=question), check=check,
                          max_new_tokens=200)

    def describe(self, som_image: np.ndarray, ids: list[int]) -> Descriptions:
        def check(d: Descriptions):
            got = sorted(o.id for o in d.objects)
            if got != sorted(ids):
                raise ValueError(f"ids {got} do not match the drawn ids {sorted(ids)}")
        return self._call("describe", Descriptions, DESCRIBE_PROMPT.format(ids=", ".join(map(str, ids))), image=som_image, check=check,
                          max_new_tokens=80 * len(ids) + 300)

    def shortlist(self, grid_image: np.ndarray, phrases: dict[str, str], ids: list[int]) -> dict[str, list[int]]:
        """One call for all objects: per phrase, the numbered objects that could match (high recall)."""
        def check(sl: Shortlist):
            if set(sl.lists) != set(phrases):
                raise ValueError(f"keys {sorted(sl.lists)} must be exactly {sorted(phrases)}")
            bad = sorted({i for v in sl.lists.values() for i in v} - set(ids))
            if bad:
                raise ValueError(f"numbers {bad} are not in the picture")
        text = SHORTLIST_PROMPT.format(n=len(ids), phrases="\n".join(f'{k}: "{v}"' for k, v in phrases.items()),
                                       keys=", ".join(f'"{k}": [<numbers>]' for k in phrases))
        return self._call("shortlist", Shortlist, text, image=grid_image, check=check, max_new_tokens=60 + 40 * len(phrases)).lists

    def describe_many(self, crops: dict[int, np.ndarray], batch: int = 8) -> dict[int, ObjectDescription]:
        """describe_one for many crops: batched calls with the identical prompt (if the VLM supports it);
        an answer that fails validation gets the usual single retry on its own."""
        if not hasattr(self.vlm, "generate_batch"):
            return {k: self.describe_one(c) for k, c in crops.items()}
        out, keys = {}, list(crops)
        for i in range(0, len(keys), batch):
            ks = keys[i:i + batch]
            t0 = time.monotonic()
            raws = self.vlm.generate_batch(DESCRIBE_ONE_PROMPT, [crops[k] for k in ks], max_new_tokens=120)
            dt = (time.monotonic() - t0) / len(ks)
            for k, raw in zip(ks, raws):
                try:
                    out[k] = ObjectDescription.model_validate(_json(raw))
                    if self.log:
                        self.log("vlm", role="describe", attempt=1, prompt=DESCRIBE_ONE_PROMPT, answer=raw, valid=True,
                                 seconds=round(dt, 2), batched=len(ks))
                except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                    if self.log:
                        self.log("vlm", role="describe", attempt=1, prompt=DESCRIBE_ONE_PROMPT, answer=raw, valid=False,
                                 error=str(exc).splitlines()[0][:200], seconds=round(dt, 2), batched=len(ks))
                    out[k] = self._retry_one(crops[k], str(exc).splitlines()[0][:200])  # the one retry
        return out

    def _retry_one(self, crop, error: str) -> ObjectDescription:
        prompt = f"{DESCRIBE_ONE_PROMPT}\n\nYour previous answer was not valid ({error}). Answer again with only the JSON."
        t0 = time.monotonic()
        raw = self.vlm.generate(prompt, image=crop, max_new_tokens=120, thinking=self.thinking)
        try:
            d = ObjectDescription.model_validate(_json(raw))
            ok, err = True, None
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            d, ok, err = None, False, str(exc).splitlines()[0][:200]
        if self.log:
            self.log("vlm", role="describe", attempt=2, prompt=prompt, answer=raw, valid=ok, error=err,
                     seconds=round(time.monotonic() - t0, 2))
        if d is None:
            raise VLMFailure(f"describe: invalid answer after one retry ({err})")
        return d

    def describe_one(self, crop_image: np.ndarray) -> ObjectDescription:
        return self._call("describe", ObjectDescription, DESCRIBE_ONE_PROMPT, image=crop_image, max_new_tokens=120)

    def resolve(self, som_image: np.ndarray, descriptions: list[dict], text: str, ids: list[int]) -> Resolution:
        def check(r: Resolution):
            if r.id is not None and r.id not in ids:
                raise ValueError(f"id {r.id} is not one of the drawn ids")
            if r.id is None and not r.ambiguous:
                raise ValueError("id is null but ambiguous is false")
        desc = "\n".join(f"{d['id']}: {d.get('name', '')} ({d.get('colour', '')}, {d.get('shape', '')}, {d.get('material', '')})"
                         for d in descriptions)
        return self._call("resolve", Resolution, RESOLVE_PROMPT.format(descriptions=desc, text=text), image=som_image, check=check)

    def compare(self, montage_image: np.ndarray, text: str, ids: list[int]) -> Resolution:
        def check(r: Resolution):
            if r.id is not None and r.id not in ids:
                raise ValueError(f"id {r.id} is not one of the shown numbers {ids}")
            if r.id is None and not r.ambiguous:
                raise ValueError("id is null but ambiguous is false")
        return self._call("compare", Resolution, COMPARE_PROMPT.format(n=len(ids), text=text), image=montage_image, check=check)

    def confirm(self, crop_image: np.ndarray, text: str) -> Confirmation:
        return self._call("confirm", Confirmation, CONFIRM_PROMPT.format(text=text), image=crop_image)
