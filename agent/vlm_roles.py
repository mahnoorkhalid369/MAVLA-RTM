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
        out = self.model.generate(text, image=image, system=system, max_new_tokens=max_new_tokens, thinking=thinking)
        try:  # instrumentation only: text tokens in / out (image tokens are a fixed per-image cost, counted as images)
            tok = self.model.processor.tokenizer
            self.last_tokens = {"prompt": len(tok.encode((system or "") + text)), "answer": len(tok.encode(out)),
                                "images": int(image is not None)}
        except Exception:
            self.last_tokens = None
        return out


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


HIDING_WORDS = re.compile(r"\b(behind|under|beneath|below|inside)\b", re.I)  # English prepositions of hiding
# A location phrase in the user's request ("in the buffer tray", "at the front of the incoming rack"): its anchor must
# survive into the plan's steps (validation only: a missing anchor is rejected and asked again, never invented).
LOCATION_PHRASE = re.compile(
    r"\b(?:at the front of|at the back of|in front of|on top of|next to|left of|right of|inside|behind|beside|near|in|on)"
    r"\s+(?:the|a|an)\s+([a-z0-9][a-z0-9\- ]*?)(?=\s*(?:[,.?!;]|$|\band\b|\bthat\b|\bwhich\b|\bwith\b|\bfrom\b|\bto\b|\bfor\b))",
    re.I)
WORKSPACE_WORDS = {"table", "bench", "workspace", "floor", "scene", "work surface"}  # the whole scene, not an anchor


def location_anchors(instruction: str) -> list[str]:
    return [m.group(1).strip().lower() for m in LOCATION_PHRASE.finditer(instruction or "")
            if m.group(1).strip().lower() not in WORKSPACE_WORDS]


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

PLAN_SYSTEM = """You plan for a robot arm with a gripper and two cameras, looking at a table with objects on it.
Work out what the user wants to be TRUE at the end, then list the steps that get there. Each step uses ONE of:
{menu}
Rules: copy the user's own words for objects. "pick" only grasps and holds (no destination). Use pick_place only when
the user gives a destination; if the robot already holds the object, use "place" (no new grasp). Words like "it" /
"that" may name the held or last-handled object: write "it". When the goal involves several objects chosen by a
property (all <A>s, the largest, sort, arrange, line up), use "observe" then "decompose" with the goal.
Only steps the user asked for. If no capability can achieve it, return "steps": [] and say what is missing in "reason".
Answer with ONE JSON object only. Each step is one flat object with "do" and its own keys:
{{"goal": "<one line>", "steps": [...], "reason": "<short>"}}
Examples:
"pick up the <A>" -> {{"goal": "hold <A>", "steps": [{{"do": "pick", "object": "<A>"}}], "reason": "grasp only"}}
"put the <A> on the <B>" -> {{"goal": "<A> on <B>", "steps": [{{"do": "pick_place", "object": "<A>", "relation": "on", "target": "<B>"}}], "reason": "one move"}}
(holding something) "put it next to the <B>" -> {{"goal": "held object next to <B>", "steps": [{{"do": "place", "object": "it", "relation": "next_to", "target": "<B>"}}], "reason": "already held"}}
"what is on the table?" -> {{"goal": "report the scene", "steps": [{{"do": "observe"}}, {{"do": "describe_scene"}}], "reason": "look and tell"}}
"is there a <A> left of the <B>?" -> {{"goal": "answer", "steps": [{{"do": "observe"}}, {{"do": "answer", "question": "is there a <A> left of the <B>?"}}], "reason": "look and answer"}}
"where is the <A>?" -> {{"goal": "find <A>", "steps": [{{"do": "observe"}}, {{"do": "locate", "object": "<A>"}}], "reason": "look and find"}}
"move closer and inspect the <A>" -> {{"goal": "close look at <A>", "steps": [{{"do": "observe"}}, {{"do": "inspect", "object": "<A>"}}], "reason": "close-up"}}
"inspect the <A> in the <B>" -> {{"goal": "close look at <A> in <B>", "steps": [{{"do": "observe"}}, {{"do": "inspect", "object": "<A>", "object_relation": "in", "object_anchor": "<B>"}}], "reason": "close-up of the one in <B>"}}
"find the thing behind the <B>" -> {{"goal": "find what is behind <B>", "steps": [{{"do": "observe"}}, {{"do": "explore", "object": "thing", "near": "<B>"}}, {{"do": "locate", "object": "thing"}}], "reason": "hidden region"}}
"line up all <A> by size" -> {{"goal": "<A> in a row by size", "steps": [{{"do": "observe"}}, {{"do": "decompose", "goal": "line up all <A> by size"}}], "reason": "depends on the measured objects"}}"""

DECOMPOSE_PROMPT = """Goal: {goal}
The robot measured these objects on the table (#id: description; position in mm, x = robot's right, y = forward;
footprint and height in mm). Descriptions are a vision model's words (may be imperfect); sizes and positions are measured.
{facts}
{context}
Plan the steps that make the goal true, referring to objects ONLY as "#<id>". Available steps: pick_place (object,
relation, target), pick (object), place (relation, target: the held object), locate / inspect (object).
Relations: on, next_to, left_of, right_of, in_front_of, behind. Also list the final conditions to check:
holding, on, next_to, left_of, right_of, in_front_of, behind (args ["#a", "#b"]), located, observed, or
ordered (args = all the "#ids" in order, with "by": size | height | width and "axis": x | y).
If the goal cannot be achieved with these steps, return "steps": [] and say why in "reason".
Answer with ONE JSON object only:
{{"steps": [{{"do": "pick_place", "object": "#3", "relation": "right_of", "target": "#5"}}, ...], "goal": [{{"pred": "...", "args": ["#3", "#5"]}}], "reason": "<short>"}}"""

ANSWER_PROMPT = """These objects were measured on the table by the robot's camera (number: description, position):
{facts}
Answer the user's question using only these objects. If the descriptions do not settle it (e.g. the shape or
colour of a small object is unclear), say so: set "certain" to false and list in "check_closer" the objects a
closer look could settle. Answer with ONE JSON object only:
{{"answer": "<short answer>", "object_ids": [<numbers your answer refers to>], "certain": true|false, "check_closer": [<numbers>]}}
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


class RequestKind(BaseModel):
    kind: Literal["question", "description", "action"]
    evidence: str = ""


REQUEST_SYSTEM = """Read the user's request to a robot arm and say what KIND of request it is. Use only the user's words.
"question": the user wants information (an answer): which, what, how many, where, is there ...
"description": the user wants the robot to look and report or describe, or to look closer / inspect something.
"action": the user asks the robot to CHANGE the world: move, pick up, put, place, stack, sort, arrange, hand over objects.
A question that asks the robot to do something ("can you put the <A> on the <B>?") is an action.
"evidence": copy, word for word, the part of the request that shows the kind.
Answer with ONE JSON object only: {"kind": "question" | "description" | "action", "evidence": "<words copied from the request>"}"""


class Roles:
    def __init__(self, vlm: VLM, log=None, thinking: bool = False) -> None:
        self.vlm, self.log, self.thinking = vlm, log, thinking
        self.calls, self.tokens = 0, {"prompt": 0, "answer": 0, "images": 0}  # instrumentation (per session)

    def _call(self, role: str, schema, text: str, image=None, system=None, check=None, max_new_tokens: int = 400):
        error = None
        for attempt in (1, 2):  # one retry with the validation error
            prompt = text if error is None else f"{text}\n\nYour previous answer was not valid ({error}). Answer again with only the JSON."
            t0 = time.monotonic()
            raw = self.vlm.generate(prompt, image=image, system=system, max_new_tokens=max_new_tokens, thinking=self.thinking)
            dt = time.monotonic() - t0
            self.calls += 1
            tk = getattr(self.vlm, "last_tokens", None) or {}
            for k in self.tokens:
                self.tokens[k] += int(tk.get(k, 0))
            try:
                obj = schema.model_validate(_json(raw))
                if check is not None:
                    check(obj)
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, prompt=prompt, answer=raw, valid=True, seconds=round(dt, 2), tokens=tk)
                return obj
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                if isinstance(exc, ValidationError):  # say WHICH field is wrong and why (the retry needs it)
                    error = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'answer'}: {e['msg']}" for e in exc.errors())[:300]
                else:
                    error = str(exc).splitlines()[0][:300]
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, prompt=prompt, answer=raw, valid=False, error=error, seconds=round(dt, 2), tokens=tk)
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

    def classify_request(self, instruction: str) -> RequestKind:
        """What the request asks for (agent/authority.py turns it into permissions). Sees only the user's words."""
        def check(r: RequestKind):
            if r.kind == "action" and not r.evidence.strip():
                raise ValueError("an action needs the user's words that ask for it in evidence")
        return self._call("request", RequestKind, f"Request: {instruction}", system=REQUEST_SYSTEM, check=check, max_new_tokens=80)

    def plan_goal(self, instruction: str, menu: str, max_steps: int = 6, context: str = "") -> GoalPlan:
        """Goal -> capability plan (one call). Validated deterministically: capability names (schema enum), the keys
        each capability needs (placeholder strings like "null" count as missing), step count, consistency rules."""
        def check(p: GoalPlan):
            if len(p.steps) > max_steps:
                raise ValueError(f"at most {max_steps} steps")
            for i, st in enumerate(p.steps, 1):
                if st.do == "pick_place":
                    if not (st.object and st.relation and st.target):
                        raise ValueError(f"step {i}: pick_place needs object, relation and a real target object; with no "
                                         "destination use pick")
                    if st.object_anchor and st.object_anchor.strip().lower() == st.target.strip().lower() \
                            and st.object_relation == st.relation:
                        raise ValueError(f"step {i}: object_relation / object_anchor repeat the destination; leave them "
                                         "out unless the user describes where the object to move is now")
                    if bool(st.object_anchor) != bool(st.object_relation):
                        raise ValueError(f"step {i}: object_relation and object_anchor go together")
                elif st.do == "pick" and not st.object:
                    raise ValueError(f"step {i}: pick needs object")
                elif st.do == "place" and not (st.relation and st.target):
                    raise ValueError(f"step {i}: place needs relation and a real target object")
                elif st.do == "decompose" and not st.goal:
                    raise ValueError(f"step {i}: decompose needs goal")
                elif st.do == "explore" and not st.near and HIDING_WORDS.search(instruction):
                    raise ValueError(f"step {i}: the instruction says where the thing is hidden; set explore.object to "
                                     "the thing to find and explore.near to the object it is behind / under")
                elif st.do in ("locate", "inspect", "explore") and not st.object:
                    raise ValueError(f"step {i}: {st.do} needs object")
                elif st.do == "answer" and not (st.question or "").strip():
                    raise ValueError(f"step {i}: answer needs question")
            said = " ".join(str(x or "") for st in p.steps for x in (st.object, st.object_anchor, st.target, st.near,
                                                                    st.question, st.goal)).lower()
            lost = [a for a in location_anchors(instruction) if a not in said]
            if lost and p.steps:
                raise ValueError(f"the user located the object with {lost}: keep it in the step (object_relation + "
                                 "object_anchor, e.g. \"object_relation\": \"in\", \"object_anchor\": \"" + lost[0] + "\")")
        text = f"{context}\nInstruction: {instruction}" if context else f"Instruction: {instruction}"
        return self._call("plan", GoalPlan, text, system=PLAN_SYSTEM.format(menu=menu), check=check, max_new_tokens=300)

    def decompose(self, goal: str, facts: str, ids: list[int], context: str = "", max_steps: int = 8) -> SubPlan:
        """Planning over the measured scene: steps and final conditions referring to world objects as #id."""
        import re as _re
        valid = {f"#{i}" for i in ids}

        def check(p: SubPlan):
            if len(p.steps) > max_steps:
                raise ValueError(f"at most {max_steps} steps")
            refs = [x for st in p.steps for x in (st.object, st.target) if x] + [a for c in p.goal for a in c.args]
            bad = sorted({r for r in refs if not _re.fullmatch(r"#\d+", r) or r not in valid})
            if bad:
                raise ValueError(f"unknown references {bad}: use only the listed #ids")
            for st in p.steps:
                if st.do not in ("pick_place", "pick", "place", "locate", "inspect"):
                    raise ValueError(f"step '{st.do}' is not available here")
                if st.do == "pick_place" and not (st.object and st.relation and st.target):
                    raise ValueError("pick_place needs object, relation, target")
        return self._call("decompose", SubPlan, DECOMPOSE_PROMPT.format(goal=goal, facts=facts, context=context),
                          check=check, max_new_tokens=500)

    def answer(self, question: str, facts: str, ids: list[int]) -> SceneAnswer:
        def check(a: SceneAnswer):
            bad = sorted(set(a.object_ids) - set(ids))
            if bad:
                raise ValueError(f"object_ids {bad} are not in the list")
            num = re.fullmatch(r"\s*(\d+)\b.*", a.answer)  # a count must agree with the objects it cites
            if num and a.object_ids and int(num.group(1)) != len(set(a.object_ids)):
                raise ValueError(f"the answer says {num.group(1)} but lists {len(set(a.object_ids))} objects; "
                                 "count again and list exactly the objects you counted")
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
        self.calls += 1
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
