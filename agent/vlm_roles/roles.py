"""Roles: one method per question asked of Gemma (classify, plan, decompose, answer, describe, resolve, compare,
confirm), each with a strict schema, a deterministic check and one retry with the validation error."""

from __future__ import annotations

import json
import re
import time

import numpy as np
from pydantic import ValidationError

from agent.vlm_roles.gemma import VLM
from agent.vlm_roles.prompts import (ANSWER_PROMPT, COMPARE_PROMPT, CONFIRM_PROMPT, DECIDE_SYSTEM, DECOMPOSE_PROMPT,
                                     DESCRIBE_ONE_PROMPT, DESCRIBE_PROMPT, IDENTITY_SYSTEM, PARSE_SYSTEM, PLAN_SYSTEM,
                                     REQUEST_SYSTEM, RESOLVE_PROMPT, SHORTLIST_PROMPT)
from agent.vlm_roles.schemas import (Confirmation, Decision, Descriptions, GoalPlan, ObjectDescription, PlanStep, RequestKind,
                                     Resolution, SceneAnswer, Shortlist, SubPlan, TaskSpec, VLMFailure)

def check_decision(d: Decision, has_pending: bool) -> None:
    """Structural validation of a closed-loop decision (no language rules): raises ValueError with the reason."""
    if d.action == "replan":
        if not d.steps:
            raise ValueError("replan needs the remaining steps")
        check_steps(d.steps)
    if d.action == "continue" and not has_pending:
        raise ValueError("no steps remain: choose replan, finish, ask_user or stop")
    if d.action == "ask_user" and not (d.question or "").strip():
        raise ValueError("ask_user needs a question")


def check_steps(steps: list[PlanStep], max_steps: int = 6) -> None:
    """Structural validation of planned steps (schema-level: each capability gets the arguments it needs); raises
    ValueError with what to fix, which Gemma sees on its one retry. No interpretation of the user's words."""
    if len(steps) > max_steps:
        raise ValueError(f"at most {max_steps} steps")
    for i, st in enumerate(steps, 1):
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
        elif st.do in ("locate", "inspect", "explore") and not st.object:
            raise ValueError(f"step {i}: {st.do} needs object")
        elif st.do == "answer" and not (st.question or "").strip():
            raise ValueError(f"step {i}: answer needs question")
        elif st.do == "look" and not st.direction:
            raise ValueError(f"step {i}: look needs direction (left, right, forward, backward or down) and frame")


# Roles that reason as the robot (what it is, what it can do, its directions) get IDENTITY_SYSTEM in front of their
# system prompt. The perception roles (describe, shortlist, resolve, compare, confirm) judge images only: no identity.
EMBODIED_ROLES = {"request", "plan", "decide", "decompose", "answer"}


def _json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in the answer")
    s = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[s:end + 1])


class Roles:
    def __init__(self, vlm: VLM, log=None, thinking: bool = False) -> None:
        self.vlm, self.log, self.thinking = vlm, log, thinking
        self.calls, self.tokens = 0, {"prompt": 0, "answer": 0, "images": 0}  # instrumentation (per session)

    def _call(self, role: str, schema, text: str, image=None, system=None, check=None, max_new_tokens: int = 400):
        if role in EMBODIED_ROLES:  # the robot's identity, once, in front of the role's own instructions
            system = IDENTITY_SYSTEM + ("\n\n" + system if system else "")
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
            check_steps(p.steps, max_steps)
        text = f"{context}\nInstruction: {instruction}" if context else f"Instruction: {instruction}"
        return self._call("plan", GoalPlan, text, system=PLAN_SYSTEM.format(menu=menu), check=check, max_new_tokens=300)

    def decide(self, goal: str, progress: str, event: str, menu: str, has_pending: bool, context: str = "") -> Decision:
        """The supervisor's next decision in the closed loop, from the goal, the progress (task ledger) and the latest
        event (a worker result / a check / the goal not shown). Schema-validated; replan steps are checked like a plan."""
        def check(d: Decision):
            check_decision(d, has_pending)
        text = "\n".join(x for x in (context, f"Goal: {goal}", progress, f"Latest: {event}") if x)
        return self._call("decide", Decision, text, system=DECIDE_SYSTEM.format(menu=menu), check=check, max_new_tokens=350)

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
