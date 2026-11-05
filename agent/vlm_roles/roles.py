"""Roles: one method per question asked of Gemma (classify, plan, decompose, answer, describe, resolve,
confirm), each with a strict schema, a deterministic check and one retry with the validation error."""

from __future__ import annotations

import json
import re
import time

import numpy as np
from pydantic import ValidationError

from agent.vlm_roles.gemma import VLM
from agent.vlm_roles.prompts import (ANSWER_PROMPT, CLARIFY_SYSTEM, CONFIRM_PROMPT, DECIDE_SYSTEM, DECOMPOSE_PROMPT, DESCRIBE_ONE_PROMPT,
                                     DESCRIBE_PROMPT, IDENTITY_SYSTEM, PLAN_SYSTEM, REQUEST_SYSTEM, RESOLVE_PROMPT,
                                     SHORTLIST_PROMPT, REFERENCE_PROMPT, APPEARANCE_PROMPT,
                                     APPEARANCE_ONE_PROMPT)
from agent.vlm_roles.schemas import (AppearanceMatch, Clarification, Confirmation, Decision, Descriptions, GoalPlan, ObjectDescription,
                                     PlanStep, ReferenceSpec, RequestKind, Resolution, SceneAnswer, Shortlist, SubPlan,
                                     VLMFailure)


def check_decision(d: Decision, has_pending: bool) -> None:
    """Structural validation of a closed-loop decision (no language rules): raises ValueError with the reason."""
    if d.action == "replan":
        if not d.steps:
            raise ValueError("replan needs the remaining steps")
        check_steps(d.steps)
    if d.action == "continue" and not has_pending:
        if not d.steps:
            raise ValueError("no steps remain: choose replan, finish, ask_user or stop")
        check_steps(d.steps)  # "continue" with new steps when none remain: the supervisor takes them as a replan
    if d.action == "ask_user" and not (d.question or "").strip():
        raise ValueError("ask_user needs a question")
    if d.action == "new_goal" and not (d.request or "").strip():
        raise ValueError("new_goal needs the user's new request in \"request\"")


def check_binds(binds, candidates: dict, bound: dict | None = None) -> None:
    """A BIND settles goal words Gemma was shown as unsettled, with one of the candidates shown for them; repeating a
    binding the task already has (same words, same id) is a no-op and accepted (live 2026-11-04: the restated
    "<words> = #9" was rejected and the decision failed); another id for bound words is not a change it can make.
    Structural: the words are compared as given (they are the task's own reference text), never interpreted."""
    bound = bound or {}
    for b in binds:
        if b.text in bound:
            if b.id.strip() != f"#{bound[b.text]}":
                raise ValueError(f"BIND {b.text!r} = {b.id}: those words are already bound to #{bound[b.text]} in this "
                                 f"task; a binding is not changed (keep #{bound[b.text]}, or leave the BIND line out)")
            continue
        if b.text not in candidates:
            have = "; ".join(f'"{t}"' for t in candidates) or "none"
            raise ValueError(f"BIND {b.text!r}: not an unsettled reference of this task (unsettled: {have}); write the "
                             f"words exactly as listed")
        ids = candidates[b.text]
        if not re.fullmatch(r"#\d+", b.id.strip()) or int(b.id.strip()[1:]) not in ids:
            raise ValueError(f"BIND {b.text!r} = {b.id}: use one of its measured candidates: "
                             f"{', '.join(f'#{i}' for i in ids)}")


def check_steps(steps: list[PlanStep], max_steps: int = 6) -> None:
    """Structural validation of planned steps: a bounded plan, and each implemented operation's arguments through
    its own interface (a ValueError says what to fix; Gemma sees it on its one retry). An operation that is not
    implemented is not an error here: it stays in the plan and is reported unavailable when reached. No
    interpretation of the user's words."""
    if len(steps) > max_steps:
        raise ValueError(f"at most {max_steps} steps")
    from agent.capabilities.specs import check_worker  # the operations' interfaces (lazy: no import cycle)
    for i, st in enumerate(steps, 1):
        try:
            check_placeholders(st.args)
            check_worker(st.do, st.worker)
        except ValueError as exc:
            raise ValueError(f"step {i}: {exc}") from None


def check_relations_agree(steps, success, request: str = "") -> None:
    """A plan that puts something IN a destination must not state ON (or another relation) for that destination as its
    success, and the other way round (live 2026-10-26: plan "place relation=in <tray>", success "on(<block>, <tray>)";
    the block landed in the tray and the goal check, correctly, said "not on top"). Structural: the plan is
    compared with itself; Gemma fixes it on its retry."""
    from agent.capabilities.specs import PLACE_RELATIONS
    placed = {}
    for st in steps:
        raw = st.model_extra or {}
        if st.do in ("place", "pick_place") and raw.get("target") and raw.get("relation"):
            placed[str(raw["target"]).strip().lower()] = str(raw["relation"]).strip().lower()
    for c in success:
        if c.pred in PLACE_RELATIONS and len(c.args) > 1:
            rel = placed.get(str(c.args[1]).strip().lower())
            if rel is not None and rel != c.pred:
                asked = f' (the user asked: "{request}")' if request else ""
                fixed = f"{rel}({', '.join(c.args)})"
                raise ValueError(f"the steps put the object {rel} {c.args[1]!r} but the success criterion says "
                                 f"{c.pred}({', '.join(c.args)}){asked}. Keep the steps and write the criterion as "
                                 f"{fixed}, unless the steps are what is wrong (in = inside a container, on = on top of "
                                 f"a solid object)")


def check_end_state(steps, success) -> None:
    """Success criteria are checked on the FINAL state: a plan whose last gripper step puts the object down cannot also
    state holding(...) (live 2026-10-26: "holding(<block>); in(<block>, <tray>)"; the block was placed and
    verified in the tray, and the task failed on holding). Structural: the plan is compared with itself; Gemma fixes it
    on its retry."""
    last = next((st.do for st in reversed(steps) if st.do in ("pick", "place", "pick_place")), None)
    if last not in ("place", "pick_place"):
        return
    for c in success:
        if c.pred == "holding":
            raise ValueError(f"the success criteria describe the end state: after the last step ({last}) nothing is held, "
                             f"so holding({', '.join(c.args)}) can never be met. Keep the steps and remove that criterion")


def check_predicate(pred: str, args=None) -> None:
    """A success condition must be one the verification worker can evaluate (agent/goals.py PREDICATES); a free-text
    condition ("all <X> are in the <Y>") could never be shown and kept the goal unknown (live 2026-10-26). With args,
    it must also have the arguments the predicate needs ("holding()" names no object, live 2026-11-04)."""
    from agent.goals import MIN_ARGS, PREDICATES
    if pred not in PREDICATES:  # never mapped to a supported one here: Gemma chooses the replacement on its retry
        raise ValueError(f"{pred} is not a supported condition: it cannot be measured. Write each SUCCESS condition with "
                         f"one of these names: {', '.join(PREDICATES)}")
    if args is not None and len(args) < MIN_ARGS.get(pred, 0):
        raise ValueError(f"{pred}({', '.join(args)}) is malformed: {pred} needs {MIN_ARGS[pred]} argument(s)")


def check_region_args(conditions) -> None:
    """Region ids in conditions (config/layout.py): each must be a region of the layout, and a region is a place, so
    it is never what is held, located or moved (the first argument). Structural: ids, never a region's name."""
    from config import layout
    from agent.goals import OBJECT_ARGS
    have = ", ".join(layout.REGIONS) or "none"
    for c in conditions:
        for i, a in enumerate(c.args):
            if not layout.is_region_id(a):
                continue
            if layout.region(a) is None:
                raise ValueError(f"{c.pred}({', '.join(c.args)}): {a} is not a region of this work cell (its regions: {have})")
            if i == 0 and c.pred in OBJECT_ARGS:
                raise ValueError(f"{c.pred}({', '.join(c.args)}): {a} is a fixed region, not an object; a region can "
                                 f"only be where an object is (the second argument)")


def check_progress(steps, preds) -> None:
    """A plan whose conditions need an object to change (holding, on, in, ...) must contain an operation that changes
    an object (live 2026-11-04: a decomposition of a pick-and-place was only "observe"). Structural: the operations'
    declared effects are compared with the conditions' kind; which operation to use stays Gemma's choice."""
    from agent.capabilities.specs import EFFECT
    from agent.goals import OBJECT_STATE
    if any(p in OBJECT_STATE for p in preds) and not any(EFFECT.get(st.do) == "object_change" for st in steps):
        ops = sorted(k for k, v in EFFECT.items() if v == "object_change")
        raise ValueError(f"the steps ({', '.join(st.do for st in steps) or 'none'}) only sense: none of them can change "
                         f"an object, so the goal can never be reached. Plan the steps that change the objects "
                         f"(operations that change objects: {', '.join(ops)})")


_WRAPPED = re.compile(r"\s*<([^<>]{1,60})>\s*")


def unwrap_user_words(steps, conditions, words: str) -> None:
    """Gemma sometimes writes the user's own words inside the prompts' template brackets: "<yellow block>" (live
    2026-10-26: the plan was rejected twice as a placeholder). When the bracketed text occurs word for word in the
    user's instruction it IS the user's words: the brackets are removed (in place). Anything else in brackets stays a
    placeholder (check_placeholders rejects it). Structural: the words are compared, not interpreted."""
    low = (words or "").lower()

    def fix(v):
        m = _WRAPPED.fullmatch(v) if isinstance(v, str) else None
        return m.group(1).strip() if m and m.group(1).strip().lower() in low else v
    for st in steps or []:
        extra = st.model_extra or {}
        for k, v in list(extra.items()):
            extra[k] = fix(v)
    for c in conditions or []:
        c.args = [fix(a) for a in c.args]


_PLACEHOLDER = re.compile(r"<[^<>]{1,60}>")  # a template slot copied from a prompt, e.g. "<object_id>", "#<held id>"


def check_placeholders(value) -> None:
    """Gemma sometimes copies a prompt template's slot instead of filling it (live 2026-10-21: "#<held id>",
    "<object_id>", "<the user's words in brackets>"); such an argument would be matched to an arbitrary object. Rejected, so the one
    retry says what to write instead. Structural: no words are interpreted."""
    if isinstance(value, dict):
        for v in value.values():
            check_placeholders(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            check_placeholders(v)
    elif isinstance(value, str) and _PLACEHOLDER.search(value):
        raise ValueError(f"{value!r} is a template placeholder: write the user's own words for the object, or a #id "
                         f"given in the context")


# Roles that reason as the robot (what it is, what it can do, its directions) get IDENTITY_SYSTEM in front of their
# system prompt. The perception roles (describe, shortlist, resolve, confirm) judge images only: no identity.
EMBODIED_ROLES = {"request", "clarify", "plan", "decide", "decompose", "answer", "reference"}  # "reference": frames


class _LineError(ValueError):
    """A line-protocol answer whose structure is broken (not its content)."""


def _parser_input(text: str) -> str:
    """What the JSON parser reads: the answer without code fences, from its first "{" or "[" on."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in the answer")
    return text[min(starts):]


def _json(text: str):
    """The FIRST complete JSON value in the answer: code fences, text before it and anything after it (a second object,
    a remark) are ignored (live 2026-10-26: "Extra data" after a valid object cost a retry and then the task)."""
    value, _ = json.JSONDecoder().raw_decode(_parser_input(text))
    return value


# ----------------------------------------------------------------------- line protocol (plan, decompose, decide)
# Gemma answers these roles one labelled item per line (2026-11-04: its JSON plans were sensible but malformed):
#   GOAL: <final state>          SUCCESS: pred(arg, arg) | key=value          STEP: worker | operation | key=value
#   ACTION: / QUESTION: / REQUEST: / REASON: <text>
# The parser reads labels, "|", "key=value" and a predicate's arguments: serialization only. It never chooses or changes
# a worker, an operation, an argument or a condition; the result goes through the same schemas and checks as before.
LINE_LABELS = {"plan": ("GOAL", "SUCCESS", "STEP", "REASON"), "decompose": ("STEP", "SUCCESS", "REASON"),
               "decide": ("ACTION", "BIND", "STEP", "QUESTION", "REQUEST", "ANSWER", "REASON")}
_TEXT_LABELS = ("GOAL", "REASON", "QUESTION", "REQUEST", "ANSWER")
_SINGLE_LABELS = ("ACTION", "QUESTION", "REQUEST", "ANSWER", "GOAL")  # at most once in an answer  # free text: a following unlabelled line continues it
_LABEL = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?\**([A-Z_]+)\**\s*:\s*(.*)$")
_PRED = re.compile(r"^(\w+)\s*\((.*)\)$")


def _has_labels(role: str, text: str) -> bool:
    return any((m := _LABEL.match(line)) and m.group(1) in LINE_LABELS[role] for line in text.splitlines())


def _value(v: str):
    """One value as written: surrounding quotes dropped; a [list] read as a list (a direction)."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    if v.startswith("[") and v.endswith("]"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return v


def _pairs(parts, where: str) -> dict:
    out = {}
    for part in parts:
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"{where}: {part.strip()!r} is not key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = _value(v)
    return out


def _step_line(v: str, n: int) -> dict:
    parts = v.split("|")
    if len(parts) < 2 or "=" in parts[0] or "=" in parts[1] or not parts[1].strip():
        raise ValueError(f"line {n}: a STEP line is: worker | operation | key=value | ... (got {v.strip()!r})")
    return {"worker": parts[0].strip(), "do": parts[1].strip(), **_pairs(parts[2:], f"line {n}")}


def _condition_line(v: str, n: int) -> dict:
    head, *rest = v.split("|")
    m = _PRED.match(head.strip())
    if not m:
        raise ValueError(f"line {n}: a SUCCESS line is a condition with its arguments, such as holding(x) "
                         f"(got {head.strip()!r})")
    from agent.goals import ARITY
    inner, n_args = m.group(2).strip(), ARITY.get(m.group(1))
    if not inner:
        parts = []
    elif n_args == 1:  # the whole text is the one argument (the user's words may contain commas)
        parts = [inner]
    elif n_args == 2:  # two arguments: the destination after the last comma
        parts = inner.rsplit(",", 1)
    else:
        parts = inner.split(",")
    args = [str(_value(a)) for a in parts if a.strip()]
    return {"pred": m.group(1), "args": args, **_pairs(rest, f"line {n}")}


def _bind_line(v: str, n: int) -> dict:
    if "=" not in v:
        raise ValueError(f"line {n}: a BIND line is: the user's words = #id (got {v.strip()!r})")
    text, wid = v.rsplit("=", 1)
    return {"text": str(_value(text)), "id": str(_value(wid))}


def parse_lines(role: str, text: str) -> tuple[str, dict]:
    """Gemma's line-protocol answer -> (the lines the parser read, the fields of the role's schema). ValueError says
    which line breaks the format (structure only)."""
    body = re.sub(r"```\w*", "", text).strip()
    labels, items = LINE_LABELS[role], []
    for n, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        m = _LABEL.match(line)
        if m and m.group(1) in labels:
            items.append([m.group(1), m.group(2).strip(), n])
        elif m and m.group(1).isupper() and len(m.group(1)) > 1:
            raise ValueError(f"line {n}: {m.group(1)}: is not a label of this answer (the labels: {', '.join(labels)})")
        elif items and items[-1][0] in _TEXT_LABELS:
            items[-1][1] = f"{items[-1][1]} {line.strip()}".strip()
        else:
            raise ValueError(f"line {n} has no label ({line.strip()[:60]!r}); every line starts with one of: "
                             f"{', '.join(label + ':' for label in labels)}")
    if not items:
        raise ValueError(f"no labelled lines; answer with lines starting {', '.join(label + ':' for label in labels)}")
    for label in _SINGLE_LABELS:  # never joined, never one of them chosen: the answer is sent back (live 2026-11-04:
        lines = [n for k, _, n in items if k == label]  # "ACTION: finish ... Wait ... ACTION: replan" read as one)
        if len(lines) > 1:
            raise ValueError(f"line {lines[1]}: a second {label} line was written. An answer has exactly one {label}. "
                             f"Give only your final decision and do not write corrections or reasoning between the "
                             f"format lines")
    one = lambda label: " ".join(v for k, v, _ in items if k == label)  # noqa: E731
    steps = [_step_line(v, n) for k, v, n in items if k == "STEP"]
    conds = [_condition_line(v, n) for k, v, n in items if k == "SUCCESS"]
    if role == "plan":
        out = {"goal": one("GOAL"), "success": conds, "steps": steps, "reason": one("REASON")}
    elif role == "decompose":
        out = {"steps": steps, "goal": conds, "reason": one("REASON")}
    else:
        out = {"action": one("ACTION").strip().lower(), "steps": steps, "question": one("QUESTION") or None,
               "request": one("REQUEST") or None, "answer": one("ANSWER") or None, "reason": one("REASON"),
               "binds": [_bind_line(v, n) for k, v, n in items if k == "BIND"]}
    return body, out


def line_feedback(role: str, error: str) -> str:
    return (f"Your previous answer did not follow the line format: {error}.\nReturn the same intended answer in the line "
            f"format only, one item per line starting with {', '.join(label + ':' for label in LINE_LABELS[role])}; "
            f"a STEP line is worker | operation | key=value | ..., a SUCCESS line is a condition such as holding(x). "
            f"No other text.")


_CLOSER = {"{": "}", "[": "]"}


def _bracket_mismatch(src: str) -> str:
    """The first closing bracket that does not close the bracket opened last (strings skipped), described by position:
    a structural fact about the text, nothing about what it means. "" when the brackets pair up."""
    stack, in_str, esc = [], False, False
    line, col = 1, 0
    for ch in src:
        col += 1
        if ch == "\n":
            line, col = line + 1, 0
        if in_str:
            esc, in_str = (not esc and ch == "\\"), in_str and not (ch == '"' and not esc)
            continue
        if ch == '"':
            in_str, esc = True, False
        elif ch in _CLOSER:
            stack.append((ch, line, col))
        elif ch in "}]":
            if not stack:
                return f'"{ch}" at line {line} column {col} closes nothing.'
            o, ol, oc = stack.pop()
            if _CLOSER[o] != ch:
                kind = "object" if o == "{" else "array"
                return (f'A JSON {kind} opened with "{o}" at line {ol} column {oc} must be closed with "{_CLOSER[o]}", '
                        f'but "{ch}" closes it at line {line} column {col}.')
    return ""


def json_error(text: str, exc: json.JSONDecodeError) -> dict:
    """A JSON parse error of Gemma's answer, located in the parser's input: message, line, column, offset, the lines
    around it (the failing line marked) and the bracket that is closed wrongly, if one is."""
    src = exc.doc if isinstance(getattr(exc, "doc", None), str) else _parser_input(text)
    lines = src.splitlines()
    lo, hi = max(exc.lineno - 4, 0), min(exc.lineno + 1, len(lines))
    near = "\n".join(f"{'>>' if k + 1 == exc.lineno else '  '} {lines[k]}" for k in range(lo, hi))
    return {"message": exc.msg, "line": exc.lineno, "column": exc.colno, "offset": exc.pos, "context": near,
            "mismatch": _bracket_mismatch(src)}


def json_feedback(err: dict) -> str:
    """The retry's feedback for an answer that is not valid JSON: where the parser failed and what is structurally
    wrong there. It names no content: the plan stays Gemma's."""
    return "\n".join(x for x in (
        "Your previous response was not valid JSON.",
        f"Parser error: {err['message']} at line {err['line']} column {err['column']}.",
        f"Near the error (the line marked >>):\n{err['context']}",
        err["mismatch"],
        "Return the same intended answer, correcting the JSON structure only. Return one JSON object and no explanation.")
        if x)


class Roles:
    def __init__(self, vlm: VLM, log=None, thinking: bool = False) -> None:
        self.vlm, self.log, self.thinking = vlm, log, thinking
        self.calls, self.tokens = 0, {"prompt": 0, "answer": 0, "images": 0}  # instrumentation (per session)

    def _call(self, role: str, schema, text: str, image=None, system=None, check=None, max_new_tokens: int = 400):
        if role in EMBODIED_ROLES:  # the robot's identity, once, in front of the role's own instructions
            system = IDENTITY_SYSTEM + ("\n\n" + system if system else "")
        error = feedback = None
        for attempt in (1, 2):  # one retry with the validation error
            prompt = text if error is None else f"{text}\n\n{feedback}"
            t0 = time.monotonic()
            raw = self.vlm.generate(prompt, image=image, system=system, max_new_tokens=max_new_tokens, thinking=self.thinking)
            dt = time.monotonic() - t0
            self.calls += 1
            tk = getattr(self.vlm, "last_tokens", None) or {}
            for k in self.tokens:
                self.tokens[k] += int(tk.get(k, 0))
            # every stage is logged as it is: the raw answer, what the parser read, what it parsed, the validated
            # (normalized) result; nothing in the answer is repaired
            io = {"answer": raw}
            lines = role in LINE_LABELS and _has_labels(role, raw)  # the line protocol (an old JSON answer still reads)
            try:
                if role in LINE_LABELS and not lines and "{" not in raw:
                    raise _LineError(f"no labelled lines; answer with lines starting "
                                     f"{', '.join(label + ':' for label in LINE_LABELS[role])}")
                if lines:
                    try:
                        io["parser_input"], parsed = parse_lines(role, raw)
                    except ValueError as exc:
                        raise _LineError(str(exc)) from None
                else:
                    io["parser_input"] = _parser_input(raw)
                    parsed = _json(raw)
                io["format"] = "lines" if lines else "json"
                io["parsed"] = parsed
                obj = schema.model_validate(parsed)
                io["normalized"] = obj.model_dump(mode="json")
                if check is not None:
                    check(obj)
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, system=system, prompt=prompt, **io, valid=True, seconds=round(dt, 2), tokens=tk)
                return obj
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                if isinstance(exc, _LineError):  # the line format is broken: which line, and the format again
                    io["parse_error"] = {"message": str(exc)}
                    error = f"line format: {exc}"
                    feedback = line_feedback(role, str(exc))
                elif isinstance(exc, json.JSONDecodeError):  # not JSON: where and what is structurally wrong
                    io["parse_error"] = json_error(raw, exc)
                    error = f"not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
                    feedback = json_feedback(io["parse_error"])
                else:
                    if isinstance(exc, ValidationError):  # say WHICH field is wrong and why (the retry needs it)
                        error = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'answer'}: {e['msg']}" for e in exc.errors())[:300]
                    else:
                        error = str(exc).splitlines()[0][:300]
                    again = ("in the line format only" if lines else "with only the JSON")  # the format it answered in
                    feedback = f"Your previous answer was not valid ({error}). Answer again {again}."
                if self.log:
                    self.log("vlm", role=role, attempt=attempt, system=system, prompt=prompt, **io, valid=False, error=error, seconds=round(dt, 2), tokens=tk)
        raise VLMFailure(f"{role}: invalid answer after one retry ({error})")

    def classify_request(self, instruction: str) -> RequestKind:
        """What the request asks for (agent/authority.py turns it into permissions). Sees only the user's words."""
        def check(r: RequestKind):
            if r.kind == "action" and not r.evidence.strip():
                raise ValueError("an action needs the user's words that ask for it in evidence")
        return self._call("request", RequestKind, f"Request: {instruction}", system=REQUEST_SYSTEM, check=check, max_new_tokens=80)

    ALREADY_MET = ("Your SUCCESS conditions are already true before the requested action is performed. State what "
                   "measurable conditions must be true after the request is completed.")

    def plan_goal(self, instruction: str, menu: str, max_steps: int = 6, context: str = "",
                  already_met=None) -> GoalPlan:
        """Goal -> capability plan (one call). Validated deterministically: capability names (schema enum), the keys
        each capability needs (placeholder strings like "null" count as missing), step count, consistency rules.
        already_met: for a request that asks for a change, a callback (conditions -> True when every one of them is
        already measurably true now); such a goal asks for no change and is sent back on the retry (live 2026-11-04:
        "pick ..." with SUCCESS not_holding() completed after an observation). Which condition to write stays Gemma's."""
        def check(p: GoalPlan):
            unwrap_user_words(p.steps, p.success, instruction)
            check_steps(p.steps, max_steps)
            check_relations_agree(p.steps, p.success, instruction)
            check_end_state(p.steps, p.success)
            for c in p.success:
                try:
                    check_placeholders(c.model_dump().get("args"))
                    check_predicate(c.pred, c.args)  # only measurable conditions (live 2026-11-04: picked(...))
                    check_region_args([c])
                except ValueError as exc:
                    raise ValueError(f"success criterion {c.pred}: {exc}") from None
            if p.steps and not p.success:
                raise ValueError("state the success criteria (\"success\"): the measurable conditions that are true "
                                 "when the user's goal is reached")
            if already_met is not None and p.success and already_met([c.model_dump(exclude_none=True) for c in p.success]):
                raise ValueError(self.ALREADY_MET)
        text = f"{context}\nInstruction: {instruction}" if context else f"Instruction: {instruction}"
        return self._call("plan", GoalPlan, text, system=PLAN_SYSTEM.format(menu=menu), check=check, max_new_tokens=700)

    def clarify(self, request: str, conversation: list, context: str = "") -> Clarification:
        """Understanding before planning: is the request clear with what the robot knows and what the user has said so
        far? -> clear (restated completely) or one question for the user. conversation: [(question, answer)]."""
        def check(c: Clarification):
            if c.status == "ask" and not (c.question or "").strip():
                raise ValueError("ask needs the question in \"question\"")
            if c.status == "clear" and not c.request.strip():
                raise ValueError("clear needs the complete request in \"request\"")
        talk = "\n".join(f"You asked: {q}\nThe user answered: {a}" for q, a in conversation)
        text = "\n".join(x for x in (context, f"The user's request: {request}", talk and f"Conversation so far:\n{talk}") if x)
        return self._call("clarify", Clarification, text, system=CLARIFY_SYSTEM, check=check, max_new_tokens=300)

    def decide(self, goal: str, progress: str, event: str, menu: str, has_pending: bool, context: str = "",
               candidates: dict | None = None, bound: dict | None = None) -> Decision:
        """The supervisor's next decision in the closed loop, from the goal, the progress (task ledger) and the latest
        event (a worker result / a check / the goal not shown). Schema-validated; replan steps are checked like a plan.
        candidates: the goal's unsettled words -> the measured world ids shown to Gemma for them (a BIND must use one).
        bound: the task's established bindings (words -> world id): a BIND may restate one, never change it."""
        def check(d: Decision):
            unwrap_user_words(d.steps, [], goal)
            check_decision(d, has_pending)
            check_binds(d.binds, candidates or {}, bound)
        text = "\n".join(x for x in (context, f"Goal: {goal}", progress, f"Latest: {event}") if x)
        return self._call("decide", Decision, text, system=DECIDE_SYSTEM.format(menu=menu), check=check, max_new_tokens=600)

    def decompose(self, goal: str, facts: str, ids: list[int], context: str = "", max_steps: int = 8,
                  menu: str = "", task_preds=()) -> SubPlan:
        """Planning over the measured scene: steps and final conditions referring to world objects as #id and to the
        layout's fixed regions as R<n> (config/layout.py; in `facts`). task_preds: the predicates of the task's own
        success criteria (a decomposition must be able to reach them)."""
        import re as _re
        from config import layout
        valid = {f"#{i}" for i in ids}
        regions = set(layout.REGIONS)

        def check(p: SubPlan):
            if len(p.steps) > max_steps:
                raise ValueError(f"at most {max_steps} steps")
            if any(st.do == "decompose" for st in p.steps):  # live 2026-10-26: decompose -> decompose -> ... (a loop)
                raise ValueError("a decomposition lists the concrete steps (pick, place, ...); it cannot contain decompose")
            for c in p.goal:
                check_predicate(c.pred, c.args)
            refs = [str(st.raw.get(k)) for st in p.steps for k in ("object", "target") if st.raw.get(k)]
            bad = sorted({r for r in refs + [a for c in p.goal for a in c.args]
                          if not (_re.fullmatch(r"#\d+", r) and r in valid) and r.strip().upper() not in regions})
            if bad:
                have = f"; these region ids: {', '.join(sorted(regions, key=lambda r: int(r[1:])))}" if regions else ""
                raise ValueError(f"unknown references {bad}: use only these #ids: "
                                 f"{', '.join(sorted(valid, key=lambda r: int(r[1:])))}{have}")
            check_region_args(p.goal)
            check_steps(p.steps, max_steps)
            check_end_state(p.steps, p.goal)
            check_progress(p.steps, [c.pred for c in p.goal] + list(task_preds))
        return self._call("decompose", SubPlan, DECOMPOSE_PROMPT.format(goal=goal, facts=facts, context=context, menu=menu),
                          check=check, max_new_tokens=700)

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
                        self.log("vlm", role="describe", attempt=1, system=None, prompt=DESCRIBE_ONE_PROMPT, answer=raw, valid=True,
                                 seconds=round(dt, 2), batched=len(ks))
                except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                    if self.log:
                        self.log("vlm", role="describe", attempt=1, system=None, prompt=DESCRIBE_ONE_PROMPT, answer=raw, valid=False,
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
            self.log("vlm", role="describe", attempt=2, system=None, prompt=prompt, answer=raw, valid=ok, error=err,
                     seconds=round(time.monotonic() - t0, 2))
        if d is None:
            raise VLMFailure(f"describe: invalid answer after one retry ({err})")
        return d

    def describe_one(self, crop_image: np.ndarray) -> ObjectDescription:
        return self._call("describe", ObjectDescription, DESCRIBE_ONE_PROMPT, image=crop_image, max_new_tokens=120)

    def resolve(self, som_image: np.ndarray, descriptions: list[dict], text: str, ids: list[int],
                allowed: list[int] | None = None) -> Resolution:
        """allowed: every object numbered in the picture (default: the described ones). Gemma's answer stands when it
        names any of them: the described ones are only where to look first (live 2026-10-26: Gemma answered the tray,
        #13, correctly, and was rejected 8 times because #13 was not among the two described objects)."""
        valid = set(ids) | set(allowed or [])

        def check(r: Resolution):
            if r.id is not None and r.id not in valid:
                raise ValueError(f"id {r.id} is not one of the numbered objects")
            if r.id is None and not r.ambiguous:
                raise ValueError("id is null but ambiguous is false")
        desc = "\n".join(f"{d['id']}: {d.get('name', '')} ({d.get('colour', '')}, {d.get('shape', '')}, {d.get('material', '')})"
                         + (f"; measured: {d['measured']}" if d.get("measured") else "") for d in descriptions)
        return self._call("resolve", Resolution, RESOLVE_PROMPT.format(descriptions=desc, text=text), image=som_image, check=check)

    def parse_reference(self, text: str) -> ReferenceSpec:
        """The user's words for one object -> appearance (judged by Gemma on images), size and side (judged on
        measured geometry). Schema-checked; a side must be a valid direction in a known frame."""
        def check(r: ReferenceSpec):
            if r.side is not None:
                from agent.directions import direction_vector
                direction_vector(r.side.direction, r.side.frame)  # ValueError -> one retry with the reason
        return self._call("reference", ReferenceSpec, REFERENCE_PROMPT.format(text=text), check=check, max_new_tokens=160)

    def appearance_each(self, crops: dict[int, np.ndarray], appearance: str, batch: int = 8) -> dict[int, Confirmation]:
        """Gemma judges each candidate's isolated crop on its own: does it look like the words? Batched calls with the
        identical prompt (if the VLM supports it); an invalid answer gets one retry on its own."""
        prompt = APPEARANCE_ONE_PROMPT.format(appearance=appearance)
        if not hasattr(self.vlm, "generate_batch"):
            return {k: self._call("appearance_one", Confirmation, prompt, image=c, max_new_tokens=80) for k, c in crops.items()}
        out, keys = {}, list(crops)
        for i in range(0, len(keys), batch):
            ks = keys[i:i + batch]
            t0 = time.monotonic()
            raws = self.vlm.generate_batch(prompt, [crops[k] for k in ks], max_new_tokens=80)
            dt = (time.monotonic() - t0) / len(ks)
            for k, raw in zip(ks, raws):
                try:
                    out[k] = Confirmation.model_validate(_json(raw))
                    ok, err = True, None
                except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                    ok, err = False, str(exc).splitlines()[0][:200]
                if self.log:
                    self.log("vlm", role="appearance_one", attempt=1, system=None, prompt=prompt, answer=raw, valid=ok, error=err,
                             seconds=round(dt, 2), batched=len(ks), mark=k)
                if not ok:
                    out[k] = self._call("appearance_one", Confirmation, prompt, image=crops[k], max_new_tokens=80)
        return out

    def match_appearance(self, grid_image: np.ndarray, appearance: str, ids: list[int]) -> AppearanceMatch:
        """Gemma looks at close-up tiles of the candidates and lists those that look like the words."""
        def check(m: AppearanceMatch):
            bad = sorted((set(m.matches) | set(m.unsure)) - set(ids))
            if bad:
                raise ValueError(f"numbers {bad} are not in the picture; use only {sorted(ids)}")
        return self._call("appearance", AppearanceMatch, APPEARANCE_PROMPT.format(appearance=appearance), image=grid_image,
                          check=check, max_new_tokens=120)

    def confirm(self, crop_image: np.ndarray, text: str) -> Confirmation:
        return self._call("confirm", Confirmation, CONFIRM_PROMPT.format(text=text), image=crop_image)
