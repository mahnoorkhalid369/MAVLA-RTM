"""Asking the user (ask_user): questions, answers and the two responders.

Only a person (TerminalResponder) or an explicitly configured script (ScriptedResponder, for unattended
rehearsals) answers. Model output is never an answer. Answers are parsed deterministically:

    "7", "#7", "object 7"     -> candidate 7 (must be one of the listed candidates)
    "the gold one"            -> the only candidate whose description contains every word given
    "nearest" / "farthest"    -> the candidate closest to / farthest from the robot base
    "cancel" (stop, abort, quit) -> cancel
    anything else / several matches -> explained, and the question is asked again

Cancellation policy (documented): explicit "cancel"; end of input (EOF: nobody can answer); or no
answer within the timeout (default Budgets.question_timeout_s = 120 s; 0 = wait indefinitely).
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field

CANCEL_WORDS = {"cancel", "stop", "abort", "quit"}
NEAR_WORDS = {"nearest", "closest", "near", "close"}
FAR_WORDS = {"farthest", "furthest", "far"}
YES_WORDS = {"yes", "y", "yeah", "correct"}
NO_WORDS = {"no", "n", "nope"}
FILLER = {"the", "a", "an", "one", "object", "number", "id", "it", "is", "that", "this", "i", "mean", "please", "#"}


@dataclass
class Candidate:
    mark: int
    description: str  # the VLM's own words (shown to the user; also matched against their answer)
    centre_mm: tuple[float, float]  # robot base frame: +x = robot's right, +y = forward
    distance_m: float  # from the robot base axis
    size_mm: tuple[float, float, float]  # footprint long, short, height


@dataclass
class Question:
    kind: str  # "choose": pick one candidate | "confirm": yes / no / another number
    text: str
    role: str  # "object" (to move) | "reference" (destination)
    candidates: list[Candidate]
    image: str | None = None  # picture of the candidates for the user


@dataclass
class Answer:
    kind: str  # "choice" | "yes" | "no" | "cancel" | "eof" | "timeout"
    mark: int | None = None
    raw: str = ""
    source: str = "terminal"  # "terminal" (a person) | "scripted" (configured rehearsal answer)
    attempts: list = field(default_factory=list)  # rejected inputs before this answer, with the reason


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9#]+", text.lower()) if w]


def parse_answer(text: str, q: Question) -> tuple[Answer | None, str]:
    """(Answer, "") for a valid answer, or (None, why it is not valid)."""
    words = _words(text)
    if not words:
        return None, "empty answer"
    if set(words) & CANCEL_WORDS:
        return Answer("cancel", raw=text), ""
    marks = {c.mark for c in q.candidates}
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if q.kind == "confirm" and not numbers:
        if set(words) & YES_WORDS:
            return Answer("yes", raw=text), ""
        if set(words) & NO_WORDS:
            return Answer("no", raw=text), ""
    if len(numbers) > 1:
        return None, f"several numbers given ({numbers}); give one of {sorted(marks)}"
    if numbers:
        if numbers[0] in marks:
            return Answer("choice", numbers[0], raw=text), ""
        return None, f"{numbers[0]} is not one of the candidates {sorted(marks)}"
    pool = list(q.candidates)
    content = [w for w in words if w not in FILLER | NEAR_WORDS | FAR_WORDS]
    if content:  # description words: keep candidates whose description contains all of them
        pool = [c for c in pool if set(content) <= set(_words(c.description))]
        if not pool:
            return None, f"no candidate is described with {' '.join(content)!r}"
    if set(words) & NEAR_WORDS:
        pool = [min(pool, key=lambda c: c.distance_m)]
    elif set(words) & FAR_WORDS:
        pool = [max(pool, key=lambda c: c.distance_m)]
    if len(pool) == 1 and (content or set(words) & (NEAR_WORDS | FAR_WORDS)):
        return Answer("choice", pool[0].mark, raw=text), ""
    if len(pool) > 1 and content:
        return None, f"{' '.join(content)!r} fits several candidates {[c.mark for c in pool]}; give a number"
    return None, "not understood"


def format_question(q: Question) -> str:
    lines = ["", f"[agent] QUESTION ({q.role}): {q.text}"]
    for c in q.candidates:
        lines.append(f"  {c.mark:>3}: {c.description:<40.40s} at x {c.centre_mm[0]:+.0f} mm, y {c.centre_mm[1]:+.0f} mm "
                     f"({c.distance_m:.2f} m from the robot), {c.size_mm[0]:.0f} x {c.size_mm[1]:.0f} x {c.size_mm[2]:.0f} mm")
    if q.image:
        lines.append(f"  picture of the candidates: {q.image}")
    lines.append("  (x = robot's right, y = forward from the robot base)")
    if q.kind == "confirm":
        lines.append("  Answer yes, no, another object's number, or cancel.")
    else:
        lines.append("  Answer with a number, a description that fits one of them (e.g. 'nearest'), or cancel.")
    return "\n".join(lines)


class LineReader:
    """The process's one reader of standard input: a background thread puts lines on a queue, so the
    instruction prompt and the questions share stdin (tty or pipe) and both can wait with a timeout."""

    _instance: "LineReader | None" = None

    def __init__(self, stream=None) -> None:
        self.q: queue.Queue = queue.Queue()
        self.stream = stream or sys.stdin
        self.on_directive = None  # callable(line): lines starting with "@" go here, never to a reader (harnesses)
        threading.Thread(target=self._pump, daemon=True).start()

    @classmethod
    def shared(cls) -> "LineReader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _pump(self) -> None:
        for line in iter(self.stream.readline, ""):
            self.q.put(line.rstrip("\n"))
        self.q.put(None)  # EOF

    def get(self, timeout: float | None = None):
        """Next line; None at EOF; raises TimeoutError if nothing arrives within timeout (None = wait).
        With on_directive set, "@..." lines are handed to it and the next line is read."""
        while True:
            try:
                line = self.q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError from None
            if line is None:
                self.q.put(None)  # EOF stays EOF for every later reader
                return line
            if self.on_directive is None or not line.startswith("@"):
                return line
            self.on_directive(line)


class TerminalResponder:
    """A person answers at the terminal. Invalid answers are explained and the question is repeated."""

    def __init__(self, reader: LineReader | None = None, out=None, timeout_s: float = 120.0) -> None:
        self.reader, self.out, self.timeout_s = reader or LineReader.shared(), out or sys.stdout, timeout_s

    def __call__(self, q: Question) -> Answer:
        print(format_question(q), file=self.out, flush=True)
        rejected = []
        deadline = time.monotonic() + self.timeout_s if self.timeout_s > 0 else None
        while True:
            print("answer> ", end="", file=self.out, flush=True)
            try:
                left = None if deadline is None else max(0.0, deadline - time.monotonic())
                line = self.reader.get(left)
            except TimeoutError:
                print(f"\n[agent] no answer within {self.timeout_s:.0f} s: cancelling", file=self.out, flush=True)
                return Answer("timeout", attempts=rejected)
            if line is None:
                print("\n[agent] end of input: nobody can answer, cancelling", file=self.out, flush=True)
                return Answer("eof", attempts=rejected)
            a, why = parse_answer(line, q)
            if a is not None:
                a.attempts = rejected
                return a
            rejected.append({"input": line, "reason": why})
            print(f"[agent] {why}. Please answer again.", file=self.out, flush=True)

    def ask(self, text: str) -> str | None:
        """A free-text question from the supervisor (Gemma's ask_user decision). The person's answer is returned as
        typed (Gemma interprets it in its next decision); None = no answer (timeout or end of input)."""
        print(f"\n[QUESTION] {text}", file=self.out, flush=True)
        print("answer> ", end="", file=self.out, flush=True)
        try:
            line = self.reader.get(self.timeout_s if self.timeout_s > 0 else None)
        except TimeoutError:
            print(f"\n[agent] no answer within {self.timeout_s:.0f} s", file=self.out, flush=True)
            return None
        return None if line is None or not line.strip() else line.strip()


class ScriptedResponder:
    """Unattended runs only (rehearsal harness): fixed answers, logged as scripted, never a person."""

    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or ["cancel"])

    def __call__(self, q: Question) -> Answer:
        raw = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        a, why = parse_answer(raw, q)
        a = a or Answer("cancel", raw=raw)
        a.source = "scripted"
        return a

    def ask(self, text: str) -> str | None:
        """Free-text question: the next scripted answer ("cancel" counts as no answer)."""
        raw = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return None if raw.strip().lower() == "cancel" else raw
