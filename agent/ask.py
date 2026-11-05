"""Asking the user (ask_user): questions, answers and the two responders.

Only a person (TerminalResponder) or an explicitly configured script (ScriptedResponder, for unattended
rehearsals) answers. Model output is never an answer. A "which object?" question has an explicit protocol, with no
interpretation of words in Python:

    "7" or "#7"      -> object 7: a listed candidate, or any other object in the current view (its number in the
                        picture); a number that is not an object of the current observation is refused
                        (for "is it object 7?", 7 confirms it)
    an empty line    -> no choice: the step is cancelled (Gemma then decides what to do)
    anything else    -> explained, and the question is asked again

Gemma's own free-text questions (ask) return the typed text as it is; Gemma interprets it.
Cancellation policy (documented): an empty line; end of input (EOF: nobody can answer); or no answer within the
timeout (default Budgets.question_timeout_s = 120 s; 0 = wait indefinitely). A ScriptedResponder answer that is not a
candidate number (e.g. "cancel") is no choice.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field

@dataclass
class Candidate:
    mark: int
    description: str  # the VLM's own words (shown to the user)
    centre_mm: tuple[float, float]  # robot base frame: +x = robot's right, +y = forward
    distance_m: float  # from the robot base axis
    size_mm: tuple[float, float, float]  # footprint long, short, height


@dataclass
class Question:
    kind: str  # "choose": pick one candidate | "confirm": the asked object's number, another number, or none
    text: str
    role: str  # "object" (to move) | "reference" (destination)
    candidates: list[Candidate]
    image: str | None = None  # picture of the candidates for the user
    visible: list[int] = field(default_factory=list)  # every object number of the current observation (also valid)


@dataclass
class Answer:
    kind: str  # "choice" | "text" (the person answered in words) | "cancel" | "eof" | "timeout"
    mark: int | None = None
    raw: str = ""
    source: str = "terminal"  # "terminal" (a person) | "scripted" (configured rehearsal answer)
    attempts: list = field(default_factory=list)  # rejected inputs before this answer, with the reason


def parse_answer(text: str, q: Question) -> tuple[Answer | None, str]:
    """(Answer, "") for a valid answer, or (None, why it is not valid): a listed number, words, or an empty line (no
    choice). Words are the person's own answer ("you already have it in your grasp"): they are passed on as typed for
    the supervisor (Gemma) to understand, never rejected (live 2026-10-26: such an answer was refused as "not a
    number")."""
    t = text.strip()
    if not t or t.lower() in ("cancel", "none", "stop"):
        return Answer("cancel", raw=text), ""
    m = re.fullmatch(r"#?\s*(\d+)", t)
    marks = {c.mark for c in q.candidates}
    if m is None:
        return Answer("text", raw=t), ""
    n = int(m.group(1))
    if n in marks or n in set(q.visible):
        return Answer("choice", n, raw=text), ""
    if q.visible:
        return None, f"{n} is not an object in the current view (objects: {sorted(set(q.visible) | marks)})"
    return None, f"{n} is not one of the candidates {sorted(marks)}"


def format_question(q: Question) -> str:
    lines = ["", f"[agent] QUESTION ({q.role}): {q.text}"]
    for c in q.candidates:
        lines.append(f"  {c.mark:>3}: {c.description:<40.40s} at x {c.centre_mm[0]:+.0f} mm, y {c.centre_mm[1]:+.0f} mm "
                     f"({c.distance_m:.2f} m from the robot), {c.size_mm[0]:.0f} x {c.size_mm[1]:.0f} x {c.size_mm[2]:.0f} mm")
    if q.image:
        lines.append(f"  picture of the candidates: {q.image}")
    others = sorted(set(q.visible) - {c.mark for c in q.candidates})
    if others:
        lines.append(f"  other objects in view (numbers as in the camera picture): {', '.join(map(str, others))}")
    lines.append("  (x = robot's right, y = forward from the robot base)")
    if q.kind == "confirm":
        lines.append(f"  Type {q.candidates[0].mark} to confirm it, another object's number, answer in your own words, or an "
                     "empty line for none.")
    else:
        lines.append("  Type one of the numbers" + (" (or another object's number)" if others else "") + ", answer in your "
                     "own words, or an empty line for none.")
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
