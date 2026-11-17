"""Internal handoffs: status and measured effects stay separate from model-written prose."""

from __future__ import annotations

from dataclasses import dataclass, field
from agent import prompts


def sensor_epoch(tools):
    return getattr(getattr(tools, "bridge", None), "epoch", getattr(tools, "_sensor_epoch", None))


@dataclass(frozen=True)
class Outcome:
    kind: str  # holding | released | placed | unknown | home
    object_id: int | None = None
    relation: str | None = None
    reference_id: int | None = None
    region_id: str | None = None


@dataclass(frozen=True)
class Report:
    word: str
    text: str
    code: str = ""
    frame_id: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.word not in {"ok", "done", "yes", "failed", "unsure", "no"}:
            raise ValueError(f"invalid worker status: {self.word!r}")

    @property
    def ok(self) -> bool:
        return self.word in {"ok", "done", "yes"}

    def __str__(self) -> str:
        return f"{self.word}: {self.text}"


@dataclass
class TaskContext:
    """One instruction's execution record, shared with verification, never written by Gemma.

    Outcomes describe the latest successful physical effect per object; history is
    separate. The final visual check still evaluates the complete original request.
    """
    outcomes: dict[int | str, Outcome] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    motion_attempted: bool = False
    motion_requested: bool = False
    pending: set[int | str] = field(default_factory=set)
    current: dict[int | str, str] = field(default_factory=dict)
    revision: int = 0
    placements: dict = field(default_factory=dict)  # evidence receipts, never master memory
    failures: list[str] = field(default_factory=list)  # "<what was asked> -> <why it failed>", latest last

    def note_failure(self, message: str, why: str) -> None:
        self.failures = (self.failures + [f"{message} -> {why}"])[-3:]

    def record(self, outcome: Outcome, text: str) -> None:
        key = outcome.object_id if outcome.object_id is not None else "home"
        self.outcomes[key] = outcome
        self.pending.discard(key)
        self.current[key] = text
        self.history.append(text)
        self.revision += 1

    def summary(self, held: str | None = None) -> str:
        """held: the catalogue name of what the gripper holds now (it may come from an earlier instruction)."""
        done = "\n".join(f"- {x}" for x in self.current.values()) or (
            prompts.EXECUTION_ALREADY_HELD.format(held=held) if held else prompts.EXECUTION_NOTHING)
        failed = prompts.EXECUTION_FAILURES.format(failures="\n".join(f"- {x}" for x in self.failures)) if self.failures else ""
        pending = prompts.EXECUTION_UNRESOLVED if self.pending else ""
        return prompts.EXECUTION_CONTEXT.format(done=done, failed=failed, pending=pending)


def response_lines(answer: str) -> tuple[list[str], str | None]:
    """Validate the whole response and extract at most one final memory line."""
    import re
    if re.search(r"<[^>\n]*\|[^>\n]*>|<eos>|<bos>", answer):
        raise ValueError("response contains model control tokens; return only the requested answer")
    lines = [line.strip().strip("*").strip() for line in answer.splitlines() if line.strip()]
    memory = None
    for index, line in enumerate(lines):
        if line.lower().startswith("memory:"):
            if memory is not None or index != len(lines) - 1:
                raise ValueError("memory must occur once, on the last line")
            memory = line.partition(":")[2].strip()
    if memory is not None:
        lines.pop()
    if any(re.search(r"->\s*(ok|done|yes|no|failed|unsure)\s*:", line, re.I) for line in lines):
        raise ValueError("response echoes an execution report; return new instructions only")
    if not lines:
        raise ValueError("response has no action or decision")
    return lines, memory
