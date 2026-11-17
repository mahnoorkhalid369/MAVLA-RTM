"""TaskPlanner: camera image + instruction -> validated TaskPlan; speech -> text.

No ROS: give it numpy arrays. The model is passed in, so the planner can be tested with a stub.
Speech is handled in two steps: transcribe() turns audio into text (quick, no reasoning), then
plan() treats it exactly like a typed instruction.
"""

from __future__ import annotations

import re
import time

import numpy as np

from vlm.plan import PlanError, Step, TaskPlan, parse_detections, parse_plan
from vlm.prompts import (
    DETECT_PROMPT,
    NO_SPEECH,
    RETRY_PROMPT,
    SYSTEM_PROMPT,
    TASK_PROMPT_TEXT,
    TRANSCRIBE_PROMPT,
)

MIN_WORDS = 2  # shorter transcripts are treated as noise
# When the audio has no clear speech the model sometimes answers like a chat assistant instead of
# transcribing ("Please provide the audio."); such replies are not what the user said.
NOT_A_TRANSCRIPT = re.compile(
    r"\b(audio|transcri\w*|recording|provide|i'?m not sure|i cannot|i can'?t|unable|sorry|no speech|"
    r"not clear|inaudible|silence|background noise)\b",
    re.IGNORECASE,
)


PLACE_WORDS = re.compile(r"\b(put|place|stack|move|onto|on top|on the|next to|beside|into|in the|sort|line up|arrange)\b",
                         re.IGNORECASE)


def check_complete(plan: TaskPlan, instruction: str) -> None:
    """Reject a plan that stops while still holding an object although the user asked to put it somewhere."""
    actions = [s.action for s in plan.steps]
    if "search" in actions or not actions:
        return
    holding = actions.count("pick") > actions.count("place")
    if holding and PLACE_WORDS.search(instruction):
        raise PlanError(
            "the plan ends while still holding an object, but the instruction says where to put it: "
            "add the place step(s)"
        )


TARGET_PHRASE = re.compile(r"\b(?:on|onto|on top of)\s+(?:the\s+)?([a-z]+\s+(?:cube|box|block|cylinder|sphere|ball))\b",
                           re.IGNORECASE)


def complete_from_instruction(answer: str, instruction: str) -> TaskPlan | None:
    """Fallback: the model stopped after the pick; take the target from "... on the yellow cube"."""
    try:
        plan = parse_plan(answer)
    except PlanError:
        return None
    m = TARGET_PHRASE.search(instruction)
    actions = [s.action for s in plan.steps]
    if m is None or actions.count("pick") != actions.count("place") + 1:
        return None
    plan.steps.append(Step("place", m.group(1).lower()))
    plan.reply += f" (added the missing step: place on the {m.group(1).lower()})"
    return plan


class TaskPlanner:
    def __init__(self, model, retries: int = 1) -> None:
        self.model = model  # anything with generate(text, image=, audio=, system=, thinking=) -> str
        self.retries = retries
        self.last_seconds = 0.0

    def transcribe(self, audio: np.ndarray) -> str | None:
        """Speech (mono float32, 16 kHz) -> text, or None if there is no clear speech."""
        start = time.monotonic()
        text = self.model.generate(TRANSCRIBE_PROMPT, audio=audio, max_new_tokens=120, thinking=False).strip()
        self.last_seconds = time.monotonic() - start
        text = text.strip().strip('"').strip()
        if not text or NO_SPEECH in text or len(re.findall(r"[A-Za-z]+", text)) < MIN_WORDS:
            return None
        if NOT_A_TRANSCRIPT.search(text):
            return None
        return text

    def detect(self, image: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Every object in the image as (label, box 0-1000). Quick: no reasoning."""
        start = time.monotonic()
        answer = self.model.generate(DETECT_PROMPT, image=image, max_new_tokens=600, thinking=False)
        self.last_seconds = time.monotonic() - start
        return parse_detections(answer)

    def plan(self, image: np.ndarray, text: str, memory: str = "(nothing remembered yet)") -> TaskPlan:
        """image: HxWx3 uint8 RGB from the wrist camera; text: the instruction; memory: remembered objects."""
        if not text:
            raise ValueError("empty instruction")
        prompt = TASK_PROMPT_TEXT.format(instruction=text, memory=memory)
        start = time.monotonic()
        answer, error, thinking = "", None, None  # None = config default
        for _ in range(self.retries + 1):
            ask = prompt if error is None else f"{prompt}\n\n{RETRY_PROMPT.format(error=error)}"
            answer = self.model.generate(ask, image=image, system=SYSTEM_PROMPT, thinking=thinking)
            try:
                plan = parse_plan(answer)
                check_complete(plan, text)
                self.last_seconds = time.monotonic() - start
                return plan
            except PlanError as exc:
                error = str(exc)
                if not answer.strip():  # reasoning used up the budget: answer directly next time
                    thinking = False
        self.last_seconds = time.monotonic() - start
        completed = complete_from_instruction(answer, text)
        if completed is not None:
            return completed
        raise PlanError(f"{error} (model said: {answer[:200]!r})")
