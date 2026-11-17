"""The master: Gemma turns the user's request into a plan, sends each step to a worker, and decides after a report,
keeping its own memory in plain words, until the request is done or the budget is used up.

    steps, memory = plan(gemma, request, state, image, log)        # [("manipulation", "put the small green cube on ..."), ...]
    action, value, memory = decide(gemma, request, state, memory, steps_text, last, budget, log, image)
        action: "continue" | "plan" (value = new steps) | "ask" (value = question to the user)
                | "done" (value = the answer / report to the user) | "stop" (value = the reason, to the user)

What the master gets each time: the request, the CURRENT camera picture of the table (taken after the last arm move;
there are no pre-made object names: it sees and names things itself), the places and what the gripper holds, its own
memory from the last call (what is done, what is left, what did not work), the plan still to do, the latest
worker reports and the budget left. It writes its memory again with every answer, so nothing depends on a long
history. Messages stay in words; the whole control response is validated before dispatch. A malformed response is
sent back once with what was wrong. Thinking is on: these are the calls that reason.
"""

from __future__ import annotations

import re

from agent.gemma import VLMFailure
from agent.contracts import response_lines
from agent import prompts
from agent.plan_validation import complete_transfers

WORKERS = ("perception", "manipulation", "verification")


def parse_steps(text: str) -> list[tuple[str, str]]:
    steps = []
    lines, _ = response_lines(text)
    for line in re.split(r"[\n;]", "\n".join(lines)):
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip().strip("*")
        if not line or line.lower().startswith("reason:"):
            continue
        if line.lower().startswith("plan:"):
            line = line.partition(":")[2].strip()
        worker, sep, message = line.partition(":")
        worker = worker.strip().strip("*").lower()
        if not sep or worker not in WORKERS or not message.strip():
            raise ValueError(f"{line!r} does not start with a worker's name and a colon ({', '.join(WORKERS)})")
        steps.append((worker, message.strip()))
    if not steps:
        raise ValueError("no steps")
    return complete_transfers(steps)


def memory_of(answer: str, before: str = "") -> str:
    _, memory = response_lines(answer)
    return " ".join(memory.split()) if memory is not None else before


def parse_decision(answer: str, memory: str = "") -> tuple[str, object, str]:
    lines, new_memory = response_lines(answer)
    word, sep, rest = lines[0].partition(":")
    word, rest = word.strip().lower(), rest.strip()
    memory = new_memory if new_memory is not None else memory
    if word == "plan" and sep:
        return "plan", parse_steps("\n".join([rest, *lines[1:]])), memory
    if len(lines) != 1:
        raise ValueError("unexpected additional lines after the decision")
    if word == "continue" and not sep:
        return word, None, memory
    if word in ("ask", "done", "stop") and sep and rest:
        return word, rest, memory
    raise ValueError("decision is not one of: continue, plan: ..., ask: ..., done: ..., stop: ...")


def plan(gemma, request: str, state: str, image, log, image_info=None) -> tuple[list[tuple[str, str]], str]:
    text, error = prompts.MASTER_PLAN.format(request=request, state=state), ""
    for _ in range(2):
        prompt = text + (prompts.RESPONSE_REPAIR.format(error=error) if error else "")
        log("model_input", role="master", phase="plan", prompt=prompt, system=prompts.MASTER, images=[image_info],
            call_id=getattr(gemma, "call_count", 0) + 1)
        answer = gemma.ask(prompt,
                           images=[image], system=prompts.MASTER, thinking=True, max_new_tokens=600)
        log("master", call="plan", answer=answer, thinking=gemma.last_thinking)
        try:
            return complete_transfers(parse_steps(answer), request), memory_of(answer)
        except ValueError as exc:
            error = str(exc)
    raise VLMFailure(f"plan: {error}")


def decide(gemma, request: str, state: str, memory: str, plan_text: str, last: list[str], budget: str,
           log, image=None, image_info=None) -> tuple[str, object, str]:
    text = prompts.MASTER_DECIDE.format(request=request, state=state, memory=memory or "(nothing yet)", plan=plan_text or "(nothing)",
                         last="\n".join(last) or "(nothing yet)", budget=budget)
    error = ""
    for _ in range(2):
        prompt = text + (prompts.RESPONSE_REPAIR.format(error=error) if error else "")
        log("model_input", role="master", phase="decide", prompt=prompt, system=prompts.MASTER,
            images=[image_info] if image is not None else [], call_id=getattr(gemma, "call_count", 0) + 1)
        answer = gemma.ask(prompt,
                           images=[image] if image is not None else [], system=prompts.MASTER, thinking=True,
                           max_new_tokens=600)
        log("master", call="decide", answer=answer, thinking=gemma.last_thinking)
        try:
            decision, value, updated_memory = parse_decision(answer, memory)
            if decision == "plan":
                value = complete_transfers(value, request)
            return decision, value, updated_memory
        except ValueError as exc:
            error = str(exc)
    raise VLMFailure(f"decide: {error}")
