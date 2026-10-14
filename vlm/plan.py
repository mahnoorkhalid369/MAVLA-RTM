"""The task plan the model returns, and strict parsing/validation of its JSON.

No ROS, no model: run `python3 -m vlm.plan` for a self-test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

ACTIONS = ("pick", "place", "home", "search", "move")


class PlanError(ValueError):
    """The model's answer is not a usable plan."""


@dataclass
class Step:
    action: str  # "pick" | "place" | "home"
    label: str = ""  # object (pick) or target (place) description
    box: tuple[int, int, int, int] | None = None  # (ymin, xmin, ymax, xmax), 0-1000, in the current image
    object_id: int | None = None  # a remembered object (perception/world_memory.py)
    position_mm: tuple[float, float] | None = None  # place: a free table spot (base frame, mm)

    def describe(self) -> str:
        if self.action == "home":
            return "return home"
        if self.action == "search":
            return f"search for {self.label}"
        if self.action == "move":
            return self.label
        ref = f" (#{self.object_id})" if self.object_id is not None else ""
        return f"{self.action} {self.label}{ref}".strip()


@dataclass
class TaskPlan:
    reply: str
    heard: str = ""
    steps: list[Step] = field(default_factory=list)
    raw: str = ""  # the model's text, for logs


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PlanError("no JSON object in the answer")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise PlanError("the JSON is not an object")
    return data


def _box(value, where: str) -> tuple[int, int, int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value)):
        raise PlanError(f"{where}: box_2d must be 4 numbers [ymin, xmin, ymax, xmax]")
    y0, x0, y1, x1 = (int(round(float(v))) for v in value)
    if not all(0 <= v <= 1000 for v in (y0, x0, y1, x1)):
        raise PlanError(f"{where}: box_2d values must be 0-1000")
    if y1 <= y0 or x1 <= x0:
        raise PlanError(f"{where}: box_2d needs ymin < ymax and xmin < xmax")
    return y0, x0, y1, x1


def parse_detections(text: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """[{"label", "box_2d"}, ...] from the detection prompt; invalid entries are skipped."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            found.append((str(item.get("label", "object")).strip().lower(), _box(item.get("box_2d"), "detection")))
        except PlanError:
            continue
    return found


def parse_plan(text: str) -> TaskPlan:
    data = _extract_json(text)
    reply = str(data.get("reply", "")).strip()
    steps_in = data.get("steps", [])
    if not isinstance(steps_in, list):
        raise PlanError('"steps" must be a list')
    steps, holding = [], False
    for i, s in enumerate(steps_in, 1):
        if not isinstance(s, dict) or s.get("action") not in ACTIONS:
            raise PlanError(f"step {i}: action must be one of {', '.join(ACTIONS)}")
        action = s["action"]
        if action == "home":
            steps.append(Step("home"))
            continue
        if action == "move":  # a direct robot command: "move left 100", "look up 20", "turn around"
            command = str(s.get("command", "")).strip()
            if not command:
                raise PlanError(f"step {i}: move needs a command, e.g. \"move left 100\"")
            steps.append(Step("move", command))
            continue
        if action == "search":  # object not in view or memory: the robot looks around for it
            label = str(s.get("object", "")).strip()
            if not label:
                raise PlanError(f"step {i}: search needs an object description")
            steps.append(Step("search", label))
            break  # the robot re-plans once it has found it
        label = str(s.get("object" if action == "pick" else "target", "")).strip()
        where = f"step {i} ({action})"
        box = object_id = position = None
        if s.get("box_2d") is not None:
            box = _box(s["box_2d"], where)
        elif s.get("object_id") is not None:
            try:
                object_id = int(s["object_id"])
            except (TypeError, ValueError) as exc:
                raise PlanError(f"{where}: object_id must be a number") from exc
        elif action == "place" and s.get("position_mm") is not None:
            value = s["position_mm"]
            if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
                raise PlanError(f"{where}: position_mm must be [x, y] in millimetres")
            position = (float(value[0]), float(value[1]))
        elif not label:
            options = "a description, box_2d or object_id" + (" or position_mm" if action == "place" else "")
            raise PlanError(f"{where}: needs {options}")
        # only a description: the agent finds it (detection in this image, then memory, then search)
        if action == "pick":
            if holding:
                raise PlanError(f"step {i}: pick while already holding something (place it first)")
            holding = True
        else:
            if not holding:
                raise PlanError(f"step {i}: place without a pick before it")
            holding = False
        steps.append(Step(action, label, box, object_id, position))
    return TaskPlan(reply=reply, heard=str(data.get("heard", "")).strip(), steps=steps, raw=text)


if __name__ == "__main__":
    good = """```json
{"heard": "", "reply": "Putting the red block on the green one.",
 "steps": [{"action": "pick", "object": "red block", "box_2d": [400, 450, 520, 560]},
           {"action": "place", "target": "green block", "box_2d": [380, 700, 480, 800]},
           {"action": "home"}]}
```"""
    plan = parse_plan(good)
    print("parsed:", plan.reply, [(s.describe(), s.box) for s in plan.steps])
    for bad in ['{"steps": [{"action": "place", "target": "x", "box_2d": [1,2,3,4]}]}',
                '{"steps": [{"action": "pick", "object": "x", "box_2d": [500, 2, 100, 4]}]}',
                '{"steps": [{"action": "dance"}]}', "I will pick the red block."]:
        try:
            parse_plan(bad)
        except PlanError as e:
            print("rejected:", e)
