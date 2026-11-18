"""The conversation loop without Gemma, simulator or GPU: scripted model answers and a pretend robot.

    PYTHONPATH=. python3 tests/test_dialogue.py
"""
import numpy as np

import agents


class FakeGemma:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def ask(self, text, images=(), **options):
        self.prompts.append(text)
        return self.answers.pop(0)


class FakeRobot:
    """A tray (hollow) at the right and two objects at the left of a 1000 x 1000 picture."""
    def __init__(self):
        self.moved, self.done, self.boxes = False, [], {1: (100, 100, 100, 100), 2: (100, 400, 100, 100), 3: (600, 300, 300, 300)}

    def look(self):
        objects = {i: {"width": 60, "length": 60, "height": 100, "x": 500, "y": 100 * i, "crop": None, "hollow": 0} for i in (1, 2)}
        objects[3] = {"width": 360, "length": 250, "height": 25, "x": -500, "y": 500, "crop": None, "hollow": 18}
        return np.zeros((1000, 1000, 3), np.uint8), None, objects

    def holding(self):
        return None

    def meant(self, box, skip=None):
        best = (None, 0.0)
        for oid, (x, y, w, h) in self.boxes.items():
            both = max(0, min(box[2], x + w) - max(box[0], x)) * max(0, min(box[3], y + h) - max(box[1], y))
            share = both / (w * h + (box[2] - box[0]) * (box[3] - box[1]) - both)
            if oid != skip and share > best[1]:
                best = (oid, share)
        return best

    def zoom(self, box, around=0):
        return "close-up", (box[0], box[1], box[2] - box[0], box[3] - box[1], 1)

    def aim(self, oid, support, u, v):
        return (oid, support, round(u), round(v)), None

    def preview(self, plan, picture, view):
        return "outlined", 12

    def why_moved(self):
        return "out of reach"

    def carry_out(self, plan):
        self.moved = True
        self.done.append(plan)
        return True, "carried and released"

    def pick(self, oid):
        self.moved = True
        self.done.append(("pick", oid))
        return True, "held"


NAMES = ["milk carton", "apple", "blue tray"]
TRAY = "box: 300, 600, 600, 900 | the tray"
MILK = ["pick up: milk carton\nput down: blue tray\nwhy: the milk goes to the tray", "box: 100, 100, 200, 200 | the milk", TRAY]
APPLE = ["pick up: apple\nput down: blue tray\nwhy: the apple goes to the tray", "box: 400, 100, 500, 200 | the apple", TRAY]


def case(name, answers, request, expected, replies=()):
    gemma, robot, asked, replies = FakeGemma(NAMES + answers), FakeRobot(), [], list(replies)
    a = agents.Agents(gemma, robot, lambda q: asked.append(q) or replies.pop(0), say=lambda *line: None)
    got = (a.run(request)[0], robot.done, len(asked), len(gemma.answers))
    print("PASS" if got == expected else "FAIL", name, "|", got if got != expected else "")
    return a


case("one object, the worker accepts the outlined place", [
    "Plan: milk to the tray.\nto manipulation: Put the milk carton into the blue tray.",
    *MILK, "At the side of the tray.", "box: 500, 500, 600, 600 | at the side", "fine: beside the rim",
    "to verification: Is the milk carton in the blue tray?", "The milk carton stands in the blue tray.",
    "done: The milk carton is in the blue tray."],
    "put the milk in the tray", ("COMPLETED", [(1, 3, 765, 465)], 0, 0))

a = case("two objects, one correction of the place", [
    "to manipulation: Put the milk carton into the blue tray, at a side; the apple follows.",
    *MILK, "In the upper left corner.", "box: 0, 0, 200, 200 | corner",
    "move it: into the lower right corner", "box: 800, 800, 1000, 1000 | other corner", "fine: good",
    "to manipulation: Put the apple into the blue tray, beside the milk.",
    *APPLE, "Close beside the milk.", "box: 400, 400, 600, 600 | beside the milk", "fine: ok",
    "to verification: Are both in the tray?", "Both stand in the tray.", "done: Both are in the tray."],
    "put the milk and the apple in the tray", ("COMPLETED", [(1, 3, 870, 570), (2, 3, 750, 450)], 0, 0))
print("PASS" if "robot -> manipulation: The white outline" in a.gemma.prompts[-4] and "middle at x" not in a.gemma.prompts[-4] and "master -> manipulation: Put the apple" in a.gemma.prompts[-4]
      else "FAIL", "every agent reads the whole conversation, the pointing worker without positions |")

case("the worker points at nothing measured: the master hears it and stops", [
    "to manipulation: Put the banana into the blue tray.",
    "pick up: banana\nput down: blue tray\nwhy: asked", "There is no banana in this image.",
    "stop: There is no banana on the table."],
    "put the banana in the tray", ("STOPPED", [], 0, 0))

case("the worker cannot find the object, the master asks the user, the user cancels", [
    "to manipulation: Put the banana into the blue tray.",
    "There is no banana in the picture.",
    "to user: I see no banana. Which object do you mean?",
    "stop: The user cancelled."],
    "put the banana in the tray", ("STOPPED", [], 1, 0), replies=["never mind"])

case("a question only, no motion", [
    "to perception: Which objects are on the table?", "A milk carton, an apple and a blue tray.",
    "done: A milk carton, an apple and a blue tray."],
    "what is on the table?", ("COMPLETED", [], 0, 0))

case("the master's answer has no message line, then it is repaired", [
    "I think the milk should go to the tray.",
    "to manipulation: Put the milk carton into the blue tray.",
    *MILK, "In the middle.", "box: 500, 500, 600, 600 | middle", "fine: ok",
    "done: Done."],
    "put the milk in the tray", ("COMPLETED", [(1, 3, 765, 465)], 0, 0))

case("onto the table beside another object", [
    "to manipulation: Put the apple on the table close beside the milk carton.",
    "pick up: apple\non the table by: milk carton\nwhy: asked", "box: 400, 100, 500, 200 | the apple", "box: 100, 100, 200, 200 | the milk",
    "Just right of the milk carton.", "box: 400, 600, 600, 800 | right of the milk", "fine: close beside it",
    "done: The apple stands beside the milk carton."],
    "put the apple next to the milk", ("COMPLETED", [(2, None, 170, 150)], 0, 0))

case("pick and hold", [
    "to manipulation: Pick up the apple and hold it.", "pick up: apple\nwhy: only pick it up", "box: 400, 100, 500, 200 | the apple",
    "done: The apple is in the gripper."],
    "pick up the apple", ("COMPLETED", [("pick", 2)], 0, 0))
