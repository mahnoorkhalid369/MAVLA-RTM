"""Gemma alone on the saved table picture of a finished request, with the manipulation worker's prompts: no simulator, no robot.

    cd v3/dialogue
    env -u AMENT_PREFIX_PATH PYTHONPATH=.:../../v2/src ../../v2/.venv-seg/bin/python tests/gemma_only.py \\
        ../out/v3d/<date>/run_<time>/01 "Put the milk carton on the light blue serving tray, in a corner."

It takes the first table picture and the measured facts of that request, asks the worker what to pick up and where it
goes, lets it point at both, then describe and point at the place on a close-up. The pictures with the boxes drawn are
saved in <folder>/gemma_only/. Unlike a live run, the close-up is cut around Gemma's own box, not around the measured
outline, and nothing checks whether the object fits there.
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2

import prompts
from agent.gemma import Gemma
from agents import Agents

folder, message = Path(sys.argv[1]), sys.argv[2]
events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
request = next(e["text"] for e in events if e.get("kind") == "instruction")
first = next(e["text"] for e in events if e.get("kind") == "prompt" and "\n- " in e["text"] and "In the gripper" in e["text"])
facts = first[first.index("\n- ") + 1:first.index("In the gripper")] + "In the gripper: nothing"
things = re.sub(r", middle at x [+-]\d+, y [+-]\d+ mm", "", facts)  # the pointing worker gets no positions
talk = f"user -> master: {request}\nmaster -> manipulation: {message}"
rgb = cv2.cvtColor(cv2.imread(str(folder / "images/table_001_rgb.png")), cv2.COLOR_BGR2RGB)
out = folder / "gemma_only"
out.mkdir(exist_ok=True)
gemma = Gemma(None)


def ask(text, picture, tokens):
    started = time.monotonic()
    answer = gemma.ask(text, images=[picture], max_new_tokens=tokens).strip()
    print(f"{answer}\n    ({time.monotonic() - started:.1f} s)", flush=True)
    return answer


def draw(picture, box, text):
    h, w = picture.shape[:2]
    corner, opposite = (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))
    for colour, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.rectangle(picture, corner, opposite, colour, width)
        cv2.putText(picture, text, (corner[0] + 4, corner[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, width)


def save(picture, name):
    cv2.imwrite(str(out / name), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
    print(f"saved {out / name}")


print(f"--- the master's message: {message}")
answer = ask(prompts.WHAT.format(things=things, talk=talk, message=message), rgb, 120)
pick = Agents.line(answer, "pick up")
place = Agents.line(answer, "put down") or Agents.line(answer, "on the table by")
table, boxes = rgb.copy(), {}
for thing in filter(None, (pick, place)):
    print(f"--- point at: {thing}")
    boxes[thing] = Agents.box(ask(prompts.POINT.format(thing=thing), rgb, 60), "box")
    if boxes[thing]:
        draw(table, boxes[thing], thing)
save(table, "table.jpg")

if place and boxes[place]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(v * s) for v, s in zip(boxes[place], (w, h, w, h)))
    pad = max(60, (max(x1 - x0, y1 - y0)) // 4)
    crop = rgb[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    close = cv2.resize(crop, None, fx=(scale := max(1, round(900 / max(crop.shape[:2])))), fy=scale, interpolation=cv2.INTER_CUBIC)
    size = next((line.split(": ", 1)[1] for line in things.splitlines() if line[2:].lower().startswith(pick.lower())), "size not measured")
    print(f"--- where on the close-up of: {place}")
    words = ask(prompts.WHERE.format(things=things, talk=talk, object=pick, size=size, where=place), close, 80)
    spot = Agents.box(ask(prompts.SPOT.format(object=pick, words=" ".join(words.split())), close, 60), "box")
    if spot:
        draw(close, spot, pick)
    save(close, "place.jpg")
