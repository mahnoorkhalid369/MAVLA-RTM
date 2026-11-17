"""The session's object catalogue: one name per world id, given by Gemma once and kept for the whole session.

    build(tools, gemma, matcher, log)        # at startup, before the first instruction: numbers 1..N, then names
    name_new(tools, gemma, matcher, obs, log) # later: only objects that were not seen before get a name
    names(tools)                              # every catalogue name, for the identity check (perception/matcher.py)

Naming: one Gemma call per object, on the WHOLE table image with only that object outlined and numbered (no crop
board: a crop loses the object's real size and its surroundings), and its measured size in the text. The names go
to tools.object_words, which grounding (numbered list), the master (plain list, no numbers) and the reports use.
SigLIP 2, when loaded, cross-checks each name against the object's table crop; a disagreement is only logged.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from agent import prompts
from perception.somark import draw_layout, isolate, layout


def _reading_order(obs) -> None:
    """Renumber the marks of a first observation in reading order on the image (rows top to bottom, then left to
    right), so the session's ids 1..N follow the picture instead of detector confidence."""
    def key(k):
        ys, xs = np.nonzero(obs.marks[k])
        return (int(ys.mean() // 120), float(xs.mean()))
    order = sorted(obs.marks, key=key)
    for name in ("marks", "measured", "scores"):
        data = getattr(obs, name)
        setattr(obs, name, {i + 1: data[k] for i, k in enumerate(order) if k in data})


def _clean(answer: str) -> str:
    line = next((x for x in answer.splitlines() if x.strip()), "")
    line = re.sub(r"^\s*(#\d+\s*[:=-]?\s*|object\s+#?\d+\s*(is)?\s*[:=-]?\s*)", "", line, flags=re.I)
    line = re.sub(r"^(it is|this is|a|an|the)\s+", "", line.strip(" *`'\".:"), flags=re.I)
    words = re.sub(r"[^\w\s-]", "", line).lower().split()[:5]
    return " ".join(words) or "object"


def _size(g: dict) -> str:
    fx, fy = (round(v * 1000) for v in g["footprint"])
    return f"{fx} x {fy} mm, {round(g['height'] * 1000)} mm tall"


def names(tools) -> list[str]:
    return list(dict.fromkeys(w for w in tools.object_words.values() if w and w != "object"))


def name_new(tools, gemma, matcher, obs, log) -> dict[int, str]:
    """Name every object of a table observation that has no catalogue name yet (not the held object)."""
    new = {}
    for k, oid in sorted(obs.object_ids.items(), key=lambda item: item[1]):
        if oid in tools.object_words or oid == tools.held_object or k not in obs.marks:
            continue
        g = tools.world.objects[oid].geometry
        image = draw_layout(obs.rgb, layout({oid: obs.marks[k]}))
        prompt = prompts.NAME.format(oid=oid, size=_size(g))
        log("model_input", role="catalogue", prompt=prompt, object_id=oid,
            images=[{"camera": "table", "frame_id": obs.frame_id, "variant": "one_outlined", "object_id": oid}],
            call_id=getattr(gemma, "call_count", 0) + 1)
        answer = gemma.ask(prompt, images=[image], thinking=False, max_new_tokens=24)
        tools.object_words[oid] = new[oid] = _clean(answer)
        path = tools.run_dir / "images" / f"catalogue_{oid}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if new:
        disagree = cross_check(tools, matcher, obs, new, log)
        log("catalogue", frame_id=obs.frame_id, named={oid: tools.object_words[oid] for oid in new},
            sizes={oid: _size(tools.world.objects[oid].geometry) for oid in new}, siglip_disagrees=disagree)
    return new


def cross_check(tools, matcher, obs, oids, log) -> dict:
    """SigLIP 2 on each newly named object's table crop against all catalogue names: {id: what it ranked first}
    where that is not the object's own name. Logged as a warning only; the name is kept."""
    if matcher is None:
        return {}
    every = names(tools)
    marks = {oid: k for k, oid in obs.object_ids.items() if oid in oids and k in obs.marks}
    if len(every) < 2 or not marks:
        return {}
    ids = sorted(marks)
    crops = [isolate(obs.rgb, obs.marks[marks[oid]], margin=0.15) for oid in ids]
    p = matcher.scores(crops, every)
    out = {}
    for oid, row in zip(ids, p):
        top = every[int(np.argmax(row))]
        if top != tools.object_words[oid]:
            out[oid] = {"siglip_top": top, "scores": {n: round(float(s), 3) for n, s in zip(every, row) if s > 0.05}}
    return out


def build(tools, gemma, matcher, log) -> list[str]:
    """Startup: one table look, ids 1..N in reading order, a Gemma name for each. Returns printable lines."""
    obs = tools.perceive("table")
    if not tools.world.objects:
        _reading_order(obs)
    tools.update_world(obs)
    name_new(tools, gemma, matcher, obs, log)
    return [f"#{oid} {tools.object_words.get(oid, 'object')} ({_size(tools.world.objects[oid].geometry)})"
            for oid in sorted(set(obs.object_ids.values()))]
