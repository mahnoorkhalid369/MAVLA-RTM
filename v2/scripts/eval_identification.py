"""Object identification benchmark on recorded sessions, scored against the simulator's ground truth.

From v2 (no simulator needed; Gemma, YOLO and SigLIP 2 load on the GPU):
    .venv-seg/bin/python scripts/eval_identification.py out/demo/2026-11-12/live_051905 [more session folders ...]

Each session folder holds sim.log ("[run_scenario] <scenario> (<difficulty>, seed <n>)") and the first instruction's
first table frame (01/images/table_001_rgb.png + _depth.npz), taken before anything moved. The frame is replayed
through the agent's own perception (YOLO, depth measurement, robot removal, part merging, ids 1..N); every scenario
object and container is matched to the measured object nearest its ground-truth position (within MATCH_M), and each
method has to find it from its ground-truth name:

    siglip           SigLIP 2 alone: the name against every object's table crop, best score wins
    gemma_numbers    Gemma grounding on the numbered image with sizes, no names (the design before the catalogue)
    gemma_catalogue  Gemma names every object once (agent/catalogue.py), then grounds with those names (current design)

Items are asked as "pick up the <name>", containers as "pick up the <first item> and place it in the <name>".
Results: printed table, and <out>/identification.json with every answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MATCH_M = 0.06  # a ground-truth object and a measured object within this (xy) are the same object
CONTAINERS = {"serving_tray": "serving tray", "storage_basket": "storage basket"}


class ReplayBridge:
    """A saved RGB-D table frame in place of the ROS bridge."""

    def __init__(self, stem: Path) -> None:
        from perception.capture import Frame
        d = np.load(f"{stem}_depth.npz")
        rgb = cv2.cvtColor(cv2.imread(f"{stem}_rgb.png"), cv2.COLOR_BGR2RGB)
        self.f = Frame("table", rgb, d["depth"], tuple(d["K"]), float(d["stamp"]), d["q"], float(d["finger"]),
                       d["T_world_cam"])

    def snapshot(self, camera):
        return self.f

    def q(self):
        return self.f.q

    def finger(self):
        return self.f.finger

    def wait_still(self):
        pass


def scene_of(session: Path) -> tuple[str, str, int]:
    m = re.search(r"\[run_scenario\] (\S+) \((\w+), seed (\d+)\)", (session / "sim.log").read_text())
    if not m:
        raise ValueError(f"{session}/sim.log does not name the scenario")
    return m[1], m[2], int(m[3])


def truth(name: str, difficulty: str, seed: int) -> list[dict]:
    import scenarios
    sc = scenarios.generate(name, difficulty, seed)
    out = [{"key": o.id, "name": o.attributes.get("name", o.id.replace("_", " ")), "xy": o.pos[:2], "kind": "item"}
           for o in sc.objects]
    out += [{"key": f.id, "name": CONTAINERS[f.id], "xy": f.pos[:2], "kind": "container"}
            for f in sc.fixtures if getattr(f, "id", None) in CONTAINERS]
    return out


def ground(gemma, scene, obs, request: str, names: bool) -> tuple[str, int | None, int | None]:
    from agent import prompts
    tools = scene.tools
    words, tools.object_words = tools.object_words, (tools.object_words if names else {})
    try:
        text = scene.text(numbers=True, observation=obs)
    finally:
        tools.object_words = words
    prompt = prompts.GROUND.format(scene=text, message=request)
    prompt += "\nAllowed current object IDs: " + ", ".join(f"#{k}" for k in sorted(obs.object_ids.values()))
    answer = gemma.ask(prompt, images=[obs.id_image], thinking=True, max_new_tokens=120)
    src = re.search(r"pick #(\d+)", answer)
    dst = re.search(r"place \w+ #(\d+)", answer)
    return answer, int(src[1]) if src else None, int(dst[1]) if dst else None


def evaluate(session: Path, gemma, detector, matcher, out: Path) -> dict:
    from agent import catalogue
    from agent.eventlog import EventLog
    from agent.tools import Tools
    from agent.workers import Scene
    from perception.somark import isolate

    name, difficulty, seed = scene_of(session)
    work = out / session.name
    (work / "images").mkdir(parents=True, exist_ok=True)
    log = EventLog(work / "events.jsonl", echo=False)
    gemma.log, gemma.run_dir = log, work
    tools = Tools(ReplayBridge(session / "01/images/table_001"), detector, gemma, None, log, work)
    tools.matcher = matcher
    catalogue.build(tools, gemma, matcher, log)
    scene = Scene(tools, work)
    obs = scene.current(force=True)
    ids = sorted(set(obs.object_ids.values()))
    centre = {oid: np.array(tools.world.objects[oid].geometry["centre"][:2]) for oid in ids}
    gt = truth(name, difficulty, seed)
    for g in gt:
        d = {oid: float(np.linalg.norm(c - g["xy"])) for oid, c in centre.items()}
        best = min(d, key=d.get) if d else None
        g["id"] = best if best is not None and d[best] < MATCH_M else None
        g["catalogue_name"] = tools.object_words.get(g["id"]) if g["id"] is not None else None
    found = [g for g in gt if g["id"] is not None]
    mark = {oid: k for k, oid in obs.object_ids.items()}
    crops = [isolate(obs.rgb, obs.marks[mark[oid]], margin=0.15) for oid in ids]
    scores = matcher.scores(crops, [g["name"] for g in found]) if found else np.zeros((0, 0))
    first_item = next((g for g in found if g["kind"] == "item"), None)
    rows = []
    for j, g in enumerate(found):
        row = {"object": g["name"], "id": g["id"], "catalogue_name": g["catalogue_name"],
               "siglip": ids[int(np.argmax(scores[:, j]))] == g["id"]}
        if g["kind"] == "item":
            request, field = f"pick up the {g['name']}", 1
        else:
            if first_item is None:
                continue
            request, field = f"pick up the {first_item['name']} and place it in the {g['name']}", 2
        for method, names in (("gemma_numbers", False), ("gemma_catalogue", True)):
            answer, src, dst = ground(gemma, scene, obs, request, names)
            row[method] = (src, dst)[field - 1] == g["id"]
            row[method + "_answer"] = answer
        rows.append(row)
        print(f"  {g['name']:24s} #{g['id']:<3d} named '{g['catalogue_name']}':  siglip {'ok' if row['siglip'] else 'WRONG':5s} "
              f"gemma_numbers {'ok' if row['gemma_numbers'] else 'WRONG':5s} gemma_catalogue {'ok' if row['gemma_catalogue'] else 'WRONG'}",
              flush=True)
    log.close()
    return {"session": str(session), "scene": [name, difficulty, seed], "rows": rows,
            "unmatched": [g["name"] for g in gt if g["id"] is None]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "out/eval/identification" / time.strftime("%Y-%m-%d_%H%M%S"))
    a = ap.parse_args(argv)
    from agent.gemma import Gemma
    from config import models
    from perception.detector import YoloDetector
    from perception.matcher import Matcher
    gemma, matcher = Gemma(), Matcher()
    detector = YoloDetector(str(models.YOLO_WEIGHTS), models.DETECTOR_IMGSZ, models.DETECTOR_CONF)
    results = []
    for s in a.sessions:
        print(f"[eval] {s}", flush=True)
        results.append(evaluate(s, gemma, detector, matcher, a.out))
    rows = [r for res in results for r in res["rows"]]
    print(f"\n{len(rows)} lookups over {len(results)} scene(s):")
    for method in ("siglip", "gemma_numbers", "gemma_catalogue"):
        n = sum(r[method] for r in rows)
        print(f"  {method:16s} {n}/{len(rows)} = {100 * n / max(1, len(rows)):.0f} %")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "identification.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"[eval] {a.out / 'identification.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
