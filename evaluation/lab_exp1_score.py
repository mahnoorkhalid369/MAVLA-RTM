"""Laboratory Experiment 1 (observation only): score the agent's perception against simulator ground truth.

EVALUATION ONLY. Reads the agent's outputs (per-instruction task_state.json, events.jsonl, result.json) and
ground truth that the agent never sees: the scenario metadata written by sim/run_scenario.py
(out/scenarios/live/laboratory_easy_s42.json) and a /sim/ground_truth/objects snapshot (evaluation/truth_snapshot.py).

    python3 evaluation/lab_exp1_score.py --run out/lab_exp1/<run> \
        --meta out/scenarios/live/laboratory_easy_s42.json --truth out/lab_exp1/<run>/truth.json

Measures (per ground-truth sample container):
  detection     a table-camera measured object within MATCH_XY of the container (instructions 01, 02)
  localisation  xy error and top-height error of the matched measurement
  identity      the same world id in 01 and 02 (re-observation, no motion in between)
  Gemma         cap colour named in Gemma's description; container type (tube / vial) named
  labels        sample ID text anywhere in descriptions, inspections or answers (no label-reading field exists in
                the current perception schema, so a correct ID can only come from free-text answers)
  inspection    wrist-camera inspection reached the intended container; its description vs ground truth
  cost          per-instruction seconds; Gemma calls, seconds and tokens by role
Plus false positives: measured objects that match no container, attributed to the nearest fixture when inside it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

MATCH_XY = 0.030  # m: the world model's own association radius
ID_RE = re.compile(r"\b[A-Z]\d{3}\b")
COLOUR_WORDS = {"red": ["red", "crimson", "scarlet"], "blue": ["blue", "navy"], "green": ["green"],
                "yellow": ["yellow", "gold"], "purple": ["purple", "violet", "magenta", "lilac"],
                "white": ["white", "ivory", "cream"]}


def load_jsonl(p: Path) -> list:
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def instruction_dirs(run: Path) -> list[Path]:
    return sorted(d for d in run.iterdir() if d.is_dir() and re.fullmatch(r"\d\d", d.name))


def fixture_at(meta: dict, xy) -> str | None:
    """Fixture whose footprint (bounding box of its colliders, in its frame) contains xy."""
    for f in meta["fixtures"]:
        a = math.radians(f["yaw_deg"])
        dx, dy = xy[0] - f["pos"][0], xy[1] - f["pos"][1]
        lx, ly = math.cos(a) * dx + math.sin(a) * dy, -math.sin(a) * dx + math.cos(a) * dy
        for p in f["parts"]:
            if not p["collide"]:
                continue
            hx = p["size"][0] / 2
            hy = (p["size"][0] if p["shape"] == "cylinder" else p["size"][1]) / 2
            if abs(lx - p["pos"][0]) <= hx + 0.01 and abs(ly - p["pos"][1]) <= hy + 0.01:
                return f["id"]
    return None


def match(objs: list, truth_xy: dict) -> dict:
    """Greedy nearest matching of measured objects to ground-truth containers within MATCH_XY."""
    pairs = sorted((math.dist(o["centre_m"][:2], xy), gid, i) for gid, xy in truth_xy.items() for i, o in enumerate(objs))
    used_g, used_o, out = set(), set(), {}
    for d, gid, i in pairs:
        if d <= MATCH_XY and gid not in used_g and i not in used_o:
            out[gid] = (i, d)
            used_g.add(gid)
            used_o.add(i)
    return out


def score(run: Path, meta: dict, truth: dict) -> dict:
    gt = {o["id"]: o for o in meta["objects"] if o["category"] in ("sample_tube", "sample_vial")}
    pos = {k: v["position"] for k, v in truth["objects"].items()}
    truth_xy = {k: pos[k][:2] for k in gt if k in pos}
    rep = {"scenario": f"{meta['scenario_id']} {meta['difficulty']} seed {meta['seed']}", "containers": len(gt),
           "instructions": [], "per_container": {k: {"cap_colour": o["attributes"]["cap_colour"],
                                                     "sample_id": o["attributes"]["sample_id"], "type": o["attributes"]["container"]}
                                                 for k, o in gt.items()}}
    world_ids = defaultdict(dict)
    gemma = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "answer_tokens": 0})
    label_hits = defaultdict(list)
    for d in instruction_dirs(run):
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
        st = json.loads((d / "task_state.json").read_text()) if (d / "task_state.json").exists() else {}
        ev = load_jsonl(d / "events.jsonl")
        calls = [e for e in ev if e["kind"] == "vlm"]
        for e in calls:
            g = gemma[e.get("role", "?")]
            g["calls"] += 1
            g["seconds"] += float(e.get("seconds", 0))
            tk = e.get("tokens") or {}
            g["prompt_tokens"] += int(tk.get("prompt", 0))
            g["answer_tokens"] += int(tk.get("answer", 0))
        inst = {"dir": d.name, "instruction": res.get("instruction"), "status": res.get("status"),
                "seconds": res.get("seconds"), "gemma_calls": len(calls),
                "gemma_seconds": round(sum(float(e.get("seconds", 0)) for e in calls), 1)}
        objs = (st.get("facts") or {}).get("objects") or []
        descs = (st.get("hypotheses") or {}).get("descriptions") or {}
        if objs and (st.get("facts") or {}).get("camera") == "table":
            m = match(objs, truth_xy)
            inst["detected"] = len(m)
            matched_marks = set()
            for gid, (i, dist) in m.items():
                o = objs[i]
                matched_marks.add(str(o["mark"]))
                world_ids[gid][d.name] = o.get("world_id")
                top_true = pos[gid][2] + gt[gid]["dimensions"]["height"]
                top_meas = o["centre_m"][2] + o["height_m"] / 2
                desc = descs.get(str(o["mark"])) or {}
                text = " ".join(str(desc.get(k, "")) for k in ("name", "colour", "shape", "material")).lower()
                pc = rep["per_container"][gid].setdefault("observations", {})
                pc[d.name] = {"xy_error_mm": round(dist * 1000, 1), "top_error_mm": round((top_meas - top_true) * 1000, 1),
                              "world_id": o.get("world_id"), "gemma": desc,
                              "cap_colour_named": any(w in text for w in COLOUR_WORDS[gt[gid]["attributes"]["cap_colour"]]),
                              "type_named": gt[gid]["attributes"]["container"] in text}
            fps = [o for o in objs if str(o["mark"]) not in matched_marks]
            inst["unmatched_measurements"] = [{"mark": o["mark"], "centre_mm": [round(v * 1000) for v in o["centre_m"]],
                                               "fixture": fixture_at(meta, o["centre_m"][:2]),
                                               "gemma": (descs.get(str(o["mark"])) or {}).get("name")} for o in fps]
        for e in ev:
            if e["kind"] == "inspection":
                wxy = (e.get("wrist_centre") or e.get("table_centre") or [None, None])[:2]
                near = min(truth_xy, key=lambda g: math.dist(truth_xy[g], wxy)) if wxy[0] is not None else None
                inst.setdefault("inspections", []).append({
                    "target_text": e.get("target"), "nearest_container": near,
                    "distance_mm": round(math.dist(truth_xy[near], wxy) * 1000, 1) if near else None,
                    "description": e.get("description")})
            if e["kind"] == "answer":
                inst.setdefault("answers", []).append({"question": e.get("question"), "answer": e.get("answer"),
                                                       "ids_in_answer": ID_RE.findall(str(e.get("answer", "")))})
            blob = json.dumps({k: v for k, v in e.items() if k in ("description", "answer", "text")})
            for gid, o in gt.items():
                if o["attributes"]["sample_id"] in blob:
                    label_hits[gid].append(f"{d.name}:{e['kind']}")
        rep["instructions"].append(inst)
    rep["identity_stable"] = {g: (len(set(v.values())) == 1 and len(v) >= 2) for g, v in world_ids.items()}
    rep["label_read_correctly"] = dict(label_hits)
    rep["hallucinated_ids"] = sorted({i for inst in rep["instructions"] for a in inst.get("answers", [])
                                      for i in a["ids_in_answer"]} - {o["attributes"]["sample_id"] for o in gt.values()})
    rep["gemma_by_role"] = {k: {**v, "seconds": round(v["seconds"], 1)} for k, v in gemma.items()}
    first = next((i for i in rep["instructions"] if "detected" in i), None)
    if first:
        obs = [c["observations"].get(first["dir"]) for c in rep["per_container"].values() if "observations" in c]
        obs = [o for o in obs if o]
        rep["summary"] = {
            "detected": f"{first['detected']}/{len(gt)}",
            "max_xy_error_mm": max((o["xy_error_mm"] for o in obs), default=None),
            "cap_colour_named": f"{sum(o['cap_colour_named'] for o in obs)}/{len(obs)}",
            "type_named": f"{sum(o['type_named'] for o in obs)}/{len(obs)}",
            "identity_stable": f"{sum(rep['identity_stable'].values())}/{len(rep['identity_stable'])}",
            "unmatched_measurements": len(first.get("unmatched_measurements", [])),
            "labels_read": f"{len(label_hits)}/{len(gt)}",
            "hallucinated_ids": rep["hallucinated_ids"]}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    rep = score(run, json.loads(Path(a.meta).read_text()), json.loads(Path(a.truth).read_text()))
    (run / "score.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep.get("summary", {}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
