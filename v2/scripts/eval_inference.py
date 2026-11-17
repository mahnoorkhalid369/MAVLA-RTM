#!/usr/bin/env python3
"""Replay saved Gemma inputs without robot motion. Labelled placement datasets come from eval_pipeline.py.

Compare exact outputs for KV caching; compare verdicts and ground-truth false-yes/false-no for cheaper vision.
Example: python scripts/eval_inference.py --events <run>/events.jsonl --labels <eval>/placement_labels.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from evaluation.benchmark import PROFILES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="independent evaluation labels; agreement alone is not accuracy")
    ap.add_argument("--profiles", nargs="+", choices=["baseline", "no_thinking", "wrist_560", "reasoning_1024", "prefix_cache"],
                    default=["baseline", "no_thinking", "wrist_560", "reasoning_1024", "prefix_cache"])
    ap.add_argument("--out", type=Path, default=ROOT/"out/inference_benchmark")
    ap.add_argument("--manifest-only", action="store_true", help="inspect replay inputs without loading a model")
    a = ap.parse_args()
    events = [json.loads(s) for s in a.events.read_text().splitlines()]
    images = {e["call_id"]: e for e in events if e["kind"] == "model_images"}
    labels = {e["frame_id"]:e["label"] for e in json.loads(a.labels.read_text())} if a.labels else {}
    inputs = []
    for e in events:
        if e["kind"] != "model_input" or e.get("call_id") not in images:
            continue
        frame = next((im.get("frame_id") for im in e.get("images", []) if im), None)
        paths = []
        recorded = images[e["call_id"]]
        for im in recorded["images"]:
            p = Path(im["path"]) if im.get("path") else None
            if p is not None and not p.exists() and (ROOT/p).exists():
                p = ROOT/p
            if p is None or not p.exists():
                raise SystemExit(f"Missing saved model input for call {e['call_id']}: {p}")
            paths.append(str(p.resolve()))
        default_budget = 600 if e.get("role") == "master" else (300 if e.get("phase") == "final_check" else 120)
        inputs.append({"call_id": e["call_id"], "phase": e.get("phase"), "role": e.get("role"),
                       "prompt": e["prompt"], "system": e.get("system"), "images": paths,
                       "thinking": recorded["thinking"],
                       "max_new_tokens": recorded.get("requested_max_new_tokens", default_budget if recorded["thinking"] else recorded["max_new_tokens"]),
                       "label": labels.get(frame) if e.get("phase") == "placement" else None})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/"inputs.json").write_text(json.dumps(inputs, indent=2))
    if a.manifest_only:
        print(f"Prepared {len(inputs)} recorded calls; {sum(i['label'] is not None for i in inputs)} independently labelled.")
        return
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True)
    if gpu.returncode:
        raise SystemExit("No accessible NVIDIA driver; inference benchmark was not started.")
    from agent.gemma import Gemma
    from agent.performance import phase
    from agent.eventlog import EventLog
    gemma = Gemma()
    rows, baseline = [], {}
    for profile in dict.fromkeys(["baseline", *a.profiles]):
        for key in list(os.environ):
            if key.startswith("V2_"):
                del os.environ[key]
        os.environ.update(PROFILES[profile])
        log = EventLog(a.out/f"{profile}.jsonl", echo=False)
        gemma.log, gemma.run_dir = log, a.out/profile
        # Cold and warm passes expose cache reuse and allocator warmup, with equal prompts/images.
        for repeat in (0, 1):
            for item in inputs:
                pics = [Image.open(p).convert("RGB") for p in item["images"]]
                thinking = item["thinking"]
                t0 = time.monotonic()
                with phase("placement" if item["phase"] == "placement" else "default"):
                    answer = gemma.ask(item["prompt"], images=pics, system=item["system"], thinking=thinking,
                                       max_new_tokens=item["max_new_tokens"])
                word = re.split(r"[:\s]", answer.strip().lower(), maxsplit=1)[0]
                if profile == "baseline":
                    baseline[item["call_id"], repeat] = answer
                label = item["label"]
                rows.append({"profile": profile, "call_id": item["call_id"], "repeat": repeat,
                             "seconds": time.monotonic()-t0, "answer": answer,
                             "exact_baseline_match": answer == baseline[item["call_id"], repeat],
                             "label": label, "phase": item["phase"], "false_yes": label is False and word == "yes",
                             "false_no": label is True and word == "no", "unsure": word not in {"yes", "no"}})
        log.close()
        (a.out/"results.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({p: {"calls": len(rs := [r for r in rows if r["profile"]==p]),
                         "mean_s": sum(r["seconds"] for r in rs)/len(rs),
                         "labelled": sum(r["label"] is not None for r in rs),
                         "false_yes": sum(r["false_yes"] for r in rs), "false_no": sum(r["false_no"] for r in rs),
                         "exact_output_matches": sum(r["exact_baseline_match"] for r in rs)}
                      for p in dict.fromkeys(r["profile"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
