"""Offline benchmark (evaluation): describe every object one call at a time ("single"), batched ("batched"), or
via a one-image shortlist ("shortlist", rejected 2026-10-14) on
today's capture, with the round-1 detector and the real Gemma. Ground truth (mask -> scene name) only
scores the answers. Same Tools.resolve path (lexical cross-check, comparison) as the live agent.

    .venv-seg/bin/python -m evaluation.demo_shortlist_bench --out out/perf/shortlist_bench.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = [("red cube", "green cube"), ("yellow cube", "tan box"), ("orange cube", "purple cylinder"),
         ("pink cube", "white tall cylinder")]
TRUTH = {"red cube": "red_cube", "green cube": "green_cube", "yellow cube": "yellow_cube", "tan box": "tan_box_wide",
         "orange cube": "orange_cube", "purple cylinder": "purple_cylinder", "pink cube": "pink_cube",
         "white tall cylinder": "white_tall_cylinder"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=["single", "batched"])
    a = ap.parse_args()
    from agent.detector import YoloDetector
    from agent.somark import draw_marks
    from agent.tools import Observation, Tools
    from agent.vlm_roles import GemmaVLM, Roles
    from seg.evaluate import iou_matrix

    events = []
    log = lambda kind, **f: events.append({"kind": kind, **f})  # noqa: E731
    vlm = GemmaVLM()
    det = YoloDetector(str(ROOT / "out/seg/runs/yolo11s_seg_r1/weights/best.pt"))
    d = ROOT / "data/perception/eval_layouts/today"
    A = json.loads((d / "annotations.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(d / "images/today_000.png")), cv2.COLOR_BGR2RGB)
    ids = cv2.imread(str(d / "masks/today_000.png"), cv2.IMREAD_UNCHANGED)
    marks = {k + 1: x.mask for k, x in enumerate(sorted(det.detect(rgb), key=lambda x: -x.score))}
    names = [x["name"] for x in A["annotations"]]
    M = iou_matrix(list(marks.values()), np.stack([ids == x["object_uid"] for x in A["annotations"]]))
    truth = {k: (names[int(np.argmax(M[i]))] if M[i].max() >= 0.5 else None) for i, k in enumerate(marks)}
    run_dir = Path(a.out).parent / "shortlist_bench_images"
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    report = {"objects": len(marks), "modes": {}}
    import torch

    class Single:  # the same model without the batched path
        def __init__(self, v):
            self.v = v

        def generate(self, *a, **k):
            return self.v.generate(*a, **k)

    for mode in a.modes:
        model = Single(vlm) if mode == "single" else vlm
        torch.cuda.reset_peak_memory_stats()
        rows, calls_before, t_mode = [], 0, 0.0
        for obj, ref in PAIRS:
            events.clear()
            tools = Tools(None, None, Roles(model, log), None, log, run_dir)
            obs = Observation("today", "table", rgb, draw_marks(rgb, marks), dict(marks), {k: {} for k in marks})
            t0 = time.monotonic()
            if mode == "shortlist":
                tools.shortlist(obs, [obj, ref])
            else:
                tools.describe(obs)
            t_desc = time.monotonic() - t0
            descs = dict(obs.descriptions)
            for text in (obj, ref):
                r = tools.resolve(obs, text)
                sl = obs.shortlists.get(text)
                rows.append({"text": text, "resolved": r.id, "resolved_truth": truth.get(r.id), "ambiguous": r.ambiguous,
                             "candidates": r.candidates, "correct": truth.get(r.id) == TRUTH[text],
                             "shortlist": sl, "shortlist_has_truth": None if sl is None else any(truth.get(k) == TRUTH[text] for k in sl)})
            dt = time.monotonic() - t0
            t_mode += dt
            vl = [e for e in events if e["kind"] == "vlm"]
            rows[-1]["pair_seconds"] = round(dt, 1)
            rows[-1]["pair_vlm_calls"] = len(vl)
            rows[-1]["pair_vlm_seconds"] = round(sum(e["seconds"] for e in vl), 1)
            rows[-1]["describe_seconds"] = round(t_desc, 1)
            rows[-1]["descriptions"] = {k: {**v, "truth": truth[k]} for k, v in descs.items()}
            print(mode, obj, "/", ref, f"{dt:.1f} s, {len(vl)} VLM calls", [(x["text"], x["resolved_truth"], x["shortlist"]) for x in rows[-2:]], flush=True)
        report["modes"][mode] = {"rows": rows, "seconds_total": round(t_mode, 1),
                                 "correct": sum(r["correct"] for r in rows), "asked": sum(r["ambiguous"] for r in rows),
                                 "wrong": sum((not r["correct"]) and (not r["ambiguous"]) for r in rows), "n": len(rows),
                                 "vlm_calls": sum(r.get("pair_vlm_calls", 0) for r in rows),
                                 "describe_seconds": [r["describe_seconds"] for r in rows if "describe_seconds" in r],
                                 "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    for m, v in report["modes"].items():
        print(m, {k: v[k] for k in ("seconds_total", "vlm_calls", "correct", "asked", "wrong", "n", "describe_seconds", "gpu_peak_gb")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
