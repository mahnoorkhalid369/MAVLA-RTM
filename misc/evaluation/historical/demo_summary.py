"""Summarise rehearsal cycles (evaluation/historical/demo_rehearsal.py summary.json files) into the tables used in
DEMO_REPORT.md: per instruction success vs ground truth with Wilson intervals, false done / false
failed, asked, placement error, time per stage, failures by class and by agent.

    python3 evaluation/historical/demo_summary.py out/demo/2026-10-14/rehearsal/cycle2 [more cycles ...]
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

AGENT_OF_STATE = {"UNDERSTANDING": "supervisor/parser (VLM)", "OBSERVING": "perception", "LOCALIZING": "perception (VLM)",
                  "PLANNING_GRASP": "grasp+motion planner", "APPROACHING": "executor", "GRASPING": "executor",
                  "VERIFYING_GRASP": "verifier", "TRANSPORTING": "executor", "PLACING": "executor",
                  "VERIFYING_PLACEMENT": "verifier", "RETREATING": "executor", "RETURNING_HOME": "executor",
                  "VERIFYING_HOME": "verifier", "RECOVERING": "supervisor"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def failing_state(r: dict) -> str | None:
    tl = r.get("timeline") or []
    states = [t["state"] for t in tl if t["state"] not in ("FAILED", "CANCELLED", "COMPLETED", "RECOVERING")]
    return states[-1] if states and r.get("result_status") in ("FAILED", "CANCELLED") else None


def main(dirs: list[str]) -> str:
    recs = []
    for d in dirs:
        recs += json.loads((Path(d) / "summary.json").read_text())
    by = collections.defaultdict(list)
    for r in recs:
        by[r["instruction"]].append(r)
    lines = ["| Instruction | Runs | Agent COMPLETED | Truth success (95 % Wilson) | False done | False failed | Asked | Placement error vs truth (mm) |",
             "|---|---|---|---|---|---|---|---|"]
    for ins, rs in by.items():
        sc = [r.get("score", {}) for r in rs]
        n = len(rs)
        done = sum(r.get("result_status") == "COMPLETED" for r in rs)
        ok = sum(bool(s.get("truth_success")) for s in sc)
        lo, hi = wilson(ok, n)
        err = [s.get("xy_error_mm") for s in sc if s.get("truth_success") and s.get("xy_error_mm") is not None]
        lines.append(f"| {ins} | {n} | {done} | {ok}/{n} ({lo:.2f}-{hi:.2f}) | {sum(bool(s.get('false_done')) for s in sc)} | "
                     f"{sum(bool(s.get('false_failed')) for s in sc)} | {sum(bool(s.get('asked')) for s in sc)} | "
                     f"{', '.join(f'{e:.1f}' for e in err) or '-'} |")
    stage = collections.defaultdict(list)
    for r in recs:
        for t in r.get("timeline") or []:
            stage[t["state"]].append(t["seconds"])
    lines += ["", "| Stage | n | median s | max s |", "|---|---|---|---|"]
    for s, v in stage.items():
        v = sorted(v)
        lines.append(f"| {s} | {len(v)} | {v[len(v) // 2]:.1f} | {v[-1]:.1f} |")
    fails = collections.Counter()
    for r in recs:
        if r.get("result_status") in ("FAILED", "CANCELLED") or r.get("error"):
            f = r.get("failure") or {}
            st = failing_state(r)
            fails[(f.get("cls", r.get("error", "?")), AGENT_OF_STATE.get(st, st or "-"), (f.get("reason") or r.get("error") or "")[:90])] += 1
    lines += ["", "| Failure class | Agent (state) | Reason | Count |", "|---|---|---|---|"]
    lines += [f"| {c} | {a} | {why} | {k} |" for (c, a, why), k in fails.items()] or ["| - | - | none | 0 |"]
    runs = ["", "| Run | Agent status | Truth | xy / z vs truth (mm) | Agent's verification (xy / z mm, tilt deg) | Total s |", "|---|---|---|---|---|---|"]
    for r in recs:
        s = r.get("score", {})
        v = [x for x in (r.get("verifications") or []) if x.get("check") == "placement"]
        vv = f"{v[-1].get('xy_error_mm')} / {v[-1].get('z_error_mm')}, {v[-1].get('tilt_deg')}" if v else "-"
        runs.append(f"| i{r['key']} run{r['run']} | {r.get('result_status')} | {'success' if s.get('truth_success') else 'fail'} | "
                    f"{s.get('xy_error_mm', '-')} / {s.get('z_error_mm', s.get('gap_mm', '-'))} | {vv} | {r.get('seconds', '-')} |")
    return "\n".join(lines + runs)


if __name__ == "__main__":
    print(main(sys.argv[1:]))
