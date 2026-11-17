#!/usr/bin/env python3
"""Two-stage live review: bounded smoke, then a gated small paired subset. No experimental profiles."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from evaluation.benchmark import cases, compare
from evaluation.smoke import smoke_checks, table


def load_runner():
    spec = importlib.util.spec_from_file_location("pipeline_runner", ROOT/"scripts/eval_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["smoke", "subset"])
    ap.add_argument("--budget", type=float, default=900, help="total smoke execution budget, including startup; cleanup can add time")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--domain", type=int, default=88)
    ap.add_argument("--smoke-report", type=Path, help="subset requires both smoke profiles to have passed automatic checks")
    ap.add_argument("--images-reviewed", action="store_true", help="operator confirmed sharp, stationary wrist pictures in the smoke report")
    a = ap.parse_args()
    if a.budget <= 0 or not 0 <= a.domain <= 232:
        ap.error("invalid budget or ROS domain")
    if a.mode == "subset":
        if not a.smoke_report or not a.images_reviewed:
            ap.error("subset requires --smoke-report <smoke-report.json> and --images-reviewed after inspecting its pictures")
        gate = json.loads(a.smoke_report.read_text())
        if not gate.get("automatic_checks_pass") or {r["profile"] for r in gate.get("runs", [])} != {"baseline", "deduplicated"}:
            ap.error("smoke is incomplete or failed; do not run the paired subset yet")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    if gpu.returncode:
        raise SystemExit("GPU unavailable: smoke/subset NOT STARTED. No live result or pass was produced.")
    folder = (a.out or ROOT/"out/pipeline_review"/time.strftime("%Y%m%d_%H%M%S",time.gmtime())/a.mode).resolve()
    if folder.exists():
        ap.error(f"refusing to overwrite {folder}")
    folder.mkdir(parents=True)
    runner = load_runner()
    wanted = (["easy_s42_apple"] if a.mode=="smoke" else
              ["easy_s42_apple", "easy_s42_two_items", "easy_s42_impossible", "easy_s42_on_the_tray", "easy_s42_occupied_tray"])
    profiles = ["baseline", "deduplicated"] + (["headless"] if a.mode=="subset" else [])
    catalog = {c["id"]:c for c in cases("all")}
    rows, reviews = [], []
    deadline = time.monotonic()+a.budget if a.mode=="smoke" else None
    for key in wanted:
        for profile in profiles:
            remaining = deadline-time.monotonic() if deadline else 2400
            if remaining <= 0:
                print("SMOKE INCOMPLETE: configured budget exhausted before both profiles finished.")
                break
            case_dir = folder/profile/key
            print(f"[{a.mode.upper()}] starting {key}: {profile}; output: {case_dir}", flush=True)
            row = runner.collect_isolated(catalog[key], profile, case_dir, a.domain, remaining)
            rows.append(row)
            if a.mode=="smoke":
                events = runner.session_events(case_dir/"agent")
                check = smoke_checks(row, events, ROOT)
                reviews.append({"profile":profile, "status":row.get("score", {}).get("status"), **check})
                print(f"[{profile}] timing={'PASS' if check['timing_complete'] else 'FAIL'}; "
                      f"motion steps={check['motion_steps']}; settling captures={check['successful_settling_captures']}; "
                      f"task={row.get('score', {})}", flush=True)
                for e in events:
                    if e.get("kind")=="tool" and e.get("tool")=="execute":
                        for step in e.get("steps", []):
                            print(f"  {step.get('label')}: planned={step.get('planned_s')} sim={step.get('sim_elapsed_s')} "
                                  f"wall={step.get('wall_s')} settling={step.get('settling_wall_s', 'gripper command')}")
                for image in check["pictures"]:
                    print("  REVIEW WRIST:", image["path"])
                passed = len(reviews)==2 and all(r["automatic_checks_pass"] for r in reviews)
                (folder/"smoke-report.json").write_text(json.dumps({"automatic_checks_pass":passed, "runs":reviews,
                    "status":"AUTOMATIC PASS; VISUAL REVIEW REQUIRED" if passed else "INCOMPLETE OR FAILED",
                    "budget_s":a.budget, "images_reviewed":False}, indent=2))
            (folder/"comparison.json").write_text(json.dumps(compare(rows),indent=2))
            (folder/"table.md").write_text(table(rows))
        if deadline and time.monotonic() >= deadline:
            break
    print(table(rows))
    if a.mode=="smoke":
        passed = len(reviews)==2 and all(r["automatic_checks_pass"] for r in reviews)
        print("SMOKE: AUTOMATIC PASS; inspect wrist images before step 4." if passed else
              "SMOKE: INCOMPLETE OR FAILED; step 4 remains blocked. Increase --budget only if startup/runtime exceeded it.")
        print("Report:", folder/"smoke-report.json")
        return 0 if passed else 1
    print("Paired subset table:",folder/"table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
