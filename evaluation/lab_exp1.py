"""Laboratory Experiment 1, observation only (no grasp, no placement): perception baseline in laboratory/easy.

One agent session (agent.run, interactive, unchanged) receives the instructions below through its prompt;
Gemma and YOLO load first, then the simulator starts (12 GB GPU rule). Ground truth is captured separately for
evaluation (evaluation/truth_snapshot.py) and the scenario metadata comes from sim/run_scenario.py; neither is
given to the agent. The only robot motion is the validated wrist-camera viewpoint motion of "inspect"
(collision-checked, back to HOME); --no-motion leaves those instructions out.

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python evaluation/lab_exp1.py --out out/lab_exp1/run1 [--no-motion] [--seed 42]
Then: python3 evaluation/lab_exp1_score.py --run <out> --meta <out>/scenario_metadata.json --truth <out>/truth.json
(run automatically at the end).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# laboratory/easy seed 42: purple caps = tube in B4 and the vial in buffer spot T3; white caps = tubes in A4 (front
# row) and B1. The references are chosen to be unique in that layout (they are words a person would use).
INSTRUCTIONS = [
    ("Describe the objects on the table.", False),
    ("Describe the objects on the table.", False),  # re-observation, no motion in between: identity stability
    ("Inspect the purple-capped vial in the buffer tray.", True),
    ("Inspect the white-capped tube at the front of the incoming rack.", True),
    ("What sample ID is printed on the cap of the purple-capped vial?", False),
    ("Which objects are sample tubes and which are vials?", False),
]
PER_INSTRUCTION_S = 900


def stop(p, sig=signal.SIGINT, wait=30):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, sig)
        p.wait(wait)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-motion", action="store_true", help="skip the wrist-camera inspection instructions")
    a = ap.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "42" or not os.environ.get("AMENT_PREFIX_PATH"):
        print("source /opt/ros/jazzy/setup.bash and export ROS_DOMAIN_ID=42 first", flush=True)
        return 2
    busy = subprocess.run(["pgrep", "-f", "sim.run_live|sim.run_scenario|validate_scenarios"], capture_output=True, text=True)
    if busy.stdout.strip():
        print(f"another Isaac Sim process is running (pids {busy.stdout.split()}): one instance at a time", flush=True)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t, motion in INSTRUCTIONS if not (motion and a.no_motion)]
    (out / "instructions.txt").write_text("\n".join(todo) + "\n")
    flag = out / "models_loaded.flag"
    agent = subprocess.Popen([str(ROOT / ".venv-seg/bin/python"), "-u", "-m", "agent.run", "--out", str(out), "--sim-flag",
                              str(flag), "--ask", "scripted", "--live-dir", ""], cwd=ROOT, stdin=subprocess.PIPE, text=True,
                             stdout=open(out / "agent.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    sim = None
    t0 = time.monotonic()
    try:
        while not flag.exists():
            if agent.poll() is not None or time.monotonic() - t0 > 600:
                print("[exp1] the agent did not load its models", flush=True)
                return 1
            time.sleep(1)
        print(f"[exp1] models loaded after {time.monotonic() - t0:.0f} s; starting laboratory easy seed {a.seed}", flush=True)
        sim = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", "-m", "sim.run_scenario", "--scenario", "laboratory",
                                "--difficulty", "easy", "--seed", str(a.seed), "--headless"], cwd=ROOT,
                               stdout=open(out / "sim.log", "w"), stderr=subprocess.STDOUT,
                               env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, start_new_session=True)
        while "[agent] ready" not in (out / "agent.log").read_text():
            if agent.poll() is not None or sim.poll() is not None or time.monotonic() - t0 > 1500:
                print("[exp1] agent or simulator did not get ready", flush=True)
                return 1
            time.sleep(2)
        meta = ROOT / "out" / "scenarios" / "live" / f"laboratory_easy_s{a.seed}.json"
        shutil.copy(meta, out / "scenario_metadata.json")  # evaluation only
        snap = subprocess.run([str(ROOT / ".venv-seg/bin/python"), "-m", "evaluation.truth_snapshot", str(out / "truth.json")],
                              cwd=ROOT, timeout=120)
        print(f"[exp1] ready after {time.monotonic() - t0:.0f} s; ground truth {'saved' if snap.returncode == 0 else 'MISSING'}",
              flush=True)
        for n, text in enumerate(todo, 1):
            res = out / f"{n:02d}" / "result.json"
            ti = time.monotonic()
            print(f"[exp1] {n:02d} > {text}", flush=True)
            agent.stdin.write(text + "\n")
            agent.stdin.flush()
            while not res.exists():
                if agent.poll() is not None:
                    print("[exp1] the agent exited", flush=True)
                    return 1
                if time.monotonic() - ti > PER_INSTRUCTION_S:
                    print(f"[exp1] {n:02d} timed out after {PER_INSTRUCTION_S} s", flush=True)
                    return 1
                time.sleep(2)
            print(f"[exp1] {n:02d} done in {time.monotonic() - ti:.0f} s", flush=True)
        agent.stdin.write("\n")  # empty line ends the session
        agent.stdin.flush()
        agent.wait(60)
    finally:
        stop(agent)
        stop(sim)
    score = subprocess.run([sys.executable, str(ROOT / "evaluation" / "lab_exp1_score.py"), "--run", str(out), "--meta",
                            str(out / "scenario_metadata.json"), "--truth", str(out / "truth.json")], cwd=ROOT)
    return score.returncode


if __name__ == "__main__":
    raise SystemExit(main())
