"""Acts as the operator for a full demo test: runs ./launch/start_demo.sh, types one instruction, answers a
clarification question if one appears, then quits. The driver chooses the candidate whose printed
position is nearest to --pick-near (mm, the operator's knowledge of which object is meant: test only).

    DEMO_VIEWER_ARGS="--snapshot-every 3 --snapshot-dir <dir>" .venv-seg/bin/python -m evaluation.demo_live_driver \
        --instruction "..." --pick-near 100 720 --console <file>
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True, action="append", help="repeat for several instructions in one session")
    ap.add_argument("--pick-near", type=float, nargs=2, required=True)
    ap.add_argument("--console", required=True)
    ap.add_argument("--launcher-args", default="", help='e.g. "--headless --scenario ap_occlusion" (one quoted string)')
    ap.add_argument("--timeout", type=float, default=3000)
    ap.add_argument("--launcher", default="launch/start_demo.sh")
    ap.add_argument("--gpu-log", help="log nvidia-smi memory / utilisation every 2 s to this file")
    ap.add_argument("--truth-out", help="save one simulator ground-truth message here once the agent is ready (evaluation only)")
    a = ap.parse_args()
    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu", "--format=csv,noheader", "-l", "2"],
                           stdout=open(a.gpu_log, "w")) if a.gpu_log else None
    launcher = Path(a.launcher)
    p = subprocess.Popen([str(launcher if launcher.is_absolute() else ROOT / launcher), *a.launcher_args.split()],
                         cwd=(launcher.parents[1] if launcher.is_absolute() else ROOT), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = open(a.console, "w")
    t0, cands, answered, finals = time.time(), [], [], []
    todo = list(a.instruction)
    for line in p.stdout:
        log.write(line)
        log.flush()
        if time.time() - t0 > a.timeout:
            print("driver timeout", flush=True)
            break
        if "[agent] ready" in line and a.truth_out:
            subprocess.Popen(["bash", "-c", f"source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=${{ROS_DOMAIN_ID:-42}} && "
                              f"{ROOT}/.venv-seg/bin/python -m evaluation.truth_snapshot {a.truth_out}"], cwd=ROOT)
        if "[agent] ready" in line and todo:
            p.stdin.write(todo.pop(0) + "\n")
            p.stdin.flush()
        if "QUESTION" in line:
            cands = []
        m = re.match(r"\s+(\d+): .* at x ([+-]\d+) mm, y ([+-]\d+) mm", line)
        if m:
            cands.append((int(m[1]), float(m[2]), float(m[3])))
        if line.strip().startswith(("Answer with a number", "Answer yes")):
            k = min(cands, key=lambda c: (c[1] - a.pick_near[0]) ** 2 + (c[2] - a.pick_near[1]) ** 2)[0]
            p.stdin.write(f"{k}\n")
            p.stdin.flush()
            answered.append(k)
        if re.search(r"\[agent\] (COMPLETED|FAILED|CANCELLED)", line):
            finals.append(line.strip())
            p.stdin.write((todo.pop(0) if todo else "") + "\n")  # next instruction, or quit
            p.stdin.flush()
    p.wait(60)
    if gpu:
        gpu.terminate()
    print("finals:\n  " + "\n  ".join(finals) + f"\nanswered: {answered}\nlauncher exit: {p.returncode}\nseconds: {time.time() - t0:.0f}")
    return 0 if finals and all("COMPLETED" in f for f in finals) else 1


if __name__ == "__main__":
    sys.exit(main())
