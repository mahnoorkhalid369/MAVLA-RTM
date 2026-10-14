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
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--pick-near", type=float, nargs=2, required=True)
    ap.add_argument("--console", required=True)
    ap.add_argument("--launcher-args", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=1500)
    a = ap.parse_args()
    p = subprocess.Popen([str(ROOT / "launch/start_demo.sh"), *a.launcher_args], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = open(a.console, "w")
    t0, sent, cands, answered, final = time.time(), False, [], [], None
    for line in p.stdout:
        log.write(line)
        log.flush()
        if time.time() - t0 > a.timeout:
            print("driver timeout", flush=True)
            break
        if "[agent] ready" in line and not sent:
            p.stdin.write(a.instruction + "\n")
            p.stdin.flush()
            sent = True
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
            final = line.strip()
            p.stdin.write("\n")  # quit
            p.stdin.flush()
    p.wait(60)
    print(f"final: {final}\nanswered: {answered}\nlauncher exit: {p.returncode}\nseconds: {time.time() - t0:.0f}")
    return 0 if final and "COMPLETED" in final else 1


if __name__ == "__main__":
    sys.exit(main())
