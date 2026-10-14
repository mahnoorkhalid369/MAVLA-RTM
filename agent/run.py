"""Run the demo slice on typed instructions (text only).

Order (12 GB GPU): Gemma first, then the YOLO-seg detector, then the simulator. With --start-sim
this process starts Isaac Sim itself after the models are loaded; otherwise start it yourself once
"models loaded" is printed (or when --sim-flag appears, for the rehearsal harness).

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python -m agent.run --start-sim --gui                     # interactive prompt
    .venv-seg/bin/python -m agent.run --instruction "put A on B" --out out/demo/x

Outputs per instruction in --out: events.jsonl, checkpoints.sqlite, images/, result.json.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "out/seg/runs/yolo11s_seg/weights/best.pt"


def start_sim(gui: bool, log: Path) -> subprocess.Popen:
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    cmd = [str(ROOT / ".venv/bin/python"), "-u", "-m", "sim.run_live"] + ([] if gui else ["--headless"])
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_sim(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def run_one(instruction: str, out: Path, bridge, detector, vlm, executor) -> dict:
    from agent.graph import EventLog, Supervisor
    from agent.tools import Tools
    from agent.vlm_roles import Roles

    (out / "images").mkdir(parents=True, exist_ok=True)
    log = EventLog(out / "events.jsonl")
    try:
        tools = Tools(bridge, detector, Roles(vlm, log), executor, log, out)
        report = Supervisor(tools, log, out).run(instruction, thread_id=out.name)
    except Exception as exc:  # an unexpected error is a failure, never a success
        log("error", error=repr(exc))
        report = {"instruction": instruction, "status": "FAILED", "failure": {"cls": "internal", "reason": repr(exc)},
                  "message": f"Failed (internal error): {exc!r}. I did not report success."}
    finally:
        log.close()
    (out / "result.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", help="one instruction (otherwise: interactive prompt)")
    ap.add_argument("--out", default=str(ROOT / "out/demo" / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S")))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--start-sim", action="store_true", help="start Isaac Sim after loading the models")
    ap.add_argument("--gui", action="store_true", help="with --start-sim: show the Isaac Sim window")
    ap.add_argument("--sim-flag", help="write this file when the models are loaded (rehearsal harness)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from agent.vlm_roles import GemmaVLM
    print("[agent] loading Gemma ...", flush=True)
    vlm = GemmaVLM()
    from agent.detector import YoloDetector
    print("[agent] loading the detector ...", flush=True)
    detector = YoloDetector(a.weights, a.imgsz, a.conf)
    print("[agent] models loaded", flush=True)
    if a.sim_flag:
        Path(a.sim_flag).write_text("models loaded\n")
    sim = start_sim(a.gui, out / "sim.log") if a.start_sim else None
    try:
        from agent.ros_bridge import RosBridge
        from execution.ros_executor import RosExecutor
        bridge = RosBridge()
        executor = RosExecutor()
        print("[agent] waiting for the simulator (cameras, joints, clock) ...", flush=True)
        if not (bridge.ready(600) and executor.wait_ready(120)):
            print("[agent] simulator not ready: stopping", flush=True)
            return 2
        # DDS discovery: the first trajectory is delayed if the simulator has not matched our publisher yet
        end = time.monotonic() + 60
        while executor.traj_pub.get_subscription_count() == 0 and time.monotonic() < end:
            executor.spin(0.2)
        executor.wait_sim(3.0)
        print("[agent] ready", flush=True)
        if a.instruction:
            r = run_one(a.instruction, out, bridge, detector, vlm, executor)
            print(f"[agent] {r['status']}: {r['message']}", flush=True)
            return 0
        n = 0
        while True:
            try:
                text = input("instruction (empty line to quit)> ").strip()
            except EOFError:
                break
            if not text:
                break
            n += 1
            r = run_one(text, out / f"{n:02d}", bridge, detector, vlm, executor)
            print(f"[agent] {r['status']}: {r['message']}", flush=True)
        return 0
    finally:
        stop_sim(sim)


if __name__ == "__main__":
    sys.exit(main())
