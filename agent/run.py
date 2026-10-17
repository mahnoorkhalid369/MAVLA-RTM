"""Run the demo slice on typed instructions (text only).

Order (12 GB GPU): Gemma first, then the YOLO-seg detector, then the simulator. With --start-sim
this process starts Isaac Sim itself after the models are loaded; otherwise start it yourself once
"models loaded" is printed (or when --sim-flag appears, for the rehearsal harness).

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python -m agent.run --start-sim --gui                     # interactive prompt
    .venv-seg/bin/python -m agent.run --instruction "put A on B" --out out/demo/x

Outputs per instruction in --out: events.jsonl, checkpoints.sqlite, images/, result.json.
Instructions typed in one interactive process form one session (see Session): follow-ups like "put it ..." refer
to the previous instruction's object; a new process starts a new, empty session.
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

from config.models import DETECTOR_CONF, DETECTOR_IMGSZ, YOLO_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]


def start_sim(gui: bool, log: Path, scenario: str | None = None) -> subprocess.Popen:
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    mod = ["sim.run_scenario", "--scenario", scenario] if scenario else ["sim.run_live"]
    cmd = [str(ROOT / ".venv/bin/python"), "-u", "-m", *mod] + ([] if gui else ["--headless"])
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_sim(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


class Session:
    """One robot session = one agent process with one simulator connection. The Tools object - world model,
    held object, last referent, remembered clarifications - lives as long as the session, so a follow-up such as
    "put it in the processing rack" can refer to the previous instruction's object. Each instruction still gets
    its own event log, VLM roles (call counters), supervisor, task ledger, goal conditions and LangGraph thread.
    The held-object claim is re-verified from finger feedback at the start of every instruction (and again before
    a place). Nothing is written to disk for the next session: a new process starts with an empty world and
    nothing held."""

    def __init__(self, bridge, detector, vlm, executor, responder, live=None, memory=None) -> None:
        self.id = time.strftime("session_%Y%m%d_%H%M%S")
        self.bridge, self.detector, self.vlm, self.executor = bridge, detector, vlm, executor
        self.responder, self.live, self.memory = responder, live, memory
        self.tools = None
        self.count = 0

    def run(self, instruction: str, out: Path) -> dict:
        from agent.eventlog import EventLog
        from agent.supervisor import GoalSupervisor
        from agent.tools import Tools
        from agent.vlm_roles import Roles

        (out / "images").mkdir(parents=True, exist_ok=True)
        log = EventLog(out / "events.jsonl", echo=True)
        self.count += 1
        if self.live:
            self.live.start_run(out, instruction)
        try:
            log("session", session_id=self.id, instruction_index=self.count, reused_state=self.tools is not None)
            roles = Roles(self.vlm, log)
            if self.tools is None:
                self.tools = Tools(self.bridge, self.detector, roles, self.executor, log, out, self.live, self.memory)
                self.tools.revalidate_held()
            else:
                self.tools.begin_instruction(log, roles, out)
            report = GoalSupervisor(self.tools, log, out, self.responder).run(instruction, thread_id=f"{self.id}_{self.count:02d}")
        except Exception as exc:  # an unexpected error is a failure, never a success
            log("error", error=repr(exc))
            report = {"instruction": instruction, "status": "FAILED", "failure": {"cls": "internal", "reason": repr(exc)},
                      "message": f"Failed (internal error): {exc!r}. I did not report success."}
        finally:
            log.close()
        report["session"] = {"id": self.id, "instruction_index": self.count}
        (out / "result.json").write_text(json.dumps(report, indent=1, default=str))
        return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", help="one instruction (otherwise: interactive prompt)")
    ap.add_argument("--out", default=str(ROOT / "out/demo" / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S")))
    ap.add_argument("--weights", default=str(YOLO_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    ap.add_argument("--conf", type=float, default=DETECTOR_CONF)
    ap.add_argument("--start-sim", action="store_true", help="start Isaac Sim after loading the models")
    ap.add_argument("--gui", action="store_true", help="with --start-sim: show the Isaac Sim window")
    ap.add_argument("--scenario", help="with --start-sim: an active-perception test scene (sim/scenarios/<name>.json)")
    ap.add_argument("--sim-flag", help="write this file when the models are loaded (rehearsal harness)")
    ap.add_argument("--ask", choices=["terminal", "scripted"], default="terminal",
                    help="who answers clarification questions: the person at this terminal (default) or, for "
                         "unattended rehearsals only, a scripted 'cancel'")
    ap.add_argument("--no-reuse", action="store_true", help="describe every object again for each instruction")
    ap.add_argument("--live-dir", default=str(ROOT / "out/demo/live"),
                    help="where the optional demo viewer reads the latest perception and the active run ('' = off)")
    ap.add_argument("--ask-timeout", type=float, default=None,
                    help="seconds to wait for an answer before cancelling (default Budgets.question_timeout_s; 0 = no limit)")
    a = ap.parse_args()
    out = Path(a.out)
    from agent.ask import LineReader, ScriptedResponder, TerminalResponder
    from config.agent import Budgets
    timeout = Budgets().question_timeout_s if a.ask_timeout is None else a.ask_timeout
    responder = TerminalResponder(timeout_s=timeout) if a.ask == "terminal" else ScriptedResponder(["cancel"])
    from agent.live import LiveChannel
    from agent.tools import DescriptionMemory
    live = LiveChannel(a.live_dir) if a.live_dir else None
    memory = None if a.no_reuse else DescriptionMemory()  # descriptions reused across this session's instructions
    out.mkdir(parents=True, exist_ok=True)

    from agent.vlm_roles import GemmaVLM
    print("[agent] loading Gemma ...", flush=True)
    vlm = GemmaVLM()
    from perception.detector import YoloDetector
    print("[agent] loading the detector ...", flush=True)
    detector = YoloDetector(a.weights, a.imgsz, a.conf)
    print("[agent] models loaded", flush=True)
    if a.sim_flag:
        Path(a.sim_flag).write_text("models loaded\n")
    sim = start_sim(a.gui, out / "sim.log", a.scenario) if a.start_sim else None
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
        session = Session(bridge, detector, vlm, executor, responder, live, memory)
        if a.instruction:
            r = session.run(a.instruction, out)
            print(f"[agent] {r['status']}: {r['message']}", flush=True)
            return 0
        n = 0
        while True:
            print("instruction (empty line to quit)> ", end="", flush=True)
            text = LineReader.shared().get()  # same reader as the clarification questions (shared stdin)
            if text is None or not text.strip():
                break
            text = text.strip()
            n += 1
            r = session.run(text, out / f"{n:02d}")
            print(f"[agent] {r['status']}: {r['message']}", flush=True)
        return 0
    finally:
        stop_sim(sim)


if __name__ == "__main__":
    sys.exit(main())
