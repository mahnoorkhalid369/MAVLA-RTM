"""The robot session: load the models, start Isaac Sim, connect, then run typed instructions one by one.

    ./launch/agent.sh                                   # the normal way: settings from agent/settings.py
    python -m agent.run --settings                      # what agent.sh runs
    python -m agent.run --start-sim --scenario workcell --motion real --instruction "pick the small red block"

Order on the 12 GB GPU: Gemma first, then YOLO, then the simulator. The session only commands Isaac Sim: before
anything can move it checks that the simulator's own node is on the ROS 2 graph. Motion is "blocked" unless
--motion real (or MOTION in agent/settings.py). One process = one session: the world model and the held object carry
over between instructions. Each instruction writes <out>/<nn>/: events.jsonl, result.json, images/.
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

ROOT = Path(__file__).resolve().parents[2]  # this version's folder (v1/ or v2/)
SIMULATOR_NODE, SIMULATOR_TOPIC = "vt6l_sim", "/safety/events"  # published by the simulator (sim/ros_control.py)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", action="store_true", help="take every default from agent/settings.py")
    instruction_group = ap.add_mutually_exclusive_group()
    instruction_group.add_argument("--instruction", help="run one instruction, then exit (otherwise: interactive prompt)")
    instruction_group.add_argument("--instructions-file", type=Path, help="scripted user requests/clarifications in one persistent session")
    ap.add_argument("--evaluation-setup", type=Path, help="forward initial scene variation to the simulator only")
    ap.add_argument("--out", default=str(ROOT / "out/demo" / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S")))
    ap.add_argument("--start-sim", action="store_true", help="start Isaac Sim after loading the models")
    ap.add_argument("--gui", action="store_true", help="with --start-sim: show the Isaac Sim window")
    ap.add_argument("--scenario", default="workcell")
    ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion", default="blocked", choices=["blocked", "dry_run", "real"])
    ap.add_argument("--gemma", default=None, help="local Gemma model folder (default: vlm/config.py)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--identity-check", action=argparse.BooleanOptionalAction, default=True,
                    help="SigLIP 2 checks the wrist close-up against the catalogue names (--no-identity-check: off)")
    ap.add_argument("--live-dir", default=str(ROOT / "out/demo/live"), help="files the dashboard reads ('' = off)")
    return ap


def start_sim(a, log: Path) -> subprocess.Popen:
    cmd = [str(ROOT / ".venv/bin/python"), "-u", "-m", "sim.run_scenario", "--scenario", a.scenario,
           "--difficulty", a.difficulty, "--seed", str(a.seed)] + ([] if a.gui else ["--headless"])
    if getattr(a, "evaluation_setup", None):
        cmd += ["--evaluation-setup", str(a.evaluation_setup.resolve())]
    with log.open("w") as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, start_new_session=True)
    # Lets the evaluation runner clean up only this simulator if the agent is killed.
    stat = Path(f"/proc/{process.pid}/stat")
    if stat.exists():
        ticks = stat.read_text().rsplit(")", 1)[1].split()[19]
        log.with_suffix(".process.json").write_text(json.dumps({"pid": process.pid, "start_ticks": ticks}))
    return process


def stop_sim(p) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def connect(bridge, executor) -> bool:
    print("[agent] waiting for the simulator (cameras, joints, clock) ...", flush=True)
    if not (bridge.ready(600) and executor.wait_ready(120)):
        print("[agent] simulator not ready: stopping", flush=True)
        return False
    end = time.monotonic() + 60  # DDS discovery: the simulator must have matched our trajectory publisher
    while executor.traj_pub.get_subscription_count() == 0 and time.monotonic() < end:
        executor.spin(0.2)
    executor.wait_sim(3.0)
    end = time.monotonic() + 30  # the graph must be the simulator, never a physical robot
    while time.monotonic() < end:
        if SIMULATOR_NODE in {i.node_name for i in bridge.node.get_publishers_info_by_topic(SIMULATOR_TOPIC)}:
            print(f"[agent] connected to Isaac Sim (ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID', '0')})", flush=True)
            return True
        time.sleep(0.5)
    print(f"[agent] the ROS 2 graph is not the Isaac Sim simulator (no {SIMULATOR_NODE}): stopping", flush=True)
    return False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = parser()
    if "--settings" in argv:
        from agent import settings
        problems = settings.validate()
        if problems:
            print("[agent] agent/settings.py has problems:\n  - " + "\n  - ".join(problems), flush=True)
            return 2
        print(settings.summary(), flush=True)
        ap.set_defaults(**settings.run_defaults())
    a = ap.parse_args(argv)
    scripted = json.loads(a.instructions_file.read_text()) if a.instructions_file else None
    if scripted is not None and (not isinstance(scripted, list) or not scripted or any(
            not isinstance(item, dict) or not isinstance(item.get("request"), str) or not item["request"].strip()
            or not isinstance(item.get("answers", []), list)
            or not all(isinstance(answer, str) for answer in item.get("answers", [])) for item in scripted)):
        ap.error("instructions file must be a nonempty list of {request: text, answers: [text, ...]}")
    if a.start_sim:  # perception must use the camera pose and regions of the scene the simulator renders
        import scenarios
        if (os.environ.get("TABLE_CAMERA", ""), os.environ.get("WORKCELL_LAYOUT", "")) != \
                (scenarios.table_camera(a.scenario), scenarios.layout(a.scenario)):
            print(f"[agent] TABLE_CAMERA / WORKCELL_LAYOUT do not match scenario {a.scenario!r}: start through "
                  "./launch/agent.sh", flush=True)
            return 2

    from agent.controls import ExecutionControls
    from agent.eventlog import EventLog
    from agent.gemma import Gemma
    from agent.live import LiveChannel
    from agent import catalogue
    from agent.loop import run_instruction
    from agent.tools import Tools
    from config import models
    from perception.detector import YoloDetector

    controls = ExecutionControls(motion=a.motion)
    print(f"[agent] motion: {a.motion.upper()}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("[agent] loading Gemma ...", flush=True)
    gemma = Gemma(a.gemma)
    print("[agent] loading the detector ...", flush=True)
    detector = YoloDetector(a.weights or str(models.YOLO_WEIGHTS), a.imgsz or models.DETECTOR_IMGSZ,
                            a.conf or models.DETECTOR_CONF)
    matcher = None
    if a.identity_check:
        print("[agent] loading SigLIP 2 (identity check) ...", flush=True)
        from perception.matcher import Matcher
        matcher = Matcher()
    print("[agent] models loaded", flush=True)
    sim = start_sim(a, out / "sim.log") if a.start_sim else None
    try:
        from agent.ros_bridge import RosBridge
        from execution.ros_executor import RosExecutor
        bridge, executor = RosBridge(), RosExecutor()
        if not connect(bridge, executor):
            return 2
        print("[agent] ready", flush=True)
        live = LiveChannel(a.live_dir) if a.live_dir else None
        # the session's catalogue, before the first instruction: ids 1..N, and a Gemma name for each object
        start_dir = out / "00_catalogue"
        (start_dir / "images").mkdir(parents=True, exist_ok=True)
        log = EventLog(start_dir / "events.jsonl", echo=False)  # printed below, with sizes
        gemma.log, gemma.run_dir = log, start_dir
        if live:
            live.start_run(start_dir, "(naming the objects on the table)")
        tools = Tools(bridge, detector, gemma, executor, log, start_dir, live, controls)
        tools.matcher = matcher
        tools.revalidate_held()
        print("[agent] naming the objects on the table ...", flush=True)
        try:
            for line in catalogue.build(tools, gemma, matcher, log):
                print(f"[CATALOGUE   ] {line}", flush=True)
        except Exception as exc:  # the instructions still run; objects are then named when first grounded
            log("error", error=repr(exc))
            print(f"[agent] catalogue failed: {exc!r}", flush=True)
        finally:
            log.close()
        n = 0
        while True:
            if scripted is not None:
                text = scripted[n]["request"] if n < len(scripted) else ""
            elif a.instruction:
                text = a.instruction if n == 0 else ""
            else:
                try:
                    text = input("instruction (empty line to quit)> ").strip()
                except EOFError:
                    text = ""
            if not text:
                return 0
            n += 1
            run_dir = out / f"{n:02d}"
            (run_dir / "images").mkdir(parents=True, exist_ok=True)
            log = EventLog(run_dir / "events.jsonl")
            if live:
                live.start_run(run_dir, text)
            tools.begin_instruction(log, run_dir)
            log("instruction", text=text)
            try:
                if scripted is not None:
                    answers = iter(scripted[n-1].get("answers", []))
                    def answer_user(question):
                        try:
                            answer = next(answers)
                        except StopIteration:
                            raise EOFError()
                        log("scripted_user_answer", question=question, answer=answer)
                        print(f"[SCRIPTED USER] {answer}", flush=True)
                        return answer
                    result = run_instruction(text, tools, gemma, log, run_dir, ask_user=answer_user)
                    if result["status"] != "COMPLETED":
                        return 1  # a failed setup request must not masquerade as a completed session test
                else:
                    run_instruction(text, tools, gemma, log, run_dir)
            except Exception as exc:  # an unexpected error is a failure, never a success
                log("error", error=repr(exc))
                print(f"[RESULT      ] FAILED (internal error): {exc!r}", flush=True)
            finally:
                log.close()
    finally:
        stop_sim(sim)


if __name__ == "__main__":
    sys.exit(main())
