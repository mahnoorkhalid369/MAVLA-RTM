"""Run the demo slice on typed instructions (text only).

Order (12 GB GPU): Gemma first, then the YOLO-seg detector, then the simulator. With --start-sim
this process starts Isaac Sim itself after the models are loaded; otherwise start it yourself once
"models loaded" is printed (or when --sim-flag appears, for the rehearsal harness).

    source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
    .venv-seg/bin/python -m agent.run --start-sim --gui                     # interactive prompt, motion BLOCKED
    .venv-seg/bin/python -m agent.run --start-sim --scenario laboratory --difficulty easy --seed 42 --motion real
    .venv-seg/bin/python -m agent.run --instruction "put A on B" --out out/demo/x
(launch/agent.sh wraps this with the viewer.) Motion is blocked unless --motion real is given.

Outputs per instruction in --out: events.jsonl, checkpoints.sqlite, images/, result.json.
Instructions typed in one interactive process form one session (agent/session.py): follow-ups like "put it ..." refer
to the previous instruction's object; a new process starts a new, empty session.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agent.session import Session
from config.models import DETECTOR_CONF, DETECTOR_IMGSZ, YOLO_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]


def sim_command(gui: bool, scenario: str | None = None, difficulty: str = "easy", seed: int = 0) -> list[str]:
    """The simulator process. A named scenario (any of scenarios.available(), "default" included) starts through
    sim.run_scenario; no scenario starts the frozen sim.run_live directly (harnesses). Difficulty and seed apply to
    procedural scenarios such as laboratory."""
    mod = ["sim.run_scenario", "--scenario", scenario, "--difficulty", difficulty, "--seed", str(seed)] if scenario else ["sim.run_live"]
    return [str(ROOT / ".venv/bin/python"), "-u", "-m", *mod] + ([] if gui else ["--headless"])


def start_sim(gui: bool, log: Path, scenario: str | None = None, difficulty: str = "easy", seed: int = 0) -> subprocess.Popen:
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    cmd = sim_command(gui, scenario, difficulty, seed)
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop_sim(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


SIMULATOR_TOPIC = "/safety/events"  # published by the simulator's command interface (frozen sim/ros_control.py)
SIMULATOR_NODE = "vt6l_sim"         # ... whose ROS 2 node has this name


def simulator_on_graph(publisher_nodes) -> bool:
    """True if the Isaac Sim command interface publishes on the connected ROS 2 graph (a publisher node named
    vt6l_sim on /safety/events). A physical robot driver is not that node, so a session that targets Isaac Sim never
    commands hardware by mistake. (The executor subscribes to that topic itself, so the topic alone proves nothing.)"""
    return SIMULATOR_NODE in set(publisher_nodes)


def simulator_publishers(node) -> list[str]:
    return [i.node_name for i in node.get_publishers_info_by_topic(SIMULATOR_TOPIC)]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", action="store_true",
                    help="take every default from agent/settings.py (validated first); launch/agent.sh uses this")
    ap.add_argument("--instruction", help="one instruction (otherwise: interactive prompt)")
    ap.add_argument("--out", default=str(ROOT / "out/demo" / time.strftime("%Y-%m-%d") / time.strftime("live_%H%M%S")))
    ap.add_argument("--weights", default=str(YOLO_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    ap.add_argument("--conf", type=float, default=DETECTOR_CONF)
    ap.add_argument("--start-sim", action="store_true", help="start Isaac Sim after loading the models")
    ap.add_argument("--gui", action="store_true", help="with --start-sim: show the Isaac Sim window")
    ap.add_argument("--headless", dest="gui", action="store_false", help="with --start-sim: no Isaac Sim window")
    ap.add_argument("--scenario", help="with --start-sim: a scenario registered in scenarios/ (default, laboratory, "
                                       "ap_occlusion, ...); none: the frozen sim.run_live directly")
    ap.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"], help="with --scenario laboratory")
    ap.add_argument("--seed", type=int, default=0, help="with --scenario laboratory: layout seed")
    ap.add_argument("--motion", default="blocked", choices=["blocked", "dry_run", "real"],
                    help="robot motion (agent/controls.py): blocked (default: nothing is sent), dry_run (planned and "
                         "logged, not sent) or real. Authorization and the simulator's safety layer apply in every mode")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="Gemma decisions per instruction after the first plan (default: agent/controls.py)")
    ap.add_argument("--sim-flag", help="write this file when the models are loaded (rehearsal harness)")
    ap.add_argument("--ask", choices=["terminal", "scripted"], default="terminal",
                    help="who answers clarification questions: the person at this terminal (default) or, for "
                         "unattended rehearsals only, a scripted 'cancel'")
    ap.add_argument("--no-reuse", action="store_true", help="describe every object again for each instruction")
    ap.add_argument("--live-dir", default=str(ROOT / "out/demo/live"),
                    help="where the optional demo viewer reads the latest perception and the active run ('' = off)")
    ap.add_argument("--gemma", default=None, help="local Gemma model folder (default: vlm/config.py)")
    ap.add_argument("--trace", action="store_true", help="print every agent handoff (delegation input, worker output, "
                                                         "authorization, bindings, goal checks) in the terminal")
    ap.add_argument("--target", default="isaac_sim", choices=["isaac_sim"],
                    help="execution target; only Isaac Sim is supported (physical-robot execution is not part of this entry point)")
    ap.add_argument("--ask-timeout", type=float, default=None,
                    help="seconds to wait for an answer before cancelling (default Budgets.question_timeout_s; 0 = no limit)")
    return ap


def controls_for(args, controls=None):
    """The execution controls of this process: the harness's own, or the --motion mode (default blocked)."""
    from agent.controls import ExecutionControls
    if controls is not None:
        return controls
    budget = getattr(args, "max_decisions", None)
    return ExecutionControls(motion=args.motion) if budget is None else ExecutionControls(motion=args.motion, max_replans=budget)


def main(controls=None, on_directive=None, argv=None) -> int:
    """controls: agent.controls.ExecutionControls for this process; when given (experiment harnesses) it is used
    as is and --motion is ignored; otherwise built from --motion (default blocked).
    on_directive: called with every input line that starts with "@" (harness directives); such lines are not
    instructions or answers (agent.ask.LineReader.on_directive)."""
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
    if a.scenario:  # perception must use the camera pose the simulator renders (control/robot.py TABLE_CAMERA)
        import scenarios
        want = scenarios.table_camera(a.scenario)
        if os.environ.get("TABLE_CAMERA", "") != want:
            print(f"[agent] scenario {a.scenario!r} mounts the table camera at TABLE_CAMERA={want or '(frozen cell)'}, but "
                  f"this process has {os.environ.get('TABLE_CAMERA') or '(frozen cell)'}: start through ./launch/agent.sh, "
                  f"or export TABLE_CAMERA={want}", flush=True)
            return 2
        if os.environ.get("WORKCELL_LAYOUT", "") != scenarios.layout(a.scenario):  # its calibrated regions (config/layout.py)
            print(f"[agent] scenario {a.scenario!r} has another calibrated layout than this process's WORKCELL_LAYOUT: "
                  f"start through ./launch/agent.sh", flush=True)
            return 2
    controls = controls_for(a, controls)
    print(f"[agent] motion: {controls.motion.upper()}" + {"blocked": " (no robot command is sent; --motion real to move)",
                                                         "dry_run": " (motions are planned and logged, not sent)",
                                                         "real": " (motions are executed)"}[controls.motion], flush=True)
    out = Path(a.out)
    from agent.ask import LineReader, ScriptedResponder, TerminalResponder
    if on_directive is not None:
        LineReader.shared().on_directive = on_directive
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
    vlm = GemmaVLM(a.gemma)
    from perception.detector import YoloDetector
    print("[agent] loading the detector ...", flush=True)
    detector = YoloDetector(a.weights, a.imgsz, a.conf)
    print("[agent] models loaded", flush=True)
    if a.sim_flag:
        Path(a.sim_flag).write_text("models loaded\n")
    sim = start_sim(a.gui, out / "sim.log", a.scenario, a.difficulty, a.seed) if a.start_sim else None
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
        end = time.monotonic() + 30  # target isaac_sim: the graph must be the simulator before anything can move
        while not simulator_on_graph(simulator_publishers(bridge.node)) and time.monotonic() < end:
            time.sleep(0.5)
        if not simulator_on_graph(simulator_publishers(bridge.node)):
            print(f"[agent] the connected ROS 2 graph is not the Isaac Sim simulator (no {SIMULATOR_NODE} publishing "
                  f"{SIMULATOR_TOPIC}); stopping "
                  "(this entry point never commands physical hardware)", flush=True)
            return 2
        print(f"[agent] connected to Isaac Sim (ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID', '0')})", flush=True)
        print("[agent] ready", flush=True)
        session = Session(bridge, detector, vlm, executor, responder, live, memory, controls=controls, trace=a.trace)
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
