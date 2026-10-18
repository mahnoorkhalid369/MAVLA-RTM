# Developer guide: run and test each part on its own

Read first:

- `docs/architecture/MODULE_GUIDE.md`: where things are.
- `docs/architecture/EXECUTION_FLOW.md`: how an instruction flows through the code.
- `docs/architecture/FUNCTIONALITY_MAP.md`: what each function does and its status.

All commands run from the repository root.

> **Restructured 2026-10-17;** automated verification passed on 2026-10-18 (unit, integration, saved-data, model inference, Isaac Sim + ROS with motion blocked); manual acceptance pending: `docs/validation/FINAL_MAIN_READINESS.md`. Expected results below are from that validation.

## 0. Which Python for what

| Environment | Use it for | Notes |
|---|---|---|
| `.venv/bin/python` | **Isaac Sim** processes: `sim/*`, `evaluation/validate_scenarios.py`, `evaluation/lab_capture_frames.py` | numpy 2.3; one Isaac instance at a time (12 GB GPU) |
| `.venv-seg/bin/python` | **the agent**: `agent.run`, Gemma, YOLO, LangGraph; the agent tests; perception studies; ROS probes | no pip inside. For pytest, start it without the ROS paths: `env -u PYTHONPATH -u AMENT_PREFIX_PATH …` |
| `/usr/bin/python3` | **ROS tools and viewers** (`ros_nodes/camera_viewer.py`, `agent/viewer.py`); the default `pytest`; scorers | numpy 1.26, no pydantic or langgraph, so the agent tests are *skipped* here |
| `.venv-vlm` | legacy VLM stack only (`launch/run.sh`) | not needed for current work |

**For anything that talks to ROS**, start the shell like this:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42        # keeps tests away from your normal session (domain 0)
```

**GPU order:** load Gemma first, then YOLO, then Isaac Sim. `agent.run --start-sim` does this for you. Never train
while the simulator runs.

## 1. Offline tests (no simulator, no GPU, no motion)

```bash
# Robot, safety, perception helpers, scenarios, scorers (agent tests are skipped here):
/usr/bin/python3 -m pytest                                  # expect: 199 passed, 29 skipped (agent tests need the agent env)

# The agent: supervisor, workflows, authorization, session, ask (mocked tools, scripted Gemma):
env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python -m pytest \
    tests/test_agent.py tests/test_agent_ask.py tests/test_supervisor_goal.py \
    tests/test_orchestration.py tests/test_session.py tests/test_authority.py   # expect: 142 passed

# Everything (all 383 tests, incl. controls, structure, saved frames + YOLO) in the agent environment:
env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python -m pytest       # expect: 383 passed

# Frozen manipulation baseline unchanged:
grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c --quiet && echo FROZEN-OK
```

**One subsystem at a time:**

| Subsystem | Command (environment) |
|---|---|
| Kinematics, collision, grasp / place planning (frozen core) | `/usr/bin/python3 -m pytest tests/test_kinematics.py tests/test_collision.py tests/test_skills_and_plans.py` |
| Joint-limit safety | `/usr/bin/python3 -m pytest tests/test_safety.py` |
| Scenario library | `/usr/bin/python3 -m pytest tests/test_scenario_library.py tests/test_scene_reachability.py` |
| Perception geometry / world model / association | agent env: `… -m pytest tests/test_agent.py -k "world or geometry or association or wrist"` |
| Authorization | agent env: `… -m pytest tests/test_authority.py` |
| Goal verification | agent env: `… -m pytest tests/test_supervisor_goal.py` |
| Session memory across instructions | agent env: `… -m pytest tests/test_session.py` |
| Scorers | `/usr/bin/python3 -m pytest tests/test_lab_exp1_score.py tests/test_demo_scoring.py` |

(`…` = `env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python`)

## 2. Test the supervisor without robot motion

**Mocked: no GPU, no simulator.** The fakes in `tests/support/fakes.py` replace the `Tools` class:

- `GoalTools(plans)` returns the plans you give it in place of Gemma;
- `StubTools` provides measured objects (`geom(x, y, top)`) and records every call.

A new test looks like this:

```python
from tests.support.fakes import GoalTools
from tests.support.runners import plan, run_supervisor

def test_my_case(tmp_path):
    t = GoalTools([plan(("observe", {}), ("describe_scene", {}))])
    report, events = run_supervisor(tmp_path, t, instruction="Take a look at your surroundings.")
    assert report["status"] == "COMPLETED"
    assert t.executed == []      # StubTools records every Tools.execute call here: no motion was requested
```

Other recorded calls: `t.plan_calls`, `t.home_calls`, `t.parks`. For the pick-and-place skill alone, use
`run_workflow(tmp_path, StubTools(...))`.

**Real Gemma planning, no simulator.** This calls the planner on fixed instructions and checks the plan only:

```bash
.venv-seg/bin/python -m evaluation.planner_check --out out/perf/planner_check.json
```

Expect failures at the current HEAD (BD-1).

**Execution controls: blocked, dry-run, disabled, no replanning** (`agent/controls.py`). These are the explicit
switches that replaced the Experiment 1 monkeypatches. Use them to run real code without robot motion:

```python
from pathlib import Path
from agent.controls import ExecutionControls
from agent.tools import Tools

# Tools with no robot at all: a blocked or dry-run execute never touches the executor.
t = Tools(bridge=None, detector=None, roles=None, executor=None, log=lambda *a, **k: print(a, k), run_dir=Path("."),
          controls=ExecutionControls(motion="blocked"))
r = t.execute([], "test move")      # expected: r.ok is False, a "tool execute_refused" event, motion_count unchanged
t.controls.motion = "dry_run"
r = t.execute([], "test move")      # expected: r.ok is False, a "tool execute_dry_run" event
```

```python
from agent.ask import ScriptedResponder
from agent.eventlog import EventLog
from agent.supervisor import GoalSupervisor
from tests.support.fakes import GoalTools
from tests.support.runners import plan
from agent.controls import ExecutionControls

def test_disabled_capability(tmp_path):
    t = GoalTools([plan(("pick_place", {"object": "a", "relation": "on", "target": "b"}))])
    log = EventLog(tmp_path / "events.jsonl")
    controls = ExecutionControls(disabled={"pick_place"}, max_replans=0)
    r = GoalSupervisor(t, log, tmp_path, ScriptedResponder(["cancel"]), controls=controls).run("put a on b")
    # expected (GoalTools classifies every request as an action): the step's outcome is "unsupported"
    # ("pick_place is disabled by the execution controls"), r["status"] == "UNSUPPORTED", t.executed == []
```

The same controls go to the whole agent through `agent.run.main(controls=...)`. That is how
`evaluation/lab_exp1_agent.py` runs Experiment 1: motion is blocked until an "@motion on" line, five
capabilities are disabled, and replanning is off.

## 3. Load the laboratory scenario (Isaac Sim, no agent)

The scenario code is in `scenarios/` (laboratory definition `scenarios/laboratory.py`; JSON scenes in
`scenarios/scenes/`). Offline design checks, with no Isaac, run on the system Python:

```bash
/usr/bin/python3 -m scenarios.checks --scenario laboratory --difficulty easy --seed 42 --out out/dev/design.json
```

```bash
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42 OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python -m sim.run_scenario --scenario laboratory --difficulty easy --seed 42            # window
.venv/bin/python -m sim.run_scenario --scenario laboratory --difficulty easy --seed 42 --headless
```

- Difficulties: `easy`, `medium`, `hard`.
- The same seed gives the same layout.
- The default scene (cubes) is `.venv/bin/python -m sim.run_live`.
- The scene description is in `docs/scenarios/LABORATORY.md`.

**Validate the scenario.** This is slow: run it alone, in a separate session.

```bash
.venv/bin/python evaluation/validate_scenarios.py --scenario laboratory --seed 42 --reload --drop-mm 10   # add --resume to continue
.venv/bin/python evaluation/validate_scenarios.py --scenario laboratory --difficulty easy --no-design     # quick
```

## 4. Test cameras and ROS (simulator running, same shell setup)

```bash
ros2 topic list                                   # expect /camera/color/image_raw (wrist), /table_camera/color/image_raw, /joint_states, /clock
ros2 topic hz /table_camera/color/image_raw
/usr/bin/python3 ros_nodes/camera_viewer.py                                         # wrist camera window
/usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera" \
    --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
    --depth-min 1.2 --depth-max 3.2 --ros-args -r __node:=table_viewer
/usr/bin/python3 ros_nodes/camera_viewer.py --snapshot out/view.png                 # one frame, then exit
.venv-seg/bin/python -m evaluation.lab_probe snapshot out/probe.json                # joint states + ground truth (evaluation only)
```

## 5. Test perception on its own

**Offline, on the 27 saved laboratory frames** (`out/lab_seg_frames/`; no simulator, GPU optional):

```bash
.venv-seg/bin/python evaluation/seg_generalization.py out/lab_seg_frames      # approaches A / B / B2 / C vs simulator masks
.venv-seg/bin/python evaluation/seg_diagnosis.py out/lab_exp1/current_system_test
```

**Capture new frames:** `.venv/bin/python evaluation/lab_capture_frames.py --out out/lab_seg_frames` (Isaac only;
no agent running).

**Live perception, no motion.** Start the agent and give an observation-only instruction:

```bash
.venv-seg/bin/python -m agent.run --start-sim --scenario laboratory --instruction "Take a look at your surroundings." \
    --out out/dev/observe
```

Authorization grants no object changes for this request, and the plan is observe + describe_scene. The arm can
still move once: `Tools.park_for_view` parks it out of the table camera's view if it is in the way. That is a real
motion through the frozen executor, so don't run it during refactoring work.

## 6. Run the whole agent

```bash
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
./launch/agent.sh                           # Gemma → YOLO → Isaac (window) + viewer; motion BLOCKED
./launch/agent.sh --headless --scenario laboratory --difficulty easy --seed 42
./launch/agent.sh --motion real                 # the robot moves (simulator)
.venv-seg/bin/python -m agent.run --start-sim --gui         # the same without the viewer
```

**Motion.** Instructions that ask for object changes ("Put … on …") move the robot. Avoid them during refactoring
work.

## 7. Inspect logs and outcomes

Each instruction writes `out/demo/<date>/live_<time>/NN/` (or `--out DIR/NN/`):

| File | What it tells you |
|---|---|
| `result.json` | final status, message, verifications, `refused` steps |
| `task_state.json` | the plan with each step's outcome; facts (measured), hypotheses (Gemma), authority |
| `events.jsonl` | every event in order: `state`, `vlm`, `authorization`, `worker_result`, `verify`, … |
| `images/` | the numbered mark images and crops Gemma saw |

```bash
# readable timeline (system Python; agent/events.py has no heavy imports):
/usr/bin/python3 -c "import json,sys; from agent.events import format_event
for l in open(sys.argv[1]):
    s = format_event(json.loads(l))
    if s: print(s)" out/demo/<date>/live_<time>/01/events.jsonl
# which states the supervisor went through:
grep '"kind": "state"' out/demo/<date>/live_<time>/01/events.jsonl | head
# live viewer while the agent runs:
/usr/bin/python3 agent/viewer.py --live-dir out/demo/live
```

**Simulator logs:** `out/logs/` (from the launchers), or `sim.log` in the `--out` folder (with `agent.run --start-sim`).

**Experiment reports:** `out/lab_exp1/*/REPORT.md`.
