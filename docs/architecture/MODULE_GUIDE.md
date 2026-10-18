# Module guide and repository navigation

> **Restructured 2026-10-17;** automated verification passed on 2026-10-18 (unit, integration, saved-data, model inference, Isaac Sim + ROS with motion blocked); manual acceptance pending: `docs/validation/FINAL_MAIN_READINESS.md`.

This guide lists what each folder and module is for and where to look for a given behaviour.

- What each function does: `FUNCTIONALITY_MAP.md`.
- How the functions call each other: `EXECUTION_FLOW.md`.
- Commands: `../DEVELOPER_GUIDE.md`.

## 1. Where to find X

| I want to… | Look at |
|---|---|
| configure a robot session | `agent/settings.py` (scene, motion, models, windows, tracing, outputs; validated at startup) |
| start the agent on typed instructions | `launch/agent.sh` → `agent/run.py --settings` (`main`) |
| follow the agent-to-agent handoffs | `TRACE = True` in `agent/settings.py`; lines from `agent/events.py` `format_trace`; full record in `events.jsonl` |
| see what lives across instructions | `agent/session.py` (`Session`) |
| see how an instruction becomes steps | `agent/supervisor.py` `_interpreting` → `agent/vlm_roles/roles.py` `Roles.plan_goal` (prompt `PLAN_SYSTEM` in `agent/vlm_roles/prompts.py`) → `agent/planning.py` `complete_plan` |
| see what the robot is allowed to do for a request | `agent/authority.py`; enforced in `agent/workers.py` `WorkerRegistry.dispatch` |
| block motion, dry-run, disable capabilities, limit replanning | `agent/controls.py` `ExecutionControls` (pass to `agent.run.main`, `Session`, `Tools`, `GoalSupervisor`) |
| find the list of capabilities | `agent/capabilities/specs.py` `SPECS` |
| see how a capability runs | `agent/capabilities/perception.py` (observe, describe_scene, answer, locate), `inspection.py` (inspect, explore), `manipulation.py` (pick, place, pick_place, return_home), `planning.py` (decompose) |
| change or read a Gemma prompt | `agent/vlm_roles/prompts.py` (answer schemas: `schemas.py`) |
| see how the camera image becomes objects | `agent/tools/perception.py` `Tools.perceive` → `perception/detector.py` (YOLO) → `perception/geometry.py` `measure` |
| see how objects keep their ids | `perception/world_model.py` `WorldModel.observe` / `add_observation` |
| see Gemma's descriptions and their reuse | `agent/tools/perception.py` `Tools.describe`; `agent/tools/memory.py` `DescriptionMemory` |
| see how a phrase picks one object | `agent/tools/resolution.py` `Tools.resolve` / `confirm`; location check `agent/capabilities/perception.py` `_location_check` |
| see what the robot is holding, and who owns that | `agent/tools/held.py` (module docstring) |
| see what one instruction recorded (facts, hypotheses, plan) | `agent/task_state.py` `TaskState` → `task_state.json` |
| see how the wrist camera is pointed at something | `perception/viewpoints.py` `ViewpointPlanner` |
| see the pick-and-place state machine | `agent/pick_place.py` `PickPlaceWorkflow` |
| see grasp and motion planning | `agent/tools/manipulation.py` `Tools.plan_pick_place` / `plan_pick` / `plan_place_held` → frozen `control/skills.py`, `control/collision.py` |
| see where motion is sent to the robot | `agent/tools/execution.py` `Tools.execute` → frozen `execution/ros_executor.py` |
| see the joint-limit safety layer | frozen `safety/joint_limits.py`, applied in frozen `sim/ros_control.py` |
| see how success is checked | `agent/goals.py` `GoalEvaluator`; `agent/tools/verification.py` |
| ask the user a question during a task | `agent/ask.py` |
| change thresholds, tolerances, budgets, model paths | `config/` (`perception.py`, `agent.py`, `models.py`); robot and camera constants: frozen `control/robot.py`; Gemma: `vlm/config.py` |
| build or change a test scene | `scenarios/` (`laboratory.py`, `parts.py`, `core.py`); small JSON scenes in `scenarios/scenes/` |
| start the simulator | `launch/sim.sh` → `sim/run_scenario.py` (test scenes) or frozen `sim/run_live.py` (`--scenario default`) |
| run the tests | `launch/tests.sh` (all / agent / system / frozen / sim) |
| run an experiment | `launch/evaluate.sh` (planner / segmentation / validate-scenarios / lab-exp1 / score / ...) |
| validate a scenario | `scenarios/checks.py` (offline); `launch/evaluate.sh validate-scenarios` → `evaluation/validate_scenarios.py` (Isaac) |
| see the ROS topics the simulator publishes | `sim/ros_bridge.py` |
| read the agent's ROS input (cameras, joints) | `agent/ros_bridge.py` `RosBridge` |
| watch or replay a run | `agent/viewer.py` (live window); `agent/events.py` (event → readable line) |
| score an experiment | `evaluation/lab_exp1_score.py`, `evaluation/historical/demo_summary.py`, `evaluation/historical/active_summary.py` |
| evaluate segmentation offline | `evaluation/seg_diagnosis.py`, `evaluation/seg_generalization.py` |
| write a test with fake tools | `tests/support/fakes.py` (`StubTools`, `GoalTools`, `geom`), `tests/support/runners.py`; put it in `tests/unit/` or `tests/integration/` |

## 2. Final directory structure

**Legend:**

- **F**: frozen (do not edit; checksummed in `docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256`).
- **L**: legacy, kept but not used by the current agent.
- **P**: protected legacy file (do not modify).

```
agent/                       orchestration: one instruction -> verified result
  settings.py                THE run configuration (edit here; ./launch/agent.sh uses it)
  run.py                     entry point: python -m agent.run (models -> simulator -> Session)
  session.py                 Session: what lives across instructions
  supervisor.py              GoalSupervisor: LangGraph state machine + review policy
  planning.py                plan completion rules, achieves()
  workers.py                 typed requests / results, registry, dispatch + authorization
  authority.py               what the user's request permits (none / sensing / object_change)
  goals.py                   goal conditions, GoalEvaluator over measured state
  task_state.py              TaskState ledger (task_state.json)
  controls.py                ExecutionControls (real / dry_run / blocked motion, disabled capabilities, replans)
  pick_place.py              PickPlaceWorkflow (pick / place / pick_place skill, recovery budgets)
  ask.py                     questions to the user, LineReader
  capabilities/              what the planner may ask for
    specs.py                 SPECS, available(), planner_menu(), StepResult, Context
    perception.py            observe, describe_scene, answer, locate
    inspection.py            inspect, explore (wrist camera)
    manipulation.py          pick, place, pick_place, return_home, failure mapping
    planning.py              decompose
  tools/                     Tools: the agent's only way to touch the world
    __init__.py              class Tools (composed from the parts below)
    held.py                  held-object / session state
    perception.py            perceive, describe, update_world, is_fresh, Observation
    memory.py                DescriptionMemory
    resolution.py            resolve, lexical check, compare, confirm, relation_holds
    manipulation.py          grasp / place planning, park_for_view, return_home, PickPlan
    execution.py             execute (motion gate)
    verification.py          verify_contact, verify_home, verify_placement
  vlm_roles/                 Gemma roles
    gemma.py                 GemmaVLM, RecordedVLM
    prompts.py               every prompt text
    schemas.py               answer schemas, VLMFailure
    roles.py                 Roles, plan validation rules
  ros_bridge.py              ROS input (cameras, joints, clock)
  eventlog.py, events.py     events.jsonl writer; readable event lines
  live.py, viewer.py         live viewer channel; optional viewer window (system Python)
perception/                  camera frames -> measured, described objects
  detector.py                YoloDetector
  geometry.py                measure(), RobotVolume
  somark.py                  mark images, crops
  world_model.py             WorldModel, same_place()
  viewpoints.py              ViewpointPlanner, shadow_points (also used by scenarios/checks.py)
  grounding.py               shared geometry helpers (used by geometry.py, frozen and legacy code)
  classical_detector.py      L (used by the frozen known-pose suite)
  world_memory.py            L (legacy node)
config/                      configuration
  perception.py              depth thresholds, fixtures
  agent.py                   Tolerances, Budgets, Relations
  models.py                  YOLO weights, detector settings
scenarios/                   scenario library (pure data) + Isaac builder + offline checks
  core.py, parts.py, laboratory.py, build.py, checks.py
  scenes/*.json              small active-perception scenes
sim/                         Isaac Sim processes
  run_live.py F, scene.py F, scene_objects.py F, ros_control.py F
  run_scenario.py            python -m sim.run_scenario (test scenes)
  ros_bridge.py, import_vt6l.py
control/ F                   robot model, kinematics, collision, skill planner
execution/ F                 RosExecutor
safety/ F                    joint limits
vlm/                         Gemma loader (model.py, config.py: shared) + legacy planner (L)
evaluation/                  experiments, probes, scorers, scenario validation, studies (README.md)
  historical/                earlier-phase scripts (default-scene demo, active perception)
experiments/                 manipulation baseline experiments (known_pose_* F, payload_study F)
scripts/                     validation scripts (validate_robot_m1.py F)
ros_nodes/                   camera viewers; legacy nodes (vlm_agent_node.py P)
launch/                      agent.sh (reads agent/settings.py), sim.sh, tests.sh, evaluate.sh
  run.sh P, legacy/          the legacy stack (start_live.sh, start_vlm_agent.sh, tasks.sh)
tests/                       automated tests
  unit/, integration/        14 + 9 test files
  support/                   shared fakes and runners
datagen/, seg/               synthetic data generation; segmentation training (offline tooling)
docs/                        DEVELOPER_GUIDE.md, architecture/, scenarios/, validation/, baselines/ (frozen
                             evidence), history/ (earlier designs, audits, demo reports, LEGACY_SETUP.md)
assets/ F                    URDF and USD of the VT6L
```

## 3. Dependency direction

```
agent/run -> agent/session -> agent/supervisor -> agent/workers, agent/capabilities -> agent/pick_place
          -> agent/tools -> perception/*, agent/vlm_roles -> frozen control/, execution/
config/ <- perception/, agent/          scenarios/ -> perception/viewpoints, frozen control/
```

**What changed in the restructuring:**

- `perception/` and `scenarios/` no longer import `agent/`.
- `config/` imports only the frozen `control/robot.py`.
- Nothing in `agent/` imports a simulator module.

**What remains:**

- `perception/geometry.py` uses `perception/grounding.py`, which is shared with legacy code.
- `sim/run_scenario.py` fills the frozen `sim.scene_objects.OBJECTS` before the frozen simulator starts. That
  frozen module is global state, and it can't be changed.

## 4. Before → after mapping

| Before | After |
|---|---|
| `agent/config.py` | `config/perception.py`, `config/agent.py` (+ `unit()` → `agent/tools/resolution.py`) |
| `agent/run.py` `DEFAULT_WEIGHTS`, 1024, 0.25 | `config/models.py` |
| `agent/run.py` `Session` | `agent/session.py` |
| `agent/supervisor.py` `complete_plan`, `achieves`, `ALTERNATIVES`, `CONSUMERS`, `_ref_texts` | `agent/planning.py` |
| `agent/supervisor.py` `MAX_REPLANS` | `agent/controls.py` `ExecutionControls.max_replans` |
| `agent/world_state.py` | `agent/task_state.py` |
| `agent/tools.py` | `agent/tools/` (see §2) |
| `agent/capabilities.py` | `agent/capabilities/` (see §2) |
| `agent/vlm_roles.py` | `agent/vlm_roles/` (see §2) |
| `agent/detector.py`, `geometry.py`, `somark.py` | `perception/detector.py`, `geometry.py`, `somark.py` |
| `agent/world.py` | `perception/world_model.py` |
| `agent/exploration.py` | `perception/viewpoints.py` |
| `sim/scenario_lib/*.py` | `scenarios/*.py` |
| `sim/scenarios/*.json` | `scenarios/scenes/*.json` |
| monkeypatches in `evaluation/lab_exp1_agent.py` | `ExecutionControls` + `LineReader.on_directive` |

**Entry points kept:**

- `python -m agent.run`, `python -m sim.run_scenario` and `python -m sim.run_live`;
- `agent/viewer.py`, the active `evaluation/*.py` scripts, `launch/run.sh`;
- changed on 2026-10-18: `launch/start_demo.sh` → `launch/agent.sh` (motion blocked unless `--motion real`);
  earlier-phase evaluation scripts → `evaluation/historical/`; legacy launchers → `launch/legacy/`;
  tests → `tests/unit/`, `tests/integration/`; historical docs → `docs/history/`.
- `from agent.tools import Tools`, `from agent.capabilities import …`, `from agent.vlm_roles import …`.

**Renamed import paths** (no compatibility module was left at the old name):

| Old | New |
|---|---|
| `agent.detector` | `perception.detector` |
| `agent.geometry` | `perception.geometry` |
| `agent.somark` | `perception.somark` |
| `agent.world` | `perception.world_model` |
| `agent.exploration` | `perception.viewpoints` |
| `agent.config` | `config.*` |
| `agent.world_state` | `agent.task_state` |
| `sim.scenario_lib` | `scenarios` |
| `python -m sim.scenario_lib.checks` | `python -m scenarios.checks` |
