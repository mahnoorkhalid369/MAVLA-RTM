# Module guide and repository navigation

This guide lists what each folder and module is for, and where to look for a given behaviour. It also sets out the
simple target structure the refactor moves towards.

- What each function does: `FUNCTIONALITY_MAP.md`.
- How the functions call each other: `EXECUTION_FLOW.md`.
- Commands to run things: `../DEVELOPER_GUIDE.md`.

## 1. Where to find X

| I want to… | Look at |
|---|---|
| start the agent on typed instructions | `agent/run.py` (`main`, `Session`); launcher `launch/start_demo.sh` |
| see how an instruction becomes steps | `agent/supervisor.py` `_interpreting` → `agent/vlm_roles.py` `Roles.plan_goal` (`PLAN_SYSTEM`) |
| see what the robot is allowed to do for a request | `agent/authority.py`; enforcement in `agent/workers.py` `WorkerRegistry.dispatch` |
| find the list of capabilities (observe, locate, inspect, pick_place, …) | `agent/capabilities.py` `SPECS` and the `run_<name>` functions |
| change a Gemma prompt or output schema | `agent/vlm_roles.py` (all agent prompts and pydantic schemas live here) |
| see how the camera image becomes objects | `agent/tools.py` `Tools.perceive` → `agent/detector.py` (YOLO) → `agent/geometry.py` `measure` |
| see how objects keep their ids over time | `agent/world.py` `WorldModel.observe` / `add_observation` |
| see what the run remembers (facts, hypotheses, plan) | `agent/world_state.py` `TaskState` (`task_state.json`) |
| see how a phrase picks one object | `agent/tools.py` `Tools.resolve` / `confirm`; location check in `agent/capabilities.py` `_location_check` |
| see how the wrist camera is pointed at something | `agent/exploration.py` `ViewpointPlanner` |
| see the pick-and-place state machine | `agent/pick_place.py` `PickPlaceWorkflow` |
| see grasp and motion planning | `Tools.plan_pick_place` → frozen `control/skills.py` `SkillPlanner`, `control/collision.py` |
| see where motion is sent to the robot | `Tools.execute` → frozen `execution/ros_executor.py` `RosExecutor` |
| see the joint-limit safety layer | frozen `safety/joint_limits.py`, applied in frozen `sim/ros_control.py` |
| see how success is checked | `agent/goals.py` `GoalEvaluator`; `Tools.verify_contact` / `verify_placement` / `verify_home` |
| ask the user a question during a task | `agent/ask.py` |
| change robot, camera or tolerance constants | `agent/config.py` (agent); frozen `control/robot.py` (robot, camera rig) |
| build or change a test scene | `sim/scenario_lib/` (library: laboratory); `sim/scenarios/*.json` (small active-perception scenes) |
| start the simulator | frozen `sim/run_live.py` (default scene); `sim/run_scenario.py` (test scenes) |
| see the ROS topics the simulator publishes | `sim/ros_bridge.py` |
| read the agent's ROS input (cameras, joints) | `agent/ros_bridge.py` `RosBridge` |
| replay or watch a run | `agent/viewer.py` (live window); `agent/events.py` (event → readable line) |
| score an experiment | `evaluation/lab_exp1_score.py`, `evaluation/demo_summary.py`, `evaluation/active_summary.py` |
| evaluate segmentation offline | `evaluation/seg_diagnosis.py`, `evaluation/seg_generalization.py` |
| write a test with fake tools | `tests/support/fakes.py` (`StubTools`, `GoalTools`, `geom`), `tests/support/runners.py` |

## 2. Folders as they are today

**Legend:**

- **F**: frozen (do not edit; checksummed in `docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256`).
- **L**: legacy, kept but not used by the current agent.

| Folder | Role | Key modules |
|---|---|---|
| `agent/` (5.3 k lines) | the current agent: supervisor, capabilities, perception pipeline, world model, Gemma roles | see §3 |
| `control/` **F** | robot model, kinematics, collision checking, skill (grasp / place) planner | `robot.py`, `kinematics.py`, `collision.py`, `skills.py`, `motion.py`, `instructions.py` |
| `execution/` **F** | streams planned trajectories to the simulator over ROS | `ros_executor.py` |
| `safety/` **F** | joint limits and speed checks | `joint_limits.py` |
| `sim/` | Isaac Sim processes | `run_live.py` **F**, `scene.py` **F**, `scene_objects.py` **F**, `ros_control.py` **F**, `ros_bridge.py`, `run_scenario.py`, `import_vt6l.py` |
| `sim/scenario_lib/` | scenario library: parts, the laboratory, building, checks | `core.py`, `parts.py`, `laboratory.py`, `build.py`, `checks.py` |
| `sim/scenarios/` | small JSON test scenes (`ap_categories`, `ap_occlusion`) | — |
| `perception/` | older perception helpers. `grounding.py` is **shared**: `agent/geometry.py` uses `footprint_yaw` / `grasp_pose`, and the frozen `experiments/known_pose_*.py` use it too. `world_memory.py` and `classical_detector.py` belong to the legacy stack and its tests | `grounding.py` (shared); the rest **L** |
| `vlm/` | Gemma model loader `model.py` (**shared**: the current agent loads Gemma through it) + legacy planner (`planner.py`, `plan.py`, `prompts.py`) | `model.py`, `config.py` |
| `ros_nodes/` | ROS nodes: camera viewers (used), legacy voice / text / VLM nodes | `camera_viewer.py`; `vlm_agent_node.py` **L** (protected) |
| `launch/` | shell launchers | `start_demo.sh` (current), `start_live.sh` (sim + viewers), `run.sh` **L** (protected), `start_vlm_agent.sh` / `tasks.sh` **L** |
| `evaluation/` | experiment runners, probes, scorers, offline studies (simulator ground truth allowed here only) | lab: `lab_exp1*.py`, `lab_probe.py`, `lab_capture_frames.py`, `seg_*.py`, `validate_scenarios.py`; older phases: `demo_*.py`, `inspect_check.py`, `dual_camera_check.py` |
| `experiments/` | manipulation baseline experiments | `known_pose_*.py` **F**, `payload_study.py` **F**; `pick_place.py` / `trajectory_demo.py` are early standalone demos |
| `scripts/` | robot and camera validation | `validate_robot_m1.py` **F**, `validate_cameras_phaseA.py` |
| `tests/` | offline tests (no simulator, no GPU) | `support/` (shared fakes), `test_agent*.py`, `test_supervisor_goal.py`, `test_orchestration.py`, `test_authority.py`, `test_session.py`, … |
| `docs/` | plans, audits, architecture, scenario docs, frozen baseline | `architecture/`, `scenarios/`, `baselines/` |
| `assets/` | URDF and USD of the VT6L (**F**) | — |
| `out/` (git-ignored) | run outputs, reports, YOLO weights (`out/seg/runs/yolo11s_seg_r1/weights/best.pt`) | — |

## 3. The agent, module by module

**Layer 1: entry and orchestration.**

| Module | Lines | Responsibility | Owns state? |
|---|---|---|---|
| `run.py` | 172 | CLI entry; loads Gemma → YOLO → (optionally) Isaac; `Session` keeps one `Tools` across instructions | `Session` |
| `supervisor.py` | 464 | `GoalSupervisor`: the LangGraph state machine of §2 in `EXECUTION_FLOW.md` | per-instruction graph state |
| `workers.py` | 148 | typed `WorkerRequest` / `WorkerResult`, `WorkerRegistry` (dispatch + authorization) | — |
| `authority.py` | 74 | `RequestAuthority` from the request classification | — |
| `goals.py` | 173 | goal conditions and `GoalEvaluator` | — |
| `world_state.py` | 88 | `TaskState` ledger, written to `task_state.json` | yes (per instruction) |

**Layer 2: capabilities and skills.**

| Module | Lines | Responsibility |
|---|---|---|
| `capabilities.py` | 620 | `SPECS` (effect, safety class, inputs) and one `run_<name>` per capability |
| `pick_place.py` | 424 | `PickPlaceWorkflow`: nested LangGraph pick / place / pick_place skill with recovery budgets |
| `exploration.py` | 224 | `ViewpointPlanner` for wrist-camera views |
| `ask.py` | 189 | `TerminalResponder`, `ScriptedResponder`, deterministic answer parsing |

**Layer 3: tools, perception and the world model.**

| Module | Lines | Responsibility | Owns state? |
|---|---|---|---|
| `tools.py` | 661 | `Tools`: perceive, describe, resolve, confirm, plan, execute, verify, held object; `DescriptionMemory` | yes (session): held object, description memory, resolved refs |
| `detector.py` | 51 | `YoloDetector` (class-agnostic masks) | — |
| `geometry.py` | 187 | `measure`: mask + depth → 3-D geometry, robot / fixture removal | — |
| `somark.py` | 113 | numbered mark images and crops for Gemma | — |
| `world.py` | 157 | `WorldModel`: persistent ids, association, fusion | yes (session) |
| `vlm_roles.py` | 570 | `GemmaVLM` adapter + `Roles` (classify, plan, describe, resolve, confirm, compare, answer, decompose), prompts and schemas | — |
| `config.py` | 84 | agent constants: robot volume, tolerances, budgets, camera settings | — |

**Layer 4: I/O and observability.**

| Module | Lines | Responsibility |
|---|---|---|
| `ros_bridge.py` | 128 | ROS node for cameras, joints and clock (`snapshot`, `wait_still`) |
| `eventlog.py` | 39 | `EventLog`: JSON-lines writer |
| `events.py` | 220 | readable lines for events (pure Python, also used by the viewer) |
| `live.py` | 61 | files for the live viewer (`current.json`, `perception.json`) |
| `viewer.py` | 441 | optional read-only viewer window (system Python) |

**Dependency direction.** It mostly runs downward:

```
run → supervisor → workers / capabilities → pick_place / exploration → tools → (detector, geometry, world, vlm_roles) → frozen control / execution
```

Known exceptions (`DEPENDENCY_ANALYSIS.md`):

- `agent/config.py` imports the frozen `sim/scene_objects.py`;
- `sim/scenario_lib/checks.py` imports `agent.exploration.ViewpointPlanner` (inside a function) for its viewpoint check.

## 4. Simple target structure

**Principles:**

- Plain top-level folders of ordinary Python modules: no `src/` package, no packaging framework.
- A move happens only when it makes a file easier to find or smaller to read.
- Frozen files never move, and nor do the entry points people already use (`python -m agent.run`,
  `python -m sim.run_scenario`, `evaluation/*.py`). When a module moves, its old name stays importable for one
  increment as a two-line re-export, and the move is logged in `REFACTOR_CHANGELOG.md`.

| Target folder | What goes there | From today's files | Moves? |
|---|---|---|---|
| `agent/` | orchestration only: `run.py`, `supervisor.py`, `workers.py`, `authority.py`, `goals.py`, `world_state.py`, `capabilities.py`, `pick_place.py`, `ask.py` | same | stays |
| `agent/tools/` (small folder, not a framework) | the pieces of today's `Tools` class: `memory.py` (DescriptionMemory), `perceive.py`, `resolve.py`, `plan.py`, `verify.py`, with `Tools` kept as a thin class combining them | `agent/tools.py` | split in increment 4; `from agent.tools import Tools` keeps working |
| `agent/vlm/` *(or kept in one file)* | `prompts.py`, `schemas.py`, `roles.py`, `gemma.py` | `agent/vlm_roles.py` | split in increment 5 only if the prompt-hash check passes |
| `perception/` | the current perception code: detector, geometry, set-of-mark images, world model, viewpoint planner | `agent/detector.py`, `geometry.py`, `somark.py`, `world.py`, `exploration.py` | later; legacy files go to `legacy/` first |
| `robot/` | **not a folder move**: the name for the frozen `control/`, `execution/`, `safety/` folders, which stay where they are | — | never |
| `sim/` | simulator processes (frozen files stay) | `sim/*.py` | stays |
| `scenarios/` | the scenario library and JSON scenes | `sim/scenario_lib/`, `sim/scenarios/` | optional; only after its checks stop importing the agent (increment 7) |
| `evaluation/` | runners, probes, scorers, studies; a `README.md` table marks each script as active or historical | same | stays |
| `config/` | constants that are configuration, not code: camera rig, tolerances, model paths | `agent/config.py`, parts of `vlm/config.py` | increment 6 |
| `legacy/` | the old VLM stack, once nothing current imports it | `vlm/planner.py`, `vlm/plan.py`, `vlm/prompts.py`, `perception/world_memory.py`, `perception/classical_detector.py`, legacy `ros_nodes/` (`perception/grounding.py` stays: the frozen experiments import it) | later; the protected `launch/run.sh` and `ros_nodes/vlm_agent_node.py` stay in place |
| `tests/` | flat test files + `tests/support/` for shared fakes and runners | same | done in increment 1 |

**Deliberately not proposed:**

- a `src/vt6l/` package or `pyproject` packaging;
- protocol and interface classes for single implementations;
- adapter layers between the agent and the frozen core;
- renaming capability names, outcomes or event kinds (they are the experiment record format).
