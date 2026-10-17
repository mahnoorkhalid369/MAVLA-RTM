# Technical debt register, 2026-10-17 (`f094d2f`)

**Priority:**

| Level | Meaning |
|---|---|
| P1 | blocks safe evolution or causes defects now |
| P2 | slows every change |
| P3 | hygiene |

Evidence references point to files and lines in the audited commit. Functional defects are **not** debt items:
they are listed separately in §2 and are not to be fixed during the restructuring.

## 1. Architectural debt

### TD-1 (P1): `Tools` is a god object with implicit, shared mutable state

- **Evidence.** `agent/tools.py` (661 lines). ~40 methods spanning perception, world-model writes, grasp/place
  planning, held-object state, motion execution, verification and observation freshness. `DescriptionMemory` is
  defined in the same file.
- **State.** `world`, `held*`, `resolved_refs`, `last_referent`, `motion_count`, `view_cleared_at`, plus `log`,
  `roles` and `run_dir`, which are rebound per instruction (`begin_instruction`).
- **Impact.**
  - Every capability and the workflow call `Tools` directly.
  - Tests mirror it by hand (`tests/test_agent.py` `StubTools`, 23 methods; `tests/test_supervisor_goal.py`
    `GoalTools`).
  - New methods broke the stubs in each of the last three milestones (`revalidate_held`, `classify_request`,
    `begin_instruction`).
  - Held state is duplicated as `Tools.held_object` and `world.holding`, synchronised manually in 5 places.
- **Direction.** Split along responsibilities into explicit services, behind typed `Protocol`s (see
  TARGET_ARCHITECTURE §4). Keep the method bodies.

### TD-2 (P1): Three world/memory representations, unclear ownership

- **Evidence.**

  | Representation | Location |
  |---|---|
  | `WorldModel` | `agent/world.py` |
  | `TaskState.facts` / `hypotheses` | `agent/world_state.py` |
  | `DescriptionMemory` | `agent/tools.py` |
  | `WorldMemory` | `perception/world_memory.py` (legacy) |

- **Keys are inconsistent.** Descriptions are keyed by world id in `TaskState` (`world_state.py:78`), by mark in
  `Observation.descriptions`, and by geometry+colour in `DescriptionMemory`. The Experiment 1 scorer misread them
  (fixed in `44f8b2b`).
- **Duplicate ids.** Two masks of one fixture can share a world id, and one description is silently overwritten.
- **Direction.** One `WorldModel` (a session-scoped store with a write API and typed `ObjectId`), plus
  `TaskLedger` (per instruction) as a *view* of it. Retire the legacy memory with the legacy stack.

### TD-3 (P2): Mixed responsibilities in the orchestration modules

- **Evidence.**

  | Module | Mixes |
  |---|---|
  | `agent/supervisor.py` | graph wiring, policy constants, completion rules, binding, adaptation, review policy, replanning, verification orchestration, NL messages |
  | `agent/vlm_roles.py` | model adapter, 12 schemas, 7 prompts, validation rules, role methods |
  | `agent/capabilities.py` | specs, context, 12 executors, failure mapping, NL formatting |

- **Impact.** Prompt changes are made in the same file as schema and validation changes. The milestone-3
  regression (planner 6/8 → 3/8 valid) came from such a change, with no isolated test.
- **Direction.** Split by responsibility: `prompts/`, `schemas`, `model adapter`, `policies`. No new behaviour.

### TD-4 (P1): No injection points: safety-relevant behaviour is changed by monkeypatching

- **Evidence.**
  - `evaluation/lab_exp1_agent.py:31-78` replaces `supervisor.MAX_REPLANS`, `capabilities.SPECS`, `run_pick`,
    `run_place`, `run_pick_place`, `run_explore`, `run_decompose`, `Tools.execute` and `LineReader.get`.
  - The milestone harnesses also replace `EventLog.__call__`.
  - Tests patch `caps.run_inspect/run_explore/run_locate` and `Tools.plan_pick`.
- **Impact.**
  - Dry-run, motion gates and capability disabling are not product features, so they are untested as such.
  - A refactor that renames a function silently disables an experiment's safety gate.
- **Direction.**
  - Constructor-injected `MotionExecutor` (real / dry-run / gated).
  - A `CapabilityRegistry` built from configuration (enable/disable).
  - `SupervisorPolicy` config (replans, step bounds).
  - An `InputSource` interface.

### TD-5 (P1): Layering violations and configuration bound to the simulator

- **Evidence.**

  | # | Violation | Where |
  |---|---|---|
  | D1 | agent config imports the simulator's frozen test-field module | `agent/config.py:13` |
  | D4 | the scenario library imports the agent | `sim/scenario_lib/checks.py` → `agent.exploration` |
  | D5 | the camera rig is defined in the robot-constants module | `control/robot.py:64-77` |
  | — | scene selection by mutating `sim.scene_objects.OBJECTS` and patching `run_live.setup_scene` | `sim/run_scenario.py` |

- **Impact.**
  - The agent cannot be configured for another cell or a real robot without editing code.
  - Scenario checks can't run without the agent.
  - Scene injection depends on import order.
- **Direction.**
  - A `RigConfig` (robot base, cameras, static fixtures) loaded once, with adapters reading the frozen constants as
    defaults.
  - Move the viewpoint geometry the scenario checks need into a shared geometry module.

### TD-6 (P2): Stringly-typed outcomes and failure classes across four layers

- **Evidence.**

  | Layer | Values |
  |---|---|
  | `world_state.OUTCOMES` | tuple of str |
  | `workers.STATUS_OF` | dict of str |
  | `capabilities._failure_outcome` | maps workflow `cls` str → outcome str |
  | `pick_place` | `{"cls": "infeasible" \| "planning" \| "home" \| …}` |
  | supervisor | `step.status` / `outcome` comparisons |

  `"refused"` and `"unauthorized"` were added in two places.
- **Impact.** Mapping gaps became behaviour bugs: `infeasible` was mapped to `failed` (fixed in `c636c6e`, the cherry-pick of `9e10f59`), and
  `planning` to `infeasible`.
- **Direction.** `enum.StrEnum` types (`Outcome`, `StepStatus`, `FailureClass`, `Effect`) plus one mapping table
  with exhaustive tests. String values unchanged, so JSON logs stay compatible.

### TD-7 (P2): Two agent stacks

- **Evidence.** The legacy stack is still launchable (`launch/run.sh`, `start_vlm_agent.sh`, `tasks.sh`):
  `ros_nodes/vlm_agent_node.py` (789 lines), `vlm/plan.py`, `vlm/planner.py`, `perception/world_memory.py`,
  `perception/classical_detector.py`, `control/instructions.py` (frozen).
- **Leakage.** The current agent reuses `vlm/model.py`, `vlm/config.py` and `perception/grounding.py` (D2, D3).
- **Impact.** Duplicate concepts: two planners, two memories, two JSON parsers. Readers can't tell which is live.
- **Direction.** Freeze the legacy stack as `legacy/` *in place* (no moves; mark it in docs and launchers), and copy
  the two shared helpers behind new names. Retire it after the user decides. The legacy `run.sh` /
  `vlm_agent_node.py` are protected files: do not modify them.

### TD-8 (P3): Duplicated geometry and process utilities

- **Evidence.** DEPENDENCY_ANALYSIS §5:
  - 3 segment-box tests;
  - 3 `rot_z`;
  - 3 depth back-projections;
  - 3–4 process start/stop helpers;
  - 3 "bring-up in GPU order" sequences.
- **Direction.** `vt6l.geometry` (pure numpy) and `vt6l.runtime.bringup`, each with unit tests. The frozen copies
  stay.

### TD-9 (P2): No packaging, no dependency manifest, four interpreters

- **Evidence.**
  - No `pyproject.toml` / `requirements*.txt`.
  - 32 modules insert into `sys.path`.
  - `.venv` (Isaac), `.venv-seg` (agent), `.venv-vlm` (legacy), system python (default tests). numpy is 1.26 /
    2.3 / 2.5 across them.
  - The production YOLO weights are under the ignored `out/` (`agent/run.py` `DEFAULT_WEIGHTS`).
  - Stray `*.log` files at the repo root.
- **Impact.** Environment reproduction depends on memory. Agent tests silently skip under the default interpreter
  (7 files `importorskip`). A clean checkout cannot find its model weights.
- **Direction.** A `pyproject.toml` with optional dependency groups per environment, `pip install -e .`, and a
  model registry file (`config/models.toml`) pointing at `models/` (weights copied out of `out/`).

### TD-10 (P2): Test architecture

- **Evidence.**
  - Shared fixtures by importing test modules (D7).
  - No markers (unit / integration / sim / gpu / slow).
  - Simulator validation lives in `evaluation/validate_scenarios.py`, not tests.
  - No real-model regression for prompts.
  - Duck-typed stubs drift from `Tools`.
- **Direction.** `tests/conftest.py` + `tests/fakes/` (protocol-checked fakes); markers; a `tests/model/` suite with
  recorded Gemma I/O fixtures plus an opt-in live-model run (`-m gpu`).

### TD-11 (P3): Logging and error handling inconsistent

- **Evidence.**
  - The agent uses `EventLog` (structured JSONL, line-buffered).
  - `sim/`, `control/`, `evaluation/` use `print` (38 calls in library packages); the legacy nodes use rclpy
    loggers.
  - 10 `except Exception` in library code, most of them converting exceptions to failure strings without the type.
- **Direction.** Stdlib `logging` with a JSON formatter for libraries. `EventLog` stays as the *experiment record*
  (a separate concern). A small exception hierarchy (`VT6LError` → `PerceptionError`, `PlanningError` (wrap),
  `ExecutionError`, `ModelError`, `AuthorizationError`).

### TD-12 (P3): Evaluation code mixes libraries, drivers and harnesses

- **Evidence.** `evaluation/` holds reusable scoring (`lab_exp1_score.score`, `seg_diagnosis.approach_*`), drivers,
  monkeypatching harnesses, Isaac validators, and milestone scripts kept under `out/…/scripts`.
- **Direction.** `vt6l/evaluation/` (importable metrics, scorers, ground-truth access, which is the only place
  allowed to read simulator ground truth) + `experiments/<name>/` (drivers + config + README) + `scripts/`
  (one-off).

### TD-13 (P3): Import-time side effects

- **Evidence.** 8 modules create `SimulationApp` or parse argv at import (DEPENDENCY_ANALYSIS §4). One of them
  (`sim/run_live.py`) is frozen.
- **Direction.** New entry points use `main()` guards. The frozen module is wrapped, not edited (see the
  `sim/run_scenario.py` approach).

## 2. Known baseline defects (functional; must NOT be fixed silently during the refactor)

| Id | Defect | Evidence | Status |
|---|---|---|---|
| BD-1 | Planner reliability regression from the `91429c4` prompt change: 3/8 valid plans vs 6/8 before | `out/lab_exp1/milestone3/trackA/planner_ab/results.json` | open, needs its own fix commit |
| BD-2 | Location-constrained requests are rejected by the anchor-preservation validation when Gemma omits `object_relation`/`object_anchor` | same | open |
| BD-3 | Negated command "Do not pick up the vial." classified as action → `object_change` granted | `out/lab_exp1/milestone3/trackA/results.json` (`classification_only`) | open, safety-relevant |
| BD-4 | Table-camera segmentation merges containers with racks / trays (recall 0.29 on 8 independent frames) | `out/lab_seg_frames/generalization_results.json` | open; B2 prototype evaluated, not integrated |
| BD-5 | Split-instance heights unobservable from the table camera (−61 to −71 mm mean) | same | open |
| BD-6 | Label reading unsupported, but the planner claims it ("inspect … to read the label") | `trackA/results.json` request 2 | open |
| BD-7 | `reported()` accepts any scene report; describe coverage not verified | `agent/goals.py` `_reported` | open |
| BD-8 | YOLO detections vary between runs on the same scene (14 vs 16 kept of 18) | Experiment 1 vs current-system test | open, unexplained |

**Rule for the refactor.**

- Every refactoring increment must reproduce these defects identically. They act as characterization tests: if a
  refactor "fixes" one, the increment changed behaviour and must be split.
- Fixes go in separate, labelled commits after the structural increment that makes them easy.
