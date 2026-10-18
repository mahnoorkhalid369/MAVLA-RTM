# Architecture audit before the goal-driven supervisor refactor (2026-10-14)

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

Baseline: commit `c140d64` (tag `baseline-before-goal-supervisor`), branch `refactor/goal-supervisor`.
Working tree clean. Tests: `/usr/bin/python3 -m pytest` 135 passed / 2 skipped; `.venv-seg/bin/python -m
pytest tests/test_agent.py tests/test_agent_ask.py` 73 passed. Frozen baseline: `docs/baselines/
manipulation_A_2026-10-14/MANIFEST.sha256` (21 files: control/, execution/ros_executor.py, safety/,
sim/run_live.py, sim/scene*.py, sim/ros_control.py, assets/vt6l, experiments/, scripts/validate_robot_m1.py) —
manifest OK. Legacy agent (`launch/run.sh` → `ros_nodes/vlm_agent_node.py`, `.venv-vlm`) is separate and
untouched. Active: `./launch/start_demo.sh [--headless]` → `.venv-seg/bin/python -m agent.run --start-sim`.
Validated performance: two-instruction headless session 414 s (yellow→tan 225.5 s, red→green 161.5 s).

## What exists (call paths followed from agent/run.py)

| Responsibility (target agent) | Where it lives now | Coupling |
|---|---|---|
| Supervisor / orchestration | `agent/graph.py` `Supervisor`: ONE LangGraph with a fixed pick-and-place chain UNDERSTANDING → OBSERVING → LOCALIZING → PLANNING_GRASP → APPROACHING → GRASPING → VERIFYING_GRASP → TRANSPORTING → PLACING → RETREATING → VERIFYING_PLACEMENT → RETURNING_HOME → VERIFYING_HOME → COMPLETED, RECOVERING with budgets; SQLite checkpointer; EventLog | The task type IS the graph: any goal that is not one pick-and-place cannot be represented |
| Goal interpretation | `vlm_roles.Roles.parse` + `PARSE_SYSTEM`; schema `TaskSpec.task ∈ {pick_place, pick, unsupported}` | **Hard restriction:** `_understanding` fails everything that is not `pick_place` ("I can only move one object…"). This is why "take a look at your surroundings" is rejected |
| Perception | `tools.Tools.perceive / describe / update_world / resolve / lexical_candidates / confirm / candidate_image`; `geometry.py` (depth measurement), `detector.py` (YOLO), `somark.py`, `world.py` (WorldModel), `DescriptionMemory` (session cache) | In the same `Tools` class as manipulation and verification; fine as a toolbox, but no perception-level interface |
| Exploration | `Tools.park_for_view` (collision-checked ARM_HOME + J1 poses scored by blocked camera rays) | Only "move the arm out of the camera's view"; no viewpoint search, wrist camera unused for perception |
| Planning (task level) | None. The only plan is the fixed graph. `Tools.plan_pick_place` is motion/grasp planning (SkillPlanner + CollisionChecker) | Missing: multi-step plans, success conditions, replanning |
| Manipulation | `Tools.plan_pick_place / execute / return_home`, graph nodes APPROACHING…RETREATING; frozen `control/skills.py`, `execution/ros_executor.py`, safety layer in the simulator | Validated; reuse as is |
| Verification | `Tools.verify_contact / verify_placement / verify_home`; graph nodes VERIFYING_* | Validated per action; no goal-level verification |
| Clarification | `agent/ask.py` (TerminalResponder / ScriptedResponder), `Supervisor._question` | Only inside pick-and-place resolution |
| Shared state | LangGraph `AgentState` (instruction, task, ids, retries, questions, timeline, verifications, failure, plan summary) + `Supervisor.ctx` (observation, plan objects) + `Tools.world` | Facts, VLM hypotheses and verified outcomes are not separated |
| Logging / viewer | `EventLog` → events.jsonl (+ terminal echo via `events.py`), `live.py` → viewer feed, `viewer.py` | Works for any events; formatter knows only pick-and-place events |
| Model instances | One `GemmaVLM` per process (run.py), one YOLO; shared by all roles | Good — keep |

## Gaps (genuinely missing)
1. Goal-level supervisor loop (interpret → plan → delegate → verify → replan/complete).
2. Task-level planner with a capability vocabulary and success conditions.
3. Capability interfaces with preconditions / safety class / expected effects.
4. Observation and scene-reporting as goals in their own right (they exist only as steps inside pick-and-place).
5. Separation of observed facts / VLM hypotheses / planned actions / verified outcomes in shared state.
6. Exploration beyond arm parking (viewpoint planning, wrist camera) — out of scope for this refactor.

## Smallest safe refactor (chosen)
- Keep `agent/graph.py`'s validated chain unchanged in behaviour but make it a **skill**: `PickPlaceWorkflow`,
  started with an already-interpreted task, optionally reusing a fresh observation, optionally skipping the
  final HOME (when more manipulation follows). Its states, checks, budgets and events stay as they are.
- New, thin but real layers: `agent/world_state.py` (shared state schema), `agent/capabilities.py`
  (registry: specs + executors over the existing Tools / workflow), `agent/planner.py` (Gemma goal → plan
  constrained to the registry; deterministic validation; replaces the single parse call, so no extra Gemma
  call), `agent/supervisor.py` (bounded LangGraph plan–act–observe–verify loop).
- `agent/run.py` switches to the goal supervisor; events/viewer learn the new events.
- One new end-to-end capability: observe + grounded scene report (no robot motion beyond the existing
  view-parking rule, no extra Gemma call per object when descriptions are cached).
