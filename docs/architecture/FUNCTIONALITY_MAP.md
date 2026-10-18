# Functionality map (what exists, where, and whether it works)

> **Restructured 2026-10-17;** automated verification passed on 2026-10-18 (unit, integration, saved-data, model inference, Isaac Sim + ROS with motion blocked); manual acceptance pending: `docs/validation/FINAL_MAIN_READINESS.md`. The status column was recorded before the restructuring; the
> restructured code reproduced it wherever it could be re-run (see the readiness report, section 4).

**Basis.** Mapped from the code at `experiment/lab-exp1` after the restructuring.

- **Callers** come from a reference scan over all `.py` / `.sh` files.
- **Dynamic dispatch** is noted where it applies: LangGraph nodes and goal predicates are reached via
  `getattr(self, "_" + name)`, and the scenario registry via `import_module`.
- **Only implemented code is listed.**

**Status vocabulary:**

| Status | Meaning |
|---|---|
| **live-verified** | on the live path (`python -m agent.run` → supervisor) and observed working in a recorded live run (run named) |
| **offline-tested** | on the live path, covered by mocked tests, never (successfully) exercised live |
| **failing** | on the live path, currently fails because of a known baseline defect (BD-n, `TECHNICAL_DEBT.md` §2) |
| **new** | added by the restructuring (none left untested after 2026-10-18) |
| **experimental** | evaluation-only code, or code with no production caller |
| **inactive** | present in production code but not reachable on the live path |
| **legacy** | previous agent stack (`launch/run.sh`); still launchable, not maintained |
| **frozen** | part of the frozen manipulation baseline (checksummed; never edited) |

**Python environments** (details in `docs/DEVELOPER_GUIDE.md`):

| Code | Environment |
|---|---|
| agent, YOLO, Gemma | `.venv-seg` |
| Isaac Sim | `.venv` |
| legacy agent | `.venv-vlm` |
| viewers, `/usr/bin/python3` test run | system Python |

## A. Entry point, session, supervisor and planning

| Functionality | Implementation | Called by | Inputs → outputs | State | Status | Test independently |
|---|---|---|---|---|---|---|
| CLI / interactive loop | `agent/run.py` `main(controls=None, on_directive=None)` | `launch/agent.sh`, `evaluation/lab_exp1_agent.py`, user | instructions (stdin or `--instruction`) → `out/demo/<date>/live_*/NN/{events.jsonl, task_state.json, result.json, images/}` | process | live-verified (Experiment 1, current-system test) | manual: DEVELOPER_GUIDE §6 |
| Session (state across instructions) | `agent/session.py` `Session.run()` | `agent/run.py` | instruction → report; one `Tools` per process | owns the session `Tools` (world model, held object, memory, referents) | live-verified | `tests/integration/test_session.py` (mocked `Tools`) |
| GPU bring-up order (Gemma → YOLO → simulator → ROS ready) | `agent/run.py` `main()` (inline), `start_sim()` | — | flags → loaded models, optional sim process | process | live-verified | manual: DEVELOPER_GUIDE §6 |
| Goal supervisor (LangGraph) | `agent/supervisor.py` `GoalSupervisor.run()`; nodes `_interpreting`, `_dispatching`, `_delegating`, `_reviewing`, `_replanning`, `_verifying_goal` (via `_wrap` / `getattr`) | `Session.run` | instruction → report dict (status, outcome, message, plan, conditions, goal_check, authority, refused, timeline) | per instruction: `TaskState`, `Context`, `authority`, `specs`; reads `Tools` | live-verified | `tests/integration/test_supervisor_goal.py`, `tests/integration/test_orchestration.py` |
| Execution controls (real / dry-run / blocked motion, disabled capabilities, replanning limit) | `agent/controls.py` `ExecutionControls`; read by `Tools.execute`, `GoalSupervisor` (`specs`, `_refuse_disabled`, `max_replans`); `agent/ask.py` `LineReader.on_directive` | `evaluation/lab_exp1_agent.py`; default everywhere else | settings → refused / logged motions, `unsupported` steps | one object per process (motion may change at runtime) | offline-tested; live-verified with blocked and dry-run motion (2026-10-18, laboratory) | `tests/integration/test_controls.py` |
| Request authority (question / description / action → allowed effects) | `agent/authority.py` `classify()`, `from_classification()`; `agent/vlm_roles/roles.py` `Roles.classify_request()` (`REQUEST_SYSTEM` in `prompts.py`) | `GoalSupervisor._interpreting` | instruction → `RequestAuthority(kind, grants, evidence)` | none | live-probed; **failing for negated commands (BD-3)** | `tests/integration/test_authority.py` (decision tests) |
| Goal planning (instruction → capability steps) | `agent/vlm_roles/roles.py` `Roles.plan_goal()` (Gemma, `PLAN_SYSTEM`, structural validation `check_steps`) | `_interpreting` | instruction + capability menu + context → `GoalPlan(goal, steps, reason)` | none | **failing: 3/8 valid plans offline (BD-1); located targets rejected (BD-2)** | `tests/integration/test_supervisor_goal.py::test_planner_*` (recorded answers); live: `out/lab_exp1/milestone3/trackA/scripts/planner_ab.py` |
| Closed-loop decision (continue / replan / ask_user / finish / stop) | `agent/vlm_roles/roles.py` `Roles.decide()` (Gemma, `DECIDE_SYSTEM`, `check_decision`) | `GoalSupervisor._deciding` | goal + progress + latest event → `Decision` | `TaskState.consults`, `questions`, `replans` | offline-tested (scripted decisions); not yet live | `tests/integration/test_feedback_loop.py::test_gemma_*`, `test_roles_decide_validates_*` |
| Multi-object decomposition (goal → `#id` steps + final conditions) | `agent/capabilities/planning.py` `run_decompose()`, `world_facts()`; `Roles.decompose()`; insertion in `_reviewing` | planner plans `decompose` | goal + measured facts → `SubPlan(steps, goal conditions)` | reads `Tools.world` | **experimental**: live once, proposed manipulation for a question (now refused by authorization) | `test_decomposition_*`, `test_question_decomposed_into_pick_and_place_moves_nothing` |
| Pronoun binding ("it" → held / last object) | `GoalSupervisor._bind_refs()`; `_adapt_to_robot_state()` | `_dispatching` | step args → args with `#id` | reads `Tools.held_object`, `last_referent` | offline-tested | `tests/integration/test_session.py`, `test_follow_up_pronoun_*` |
| Decision bounds and guards | `GoalSupervisor._deciding`; budget `controls.max_replans` (settings `MAX_GEMMA_DECISIONS`, default 3); repeated-first-step guard; dropped objectives kept in the goal (`_refresh_conditions`) | `_reviewing`, `_verifying_goal` | decision → accepted / sent back / stop | `TaskState.attempted`, `replans` | offline-tested | `test_a_repeated_failed_step_is_sent_back_to_gemma`, `test_replan_that_drops_the_objective_cannot_complete_the_goal`, `test_the_loop_stops_when_the_budget_is_used` |
| Uncertainty → closer look (decided by Gemma, no rule) | `_reviewing` sends an uncertain result to `_deciding` with the worker's advice (`needs: closer_look`) | after `answer` | answer data `certain=False`, `check_closer_world` → Gemma's re-plan | `TaskState.plan` | offline-tested | `tests/integration/test_orchestration.py::test_uncertain_answer_*` |
| Not visible → explore then locate | `_reviewing` (locate branch) | after `locate` insufficient_info | → inserted `explore`, `locate` | `TaskState.plan` | live-verified (active-perception runs, default scene) | `test_not_visible_target_triggers_one_exploration_*` |
| Sub-goal and goal verification (verification worker) | `agent/capabilities/verification.py` `run_verify()` (delegated by `GoalSupervisor._reviewing` after each physical / locating step and by `_verifying_goal`); unmet goal with budget left → `_replanning` (Gemma) | supervisor | conditions → satisfied / not / undecidable with evidence | reads `Tools.world`, `TaskState` | offline-tested (`tests/integration/test_feedback_loop.py`) | `test_a_failed_subgoal_*`, `test_a_goal_not_shown_*` |
| Goal conditions and their evaluation | `agent/goals.py` `derive_conditions()`, `GoalEvaluator.evaluate()` + predicates `_holding`, `_not_holding`, `_on`, `_relation`, `_located`, `_observed`, `_ordered`, `_answered`, `_reported`, `_at_home` (via `getattr`) | `_verifying_goal` | conditions + bindings → `ConditionResult(satisfied, confidence, evidence)` | reads `Tools.world`, finger, `TaskState` | offline-tested; live verdicts recorded. **`_reported` accepts any report (BD-7)** | `tests/integration/test_orchestration.py` (goals), `tests/integration/test_authority.py` (answered) |
| Result message text | `_result_message()`, `_done_so_far()`, `_capabilities_line()` | `_finish` paths | steps → text | none | live-verified | indirect (supervisor tests assert messages) |
| Typed contracts (sender / receiver, goal, sub-goal, context, constraints, expected; status, outputs, evidence, uncertainty, needs, tools), registry, dispatch + authorization check | `agent/workers.py` `WorkerRequest`, `WorkerResult`, `assess()`, `WorkerRegistry.dispatch()` | `_delegating` | request + context + authority → result (or `unauthorized` before execution) | per-worker status | offline-tested; live dispatch logged | `test_registry_routes_capabilities_*` |
| Task ledger | `agent/task_state.py` `TaskState` (`record_observation`, `save`) | supervisor, capabilities | → `task_state.json` | per instruction | live-verified | indirect |
| Clarification questions to the user | `agent/ask.py` `TerminalResponder`, `ScriptedResponder`, `LineReader` (singleton; `on_directive` for harness lines), `parse_answer` | `PickPlaceWorkflow._question` | question → `Answer(kind, mark)` | stdin reader thread | live-verified | `tests/integration/test_agent_ask.py` |

## B. Capabilities (what the supervisor calls)

The capabilities live in `agent/capabilities/`, and every name is importable from `agent.capabilities`.

- **Calling convention.** Each capability is a `run_<name>(ctx, args, …) -> StepResult`, reached via
  `GoalSupervisor._executors()` → `WorkerRegistry.dispatch`.
- **Authorization** depends on the effect class (`SPECS[name].effect`, `agent/capabilities/specs.py`).
- **Availability per supervisor** comes from `capabilities.available(controls.disabled)`.

| Capability | File and function | Effect | Inputs → outputs | Status | Test |
|---|---|---|---|---|---|
| observe | `perception.py` `run_observe()` (→ `Tools.perceive`, `describe`, `update_world`, optional `park_for_view`) | none (park = arm motion) | → fresh `Observation` in `ctx.obs`, facts in ledger | live-verified | `test_observation_only_goal_*` |
| describe_scene | `perception.py` `run_describe_scene()`, `scene_report()` | none | obs → text report | live-verified | `test_result_comes_before_*` |
| answer | `perception.py` `run_answer()`, `facts_text()`; `Roles.answer()` | none | question + measured list → `SceneAnswer` (unverified) | live-verified (answers unverified by design) | `test_scene_question_*`, `test_counting_answer_*` |
| locate | `perception.py` `run_locate()`, `_location_check()`, `_find_in()` (wrist); `Tools.resolve/confirm` | none | reference (+ relation/anchor) → mark, world id, position | live-verified (default scene); location check offline-tested only | `test_locate_*`, `test_wrong_location_target_*` |
| inspect | `inspection.py` `run_inspect()`, `_wrist_observation()`; `perception.py` `_target_mark()`; `perception/viewpoints.py` `ViewpointPlanner` | sensing | target → wrist measurement + description; arm back HOME | live-verified (active perception, default scene); **lab: not exercised (targets unresolved, BD-4)** | `test_camera_inspection_stays_available_*`, `evaluation/historical/inspect_check.py` (live) |
| explore | `inspection.py` `run_explore()`; `perception/viewpoints.py` `shadow_points()`, `ViewpointPlanner` | sensing | target (+ near) → wrist views of hidden regions | live-verified (active perception C / C′ / D) | `explore_stub` tests (logic mocked) |
| pick | `manipulation.py` `run_pick()` → `_skill("pick")` → `PickPlaceWorkflow(mode="pick")` | object_change | object → held, retention verified, no HOME | offline-tested | `test_pick_only_ends_holding_*` |
| place | `manipulation.py` `run_place()` (re-verifies the held object) → `_skill("place")` | object_change | relation + reference → placed + HOME | offline-tested | `test_follow_up_pronoun_*`, `test_losing_the_object_*` |
| pick_place | `manipulation.py` `run_pick_place()` → `_skill("pick_place")` | object_change | object, relation, reference → placed + verified + HOME | live-verified (default scene, regression F: yellow cube → tan box, red cube → green cube) | `tests/integration/test_agent.py` workflow tests, `test_pick_and_place_goal_*` |
| decompose | `planning.py` `run_decompose()`, `world_facts()` | none (adds steps) | goal → steps + conditions | experimental (see A) | see A |
| return_home | `manipulation.py` `run_return_home()` → `Tools.return_home`, `verify_home` | sensing | → arm at HOME (joint check 1.5°) | live-verified | `test_return_home_*` |
| failure mapping | `manipulation.py` `_failure_outcome()` | — | workflow failure class → outcome | offline-tested | `test_failure_classes_map_*` |
| capability registry, menu | `specs.py` `SPECS`, `CapabilitySpec`, `available()`, `planner_menu(specs)`, `StepResult`, `Context` | — | → planner menu text, specs per supervisor | offline-tested (`available` is new) | `test_planner_menu_lists_only_registry_capabilities` |

## C. Perception, Gemma vision, localization, world model, memory

| Functionality | Implementation | Called by | Inputs → outputs | State | Status | Test |
|---|---|---|---|---|---|---|
| Camera snapshot (RGB-D + joints, after the arm is still) | `agent/ros_bridge.py` `RosBridge.snapshot()`, `q()`, `finger()`, `wait_still()` | `Tools.perceive`, many | camera name → `Frame(rgb, depth, K, stamp, q, finger)` | background ROS node | live-verified | `evaluation/lab_probe.py` (raw ROS), DEVELOPER_GUIDE §4 |
| Instance segmentation (YOLO11s-seg, round-1 weights) | `perception/detector.py` `YoloDetector.detect()`; weights and settings in `config/models.py` | `Tools.perceive` | rgb → `[Detection(mask, score)]` | model on GPU | live-verified; **merges containers with racks / trays on the lab table view (BD-4); run-to-run variability (BD-8)** | `evaluation/seg_generalization.py` (27 saved frames) |
| Mask + depth → measured object | `perception/geometry.py` `measure()`, `back_project()`, `RobotVolume`, `snap_supports()`; thresholds in `config/perception.py`; `perception/grounding.py` `footprint_yaw`, `grasp_pose` | `Tools.perceive`, `evaluation/seg_*` | mask, depth, K, camera pose → `Measured` (centre, top, height, footprint, yaw, grasp width, tilt) | none | live-verified (positions 1–5 mm, tests) | `tests/integration/test_agent.py::test_measure_*`, `evaluation/seg_diagnosis.py` |
| Perception step | `agent/tools/perception.py` `Tools.perceive()` | `run_observe`, `verify_placement`, `_wrist_observation`, `PickPlaceWorkflow._observing` | camera → `Observation(marks, measured, scores, som)` + `images/*_som.jpg` | `Tools._frames`, `motion_count` | live-verified | via fakes |
| Gemma object descriptions (per isolated crop, cached) | `agent/tools/perception.py` `Tools.describe()`; `agent/tools/memory.py` `DescriptionMemory`; `Roles.describe_many()` / `describe_one()` | `run_observe`, `resolve` | marks → `{name, colour, shape, material}` (hypotheses) | `DescriptionMemory` per session | live-verified (names often wrong for lab objects; cap colours right) | `test_describe_one_per_isolated_crop`, current-system test Step 3 |
| Batched descriptions / VLM shortlist | `Roles.describe_many()` batch branch (only if the VLM has `generate_batch`; none does), `Tools.shortlist()` | no production caller (`Tools.shortlist` only in `evaluation/historical/demo_describe_bench.py`, tests) | — | — | **inactive** (rejected after the latency benchmark) | `test_shortlist_describes_only_candidates`, `test_describe_many_*` |
| World model (persistent ids by geometry, per-camera observations, status) | `perception/world_model.py` `WorldModel.observe()`, `associate()`, `match()`, `add_observation()`, `summary()`, `same_place()`; `Tools.update_world()` | `Tools.perceive` path, capabilities | `Measured` (+ description) → world id | `Tools.world` (session) | live-verified; duplicate masks can share one id | `test_world_model_keeps_ids_by_geometry_only`, `test_cross_camera_association_and_fusion_rule`, `test_partial_wrist_view_of_a_large_object_is_associated_not_new` |
| Target resolution (reference text → mark) | `agent/tools/resolution.py` `Tools.resolve()` (Gemma over the SoM image + descriptions), `lexical_candidates()`, `Roles.compare()`, `confirm()` on the crop | `run_locate`, `PickPlaceWorkflow._resolve_one` | text → `Resolution(id, ambiguous, candidates)` | reads `Observation` | live-verified (default scene); lab: wrong-object confirmations observed | `test_resolve_*`, `test_lexical_cross_check_*` |
| Location constraint on targets ("in the buffer tray") | `agent/capabilities/perception.py` `_location_check()`; `agent/tools/resolution.py` `Tools.relation_holds()` (incl. `in`) | `run_locate` | candidate + anchor → reject / accept (logged `location_check`) | none | offline-tested (live: never reached, BD-1 / BD-2) | `test_wrong_location_target_*`, `test_in_relation_is_measured_containment` |
| Observation freshness | `agent/tools/perception.py` `Tools.is_fresh()`; `motion_count`, `view_cleared_at` | supervisor `_dispatching`, capabilities | obs → bool | `Tools` counters | offline-tested | `test_observe_then_manipulate_reuses_*` |
| Held object state + re-verification | `agent/tools/held.py` `set_held()`, `clear_held()`, `begin_instruction()`, `revalidate_held()` (ownership in the module docstring) | workflow, `Session`, `run_place` | finger feedback → verified / invalidated | `Tools.held*`; `world.holding` mirrors `held_object` | offline-tested | `tests/integration/test_session.py` |
| Clarification memory ("which one?" answers) | `Tools.resolved_refs`; `PickPlaceWorkflow._resolve_one` | workflow | text → world id | session | offline-tested | `test_id_reference_and_remembered_clarification_*` |
| Set-of-mark images, crops, montages | `perception/somark.py` `draw_marks`, `draw_names`, `isolate`, `grid`, `montage` | `Tools`, capabilities | masks → images | none | live-verified | indirect |
| Gemma adapter, prompts, schemas | `agent/vlm_roles/gemma.py` (`GemmaVLM`, `RecordedVLM`), `prompts.py`, `schemas.py`, `roles.py` | `Session`, `Tools`, capabilities | text (+ image) → validated pydantic objects or `VLMFailure` | call counters per instruction | live-verified | `tests/integration/test_agent.py` (`RecordedVLM`) |
| Live viewer channel + viewer | `agent/live.py` `LiveChannel`; `agent/viewer.py`, `agent/events.py` `format_event` | `Session`, `Tools.perceive` | events → files → window | files under `out/demo/live/` | live-verified | `tests/integration/test_demo_viewer.py` |
| Experiment record (JSONL events) | `agent/eventlog.py` `EventLog` | everything in `agent/` | kind + fields → `events.jsonl` (line-buffered) | file | live-verified | indirect |

## D. Camera inspection and exploration geometry

| Functionality | Implementation | Called by | Status | Test |
|---|---|---|---|---|
| Wrist-camera viewpoint planning (look-at IK, line of sight, collision-checked path) | `perception/viewpoints.py` `ViewpointPlanner.plan()`, `look_at_ik()`, `reached()` | `run_inspect`, `run_explore`, `scenarios/checks.py` `view_feasibility`, evaluation | live-verified | `evaluation/validate_scenarios.py` (wrist views), scenario checks |
| Hidden regions behind measured objects | `perception/viewpoints.py` `shadow_points()` | `run_explore` | live-verified | indirect |
| Park the arm out of the table camera's view | `agent/tools/manipulation.py` `Tools.park_for_view()`, `view_blockage()`, `workspace_points()`, `_rays_blocked()` | `run_observe(clear_view)`, workflow | live-verified | `test_park_pose_choice_*` |

## E. Grasping, placement, motion planning (agent side) and the frozen core

| Functionality | Implementation | Called by | Status | Test |
|---|---|---|---|---|
| Pick-and-place workflow (states OBSERVING … VERIFYING_HOME, recovery budgets, 3 modes) | `agent/pick_place.py` `PickPlaceWorkflow.run()` + state methods (via `getattr`); budgets in `config/agent.py` | `capabilities._skill` | pick_place live-verified (default scene); pick / place modes offline-tested | `tests/integration/test_agent.py` workflow tests |
| Grasp / place planning from measured geometry | `agent/tools/manipulation.py` `Tools.plan_pick_place()`, `plan_pick()` (both via `_plan_grasp()`), `plan_place_held()`, `_plan_place()`, `_too_close()`, `deterministic()`; relations in `config/agent.py` | workflow | live-verified (pick_place) | `test_plan_on_*`, `test_plan_next_to_*` |
| Collision obstacles from the world model + static fixtures | `Tools.obstacles()`; `config/perception.py` `FIXTURES` | planning, park, viewpoints | live-verified | indirect |
| Contact / placement / HOME verification | `agent/tools/verification.py` `verify_contact()`, `verify_placement()`, `verify_home()`; `agent/tools/manipulation.py` `assume_placed()`, `return_home()`; tolerances in `config/agent.py` | workflow, goals, capabilities | live-verified | `test_home_*`, `test_verify_home_*` |
| Kinematics, IK | `control/kinematics.py` `Kinematics` | everywhere | **frozen** | `tests/unit/test_kinematics.py`, `test_vt6l_model.py` |
| Trajectories | `control/motion.py` `joint_move`, `linear_move`, `PlanningError` | skills, tools | **frozen** | `tests/unit/test_skills_and_plans.py` |
| Collision checking (robot spheres vs boxes) | `control/collision.py` `CollisionChecker`, `Box` | skills, tools, viewpoints, scenario checks | **frozen** | `tests/unit/test_collision.py` |
| Pick / place skill plans | `control/skills.py` `SkillPlanner.pick/place`, `Held`, `GripperStep`, `MoveStep` | tools, scenario checks | **frozen** | `tests/unit/test_skills_and_plans.py`, `test_scene_reachability.py` |
| Robot constants, gripper calibration, **table-camera pose** | `control/robot.py` | most modules | **frozen** | `tests/unit/test_vt6l_model.py` |
| Legacy text → motion parser | `control/instructions.py` | `ros_nodes/command_interpreter.py`, legacy agent | **frozen**, legacy | `tests/unit/test_skills_and_plans.py` |

## F. Robot execution and safety

| Functionality | Implementation | Called by | Status | Test |
|---|---|---|---|---|
| Execute planned steps over ROS (trajectory streaming, reach check) | `execution/ros_executor.py` `RosExecutor.execute()` | `Tools.execute` | **frozen**, live-verified | live only; mocked in agent tests |
| Agent-side execute (motion gate, motion counter, logging) | `agent/tools/execution.py` `Tools.execute()`, `_not_executed()` (dry_run / blocked) | all motion | real mode live-verified; dry_run / blocked offline-tested and live-verified (2026-10-18; nothing sent) | `tests/integration/test_controls.py` |
| Joint limits + speed monitor (inside the simulator) | `safety/joint_limits.py` `JointLimits`; enforced by `sim/ros_control.py` `JointCommandInterface` | sim process | **frozen**, live-verified | `tests/unit/test_safety.py` |
| Authorization before execution | `agent/workers.py` `WorkerRegistry.dispatch` + `agent/authority.py` + supervisor refusals | supervisor | offline-tested; live: no violation in 4 requests (milestone 3) | `tests/integration/test_authority.py` |

## G. Isaac Sim, ROS 2 and scenarios

| Functionality | Implementation | Called by | Status | Test |
|---|---|---|---|---|
| Live simulator (scene, physics loop, ROS publishers, command interface, ground-truth topic) | `sim/run_live.py` `main()` (creates `SimulationApp` at import), `sim/scene.py` `setup_scene()`, `sim/ros_control.py`, `sim/ros_bridge.py` | `launch/*`, `agent/run.py start_sim`, `sim/run_scenario.py` | **frozen** (except `ros_bridge.py`), live-verified | `evaluation/validate_scenarios.py`, `scripts/validate_robot_m1.py` |
| Scene selection (JSON scenes, scenario library) | `sim/run_scenario.py` (fills the frozen `OBJECTS`, wraps `setup_scene`) | `launch/agent.sh --scenario`, `launch/sim.sh`, `agent/run.py --scenario`, experiment drivers | live-verified | sim smoke via the validator |
| Scenario library: data model, placement, metadata | `scenarios/core.py` (`Scenario`, `Placer`, `penetration`, `overlaps`, `metadata()`), `scenarios/parts.py` (rack, tray, plinth, capped container, …), `scenarios/laboratory.py` `build()` | `generate()` in `scenarios/__init__.py` | active, validated (seed 42, 3 levels) | `tests/unit/test_scenario_library.py` |
| Scenario build in Isaac | `scenarios/build.py` `install()`, `populate()`, `overview_camera()`, `label_texture()` | `run_scenario`, validator, frame capture | active, validated | `evaluation/validate_scenarios.py` (`.venv`) |
| Scenario design checks (spawn, visibility, grasp / placement / viewpoint feasibility) | `scenarios/checks.py` `static_report()` (+ CLI `python -m scenarios.checks`) | tests, validator | active | `tests/unit/test_scenario_library.py` |
| JSON scenes (active perception) | `scenarios/scenes/ap_occlusion.json`, `ap_categories.json` | `run_scenario` | active (historical phase) | `evaluation/scenario_design.py` |
| URDF → USD conversion | `sim/import_vt6l.py` | manual, once | active (setup tool) | manual |
| Synthetic training data | `datagen/*` (layout, generate in Isaac, coco, textures, verify) | manual | active (offline tooling) | `tests/unit/test_datagen.py` |

## H. Experiments, evaluation, test utilities

| Group | Files | Status |
|---|---|---|
| Laboratory Experiment 1 (protocol harness, agent wrapper with execution controls, ROS probe, scorer) | `evaluation/lab_exp1.py`, `lab_exp1_agent.py`, `lab_probe.py`, `lab_exp1_score.py`, `truth_snapshot.py` | experimental (in use) |
| Segmentation diagnosis / generalization, frame capture | `evaluation/seg_diagnosis.py`, `seg_generalization.py`, `lab_capture_frames.py` | experimental (in use) |
| Scenario validation | `evaluation/validate_scenarios.py`, `scenario_design.py` | active |
| Live demo driving | `evaluation/demo_live_driver.py` | active tool |
| Historical phase checks (still runnable, not maintained) | `demo_rehearsal.py`, `demo_summary.py`, `demo_latency.py`, `demo_describe_bench.py`, `demo_plan_dryrun.py`, `demo_vlm_check.py`, `demo_home_check.py`, `dual_camera_check.py`, `inspect_check.py`, `active_summary.py`, `planner_check.py`, `grasp_feasibility_map.py` | experimental / historical |
| Manipulation experiments | `experiments/known_pose_pick_place.py`, `known_pose_suite.py`, `payload_study.py` (**frozen**); `experiments/pick_place.py`, `trajectory_demo.py` (early demos) | frozen / historical |
| Validation scripts | `scripts/validate_robot_m1.py` (**frozen**), `scripts/validate_cameras_phaseA.py` | historical |
| Segmentation training / evaluation | `seg/train_yolo.py`, `train_maskrcnn.py`, `evaluate.py`, `eval_rounds.py`, `pretrained_recall.py`, `data.py` | offline tooling |
| Shared test fakes and runners | `tests/support/fakes.py` (`geom`, `StubTools`, `GoalTools`), `tests/support/runners.py` (`run_workflow`, `run_supervisor`, `plan`, `TASK`) | active |

## I. Legacy agent stack (not used by `agent/`, except the noted shared parts)

| Part | Files | Status |
|---|---|---|
| Previous agent node | `ros_nodes/vlm_agent_node.py` (protected: do not modify), `launch/run.sh` (protected), `launch/legacy/start_vlm_agent.sh`, `launch/legacy/tasks.sh` | legacy |
| Planner / plan schema | `vlm/planner.py`, `vlm/plan.py`, `vlm/prompts.py` | legacy |
| Model loader (**shared**) | `vlm/model.py` `VisionLanguageModel`, `vlm/config.py` | **active** (used by `agent/vlm_roles/gemma.py` `GemmaVLM`) |
| Box grounding (**shared**) | `perception/grounding.py`: `footprint_yaw`, `grasp_pose` used by `perception/geometry.py`; also used by the legacy node and (through the classical detector) the frozen known-pose experiments | mixed |
| Memory, classical detector | `perception/world_memory.py` (legacy node), `perception/classical_detector.py` (frozen `experiments/known_pose_suite.py`, `scripts/validate_cameras_phaseA.py`, tests) | legacy / shared with frozen code |
| Text command path | `ros_nodes/command_interpreter.py`, `text_command_node.py`, `microphone_node.py`, `launch/legacy/start_live.sh` | legacy (text only; voice off) |
| Camera viewers | `ros_nodes/camera_viewer.py`, `cameras_view.py` | active tools (used by `agent.sh`, `run.sh`) |

## J. Code removed by the restructuring

These had no caller anywhere. The check covered `getattr` and string dispatch, decorators, entry points, shell
scripts, the legacy node and the frozen files.

| Symbol | Former file |
|---|---|
| `run_unimplemented`, `elapsed`, `Kinematics_tcp` (a wrapper of `Kinematics()`) | `agent/capabilities.py` |
| `WorldModel.mark_unseen`, `WorldModel.to_dict` | `agent/world.py` |
| `GemmaVLM._generate_batch_disabled` | `agent/vlm_roles.py` |
| `FakeDetector` | `agent/detector.py` |
| `ObjectEstimate.grasp_text` | `perception/grounding.py` |
