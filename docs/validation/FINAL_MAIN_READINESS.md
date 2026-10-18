# Final main-branch readiness: `experiment/lab-exp1` (2026-10-18)

## Decision: **READY FOR MANUAL ACCEPTANCE**

Automated verification passed at every level that could be run without commanding robot motion:

- the offline unit and integration suites;
- saved-sensor-data perception, including real YOLO inference;
- real Gemma inference;
- Isaac Sim scenario loading and validation;
- ROS 2 interfaces;
- a full live agent run on the laboratory scene with motion **blocked** and **dry-run**.

No refactoring regression was found. Every comparison with the pre-restructuring code (`aa9a6a4`) that could be
made was identical.

**Not READY TO MERGE yet**, because it needs:

1. your manual acceptance (section 11);
2. a manipulation run with real (simulated) motion, which this task excluded by instruction;
3. your explicit approval to merge.

## 1. Final directory structure

```
agent/            orchestration: run.py (entry), session.py, supervisor.py, planning.py, workers.py, authority.py,
                  goals.py, task_state.py, controls.py, pick_place.py, ask.py, ros_bridge.py, eventlog.py,
                  events.py, live.py, viewer.py
  capabilities/   specs.py, perception.py, inspection.py, manipulation.py, planning.py
  tools/          held.py, perception.py, memory.py, resolution.py, manipulation.py, execution.py, verification.py
  vlm_roles/      gemma.py, prompts.py, schemas.py, roles.py
perception/       detector.py, geometry.py, somark.py, world_model.py, viewpoints.py
                  (+ grounding.py shared; classical_detector.py, world_memory.py legacy, used by frozen / protected code)
config/           perception.py, agent.py, models.py
scenarios/        core.py, parts.py, laboratory.py, build.py (Isaac), checks.py, scenes/*.json
sim/              run_live.py*, scene.py*, scene_objects.py*, ros_control.py*, run_scenario.py, ros_bridge.py, import_vt6l.py
control/*, execution/*, safety/*      frozen manipulation core
evaluation/       experiments, probes, scorers, studies (README.md: active vs historical)
experiments/      known_pose_*.py*, payload_study.py* (frozen); early demos
tests/            23 test files + tests/support/ (shared fakes, runners)
vlm/, ros_nodes/, launch/run.sh, launch/start_vlm_agent.sh, launch/tasks.sh, launch/start_live.sh   legacy stack
datagen/, seg/    synthetic data generation, segmentation training (offline tooling)
docs/             architecture/, scenarios/, validation/, baselines/ (frozen manifest), DEVELOPER_GUIDE.md
assets/*          URDF / USD                                              (* = frozen, checksummed)
```

Per-file detail: `docs/architecture/MODULE_GUIDE.md` §2.

## 2. Module responsibility map

| Responsibility | Owner |
|---|---|
| Process entry, model loading order, optional simulator start | `agent/run.py` |
| State across instructions (one `Tools`) | `agent/session.py` |
| Goal state machine, review policy | `agent/supervisor.py` |
| Plan completion / objective rules | `agent/planning.py` |
| Request authority; enforcement before execution | `agent/authority.py`; `agent/workers.py` `WorkerRegistry.dispatch` |
| Goal conditions over measured state | `agent/goals.py` |
| Per-instruction ledger | `agent/task_state.py` |
| Experiment switches (motion real / dry_run / blocked, disabled capabilities, replans) | `agent/controls.py` |
| Capability registry, menu; capability executors | `agent/capabilities/` |
| Pick / place / pick-and-place workflow | `agent/pick_place.py` |
| The agent's only path to sensors and motion | `agent/tools/` (`Tools`) |
| Gemma adapter, prompts, schemas, role calls | `agent/vlm_roles/` |
| Detection, depth geometry, marks, world model, viewpoints | `perception/` |
| Configuration values, model paths | `config/` (robot constants: frozen `control/robot.py`; Gemma: `vlm/config.py`) |
| Scenario definitions, builders, checks, scenes | `scenarios/` |
| Simulator processes | `sim/` |
| Kinematics, planning, collision, execution, joint limits | `control/`, `execution/`, `safety/` (frozen) |
| Experiments, diagnostics, scoring | `evaluation/` |

**Session state ownership** (documented in `agent/session.py` and `agent/tools/held.py`):

- `Tools` owns the world model, the held object, the description memory, clarifications and the last referent.
- Each instruction gets new copies of the rest: event log, Gemma roles, supervisor, `TaskState`, LangGraph thread.

## 3. Redundancy and dead-code cleanup (whole restructuring)

**Removed (no caller anywhere, including dynamic and string dispatch, launch scripts, frozen and protected
files):**

- `run_unimplemented`, `elapsed`, the `Kinematics_tcp` wrapper;
- `WorldModel.mark_unseen`, `WorldModel.to_dict`;
- `GemmaVLM._generate_batch_disabled`;
- `FakeDetector`;
- `ObjectEstimate.grasp_text`;
- unused imports.

**Consolidated into one definition:**

| What | Now in |
|---|---|
| same-place gate (world model + description memory) | `perception.world_model.same_place` |
| grasp-planning prefix | `_plan_grasp` |
| camera-ray blockage count | `_rays_blocked` |
| configuration | `agent/config.py` + model constants → `config/` |
| reach ring 0.45–0.90 m (two copies) | `config.agent.REACH_MIN_M / REACH_MAX_M` |
| executor reach tolerance 1.5° (two copies) | `config.agent.REACH_TOLERANCE_DEG` |
| wrist-camera frame name (two copies) | `perception.viewpoints.CAM_TIP` |
| YOLO weights path (six copies) | `config.models.YOLO_WEIGHTS` |
| Experiment 1 monkeypatches (six patch sites) | explicit `ExecutionControls` |

**Kept deliberately, documented:**

| Kept | Why |
|---|---|
| legacy stack (`vlm/` planner, `ros_nodes/vlm_agent_node.py`, `perception/world_memory.py`, `perception/classical_detector.py`, legacy launchers) | protected files and frozen experiments import them |
| historical evaluation scripts | they reproduce reported results (`evaluation/README.md`) |
| `Tools.shortlist` and the batched-description branch | inactive; benchmarked and tested |
| three `rot_z` copies | two of them are in frozen files |
| two ray-vs-box helpers | different inputs: measured boxes vs scenario parts |

**Debt register status** (`docs/architecture/TECHNICAL_DEBT.md`):

| Item | Status |
|---|---|
| TD-4 (monkeypatching) | **resolved** |
| TD-1 (`Tools` god object) | **resolved structurally**: one responsibility per file; still one shared object by design |
| TD-5 (layering) | **mostly resolved**: agent → sim and scenarios → agent removed; the frozen scene-object global remains |
| TD-8 (duplicate utilities) | partly resolved |
| TD-10 (tests) | improved: no cross-test imports, 69 new tests |
| TD-12 (evaluation mixing) | partly resolved: README; no runtime patching |
| TD-2 / TD-3 | partly resolved (see limitations L1, L2) |
| TD-6, TD-7, TD-11, TD-13 | open |
| TD-9 (packaging) | open **by your decision**: no packaging |

## 4. Automated test commands and results

Run from the repository root on 2026-10-18, at commit `f5c2557` plus documentation.

| Tier | Command | Result |
|---|---|---|
| **Unit + integration (full suite, agent env)** | `env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python -m pytest` | **383 passed**, 0 failed, 0 skipped |
| Unit + integration (system Python) | `/usr/bin/python3 -m pytest` | **199 passed**, 29 skipped (agent tests need pydantic / langgraph; they run in the row above) |
| Import smoke | the 30 `CURRENT_MODULES` in `tests/test_structure.py` | all import without Isaac or ROS |
| Static | `importcheck` (every local import resolves), `/usr/bin/python3 -m pyflakes agent perception config scenarios` | clean (one deliberate Isaac-side unused import in `evaluation/validate_scenarios.py`) |
| Frozen manifest | `grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 \| sha256sum -c` | **OK** |

**New tests in this phase:**

| File | Tests | What it covers |
|---|---|---|
| `tests/test_controls.py` | 18 | real / dry_run / blocked modes with a recording and a *forbidden* executor; disabled capabilities; replanning limit; Session wiring; stdin directives; the Experiment 1 harness and its `harness_gates.jsonl` records |
| `tests/test_structure.py` | 43 | copied constants vs frozen sources; paths and assets; registry = planner schema = executors; prompts pinned by hash; imports; no removed paths; no runtime patching (shown to flag all six patch sites of the old harness) |
| `tests/test_saved_frames.py` | 5 | depth localization on the saved Experiment 1 scenes; `Tools.perceive` + world model on a replayed frame; real YOLO inference |
| `tests/test_state_ownership.py` | 3 | the held-object invariant |

**Equivalence with the code before the restructuring (`aa9a6a4`, run side by side):**

| Check | Result |
|---|---|
| `measure()` on all 466 ground-truth masks of the 27 saved lab frames | **identical** (329 measured, 137 excluded with the same reasons) |
| `evaluation/seg_generalization.py` (real YOLO on the 27 frames, approaches A / B / B2 / C) | **identical** apart from timings |
| `evaluation/planner_check.py` (real Gemma, 12 instructions) | **identical plans**: 5 / 12 valid in both (BD-1) |
| prompt texts (11) | identical SHA-256 |
| moved definitions (`tools` 51, `capabilities` 31, `vlm_roles` 37, `config`) | identical ASTs |
| Isaac scenario validation, laboratory easy seed 42 (cameras, physics, design checks) | **identical** stage results to the pre-restructuring validation; 0 problems |

**Regressions found: none.** No test was weakened, skipped or rebaselined, and no BD defect was changed.

## 5. Integration coverage matrix

**Tiers:**

- **U** = unit / mocked;
- **I** = integration with real production code and saved data;
- **M** = real model inference;
- **S** = Isaac Sim + ROS 2 live (motion blocked);
- **—** = not exercised.

| Area | U | I | M | S | Evidence |
|---|---|---|---|---|---|
| Agent startup, session lifecycle | ✓ | ✓ | ✓ | ✓ | `test_session.py`, `test_controls.py`; live run (4 instructions, one session) |
| Supervisor planning and delegation | ✓ | ✓ | ✓ | ✓ | `test_supervisor_goal.py`, `test_orchestration.py`; `planner_check`; live run |
| Capability registration and dispatch | ✓ | ✓ | — | ✓ | `test_structure.py` (registry = schema = executors); live run |
| Gemma adapters and schemas | ✓ | ✓ | ✓ | ✓ | `RecordedVLM` tests; `planner_check`; live describe / plan / classify |
| YOLO perception, segmentation | — | ✓ | ✓ | ✓ | `test_saved_frames.py`; `seg_generalization` (identical); live (14 objects measured) |
| RGB-D localization | ✓ | ✓ | — | ✓ | `test_measure_*`; saved frames (xy ≤ 10 mm, z ≤ 5 mm on 3 scenes) |
| World model, target grounding | ✓ | ✓ | ✓ | partial | `test_agent.py` (resolve, lexical check, association); replayed frame; live grounding not reached (BD-2) |
| Camera inspection | ✓ | — | — | — | mocked inspect tests. Live inspection moves the arm, so it is left to manual acceptance |
| Grasp and motion planning interfaces | ✓ | ✓ | — | ✓ (planned, not executed) | `test_skills_and_plans.py`, `test_agent.py`; live: park planned and refused at the gate |
| Authorization, execution controls | ✓ | ✓ | ✓ | ✓ | `test_authority.py`, `test_controls.py`; live: 3 refused + 3 dry-run motions, 0 sent |
| Robot and gripper interfaces | ✓ | — | — | ✓ (read only) | `test_safety.py`, `test_kinematics.py`; live `/joint_states` 15.7 Hz, finger feedback used by `revalidate_held` |
| Isaac Sim scenario loading | ✓ | — | — | ✓ | `test_scenario_library.py`; `run_scenario` (lab easy s42); validator (0 problems); agent-started lab (easy s0) |
| ROS 2 interfaces | — | — | — | ✓ | 18 topics; table camera 14.5 Hz; both camera snapshots; `lab_probe` (ground truth + joints) |
| Experiment 1 harness and scoring | ✓ | ✓ | — | — | `test_controls.py` (harness guards), `test_lab_exp1_score.py`; a full protocol run includes motion, so it is left for manual acceptance |
| Configuration and asset paths | ✓ | ✓ | — | ✓ | `test_structure.py`; textures resolve in Isaac |
| Error handling, recovery | ✓ | — | ✓ | ✓ | workflow recovery tests; live: refused motion → FAILED with replan attempt, no crash |

## 6. Simulator and model-inference validation

All of these ran in `ROS_DOMAIN_ID=42`, one Isaac instance at a time, with nothing else on the GPU. Evidence is
in `out/validation/` (not in git).

1. **Scene load:** `sim.run_scenario --scenario laboratory --difficulty easy --seed 42 --headless`.
   - Log: "8 objects, 6 fixtures, 14 decor groups", no errors.
   - 18 ROS topics, including `/sim/ground_truth/objects` and `/safety/events`.
   - Camera snapshots: `sim/table.png`, `sim/wrist.png`. Labels render, and textures are read from
     `out/scenario_textures`.
2. **Validation:** `evaluation/validate_scenarios.py --scenario laboratory --difficulty easy --seed 42` gave
   0 problems.
   - The design checks ran through the new `python -m scenarios.checks`.
   - The results equal the earlier validation.
   - The earlier results in `out/scenarios/laboratory/` were preserved; the new ones are in
     `out/validation/sim/scenarios_laboratory/`.
3. **Live agent with motion blocked:** `agent.run.main(controls=ExecutionControls(motion="blocked"))`, which
   loads Gemma → YOLO → Isaac (headless) on the laboratory scene, and then a directive switches it to dry-run.

   | Instruction | Result |
   |---|---|
   | "Take a look at your surroundings." | **COMPLETED**: 14 objects measured and described |
   | "Which objects are in the incoming rack?" | FAILED at planning: the location-anchor validation (**BD-2**, known) |
   | "Pick up the vial." (blocked) | 3 park motions planned and **refused**; FAILED, nothing sent |
   | "Pick up the vial." (dry run) | 3 park motions planned and **logged, not executed**; FAILED |

   No `execute_start` event in any run, and no internal error. The simulator shut down cleanly.
4. **Model inference:**
   - Gemma 4 E2B (4-bit) through `GemmaVLM`: planner check plus all live roles.
   - YOLO11s-seg on the GPU: saved frames and live.

**Not exercised (by instruction or scope):**

- any motion in the simulator: pick, place, pick-and-place, inspection, explore, return HOME from away;
- the full Experiment 1 protocol (its inspection phase opens the motion gate);
- medium / hard re-validation and the reload / drop stages;
- the GUI window and the live demo viewer window;
- a physical robot (none exists).

## 7. Known defects and remaining limitations

**Known baseline defects** (unchanged; not fixed, per instruction):

| Id | Defect | Seen in this validation |
|---|---|---|
| BD-1 | planner regression from the `91429c4` prompt example | 5 / 12 planner_check plans valid, identical before and after |
| BD-2 | location-preservation validation rejects located requests | live instruction 2 |
| BD-3 | "Do not pick up the vial." classified as an action | — |
| BD-4 | segmentation merges containers with racks / trays | table-view recall of the production approach (A) 0.13–0.50 per saved scene, identical before / after |
| BD-5 | split-instance heights under-estimated | — |
| BD-6 | planner claims label reading | — |
| BD-7 | `reported()` accepts any report | — |
| BD-8 | YOLO run-to-run variability | none between the two YOLO runs here |

**Remaining limitations (architectural, documented rather than forced):**

| # | Limitation | Why it remains |
|---|---|---|
| L1 | The held object is recorded twice (`Tools.held_object`, `WorldModel.holding`) | Merging changes reads in `goals._holding`, `return_home`, the inspection guards and test fixtures that set `world.holding` directly. `tests/test_state_ownership.py` now pins the invariant that the two never disagree. Merge only with that test plus updated fixtures |
| L2 | The supervisor's review rules are not extracted into a pure function | No golden event-sequence tests pin `_reviewing` across all branches yet; extraction without them could change semantics |
| L3 | A motion refused by the controls during "park for view" is reported as "no collision-free pose that leaves the table camera's view clear" | This is the pre-existing workflow message (the old harness behaved the same). The events show `execute_refused` / `execute_dry_run` correctly |
| L4 | Old import paths were removed without compatibility modules | Scripts outside the repository that use them fail with `ModuleNotFoundError`; mapping in `MODULE_GUIDE.md` §4 |
| L5 | The legacy stack remains | Imported by protected / frozen files |
| L6 | `sim` ↔ `scenarios` mutual package dependency through the Isaac builder; legacy `classical_detector` → `sim.scene_objects` | No module cycle; the builder's job is to bridge them |
| L7 | No dependency manifest or lock files; four interpreters | By your decision against packaging; versions in section 9 |
| L8 | `agent/run.py` keeps harness options (`--sim-flag`, `--ask scripted`) | Used by `evaluation/historical/demo_rehearsal.py`; explicit and documented |

## 8. Protected-file integrity

| Check | Result |
|---|---|
| Frozen manifest (`sha256sum -c`, 21 files) | **OK** |
| `git diff main` on every frozen file + `launch/run.sh` + `ros_nodes/vlm_agent_node.py` | **0 files changed** |
| Experiment reports and evidence under `out/` | untouched. Validation outputs went to `out/validation/`; the overwritten scenario-validation folder was backed up and restored byte-identically |
| Tags `lab-env-baseline-v2.0`, `baseline-before-goal-supervisor`, `before-*` | present |

## 9. Required assets and environment dependencies

| Asset (not in git) | Path | Size | Needed by |
|---|---|---|---|
| Gemma 4 E2B-it | `models/gemma-4-E2B-it/` | 9.6 GB | agent (`vlm/config.py`) |
| YOLO11s-seg round-1 weights | `out/seg/runs/yolo11s_seg_r1/weights/best.pt` | 20 MB | agent (`config/models.py`) |
| YCB asset table and meshes | `data/assets/` | 209 MB | JSON scenes (`sim/run_scenario.py`) |
| Saved laboratory frames | `out/lab_seg_frames/` | 22 MB | `tests/test_saved_frames.py` (skips without them), segmentation studies |
| Isaac Sim 6.1 | `.venv/` | — | simulator |

`tests/test_structure.py::test_required_untracked_assets_are_present` lists any of these that are missing.

| Environment | Versions | Role |
|---|---|---|
| `.venv-seg` | Python 3.12.3, torch 2.11.0+cu130, transformers 5.19.0, bitsandbytes 0.50.2, ultralytics 8.4.177, langgraph 1.2.15, pydantic 2.14.1, numpy 2.5.2, opencv 5.0.0 | agent, models, full test suite |
| `.venv` | Python 3.12.3, isaacsim 6.1.0.0, numpy 2.3.1 | simulator, scenario validation |
| system | Python 3.12.3, numpy 1.26.4, opencv 4.6.0, pytest 7.4.4 + ROS 2 Jazzy | ROS tools, viewers, offline subset |
| `.venv-vlm` | — | legacy stack only |

GPU: 12 GB (RTX 4000 Ada Laptop). Load Gemma before the simulator, and run one Isaac instance at a time.

## 10. Git status and history

- Branch `experiment/lab-exp1`.
- Working tree clean after the final commit.
- No remote is configured.
- `main` (`c140d64`) is a direct ancestor: 0 commits ahead; this branch is ahead by `git rev-list --count main..HEAD` commits (45 at the time of writing).
- The only branch with commits outside this one, `feature/orchestration` (`9e10f59`), is patch-equivalent to
  `c636c6e` here (`git cherry` → `-`).

**Restructuring and validation commits** (oldest first; earlier milestone commits are in `git log main..HEAD`):

| Commits | Step |
|---|---|
| `aa9a6a4` | shared test fakes, navigation docs |
| `761ec65` … `46f0802` | restructuring steps 1–11 |
| `1a13abb` | restructuring docs |
| `902ef74` | constant consolidation |
| `f5c2557` | new tests |
| (this commit) | readiness report, evaluation index, historical-doc notes, `CLAUDE.md` pointer |

## 11. Manual acceptance checklist

Commands are in `docs/DEVELOPER_GUIDE.md`. Tick each item:

- [ ] Both suites pass on your machine (section 4 commands).
- [ ] `./launch/agent.sh --scenario laboratory`: Isaac window, demo viewer and prompt appear (motion BLOCKED).
- [ ] "Take a look at your surroundings." → COMPLETED with a scene list; the viewer shows masks and ids.
- [ ] A question → an answer, or a known BD-1 / BD-2 planning failure (no crash, no motion).
- [ ] With `--motion real`: "Inspect …" on a visible object → the wrist camera moves, inspects and returns HOME. *(first motion)*
- [ ] Default scene (`./launch/agent.sh --motion real`): "Put the red cube on the green cube." → COMPLETED with
      placement verification. *(manipulation)*
- [ ] `evaluation/lab_exp1.py` dry protocol, if you want the Experiment 1 harness end to end.
- [ ] Any of your own scripts outside the repo updated for the renamed imports (`MODULE_GUIDE.md` §4).
- [ ] `grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c` → OK.

## 12. Main integration plan (not executed)

`main` is an ancestor of `experiment/lab-exp1`, so the integration cannot conflict. A fast-forward is possible,
but a **no-fast-forward merge** keeps a single merge commit that marks the integration point, as preferred.

```bash
cd ~/Documents/PhD-CE/Research-LF/6DOF-Arm
git status --short                                   # must be empty
git tag main-before-restructure-2026-10-18 main      # backup of the current main (c140d64)
git branch backup/main-2026-10-18 main               # second, branch-shaped backup
git checkout main
git merge --no-ff experiment/lab-exp1 -m "Merge experiment/lab-exp1: laboratory scenario, orchestration, authorization, restructured modular codebase (validated 2026-10-18)"
env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv-seg/bin/python -m pytest      # re-run on main
grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c --quiet && echo FROZEN-OK
```

**Rollback** before anything is shared: `git checkout main && git reset --hard main-before-restructure-2026-10-18`.
This rewrites local `main` only, and only with your approval.

**Merge-conflict risk:** none, since `main` has no commits of its own. The other branches stay as they are
(`feature/active-perception`, `feature/scenario-library`, `refactor/goal-supervisor` are already contained; see
section 10). Delete none without your approval.
