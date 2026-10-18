# Refactor changelog and revised roadmap

This file records every move, rename or removal made by the refactor, with a before → after mapping, the
verification done, and how to undo it.

**Rules for every increment:**

- Behaviour-preserving.
- Frozen files untouched.
- Entry points kept.
- Known defects (BD-1 … BD-8) neither fixed nor hidden.

## Revised objective (2026-10-17)

The architecture review at `8b99000` proposed a `src/vt6l/` package, `pyproject` packaging, protocols and import
contracts (`REFACTOR_ROADMAP.md`, `TARGET_ARCHITECTURE.md`). **That roadmap is superseded by this one.**

The goal now is simpler modular Python:

- consolidate related code;
- remove verified dead code;
- replace monkeypatches with explicit switches;
- make the execution flow readable (`EXECUTION_FLOW.md`, `MODULE_GUIDE.md`).

The debt register (`TECHNICAL_DEBT.md`) and the defect list BD-1 … BD-8 remain valid.

## Increment 1: shared test utilities (done, 2026-10-17)

**Problem.** Test modules imported each other:

- `tests/test_supervisor_goal.py` imported `StubTools` and `geom` from `tests/test_agent.py`;
- four other test files imported `StubTools`, `GoalTools`, `run` and `plan` from those two.

As a result, a change to one test file could break five others, and the fakes were hard to find.

**Change.** The shared fakes and runners moved, unchanged, into `tests/support/`.

| Before | After |
|---|---|
| `tests/test_agent.py::geom` | `tests/support/fakes.py::geom` |
| `tests/test_agent.py::StubTools` | `tests/support/fakes.py::StubTools` |
| `tests/test_supervisor_goal.py::GoalTools` | `tests/support/fakes.py::GoalTools` |
| `tests/test_agent.py::TASK` | `tests/support/runners.py::TASK` |
| `tests/test_agent.py::run` (runs `PickPlaceWorkflow`) | `tests/support/runners.py::run_workflow` |
| `tests/test_supervisor_goal.py::plan` | `tests/support/runners.py::plan` |
| `tests/test_supervisor_goal.py::run` (runs `GoalSupervisor`) | `tests/support/runners.py::run_supervisor` |

**Importers updated.** Each file imports the new names under the old local names (`run_workflow as run`,
`run_supervisor as run`), so the test bodies did not change:

- `tests/test_agent.py`, `test_agent_ask.py`, `test_supervisor_goal.py`, `test_authority.py`,
  `test_orchestration.py`, `test_session.py`.

**Production code changed:** none.

**Verification:**

| Check | Before | After |
|---|---|---|
| Collected test ids (all 20 test files) | 314 | 314, identical list |
| Agent suite (`.venv-seg`, 6 files) | 142 passed | 142 passed |
| System suite (`/usr/bin/python3 -m pytest`) | 172 passed, 6 skipped | 172 passed, 6 skipped |
| Frozen manifest (`sha256sum -c`) | OK | OK |
| `grep -rn "^from tests.test_" tests/` (test modules importing each other) | 7 hits | 0 hits |

**What became simpler.**

- No test module imports another.
- Every fake used by more than one test lives in one folder.
- `run` is now two clearly named functions (`run_workflow`, `run_supervisor`), where before two different functions
  in two files shared the name.

**Undo:** `git revert` of the increment commit.

## Restructuring (2026-10-17)

**Update 2026-10-18:** automated verification passed with no regression, including side-by-side equivalence with
`aa9a6a4`, Isaac Sim and live model inference (`docs/validation/FINAL_MAIN_READINESS.md`). The text below is the
record as written on 2026-10-17, when nothing had been run yet; the risks listed there were checked on
2026-10-18 (readiness report, sections 4–6).

**Instruction.** Code refactoring and restructuring only: no tests, simulations, experiments or validation were
run. The user tests every subsystem before approving new development.

**Static checks that were done instead.** None of them imports or executes project code.

- Every local import in every `.py` file resolves to an existing module and a defined name (an `ast`-based check;
  dynamic `import_module` / `__import__` strings and `-m` module names in `.py` / `.sh` were checked too).
- `pyflakes` (system Python) reports no undefined names and no unused imports in `agent/`, `perception/`,
  `config/` and `scenarios/`.
- Code moved between files was compared as ASTs with the original:
  - `vlm_roles`: 37 / 37 definitions identical, including every prompt string;
  - `tools`: 51 / 51 identical;
  - `capabilities`: 31 / 31 identical;
  - `config`: every constant and class identical.
- `git diff` over the frozen manifest's files and the protected legacy files (`launch/run.sh`,
  `ros_nodes/vlm_agent_node.py`) is empty.

### Commits

| Commit | Step |
|---|---|
| `761ec65` | 1. remove confirmed dead code and unused imports |
| `88abf52` | 2. one `config/` package for configuration |
| `5051e18` | 3. the current perception pipeline moves to `perception/` |
| `aaadb67` | 4. `agent/vlm_roles.py` → package `agent/vlm_roles/` |
| `b5cee7b` | 5. the `Tools` god object → `agent/tools/` |
| `56b94d3` | 6. `agent/capabilities.py` → package `agent/capabilities/` |
| `e9c0aee` | 7. consolidate duplicated helpers in the tools |
| `df04b84` | 8. agent structure: `session.py`, `planning.py`, `task_state.py`, declared state |
| `1b6126d` | 9. explicit execution controls replace the Experiment 1 monkeypatching |
| `36d4814` | 10. scenario library and scenes → `scenarios/` |
| `46f0802` | 11. tidy: plain import in inspection; config index |
| (this commit) | 12. documentation |

**Undo.** Each commit reverts on its own with `git revert <hash>`, newest first. Reverting 3, 4, 5, 6 or 10
alone also needs the later commits that edited the same imports reverted.

### Files moved, split, consolidated or removed

| Kind | Before | After |
|---|---|---|
| moved | `agent/detector.py`, `geometry.py`, `somark.py` | `perception/detector.py`, `geometry.py`, `somark.py` |
| moved + renamed | `agent/world.py` | `perception/world_model.py` |
| moved + renamed | `agent/exploration.py` | `perception/viewpoints.py` |
| renamed | `agent/world_state.py` | `agent/task_state.py` |
| moved | `sim/scenario_lib/{__init__,core,parts,laboratory,build,checks}.py` | `scenarios/` |
| moved | `sim/scenarios/ap_categories.json`, `ap_occlusion.json` | `scenarios/scenes/` |
| split | `agent/tools.py` (661 lines) | `agent/tools/{__init__,held,perception,memory,resolution,manipulation,execution,verification}.py` |
| split | `agent/capabilities.py` (620) | `agent/capabilities/{__init__,specs,perception,inspection,manipulation,planning}.py` |
| split | `agent/vlm_roles.py` (570) | `agent/vlm_roles/{__init__,gemma,prompts,schemas,roles}.py` |
| split | `agent/run.py` | `agent/run.py` (entry) + `agent/session.py` (`Session`) |
| split | `agent/supervisor.py` | `agent/supervisor.py` (state machine) + `agent/planning.py` (plan rules) |
| consolidated | `agent/config.py`, `agent/run.py` model constants | `config/{__init__,perception,agent,models}.py` |
| consolidated | `WorldModel.match` gate + `DescriptionMemory._same_place` | `perception/world_model.py` `same_place()` |
| consolidated | grasp prefix in `plan_pick` and `plan_pick_place` | `ManipulationTools._plan_grasp()` |
| consolidated | ray count in `view_blockage` and `park_for_view` | `ManipulationTools._rays_blocked()` |
| new | — | `agent/controls.py` (`ExecutionControls`), `capabilities.available()`, `LineReader.on_directive` |
| removed | see "Redundant code removed" | — |

The full before → after module mapping is in `MODULE_GUIDE.md` §4, and the final tree in §2.

### Responsibility before → after

| Responsibility | Before | After |
|---|---|---|
| Process entry, model loading | `agent/run.py` (+ `Session`) | `agent/run.py` |
| Session state across instructions | `agent/run.py` `Session` | `agent/session.py` |
| Supervisor state machine and review policy | `agent/supervisor.py` (+ plan rules, replan limit) | `agent/supervisor.py` |
| Plan completion / objective rules | `agent/supervisor.py` | `agent/planning.py` |
| Experiment switches (motion, disabled capabilities, replans) | runtime replacement of production symbols | `agent/controls.py`, passed to constructors |
| Capability registry and menu | `agent/capabilities.py` | `agent/capabilities/specs.py` |
| Perception / inspection / manipulation / planning capabilities | one file | `agent/capabilities/{perception,inspection,manipulation,planning}.py` |
| Held-object and session state of `Tools` | `agent/tools.py` | `agent/tools/held.py` |
| Perceive, describe, world update | `agent/tools.py` | `agent/tools/perception.py` |
| Description memory | `agent/tools.py` | `agent/tools/memory.py` |
| Target resolution, relations | `agent/tools.py` | `agent/tools/resolution.py` |
| Grasp / place planning, parking, HOME | `agent/tools.py` | `agent/tools/manipulation.py` |
| Motion execution | `agent/tools.py` `execute` | `agent/tools/execution.py` (+ controls) |
| Sensor verification | `agent/tools.py` | `agent/tools/verification.py` |
| Gemma adapter / prompts / schemas / roles | `agent/vlm_roles.py` | `agent/vlm_roles/{gemma,prompts,schemas,roles}.py` |
| Detection, depth geometry, marks, world model, viewpoints | `agent/` | `perception/` |
| Configuration | `agent/config.py`, constants in `agent/run.py` | `config/` |
| Scenario library and scenes | `sim/scenario_lib/`, `sim/scenarios/` | `scenarios/` |

### Redundant code removed

| What | Where | Why it was safe |
|---|---|---|
| `run_unimplemented`, `elapsed` | `agent/capabilities.py` | no caller (incl. `getattr` / strings) |
| `Kinematics_tcp()` | `agent/capabilities.py` | a one-line wrapper of `Kinematics()`; calls now use `Kinematics()` |
| `WorldModel.mark_unseen`, `WorldModel.to_dict` | `agent/world.py` | no caller |
| `GemmaVLM._generate_batch_disabled` | `agent/vlm_roles.py` | disabled since 2026-10-14; never called |
| `FakeDetector` | `agent/detector.py` | no caller, including the tests |
| `ObjectEstimate.grasp_text` | `perception/grounding.py` | no caller, including the legacy node |
| the second copy of the same-place gate, the grasp prefix and the ray count | see consolidation | one copy each now |
| `agent/config.py`; `DEFAULT_WEIGHTS` in `agent/run.py` | — | moved into `config/` |
| `__import__("agent.somark", …)` | `_wrist_observation` | a normal import |
| unused imports | `tests/test_supervisor_goal.py`, `tests/test_agent_ask.py`, `evaluation/dual_camera_check.py`, `evaluation/lab_exp1.py` | pyflakes |

**Deliberately kept:**

- **Inactive but benchmarked code:** `Tools.shortlist` and the `describe_many` batch branch. A benchmark script
  and tests still use them.
- **Historical evaluation scripts:** they reproduce reported results.
- **Legacy `perception/` modules:** they are used by frozen or protected files.
- **`evaluation/validate_scenarios.py`'s unused `stage_utils` import:** it is Isaac-side, an import there may have
  side effects, and nothing could be run to check.

### Shared state and configuration changes

| Item | Before | After |
|---|---|---|
| Replanning limit | module global `supervisor.MAX_REPLANS`, reassigned by a harness | `ExecutionControls.max_replans` per supervisor (default 1) |
| Capability availability | `capabilities.SPECS` mutated by a harness | `SPECS` never mutated; each supervisor gets `capabilities.available(disabled)` |
| Disabled capability executors | module functions replaced by a harness | `GoalSupervisor._refuse_disabled` (same "unsupported" outcome and message) |
| Motion gate | `Tools.execute` replaced by a harness, gate in a module global `GATE` | `Tools.controls.motion` (`real` / `dry_run` / `blocked`); the harness keeps its own gate state in a closure |
| Harness directives on stdin | `LineReader.get` replaced | `LineReader.on_directive` |
| Held-object state | `Tools` attributes + `world.holding`, ownership undocumented | unchanged values; ownership and writers documented in `agent/tools/held.py` |
| Session state | `Session` inside `agent/run.py` | `agent/session.py`, with what lives per session vs per instruction |
| Description memory | in `agent/tools.py` | `agent/tools/memory.py`; same-place rule shared with the world model |
| Attributes attached at runtime | `wobs.assoc`, `ctx.vlm_calls` | declared fields `Observation.assoc` (default `{}`), `Context.vlm_calls` (default `lambda: 0`) |
| Agent → simulator import | `agent/config.py` imported `sim.scene_objects.MIC_POSITION` | `config/perception.py` holds the same value `(-0.45, 0.35, 0.0)` |
| Model paths | `agent/run.py` constant + literals in `evaluation/seg_generalization.py` | `config/models.py` |
| Remaining global state | `sim.scene_objects.OBJECTS` (frozen module, filled by `sim/run_scenario.py`); `LineReader._instance` (one stdin reader per process) | unchanged (frozen / by design) |

### Compatibility risks to verify manually

| # | Risk | Why | How to check |
|---|---|---|---|
| 1 | Any script or notebook outside this repository that imports an old path fails | old import paths were not kept as compatibility modules: `agent.detector`, `agent.geometry`, `agent.somark`, `agent.world`, `agent.exploration`, `agent.config`, `agent.world_state`, `sim.scenario_lib` | grep your own scripts; the error is an immediate `ModuleNotFoundError` |
| 2 | Isaac-side imports of `scenarios` and `perception.viewpoints` | resolved by `sys.path` order inside Kit; a Kit extension providing a module with the same name would shadow ours (none found in `.venv` site-packages) | `.venv/bin/python -m sim.run_scenario --scenario laboratory --headless`; `evaluation/validate_scenarios.py --no-design` |
| 3 | `python -m scenarios.checks` subprocess of the validator | new module path; it relies on `cwd` = repository root (unchanged) | `validate_scenarios.py` without `--no-design` |
| 4 | Scenario texture path | `scenarios/build.py` `ROOT` changed from `parents[2]` to `parents[1]`; textures must still land in `out/scenario_textures` | first scenario build; check that the label textures appear |
| 5 | Tests that replace `capabilities.run_*` | they now patch the package; calls made between capability modules (inspect → return_home, inspect → locate) no longer see such a patch. No test patches those today | agent suite: `test_supervisor_goal.py`, `test_authority.py`, `test_orchestration.py` |
| 6 | `Tools` composed from parts | attribute lookup goes through the parts; each method exists once, but a subclass or fake that relied on the old class layout could differ | agent suite (`test_agent.py`, `test_session.py`) |
| 7 | Experiment 1 harness | the controls must reproduce the old gates exactly: blocked until "@motion on", five disabled capabilities, no replanning, the same `harness_gates.jsonl` records. One known difference: the `execute_refused` event's reason now reads "motion gate closed (Experiment 1: no motion in this phase)" | an offline dry run of the harness; a mocked test of `ExecutionControls` (none exists yet) |
| 8 | `dry_run` mode | new, never exercised: the motion step fails by design | a mocked supervisor run with `ExecutionControls(motion="dry_run")` |
| 9 | Supervisor with default controls | the planner menu now comes from `available(())`, which should give identical text and order | `test_planner_menu_lists_only_registry_capabilities` |
| 10 | The source scan for object words | now scans `agent/**`, the moved perception modules and `config/` (previously `agent/*.py` only). A static replica of the scan found no hits | `test_no_object_knowledge_or_ground_truth_in_agent_code` |
| 11 | The viewer and system-Python tools | `agent/viewer.py` imports only `agent.events` (unchanged) | `tests/test_demo_viewer.py`; open the viewer once |
| 12 | Docs outside `docs/architecture/` and `docs/scenarios/` (DEMO.md, historical reports, `docs/perception_redesign/`) | they cite the old paths as historical records and were not rewritten | — |

**Suggested order for manual testing:**

1. Both offline suites (DEVELOPER_GUIDE §1).
2. The supervisor with fakes, and `ExecutionControls` in each mode (§2).
3. Offline perception on the saved frames (§5).
4. Scenario load and validation in Isaac (§3).
5. Cameras and ROS (§4).
6. A live observation-only instruction.
7. Only then, motion.

## Roadmap status

| # | Increment | Status |
|---|---|---|
| 1 | shared test utilities | done (`aa9a6a4`), tested |
| 2 | dead code removal | done (`761ec65`), **not tested** |
| 3 | explicit switches instead of monkeypatching | done (`1b6126d`), **not tested** |
| 4 | split `agent/tools.py` | done (`b5cee7b`, `e9c0aee`), **not tested** |
| 5 | split `agent/vlm_roles.py` | done (`aaadb67`; prompt ASTs identical), **not tested** |
| 6 | configuration in `config/` | done (`88abf52`), **not tested** |
| 7 | scenario checks independent of the agent | done (`5051e18`: viewpoints moved to `perception/`), **not tested** |
| 8 | one held-object record | **not done**: `held_object` and `world.holding` are still two fields. Ownership is documented; merging them changes code paths (`return_home` reads both) and needs tests first |
| 9 | consistent failure records | not done |
| 10 | supervisor review policy as a pure function | not done (`_reviewing` is unchanged) |
| later | legacy containment (`legacy/` folder) | not done: frozen and protected files import the shared legacy modules |

Functional fixes (BD-1 … BD-8) are untouched and still need separate approval.
