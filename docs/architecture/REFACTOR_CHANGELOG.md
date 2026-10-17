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

## Revised roadmap (increments 2 onward)

**Rules for each increment:**

- One concern per commit.
- The same verification as increment 1: identical test ids, both suites green, manifest OK.
- Golden checks where the increment touches runtime code: supervisor event sequences on the mocked tests, a
  prompt-text hash for prompt moves.
- **No live robot motion** at any point.

| # | Priority (user's order) | Increment | What changes | Check that behaviour is preserved |
|---|---|---|---|---|
| 2 | 7: dead code | Remove confirmed dead code | delete the functions listed in `FUNCTIONALITY_MAP.md` §J: `capabilities.run_unimplemented`, `WorldModel.mark_unseen`, `GemmaVLM._generate_batch_disabled`, `detector.FakeDetector`, `grounding.ObjectEstimate.grasp_text` (only if no frozen file uses it). Re-check each with a reference scan first (including `getattr` and string dispatch) | the reference scan comes back empty; suites green |
| 3 | 5: monkeypatches | Explicit switches instead of monkeypatching | add plain constructor arguments: `Tools(..., motion="real" / "dry_run" / "blocked")`, `GoalSupervisor(..., disabled=set(), max_replans=1)`, an input source for `ask`; migrate `evaluation/lab_exp1_agent.py` (today the only harness that patches production symbols, lines 38–78: `supervisor.MAX_REPLANS`, `capabilities.SPECS`, `capabilities.run_*`, `Tools.execute`, `ask.LineReader.get`) to those arguments | new tests prove a blocked `Tools` never calls the executor; harness dry runs produce the same event sequences as before |
| 4 | 1–2: ownership, large modules | Split `agent/tools.py` (661 lines) | into `agent/tools/` modules: description memory, perception, resolution, planning, verification. `Tools` stays as the one class callers use | `from agent.tools import Tools` unchanged; suites green |
| 5 | 2: large modules | Split `agent/vlm_roles.py` (570 lines) | prompts and schemas apart from the Gemma adapter | the hash of every prompt string is unchanged |
| 6 | 6: config | Consolidate configuration | a `config/` folder of plain dataclass modules (camera rig, tolerances, budgets, model paths); remove `agent/config.py` → `sim/scene_objects.py` | values compared one by one before / after |
| 7 | 4: cross-module dependencies | Scenario checks independent of the agent | `sim/scenario_lib/checks.py` gets the viewpoint check passed in (or moved to evaluation) | `validate_scenarios --no-design` result JSON identical (Isaac, separate session) |
| 8 | 3: duplicates | One held-object record | `Tools.held_object` and `world.holding` become one owner, the other a read-only view | mocked held / place / revalidate tests |
| 9 | 6: errors | Consistent failure records | one small `failures.py` listing the failure classes and outcomes already used (strings unchanged) | outcome strings identical in the golden event sequences |
| 10 | 8: testability | Supervisor policy readable alone | move `_reviewing`'s decision table into a pure function with its own tests | event sequences identical |
| later | 4 / 7 | Legacy containment | move the legacy VLM stack into `legacy/` (except protected files and the shared `vlm/model.py` and `perception/grounding.py`) | legacy tests still pass |

**Functional fixes are separate and need approval:**

- BD-1: planner prompt regression;
- BD-2: location validation;
- BD-3: negated commands;
- BD-4 / BD-5: segmentation;
- BD-6 / BD-7: label reading, `reported()`.

Each is its own commit, with its own live or offline evidence, after the structural increments it depends on.
