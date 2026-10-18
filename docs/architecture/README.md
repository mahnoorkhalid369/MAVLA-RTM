# Architecture review package (2026-10-17, audited at `experiment/lab-exp1` @ `f094d2f`)

| Document | Contents |
|---|---|
| [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) | subsystems as they are: responsibilities, dependencies, consumers, state, problems |
| [DEPENDENCY_ANALYSIS.md](DEPENDENCY_ANALYSIS.md) | the import graph (from `ast`), boundary violations, side effects, duplication, state ownership |
| [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) | debt register TD-1 … TD-13, and the known baseline defects BD-1 … BD-8 (not to be fixed during restructuring) |
| [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md) | proposed layers, package layout, interfaces (each tied to a debt item), state ownership, config, errors |
| [REFACTOR_ROADMAP.md](REFACTOR_ROADMAP.md) | increments R0 … R10, functional fixes F1 … F5, acceptance criteria, rollback, risks |
| [ENGINEERING_STANDARDS.md](ENGINEERING_STANDARDS.md) | packaging, naming, types, config, logging, tests, docs, artifacts, environments, CI tiers |

**Start here (2026-10-17, revised objective: simple modular Python, no packaging).** The code was restructured
on 2026-10-17 (commits `761ec65` … `46f0802`); automated verification passed on 2026-10-18 and manual
acceptance is pending (`../validation/FINAL_MAIN_READINESS.md`). REFACTOR_CHANGELOG.md lists the changes.

| Document | Contents |
|---|---|
| [MODULE_GUIDE.md](MODULE_GUIDE.md) | where to find X; the final directory structure; before → after module mapping |
| [EXECUTION_FLOW.md](EXECUTION_FLOW.md) | diagrams of the instruction path, and three traced examples with current failures marked |
| [FUNCTIONALITY_MAP.md](FUNCTIONALITY_MAP.md) | every implemented function: file, callers, inputs/outputs, state, status, how to test |
| [REFACTOR_CHANGELOG.md](REFACTOR_CHANGELOG.md) | every move, split, consolidation and removal; state / config changes; compatibility risks; roadmap status (supersedes REFACTOR_ROADMAP.md) |
| [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) | commands to run and test each subsystem, with the Python environment for each |

The review documents below describe the code as it was at `f094d2f`, before the restructuring. Their file paths are
historical; MODULE_GUIDE.md §4 maps them to the current files. They remain the reference for problems and debt. Their packaging proposals (`src/vt6l/`,
`pyproject`, protocols) are superseded by REFACTOR_CHANGELOG.md.

Earlier, phase-specific documents in this folder (`ARCHITECTURE.md`, `ACTIVE_PERCEPTION*.md`,
`ORCHESTRATION_AUDIT.md`, `AUDIT.md`) remain as historical records.

## Executive summary

The codebase is acyclic and has a sound, frozen manipulation core (`control/`, `execution/`, `safety/`). The current
agent (`agent/`, 5.3 k lines) grew by milestones. Its problems are ownership, injection and layering, not algorithms.

### The five highest-priority problems

1. **Safety-relevant behaviour can only be changed by monkeypatching (TD-4).**
   - Dry runs, motion gates, disabled capabilities and replanning limits in the experiment harnesses all replace
     production symbols at runtime: `Tools.execute`, `capabilities.run_*`, `capabilities.SPECS`,
     `supervisor.MAX_REPLANS`, `LineReader.get`, `EventLog.__call__`.
   - Any rename during refactoring would silently disable a gate.
2. **`Tools` is a god object holding the agent's mutable session state (TD-1).**
   - ~40 methods across perception, planning, execution, verification and state; no interface.
   - The held object is stored twice (`Tools.held_object` and `world.holding`).
   - Test stubs drift with every change.
3. **World and memory state has three representations with inconsistent keys (TD-2).**
   - `WorldModel`, `TaskState` and `DescriptionMemory` (plus the legacy `WorldMemory`): world id vs mark vs geometry
     keys.
   - This already produced a scoring error in Experiment 1, and duplicate world ids overwrite descriptions.
4. **Layering violations and simulator-bound configuration (TD-5).**
   - The agent's perception config imports the simulator's frozen test-field module.
   - The camera rig lives in the robot-constants module.
   - The scenario library imports the agent.
   - Scene selection works by mutating a frozen module's global.
5. **No packaging or environment definition (TD-9, with TD-10).**
   - No `pyproject` or lock files; four interpreters; 32 `sys.path` hacks.
   - Agent tests skipped by the default `pytest`.
   - Production YOLO weights under the ignored `out/`.

### Recommended first increment

**R0, then R2** (REFACTOR_ROADMAP §2, §4):

1. **R0, a safety net, with no production change:**
   - `pyproject.toml` + lock files;
   - one test interpreter + markers;
   - golden characterization tests (supervisor event sequences, prompt hashes, perception geometry on the 27 saved
     frames);
   - `xfail(strict=True)` tests pinning the known defects.
2. **R2, injection points:** `MotionExecutor` (real / dry-run / gated), a config-driven `CapabilityRegistry`,
   `SupervisorPolicy`, `InputSource` and `EventSink` injection. The harnesses are migrated off monkeypatching.

Together these remove the largest operational risk and make every later increment mechanically checkable.

### Constraints respected

- Nothing in production code was changed in this stage.
- The frozen manipulation baseline, the scenario baseline `lab-env-baseline-v2.0`, all experiment reports and git
  history are preserved.
- Known functional defects (planner regression, negated-command authorization, segmentation limits) are recorded
  as BD-1 … BD-8 and kept separate from the restructuring.
