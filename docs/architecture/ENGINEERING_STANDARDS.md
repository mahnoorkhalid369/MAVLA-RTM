# Engineering standards (proposal for review), 2026-10-17

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

These apply to **new and refactored code**. The frozen baseline files (see the manifest) and the protected legacy
files (`launch/run.sh`, `ros_nodes/vlm_agent_node.py`) are excluded from formatters, linters and edits.

## 1. Packaging and imports

**Packaging.**

- One installable project (`pyproject.toml`), package `vt6l`, installed editable: `pip install -e .[agent,dev]`.
- Optional dependency groups:

  | Group | Contents |
  |---|---|
  | `agent` | torch, transformers, bitsandbytes, accelerate, ultralytics, langgraph, langgraph-checkpoint-sqlite, pydantic, opencv |
  | `sim` | isaacsim; used in its own venv |
  | `legacy` | dependencies of the legacy stack |
  | `dev` | pytest, ruff, mypy, import-linter, hypothesis |

**Imports.**

- Absolute imports only (`from vt6l.world.model import WorldModel`). No `sys.path` manipulation outside the frozen
  files.
- Heavy or optional dependencies (`isaacsim`, `omni.*`, `rclpy`, `torch`, `ultralytics`, `transformers`) are
  imported only in adapter modules, inside functions or behind `TYPE_CHECKING`.
- Layer contracts (TARGET_ARCHITECTURE §2) are written as `importlinter` contracts and checked in CI.
- No import-time side effects: entry points use `def main() -> int` + `if __name__ == "__main__": raise SystemExit(main())`.

## 2. Naming

| What | Convention |
|---|---|
| Modules | short nouns, `snake_case` (`viewpoints.py`, `ledger.py`) |
| Classes | nouns (`WorldModel`, `GatedExecutor`) |
| Protocols | named by role, without an `I` prefix (`MotionExecutor`, `SensorSource`) |
| Functions | verbs (`measure`, `resolve_target`) |
| Predicates | `is_` / `has_` / `can_` |
| Units | in names when not SI-obvious: `_m`, `_mm`, `_deg`, `_rad`, `_s`; angles stored in radians internally, degrees only at I/O |
| Constants | module-level `UPPER_CASE` only if truly constant; tunables belong in `PlatformConfig` |
| Capability names, outcomes, event kinds | stable strings (they are the experiment record format); changed only with a migration note |

## 3. Types and data models

- Type annotations on all public functions. `mypy --strict` on `vt6l.core`, `vt6l.world`, `vt6l.safety`, `vt6l.io`;
  basic mode elsewhere.
- **Data models:**

  | Use | Model |
  |---|---|
  | Internal value objects | `@dataclass(frozen=True, slots=True)` |
  | Anything parsed from a model or a file | pydantic `BaseModel` (LLM output, config files, JSON artifacts) |
  | Contracts at boundaries | `typing.Protocol` |

- `StrEnum` for closed sets of states (outcomes, statuses, effects, failure classes). Never compare free strings.
- Ids are typed (`ObjectId = NewType("ObjectId", int)`); marks and world ids are never interchangeable.

## 4. Configuration

- One `PlatformConfig` (frozen dataclasses). Loading order: defaults → `config/platform.default.toml` →
  experiment TOML → CLI overrides.
- **The resolved config is written to every run directory**, and its hash is recorded in `result.json`.
- No configuration read from environment variables except deployment facts (`ROS_DOMAIN_ID`, CUDA device), and
  those only in `vt6l.apps`.
- Model paths come from `config/models.toml`. Weights live under `models/` (not `out/`), with a `MODELS.md` table:
  source, licence, SHA-256, training run.

## 5. Logging and exceptions

**Diagnostics.**

- Stdlib `logging`, one logger per module (`logging.getLogger(__name__)`), JSON formatter in apps.
- Levels: `DEBUG` internals, `INFO` lifecycle, `WARNING` degraded behaviour, `ERROR` failed operation.
- `print` only in `vt6l.apps`, `experiments/` and `scripts/`.

**Experiment record.**

- `EventSink` (JSONL, the current `events.jsonl` format), owned by the run.
- Event kinds are documented in `docs/architecture/EVENTS.md`, to be generated from a registry.
- The viewer and the scorers depend on it: version it (`"schema": 1`).

**Exceptions.**

- The `VT6LError` hierarchy (TARGET_ARCHITECTURE §6).
- Catch broadly only at defined boundaries: capability dispatch, apps `main`.
- Always keep the type and message in the resulting `StepResult` / log entry.
- Never convert an exception into success.

## 6. Tests

| Directory | Contents | Environment | Marker |
|---|---|---|---|
| `tests/unit/` | pure functions, classes with fakes | CPU, plain interpreter | `unit` (default) |
| `tests/integration/` | supervisor + capabilities + fakes; golden event sequences (G1); perception on saved frames (G3) | CPU | `integration` |
| `tests/model/` | recorded-model fixtures (always) + live Gemma runs (opt-in) | GPU for live | `model`, `gpu` |
| `tests/sim/` | Isaac smoke tests (scene build, scenario validation subset) | Isaac venv | `sim`, `slow` |
| `tests/characterization/` | known defects BD-*, `xfail(strict=True)` | CPU | `characterization` |

**Fixtures and fakes.**

- Shared fixtures in `tests/conftest.py`; fakes in `tests/fakes/`, each checked against its `Protocol`.
- Tests never import other test modules.

**Determinism.**

- Seeds fixed and logged; Gemma greedy.
- Golden files under `tests/data/` with a regeneration script and a reviewable diff.

## 7. Documentation

| Where | What |
|---|---|
| Module docstring | responsibility (one paragraph), owned state, side effects |
| Public class / function | contract: inputs, outputs, units, errors |
| `docs/architecture/` | the living architecture (these six files), updated with every R increment; decisions recorded as short ADRs (`docs/architecture/adr/NNNN-title.md`) |
| `experiments/<name>/README.md` | hypothesis, protocol, config, how to run, where outputs go |
| Each report | separates measured results, hypotheses and limitations (as current reports do) |

## 8. Experiment artifacts and model weights

**Run directories.**

```
out/<experiment>/<run-id>/
    config.resolved.toml
    events.jsonl
    result.json
    images/
    scoring_only/          # ground truth, never read by the agent
```

The run id is UTC timestamp + git short hash + `-dirty` flag.

**Ownership.** Artifacts are append-only. Corrections are new files (as with `score_corrected.json` and the
erratum), never overwrites.

**Weights.**

- They live under `models/<name>/<version>/`, registered in `config/models.toml` with a SHA-256.
- Training outputs stay in `out/seg/…`; promotion to `models/` is an explicit copy with a registry entry.

## 9. Dependency management and environments

| Environment | Purpose | Defined by |
|---|---|---|
| `sim` venv | Isaac Sim 6.1 (+ `isaacsim` wheels) | `requirements/sim.lock` |
| `agent` venv | agent, YOLO, Gemma, tests | `requirements/agent.lock` (= `pip install -e .[agent,dev]` + pins) |
| `legacy` venv | legacy stack only | `requirements/legacy.lock` |

- System ROS 2 Jazzy is sourced per shell; `ROS_DOMAIN_ID=42` for tests.
- One `tools/bootstrap.sh` creates the venvs from the lock files.
- The default `pytest` interpreter is the agent venv, which ends silent skips.

## 10. CI checks (hardware-aware)

| Tier | Runs on | Checks |
|---|---|---|
| T0 (every commit, CPU runner) | any machine | `ruff` (lint + format check, frozen paths excluded); `mypy` on strict packages; `import-linter` contracts; frozen-manifest `sha256sum -c`; no production-symbol monkeypatching in `evaluation/` / `experiments/` (grep); `pytest -m "unit or integration or characterization"` |
| T1 (merge to main, CPU) | same | T0 + golden suites G1–G3 + an import of every module under the plain interpreter (no Isaac / rclpy / torch at import) |
| T2 (nightly / pre-release, GPU workstation) | lab machine | `pytest -m model --gpu` (live Gemma prompt regressions, planner A/B), segmentation benchmark on the 27 saved frames |
| T3 (manual, Isaac) | lab machine, one Isaac instance | `pytest -m sim` smoke + `validate_scenarios --no-design`; never two Isaac instances; never while training |

T2 and T3 results are committed as small JSON summaries (`out/ci/<date>/…`), so regressions are visible in review
even though the hardware jobs aren't on hosted CI.
