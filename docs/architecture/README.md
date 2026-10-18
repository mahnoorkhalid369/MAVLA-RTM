# Architecture documents (current)

The top-level `README.md` is the entry point. These documents go deeper:

| Document | Contents |
|---|---|
| [MODULE_GUIDE.md](MODULE_GUIDE.md) | where to find X; the directory structure; before → after module mapping |
| [EXECUTION_FLOW.md](EXECUTION_FLOW.md) | diagrams of the instruction path, and three traced examples with current failures marked |
| [FUNCTIONALITY_MAP.md](FUNCTIONALITY_MAP.md) | every implemented function: file, callers, inputs / outputs, state, status, how to test |
| [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) | known baseline defects BD-1 … BD-8 (still open), and the debt register as of the review |
| [REFACTOR_CHANGELOG.md](REFACTOR_CHANGELOG.md) | every move, split, consolidation and removal of the 2026-10-17 restructuring |
| [../DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) | commands to run and test each subsystem, with the Python environment for each |
| [../validation/FINAL_MAIN_READINESS.md](../validation/FINAL_MAIN_READINESS.md) | verification record, coverage matrix, manual acceptance checklist |

**Historical documents** are in `../history/architecture/` and describe the code before the restructuring:

- the 2026-10-17 architecture review: `REVIEW_SUMMARY_2026-10-17.md` (its index and executive summary), current
  architecture, dependency analysis, target architecture, roadmap, engineering standards;
- earlier phase documents: `ARCHITECTURE.md`, `ACTIVE_PERCEPTION*.md`, `AUDIT.md`, `ORCHESTRATION_AUDIT.md`.

The review's packaging proposals were superseded by the simple modular layout described in `MODULE_GUIDE.md`.
