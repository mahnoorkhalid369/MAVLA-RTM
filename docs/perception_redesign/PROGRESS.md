# Perception redesign — progress (read this first)

Goal: perception and grounding with no hardcoded object knowledge (see the brief in DECISIONS.md and
the inventory in HARDCODING_INVENTORY.md). Evidence labels: measured / from files / unverified.

## Active phase
**Phase 1 — synthetic training data** (in progress)

## Phases
| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Setup: PROGRESS / DECISIONS / CLAUDE.md, hardcoding inventory, git | in progress | this folder, `CLAUDE.md` |
| 1 | Data: semantic labels (new file), Replicator generator, pilot, full run | in progress | — |
| 2 | Train class-agnostic instance segmentation (sim stopped) | not started | — |
| 3 | Detection + geometry + TF-based exclusion module, unit tests | not started | — |
| 4 | VLM description + world memory + instruction grounding, tests with recorded outputs | not started | — |
| 5 | Mass estimation + verification checks | not started | — |
| 6 | Evaluation on held-out layouts (eval layouts vs novel objects, separately) | not started | — |
| 7 | Pick and place with the new pipeline vs frozen Baseline A | not started | — |

## Done
- 2026-10-14: baseline manifest verified (all 21 files OK, measured); record files created;
  hardcoding inventory written (HARDCODING_INVENTORY.md, from files).

## Next
- git init (report repo size, exclude > 50 MB, scan for secrets, include the baseline folder).
- Phase 1 generator; ask the user before stopping their running simulation for the pilot.

## Open questions
- None.

## Blockers
- The user's own simulation session (sim + agent, ~5.3 GB GPU) is running: the pilot needs it
  stopped (ask first).

## Constraints (agreed)
- Do not modify files in docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256.
- New code in new files; all existing offline tests must keep passing.
- GPU 12 GB: one Isaac instance; VLM before heavy rendering; no training while the sim runs.
- Ground truth only for evaluation, never in the pipeline.
