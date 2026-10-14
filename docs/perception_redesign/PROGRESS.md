# Perception redesign — progress (read this first)

Goal: perception and grounding with no hardcoded object knowledge (see the brief in DECISIONS.md and
the inventory in HARDCODING_INVENTORY.md). Evidence labels: measured / from files / unverified.

## Active phase
**Phase 1 — synthetic training data** (in progress)

## Phases
| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Setup: PROGRESS / DECISIONS / CLAUDE.md, hardcoding inventory, git | done | this folder, `CLAUDE.md`, commit 8df6a38 |
| 1 | Data: semantic labels (new file), Replicator generator, pilot, full run | in progress: code done + tested offline; pilot waiting for the GPU | `datagen/`, `tests/test_datagen.py` (21 passed) |
| 2 | Train class-agnostic instance segmentation (sim stopped) | not started | — |
| 3 | Detection + geometry + TF-based exclusion module, unit tests | not started | — |
| 4 | VLM description + world memory + instruction grounding, tests with recorded outputs | not started | — |
| 5 | Mass estimation + verification checks | not started | — |
| 6 | Evaluation on held-out layouts (eval layouts vs novel objects, separately) | not started | — |
| 7 | Pick and place with the new pipeline vs frozen Baseline A | not started | — |

## Done
- 2026-10-14: baseline manifest verified (all 21 files OK, measured); record files created;
  hardcoding inventory written (HARDCODING_INVENTORY.md, from files).
- 2026-10-14: git initialised; first commit 8df6a38 "Baseline before perception redesign" (115 files,
  10.9 MB, none > 50 MB, no keys/tokens found; .venv*, models/, out/, data/, logs and the external
  epson-robot-ros2 clone excluded; baseline folder + manifest included). Measured.
- 2026-10-14: assets downloaded from NVIDIA's official Isaac Sim 6.1 asset server only
  (`datagen/fetch_assets.py`): 21 YCB objects + materials and 7 shape primitives, 49 files, 218 MB,
  manifest `data/assets/isaac_6.1/ASSETS_MANIFEST.json` with URL / size / ETag / licence note
  (licences UNVERIFIED: no licence file on the server for these folders). Measured.
- 2026-10-14: Phase 1 code (new files only): `datagen/textures.py` (procedural textures),
  `datagen/layout.py` (seeded scene sampler: families, sizes, resting faces, stacks, held object,
  robot pose collision-checked against the objects, table / wrist views, lights, floor, RGB effects,
  evaluation-object exclusion), `datagen/coco.py` (COCO from instance ids, YOLO-seg converter),
  `datagen/generate.py` (Isaac Sim renderer), `datagen/eval_scene_capture.py` (runtime semantic
  labels + GT masks for the evaluation scene), `datagen/config_v1.json` (all ranges + seed 20261014).
  Offline: 21 new tests pass, all 125 tests pass; sampler 32 ms/scene, 14.3 objects/image, 12.5 %
  stacked, 7.5 % held, 20 % wrist views over 200 samples (measured).

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
