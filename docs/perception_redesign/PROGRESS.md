# Perception redesign — progress (read this first)

Goal: perception and grounding with no hardcoded object knowledge (see the brief in DECISIONS.md and
the inventory in HARDCODING_INVENTORY.md). Evidence labels: measured / from files / unverified.

## Active phase
**Phase 1 — synthetic training data** (full run in progress) + architecture **revision 3 approved** (`ARCHITECTURE.md`; revs 1-2 kept). Then Phase 2 (stop after the comparison)

## Phases (the merged plan is ARCHITECTURE.md §12; this table tracks the data / perception track)
| # | Phase | Status | Evidence |
|---|---|---|---|
| 0 | Setup: PROGRESS / DECISIONS / CLAUDE.md, hardcoding inventory, git | done | this folder, `CLAUDE.md`, commit 8df6a38 |
| 1 | Data: semantic labels (new file), Replicator generator, pilot, full run | in progress: pilot done and verified; full run (7500 images) running | `datagen/`, `tests/test_datagen.py` (21 passed), `data/perception/synthetic_v1_pilot/STATS.json` |
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

- 2026-10-14: user's session stopped with permission (SIGINT; GPU free).
- 2026-10-14: evaluation scene labelled at runtime (`datagen/eval_scene_capture.py`): 20 of 21
  objects visible with masks from the table camera (black_cube hidden by the arm), masks checked by
  overlay (measured). Output `data/perception/eval_layouts/today/`.
- 2026-10-14: pilot, 200 images (160 train / 20 val / 20 test_novel). Three generator bugs found and
  fixed (measured, then re-checked): (1) the RGB kept the first image's floor material and robot
  pose when prim paths were re-used -> fresh paths per image; (2) the annotator's id -> prim path map
  is empty for prims created after the first frame -> object uid carried as a second semantic label;
  (3) asynchronous material / texture loading -> synchronous loading settings. Lighting ranges lowered
  (images were washed out). After the fixes (measured): one extra render of the same scene changes
  0 mask pixels in 200/200 images and the RGB by a median of 5 px (2 images > 1000 px, max 30,290 px
  = 3.3 %, masks unchanged); robot pose error <= 0.036 deg; objects with a mask 94.7 / 93.5 / 97.3 %
  (train / val / novel; the rest occluded, out of frame or < 30 px); masks per image 3-25; 0
  area/mask mismatches (`datagen/verify_dataset.py`); 0.86 s/image incl. the check render; peak GPU
  2.8 GB; 0.35 MB/image.

- 2026-10-14: read-only audit of the current agent (reported to the user): the agent is one ROS node
  using its own execution path (no collision checker, no grasp yaw, no settle, no verification), no
  spatial relations; last live tasks 7 Oct (from files: out/tasks/).
- 2026-10-14: direction change agreed with the user: LangGraph multi-agent system, no classical
  detector, YOLO-seg + VLM, sensor-only verification. Design written: `ARCHITECTURE.md` (12 sections +
  open decisions); decisions logged in DECISIONS.md. Not implemented.
- Full run status at 07:57 UTC (measured): train 0-3999 done (4 chunks; 0.89 / 0.91 / 1.22 / ~1.5
  s per image — slowing, cause unknown), train 4000-5999, val, test_novel pending.

- 2026-10-14: architecture rev 1 approved with changes; rev 2 written (wrong-object confirmation +
  metric, held-object exclusion, trials / Wilson intervals / thresholds, mass as plausibility check,
  whole-scene re-perception, visibility reasoning, CAD limitation); decisions logged.

- 2026-10-14: architecture rev 3 (correlated-error joint failure metric, relational references =
  attributes from crops + relation from geometry, held-object box from measured geometry); approved.
- 2026-10-14: rendering slowdown investigated (measured, read-only): GPU at 7 % utilisation, 54 C, no
  throttling; generator ~200 % CPU; CPU governor powersave / profile "balanced", 69 C; slowdown began
  mid-chunk train_2000 (first 200 images 0.86 s) and persisted in new processes (1.44-1.48 s); GNOME
  tracker-extract indexing new files. Conclusion: CPU-bound; likely reduced sustained CPU clocks
  and indexer competition (unverified); not changed.
- 2026-10-14: evaluation layouts prepared: `datagen/eval_layouts.py` (20 re-arranged layouts, seeds
  900000+i, 38 stacked objects over 20 layouts, offline-tested), `eval_scene_capture.py --layout i`
  (+ synchronous loading and double render).
- 2026-10-14: Phase 2 prepared (no GPU used): `.venv-seg` (torch 2.11+cu130, torchvision 0.26,
  ultralytics 8.4.177, pycocotools; Ultralytics analytics sync OFF, runs/weights into git-ignored
  out/seg, models/ultralytics); `seg/data.py`, `seg/train_maskrcnn.py` (torchvision Mask R-CNN
  R50-FPN v2, BSD-3), `seg/train_yolo.py` (YOLO11s-seg, AGPL-3.0), `seg/evaluate.py` (same metrics for
  both, val and test_novel separately); loader and IoU checked on the pilot.

## Next
1. Generator finishes -> `datagen/verify_dataset.py data/perception/synthetic_v1`; report stats,
   sample overlays, slowdown cause.
2. Capture eval layouts: `eval_scene_capture.py --name today`, `--layout 0..19`; ambiguity sets
   (needs `generate.py` scene builders refactored into an importable module AFTER the generator
   finishes: it re-imports generate.py per chunk).
3. Phase 2: train Mask R-CNN and YOLO11s-seg (2.5 h each, sim stopped), evaluate, licence proposal. STOP.

## Open questions
- None.

## Blockers
- None.

## Constraints (agreed)
- Do not modify files in docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256.
- New code in new files; all existing offline tests must keep passing.
- GPU 12 GB: one Isaac instance; VLM before heavy rendering; no training while the sim runs.
- Ground truth only for evaluation, never in the pipeline.
