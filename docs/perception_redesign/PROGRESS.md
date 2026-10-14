# Perception redesign — progress (read this first)

Goal: perception and grounding with no hardcoded object knowledge (see the brief in DECISIONS.md and
the inventory in HARDCODING_INVENTORY.md). Evidence labels: measured / from files / unverified.

## Active phase
**AUTONOMOUS DEMO RUN 2026-10-14T08:19:52Z – ~12:50Z: finished (no hard stop). Result: demo NOT READY
as specified (DEMO_REPORT.md): instructions 1 and 3 verified 3/3 each, instruction 2 asks 3/3,
instruction 4 not run. Next: reference-resolution evaluation (DEMO_REPORT §4), 20 eval layouts +
ambiguity sets, Phase 2 Mask R-CNN comparison.** (Earlier header follows.) Brief: user message of
2026-10-14 (demo thin slice of ARCHITECTURE rev 3; rules, budgets, hard stops). Steps: 1 dataset ->
2 fine-tune YOLO11s-seg (<= 3 rounds, <= 4 h) -> 3 demo slice (agent/) -> 4 rehearsal (3 runs x
instructions) -> 5 DEMO.md, DEMO_REPORT.md, out/demo/<date>/.
**Phase 1 — synthetic training data** (full run in progress) + architecture **revision 3 approved** (`ARCHITECTURE.md`; revs 1-2 kept). Then Phase 2 (stop after the comparison)

## Autonomous run log (UTC, measured unless marked)
- 08:49 dataset generation finished: 6,000 train / 1,000 val / 500 test_novel; verify_dataset: no
  problems, 0 area mismatches, objects with a mask 94.4 / 95.6 / 95.0 %, 2.82 GB, peak GPU 2.9 GB.
  YOLO labels fixed to one polygon per instance (DECISIONS). Rendering slowdown (0.89 -> 1.5 s/img
  mid train_2000, later back to ~0.9 s): CPU-bound (GPU 7 % util); cause unverified (likely CPU clock).
- 08:50 today's scene recaptured (20/21 objects, 1 image).
- 08:51-08:54 two agent smoke tests with the sim (pretrained detector): plumbing works end to end;
  4 bugs fixed (DECISIONS). Commit after this entry.
- ~08:58 YOLO round 1 started (1.75 h budget), sim stopped.
- 10:40 YOLO round 1 done: 49 epochs in 1.75 h, 139 s epoch 1, 133 s/epoch after (median). Metrics
  (`out/seg/eval/round1/metrics.json`, conf 0.25, mask IoU >= 0.5, greedy one-to-one): today's scene
  recall box 16/16, cylinder 3/3, sphere 1/1, precision 0.952 (20/21) -> **pass criteria met**
  (one image only: the Wilson 95 % lower bound for 16/16 is 0.81); val recall 0.976 / precision
  0.904; test_novel recall 0.965 / precision 0.893 (ycb_novel 0.939 lowest). No round 2/3 needed.
- 10:45 rehearsal cycle 1 started (instructions 1-3 x 3 runs, fresh sim per run).
- 10:45-12:29 rehearsal: 4 cycles, 3 fix cycles (DECISIONS). Final cycle 4 (measured, ground truth
  scored after each run): i1 3/3, i3 3/3 (2.0-3.9 mm), i2 0/3 asked (yellow look-alikes), 0 false
  done, 0 false failed; instruction 4 not run (gate). Cycle 3 had 2 false done (wrong object) before
  fix 3.
- 12:30-12:40 demo paths tested: interactive headless (COMPLETED 4.0 mm) and launch/start_demo.sh
  with GUI (COMPLETED 1.7 mm, GPU peak 7.1 GB); launcher cleanup fixed (killed its caller's group).
- ~12:45 deliverables: DEMO.md, DEMO_REPORT.md (**demo NOT READY as specified**), out/demo/2026-10-14/.

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

- 2026-10-14 08:17Z: pretrained YOLO11s-seg recall measured (CPU, 100 images): boxes 2 %, all 10 % at
  conf 0.25 (24 % at 0.05); novel 7 % — not usable (`out/seg/pretrained_recall.json`).
- 2026-10-14 08:25Z: .venv-seg extended with Gemma deps + LangGraph (versions in DECISIONS.md); rclpy OK.

## Next (autonomous run)
1. Generator finishing (train 5000-5999 chunk, then val, test_novel) -> verify_dataset.py -> capture
   today's scene (`eval_scene_capture.py --name today`).
2. Train YOLO11s-seg round 1 (sim stopped; imgsz 1024; seed 0) while writing agent/ code + tests.
3. Evaluate round 1 (val, test_novel, today separately; criteria: today recall >= 95 % boxes and
   cylinders, precision >= 0.9).
4. Integration + rehearsal.

## Open questions
- None.

## Blockers
- None.

## Constraints (agreed)
- Do not modify files in docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256.
- New code in new files; all existing offline tests must keep passing.
- GPU 12 GB: one Isaac instance; VLM before heavy rendering; no training while the sim runs.
- Ground truth only for evaluation, never in the pipeline.
