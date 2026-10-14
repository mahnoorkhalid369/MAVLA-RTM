# Demo report — autonomous run 2026-10-14 (08:19–~13:00 UTC)

**Demo NOT READY as specified.** In the final rehearsal (cycle 4, no code changes during it), instructions 1 and 3 succeeded 3/3 each, verified both by the agent's own camera check and by simulator ground truth, with no false "done". But instruction 2 ("yellow cube … tan box") never ran: in 3/3 runs the agent asked which of two yellow-looking cubes was meant, and the scripted answer was "cancel". Instruction 4 was therefore not attempted (the brief gates it on 1–3 each ≥ 2/3).

What exists is **a working, honest demo for instructions of the type of 1 and 3** in today's scene. It is not yet a reliable general demo:
- Every success number below comes from 3 runs, so the 95 % lower bound is 0.44.
- Before the last fix, the agent made 2 false "done" reports in cycle 3 by moving the wrong object.

Evidence labels: **measured** = from logs / ground truth in `out/demo/2026-10-14/`; **estimated** = stated as such.

## 1. Final rehearsal (cycle 4, measured; ground truth used only for scoring, after each run)

| Instruction | Runs | Agent COMPLETED | Truth success (95 % Wilson) | False done | False failed | Asked | Placement error vs truth (mm) |
|---|---|---|---|---|---|---|---|
| 1. Put the red cube on the green cube. | 3 | 3 | 3/3 (0.44–1.00) | 0 | 0 | 0 | 3.1, 2.5, 2.0 |
| 2. Pick up the yellow cube and put it on the tan box. | 3 | 0 | 0/3 (0.00–0.56) | 0 | 0 | 3 | – |
| 3. Put the orange cube on the purple cylinder. | 3 | 3 | 3/3 (0.44–1.00) | 0 | 0 | 0 | 3.9, 3.7, 3.9 |
| 4. Put the pink cube next to the white tall cylinder. | not run (gate: 1–3 each ≥ 2/3; instruction 2 = 0/3) | | | | | | |

- Agent's own verification on the 6 successes, from the table camera after parking the arm: xy 1.8–4.0 mm, z −0.8 mm, tilt ≤ 0.02°. Tolerances are xy 10 mm, z 5 mm, tilt 5°, unchanged.
- No neighbour moved (ground truth, all 9 runs). Each run used a fresh simulator.
- Instruction 2 asks because the scene contains, besides the yellow cube, an olive cube and a gold cube that Gemma also describes as yellow cubes. The agent's rule asks when two cues disagree or tie, so the scripted "cancel" ends the run (§4). The visual comparison picked the correct object in run 1, but the description tie still made it ask: the rule is conservative.

**Time per stage (cycle 4, measured, median / max s):**
- UNDERSTANDING 5.3 / 5.8
- OBSERVING 110.6 / 122.2 (park + detect + Gemma describes ~22 objects one crop each)
- LOCALIZING 15.9 / 24.0
- PLANNING_GRASP 9.1 / 17.4
- APPROACHING 34.7 / 36.2
- GRASPING 3.0
- TRANSPORTING 14.6
- PLACING 14.0
- VERIFYING_PLACEMENT 40.8 / 41.5

A complete run takes 251–291 s, plus about 2–3 min to start (Gemma, detector, Isaac Sim).

**VLM (cycle 4, measured):** 238 calls (195 describe, 18 resolve, 12 confirm, 9 parse, 4 compare). 3 answers were invalid, all resolve, all fixed by the single retry; 0 failures after retry. This is far below the trial counts ARCHITECTURE §9 requires for its VLM thresholds, so it is not evidence that those thresholds are met.

## 2. All rehearsal cycles (measured)

| Cycle | Code state | Result |
|---|---|---|
| 1 | first integration | i1 run 1: CANCELLED. Gemma called 15/21 objects "Yellow square": the yellow mask outlines bled into small objects. Stopped. |
| 2 | fix 1: describe from isolated crops; refresh the executor clock | i1: 2/3 success (2.6, 3.2 mm); run 3 honest FAILED (safety layer rejected a trajectory starting at J1 = 170.0°, my park pose sat on the limit). Stopped. |
| 3 | fix 2: park ≥ 10° inside limits; disturbance check; reuse descriptions | i1 3/3; i2 1/3 (2 honest FAILED: invalid resolve JSON 3×); i3 0/3: 1 honest FAILED (grasp stall 1.46° < 1.5°) and **2 FALSE DONE** (wrong object: brown cube taken for "orange cube", confirmed "yes" by the same model, placed and geometrically verified). |
| 4 | fix 3 (last allowed): field-specific retry, square width = mean of both extents, reference cross-check | table in §1 — no false done, no false failed. |

All fixes were in new code only. No safety check, limit, tolerance or pass criterion was changed, and the 1.5° contact threshold was kept.

Per-run tables: `out/demo/2026-10-14/rehearsal/cycle{1..4}_summary.md`. Per-run evidence: `rehearsal/cycleN/iK_runR/`:
- `events.jsonl`
- `result.json`
- `score.json`
- `truth_before/after.json`
- `images/`: numbered masks, Gemma names, crops, comparisons

## 3. Perception (measured)

**Dataset synthetic_v1:**
- 6,000 train / 1,000 val / 500 test_novel images. verify_dataset found no problems and 0 area mismatches.
- YOLO labels were fixed to one polygon per instance.
- No images excluded and no chunks regenerated.
- The generator was not restarted.

**YOLO11s-seg, round 1 only** (seed 0, imgsz 1024): 49 epochs in 1.75 h (139 s epoch 1, 133 s/epoch after). Recall and precision at mask IoU ≥ 0.5 and conf 0.25 (`out/demo/2026-10-14/yolo/round1/metrics.json`):

| Set | Recall | Precision | Lowest-recall type |
|---|---|---|---|
| Today's scene (1 image, 20 objects) | box 16/16, cylinder 3/3, sphere 1/1 | 0.952 (20/21) | – |
| val (1,000 images, 12,909 objects) | 0.976 | 0.904 | plate 0.965 |
| test_novel (500 images, 6,569 objects) | 0.965 | 0.893 | ycb_novel 0.939 |

- The pass criteria (today: recall ≥ 95 % for boxes and cylinders, precision ≥ 0.9) **were met in round 1**, so no further rounds were needed.
- Caveat: one image. The 95 % Wilson lower bound for 16/16 is 0.81. The 20 extra layouts are needed for a meaningful number.

**Geometry vs truth** (today's capture, ground-truth masks used only to measure the error, `out/demo/2026-10-14/geometry_check_today.json`):
- box centres 1.7–4.6 mm
- box widths −1.8…+1.0 mm (square footprints now use the mean of both extents)
- cylinder widths −0.1…+0.9 mm (thin 25 mm cylinder −4.4 mm)
- top +1.5 mm on every object
- An object whose top face is hidden (the blue cube under the lime cube) gets an unknown tilt; it is refused, not guessed.

**Gemma roles offline on today's capture** (`out/demo/2026-10-14/vlm_check/`): 8/8 rehearsal references resolved correctly and confirmed (one image, so indicative only). The side-by-side comparison picked the brown cube for "orange cube" 3/3 times on the crops from the false-done run (`compare_check.json`). That is why a single cue is not trusted.

## 4. Why "not ready", and what would make it ready (estimates)

1. **Reference resolution is the weak link (measured).**
   - Gemma's resolver took a partial match ("brown/orange-brown") over an exact "Orange" description, and the confirmation (same model) agreed: a correlated error, the joint-failure mode of ARCHITECTURE rev 3.
   - The cross-check now prevents that case, but at the price of asking when the cues tie (instruction 2).
   - Next: measure the joint-failure rate on the planned 400-reference set; evaluate a second model for confirmation if it exceeds 2 %; and test the rule "accept the visual choice if it is among the most-specific descriptions". Instruction 2 run 1 suggests that rule would have resolved it; this is unverified.
2. **Small samples (measured).** 3 runs per instruction, one layout, one image for the detector criterion. The ARCHITECTURE §9 trial counts are not met.
3. **Instruction 4 (next_to) untested in the sim.** It plans offline on today's measured geometry (`plan_dryrun_today.txt`).

## 5. Rules and records

- **Rules kept:**
  - The agent uses no object list, colour names or classical detector.
  - Ground truth is read only by the scorer process after each run.
  - The LLM never commands motion; all motion goes through SkillPlanner + CollisionChecker + RosExecutor + the safety layer.
  - No limit or tolerance was changed.
  - No sudo, no system changes; installs only into `.venv-seg` (pytest via uv).
- **Licence:** Ultralytics YOLO11s-seg is AGPL-3.0, accepted for this demo only (DECISIONS.md).
- **Baseline manifest:** verified at every milestone and at the end (all 21 files OK).
- **Tests:** `/usr/bin/python3 -m pytest` gives 129 passed, 1 skipped (the agent tests need langgraph and are skipped there). `.venv-seg/bin/python -m pytest tests/test_agent.py` gives 25 passed.
- **Hard stops:** none triggered. Disk free stayed at ~770 GB, no GPU or driver faults, ~4.7 h of the 12 h budget used.
- **Demo paths tested (measured):**
  - `agent.run` interactive headless: COMPLETED, 4.0 mm.
  - `launch/start_demo.sh` with the Isaac Sim window and camera windows: COMPLETED, 1.7 mm, GPU peak 7.1 GB of 12.
  - Clean shutdown on an empty line in both.
- Records: `PROGRESS.md`, `DECISIONS.md` (every autonomous choice), `docs/AUDIT_AND_PLAN.md` §8, `DEMO.md`.
