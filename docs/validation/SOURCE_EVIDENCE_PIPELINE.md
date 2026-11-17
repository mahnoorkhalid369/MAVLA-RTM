# Source evidence and complete transfers

Implementation: 2026-11-17. Prompt revision: `memory-2026-11-17-v8-source-evidence`.
This changes source inspection, transfer validation and placement search within the existing master,
three workers and measured tools. It does not establish a measured accuracy or latency improvement.

## Instruction and image flow

1. Master receives the original request, plain table RGB, catalogue and measured execution state.
   Matching adjacent pickup/placement messages are joined into one transfer. An incomplete pickup in
   a placement request requires the master's existing bounded repair, before dispatch. Different
   subjects are never silently joined. Explicit pick-only requests remain supported.
2. Manipulation grounds the complete transfer from numbered table RGB. A pickup-only grounding for
   a delegated/requested transfer is rejected before motion. Source and destination must be selected
   together so the existing planner checks grasp, lift, carry, lowering, retreat and HOME in advance.
3. The selected source's table crop, persistent ID, catalogue name and measured geometry become its
   reference. A wrist number is a **local mask number**, not another persistent object ID.
4. At the normal hover, fresh wrist RGB-D must uniquely associate the source. SigLIP remains the
   fast semantic check. A clipped/tiny crop or unconfirmed identity uses the existing one-side-view
   recovery. Newly observed obstacles remain in the collision scene.
5. The side planner projects the full source envelope with the captured camera calibration and
   rejects clipped/tiny views. It samples the facing vertical body surface, requires at least six of
   nine rays clear of robot and measured obstacles, and penalizes partial visibility when ranking
   safe viewpoints. Requiring every hidden rear corner to be visible was overly restrictive in the
   saved-scene replay. Full camera motion remains collision-checked with the source present.
6. The settled side frame removes invalid depth, configured fixtures and independently established
   support-plane pixels. It retains other foreground, including neighbours. At least half the valid
   mask depth must remain foreground; the existing 90% unique oriented-volume binding and body-span
   requirements still apply. **No clipping to the expected object's volume** is used to manufacture
   association. Local mark, persistent ID, foreground counts and per-object binding fractions are logged.
7. If the usable, uniquely bound side crop also leaves SigLIP uncertain, manipulation may make
   **one Gemma confirmation call**: frozen isolated table reference + isolated wrist foreground +
   `SOURCE_IDENTITY` from `src/agent/prompts.py`. Back panels, barcode views and rotation are explicitly
   distinguished from identity contradictions. `no`, `unsure`, malformed output or expired evidence
   cannot authorize a grasp. No Gemma call is made when binding/visibility fails or SigLIP passes.
8. Source confirmation is scoped after the successful approach, then sealed to that source/name,
   geometry, scene revision and sensor epoch. Later disturbance cannot rearm it. This avoids changing
   executor invalidation simply to accommodate the approach. The tool also checks freshness, joints
   and scene continuity after inference. No controller accepts model-provided coordinates.
9. Accepted side identity still requires a checked return to hover, **fresh top-view grasp geometry**
   and a newly collision-checked complete transfer. Unresolved source inspection attempts checked HOME
   and stops before any grasp. Successful HOME does not convert failure into task success.
10. Post-release placement judgement, measured completion gate and the independent check against the
    entire original request remain. Experimental inference/speed profiles and baseline shortcut defaults
    are unchanged (`V2_REUSE_PLACEMENT=0`, `V2_FINISH_PLAN=0`).

## Available placement space

Ordinary `on` references with an explicitly named tray/basket/bin/bowl/container use the existing
interior-floor planner; explicitly closed/sealed/lidded names do not get that normalization. A solid
support stays `on` and now searches free positions on its measured top instead of only its centre.
Unclear names are not sufficient to infer a cavity. Existing container wall/floor proxy assumptions
still apply; arbitrary containers require suitable depth/support measurements and configuration.

Candidates use the full oriented payload footprint, the placement gap and measured occupants,
including obstacles crossing the destination outline. Each selected orientation/position must pass
the complete carry/lower/retreat planner. Payload offsets remain handled by `SkillPlanner.place`.
The search is bounded to 12 positions per tested orientation; exhausting it does not prove that no
continuous-space solution exists. Logs show the selected position and unsuccessful path attempts.

A centred water bottle can leave no admissible position for the large cracker box with the current
tray size and margins. The regression preserves that stop. A second case offsets the bottle so a real
free position exists and confirms that oriented placement can find it. Obstacles/gaps are not removed
to make an occupied-destination test succeed. Named exact-position/stacking tasks still need a dedicated
task constraint interface; this change implements ordinary available-space transfers.

## Offline evidence and remaining validation

`tests/fixtures/cheezit_side_foreground.{json,npz}` retains the actual `live_060155/01` failure's mask
and calibrated depth (sampled every fourth pixel), with provenance and matching-RGB hash. The live
frame-specific labels were used only after verifying exact equality with that saved RGB. Other logged
object footprints are conservatively reconstructed as square envelopes because the compact log omits
their oriented extents. These are replay fixtures, not simulator ground truth supplied to the agent.

Replay from `v2`:

```bash
OPENBLAS_NUM_THREADS=1 .venv-seg/bin/python scripts/replay_source_view.py \
  --output out/review_followups/source_evidence_2026-11-17/replay.json
```

The raw source mask reproduces `new`/no binding. After support filtering, 8,354 of 8,915 valid points
remain and bind uniquely to source #3. A merged neighbour remains rejected. The replay finds a
35-degree side pose with collision-checked motion and two-thirds estimated body visibility. It does
not include unavailable live wrist-local/unexplained obstacles and does not execute/render the view.

Regressions exercise real contaminated depth, wrong/merged foreground, ambiguous masks, clipped views,
Gemma yes/no/unsure, one-call budgeting, reference expiry, changed joints/epoch/scene, post-approach
scoping, split-plan repair, dropped destinations, oriented free-space placement and preserved full-tray
failure. Some older test assertions were updated to the already-shortened prompt labels; the independent
final check was not removed or changed.

Validation completed: full agent suite **430 passed, 4 skipped** in 193.16 seconds; one NVML
initialization warning from the saved-frame detector test. The focused source-inspection suites
also passed **55 tests** after adding the source-specific inference timing phase. These are offline
results; they do not measure real pick-and-place timing or model recognition accuracy.

Live validation is still required: correct/wrong objects, back-of-box and cap views, mixed/occluded
masks, occupied trays and explicit pick-only requests. Score wrong-object accepts, correct-object
rejects, binding failures/recovery, calls and wall time on fixed paired cases. Inspect the saved table
reference and wrist confirmation crop together. A mocked `yes` tests controller routing, not Gemma's
recognition accuracy. Retain the existing smoke-image review gate before paired performance profiles.
