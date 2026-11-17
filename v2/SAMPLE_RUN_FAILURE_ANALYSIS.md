# Sample run failure analysis: yellow block to red tray

Reviewed 2026-11-11. This describes the captured run before the pipeline refactor. See [PIPELINE.md](PIPELINE.md)
for subsequent implementation and its validation limits.

## Evidence

The pasted terminal output matches [`events.jsonl`](../out/demo/2026-11-11/live_052729/01/events.jsonl) from `out/demo/2026-11-11/live_052729/01/`. Saved images and raw worker answers expose details omitted from the terminal. Line numbers below refer to that event file as inspected; the run may append further events.

- [Initial numbered scene](../out/demo/2026-11-11/live_052729/01/images/scene_1_ids.jpg): the yellow cube is world object `#6`; the red tray is `#4`.
- [After placement](../out/demo/2026-11-11/live_052729/01/images/scene_3_ids.jpg): the yellow cube is on the arrangement mat, and has no detector mark.
- [Later detected again](../out/demo/2026-11-11/live_052729/01/images/scene_4_ids.jpg): the same cube has mark `#6` on the mat.
- [Later missed again](../out/demo/2026-11-11/live_052729/01/images/scene_5_ids.jpg): the cube is still plainly visible, but unmarked.

## 1. Primary task failure: the destination was grounded incorrectly

At 05:39:44 the manipulation worker answered:

```text
place on R3
names: #6 = yellow block; R3 = red tray
```

This is event line 59. `R3` is actually **arrangement mat cell A1**, as the planner reports at line 67. The red tray is the visible measured object `#4`; these are separate namespaces and entities. The output is syntactically valid but semantically wrong.

[`Manipulation._act()`](src/agent/workers.py), lines 189–208, checks that a destination exists, accepts `R3`, and converts a region placement to `in`. It does not establish that the chosen destination matches “red tray.” The `names:` parser only reads `#<number>` entries, so the incorrect claim `R3 = red tray` is not validated against the calibrated region name.

The robot then performs the accepted A1 placement. Finger contact, lift retention, and HOME checks pass. The saved image confirms the actual destination. There is no evidence that trajectory tracking caused this placement error; the wrong target was selected before planning.

Needed change: validate the grounded destination against the requested entity before motion. For this scene, bind the tray reference to `#4`. Keep fixed regions and measured objects explicit; do not introduce a universal hard-coded “red tray means #4” rule.

## 2. Verification checked the accepted action, not the user's requested destination

At event line 91, placement verification records:

```text
object_id: 6
relation: in
reference_region: R3
planned_centre_mm: [-120.0, 500.0, 30.2]
pass: true
by: depth at the planned place (the detector did not outline the object)
```

Thus the placement PASS means depth is consistent with the planned A1 placement. The task verification at line 105 also checks A1, because its requirements come from manipulation's accepted action.

There is an additional evidence weakness: the detector did not outline the placed block. In this fallback, [`verify_placement()`](src/agent/tools/verification.py), lines 170–176, checks depth near the planned surface; it does not recover a fully measured object pose. [`assume_placed()`](src/agent/tools/manipulation.py), lines 397–406, previously set the remembered object geometry to the planned pose. The task-level verifier can subsequently use that remembered geometry and describe the result as “measured.” Depth presence is useful evidence, but should remain distinguishable from object identification and a complete measured pose.

At 05:44:04 the separate final visual check returns `no: the yellow block is not on the red tray` (line 108). That completion gate works and prevents false task success.

Needed change: preserve the original task's destination separately from action outcomes. Label verification evidence by what it establishes: contact, depth presence, measured pose, identity, relation, or user-goal agreement.

## 3. Master memory makes unsupported completion claims

At 05:39:04, after the pick but before the place, the master writes that the block has already been placed and the task is complete (line 53). At 05:43:18 it again claims the red-tray task is done, despite the worker explicitly reporting A1 (line 101). After the final check rejects success, the next response omits a new memory line, so `memory_of()` retains “Task complete” (line 111).

These are concrete inconsistencies between model-written narrative and execution evidence. The model eventually corrects its memory at 05:46:30, but several calls have already been spent recovering from an incorrect account of progress.

Needed change: record executed actions and verified conditions in task state. Use model-written memory as a reasoning summary, with verified progress supplied independently. Explicitly invalidate completion after a failed final check.

## 4. Recovery chooses an inconsistent object and loses its measurement

The master asks to pick a yellow block described as being in the pickup-area cluster. Grounding instead selects `#6`, the previously moved block on the mat (lines 138 and 166). The later answer even names it as `yellow block on arrangement mat cell A1`.

At 05:48:44 the skill rejects `#6` because that world ID is not associated with a measurement in the new frame (line 143). This is a correct refusal to execute from an unmeasured pose; it does not mean the block has disappeared or is physically unreachable.

The images show intermittent detector coverage: `#6` is marked in `scene_4_ids.jpg` but absent from `scene_5_ids.jpg`, although the yellow cube remains visible on the mat. Grounding can also see old object records because `Scene.text(numbers=True)` lists persistent on-surface/occluded records without requiring membership in the current observation.

Timing contributes to the first retry: frame `table_006` was taken at elapsed 743.424 seconds; grounding finished at 900.364 seconds, approximately 157 seconds later. The observation limit is 120 seconds. The skill therefore obtains a new frame (`table_007`), which misses the block. On the second attempt, grounding also selects a remembered `#6` absent from its available measurement. Refreshing alone is insufficient without rebinding the action to the refreshed observation.

Needed change: validate the whole object description, carry its frame/version, distinguish remembered from currently measured objects, and refresh/re-ground when the action's evidence expires or the object cannot be measured.

## 5. View recovery is blocked by residual robot geometry

Both `park_for_view` events, lines 147 and 174, show all nine candidate paths with `path_free: false`. The terminal's “no collision-free pose that leaves the table camera's view clear” does not mean all nine views were occluded; the path constraints already reject every candidate.

Earlier planning reveals the same class of issue: unexplained depth columns collide with `j1_link`, and five Gemma consultations classify those regions as robot parts before records are removed (lines 30–33 and 66–73). A new perception call regenerates the unexplained columns, so those exclusions do not persist.

Offline numerical check: reconstructing the rounded unexplained columns from `table_007`, using nominal ARM_HOME as the approximate starting joints and no other scene obstacles, produces `j1_link` contacts with `unseen_18` and `unseen_20`, about 1.5 mm into the collision margin. All nine parking paths fail on these same starting contacts. The logged HOME error was only 0.016 degrees before this recovery. This is strong evidence that residual self geometry is sufficient to prevent parking; it is an approximate replay, not an exact replay of raw depth and joint telemetry.

[`park_for_view()`](src/agent/tools/manipulation.py) does not use the collision-consultation callback available to the pick/place planner. The same residual geometry that the earlier skill worked around can therefore block camera recovery.

Needed change: fix geometric self filtering and obstacle construction consistently across planning and view recovery. Inspect why remaining robot-depth points become occupied columns overlapping the robot's starting configuration. The log establishes the resulting collision; it cannot distinguish all possible raw-depth, robot-shape, and voxelization causes. Do not solve this by disabling collision checks or globally ignoring unexplained obstacles.

## 6. Retrying does not require changed evidence

The master repeats nearly the same pick request at 05:49:18 after the first failure. It reaches the same `#6` measurement failure and the same parking failure. It also spends calls redescribing an object without establishing that the pose needed by the skill can now be measured.

Each dispatch constructs a new `PickPlace`, resetting its local retry counters. The overall task budgets still bound execution, but there is no enforced requirement that the next attempt change the object binding, measurement, view, or planning conditions.

Needed change: return explicit failure information such as `target_not_measured`, observation ID, attempted object, and `view_motion_blocked`. Require new evidence or a materially changed recovery action before repeating the same failed operation. Decide explicitly whether to relocate the already moved block or choose a different yellow block.

## Priority

1. Prevent the wrong destination from being accepted before motion.
2. Separate task-goal verification from action-execution checks and model-written memory.
3. Preserve the distinction between remembered, detector-measured, and depth-supported object state; bind actions to valid observations.
4. Resolve robot self-depth columns so recovery can move safely.
5. Make retries respond to the actual failed precondition instead of repeating the same request.

The available event file ends with another manipulation dispatch at 05:52:50 and has no final result in the inspected snapshot. This analysis diagnoses the observed failures; it does not claim a later terminal outcome.
