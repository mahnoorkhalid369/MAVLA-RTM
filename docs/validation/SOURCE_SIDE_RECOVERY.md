# Source identity recovery

Current follow-up: [Source evidence and complete transfers](SOURCE_EVIDENCE_PIPELINE.md) adds support
filtering, body/FOV viewpoint quality, a bounded two-image Gemma fallback for uncertain identity, transfer
validation and free-space placement. The history below describes the earlier SigLIP-only side check.

The 2026-11-17 water-bottle run selected the correct table object (#6) and associated it with
wrist mark #1. SigLIP rejected its cap-dominated crop twice. Repeating grounding selected the
same object and reproduced the failure.

The source inspection now follows this bounded path:

1. Inspect identity and grasp geometry from the usual pre-grasp hover.
2. If the source identity vote rejects it, freeze the object ID and attempt one side observation.
   Look at the measured body centre, from 25 or 35 degrees above horizontal, at 0.35 or 0.45 m.
   Require a viewing direction at least 20 degrees different from the original.
3. Keep every measured object, including the source, in collision checks. Only the visibility
   ray can enter the source box. Validate camera IK/reach, joint zones, and the entire motion path.
4. Capture settled wrist RGB-D. Bind the visible surface to exactly one remembered object volume:
   at least 30 finite depth points, 90% inside its oriented bounds (10 mm allowance), and a vertical
   span of at least the smaller of 30 mm or one quarter of its height. A partial side view is not
   required to reveal the stored top height. Require exactly one matching mark, an unclipped
   silhouette, visible lower body, and a positive SigLIP identity verdict on its isolated crop.
   No catalogue name or classifier threshold changes; no model-selected joint commands.
5. If accepted, return to the recorded hover through a newly checked path. Capture fresh top
   RGB-D and apply the existing geometry/clearance checks, then replan the complete transfer.
   The side observation supplies identity only; it never replaces grasp geometry.
6. If side identity, visibility, return-to-hover, or the fresh geometry check fails, abort before
   descent/closing. Attempt collision-checked HOME and verify it using joint feedback. HOME success
   does not turn the transfer into success. If no safe HOME path exists, report HOME unverified.
   A simulator reset stops motion and requires a restart rather than using stale geometry.

The side verdict is scoped to the same object, catalogue name, scene revision and sensor epoch,
and expires after the observation age limit. It cannot authorize another object or a later task.
The `source_identity` failure stops orchestration; it does not trigger another grounding call.
Initial non-identity RGB-D failures retain their existing measured recovery. Placement judgement,
grasp selection and collision thresholds are unchanged.

Readable states: `INSPECTING_SOURCE_SIDE`, `RETURNING_TO_INSPECTION`,
`RECHECKING_GRASP_GEOMETRY`, and (on failure) `ABORTING_SOURCE_HOME`.
Saved identity events include both the persistent object ID and frame-local mark.

Validation: offline regression tests exercise same-object binding, accepted/rejected side views,
stale evidence, partial images, geometry gates, motion rejection, HOME failure, and reset handling.
A planning-only replay of the saved bottle scene with conservative bounds for other objects found
a 35-degree side pose, a collision-free return to hover, and a collision-free HOME path.
This replay does not establish recognition or manipulation success.

## Live follow-up: 2026-11-17, live_031905

The live run reached the side pose and returned HOME with joint feedback passing. The side image
`wrist_006` shows the bottle body and its WATER label; the gripper hides the cap. Its measured visible
top is 95.9 mm, compared with the table's 168.4 mm. The first implementation reused the normal
top-view association rule (20 mm top-height agreement), so it rejected this valid partial view
before invoking SigLIP. The volume-based partial binding above corrects that implementation error
without changing the strict top-view association or grasp-geometry requirements.

The saved body points bind uniquely to source #6, even with conservative volumes for every other
table object; the cracker mask binds elsewhere. The failure's sampled real points are retained in
`tests/fixtures/source_side_bottle_body.json` as a regression. Invalid/infinite depth is now excluded
before projection and neighbouring invalid pixels cannot supply horizontal surface normals.

Recognition remains a separate problem: replaying the isolated, unmodified side crop on CPU with
the same eight catalogue names still rejects `white bottle` (milk carton 0.901, white bottle 0.039).
Gravity-aligning that crop also fails (milk carton 0.754, white bottle 0.223). Exploratory quarter-turn
rotations and adding `water bottle` to the candidate names change scores substantially. These are
diagnostics, not calibrated confidence or evidence of general accuracy. No orientation ensemble,
alias guessing, lower identity threshold, or automatic identity override has been enabled.

The corrected association now permits the side identity check to run, but this saved image would
still fail the unchanged classifier and trigger HOME. A successful live transfer has not been
established by this follow-up.
