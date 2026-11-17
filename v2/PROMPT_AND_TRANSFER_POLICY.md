# Pick-and-place policy and motion stages

Current prompt revision: `memory-2026-11-17-v8-source-evidence`; the source inspection, complete-transfer
validation and available-space placement follow-up is in
[Source evidence and complete transfers](docs/validation/SOURCE_EVIDENCE_PIPELINE.md).
The transfer-policy history below starts at `transfer-2026-11-12-v3`. The measured recovery controller
is described in [RECOVERY_HANDOFFS.md](RECOVERY_HANDOFFS.md).
All master/worker language policies live in
[`src/agent/prompts.py`](src/agent/prompts.py). The existing three workers and plain-language message format remain.

This is also the single source for the master planning/review templates, response-repair instructions,
scene and execution-context framing, final-check image captions and HOME text, and SigLIP's text template.
`master.py`, `workers.py`, `contracts.py`, `loop.py`, and `perception/matcher.py` fill these templates with
runtime data. Every template is included in the per-instruction `prompts.json` manifest with its hash.
The centralization refactor preserves the rendered prompt wording and the policy revision.

## Active prompts

The master plans one complete transfer per requested object, followed by verification of all requested items.
It starts manipulation directly when identities are clear. Perception answers a specific unresolved question;
routine surroundings descriptions and repeated confirmations are excluded from the normal transfer plan.
Failure instructions distinguish identity, inspection, contact/retention, collision and placement failures.
Memory remains advisory; execution and sensor evidence determine progress.

For the banana-and-bottle request, the intended initial response is:

```text
manipulation: pick up the banana and place it in the serving tray
manipulation: pick up the water bottle and place it in the serving tray
verification: check that the banana and water bottle are both in the serving tray
memory: Both transfers pending; none executed.
```

The grounding prompt explicitly separates whole objects, object parts, calibrated regions and movable containers.
An independent short crop check confirms the selected source/destination before any robot command. It uses the
original worker instruction and isolated selected masks, not the grounding model's descriptions as proof. A wrong
or ambiguous selection receives at most one repair. The final goal prompt explicitly checks every requested item.

Perception and crop confirmation use short non-thinking generations (160/120 tokens respectively). Master reasoning,
action grounding and final goal checking retain thinking. No live latency improvement is claimed from CPU tests.

## Transfer execution

```text
Confirm selected identities from table image crops
  → verify empty gripper and reach HOME
  → refresh table measurements if robot motion invalidated the grounding observation
  → plan approach, grasp, lift, carrying path, placement and final HOME
  → move to pre-grasp hover only
  → inspect wrist RGB-D and associate the SAME selected object
  → refine pose and retain local neighbours/unexplained depth as obstacles
  → replan the complete remaining transfer from actual joint feedback
  → controlled descent, close, contact check
  → lift, verify retention BEFORE any carrying motion
  → carry through validated raised waypoint with gripper closed
  → slow lowering, release, retreat
  → measure placement, return HOME, verify joints
```

The carrying waypoint uses the high central `ARM_HOME` joint configuration, with the full payload collision body
attached and the gripper kept closed. Both legs are checked. This does not call the empty-gripper `return_home()`
command, remove an obstacle, or assume HOME is safe for every payload. If either leg is infeasible, the plan fails.
An explicit pick-only request ends holding after lift verification. Place-only starts with the recorded held object.

The wrist inspection is a deterministic RGB-D tool stage, not another general VLM description call. It requires one
unambiguous geometric association to the selected table object, a visible top, fresh capture/joint state, centre
agreement within 15 mm, and top/footprint agreement within 10 mm. Partial/missing/inconsistent evidence stops before
descent. Raw wrist depth, intrinsics and capture joints are saved in `images/wrist_NNN_depth.npz`; the existing wrist
RGB/mark image and inspection event identify the frame. Full approach collision checks follow inspection; the wrist
view alone does not certify clearance.

Regular objects keep their measured centre. Curved shapes use a measured central solid cross-section rather than an
empty bounding-box centre. The full payload body bounds and grasp offset are carried through collision checks and
placement positioning; local finger width alone is insufficient as a payload collision volume. Off-centre curved
grasps currently require a top-down approach. Tilted pick-only fallback remains bounded; tilted placement is unsupported.

Final lowering uses 0.025 m/s instead of the normal 0.08 m/s TCP setting. Existing support clearance, grip squeeze,
contact thresholds and placement tolerances are retained. This is position-controlled lowering with a small release
clearance, not a new force-controlled placement method.

## Recovery and evidence

- A failed lift/carry clears the held claim and marks its old world record missing, enabling reassociation under the
  existing ID. Successful release preserves the record's on-surface status.
- Identical measured grasp/blocked actions cannot be retried just by changing their wording. Changed geometry or
  surroundings is required; the master can ask for a specific physical correction or stop.
- At most two perception calls are allowed between physical progress/user clarification. Excess requests return a
  budget report without inference; persistent requests stop. The overall 40-dispatch/20-decision limits remain.
- VLM collision adjudication and its record-deletion path have been removed. Robot depth self-exclusion uses analytic
  URDF solids in addition to the original spheres, covering flat cylinder rims without a semantic obstacle override.
- Each instruction prints `[POLICY] transfer-2026-11-12-v3 master=<hash>` and saves the loaded prompts/hashes in
  `prompts.json`. `events.jsonl` records the same policy alongside model input images, selected references and results.

## Loading and validation

Restart the agent process to load these code/prompt changes. An already-running Python process retains its imported
modules. After restarting using your normal launcher (`./launch/agent.sh` from `v2`), check the POLICY line and the new
instruction's `prompts.json` before evaluating performance. Existing running sessions were not interrupted or reloaded.

The prior transfer-planning pass had 144 passing targeted tests. The latest controller changes have 99 passing
relevant tests; see [RECOVERY_HANDOFFS.md](RECOVERY_HANDOFFS.md). Coverage spans master/worker contracts, new transfer stages, identity crops, robot self-exclusion,
collision checks, tilted grasps, safety checks, held sensors and saved-frame perception. Real CPU kinematics tests
plan a full transfer through the carrying waypoint, verify an empty-hand return, enforce lowering speed and check
placement compensation for an off-centre hold. Model responses and execution are stubbed in orchestration tests.

No live Gemma/Isaac run was performed: the GPU driver is unavailable in this environment. These checks do not measure
task success, false-success rate or real latency. Next live evaluation should include a single transfer, two-item snack
tray, ambiguous target, missing wrist view, narrow approach, occupied destination and lost-retention recovery. Inspect
both completed transfers and explicit stops. Wrist visibility, model identity judgments, geometry calibration,
collision approximation, friction and grasp stability remain empirical validation requirements.
