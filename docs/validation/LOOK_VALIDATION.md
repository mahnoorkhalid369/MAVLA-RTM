# Embodied instructions: shared identity, `look`, measured view verification (2026-10-20)

**Commits** on `experiment/lab-exp1`:

| Commit | Change |
|---|---|
| `c2cd6ba` | shared robot identity |
| `08be567` | the `look` capability |
| `05fa127` | measured goal verification |
| `6e56980` | robot frame shown as the default in the menu |

**Physical hardware.** No physical robot was connected or commanded: target `isaac_sim`, `ROS_DOMAIN_ID=42`, and no
driver process or ROS node existed in domain 0 or 42 before the run. No frozen file changed:
`grep -v '^#' docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256 | sha256sum -c` passes.

## 1. Gemma planning, before and after

The benchmark is `evaluation/planner_check.py`: real Gemma 4 E2B, plan role only, no simulator. It has 19 cases:

- 12 tabletop cases, the earlier set;
- 7 embodied cases: look left / right / down / behind, "what is on your left?", "look at the red cube",
  "move your hand to the left".

The same script ran on every version (older versions from a git worktree). Results are in
`out/eval/embodiment/planner_*.json`.

| Version | Correct plans | Valid JSON | Tabletop correct | Embodied correct | "Look at your left side." |
|---|---|---|---|---|---|
| baseline `c672491` | 4/19 | 15/19 | 4/12 | 0/7 | `observe` |
| + identity `c2cd6ba` | 5/19 | 15/19 | 5/12 | 0/7 | `observe` |
| + look, frame menu `"robot" \| "camera"` | 5/19 | 17/19 | 4/12 | 1/7 | `look left`, frame **camera** (wrong) |
| + robot frame as default `6e56980` | **9/19** | **18/19** | 5/12 | **4/7** | `look left`, frame robot ✔ |

**Observations:**

- **The identity alone does not fix directional requests.** With no capability to point the camera, Gemma still
  chose `observe`. The identity did raise tabletop JSON validity from 9/12 to 11/12.
- **With `look` and the robot frame shown as the default,** all four directional requests get the right direction
  in the robot frame. With both frames listed in the menu, Gemma picked the "camera" frame three times: "camera"
  read as "the camera that looks".
- **Still wrong in the final run:**
  - "What is on your left?" (invalid JSON);
  - "Look at the red cube." (`observe` instead of `inspect`);
  - "Move your hand to the left." (`look left`: Gemma conflates moving the hand with pointing the camera).
- **Tabletop failures are unchanged.** Gemma writes `pick` + `pick_place` instead of one `pick_place`. The
  supervisor executes that correctly, because `pick_place` of the held object becomes `place`, but the benchmark
  counts only the exact sequence.

## 2. Isaac Sim validation

Headless Isaac Sim with the `default` (blocks) scene, `--motion real` (simulator only), and eight instructions in
one session. Full trace: `out/eval/embodiment/live_030946/<nn>/events.jsonl`.

Errors are measured from joint feedback after the move:

- **reach:** the worst joint error against the planned pose, and the camera position error from forward kinematics;
- **view error:** the bearing, from the base, of the floor point the optical axis meets, compared with the
  requested direction.

| # | Instruction | Gemma's plan | Movement (wrist camera) | Reach error | View error | Result |
|---|---|---|---|---|---|---|
| 1 | Look at your left side. | `look left (robot)` | HOME (89, 500, 746) → (−426, −1, 820) mm, 722 mm, 13.2 s | joints 0.007°, camera 1.2 mm, axis 0.36° | **0.3°** (looks at (−996, −5) mm) | COMPLETED, `viewing(left)` measured |
| 2 | Look to your right. | invalid JSON twice (BD-1) | none | — | — | FAILED: could not plan |
| 3 | Look down at the floor. | `look down (robot)` | none | — | — | no safe viewpoint from the current (left) pose: look-at IK unreachable. Gemma decided `stop`. FAILED |
| 4 | Look behind you. | `look backward (robot)` | (−426, −1, 820) → (119, −321, 983) mm, 653 mm, 32.7 s | joints 0.007°, camera 1.6 mm, axis 0.21° | **0.1°** (looks at (−2, −995) mm) | COMPLETED |
| 5 | Look to the left in the fixed camera's picture. | invalid JSON twice (BD-1) | none | — | — | FAILED: could not plan |
| 6 | Move your hand to the left. | `look left (robot)` (wrong intent) | (119, −321, 983) → (−161, 146, 848) mm, 561 mm, 29.6 s | joints 0.006°, camera 3.8 mm, axis 0.72° | 0.4° | COMPLETED as *looking left*, not what was asked |
| 7 | Go back to your home position. | `return_home` | → HOME, 10.8 s | HOME 0.016°, TCP 0.16 mm | — | COMPLETED |
| 8 | Look at your left side. | `look left (robot)` | HOME → (−426, −1, 820) mm, 14.5 s | joints 0.007°, camera 1.2 mm, axis 0.36° | 0.3° | COMPLETED |

**Safety and authorization:**

- "Look …" requests were classified as **description**, which grants `none` and `sensing`. `look` has the
  `sensing` effect class, so it was authorized, and `object_change` was not granted.
- Every move was planned by the existing viewpoint planner, with the collision-free path checked by the frozen
  `CollisionChecker`, and executed by the frozen executor.
- No safety-layer event occurred.

**Nothing failed was reported as success:**

- the two planning failures, and the infeasible `down`, ended FAILED with the reason;
- for `down`, Gemma was consulted and chose `stop`.

## 3. Remaining limitations

- **Hand versus camera intent (instruction 6).** "Move your hand to the left" was executed and verified as looking
  left.
  - The identity says pointing the camera is not moving the hand, but the 2B model ignores it.
  - No capability moves the gripper to a free position.
  - The verification is correct for the effect that was planned. It cannot know the intent was different.
  - Possible next steps: a `move_hand` capability (a translation in an explicit frame, collision-checked), or
    Gemma asking when a request names the hand.
- **JSON validity (BD-1).** 1 of 19 offline plans and 2 of 8 live plans were invalid twice. Live prompts include
  the robot-state context, which changes Gemma's output.
- **`down` depends on the current pose.** From HOME it plans (5.8° offline). From the left viewpoint the
  look-at IK did not converge (seeds: the current pose and HOME).
- **The arm stays at the viewpoint after `look`** (by design, so that the view can be measured). The next motion,
  or `return_home`, moves it.
- **Directions are fixed:** left, right, forward, backward and down, in the robot frame or the fixed camera's
  picture frame. "Up", angles ("a bit to the left") and the user's own frame ("my left") are not supported; a word
  outside the vocabulary fails validation rather than being guessed.
- **What the view proves.** The verification measures where the camera points, not what it saw.
  "What is on your left?" still depends on Gemma composing `look` with `describe_scene` / `answer`, which failed
  offline (invalid JSON).
