# Global-to-local active perception (implemented 2026-10-14, branch `feature/active-perception`)

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

Rollback point: tag `before-active-perception`. Audit: `ACTIVE_PERCEPTION_AUDIT.md`. Supervisor architecture:
`ARCHITECTURE.md`. Everything marked *measured* comes from live headless runs in `out/active/` (ground truth
read only by the evaluation scripts, after or beside the agent, never by agent code).

## 1. Data flow

```mermaid
flowchart LR
  U[goal] --> S[Goal supervisor]
  S -->|observe / locate / answer| TC[Table camera\nRGB-D, fixed]
  TC --> P1[YOLO11s-seg masks → depth geometry\n→ Gemma descriptions (cached)]
  P1 --> W[(World model\nobjects + per-camera observations)]
  S -->|inspect X / explore behind Y| EX[Exploration agent\nviewpoint planner]
  EX -->|camera look-at IK, visibility,\ncollision-checked path| EXE[Tools.execute → RosExecutor → safety layer]
  EXE --> WC[Wrist camera\nRGB-D on J5 housing]
  WC --> P2[YOLO + depth geometry in the world frame\n(FK of the camera frame)]
  P2 -->|association: matched / partial / new / ambiguous\nfusion rule| W
  W --> L[locate / answer / report]
  L --> V[Verification: goal = evidence, not motion]
  EX -->|then| H[return HOME + joint check]
```

## 2. Global-to-local handoff

| Step | Who | What |
|---|---|---|
| Global look | perception (table camera) | every object measured (position, size, top visible), Gemma names (hypotheses) |
| Need more information? | supervisor (deterministic) | the planner asked to inspect / explore, **or** `locate` reports "not visible" → one exploration subtask, then `locate` again (never twice) |
| Where to look | exploration agent | `inspect X`: X's measured top; `explore behind Y`: the table camera's occlusion shadow behind Y (or behind the largest shadows) |
| Is a view possible and safe? | exploration agent | camera candidates on spheres (0.35/0.45 m, elevations 60/40° for inspect, 70/55/40° for regions, azimuth 45° steps) → camera look-at IK (5 mm / 2°), joints ≥ 5° inside limits, whole-region line of sight clear of every measured object by 10 mm (9 points ± 30 mm, 2 heights), robot not in the way, pose and the whole joint path collision-free (CollisionChecker with measured obstacles) |
| Move | validated execution | one MoveStep via RosExecutor + the simulator's safety layer; reach verified from joint feedback (≤ 1.5°) and camera FK |
| Local look | perception (wrist camera) | fresh frame after the arm is still; robot (incl. the visual-only camera bracket) excluded; objects measured in the world frame |
| Merge | world model | association by geometry: matched (≤ 30 mm, top ≤ 20 mm), partial view (centre inside a stored footprint), new, or ambiguous (2 similar candidates → not merged). Wrist geometry replaces table geometry only if its top face was seen and its footprint agrees within 10 mm |
| Decide | locate / verification | a newly seen object answers the goal only if its own description fits and confirm_object on its crop says yes (generic "the object behind…": the single newly seen complete object) |
| Finish | manipulation | return HOME (collision-checked) + HOME joint check |

The wrist camera never commands motion; the LLM never provides poses or joint values.

## 3. Frames and calibration (measured, Checkpoint 2, `out/active/cp2_dual_camera.json`)

- World = robot base frame (+x robot's right, +y forward, +z up). Table camera pose `table_camera_pose()` from
  the constants the simulator uses; wrist camera pose = URDF FK of `camera_color_optical_frame` (J1–J5 chain;
  J6 does not move it; optical axis toed 30° to the finger gap, not the gripper axis). Intrinsics from camera_info.
- Validation without robot motion: true object centres reprojected into each image landed in the matching mask
  20/20 (table) and 4/4 (wrist); wrist-camera position error 0.8–1.1 mm for fully visible objects at 0.74–0.83 m
  (table 3.8–5.5 mm); partial views near the fingers 19–23 mm → the fusion rule above. 5/5 cross-camera
  associations correct.
- Bug found and fixed before any exploratory motion: the wrist-camera bracket is a visual-only URDF part; the
  table camera measured it as a floating object and every viewpoint path "collided" with it → added to the
  robot-exclusion volume.

## 4. Test scenes (`sim/scenarios/*.json`, `python -m sim.run_scenario --scenario NAME`, or `./launch/start_demo.sh --scenario NAME`)

The default scene and the frozen sim files are untouched: `sim/run_scenario.py` fills the scene dictionary
before the frozen `run_live` starts and swaps placeholder prims for YCB meshes after the frozen `setup_scene`.
Poses are fixed (seed 0); the simulator's ground-truth topic covers every scenario object.

| Scene | Objects | Conditions covered |
|---|---|---|
| default (validated) | 21 coloured blocks / cylinders / sphere | single clear target, look-alikes, absent banana, depths, orientations |
| `ap_occlusion` | tall grey box hiding a 35 mm red cube; far grey box hiding a green cylinder; 3 visible cubes | partial / full occlusion from the table camera (design check: 0 % of both hidden tops visible), absent banana, target seen only after a viewpoint change |
| `ap_categories` | YCB banana, bowl, mug (never in YOLO training), mustard + bleach bottles (in training), apple / orange **proxies** (spheres, labelled as proxies), green mat, look-alike red cubes, two 25 mm yellow objects (cylinder vs cube) | new categories, look-alikes, close inspection |

Measured scene conditions: the mustard bottle topples while the physics settles (it lies on its side). The
no-safe-viewpoint case is exercised by the viewpoint planner (both hidden regions were first unreachable with
a single-point line of sight, `live_occlusion2`); with the whole-region test both regions are reachable.

## 5. Live results (measured, headless)

| Test | Instruction | Result | Ground truth |
|---|---|---|---|
| A | Find the red cube and tell me where it is. | completed, 0 motions, 70.8 s (22 Gemma calls: first look describes 21 objects) | (2, 577) mm vs true (0, 580) |
| B | …move closer to inspect it, tell me position and shape | completed, 1 viewpoint (plan 5.6 s, move 22.4 s), HOME 0.016°, no grasp | wrist 0.7 mm vs table 3.6 mm (CP3); "red cube", 58×58×60 mm (true 60) |
| C | Find the object behind the box. | completed: hidden object found by the wrist camera, 59.0 s, 1 viewpoint | (470, 301) vs true (470, 300), 35 mm tall (true 35), "Red Cube" |
| C' | Find the green cylinder. / Find the red cube. (both hidden) | completed after the exploration subtask; 45.9 s / 63.8 s; 1 / 2 viewpoints | (−70, 1020) vs (−70, 1020); (470, 301) vs (470, 300) |
| D | Find the banana. (absent) | failed honestly after 2 wrist views of hidden regions; no forced match | no banana in the scene |
| E | Describe the objects on the table. | 13 objects reported, 49.7 s | 12/12 objects detected and localized (incl. banana, mug, bowl unseen in YOLO training), 1 duplicate mask |
| E | Find the bottle. | asked: two candidates (both real bottles) | correct ambiguity |
| E | Which objects look like fruit? | "the yellow banana" (unverified) | banana correct; sphere proxies not called fruit |
| E | Inspect the yellow curved object. | wrist close-up: "yellow banana, shape curved", 1 mm from the table estimate | banana |
| E | Which of the two small yellow objects is cylindrical? | **wrong** ("object 5" = bleach bottle; unverified answer from table descriptions, no inspection chosen) | small_yellow_cylinder |
| F | yellow cube → tan box; red cube → green cube | completed 267.5 s / 187.8 s, 28 / 7 Gemma calls, placement 1.6 / 3.9 mm, HOME 0.016° | correct objects; no regression vs 446.7 / 257.2 s (refactor A/B, higher load) |

Gemma's names on the categories scene: banana, bottles, cubes correct; mug "roulette wheel", bowl "disc",
small cylinder "button" — semantic recognition is the weak link, not segmentation or geometry.
Safety: 0 failed or rejected motions in every active-perception session; every viewpoint reached within
0.06° / 4.1 mm / 0.72°.

## 6. Global-only vs global + local (matched instructions, same scenes)

| Case | Global only | Global + local | Extra cost |
|---|---|---|---|
| Hidden targets (2) | not visible → "not found" (0/2) | found 2/2, ≤ 1 mm | +35–55 s, 1–2 viewpoint moves |
| Absent object | "not found" | "not found" (same answer) | +~75 s, 2 viewpoint moves (no benefit) |
| Visible object position | 3.6 mm | 0.7 mm | +~60 s, 1 move |
| Shape of a small object | wrong (Gemma on table crops) | not used by the planner | — |

Active perception paid off where the table camera had no evidence (occlusion) and for precise local geometry;
it added cost without benefit for absent objects. Sample sizes are 1 run per case: these are demonstrations,
not statistics.

## 7. Limitations / next steps

- Planner (2B) still misses when to inspect (shape question answered without a close look); a rule "property
  questions about small / ambiguous objects → inspect" or a better planner model is the next step.
- Exploration regions = table-camera occlusion shadows of measured objects; regions hidden by unmeasured
  structure, or outside the table view, are not proposed. No wrist-only scanning of the whole workspace.
- Gemma names novel objects poorly (mug, bowl); YOLO segmented them well. The answer / describe outputs are
  marked unverified.
- One run per test; scene physics changed one pose (toppled bottle).
