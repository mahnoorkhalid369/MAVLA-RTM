# Active perception — camera and architecture audit (Checkpoint 1, 2026-10-14)

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

Branch `feature/active-perception` from `refactor/goal-supervisor` (tag `before-active-perception` = 5b09016).

| Question | Finding (from code) | Status |
|---|---|---|
| Timestamped RGB-D from both cameras? | `sim/run_live.py` publishes `/table_camera/*` and `/camera/*` (wrist): rgb8, 32FC1 depth (m), camera_info, sim-time stamps; `agent/ros_bridge.RosBridge` subscribes both; `snapshot(camera)` waits for the arm to be still and a frame newer than the request (RGB and depth within 0.1 s) | available; wrist path unused so far |
| Calibrated into the world frame? | table: `control.robot.table_camera_pose()` from the same constants `sim/scene.add_table_camera` uses; wrist: `sim/scene.add_wrist_camera` authors the camera ON the URDF link `camera_color_optical_frame` (+180° about X for USD→ROS optical) | consistent by construction, **not yet measured** → Checkpoint 2 validates both against simulator ground truth before any exploratory motion |
| Wrist camera pose from joints? | `Kinematics(tip="camera_color_optical_frame").fk(q)`; camera on the J5 housing side (URDF `camera_mount_joint`), toed in 30° to the finger gap; moves with J1–J5, not with J6. Optical axis ≠ gripper approach axis | available |
| Intrinsics | both D435-like: 1280×720, 69.4° HFOV (fx = fy ≈ 924 px), clip 0.1–10 m; K taken from camera_info at runtime | available |
| Agent access to the wrist camera | `Tools.perceive("wrist")` (YOLO + depth geometry + robot exclusion with the gripper at the measured finger angle) | exists, untested |
| Can the planner take a camera viewpoint? | No: `SkillPlanner` plans tool-down TCP poses. Available building blocks: `Kinematics.fk`, joint limits, `joint_move`, `CollisionChecker.check_path`, `RosExecutor` | **missing**: camera look-at IK + viewpoint evaluation (Checkpoint 3) |
| Object identity across observations | `agent/world.WorldModel` per instruction: geometric match (xy 30 mm, top 20 mm, footprint 20 mm); `DescriptionMemory` per session; no camera / source bookkeeping | **extend**: per-camera observations, explicit ambiguous association |
| Exploration | `Tools.park_for_view` only moves the arm out of the table camera's view; capability `explore` declared, not implemented | **missing** (Checkpoints 3–4) |
| Test scenes | default scene = `sim/scene_objects.OBJECTS` (frozen); `sim/scene.setup_scene` builds boxes / cylinders / spheres from it | **missing**: scenario entry point that fills the scene WITHOUT editing frozen files |
| Assets for new categories | YCB meshes on disk (NVIDIA asset server): banana, bowl, mug (never in YOLO training: "novel"), mustard bottle, bleach cleanser (in training); no apple → sphere proxy (labelled) | available |
| Ground truth | `/sim/ground_truth/objects` (all scene objects); read only by evaluation scripts | available |

Minimal changes: (1) dual-camera observation + association in the world model, validated against ground
truth; (2) camera look-at IK + viewpoint candidates scored by distance / view angle / occlusion, every path
collision-checked, executed by RosExecutor, reach verified from joint feedback; (3) `explore` / `inspect`
capabilities in the existing supervisor; (4) `sim/run_scenario.py` + scenario JSONs (frozen scene untouched).
