# VT6L manipulation system — environment audit, technology selection and milestone plan

Status date: 2026-10-13. Everything marked **measured** was executed on this machine today;
**from files** was read from the project; **unverified** needs a source this project does not have.

## 1. Environment audit

### Machine and software
| Item | Value | Status |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-146 | measured |
| CPU / RAM | 32 threads, 62 GB | measured |
| GPU | 12 GB NVIDIA (name unavailable: see blocker) | from earlier sessions |
| **NVIDIA driver** | **kernel module 595.91.07, user libraries 595.99.02 (apt upgrade on 2026-10-11)** | **measured: BLOCKER** |
| Isaac Sim | 6.1.0 (pip, `.venv`, Python 3.12.3); standalone copy in `~/isaacsim` | measured |
| Model env | `.venv-vlm`: torch 2.11.0 + CUDA 13.0, transformers 5.19.0; Gemma 4 E2B-it in `models/` | measured |
| ROS 2 | Jazzy (`/opt/ros/jazzy`); `controller_manager` present; **MoveIt 2 not installed** | measured |
| Isaac extensions | `isaacsim.robot_motion.cumotion`, `.experimental.motion_generation`, `.pink` (IK), `.controllers`; `isaacsim.robot.surface_gripper`; `isaacsim.replicator.grasping`; `isaacsim.ros2.bridge/.control` | measured |
| Not installed | cuRobo (stand-alone), Faster-Whisper, YOLO, Grounding DINO, SAM, FoundationPose, pytest in `.venv-vlm` (system pytest 7.4.4 is used) | measured |

**Blocker.** With the user-space driver newer than the loaded kernel module, NVML fails
(`NVML_ERROR_LIB_RM_VERSION_MISMATCH`) and Isaac Sim's RTX renderer cannot create a device
("No device could be created"; measured with `sim.run_live --headless`). PyTorch CUDA still
reports available. **Fix: reboot** so the 595.99 kernel module loads. No simulation test can run
until then.

### Robot description — updated 2026-10-13: `assets/vt6l/vt6l.urdf` (from the Epson manual)
Source: Epson *VT series Manipulator Manual* (`VT_Robot.pdf`, footer "Rev.6"), Appendix A
specifications (p.175-176), Outer Dimensions (p.42-43), Standard Motion Range (p.44-45).
The earlier C8L-based `vt6l_proxy.urdf` is kept only for comparison.

| Item | Value | Status |
|---|---|---|
| Kinematics | J2 axis 412 mm above base, 100 mm in front of J1; upper arm 420; J3 → P point 400 (no elbow offset); P point → flange 80; spherical wrist (P point = J4/J5/J6 intersection, manual footnote) | **from the manual's drawings**; `tests/test_vt6l_model.py` reproduces reach 920 / 1000 mm, height 1232 mm and all four motion-range radii (274.4, 527.3, 351.7, 293.9 mm) within 0.1 mm |
| Joint ranges (deg) | J1 ±170, J2 −160/+65, J3 −51/+190, J4 ±200, J5 ±125, J6 ±360 | **manual (Appendix A)** |
| Max. speeds (deg/s) | 166.2, 122.5, 141.2, 268.7, 296.8, 293.2 (AC standard; J4 188.1 / J6 234.5 on protection or DC models) | **manual**; the old proxy values (294–720 deg/s) were wrong |
| Joint directions | J2 + = upper arm back, J3 + = forearm up (decided by the radii: the manual's wording in footnotes *3/*4 contradicts *1/*2), J5 + = flange up | from the manual; **J1, J4, J6 directions unverified** (symmetric ranges, not shown) |
| Effort limits | 400, 400, 200, 60, 60, 30 N·m | **simulation-only placeholders** (Epson gives motor power 300/300/200/100/100/100 W, not torque); never to authorise physical motion |
| Payload / wrist | 3 kg rated, 6 kg max; allowable moment J4/J5 12 N·m, J6 7 N·m; allowable inertia 0.3 / 0.3 / 0.1 kg·m² | manual |
| Geometry (visual/collision) | simplified boxes and cylinders sized from the drawings | approximation (no Epson CAD in the project) |
| Inertia | approximate, total 40 kg (manual weight without cables) | **unverified** (needs CAD) |
| TCP | `tool0 → tcp` 133.7 mm (2F-85 grip frame) | from the gripper asset |
| Effect on the workspace | grasp-feasibility map (`evaluation/grasp_feasibility_map.py`, 5568 full pick plans incl. collision checks, yaws 0/30/45/60°, heights 35/60 mm): **feasible for every yaw from 0.45 to 0.90 m** in all directions except directly behind (J1 ±170°); 0.375–0.45 m depends on bearing and yaw (28–85 %); none beyond 0.90 m (`out/feasibility/grasp_map_20261014_012519.*`). Earlier radial check (yaw 0, ahead): 0.415–0.90 m; near-edge gaps at 0.375–0.41 m where the straight descent fails; proxy was 0.39–0.86 m. Edge test cubes moved (navy 0.89 m, olive 0.43 m); `tests/test_scene_reachability.py` re-checks all 21 tiers | measured with the planner |
| Motion configuration | own DLS IK + quintic joint / straight-line Cartesian moves (`control/`); **no collision checking against the environment yet** | from files |
| Simulator USD | `assets/vt6l/usd/vt6l/vt6l.usda` (`python -m sim.import_vt6l`): joint names, axes, ranges and speeds match the URDF. The importer writes speeds as `newton:velocityLimit`, which **PhysX likely does not enforce** (our planner limits speed itself) | generated and inspected 2026-10-13 |

### Gripper
Robotiq 2F-85 (Isaac Sim asset, CC-BY 4.0), "mimic" physics variant: one driven `finger_joint`
(0 = open 85 mm, closed target 45°, limit 47°), pad friction 1.2 static / 1.0 dynamic.
Grasping is **physical contact** (no attach shortcut). Driven via `/joint_command`. From files.

### Cameras (both RealSense D435-style RGB-D, simulated)
| | Wrist (eye-in-hand) | Table (eye-to-hand) |
|---|---|---|
| Mount | side of the J5 housing, toed in 30° towards the finger gap; does not spin with J6 | stand at (1.5, 0.25, 1.8) m, looking at (0, 0.25, 0), ~50° down |
| Extrinsics | URDF `camera_link` (FK of joint state) | `control/robot.py: table_camera_pose()` |
| Intrinsics | 1280×720, fx = fy = 924.3 px, ideal pinhole, no distortion | same |
| Depth | exact (no noise model), clip 0.1–10 m | same |
| Topics | `/camera/{color,depth}/…` | `/table_camera/{color,depth}/…` |

Gaps: simulated depth has **no D435 noise**, and extrinsics are **exact** (no calibration error);
both make simulation results optimistic.

### Scene
21 rigid objects on the floor (no table: the floor is the work surface, z = 0), sizes 35–100 mm,
wood density (3 kg steel cube), matte materials; graded easy → impossible. Object friction is the
PhysX default. Microphone prop and the camera stand are visual only. From files.

### Existing software (what already works, from earlier sessions)
Perception (VLM detection + depth grounding, blob segmentation, yaw, measured colour), world
memory with "not found here" records, active search with the wrist camera, Gemma task planning
with JSON schema validation, direct commands, typed/voice input (Gemma transcription), ROS 2
bridge, single-command launcher (`launch/run.sh`).

## 2. Gaps against the specification

| Spec requirement | Today | Gap |
|---|---|---|
| Real VT6L kinematics | **done** from the manual (kinematics, ranges, speeds) | Meshes/inertia from Epson CAD; J1/J4/J6 sign check on a real controller |
| Collision-aware planning | none (joint limits only) | Add environment + self-collision checking of swept paths |
| Classical-vision baseline | VLM detection only | Add colour/depth segmentation detector; VLM optional |
| Grasp candidates, ranking | single top-down grasp | Multiple candidates, symmetry, ranked (partly done: yaw, symmetry, J6 choice) |
| Grasp / placement verification | none | Finger-joint check, re-observation after lift and after release |
| Recovery | none | Bounded retry policy per failure class |
| State machine + event log | implicit in one ROS node | Supervisor with explicit states and a JSON event log |
| Speech | Gemma audio | Faster-Whisper backend (not installed) |
| Spatial relations ("next to", "left of") | "on" and table positions | Relation → placement pose, with explicit reference frame |
| Reset, randomised seeded scenes | fixed scene | Scenario generator + reset |
| Evaluation A–E | none | Harness + metrics using sim ground truth (evaluation only) |
| Hardware abstraction | ROS topics | Execution backend interface (sim now, Epson RC later) |

## 3. Technology selection

| Approach | Contribution | Training? | Fit for this project |
|---|---|---|---|
| SayCan ([arXiv 2204.01691](https://arxiv.org/abs/2204.01691)) | LLM proposes skills, value functions ground feasibility | yes (affordances) | Idea adopted: **fixed skill library + feasibility from planners**, no learned values |
| Code as Policies ([2209.07753](https://arxiv.org/abs/2209.07753)) | LLM writes robot code | no | Rejected for execution (spec forbids LLM code on the robot); structured JSON tool calls instead |
| VoxPoser ([2307.05973](https://arxiv.org/abs/2307.05973)) | LLM+VLM compose 3-D value maps | no | Useful later for spatial relations; heavy |
| VIMA ([2210.03094](https://arxiv.org/abs/2210.03094)) | multimodal-prompt policy | yes | Not needed: classical skills suffice for blocks |
| RT-2 ([2307.15818](https://arxiv.org/abs/2307.15818)) / OpenVLA ([2406.09246](https://arxiv.org/abs/2406.09246)) | end-to-end VLA | yes (robot data) | Not for the initial system (no VT6L data; 7B+ models exceed 12 GB with the sim) |
| FoundationPose ([2312.08344](https://arxiv.org/abs/2312.08344)) / DOPE ([1809.10790](https://arxiv.org/abs/1809.10790)) | 6-D object pose | FP: no (needs CAD); DOPE: per object | Not needed for cuboids: top-face geometry gives position + yaw; revisit for complex parts |
| cuRobo ([2310.17274](https://arxiv.org/abs/2310.17274)) / Isaac cuMotion | GPU collision-free motion generation | no | **Primary candidate**: the `cumotion` extension ships with Isaac Sim 6.1; needs a VT6L robot config (URDF + collision spheres) |
| MoveIt 2 | standard ROS planning framework | no | Fallback (not installed; Epson provides MoveIt configs only for other models) |
| RMPflow / Lula | reactive motion policies | no | Fallback for local, reactive moves |
| Next-best-view | view selection by information gain | no | Implement a scored view selector over the wrist camera's reachable views |
| Behaviour trees / FSM | deterministic task execution | no | **Hierarchical state machine** (simpler to verify than a BT for one arm) |

Links are from prior knowledge and were not re-checked online in this session. Decision: the LLM
(Gemma, local) only interprets language into a validated JSON task; perception, grasping, motion
and verification are deterministic modules.

## 4. Architecture (spec agents → modules)

```
                 text / voice (Faster-Whisper)                 event log (JSON lines)
                         |                                              ^
                [A] Supervisor: state machine IDLE->UNDERSTANDING->OBSERVING->SEARCHING->LOCALIZING
                    ->PLANNING_GRASP->APPROACHING->GRASPING->VERIFYING_GRASP->TRANSPORTING->PLACING
                    ->VERIFYING_PLACEMENT->COMPLETED | RECOVERING | FAILED | CANCELLED
                 |  language -> task JSON (vlm/, schema-validated)        world model (shared)
   +-------------+--------------+------------------+-------------------+------------------+
 [B] Perception  [C] Exploration  [D] Grasp planner  [E] Motion planner  [F] Execution   [G] Verify/recover
 classical seg.  NBV over wrist   candidates, sym.,  IK (multi-sol.),    ROS sim backend  grip check, lift
 + optional VLM  views, coverage  width, ranking     collision checks,   (later: Epson    check, placement
 depth -> world  (fixed camera:   pre-grasp/grasp/   time-param.         RC backend)      check, bounded
 registry        no fake views)   lift/retreat                                            retries
```
Existing code maps to: B = `perception/`, C = agent `explore()`, D = `control/skills.py` +
`perception/grounding.py: grasp_pose`, E = `control/kinematics.py` + `control/motion.py`,
F = `ros_nodes/vlm_agent_node.py` execution + `sim/ros_control.py`, A = the agent's `run_task`.
The restructure separates these into typed components with a supervisor and an event log.

## 5. Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Audit (this document), unit-test suite, VT6L model from the manual | **done**: 52 offline tests pass (`/usr/bin/python3 -m pytest`) |
| 1 | Robot + scene validation in sim | **passed 2026-10-13** (`scripts/validate_robot_m1.py`, report `out/validation/m1_20261013_192113.json`): 14 poses incl. a +20° step on each joint; our FK vs Isaac Sim worst 0.117 mm / 0.0067° (acceptance 1 mm / 0.1°) for j1–j6, wrist camera, gripper base; table camera extrinsics exact; wrist camera follows J5 not J6; J2 limit enforced. Physics ran headless despite the driver issue; **camera images still need the reboot**. Fixed on the way: static TF frames were on a separate root ("World" vs "world"). Not covered: arm–floor/object contact (self-collision disabled in the articulation). FK agreement shows implementation consistency, not Epson's real parameters |
| 2 | Classical RGB-D perception baseline + visualisation | grounding done (2–7 mm on the table camera, measured 2026-10-07); add colour-segmentation detector |
| 3 | Kinematics: multi-solution IK, collision checking, cuMotion config | IK done; **collision checking done** (`control/collision.py`, swept paths, self-collision with exclusions; 31 tests); **speed enforcement done** (`safety/joint_limits.py` in the sim's command interface + runtime monitor; 7 tests). cuMotion not adopted (own checker is sufficient for now) |
| 4 | Deterministic pick and place, verified grasp/release (no LLM) | **Manipulation baseline (ground-truth poses) COMPLETE and frozen 2026-10-14** (section 6.2). Open: Phase A camera acquisition (needs the reboot) and the camera-driven baseline B |
| 5 | Language → validated task JSON | exists (Gemma); add spatial relations and clarification |
| 6 | Voice (Faster-Whisper) | open |
| 7 | Active perception (NBV) | sweep search exists; NBV scoring open |
| 8 | Supervisor + agents + event log | open |
| 9 | Failure injection and recovery | open |
| 10 | Evaluation A–E, seeded scenarios | open |

## 6. Milestone 4 — Phases B–D with known poses (2026-10-14)

> The numbers in this section and 6.1 come from runs BEFORE the simulation-timing fix (trajectories
> played at half speed); the final, authoritative results are in 6.2.

**Phase B: collision checking** (`control/collision.py`, `tests/test_collision.py`). The robot,
the 2F-85 (palm, pads, linkages; opening-dependent) and a held object are covered by spheres;
obstacles are oriented boxes (test objects, camera stand, microphone); the floor is a plane. Every
planned motion is checked along its whole path at <= 1 deg joint steps (~3 ms per configuration).
Self-collision excludes joined parts, upper arm vs forearm (only J3 moves them relative to each
other; Epson's J3 limits are the fold limits; the simplified boxes touch at J3 -50 deg) and pairs
overlapping in the zero pose. Tested: known-free poses, a deliberate obstacle, floor contact,
three real self-collision poses, a swept path colliding only between free end points, a blocked
approach, and a 70 mm cube at 45 deg that only clears the open pads with the right grip yaw. Found:
the bottom block of a stack cannot be picked (correctly refused). Fallback: a blocked joint move
is re-planned via the home pose.

**Speed enforcement** (`safety/joint_limits.py`, `tests/test_safety.py`), independent of the
planner, inside the simulator's command interface: trajectories outside the joint ranges, faster
than the manual's joint speeds, with bad timestamps or starting away from the arm are rejected
(not executed); direct commands are clamped; measured joint speeds are monitored; all on
`/safety/events`. The Epson controller stays authoritative on a real robot.

**Phases C/D, Baseline 1: known poses** (`experiments/known_pose_pick_place.py`, suite
`experiments/known_pose_suite.py`). Poses from simulator ground truth (`/sim/ground_truth/objects`,
evaluation / this baseline only). Checks are measured, not assumed: grasp contact (fingers stop
before the commanded position), the object rising with the gripper (ground truth), and after
release + 3 s settling: position, height, tilt and drift.

| Case | Result | Placement error |
|---|---|---|
| red cube onto green cube | pass | 1.0 mm |
| brown cube rotated 45° | pass | 0.1 mm |
| yellow cube to a floor spot | pass | 0.7 mm |
| orange cube 35 mm | pass | 0.2 mm |
| lime cube from the top of a stack | pass | 0.1 mm |
| navy cube at the far edge (0.89 m) | pass | 0.6 mm |
| olive cube at the near edge (0.43 m) | pass | 0.2 mm |
| tan box 75 mm wide | pass | 0.4 mm |
| purple cylinder | pass | 1.4 mm |
| steel cube 3 kg (planned via home, around the microphone) | pass | 0.5 mm |
| pink cube with an injected 70 mm pose error | failure caught at the grasp (fingers reached the target: nothing held) | — |

Height error 0.0 mm and tilt 0.0° in every pass. Reports: `out/experiments/`.

**Findings on the way (all fixed):**
1. Closing the 2F-85 swings its fingertips up to 13.7 mm further from the flange; small blocks put
   the tips into the floor. Calibrated from the simulator (`control/robot.py: GRIPPER_CAL_*`) and
   used for the grasp height (tips clear of the supporting surface), the collision model and the
   contact check.
2. The planner assumed every object stands on the floor (wrong height for a stacked block, which
   would also have been released 50 mm high). Object height is now passed through.
3. Driving the simulated 2F-85 fully closed at full force flexes its linkage: the finger angle stops
   meaning the pad gap, and a first suite run reported 5 good grasps as failures (my check, not the
   robot). Now: close to (object width − 8 mm) like a Robotiq position/force command; the fingers
   stopping short of that = contact; the lift is the decisive check.
4. The executor declared steps done the moment they were within 1.5 deg; it now waits for a real
   settle (< 0.1 deg) before the next step (e.g. closing the gripper).
5. With a 3 kg payload the arm settles 0.2–0.4 deg off target and the block slipped ~4 mm in the
   pads during the lift; placement was still within 0.5 mm. Worth tracking for heavier payloads
   (effort limits are placeholders).

**Not yet done:** Phase A (camera images: needs the reboot), Baseline 2 (the same suite with RGB-D
poses), side grasps (top-down only), arm-vs-object contact in PhysX beyond the planner's model.

### 6.1 Follow-up to the review (2026-10-14)

**Scope of all results so far: simulation only.** They show that the software, the simulated VT6L
and the simulated 2F-85 work together; they are not a validation of the physical VT6L (kinematic
parameters, dynamics, payload, gripper behaviour). That needs the real controller, Epson CAD and
the lab cameras.

**Collision checking is now conservative along paths** (`control/collision.py: check_path`).
Conservative advancement: between two configurations no robot point moves further than
B = Σ_j |Δq_j|·R_j, with R_j an upper bound on joint j's lever arm (link offsets + sphere offset +
radius, triangle inequality: lever arms 1.30 / 1.06 / 0.64 / 0.54 / 0.24 / 0.16 m). A segment is
safe when the clearance at its ends is at least B (2B for self-collision) and is otherwise halved,
down to 1 mm. Within the sphere model the swept motion therefore keeps (margin − 1 mm), between
samples too; `tests/test_collision.py` shows a 2 mm wall that sampling misses and this finds. How
well the spheres and simplified shapes match the real robot is a separate (CAD) question.
The excluded self-collision pairs are pinned in a test (`REVIEWED_EXCLUSIONS`) so any change is a
deliberate review; upper arm vs forearm is excluded on an **assumption** (only J3 moves them;
Epson's J3 range taken as the fold limit) to be checked against CAD or the robot.

**Grasp states** (`experiments/known_pose_pick_place.py`), each with its own evidence:
*contact suspected* (fingers stopped before the commanded position; could also be interference),
*object secured* (after the lift the object rose with the TCP and sits at it), *grasp verified*
(after the carry it is still at the TCP; slip measured). Reports carry the state reached and a
failure class: perception / planning / grasp / execution / placement.

**Payload study** (`experiments/payload_study.py`): light and 3 kg blocks, speeds ×0.5 / ×1 / ×2,
3 repeats each; tracking error during motion, settling, slip in lift and carry, finger drift,
placement. Results below (simulated dynamics with unverified inertia / effort limits).

**Baseline B (camera poses), prepared, to run after the reboot:**
1. `scripts/validate_cameras_phaseA.py`: both cameras' frames (sizes, RGB/depth sync, freshness),
   intrinsics vs the D435 model, extrinsics vs TF and FK (1 mm / 0.1°), depth sanity (floor reads
   z = 0), and the classical detector scored against ground truth for every object (found /
   missed, position and yaw error).
2. `experiments/known_pose_suite.py --poses camera`: the same 11 scenarios with poses from
   `perception/classical_detector.py` (depth geometry + colour + the object catalogue; stacks
   peeled into separate blocks; min-area-rectangle centres robust to partial occlusion). Reports
   the perception error and the manipulation result per case, plus the placement error against
   the TRUE destination (a biased estimate can still place "where it believed").
   Offline test on a synthetic table-camera frame: 8/8 objects found and named, 0.9–2.8 mm,
   yaw within 2° (`tests/test_classical_detector.py`); synthetic frames have no lighting, shadows
   or depth noise, so the simulator run is the real test.
Ground truth stays evaluation-only: the camera pose source never reads it.

**Evaluation framework (A–E), kept for later milestones:** A ground-truth geometric baseline
(done), B RGB-D geometric baseline (prepared), C language task planner on the verified skills,
D + active perception (wrist-camera search), E + verification and recovery.

### 6.2 Final validation at corrected simulation speed — Milestone 4 manipulation baseline COMPLETE (2026-10-14)

**Timing bug fixed before these runs:** each app update runs two 120 Hz physics steps, but the
simulator's command interface advanced by one, so every trajectory had played back at half the
planned speed since the project began (final poses were always correct; the reported "tracking
errors" of 15–130° in the first payload run were this mismatch). `sim/run_live.py` now advances by
the measured simulation time. All three validations were re-run after the fix:

| Validation | Result |
|---|---|
| Milestone 1 (`scripts/validate_robot_m1.py`) | **PASSED**: worst FK vs simulator 0.109 mm / 0.0061° (target 1 mm / 0.1°); table camera exact; wrist camera follows J5 only; J2 limit enforced |
| Suite, ground-truth poses (`experiments/known_pose_suite.py`) | **11/11 as expected**: 10 picks reached *grasp verified*, placement error 0.1–2.0 mm, carry slip ≤ 0.7 mm; injected 70 mm error caught (fingers closed on nothing) |
| Payload (`experiments/payload_study.py`), 18 runs | **18/18 placed**; see table |

| Block | Speed | OK | Max tracking error | Settle offset | Slip in lift | Slip in carry | Placement |
|---|---|---|---|---|---|---|---|
| red 0.1 kg | ×0.5 / ×1 / ×2 | 3/3 each | 0.7 / 1.4 / 2.6° | 0.37–0.39° | 1.0–1.3 mm | ≤ 0.2 mm | 0.7–1.0 mm |
| steel 3 kg | ×0.5 / ×1 / ×2 | 3/3 each | 0.9 / 1.7 / 3.0° | 0.30–0.31° | 2.7–2.9 mm | 0.1 / 0.6 / 1.0 mm | 0.9–1.3 mm |

Tracking error grows with speed as expected; slip in the carry grows with speed for the 3 kg
block (up to ~1 mm mean); placement stays within 1.3 mm. Simulated dynamics only (inertia and
effort limits are not manufacturer values).

**Known simulation limitation — J5 offset while holding:** with a block in the gripper, J5 settles
~0.3–0.4° off target (other joints < 0.13°; 0.02° without a block), ≈ 1 mm at the TCP, with any
block mass. Most likely the simulated 2F-85's internal squeeze forces, which in this scene act on
the arm's articulation (the knuckle axes are parallel to J5) and exceed J5's placeholder effort
limit; a real gripper's internal forces stay in the gripper. It caused no grasp failure, unsafe
motion or placement outside tolerance, so it is documented, not optimised. The executor treats a
standstill as settled and reports the offset.

**Frozen baseline:** `docs/baselines/manipulation_A_2026-10-14/` holds the three reports, the
grasp-feasibility map, the offline test result (104 passed) and `MANIFEST.sha256` of the robot
model and every file the results depend on. A change to any of them means re-running the three
validations before comparing new results with this baseline.

## 7. Open items and next steps (in order)

1. **Reboot** (NVIDIA driver/library mismatch blocks camera rendering), then **Phase A**:
   `scripts/validate_cameras_phaseA.py` (frames, sync, intrinsics, extrinsics, depth sanity,
   detector scored against ground truth).
2. **RGB-D perception**: the classical detector on real simulator frames; fix what Phase A shows.
3. **YOLO detection** (not installed yet; needs a model and a decision on pretrained vs trained on
   simulator images) as an alternative / complement to the classical detector.
4. **VLM semantic labelling** (Gemma) of detected objects for language references.
5. **Camera-driven pick and place**: `experiments/known_pose_suite.py --poses camera` (Baseline B),
   compared case by case with the frozen Baseline A.
Hardware-side items (not blocking simulation work): Epson CAD for link shapes and inertia; J1/J4/J6
positive directions on a controller; the lab camera arrangement; the upper-arm/forearm exclusion
assumption.


## 8. 2026-10-14 — Perception redesign and multi-agent architecture (dated entry)

Active work is tracked in `docs/perception_redesign/PROGRESS.md` (read first), decisions in
`docs/perception_redesign/DECISIONS.md`, the design in `docs/perception_redesign/ARCHITECTURE.md`.
- **Reversal of section 3's decision** that the language model only parses language: the VLM now
  also describes detected objects and resolves references, under strict JSON validation; geometry,
  motion and verification stay deterministic (DECISIONS.md, 2026-10-14).
- No classical detector in the new system; class-agnostic learned segmentation trained on synthetic
  data; LangGraph supervisor with seven agents; sensor-only verification (ground truth for
  evaluation only).
- Milestone table update: Milestone 2 (perception) is being rebuilt on the new design — Phase 1
  (synthetic dataset) in progress (train 0-3999 of 6000 rendered, measured 07:57 UTC); Milestones 5,
  7, 8, 9, 10 are re-planned in ARCHITECTURE.md §12 (estimates). Frozen manipulation baseline A
  unchanged (manifest verified 2026-10-14).
