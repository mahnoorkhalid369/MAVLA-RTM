# Epson VT6L — Isaac Sim 6.1 simulation

## Current layout (restructured 2026-10-17; automated verification passed 2026-10-18, manual acceptance pending)

| Folder | What is there |
|---|---|
| `agent/` | the current agent: entry point `run.py`, `session.py`, `supervisor.py`, `capabilities/`, `tools/`, `vlm_roles/` (Gemma), `controls.py` |
| `perception/` | YOLO detector, depth geometry, mark images, world model, wrist-camera viewpoints (+ shared / legacy helpers) |
| `config/` | thresholds, tolerances, budgets, model paths |
| `scenarios/` | scenario library (laboratory) and JSON test scenes; offline design checks |
| `sim/` | Isaac Sim processes (`run_live.py` default scene, `run_scenario.py` test scenes) |
| `control/`, `execution/`, `safety/` | frozen manipulation core (kinematics, planning, executor, joint limits) |
| `evaluation/`, `experiments/` | experiment runners, scorers, studies; frozen baseline experiments |
| `tests/` | offline tests (`tests/support/` shared fakes) |

Start with `docs/architecture/MODULE_GUIDE.md` (where to find X), `docs/architecture/EXECUTION_FLOW.md` and
`docs/DEVELOPER_GUIDE.md` (commands and Python environments). The sections below describe the earlier,
legacy setup (`launch/run.sh`, `vlm/`, `ros_nodes/`), which is kept but not used by the current agent.

## Code layout
Each folder has one job and runs in one Python environment.

| Folder | Job | Runs with |
|---|---|---|
| `assets/` | Robot description (URDF + meshes), imported USD, Robotiq 2F-85 asset | data |
| `sim/` | Everything inside Isaac Sim: scene, ROS 2 bridge, live launcher, URDF import | `.venv` Python |
| `control/` | Robot maths (FK / IK), no simulator imports | any Python with numpy |
| `experiments/` | Scripted demos with hard-coded task plans | `.venv` Python |
| `vlm/` | Vision-language model (Gemma 4 E2B-it): loading, prompts, plan parsing, task planner | `.venv-vlm` Python |
| `perception/` | Turn model boxes into 3-D positions with the depth image | any Python with numpy |
| `ros_nodes/` | Standalone ROS 2 nodes: camera viewer, microphone, command interpreter, VLM agent | system Python + ROS 2 Jazzy (agent: `.venv-vlm`) |
| `models/` | Downloaded model weights (`gemma-4-E2B-it`, 10 GB) | data |
| `launch/` | Starts the live setup (sim + nodes) in one command | bash |

```
sim/scene.py          world: arm, 2F-85, D435 camera prim, microphone prop, blocks, materials
sim/ros_bridge.py     ROS 2 publishers (OmniGraph): camera RGB/depth/info, /tf, /tf_static, /clock
sim/run_live.py       live mode: scene + ROS 2 streams, arm moves only on command
sim/ros_control.py    /joint_trajectory (played back on sim time) + /joint_command in, /joint_states out
sim/import_vt6l.py    URDF -> USD (rerun after editing the URDF)
control/robot.py      shared robot constants: joint names, home pose, gripper values, speeds
control/kinematics.py FK + damped least-squares IK (multi-seed ik_best) read from the URDF
control/motion.py     coordinated motion planning: synchronised joint moves, straight-line TCP moves
control/instructions.py  text instruction -> joint / TCP goal (no ROS; `python3 -m control.instructions`)
control/skills.py     pick / place / home -> planned motion steps (`python3 -m control.skills`)
vlm/config.py         model path, 4bit/8bit/bf16, generation settings
vlm/model.py          Gemma wrapper: generate(text, image, audio)
vlm/prompts.py        system prompt + JSON plan format (edit to change the model's instructions)
vlm/plan.py           parse/validate the model's JSON plan (`python3 -m vlm.plan`)
vlm/planner.py        camera frame + typed or spoken task -> TaskPlan (retries once on bad JSON)
perception/grounding.py  box (0-1000) + depth + camera pose (FK) -> 3-D object position
experiments/trajectory_demo.py   joint-waypoint loop (tracking-error check)
experiments/pick_place.py        pick & place of the red block (--block-mass payload tests)
ros_nodes/camera_viewer.py       live RGB | depth window
ros_nodes/microphone_node.py     host microphone -> /microphone/*
ros_nodes/command_interpreter.py /instruction text -> plan -> /joint_trajectory (+ /instruction/feedback)
ros_nodes/text_command_node.py   terminal prompt: type instructions -> /instruction (or --topic task)
ros_nodes/vlm_agent_node.py      VLM pick-and-place agent: /task text or speech + camera -> motions
launch/start_live.sh             sim + viewer + interpreter in one go (--voice adds the microphone)
launch/start_vlm_agent.sh        the VLM agent (next to a running live sim)
```

## Setup (already done)
- Python 3.12 venv in `.venv/` (created with `uv`, installed in `~/.local/bin`)
- `isaacsim[all,extscache]==6.1.0.0` from https://pypi.nvidia.com
- ROS 2 Jazzy in `/opt/ros/jazzy` (system install)
- VLM environment `.venv-vlm/` (torch 2.11 cu130, transformers 5.19, bitsandbytes) and the model
  `google/gemma-4-E2B-it` (Apache-2.0, not gated) in `models/gemma-4-E2B-it/`

## Quick start (one command)
```bash
./launch/run.sh          # add --headless to skip the Isaac Sim window
```
It starts the simulation (Isaac Sim window), opens ONE window with both cameras (table camera on top,
wrist below; RGB | depth), loads the agent and then takes tasks typed in the same terminal, printing
its logs there in order:
```
forget
overview
put the yellow cube on the green cube
quit
```
The sections below describe the parts separately (start_live.sh / start_vlm_agent.sh / tasks.sh).

## Live mode: camera, microphone, text control
```bash
./launch/start_live.sh              # Isaac Sim + camera viewer + command interpreter (--voice: + microphone)
./launch/start_live.sh --headless   # same, without the Isaac Sim window
```
Then, in a second terminal, type instructions for the arm:
```bash
cd ~/Documents/PhD-CE/Research-LF/6DOF-Arm
source /opt/ros/jazzy/setup.bash
python3 ros_nodes/text_command_node.py
```
```
vt6l> help
vt6l> look left 30        # J1 +30 (camera view turns left)
vt6l> look up 20          # J5 +20
vt6l> j5 -60              # absolute angle in degrees;  j1 by 20 = relative
vt6l> j1 30 j5 -45        # several joints at once;     pose 0 -20 20 0 -90 0 = all six
vt6l> move down 100        # gripper (TCP) straight line, mm; up/down/left/right/forward/back
vt6l> move to 300 400 300  # straight line to a point (mm, base frame: +X right, +Y forward, +Z up)
vt6l> goto 0 500 400 down  # joint move to a point; 'down' = tool pointing straight down
vt6l> point down           # turn the tool straight down in place
vt6l> close / open / gripper 50%
vt6l> speed 45             # joint speed limit, deg/s (default 30)
vt6l> tcp speed 120        # straight-line speed, mm/s (default 80)
vt6l> status / stop / home # status also shows the TCP position from forward kinematics
```

### How a command becomes coordinated motion
```
text_command_node -> /instruction -> command_interpreter -> /joint_trajectory -> simulator -> /joint_states
                                     |  control/instructions.py  text -> goal (joints or TCP pose)
                                     |  control/motion.py        goal -> time-stamped joint path
                                     |  control/kinematics.py    FK (q -> pose), IK (pose -> q)
```
- **Joint goals** (`j5 -60`, `pose …`, `home`, `look …`): synchronised joint move. Every joint
  starts and stops together on a quintic (smooth start/stop) profile; the joint with the
  furthest to go at the speed limit sets the duration.
- **Straight lines** (`move …`, `move to …`, `point down`): the TCP path is sampled every
  1/30 s, IK is solved at each sample (warm-started), and the plan is rejected before any
  motion if a point is out of reach, beyond a joint limit, or needs a joint flip (singularity).
- **`goto`**: IK for the end pose only (several start seeds, closest solution), then a
  synchronised joint move — the TCP path curves, but it is the fastest way there.
- The simulator plays the joint path back against simulation time; the gripper still uses
  `/joint_command`. A new command replaces the running trajectory; `stop` holds in place.
- Measured in simulation: straight lines stay within 0.2 mm of the ideal line, and all joints
  of a joint move arrive together.

Targets are clamped to the VT6L joint limits, but nothing checks for collisions yet
(e.g. `j2 -160` drives the arm into the floor). A speech-to-text node can later publish
on `/instruction` without changing anything downstream.

The first launch takes 1–2 min. Ctrl-C stops everything; logs go to `out/logs/`.
Camera viewer keys: `q` quit, `s` save a snapshot to `out/snapshots/`; hover for depth.

ROS 2 interface (all from a shell with `source /opt/ros/jazzy/setup.bash`):

| Topic | Type | Notes |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | rgb8, 1280×720 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | fx = fy = 924.3 px (69.4° HFOV) |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 32FC1, metres along the optical axis |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | same intrinsics as colour |
| `/table_camera/color/image_raw`, `/table_camera/depth/image_rect_raw`, `/table_camera/*/camera_info` | `sensor_msgs/Image` / `CameraInfo` | fixed table camera, same format as the wrist one |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | World → arm and gripper links; camera optical frame; `microphone_link` |
| `/clock` | `rosgraph_msgs/Clock` | simulation time |
| `/microphone/audio` | `std_msgs/Int16MultiArray` | mono PCM 16 kHz, 100 ms chunks |
| `/microphone/level_db` | `std_msgs/Float32` | loudness, dBFS |
| `/microphone/speech_active` | `std_msgs/Bool` | level above `--threshold-db` |
| `/instruction` | `std_msgs/String` | text instruction (typed now; speech-to-text later) |
| `/instruction/feedback` | `std_msgs/String` | interpreter's reply |
| `/joint_trajectory` | `trajectory_msgs/JointTrajectory` | planned arm path (J1–J6), 30 Hz samples |
| `/joint_command` | `sensor_msgs/JointState` | direct targets (rad), used for the gripper; `velocity` = max speed (rad/s) |
| `/joint_states` | `sensor_msgs/JointState` | J1–J6 + `finger_joint`, 30 Hz |

Other viewers: `ros2 run rqt_image_view rqt_image_view`, or `rviz2` (fixed frame `World`).

### Microphone
Isaac Sim does not simulate sound. The scene has a desk microphone prop at
`sim/scene.py: MIC_POSITION` (front-left of the arm) with a `microphone_link` frame on
`/tf_static`; the audio itself comes from the computer's real microphone through
`ros_nodes/microphone_node.py` (`arecord`, ALSA device `default`; pick another with
`--device`, list them with `arecord -L`).

Speech is detected relative to the room's background noise (12 dB above the level of the last
10 s; `--threshold-db` sets a fixed level instead). The node warns when the input is clipping:
on this laptop the internal mic ships at maximum gain (+60 dB in total) and must be lowered:
```bash
amixer -c 1 sset 'Internal Mic Boost' 1
amixer -c 1 sset Capture 50%
```
With `--voice`, the agent transcribes each utterance first (shows `you said: "..."`), ignores
noise and sounds over 12 s without a pause, ignores speech while a task runs, then plans the
transcript exactly like a typed task. To stop a running task, type `stop`.

## VLM pick-and-place agent (Gemma 4 E2B-it)
Three terminals:
```bash
./launch/start_live.sh       # 1. simulation + wrist and table camera windows + interpreter
./launch/start_vlm_agent.sh  # 2. the agent (add --voice for spoken tasks)
./launch/tasks.sh            # 3. type tasks
```
```
task> put the red block on the green block
task> pick up the yellow block and put it on the table to its left
task> how many blocks can you see?          # questions get an answer, no motion
task> stop                                   # cancel after the current motion
```
The task prompt also takes direct commands (`move left 100`, `look up 20`, `turn around`, `observe`,
`open`, ...) and runs them without the model. Other agent commands: `scan` (map mid + far views all
around), `explore` (near + mid + far views everywhere), `memory` (objects with x, y, theta, and where
it looked without finding something), `forget` (clear memory: do this after restarting the sim).

Finding objects: every pick logs `target <object>: x, y, theta | distance, bearing`. If a named
object is not in view or memory, the agent records "not found here" and asks where to look; answer
`left` / `right` / `front` / `behind` (+ `near` / `far`) or `search`. It then explores that sector
first, then everything else (near, mid, far views at 45 deg steps), skipping views already searched,
and resumes the task when it finds the object.

Table camera (eye-to-hand): every task starts with a `survey` of the fixed camera on its stand,
which sees the whole field at once. Each object goes into memory and is logged with a holistic
direction, e.g. `brown cube: x 500 mm, y -250 mm, theta +44 deg | right, bearing -117 deg, 559 mm from the
base, reachable`. The model plans from this image; the wrist camera (eye-in-hand) is used for the close-up
check before grasping. `overview` (or `survey`, `look around`) runs a survey without a task. Objects
hidden behind the arm (e.g. the black cube at the home pose) are covered by memory, the wrist camera
and exploration.

How a task flows (each box is its own module, only the agent node uses ROS):
```
task text / speech + camera frame + depth + joint angles   (one snapshot)
  -> vlm/planner.py      Gemma: JSON plan [pick object (box), place target (box), home]
  -> vlm/plan.py         validate (actions, box ranges, pick-before-place)
  -> perception/grounding.py   box + depth + camera pose (FK) -> 3-D top-centre of each object
  -> control/skills.py   approach / descend / grip / lift / carry / lower / release / retreat,
                         every motion planned with control/motion.py before anything moves
  -> ros_nodes/vlm_agent_node.py  executes: /joint_trajectory + /joint_command, waits on /clock
```
Outputs: progress on `/task/feedback`; `/task/annotated` (camera frame with the model's boxes;
view with `ros2 run rqt_image_view rqt_image_view`); each task's image + raw model answer in
`out/tasks/`.

Model settings (`vlm/config.py`): 4-bit weights; the 4.7 GB per-layer embedding table lives in
system RAM, so the agent holds ~2.3 GB of GPU memory next to Isaac Sim (~2.5 GB). Loading
briefly needs ~6.5 GB. `--quantization 8bit|none` and `--thinking` are available on the agent.

Measured in simulation ("put the red block on the green block", observe pose): model plan in
22 s with two simulators sharing the GPU (~5 s on its own); depth grounding within 2 mm (red)
and 7 mm (green) of the true block positions; red block ended stacked on green.

## Experiments (scripted plans)
```bash
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES   # accepts the NVIDIA Omniverse EULA
python -m sim.import_vt6l                              # after URDF edits
python -m experiments.trajectory_demo                  # joint-waypoint loop
python -m experiments.trajectory_demo --headless --test
python -m experiments.pick_place                       # pick & place
python -m experiments.pick_place --headless --test     # prints SUCCESS / FAILED
python -m experiments.pick_place --block-mass 3.0      # payload test
python -m control.kinematics                           # IK self-test
```
Run these from the project folder (they are Python modules).

## Safety, collision checking and verified pick and place (Milestone 4)
- `control/collision.py`: every planned motion is collision-checked along its path (robot, 2F-85,
  held object vs floor, obstacles, itself). Used by `control/skills.py` when given a checker.
- `safety/joint_limits.py`: the simulator rejects trajectories outside the VT6L joint ranges or
  faster than its joint speeds, clamps direct commands and monitors measured speeds
  (`/safety/events`). Independent of the planner.
- `execution/ros_executor.py`: runs planned steps on the simulator and reports what happened.
- Known-pose baseline (object poses from simulator ground truth, verified physically):
  ```bash
  ./launch/run.sh --headless          # fresh sim
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 experiments/known_pose_pick_place.py --object red_cube --onto green_cube
  /usr/bin/python3 experiments/known_pose_suite.py      # 11 cases, table of results
  ```
- Grasp-feasibility map: `/usr/bin/python3 evaluation/grasp_feasibility_map.py` (~9 min, 24 workers).
- Robot validation against the simulator: `scripts/validate_robot_m1.py`. Tests: `/usr/bin/python3 -m pytest`.
- `/sim/ground_truth/objects` is for evaluation only; perception must never read it.
Details and results: `docs/AUDIT_AND_PLAN.md`, section 6.

## Robot model
`assets/vt6l/vt6l.urdf` is the **Epson VT6L built from Epson's VT manipulator manual**
(kinematics from the dimension drawings, joint ranges and max. speeds from the specification
table; sources in the URDF header). `tests/unit/test_vt6l_model.py` checks it against the manual's
reach (920 / 1000 mm) and motion-range radii. Still approximate: link shapes (boxes/cylinders),
inertia (total 40 kg), effort limits (simulation-only), J1/J4/J6 positive directions.
Convert it for Isaac Sim with `python -m sim.import_vt6l` (writes `assets/vt6l/usd/vt6l/`).
The earlier proxy (`vt6l_proxy.urdf`: Epson C8L kinematics + meshes) is kept for comparison.
Full audit and plan: `docs/AUDIT_AND_PLAN.md`. Offline tests: `/usr/bin/python3 -m pytest`.

The imported USD stores physics in a `Physics` variant set; `sim/scene.py` selects `physx`.

## Wrist camera (RealSense D435-style)
- Mounted on a bracket on the SIDE of the J5 wrist housing (`camera_link` in the URDF;
  the housing top has no mounting surface on the real VT6L), beside the flange and outside
  the J4 fork, toed in 30 deg so it looks into the gap between the 2F-85 finger pads
  (TCP ~155 mm in front of the camera, near image centre). It pitches with J5 but does
  not spin with J6; grasps are planned with J6 ~ 0 (`R_DOWN`) to keep that view.
- ROS frames: `camera_link` (x fwd, z up) and `camera_color_optical_frame` (z fwd, y down).
- Simulated as RGB + depth at 1280x720, 69.4 deg HFOV (D435 colour stream), clip 0.1-10 m.
- Experiments' `--save-images`: `rgb_XXXX.png` and `depth_XXXX.png` (uint16 mm, 0 = no data).

## Table camera (eye-to-hand)
- A D435 on a stand at the robot's right: `control/robot.py: TABLE_CAMERA_POSITION` (1.5, 0.25, 1.8) m,
  looking at `TABLE_CAMERA_TARGET` (0, 0.25, 0), ~50 deg down. Chosen so every test object is in view
  and none is behind the robot's base; the moving arm can still hide some, as in a real cell.
- USD: `/World/TableCamera` (camera, D435 body, pole, foot; visual only). Pose for grounding:
  `table_camera_pose()`; frame `table_camera_optical_frame` on `/tf_static`.
- Measured with the test objects: positions within 2-7 mm, rotated cube theta 44 deg.
- Colour calibration: through this camera the scene light makes colours brighter and paler, so
  `perception/grounding.py: CALIBRATED_COLOURS` holds how each named colour actually looks
  (re-measure them if you change the camera or the lights). Objects use matte materials
  (`sim/scene.py: _matte_material`) so glare does not wash out colours.
- Grounding ignores points above 0.25 m (`MAX_OBJECT_Z`): from the side, the arm's links sit
  right next to the objects in the image.
- Limit: a stacked pair is measured as one blob, so its colour is a mix (lime on blue reads green).

## Gripper: Robotiq 2F-85
- Isaac Sim's Robotiq 2F-85 asset (CC-BY 4.0), downloaded to `assets/robotiq_2f85/`
  ("mimic" physics variant: one driven `finger_joint`, 0 = open 85 mm, ~0.8 rad = closed).
- Rotated 90 deg on the flange (`GRIPPER_MOUNT_YAW_DEG`) so the fingers close
  left/right in the wrist camera image.
- Mounted by `sim/scene.py: attach_gripper()`: the gripper's own articulation root is removed
  and its `base_link` is fixed to `j6_link` at the flange, so arm + gripper are one articulation.
  No coupling plate is modelled (a real non-UR mount adds ~10 mm).
- `tcp` (URDF) = the 2F-85 `grip_frame`, 133.7 mm from the flange; used by the IK.
- Finger pads and blocks share a rubber-like physics material (friction 1.2 / 1.0).
- 12 blocks around the arm (`sim/scene.py: BLOCKS`): small 35-45 mm and medium 50-60 mm, 12 colours
  (red, green, blue, yellow, orange, purple, pink, brown, teal, black, white, gray), 0.5-0.75 m
  from the base, all pickable from above. Mass from size at wood density (60 mm = 0.1 kg);
  `--block-mass` overrides the picked one.

## Payload realism (not yet enforced)
- VT6L: 6 kg max / 3 kg rated payload. 2F-85: ~0.9 kg, 20-235 N grip, 5 kg rated.
  Largest realistic object: ~6 - 0.9 - 0.2 (camera) = 4.9 kg.
- The sim currently succeeds even at 6 kg: arm joint torque limits are placeholders
  (URDF `effort`), and the 2F-85 drive / pad friction are not calibrated to Robotiq's
  grip-force spec.
