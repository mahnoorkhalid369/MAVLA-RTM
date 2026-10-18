# Scenario library (Isaac Sim benchmark workstations)

Branch `feature/scenario-library`, rollback tag `before-scenario-library`. Scope: scene creation only. The
supervisor, the workers, the VLM, the planners, the skills, the robot, the cameras, `control/`, `execution/`,
`safety/` and the frozen `sim/scene.py`, `sim/scene_objects.py` and `sim/run_live.py` are unchanged.

| # | Scenario id | Status |
|---|---|---|
| 1 | `laboratory`: biomedical sample handling | built, validated (easy / medium / hard) |
| 2 | `warehouse`: order fulfillment | planned (Checkpoint 3) |
| 3 | `electronics`: assembly preparation | planned |
| 4 | `sorting`: contamination-aware material sorting | planned |
| 5 | `fragile`: fragile-object handling | planned |
| 6 | `maintenance`: inspection and tool organization | planned |

## All scenarios live in `scenarios/` (plug-in registry)

`scenarios.available()` lists every selectable scene. Each one is chosen the same way:

- **agent:** `SCENARIO = "<name>"` in `agent/settings.py`;
- **simulator only:** `./launch/sim.sh --scenario <name>`.

Every scene starts through `sim/run_scenario.py`.

| Name | Kind | Defined in |
|---|---|---|
| `default` | frozen | `scenarios/default.py`, which refers to the frozen cube scene of the manipulation baseline (`sim/scene_objects.py`, built by frozen `sim/scene.py`); referenced, not copied, so the checksummed baseline stays the single definition |
| `laboratory` | procedural | `scenarios/laboratory.py` (`build(difficulty, seed)`) |
| `ap_occlusion`, `ap_categories` | json | `scenarios/scenes/<name>.json` |

**To plug in a new scenario, add one file. Nothing else changes.**

- **Procedural:** `scenarios/<name>.py` defining `build(difficulty, seed) -> scenarios.core.Scenario`. Pure data, no
  Isaac imports; use `scenarios/parts.py` and see `laboratory.py`. The registry finds it automatically. The
  infrastructure modules `core`, `parts`, `build` and `checks` are not scenarios.
- **Fixed:** `scenarios/scenes/<name>.json` with `{"name", "description", "objects": [{"name", "shape", "xy", "size",
  "color", ...}]}`, as in `ap_occlusion.json`.

The agent never imports `scenarios/`: it sees only the cameras, the joint states and the objects it measures.

**What every scenario shares.** Every scenario uses the same work cell: the robot, the table camera on its stand and
the microphone prop. That cell is described by `config/perception.py` and frozen `control/robot.py`. A scenario with a
different cell (another camera pose, robot position or support height) would also need those values changed.

## Loading a scene

```bash
# simulator only (ROS 2 sensors, /joint_command, ground-truth topic), as for any other scene
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
.venv/bin/python -m sim.run_scenario --scenario laboratory --difficulty medium --seed 42 [--headless]

# the agent then attaches to the running simulator (it never reads the scenario metadata)
.venv-seg/bin/python -u -m agent.run --live-dir out/demo/live

# the existing JSON scenes and the default scene are unchanged
.venv/bin/python -m sim.run_scenario --scenario ap_occlusion
.venv/bin/python -m sim.run_live
```

To change the scene, restart the simulator process with new arguments. The agent code needs no change.
`agent.run --start-sim --scenario NAME` still passes only the scene name (the default difficulty is `easy`, seed 0).
Passing difficulty and seed through `agent/run.py` was left out on purpose: this task changes no agent
code.

## Validation

```bash
.venv/bin/python evaluation/validate_scenarios.py --scenario laboratory --seed 42 --reload --drop-mm 10   # headless, no ROS
/usr/bin/python3 -m pytest tests/unit/test_scenario_library.py                                    # offline, no simulator
```

Results go to `out/scenarios/<scenario>/<difficulty>_s<seed>/`:

- `table.png` and `wrist.png`, with `*_depth.png`
- `wrist_<fixture>.png`: wrist-camera views of the scene's inspection targets (e.g. `wrist_incoming_rack.png`)
- `overview.png`
- `metadata.json`: the ground truth, with spawn and settled poses
- `validation.json`

### What is checked

**In the simulator (`evaluation/validate_scenarios.py`)**

- Robot and gripper articulation are present: 6 arm joints plus the finger joint. The arm is held at HOME.
- Both robot cameras render RGB and depth. An extra documentation camera (`overview`) is used only for screenshots.
- Every object spawns as a rigid body.
- After a configurable settling time (default 3 s simulated):
  - displacement from the design rest pose ≤ 5 mm;
  - tilt ≤ 3°;
  - drift ≤ 0.5 mm over one more simulated second (PhysX's reported resting-contact velocity is recorded,
    not judged);
  - nothing below the floor;
  - no interpenetration of the settled colliders (> 1.5 mm).
- `--reload` rebuilds the same scenario, difficulty and seed on a new stage in the same process and compares
  the settled poses (≤ 2 mm).
- `--drop-mm D` spawns every object D mm above its support. The objects must land back on their supports. The
  first pose is read after physics has started, so the start of the fall is not observed.
- Stages: physics (GPU, as the live simulator), reload (GPU), drop (GPU), and cameras (a separate CPU-physics
  build, run first). Headless without ROS, GPU physics leaves the wrist camera prim at its zero-joint USD pose;
  see [LABORATORY.md](LABORATORY.md) §6.
- Each stage prints progress and elapsed time, plus a 60 s heartbeat. Past 600 s it dumps the Python stack and
  exits. `--resume` reuses saved stages (`out/.../stages/*.json`).
- Wrist inspection images come from viewpoints computed by the existing, unmodified `ViewpointPlanner`,
  collision-checked against the scene's colliders. The joint state is set directly; no motion is commanded and
  no agent runs. An image is saved only if the rendered camera is within 5 mm of the arm's FK.

**Offline (`scenarios/checks.py`), using the project's own planner and collision checker, unchanged.**
The validator runs these as a separate plain-Python subprocess, in parallel with the builds, with a time limit
(`python -m scenarios.checks ...`).

- Spawn interpenetration of colliders.
- The frozen props (microphone, camera stand) stay clear.
- Decor stays ≥ 1.15 m from the base, beyond the arm's reach.
- Only objects marked `unreachable` lie beyond 0.90 m.
- Table-camera visibility of each object's top face, by ray casting against all colliders and the robot at HOME.
- Grasp feasibility: a collision-checked top-down pick from HOME at four grip yaws. Boxes also try their
  face-aligned yaws.
- Placement feasibility: a held probe container (tube 34 × 114 mm or vial 26 × 60 mm) lowered into each free
  receiving region.
- Wrist-viewpoint feasibility for inspection regions and occluded objects.
- Intent rules: only `occluded` objects may be invisible, only `unreachable` objects may lack a grasp, and every
  free receiving region must accept its container.

These are planning-feasibility checks, not executed manipulation.

## Architecture

| File | Role |
|---|---|
| `scenarios/core.py` | Data model and geometry. Pure Python and numpy, no Isaac Sim. Defines `Scenario`, `Obj`, `Fixture`, `Part`, `Label`, `Region`, `Light`, the deterministic `Placer`, the interpenetration test and `metadata()`. |
| `scenarios/parts.py` | Reusable procedural fixtures and objects: `rack` (slot labels), `tray` (striped rims), `plinth`, `surface`, `zone_outline`, `panel`, `sign`, `shelf_unit`, `capped_container`, `box_obj`. |
| `scenarios/<scenario>.py` | One scenario: `build(difficulty, seed) -> Scenario`. Difficulty is a `CONFIG` table (counts, occluders, look-alikes, occupied destinations, unreachable items), not separate scene code. |
| `scenarios/build.py` | Isaac side: `install()` and `populate()` (see below). |
| `sim/run_scenario.py` | Adds `--difficulty` and `--seed` for library scenes. The JSON scenes run exactly as before. |

What `build.py` does:

- `install()` fills the frozen `OBJECTS` dict with placeholders, so `run_live`'s ground-truth publisher covers
  every object.
- `populate()`, run after the frozen `setup_scene`, does the following:
  - replaces each placeholder by a compound rigid body at `/World/Objects/<id>`;
  - builds static fixtures (colliders, no rigid body);
  - adds visual-only decor;
  - adds textured text labels (PIL to PNG, `UsdUVTexture`);
  - adds lights.

### Conventions

- Units are metres. The world is the robot base frame: +y is the arm's front, +x its right, z up. The floor is at
  z = 0, as in `sim/scene_objects.py`.
- Objects rest on the physics floor or on fixtures. The bench and floor colours are visual-only slabs
  ≤ 0.4 mm thick, so measured heights are unchanged.
- Workspace: reliable top-down picks between 0.45 and 0.90 m from the base. The gripper grips at most 80 mm.
- Spacing comes from the gripper, as measured with the planner. The open pads reach about 75 mm from the
  object centre along the closing direction. At yaw 0 the fingers close along world y. A held object needs
  10 mm of clearance from obstacles.

### Ground truth

`Scenario.metadata()` holds, for each scenario:

- id, version and library version;
- seed and difficulty;
- per object: id, category, role, support, spawn pose, footprint and height, distance from the base, mass,
  friction, attributes (cap colour, sample code, slot…), and its colliders and labels;
- fixtures with their parts;
- regions: slot and tray definitions, with occupancy;
- decor and lights.

The validator adds the settled poses. It is written only under `out/`. The agent never receives it, and the
simulator's `/sim/ground_truth/objects` topic stays evaluation-only, as before.

## Scenario 1: biomedical laboratory sample handling (`laboratory`, v2.0)

Six functional zones (incoming, processing, storage, inspection, quarantine, buffer), 8 / 10 / 12 sample
containers. Full description, results and screenshots: [LABORATORY.md](LABORATORY.md).
