# Demo: typed instruction → perception → planning → verified pick-and-place

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

First thin slice of `docs/perception_redesign/ARCHITECTURE.md` (rev 3). Results and readiness:
`docs/perception_redesign/DEMO_REPORT.md` (read its first lines first).

**Status (2026-10-14, measured in the final rehearsal, 3 runs each, fresh sim per run):**
- "Put the red cube on the green cube." 3/3 verified successes (2.0–3.1 mm vs ground truth).
- "Put the orange cube on the purple cylinder." 3/3 verified successes (3.7–3.9 mm).
- "Pick up the yellow cube and put it on the tan box." 0/3 in the rehearsal: the agent asks which yellow
  cube (an olive and a gold cube also look yellow). In the rehearsal a script answered "cancel"; in the
  interactive demo you now answer the question yourself (see "Questions" below).
- The "next to" instruction was not rehearsed.

```
"put the red cube on the green cube"
  → LangGraph supervisor (agent/graph.py)
  → table camera RGB-D → fine-tuned YOLO11s-seg masks (class-agnostic "object")
  → depth geometry inside each mask (robot / fixtures excluded)
  → Gemma names the numbered objects, resolves "red cube" / "green cube", confirms each on a crop
  → SkillPlanner + CollisionChecker (obstacles = measured objects) → RosExecutor → safety layer
  → verification: fingers stalled on the object; after placing, the table camera looks again
  → honest report: "Done (verified …)" only if xy ≤ 10 mm, z ≤ 5 mm, tilt ≤ 5°
```

No object list, colour names or simulator ground truth are used by the agent.

## What's on screen (demo viewer, 2026-10-14)

`./launch/start_demo.sh` opens a read-only **demo viewer** window once the table camera streams:
- **Image:** the live table camera at its frame rate, shown at the bottom left as "live NN fps".
- **Overlay:** coloured masks and numbers from the agent's latest perception result. YOLO runs only when the agent observes; the bar shows that frame and its age.
- **Highlights:** TARGET in green, DESTINATION in blue, question candidates in orange.
- **Panel:** the task, the supervisor state, the target and destination (description, position, detector confidence), the grasp, and retries used.
- **Live agent logs:** every event of the run, the same lines as the terminal. Use the arrow keys, PgUp/PgDn and End to scroll.
- **Status bar:** Task / Robot / HOME.

Keys: `q` closes the viewer only (the task continues, measured); `c` switches table / wrist / auto.
Questions are always answered in the terminal; the viewer never reads input.

After a verified placement the robot now returns to its validated HOME (`ARM_HOME`) and checks it
from joint feedback (every joint within 1.5 deg, the executor's validated reach tolerance) before
reporting COMPLETED. On a failure it does not move HOME (it stops where it is and says why).

## Run it

GPU order matters (12 GB): the script loads Gemma first, then the detector, then starts Isaac Sim.
Close any other Isaac Sim first.

```bash
cd ~/Documents/PhD-CE/Research-LF/6DOF-Arm
./launch/start_demo.sh                   # Isaac Sim window + demo viewer
./launch/start_demo.sh --headless        # no Isaac Sim window (demo viewer still opens)
./launch/start_demo.sh --no-viewer       # terminal only
./launch/start_demo.sh --camera-windows  # also the raw wrist + table RGB-D windows
```

- **Faster:** `./launch/start_demo.sh --headless` (the demo viewer still shows the camera): Gemma runs ~2.5x
  faster and motions ~1.5x faster without the Isaac Sim window (measured). Give several instructions in one
  session: objects that did not change keep their description, so later instructions skip most Gemma calls
  (measured: red→green after yellow→tan 161 s headless). A busy desktop (browser, video) slows Gemma too.
- Wait for `[agent] ready` (about 2–3 min: Gemma ~1 min, Isaac Sim 1–2 min). GPU peak with the
  Isaac Sim window measured at 7.1 GB of 12 GB.
- Type an instruction at the `instruction (empty line to quit)>` prompt, e.g.
  - `Put the red cube on the green cube.`
  - `Pick up the yellow cube and put it on the tan box.`
  - `Put the orange cube on the purple cylinder.`
  - `Put the pink cube next to the white tall cylinder.`
- Since the goal-supervisor refactor (2026-10-14) other goals work too, without moving the robot:
  `Take a look at your surroundings.` (grounded scene report), `How many cubes are on the table?`
  (Gemma's answer, marked unverified), `Where is the red cube?` (measured position), and sequences
  (`Put the red cube on the green cube, then put the orange cube on the purple cylinder.` — HOME once at
  the end). Searching other viewpoints ("find X" when X is hidden) is not available yet; the robot says so.
- Active perception (2026-10-14): `Find the red cube, move closer to inspect it, and tell me its position and
  shape.` (wrist close-up), `Find the object behind the box.` (looks behind objects with the wrist camera; try
  `./launch/start_demo.sh --scenario ap_occlusion`), new object types with `--scenario ap_categories`
  (banana, bottles, mug, bowl). The arm moves only along collision-checked paths and returns HOME after.
- Relations understood: on, next to, left of, right of, in front of, behind (robot's frame: +x =
  the robot's right, "in front of" = towards the robot; say "from the camera's view" for the camera
  frame).
- It uses ROS_DOMAIN_ID=42 (set another with `ROS_DOMAIN_ID=… ./launch/start_demo.sh`).

Without the launcher (two terminals):

```bash
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42
.venv-seg/bin/python -m agent.run --start-sim --gui                 # terminal 1 (prompt)
/usr/bin/python3 ros_nodes/camera_viewer.py --title "Table camera" \
  --color-topic /table_camera/color/image_raw --depth-topic /table_camera/depth/image_rect_raw \
  --depth-min 1.2 --depth-max 3.2                                    # terminal 2 (optional)
```

One instruction, no prompt: `.venv-seg/bin/python -m agent.run --start-sim --instruction "…" --out out/demo/mine`.

## What you will see

1. The arm turns about its base to the pose that hides the fewest table positions from the camera.
2. About 2 min of looking and thinking (measured median 111 s): detection, measuring, Gemma describing
   every object from its own crop, then ~16 s to resolve and confirm the two objects (watch
   `images/` fill up, see below). Nothing moves during this time; that is normal.
3. The arm moves above the object, descends, closes, checks that the fingers stalled on it,
   lifts, carries, lowers onto / next to the destination, opens and retreats.
4. The arm retreats, turns away, the table camera checks the placement (~40 s), then the arm
   returns HOME (~30 s) and the joint positions are checked. The agent prints one of (a full
   instruction took 7.3-9.4 min in the GUI runs of 2026-10-14, plus start-up):
   - `COMPLETED: Done (verified by the table camera): placed … mm from the planned spot …`
   - `FAILED (<class>): <reason>. I did not report success.`
   - `CANCELLED: <question> -> <why>` — only if you typed cancel, input ended, or nobody answered
     within 120 s.

## Questions

When several objects fit a description (for the object to move or the destination), the agent stops
before moving and asks you:

```
[agent] QUESTION (object): Several objects could be 'yellow cube'. Which one do you mean?
    5: yellow Yellow block       at x +101 mm, y +717 mm (0.72 m from the robot), 60 x 60 x 61 mm
    7: yellow Block              at x +2 mm, y +1146 mm (1.15 m from the robot), 50 x 50 x 51 mm
   14: yellow Cube               at x +2 mm, y +427 mm (0.43 m from the robot), 45 x 45 x 46 mm
  picture of the candidates: out/demo/.../images/table_001_question_5_7_14.jpg
answer>
```
(example; the numbers change every run). Answer with a number (`5`, `#5`, `object 5`), words that fit
only one candidate (`the gold one`), `nearest` / `farthest` (from the robot base), or `cancel`. An
answer that fits none or several candidates is explained and the question is repeated. The task then
continues with your choice (no restart). x = the robot's right, y = forward from its base.
Wait time: 120 s (`--ask-timeout`, 0 = no limit), then it cancels; end of input also cancels.

## Where the evidence goes

`out/demo/<date>/live_<time>/<nn>/` per instruction:
- `events.jsonl` — every state, tool call, VLM prompt/answer, verification and failure (JSON lines)
- `images/table_NNN_som.jpg` (numbered masks), `table_NNN_names.jpg` (Gemma's names),
  `table_NNN_crop_K.jpg` (confirmation crops)
- `result.json` — final report (status, plan, verifications, timeline, world model)
- `checkpoints.sqlite` — LangGraph checkpoints

## How to stop

- Empty line at the prompt: stops the agent, Isaac Sim and the camera windows.
- Ctrl-C in the launcher terminal does the same. A motion in progress is not interrupted mid-
  trajectory by the agent (the simulator's safety layer still applies).
- If something is left running: `pkill -f sim.run_live` and `pkill -f camera_viewer.py`.

## Limitations (see DEMO_REPORT.md for measured numbers)

- Simulation only; detector trained on synthetic images; AGPL (Ultralytics) — demo use only.
- One table camera for perception and verification; objects hidden behind others or the arm are not
  measured (a hidden top face is refused, not guessed).
- One object per instruction; no "pick and hold", no stacking more than one level in one command.
- Questions are answered at the terminal only (no GUI picker); the rehearsal harness uses
  `--ask scripted` (always "cancel") because nobody is at its terminal.
- Mass estimation is skipped in this slice.
