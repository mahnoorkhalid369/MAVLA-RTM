# Demo: typed instruction → perception → planning → verified pick-and-place

First thin slice of `docs/perception_redesign/ARCHITECTURE.md` (rev 3). Results and readiness:
`docs/perception_redesign/DEMO_REPORT.md` (read its first lines first).

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

## Run it

GPU order matters (12 GB): the script loads Gemma first, then the detector, then starts Isaac Sim.
Close any other Isaac Sim first.

```bash
cd ~/Documents/PhD-CE/Research-LF/6DOF-Arm
./launch/start_demo.sh             # Isaac Sim window + wrist and table camera windows
./launch/start_demo.sh --headless  # no Isaac Sim window (camera windows still open)
```

- Wait for `[agent] ready` (about 2–4 min: Gemma ~1 min, Isaac Sim 1–2 min).
- Type an instruction at the `instruction (empty line to quit)>` prompt, e.g.
  - `Put the red cube on the green cube.`
  - `Pick up the yellow cube and put it on the tan box.`
  - `Put the orange cube on the purple cylinder.`
  - `Put the pink cube next to the white tall cylinder.`
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
2. ~30–60 s of thinking: detection, measuring, Gemma naming every numbered object, resolving and
   confirming the two objects (watch `images/` fill up, see below).
3. The arm moves above the object, descends, closes, checks that the fingers stalled on it,
   lifts, carries, lowers onto / next to the destination, opens and retreats.
4. The arm turns away again, the table camera looks, and the agent prints one of:
   - `COMPLETED: Done (verified by the table camera): placed … mm from the planned spot …`
   - `FAILED (<class>): <reason>. I did not report success.`
   - `CANCELLED: I asked (…) and the answer was 'cancel'.` — whenever the agent is unsure
     (ambiguous reference, confirmation "unsure", unsupported task) it asks; in this demo the
     question is answered "cancel" automatically and logged.

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
- The scripted "cancel" answer stands in for a real user dialogue.
- Mass estimation is skipped in this slice.
