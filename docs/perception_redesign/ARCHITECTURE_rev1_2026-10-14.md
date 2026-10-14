# VT6L multi-agent manipulation system — architecture (design, 2026-10-14)

Status: **design for review, nothing implemented.** Evidence labels: **measured** (we ran it),
**from files** (read in the code / docs), **unverified** (not checked; estimates are marked).
Supersedes the agent design implied by `ros_nodes/vlm_agent_node.py`; the frozen manipulation
baseline (`docs/baselines/manipulation_A_2026-10-14/`) is reused, never modified.

## 0. Fixed principles
- No hardcoded object knowledge: no classical detector, catalogue, colour names / families, fixed
  object lists or hand-tuned classification thresholds. Objects are discovered (YOLO-seg masks),
  measured (depth) and described (VLM) at runtime.
- Fixed knowledge = physical / safety configuration only: camera intrinsics and extrinsics, the
  VT6L model, joint / speed / payload limits, gripper calibration, safety zones, numeric tolerances.
- Simulator ground truth: evaluation only, never in the pipeline (verification included).
- The LLM never commands motion. Every motion: grasp / placement geometry (deterministic) →
  `SkillPlanner` + `CollisionChecker` → `RosExecutor` → the simulator's safety layer.

## 1. Multi-agent architecture (LangGraph)

### 1.1 Structure
One **supervisor graph** (a LangGraph `StateGraph`) whose nodes are the supervisor states; each
working state delegates to an **agent subgraph**. Seven agents (AUDIT_AND_PLAN §4). LLM-driven
agents are *roles* (system prompt + allowed tools + output schema) on **one shared VLM instance**,
called sequentially — one GPU, one model. Agents exchange **typed messages** and read / write a
**shared world model**; no free-text hand-offs.

```
                        user text  (+ voice, final phase)
                              │            ▲ questions / results (/task/feedback, terminal)
                              ▼            │
 ┌──────────────────────── SUPERVISOR GRAPH (LangGraph, deterministic state machine) ───────────────────────┐
 │ IDLE → UNDERSTANDING → OBSERVING ─┬→ LOCALIZING → PLANNING_GRASP → APPROACHING → GRASPING →               │
 │                                   └→ SEARCHING ─┘        VERIFYING_GRASP → TRANSPORTING → PLACING →         │
 │                                                          VERIFYING_PLACEMENT → COMPLETED                   │
 │   any state ──failure──→ RECOVERING ──(bounded retry)──→ re-enter state | FAILED      cancel → CANCELLED  │
 └─────┬──────────────┬──────────────┬───────────────┬──────────────┬───────────────┬──────────────┬─────────┘
       │              │              │               │              │               │              │
  Supervisor     Perception     Exploration     Grasp planner  Motion planner    Execution    Verify/recover
  (LLM role:     (mixed: YOLO   (mixed: LLM     (deterministic)(deterministic)  (deterministic)(mixed: checks
  interpret,     + geometry     proposes views, │              │               │              deterministic,
  decompose,     determ.; VLM   determ. scores  │              │               │              LLM may choose
  ask)           describes /    + reachability) │              │               │              among allowed
                 resolves)                      │              │               │              recoveries)
       └──────────────┴──────────────┴───────────────┴──────────────┴───────────────┴──────────────┘
                                  shared WORLD MODEL (objects, robot, task, history)
                                                │
               VLM / Detector interfaces ── (in-process now; separate inference server later)
                                                │
          ROS 2 bridge thread (MultiThreadedExecutor): /joint_states, /clock, cameras, TF,
          /joint_trajectory, /joint_command, /safety/events  ──→  Isaac Sim (separate process)
```

### 1.2 Typed shared state (Python dataclasses / pydantic models; JSON-serialisable)
```
WorldModel:
  objects: {object_id: ObjectRecord}            # persistent ids, never names
  robot: {q, gripper_finger_rad, holding: object_id | None, tcp_pose}
  cameras: {table: CameraInfo, wrist: CameraInfo}  # intrinsics + current extrinsics (from TF / FK)
  observations: [ObservationRef]                 # frame ids, time, camera, which objects seen
ObjectRecord:
  id, first_seen, last_seen, times_seen, status (on_surface | held | missing | moved_by_robot)
  geometry: {centre_xyz, top_z, footprint_lw, yaw_deg | None, symmetry_deg, grasp_yaw_deg,
             grasp_width_m, height_m, support_id | None, points, mask_area_px}     # measured
  description: {name, colour, shape, material, free_text, frame_id}               # VLM, as seen
  uncertainty: {pos_std_m, yaw_std_deg, n_obs}
TaskState:
  instruction, frame ("robot" default | "camera" | named), plan: [SubTask], current, retries{class: n},
  clarifications_asked, status, failure: {class, agent, reason} | None
SubTask: {kind: pick | place | move | look, target_ref, relation?, reference_ref?, frame}
Reference: {text, resolved_id | None, candidates[], confidence, needs_clarification: bool}
```

### 1.3 Agents

| Agent | Responsibility | Inputs → outputs | Kind | Why |
|---|---|---|---|---|
| **Supervisor** | Interpret the instruction; decompose into sub-tasks; drive the state machine; ask the user; final report | instruction text, WorldModel → TaskState (plan, references unresolved or resolved), user questions | **Mixed**: LLM role for interpretation / decomposition / clarification wording; the state machine and retry budgets are deterministic code | Language judgment needs the LLM; legal transitions, budgets and "never done without verification" must be guaranteed by code |
| **Perception** | Observe: masks → geometry → robot/fixture removal → descriptions → memory update; resolve references to object ids | camera frames, robot state, TF → ObservationRef, updated ObjectRecords, Reference resolutions | **Mixed**: YOLO-seg + depth geometry + TF exclusion deterministic; description and reference resolution by the VLM role | Geometry is physics (must be measured); naming and "the one next to it" are judgment |
| **Exploration** | Find objects not in memory / not resolvable: choose wrist-camera views, look, re-perceive, record "not found here" | unresolved Reference, WorldModel, coverage map → new observations or "exhausted" | **Mixed**: deterministic candidate views (reachable, collision-free, coverage-scored); the LLM may rank candidates using the description ("probably behind the arm") | View feasibility is geometry; where an object is likely is judgment |
| **Grasp planner** | Grasp pose from measured geometry: yaw (symmetry), width, depth, contact length; reject infeasible | ObjectRecord.geometry, robot state → GraspPlan {top_centre, yaw, width, height, symmetry} or failure | **Deterministic** | Physics; existing validated code |
| **Motion planner** | Plan hover / descend / lift / carry / lower / retreat / re-observation moves; collision-check every path | GraspPlan / PlacePlan, WorldModel (obstacle boxes from measured geometry) → [MoveStep, GripperStep] or failure | **Deterministic** | Safety; validated `SkillPlanner` + conservative `CollisionChecker` |
| **Execution** | Run steps on the robot; settle; report per-step result; stop on cancel | steps → ExecutionResult {ok, failed_step, reason, per-step tracking / settle} | **Deterministic** | Real-time control; validated `RosExecutor` |
| **Verify / recover** | Grasp, lift, placement checks from sensors only; classify failures; choose a bounded recovery | step results, fresh observations, WorldModel → Verdict {pass, class, evidence}; RecoveryAction | **Mixed**: checks deterministic; when several recoveries are allowed, the LLM may choose with the evidence; budgets deterministic | Pass/fail must be objective; choosing between "re-observe" and "regrasp at another yaw" can use judgment |

### 1.4 Failure detection and ownership

| Failure | Detected by | Handled by |
|---|---|---|
| Malformed / schema-invalid VLM output | VLM interface (validation) | 1 retry with the error, then the calling agent fails with class `perception` (descriptions) or `understanding` (plan) |
| Instruction not understood / ambiguous reference | Supervisor / Perception (Reference.needs_clarification) | Supervisor asks the user (≤ N questions, timeout) → else FAILED (reason) |
| Object not in view / memory | Perception (unresolved) | Exploration (bounded views) → else ask user → else FAILED |
| No mask / depth for an object | Perception | Re-observe (other camera) once → Exploration |
| Grasp infeasible (width, contact, reach) | Grasp planner | Verify/recover: other grasp yaw (symmetry) → re-observe → FAILED `planning` |
| No collision-free path | Motion planner | via-home fallback (existing) → other grasp yaw → FAILED `planning` |
| Trajectory rejected by the safety layer / not reached | Execution (`/safety/events`, settle) | FAILED `execution` (no automatic retry of a rejected motion) |
| Fingers closed on nothing / no stall | Verify (grasp) | open, lift, re-perceive, regrasp (≤ 1) → FAILED `grasp` |
| Object did not leave its spot / mass over limit | Verify (lift) | put down at the pick spot, re-perceive, regrasp (≤ 1) / report payload → FAILED |
| Placement off target / unstable | Verify (placement) | re-observe once (measurement noise) → re-grasp and re-place (≤ 1) → FAILED `placement` |
| User cancel | Supervisor (cancel command) | stop after the current step, release only if safe → CANCELLED |

### 1.5 States → agents
| State | Agent(s) | Exit to |
|---|---|---|
| IDLE | — | UNDERSTANDING on an instruction |
| UNDERSTANDING | Supervisor (VLM role) | OBSERVING; or ask user; FAILED |
| OBSERVING | Perception (table camera, + wrist if needed) | LOCALIZING if all references resolved; else SEARCHING |
| SEARCHING | Exploration (+ Perception per view) | LOCALIZING; ask user; FAILED |
| LOCALIZING | Perception (fresh measurement of target / reference) | PLANNING_GRASP |
| PLANNING_GRASP | Grasp planner + Motion planner | APPROACHING; RECOVERING |
| APPROACHING | Execution (hover, descend) | GRASPING; RECOVERING |
| GRASPING | Execution (close) | VERIFYING_GRASP |
| VERIFYING_GRASP | Verify (stall + lift + re-observation + mass) | TRANSPORTING; RECOVERING |
| TRANSPORTING | Motion planner + Execution (carry) | PLACING |
| PLACING | Motion planner (placement pose from §5) + Execution (lower, release, retreat) | VERIFYING_PLACEMENT |
| VERIFYING_PLACEMENT | Verify (re-observation) | next sub-task / COMPLETED; RECOVERING |
| RECOVERING | Verify/recover (+ the agent that owns the retried step) | the retried state; FAILED |
| COMPLETED / FAILED / CANCELLED | Supervisor (report with evidence) | IDLE |

## 2. Tools (typed I/O; the only way agents touch the world)

| Tool | Input → output | Wraps (existing, **from files**) | New |
|---|---|---|---|
| `perceive(camera)` | {camera: table\|wrist} → Observation {frame_id, masks[], per-mask geometry, excluded masks + reason} | depth / geometry of `perception/grounding.py` (`locate`'s top face, `footprint_yaw`, `grasp_pose`), `_peel` stack splitting from `perception/classical_detector.py` (geometry functions only, no naming / catalogue) | `Detector` interface (YOLO-seg), mask-restricted back-projection, TF-based robot / fixture exclusion |
| `describe_objects(frame_id)` | numbered set-of-mark image → {mask_id: Description} (strict JSON) | `vlm/model.py: VisionLanguageModel.generate` | prompt, schema, set-of-mark rendering, retry logic |
| `update_memory(observation, descriptions)` | → object_id per mask (match by 3-D position / footprint / height, description as tie-breaker) | idea of `perception/world_memory.py: observe` (distance + height gates) | new memory module: persistent ids, descriptions, uncertainty, no colour lists |
| `resolve_reference(text, frame)` | → Reference {resolved_id \| None, candidates, needs_clarification} | VLM | prompt over the numbered image + memory summary |
| `propose_views(reference)` / `look(view)` | → [View] scored / Observation | `vlm_agent_node.py: look / explore` view logic, `control/robot.py: VIEW_POSES` | coverage map, reachability + collision scoring, view ranking hook for the VLM |
| `plan_grasp(object_id)` | → GraspPlan | `control/skills.py: hover / grasp` geometry (yaw symmetry, `fingertip_below_tcp`, contact length) | adapter from ObjectRecord.geometry (replaces the catalogue's `grasp_for`) |
| `plan_place(object_id, relation, reference_id, frame)` | → PlacePlan {pose, yaw} or failure | `SkillPlanner.place`, `CollisionChecker` | relation geometry (§5) |
| `plan_motion(plan)` | → [Step] | `SkillPlanner(..., collision=CollisionChecker(boxes from WorldModel))` incl. via-home fallback | obstacle boxes from measured geometry (replaces `sim/scene_objects.obstacle_boxes`); fixtures from configuration |
| `execute(steps)` | → ExecutionResult | `execution/ros_executor.py: RosExecutor.execute` | runs on the shared ROS thread (§7) |
| `gripper_state()` | → {finger_rad, gap_m (calibrated), stalled} | `control/robot.py: gripper_gap` + calibration | — |
| `estimate_mass()` | → {mass_kg, std} | — | joint-effort difference before / after lift (§6); needs efforts on `/joint_states` (§9 open item) |
| `verify_grasp / verify_lift / verify_placement` | → Verdict {pass, class, evidence} | contact check logic of `experiments/known_pose_pick_place.py: run_pick_place` | sensor-only versions (no ground truth) |
| `ask_user(question)` | → answer \| timeout \| cancel | `--stdin` reader and `/task/feedback` of the current agent | LangGraph `interrupt` / resume, limits |
| `report(status, evidence)` | → message | `/task/feedback` | — |

## 3. Perception pipeline
1. **Frames**: table RGB-D (+ wrist when asked) with frame checks reused from
   `scripts/validate_cameras_phaseA.py` (sizes, RGB/depth sync, intrinsics, extrinsics vs TF / FK).
2. **Masks**: class-agnostic instance segmentation (one class "object"), confidence per mask, behind a
   `Detector` interface (`detect(rgb) -> [Mask{id, polygon/bitmap, score}]`).
3. **Geometry per mask**: depth pixels inside the mask (eroded 2 px against edge bleeding) →
   world points (intrinsics, extrinsics); support-surface height measured (floor plane fit, or the
   top of the supporting object); top face, centre, footprint, yaw, symmetry, grasp yaw / width,
   height; stacks split by peeling layers.
4. **Exclusion by the robot model + TF**: mask pixels whose 3-D points lie inside the robot's collision
   geometry at the current joint state (`control/collision.py` spheres / boxes, inflated 1 cm) or
   inside configured fixture volumes (camera stand, microphone: physical configuration) are removed;
   a mask that is mostly robot / fixture is dropped with the reason logged. No fixed xy circles.
5. **Description**: the frame with numbered mask outlines (set-of-mark) → one VLM call per frame →
   strict JSON `{"objects": [{"mask_id": int, "name": str, "colour": str, "shape": str,
   "material": str, "notes": str}]}`; ids must match the drawn ids; retry once on invalid JSON,
   else the frame is "undescribed" (geometry is still stored).
6. **World memory**: match each measured object to a record (3-D gates on centre / height /
   footprint; descriptions only break ties), update geometry with uncertainty, append the
   description, assign a new persistent id otherwise; objects not seen where they should be visible
   become `missing` after N observations.
7. **References**: the VLM resolves "the red cube", "the one next to it" against the numbered image +
   memory summary → object id or `needs_clarification` with candidates. **Ambiguous / unresolved →
   ask the user or re-observe with the wrist camera; never a silent guess.**

## 4. VLM integration
- **Interface** `VLM.call(role, images, text, schema) -> validated object` (in-process
  `vlm/model.py` now; an inference-server implementation later with the same interface; same for
  `Detector`).
- **Structured output**: JSON schema per role (pydantic), parse + validate, one retry with the
  validation error, then a typed failure. No free-text parsing downstream.
- **Roles**: supervisor (decompose, relations, frames, clarifying questions), describer (set-of-mark
  descriptions), resolver (reference → id), explorer (rank candidate views), recovery chooser (pick
  one of the allowed actions). Each role has a fixed tool list; tool choice is part of the schema.
- **Thinking**: off for describer / resolver (latency), configurable for supervisor.
- **Reliability metrics** (measured on recorded frames and instruction sets, CPU-free replay in tests):
  JSON validity rate, schema-retry rate, tool-choice accuracy (vs labelled expected tool), naming
  accuracy (per object, does the description identify the GT object — judged per object), reference
  resolution accuracy (incl. "ask" correctness on ambiguous cases), latency per call.
- **Acceptance (proposed)**: JSON validity ≥ 98 % after one retry; reference resolution ≥ 90 % on
  unambiguous, "ask" ≥ 90 % on ambiguous; if Gemma 4 E2B misses these, evaluate a fallback.
- **Fallback candidates** (licences and memory **unverified**, to check before download): Gemma 3 4B
  (Gemma licence), Qwen2.5-VL-7B (Apache-2.0 per its model card — verify; 4-bit ~6 GB est.),
  SmolVLM2 (Apache-2.0, small; likely weaker). Selection by the metrics above, never by adding rules.

## 5. Spatial relations
- **Schema**: `place {object_ref, relation: on | next_to | left_of | right_of | in_front_of |
  behind, reference_ref, frame: robot | camera | <named>}`. Default frame **robot** (+x right,
  +y forward from the base), stated in every reply; an instruction naming another frame switches it;
  ambiguity → ask.
- **The VLM only fills the schema** (relation, reference, frame). A deterministic `plan_place`:
  - `on`: centre over the reference's top face, release at its top + held height + clearance; the
    held footprint must fit the reference's top (else ask / fail).
  - directional / `next_to`: direction vector in the chosen frame (next_to = the free side closest
    to the robot's current reach); target centre = reference centre + d·(half extents of both along d
    + gap 2 cm); search outward along d and ±30° in small steps for the first pose that is reachable,
    collision-free (held object at the target + the full place path), inside the workspace map and
    not within the gap of another object.
  - Result reported with the frame ("left of #4 in the robot's frame, 2 cm gap").

## 6. Verification and recovery (no ground truth)
- **Grasp**: close to (width − 8 mm); contact = fingers stop ≥ 1.5° before the target (existing,
  measured in the suite); measured pad gap vs measured object width (calibrated gap; known limitation:
  the simulated linkage flexes under load, so the gap is evidence, not proof — measured 2026-10-14).
- **Lift**: after lifting 3 cm, re-perceive the pick area (table camera; wrist if occluded): the
  object's mask / points are gone from the support and the gripper region shows the object; mass =
  Δ(joint efforts) projected to a vertical force (needs efforts on `/joint_states`); over the payload
  limit → set down, report.
- **Placement**: after release + settle, re-perceive: the object's measured centre / top height
  within tolerance (xy 10 mm, z 5 mm, tilt 5°, the same numbers as Baseline A) of the planned pose,
  stable over 1 s.
- **Recovery budgets** (per task, configurable): perception 2, planning 2, grasp 1, execution 0,
  placement 1; then FAILED with class, agent, evidence. COMPLETED only after VERIFYING_PLACEMENT passes.

## 7. Environments, threading, GPU
- **Processes**: (1) Isaac Sim (`.venv`, headless when combined); (2) **agent process** (`.venv-vlm`
  with ROS sourced, as today): LangGraph, VLM (Gemma), detector (YOLO-seg inference), world model,
  tools; (3) camera viewer (system Python). VLM and detector sit behind interfaces so they can move to
  an inference server process later without agent changes.
- **Threading in the agent process**: a ROS bridge thread runs a `MultiThreadedExecutor` (cameras,
  joint states, clock, TF, safety events, publishers); tools post requests and **wait on futures**
  with timeouts; the LangGraph graph runs in the main thread; `RosExecutor` is adapted to use the
  bridge's node instead of spinning itself (removes the spin-in-tool deadlock risk flagged in the
  audit). One task at a time; cancel is a flag checked between steps and inside long waits.
- **GPU (12 GB)**: measured — Isaac headless with two cameras ~2.8 GB (pilot peak incl. renders),
  Gemma 4-bit 2.2 GB steady / ~6.5 GB while loading (earlier sessions). Estimated — YOLO-seg
  inference 0.5–1.5 GB. Load order: VLM first (before Isaac's heavy rendering), then detector, then
  start the sim; sim headless; peak budget ≈ 2.8 + 2.2 + 1.5 ≈ 6.5 GB steady (estimate). Training
  never runs with the sim.

## 8. Checkpointing and event log
- LangGraph checkpointer (local SQLite file per run) stores the state after every node → resumable
  after a crash, replayable.
- An event writer (graph callbacks + tool wrappers) appends JSON lines to
  `out/runs/<run_id>/events.jsonl`: `{t, run_id, state, agent, event: transition | tool_call |
  tool_result | vlm_input | vlm_output | verification | failure | recovery | user_question |
  user_answer | cancel, payload, duration_s, retry}`; VLM images saved by frame id, referenced not
  inlined; trajectories saved per step. A replay tool re-runs perception / VLM steps from recorded
  frames for evaluation (no GPU in unit tests: recorded VLM outputs).

## 9. Inventory: what is replaced, kept or removed (from files)
| Item (current) | Decision |
|---|---|
| Agent's `SkillPlanner(self.kin)` without collision checker (`vlm_agent_node.py` l.116) | **Replaced**: motion planner tool with `CollisionChecker` (obstacles from measured geometry) |
| Agent's own `publish_trajectory` / `execute` / `wait_reached` | **Replaced** by `RosExecutor` (settle, safety rejections) on the shared ROS thread |
| `pick()` without yaw / height (l.463) | **Replaced** by `plan_grasp` from measured geometry |
| Plan schema without relations (`vlm/prompts.py`, `vlm/plan.py`) | **Replaced** by the §5 schema (new files); old parser kept for the old agent |
| Verification via simulator ground truth (`run_pick_place` secured / verified / placement) | **Kept for evaluation only**; pipeline uses §6 sensor checks |
| `world_memory.py` COLOURS / SHAPES / COLOUR_FAMILIES / `matches` | **Removed** from the new pipeline (new memory module); old file untouched |
| Gemma-based detection (`TaskPlanner.detect`) | **Removed** from the pipeline (YOLO-seg detects; Gemma describes) |
| `grounding.py` COLOUR_NAMES / CALIBRATED_COLOURS / `colour_name`, MAX_OBJECT_Z | **Removed** from the pipeline (VLM describes; TF exclusion); geometry functions **kept** |
| Classical detector (`perception/classical_detector.py`) | **Not used**, not even as an interim; stays frozen for comparison |
| Agent `MAX_OBJECT_TOP`, "microphone" label filter, `measured_label` | **Removed** (TF / fixture-volume exclusion; VLM descriptions) |
| Catalogue use in experiments (`grasp_for`, `symmetry_of`, OBJECTS heights) | **Kept** in Baseline A (frozen, by design); the new evaluation uses measured geometry in the pipeline and the catalogue only as ground truth |
| `sim/scene_objects.obstacle_boxes` (objects) | **Replaced** by boxes from measured geometry; fixture volumes kept as physical configuration |
| Prompts listing colours / shapes (`DETECT_PROMPT`, few-shot "<colour> <shape>") | **Removed** from the new prompts |
| Text-command interpreter rules (`control/instructions.py`) | **Kept** (robot jog commands, not object knowledge) |
| Joint efforts on `/joint_states` | **Missing**; needed for mass. `sim/run_live.py` / `sim/ros_control.py` are frozen → **decision needed**: new sibling sim entry point (no baseline change) vs editing them (then re-run the three validations) |

## 10. Evaluation plan
Sets, always reported **separately**: (a) evaluation layouts (today's scene + 20 re-arranged layouts
of the 21 evaluation objects, seeds disjoint from training); (b) novel-object test set; (c)
ambiguity layouts from novel objects (e.g. two red cubes of different sizes) for reference
resolution and the ask-the-user path.
- **Perception**: detection recall / precision at mask IoU ≥ 0.5 (match by 3-D position within a stated
  tolerance, 15 mm proposed), mask IoU, 3-D xy / z / yaw error (yaw modulo symmetry); robot / fixture
  exclusion false positives.
- **VLM**: JSON validity, retry rate, tool-choice accuracy, naming accuracy, reference resolution
  accuracy and "ask" correctness; latency.
- **End to end** per instruction type (pick, place on, each relation, multi-step, ambiguous,
  missing object): success rate, failures by class and which agent caught them, recoveries used,
  questions asked, re-observation time (counted), time per stage, peak GPU memory.
- **Manipulation**: the 11 known-pose cases with the new pipeline vs frozen Baseline A, case by case.
- Ground truth only in the scorer; the replay of recorded runs makes VLM / perception metrics
  repeatable.

## 11. Stopgap for an early demo (recommendation only; nothing installed)
| Option | What | Licence (**unverified**, check before use) | Risk |
|---|---|---|---|
| YOLO-World (Ultralytics) | open-vocabulary boxes from text | AGPL-3.0 (Ultralytics) | licence; boxes only (masks via SAM); text prompts reintroduce class words |
| OWLv2 (transformers) | open-vocabulary boxes | Apache-2.0 | slower; boxes only |
| Grounding DINO + SAM / SAM 2 | boxes from text + masks | Apache-2.0 | two models on the GPU; latency |
**Recommendation:** don't build the demo on a stopgap — it would need text queries (object words)
and a second set of models while the dataset finishes in hours (§12). If a demo is needed before
Phase 2 ends, OWLv2 + SAM 2 (both Apache-2.0, unverified) is the least restrictive, used class-agnostic
("object") behind the same `Detector` interface and removed afterwards.

## 12. Phase plan (efforts are estimates)
| # | Phase | Deliverables | Tests | Depends on | Effort |
|---|---|---|---|---|---|
| 1 | Dataset (running) | synthetic_v1 (6000 / 1000 / 500), STATS.json, YOLO labels; eval layouts incl. 20 re-arranged + ambiguity sets | verify_dataset.py (0 problems) | — | finishing (~1.5 h render) + 0.5 d layouts |
| 2 | Segmentation model | train a permissive model (torchvision Mask R-CNN, BSD-3, or another Apache / MIT option) **and** Ultralytics YOLO-seg; compare mask mAP / recall on val + novel; **licence decision** in DECISIONS.md | metrics on val / test_novel | 1 | 1.5–2 d (GPU, sim stopped) |
| 3 | Perception module | Detector interface, mask geometry, TF exclusion, Observation | unit tests on recorded frames; geometry vs GT on eval layouts | 2 | 1.5 d |
| 4 | VLM roles + memory + references | VLM interface (schemas, retries), set-of-mark describer, new memory, resolver | recorded-output unit tests (no GPU); reliability metrics on recorded sets | 3 | 2 d |
| 5 | Agent framework | LangGraph supervisor + subgraphs, tools, ROS bridge thread, RosExecutor adaptation, event log + checkpointer, ask / timeout / cancel | graph tests with fake tools; one live pick on GT-free pipeline | 3, 4 | 2–3 d |
| 6 | Relations + verification + recovery | plan_place relations, sensor-only checks, mass (after the efforts decision), budgets | unit tests; failure injection | 5 | 1.5–2 d |
| 7 | Evaluation | all §10 metrics, three sets separately, Baseline A comparison | — | 6 | 1.5 d |
| 8 | Voice | speech-to-text (model choice by measured WER + GPU), connection to the supervisor | WER on recorded speech; end-to-end voice tasks | 7 | 1–1.5 d |
Total ≈ 13–16 working days (estimate).

## 13. Open decisions for review
1. Joint efforts for mass estimation: new sim entry point (recommended) vs editing frozen files.
2. Recovery budgets and tolerances (proposed values in §6).
3. VLM acceptance thresholds (§4).
4. Whether a stopgap demo is wanted (§11).
