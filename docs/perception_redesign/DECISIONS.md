# Perception redesign — decisions log

One dated entry per design decision: the decision, the reasoning, the alternatives rejected.
Evidence labels as in docs/AUDIT_AND_PLAN.md: measured / from files / unverified.

## 2026-10-14 — The VLM also names objects and resolves references (reverses AUDIT_AND_PLAN §3)
**Decision.** AUDIT_AND_PLAN.md section 3 decided that the language model only parses language into
a validated task, with perception deterministic. This redesign reverses that: the VLM also (a)
describes each detected object (free-text name, colour, shape, material as it sees them) from a
numbered set-of-mark image, and (b) resolves references in instructions ("the red block", "the one
next to it") to object ids.
**Reasoning.** The goal is no hardcoded object knowledge: no catalogue, colour names, colour families
or fixed object lists. Naming and reference resolution are open-vocabulary problems; rules
(COLOUR_FAMILIES, colour_name, matches()) fail on anything outside their lists (Phase A,
2026-10-14, measured: "steel" not in the gray family; "black" in no family).
**Safeguards.** Strict JSON schema validation (retry once, else the frame is "undescribed"); the VLM
never sets geometry, grasp poses or motions — those come from measured depth only; physical
verification (grasp width vs measured width, lift / mass plausibility, placement re-observation);
unresolved or ambiguous references → ask the user or re-observe, never a silent guess.
**Alternatives rejected.** Keeping colour/shape rules (fails open-world, the failures above);
a trained classifier with a fixed label set (a fixed object list in disguise); letting the VLM
output 3D poses (not metric, not verifiable).

## 2026-10-14 — Keep depth geometry; the learned part only detects and segments
**Decision.** Class-agnostic YOLO-seg (one class "object") finds instance masks; centre, top height,
footprint, yaw, symmetry and grasp width are measured from depth inside each mask (reusing
grounding / _peel geometry).
**Reasoning.** Depth + calibration already gives 1–3 mm (measured, Phase A 2026-10-14); learned 6-D
pose needs models per object and training for no gain on rigid tabletop items. What failed was
separating and naming objects — exactly what masks and the VLM add.
**Alternatives rejected.** End-to-end learned pose (FoundationPose / DOPE: need object models,
contradict "no object knowledge"); class-specific YOLO (fixed object list).

## 2026-10-14 — Phase 1 data: COCO instance masks as the master format
**Decision.** The generator writes COCO instance-segmentation annotations (polygons + lossless
per-instance mask PNGs); a converter produces YOLO-seg labels.
**Reasoning.** Keeps Phase 2's model choice open: Ultralytics YOLO is AGPL-3.0; permissive options
(torchvision Mask R-CNN, BSD; others) read COCO. Mask PNGs allow exact IoU in evaluation.
**Alternatives rejected.** YOLO format only (locks in one toolchain).

## 2026-10-14 — Phase 1 data content (agreed with the user)
Modal masks (visible pixels only). Robot, gripper, camera stand, microphone and floor are unlabelled
background; an object held in the gripper is labelled. 80 % table-camera views with pose jitter
(±5 cm, ±3°), 20 % wrist-like viewpoints. Floor appearance and object materials randomised
(textures, colours, markings, shiny / metallic), lighting randomised, RGB effects (noise, blur,
exposure). Evaluation layouts (incl. today's scene) and exact copies of the 21 evaluation objects
are excluded from training; a separate novel-object test set uses shapes never used in training.
Assets: procedural shapes + NVIDIA's official Isaac Sim asset server only, source and licence
recorded per asset.

## 2026-10-14 — No classical detector in the new system, not even as an interim
**Decision.** The new pipeline detects with a class-agnostic learned segmentation model only;
`perception/classical_detector.py` stays frozen as a comparison baseline and is never called by
the agents. **Reasoning.** It identifies objects through the catalogue and colour names (hardcoded
object knowledge); an interim use would leak that knowledge into the agents and the evaluation.
**Rejected.** Classical candidates + VLM choice for an early demo (audit option D4).

## 2026-10-14 — LangGraph as the agent framework
**Decision.** A LangGraph `StateGraph` supervisor (states = the AUDIT_AND_PLAN task states) with
agent subgraphs; local only (no LangSmith / cloud tracing); SQLite checkpointer + our own JSON-lines
event log. **Reasoning.** Explicit, typed state machine with checkpoints, interrupts for asking the
user, and replay — the deterministic execution layer the plan requires, without a cloud service.
**Rejected.** Hand-written loop in the ROS node (today's agent: implicit states, no checkpoints, no
replay); frameworks built around hosted LLM APIs.

## 2026-10-14 — Multi-agent split: which agents use the LLM
**Decision.** LLM roles (one shared VLM, sequential calls): Supervisor (interpretation, decomposition,
clarification), Perception's describer and reference resolver, Exploration's view ranking, Recovery's
choice among allowed actions. Deterministic: Grasp planner, Motion planner, Execution, verification
checks, the state machine, budgets and safety. **Reasoning.** LLM only where judgment is needed;
anything involving physics, safety or pass/fail must be measured and reproducible. The LLM never
issues motion. **Rejected.** LLM tool-calling of motion primitives (Code-as-Policies style); separate
models per agent (GPU budget).

## 2026-10-14 — Interfaces for the VLM and the detector; one agent process
**Decision.** Agent process in `.venv-vlm` (LangGraph + VLM + detector inference), ROS 2 on a
background MultiThreadedExecutor thread, tools wait on futures; VLM and detector behind interfaces so
they can move to a separate inference server without agent changes. **Reasoning.** One torch / GPU
context; removes the spin-inside-a-tool deadlock risk; keeps a later split cheap.

## 2026-10-14 — Spatial relations: the VLM names them, geometry places them
**Decision.** Relations on / next to / left of / right of / in front of / behind with an explicit
frame (default robot frame, stated in every reply; other frames only when named; ask if ambiguous).
The VLM fills relation + reference + frame; a deterministic planner computes the pose from measured
footprints with a free-space check through the collision checker. **Rejected.** The VLM outputting
coordinates (not metric, not verifiable).

## 2026-10-14 — Licences
**Decision.** Prefer Apache-2.0 / MIT / BSD or the Gemma licence for models; AGPL or research-only
only if a permissive option is clearly worse on measured results, recorded here with the restriction.
Phase 2 compares a permissive segmentation model with Ultralytics YOLO-seg (AGPL-3.0) before deciding.

## 2026-10-14 — Architecture revision 2 (review decisions)
Design approved with changes; revision 1 kept as `ARCHITECTURE_rev1_2026-10-14.md`.
- **Wrong-object protection.** §6's checks only show the plan was executed, not that the right
  object was chosen. Added: an independent confirmation call (`confirm_object`: crop + reference text
  only, never the resolver's answer) before any motion, disagreement → ask the user; a "confident
  wrong resolution" metric with threshold ≤ 2 % on ≥ 400 references; a failure row naming the
  confirmation check as the detector. Rejected: relying on post-hoc verification (cannot detect it).
- **Held object survives robot exclusion** (its tool-frame box, as in the collision model).
- **Statistics.** Minimum trials per metric and per instruction type, 95 % Wilson intervals,
  thresholds for naming (≥ 90 %) and tool choice (≥ 95 %), reference accuracy (≥ 97 %) tied to task
  success (a place needs two correct references: ≈ p²). The interval column states what each trial
  count can show (computed).
- **Mass estimate = plausibility check only** (light vs over the payload limit) until its accuracy is
  measured on 0.1 kg and 3 kg (≥ 10 lifts each), because the simulated gripper's internal forces act on
  the arm's joints (measured, AUDIT §6.2).
- **Whole-scene re-perception after each sub-task**; **visibility reasoning**: occluded / out of view
  give no evidence; `missing` only after 2 "visible but absent" observations.
- **Joint efforts:** a new sim entry point that imports the frozen code (no copy, no baseline change).
- **Recovery budgets and tolerances** approved as proposed. **No stopgap demo.**
- **Known limitation recorded:** no Epson CAD → simplified arm in robot exclusion and in the training
  images; sim-to-real impact and mitigations in ARCHITECTURE §13.

## 2026-10-14 — Architecture revision 3 (approved rev 2 + additions)
- **Correlated errors:** the confirmation call uses the same model as the resolver; the decisive
  metric is the joint failure rate (resolver wrong AND confirmation "match") on the 400-reference set;
  above 2 % → evaluate a second model for confirmation.
- **Relational references:** confirm attributes from each object's own crop; check the relation
  deterministically from measured positions in the stated frame (same definitions as placement).
  Rejected: a crop containing both objects (the VLM would judge the relation again — not metric,
  correlated with the resolver).
- **Held-object box** for collision checking / exclusion from measured geometry (ObjectRecord), never
  the catalogue.

## 2026-10-14 — Autonomous demo run: decisions taken without the user
(Pre-authorised run; each entry records a choice the rules did not fix.)
- **Pretrained YOLO is not usable** (measured: recall 2 % on boxes, 10 % overall at conf 0.25,
  `out/seg/pretrained_recall.json`); the demo uses a YOLO11s-seg fine-tuned on synthetic_v1.
- **AGPL accepted for the demo only** (user decision): Ultralytics YOLO11s-seg code and weights are
  AGPL-3.0; the demo's detector must not be distributed beyond the AGPL terms; the permissive model
  comparison (Phase 2) and the final licence decision remain open.
- **Agent process runs in `.venv-seg`, not `.venv-vlm`** (deviation from ARCHITECTURE §7): the run's
  rules allow package installs only into `.venv-seg`, and LangGraph is needed in the same process as
  Gemma and YOLO. Installed there the same versions Gemma uses in `.venv-vlm` (transformers 5.19.0,
  bitsandbytes 0.50.2, accelerate 1.15.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface_hub
  1.33.0) + langgraph 1.2.15, langgraph-checkpoint-sqlite 3.1.2, pydantic 2.14.1. No new model
  download (Gemma weights already in models/). `.venv-vlm` untouched. rclpy works there (measured).
- **RosExecutor keeps its own node and spins it during `execute`** (it is manifest-covered and
  cannot be adapted); the agent's camera / state bridge is a second node on a background
  MultiThreadedExecutor. Two nodes in one process, each spun by exactly one executor: no shared-spin
  deadlock.
- **Executor's ground-truth subscription:** the frozen `RosExecutor` subscribes to
  `/sim/ground_truth/objects` (for Baseline A). The agent never reads it (`.truth` / `ground_truth`
  are absent from `agent/`, enforced by `tests/test_agent.py`). Scoring reads ground truth in a
  separate process (`evaluation/demo_rehearsal.py`) after the agent has finished.
- **Agent tests run in `.venv-seg`** (pytest installed there with uv): system Python lacks pydantic /
  langgraph, so `tests/test_agent.py` is skipped by `/usr/bin/python3 -m pytest` and run with
  `.venv-seg/bin/python -m pytest tests/test_agent.py` (without ROS sourced: ROS's pytest plugins need
  `lark`).
- **Geometry fixes found by the synthetic-depth tests (before any sim run):** (1) the lowest
  *visible* point of a floor object sat 8.6 mm above the floor (mask-edge erosion, 2nd percentile) and
  was read as "stacked" -> support = the highest real surface (floor, or a measured top under the
  object) within `SUPPORT_SNAP` = 20 mm below the visible bottom; (2) the top-face tilt included the
  upper side-face points (6.4 deg on a flat box, would have failed every placement check) -> the top
  face is the up-facing pixels (|normal z| > 0.9 from neighbouring depth pixels); (3) mask erosion
  shrank widths by ~5 mm -> add the eroded pixels back (2 x 2 px at the measured distance). These are
  physical/sensor constants, not object knowledge, and do not change any pass tolerance.
- **Prompts contain no object words:** the parse prompt's example uses placeholders `<A>` / `<B>`.
- **Unsupported tasks** ("pick" only, anything not pick-and-place with a destination) go to
  `ask_user`; the scripted responder answers "cancel" -> CANCELLED (counts as "asked").
- **View for verification:** the arm is parked at ARM_HOME with J1 in {170, 90, -90, 0, -170} deg,
  the first pose whose collision spheres stay >= 2 cm off the camera-to-target line and whose joint
  move is collision-free; initial observation uses the camera's image-centre floor point.
- **Directional placement search:** candidate spots in the reliable pick/place ring 0.45-0.90 m
  (grasp-feasibility map), gap 20 mm, +10 mm steps to +100 mm, directions rotated 0/+-15/+-30 deg;
  `next_to` tries the four sides, closest to mid-reach (0.65 m) first.
- **YOLO labels: one polygon per instance** (conversion fix, allowed by the brief): the converter wrote
  one YOLO instance per mask fragment (+5.5 % lines: 83,502 vs 79,104 annotations in train), which
  would teach the detector to split occluded objects. Fragments are now joined by zero-width bridges
  (as Ultralytics' own COCO converter does); labels regenerated: 79,104 / 12,909 / 6,569 lines =
  annotations (measured). No image excluded, no chunk regenerated. Test added (tests/test_datagen.py).
- **Today's scene recaptured** with the double-render capture (20 of 21 objects visible, 1 image;
  first capture kept as `eval_layouts/today_capture1_single_render`). With 16 boxes and 3 cylinders,
  "recall >= 95 %" means every box and every cylinder must be found (15/16 = 93.8 %).
- **Smoke tests before training (pretrained detector, measured):** (1) Gemma parsed "put X on Y" as
  task "pick" and duplicated the destination in object.relation/anchor -> clearer task definitions in
  the prompt + a structural consistency check that triggers the one allowed retry (no object
  knowledge); (2) describe answers were truncated (Gemma listed un-numbered objects past the token
  limit) -> budget 80 tokens per mark + 300; the exact-ids check stays strict; (3) the robot base was
  detected and measured as an object (the simplified collision spheres are smaller than the rendered
  base) -> a fixed robot-base exclusion box (0.30 x 0.32 x 0.50 m, perception only, not a planning
  obstacle; nearest workspace object 0.34 m from the base axis); (4) the first trajectory after sim
  start ran up to 94 deg behind its plan in time (DDS discovery; the simulator replays the trajectory
  from receipt, so the joint path is unchanged) -> the agent waits for the trajectory subscriber and
  3 s of sim time before the first motion.
- **YOLO round budgets:** round 1 = 1.75 h (time-limited Ultralytics training, epochs fitted to the
  time, cosine LR), leaving <= 2.25 h for rounds 2-3 within the 4 h cap.
- **Geometry checked on real simulator depth (today's capture, ground-truth masks; ground truth used
  only to MEASURE the error, `out/demo/2026-10-14/geometry_check_today.json`):** before -> after the
  method fixes, boxes: centre xy 3-7 -> 1.7-4.6 mm, width +0.5..+3.8 -> -1.8..+1.0 mm; cylinders width
  -4..-5 -> -0.1..+0.9 mm (thin 25 mm cylinder -4.4 mm); top +1.5 mm on every object (same sign:
  likely my offline intrinsics/pose for the capture; the live run uses camera_info). Fixes are method
  changes, not tuned constants: outline from the up-facing top-face pixels (upper side-face points
  leaned the centre towards the camera); round footprints measured by radius (a percentile range
  under-measures a disc). Hidden top face (blue cube under the lime cube: centre 25 mm off) -> tilt
  is now `None` (unknown, never guessed); planning refuses to pick it or place on it, and placement
  verification fails if the placed object's top face is not visible. These do not loosen any
  tolerance; they make an unmeasurable case fail instead of passing on a guess.
- **YOLO stops after round 1:** the fixed criteria passed on today's scene (recall box 16/16,
  cylinder 3/3, precision 0.952); rounds 2-3 are only for failures. Training used 1.75 of 4 h.
  Caveat recorded: the criterion is measured on ONE image (today's layout); the 20 extra layouts
  (after the demo) are needed for a statistically meaningful number.
- **Rehearsal fix cycle 1 (of 3), bugs in new code only:** (1) Gemma described 15 of 21 objects as
  "Yellow square" from the set-of-mark image: the yellow outlines bleed into small objects after
  downscaling -> each object is described from its own crop with the surroundings dimmed and nothing
  drawn on it (same for confirmation crops); offline check on today's capture with the round-1
  detector: 8/8 rehearsal references resolved correctly, all confirmed "yes", 1 invalid answer in 38
  calls (fixed by the retry), 32 s to describe 21 objects (`out/demo/2026-10-14/vlm_check/`).
  (2) The frozen RosExecutor reads its sim clock only while it spins, so after a VLM pause the first
  step started from a stale clock ("tracking 153 deg", step could end early before the settle wait)
  -> the agent spins the executor 0.3 s before each execute. Cycle 1 stopped after run 1 (CANCELLED,
  asked; scored, kept in `rehearsal/cycle1/`).
- **Rehearsal fix cycle 2 (of 3), bugs in new code only:** cycle 2 runs 1-2 succeeded (truth xy 2.6 /
  3.2 mm); run 3 FAILED honestly: the simulator's safety layer rejected the approach trajectory
  because it started at J1 = 170.0 deg, outside [-170, 170] -- my park pose was AT the J1 limit and
  the measured joint can sit a hair beyond it. The safety layer was right and is unchanged; park
  poses are now >= 10 deg inside the limits (+-160 instead of +-170; test added). Also: (a) the
  "disturbed neighbour" check flagged the support cube (its top became hidden under the placed cube,
  ground truth: it did not move) -> only objects whose top face is visible now are compared; (b)
  re-observations describe only objects that do not match a described world object (saves ~70 s per
  verification; matched objects keep their description). Cycle 2 stopped after 3 scored runs.
- **Cycle 3 results (measured, `rehearsal/cycle3/`):** i1 3/3 truth success (xy 4.0 / 2.6 / 4.3 mm);
  i2 1/3 (2 honest FAILED: resolve answers invalid 3x); i3 0/3: run 1 honest FAILED (grasp: fingers
  stopped 1.46 / 1.29 deg short of the close target, threshold 1.5 deg, unchanged), runs 2-3
  **FALSE DONE**: "orange cube" resolved to the brown cube (rendered orange-brown; described
  "brown/orange-brown"), confirm_object said "yes" (same model, correlated error — the failure mode
  ARCHITECTURE rev 3 names), the brown cube was then placed and verified geometrically on the
  cylinder. Geometric verification cannot detect a wrong object.
- **Fix cycle 3 (of 3, the last), new code only:** (1) VLM retry message names the invalid field and
  why (was "1 validation error"); resolve prompt states "id" is one number or null. (2) Square
  footprints (symmetry 90): width = mean of both measured extents (the oblique view under-reads one
  axis and over-reads the other; offline on today's capture: cubes -0.4..-1.0 mm instead of up to
  +1.0 / -3.1 mm); errs to the tighter close target. The 1.5 deg contact threshold is unchanged.
  (3) Reference cross-check: lexical candidates = objects whose VLM description contains every word of
  the reference (deterministic, the VLM's own words; only English function words are dropped). If
  the resolver's pick is not among them or several qualify, the VLM compares their isolated crops side
  by side; the choice is accepted only if it is also the single most specific description (fewest
  extra words); otherwise ambiguous -> ask the user (scripted "cancel" -> counted as asked). Offline
  on the false-done crops the visual comparison alone picked the brown cube 3/3 times
  (`vlm_check/compare_check.json`), so the agreement rule turns that case into a question instead of
  a wrong pick. Conservative by design: some correct picks will now be questions.
- **Final cycle 4 (no code change during it):** i1 3/3, i3 3/3, i2 0/3 asked, 0 false done / failed.
  **Instruction 4 not run** (brief: only if 1-3 each >= 2/3). **Instruction 2 not substituted:** the
  brief substitutes only *infeasible* instructions; instruction 2 is feasible (it planned offline and
  succeeded in cycle 3 run 3) but ambiguous to the agent's conservative rule — recorded as "asked".
  No fix cycle left; the candidate improvement (accept the visual choice when it is among the
  most-specific descriptions) is recorded in DEMO_REPORT §4 as unverified.
- **Launcher fix after the rehearsal (deliverable script, not agent code):** `launch/start_demo.sh`
  cleaned up with `kill 0`, which also killed the calling shell when started non-interactively; the
  camera viewers now run in their own process group and only that group is stopped. Re-tested: start,
  quit, nothing left running, caller unaffected (measured).
- **Run ended early (~12:50Z of the 20:19Z limit):** all steps done; the 20 extra layouts, ambiguity
  sets and Phase 2 wait for the user (brief: "after the demo").

## 2026-10-14 — Interactive clarification bug (user report) and fix
- **Root cause (from code):** `agent/run.py` built `Supervisor(tools, log, out)` without a responder,
  so the default `scripted_responder` (written for the unattended rehearsal) returned the literal
  "cancel" without printing or reading anything; `_ask` logged it as the answer -> CANCELLED. Not
  Gemma, not input handling. Even a real answer could not have continued the task (any answer other
  than "cancel" became FAILED "unhandled answer").
- **Fix:** `agent/ask.py` (new): Question / Answer / Candidate, deterministic answer parsing (ids,
  unique description words, nearest / farthest, explicit cancel words), `TerminalResponder` (one
  shared stdin reader thread for the instruction prompt and questions; invalid answers explained and
  re-asked; EOF and timeout = documented cancel policy, timeout = Budgets.question_timeout_s 120 s),
  `ScriptedResponder` (rehearsal only, tagged "scripted"). `Supervisor` requires a responder (no
  silent default). On several matches (object or destination) the user's choice is used and the task
  continues in the same run; the VLM does not re-confirm a person's choice. Questions show ids, the
  VLM's description, measured position, distance, size and a picture of the candidates. Model output
  is never an answer. Unsupported instructions, "object = destination" and an unverifiable relational
  reference are now honest FAILED messages (rephrase), not fake questions. `agent.run --ask
  terminal|scripted` (default terminal); the rehearsal harness passes `--ask scripted`.
- **Tests:** tests/test_agent_ask.py (new, 30 incl. parametrised) + updated tests/test_agent.py: 57
  passed in .venv-seg; /usr/bin/python3 -m pytest 129 passed, 2 skipped. Manipulation pipeline,
  planner, executor, safety and verification untouched; manifest OK. **Live sim test not run:** the
  user's own demo session was running on the GPU (my attempt stopped at Gemma loading with CUDA OOM
  before creating any ROS node; the user's session was not touched).
