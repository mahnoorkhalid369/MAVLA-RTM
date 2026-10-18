# Hardcoded object knowledge in the current code (inventory, 2026-10-14)

> **Historical record.** This document describes the project before the 2026-10-17 restructuring; its file paths
> may refer to the old layout (e.g. `agent/tools.py`, `agent/world.py`, `sim/scenario_lib/`). Current layout:
> `docs/architecture/MODULE_GUIDE.md` (§4 maps old paths to new). Content is kept unchanged as evidence.

Found by searching the code (from files). "Frozen" = covered by the baseline manifest
(docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256): it stays unchanged; the new pipeline
lives in new files and does not use it.

| Where | What it hardcodes | Decision |
|---|---|---|
| `perception/grounding.py: COLOUR_NAMES, CALIBRATED_COLOURS, colour_name()` | fixed colour vocabulary + reference colours tuned to this scene's lighting | **Replaced** in the new pipeline by the VLM's own description; the measured median RGB is kept as a number, not a name. File left as is (used by the old agent and the classical baseline) |
| `perception/grounding.py: GROUND_Z, ABOVE_GROUND, MAX_OBJECT_Z = 0.25` | floor height (physical) and a height band tied to this scene ("tallest stack ~0.15 m") | GROUND_Z **kept** (physical configuration: the support surface). MAX_OBJECT_Z **replaced** by TF-based robot exclusion (robot geometry at the current joint state); no fixed band |
| `perception/grounding.py: locate()` geometry (top face, footprint, yaw, grasp pose) | measurement only, no object knowledge | **Reused** (geometry stage) |
| `perception/world_memory.py: COLOURS, SHAPES, COLOUR_FAMILIES, matches()` | fixed colour / shape vocabularies and string-matching rules | **Replaced** by VLM reference resolution against stored descriptions + object ids (Phase 4); new memory module in a new file |
| `perception/classical_detector.py: MIN_Z/MAX_Z band, EXCLUDE circles, catalogue identify()` | scene height band, fixed xy circles for robot base and camera stand, OBJECTS catalogue lookup | **Frozen as the comparison baseline** (unchanged). New pipeline: masks from YOLO-seg, exclusion from the robot model + TF, no catalogue |
| `ros_nodes/vlm_agent_node.py: MAX_OBJECT_TOP = 0.14, "microphone" label filter, measured_label()` | height cutoff for props, a name-based filter, colour-name relabelling | **Replaced** when the agent moves to the new pipeline (TF exclusion, VLM descriptions); old node left as is until then |
| `vlm/prompts.py: "<colour> <shape>", "shapes: cube, box, cylinder, sphere", few-shot colour examples` | fixed vocabulary in prompts | **Replaced** by new prompts (free-text description, set-of-mark ids, strict JSON) in new files |
| `experiments/known_pose_pick_place.py: grasp_for(), symmetry_of(), OBJECTS sizes/heights` | catalogue sizes for grasp width, height, symmetry | **Frozen** (Baseline A uses known poses by design). The new pipeline's experiments take width, height and symmetry from measured geometry (Phase 7, new file) |
| `experiments/payload_study.py`, `scripts/validate_cameras_phaseA.py` (OBJECTS) | catalogue for names / masses in reports | **Kept**: evaluation side only (ground truth for scoring), never pipeline input |
| `sim/scene_objects.py: OBJECTS, obstacle_boxes()` | the evaluation scene definition and its ground-truth boxes | **Kept** (frozen): it defines the evaluation scene and ground truth; the new pipeline never reads it. Fixture boxes (camera stand, microphone) **kept as physical configuration** for collision checking until TF-based fixture geometry replaces them |
| `control/skills.py: GROUND_Z`, gripper calibration in `control/robot.py` | support surface, gripper geometry | **Kept** (physical configuration) |
| `control/collision.py`, joint/speed/payload limits | robot and safety configuration | **Kept** (physical / safety configuration) |
