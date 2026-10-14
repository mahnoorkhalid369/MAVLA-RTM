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
