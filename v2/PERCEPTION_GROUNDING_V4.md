# Perception, grounding and grasp validation — v4

Active policy: `transfer-2026-11-12-v4`. Restart both the agent and simulator: v4 requires the new optical-pose sensor topics. The six-item kitchen layout remains the default.

## Capture and coordinate consistency

The simulator publishes `/table_camera/optical_pose` and `/camera/optical_pose`, containing optical-camera poses in world coordinates (camera-to-world transforms) on the simulation clock. They are sensor calibration metadata, not object ground truth. The publisher reads the current Fabric/USDRT transforms, including wrist motion, rather than freezing a camera pose from USD at startup. Joint-state stamps use the same simulation clock.

The bridge buffers RGB, depth, both CameraInfo messages, optical poses and joints. RGB/depth exposures must agree within 3 ms, share an optical frame and image dimensions, and have matching intrinsics. Capture uses nearby stable robot/gripper feedback and a nearby optical pose (within 50 ms). Row padding, BGR/RGB encoding and depth byte order are handled explicitly.

Before accepting a table observation, perception verifies a broad horizontal work surface against the calibrated table plane, z=0 (12 mm maximum discrepancy). A coherent coordinate jump among at least three unchanged silhouettes also rejects the observation before any world IDs are allocated or updated. A simulator clock reset invalidates the old session. Wrist extrinsics are checked against capture-time joint kinematics and the camera mount.

Historical v3 table captures did not save depth/extrinsics, so the original 94 mm Y / 155 mm Z jump cannot be attributed conclusively to a particular publisher or camera edit. V4 removes the static-pose/latest-message handoff and rejects this failure pattern; it does not claim to have reconstructed missing historical evidence.

Every observation now saves raw RGB PNG, compressed depth, intrinsics, camera pose and robot state, plus a `sensor_bundle` event. Failed wrist inspections retain their captures too.

## Instances, IDs and model images

Robot points are removed using collision geometry plus URDF visual primitives at the captured joints. Numbered masks use the cleaned pixels. Contained label/duplicate masks are consolidated. A smaller adjacent round cap can join a taller round body when image contact, measured position and support-height tests agree. Ordinary stacked boxes remain separate. These geometric rules improve the existing detector; they are not a replacement segmentation model or a guarantee for arbitrary occlusions.

The session keeps persistent object IDs. Whole-frame association remains one-to-one, with mask overlap, geometry and appearance checks. Each fresh model input lists only its currently visible IDs.

| Consumer | Images/evidence |
|---|---|
| Master | Plain current table RGB, user request, execution record and worker reports |
| Manipulation grounding | Plain table context and up to nine isolated whole-object crops per board, with separated `#ID` headers |
| Binding confirmation | Separate selected source/destination crops from that same observation |
| Motion planner | Fresh table RGB-D geometry; another table observation after model inference must confirm the selected identities and poses before motion |
| Wrist inspection | Fresh wrist RGB-D, associated with the selected table object; local obstacles supplement the scene before replanning |
| Placement verification | Released-object size/appearance, destination geometry, open-gripper feedback and two stable table observations |
| Final goal check | Fresh plain table RGB, plus wrist RGB when an object is still held |

The full-scene numbering also avoids overlapping label boxes and connects displaced labels to their outlines. The dashboard draws labels on their matching captured RGB while stationary; during motion it shows live video without old masks. Captured views are explicitly captioned.

Invalid IDs trigger one bounded grounding retry with current crop boards. If still invalid, execution stops with a selection/segmentation failure. It cannot turn “invalid ID” into “object absent,” or repeatedly ask the user to locate an object that is visibly present. Genuine semantic ambiguity can still ask the user.

## Support, grasp and placement geometry

The lowest visible point is no longer assumed to be the support plane when the lower body is hidden. Local horizontal depth patches establish support, excluding the object's own pixels. This distinction is essential for an apple's hidden bottom and for a tray, whose internal floor is not the surface supporting the tray itself. Unconfirmed support keeps conservative bounds and blocks a new grasp.

Body bounds include the full measured surface. Regular cartons use their solid top/footprint for centered grips. Round objects use consistent fitted circular arcs across several pad-height slices, so a visible front half of a bottle is not interpreted as a thin slab or narrow cap. Irregular objects retain the material-aware central grasp. Wrist refinement preserves confirmed support and never silently shrinks payload bounds.

Grasp selection evaluates complete hover/descent/close/lift candidates. A transfer candidate additionally needs carrying, lowering, release/retreat and a planned return path. An orientation that only reaches the hover pose cannot prevent other valid top-down orientations from being tried. Every accepted candidate still passes the existing collision and joint-limit checks; failure returns the actual blocking evidence.

Ordinary `in`/`on` placement requires the whole object footprint inside the destination, a confirmed compatible support, consistent full-body height, confirmed release and stability across two observations. Flat-top tilt is checked only when a plane is supported by the depth data. An apple's curved surface does not establish rigid-body tilt. Exact-pose tasks can retain `PickPlan.strict_pose=True`; missing orientation evidence fails such a task. The existing numeric exact-pose tolerances were not increased. Depth-only presence is diagnostic and cannot independently certify object identity or placement.

## Reproduce the checks

Run the focused regression tests with the segmentation Python environment. The v4 tests cover timestamp mismatch, calibration rejection, row padding/byte order, coherent coordinate jumps, label collisions, cap/label consolidation, support inference, curved placement, invalid-ID recovery, changed geometry during grounding and full-path orientation search.

Replay any complete v4 capture without sending arm commands:

```bash
cd v2
PYTHONPATH=src .venv-seg/bin/python scripts/replay_rgbd.py path/to/table_001_depth.npz
```

Use `--camera wrist` for wrist captures. The adjacent `_rgb.png` is used unless RGB is embedded in the NPZ. `--source-ids 3,4 --destination-id 1` additionally checks independent offline transfer plans for those displayed IDs; it does not simulate a sequence or pack multiple objects into the tray.

The revised-kitchen software RGB-D capture and replay results are in `out/perception_v4_review/`. The review includes numbered observations, a candidate board, full events and `report.json`. Real YOLO inference on this software-rendered scene accepted six items and two containers, rejected the robot detection, consolidated the milk-pack fragments, and retained IDs over four observations. It is an input-path check, not a Gemma accuracy or live physics result.

The optional ROS callback/bundle integration test runs on isolated domain 87:

```bash
source /opt/ros/jazzy/setup.bash
ROS_LOG_DIR=/tmp/ros-v4-capture ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="src:$PYTHONPATH" \
.venv-seg/bin/python -m pytest tests/integration/test_rgbd_capture.py -q
```

Live Isaac sensor publication, Gemma grounding accuracy and physical pick/place success still require an end-to-end simulator run. Offline/replay tests do not establish those results.
