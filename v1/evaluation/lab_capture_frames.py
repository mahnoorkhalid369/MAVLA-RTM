"""Capture laboratory observations for OFFLINE segmentation evaluation (EVALUATION ONLY; Isaac Sim headless, no ROS,
no agent, nothing commanded: the arm's joint state is SET for camera poses, as in evaluation/validate_scenarios.py).

For each (difficulty, seed): build the scene (frozen setup_scene + scenarios/), CPU physics (the wrist camera prim
follows the arm only with CPU physics headless, see docs/scenarios/LABORATORY.md), settle, then capture views:
  table        the fixed table camera
  wrist_home   the wrist camera with the arm at HOME
  wrist_rack   the wrist camera at a viewpoint above the incoming rack (existing ViewpointPlanner, collision-checked
               against the scene's colliders; joint state set, render pose checked against FK)
Each view: rgb.png, depth_mm.png (uint16), instance.png (uint16 ids) + instance.json (id -> prim path / label),
camera.json (K, world pose, q). Ground truth: metadata.json + settled poses (scoring only).

    .venv/bin/python evaluation/lab_capture_frames.py --out out/lab_seg_frames [--difficulty ...] [--seed ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "out" / "lab_seg_frames"))
ap.add_argument("--difficulty", nargs="+", default=["easy", "medium", "hard"])
ap.add_argument("--seed", nargs="+", type=int, default=[0, 42, 1234])
args = ap.parse_args()
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import carb  # noqa: E402

for key, value in (("/rtx/materialDb/syncLoads", True), ("/rtx/hydra/materialSyncLoads", True),
                   ("/omni.kit.plugin/syncUsdLoads", True), ("/rtx-transient/resourcemanager/texturestreaming/async", False),
                   ("/rtx-transient/resourcemanager/enableTextureStreaming", False)):
    carb.settings.get_settings().set(key, value)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.experimental.utils import semantics as semantics_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from control.kinematics import Kinematics  # noqa: E402
from control.robot import ARM_HOME, ARM_JOINTS, GRIPPER_JOINT, GRIPPER_OPEN, table_camera_pose  # noqa: E402
from scenarios import generate  # noqa: E402
from scenarios.build import install, populate  # noqa: E402
from scenarios.core import scenario_parts  # noqa: E402
from sim.scene import (CAM_HFOV_DEG, ROBOT_PATH, GRIPPER_PATH, add_table_camera, add_wrist_camera, find_prim_path,  # noqa: E402
                       setup_scene)

W_IMG, H_IMG = 1280, 720
FX = W_IMG / (2 * np.tan(np.radians(CAM_HFOV_DEG) / 2))
K = [float(FX), float(FX), W_IMG / 2, H_IMG / 2]


def label_prims(sc) -> None:
    stage = stage_utils.get_current_stage()
    for o in sc.objects:
        semantics_utils.add_labels(stage.GetPrimAtPath(f"/World/Objects/{o.id}"), labels=f"object:{o.id}", taxonomy="class")
    for f in sc.fixtures:
        semantics_utils.add_labels(stage.GetPrimAtPath(f"/World/Scenario/Fixtures/{f.id}"), labels=f"fixture:{f.id}", taxonomy="class")
    for d in sc.decor:
        semantics_utils.add_labels(stage.GetPrimAtPath(f"/World/Scenario/Decor/{d.id}"), labels=f"decor:{d.id}", taxonomy="class")
    for p in (ROBOT_PATH, GRIPPER_PATH):
        semantics_utils.add_labels(stage.GetPrimAtPath(p), labels="robot", taxonomy="class")


def capture(cam_path: str, out: Path) -> dict:
    rp = rep.create.render_product(cam_path, (W_IMG, H_IMG))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    a_ins = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    for a in (a_rgb, a_dep, a_ins):
        a.attach([rp])
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
    rgb = np.asarray(a_rgb.get_data())[..., :3].astype(np.uint8)
    dep = np.nan_to_num(np.asarray(a_dep.get_data()), nan=0.0, posinf=0.0)
    ins = a_ins.get_data()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "depth_mm.png"), np.clip(dep * 1000, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(out / "instance.png"), np.asarray(ins["data"]).astype(np.uint16))
    (out / "instance.json").write_text(json.dumps({str(k): str(v) for k, v in ins["info"]["idToLabels"].items()}, indent=0))
    for a in (a_rgb, a_dep, a_ins):
        a.detach()
    rp.destroy()
    return {"rgb_std": round(float(rgb.std()), 1), "depth_valid": round(float(((dep > 0.1) & (dep < 10)).mean()), 3)}


def main() -> int:
    root = Path(args.out)
    kin = Kinematics(tip="camera_color_optical_frame")
    summary = {}
    for d in args.difficulty:
        for s in args.seed:
            sc = generate("laboratory", d, s)
            out = root / f"{d}_s{s}"
            install(sc)
            SimulationManager.setup_simulation(dt=1 / 120, device="cpu")
            app.run_coroutine(setup_scene())
            populate(sc)
            label_prims(sc)
            wrist_cam = add_wrist_camera()
            table_cam, _ = add_table_camera()
            ids = [o.id for o in sc.objects]
            bodies = RigidPrim([f"/World/Objects/{i}" for i in ids])
            app_utils.play()
            app.update()
            robot = Articulation(ROBOT_PATH)
            idx = robot.get_dof_indices(ARM_JOINTS + [GRIPPER_JOINT])

            def set_q(q):
                v = np.concatenate([q, [GRIPPER_OPEN]]).astype(np.float32).reshape(1, -1)
                if not app_utils.is_playing():
                    app_utils.play()
                robot.set_dof_positions(wp.array(v, dtype=wp.float32), dof_indices=idx)
                robot.set_dof_position_targets(wp.array(v, dtype=wp.float32), dof_indices=idx)
                for _ in range(40):
                    app.update()
            set_q(np.asarray(ARM_HOME, float))
            for _ in range(240):  # ~2 s settling
                app.update()
            pos, quat = (a.numpy() for a in bodies.get_world_poses())
            meta = sc.metadata()
            meta["settled"] = {i: {"position": pos[k].tolist(), "quat_wxyz": quat[k].tolist()} for k, i in enumerate(ids)}
            out.mkdir(parents=True, exist_ok=True)
            (out / "metadata.json").write_text(json.dumps(meta, indent=1))
            views = {}
            views["table"] = capture(table_cam, out / "table")
            (out / "table" / "camera.json").write_text(json.dumps({"K": K, "T_world_cam": table_camera_pose().tolist(),
                                                                   "q": list(map(float, ARM_HOME))}))

            def wrist_view(name, q):
                set_q(q)
                q_meas = robot.get_dof_positions(dof_indices=idx).numpy()[0][:6]
                from pxr import UsdGeom
                T_usd = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage_utils.get_current_stage().GetPrimAtPath(
                    find_prim_path(ROBOT_PATH, "camera_color_optical_frame")))).T
                T_fk = kin.fk(q_meas[:kin.lower.size])
                err = float(np.linalg.norm(T_usd[:3, 3] - T_fk[:3, 3])) * 1000
                if err > 5:
                    views[name] = {"skipped": f"render pose {err:.0f} mm from FK"}
                    return
                views[name] = capture(wrist_cam, out / name)
                views[name]["render_pose_error_mm"] = round(err, 2)
                (out / name / "camera.json").write_text(json.dumps({"K": K, "T_world_cam": T_fk.tolist(), "q": q_meas.tolist()}))
            wrist_view("wrist_home", np.asarray(ARM_HOME, float))
            from perception.viewpoints import ViewpointPlanner
            from control.collision import Box
            fx, ob = scenario_parts(sc)
            boxes = [Box(w.owner, w.centre, w.R, w.half) for w in fx + ob]
            rack = next(f for f in sc.fixtures if f.id == "incoming_rack")
            vp = ViewpointPlanner().plan(np.array([rack.pos[0], rack.pos[1], 0.11]), np.asarray(ARM_HOME, float), boxes,
                                         target_box_name="incoming_rack")
            if vp.ok:
                wrist_view("wrist_rack", vp.best.q)
            else:
                views["wrist_rack"] = {"skipped": vp.reason}
            set_q(np.asarray(ARM_HOME, float))
            app_utils.stop()
            app.update()
            summary[f"{d}_s{s}"] = views
            print(f"[capture] {d} s{s}: {views}", flush=True)
    (root / "capture_summary.json").write_text(json.dumps(summary, indent=1))
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
