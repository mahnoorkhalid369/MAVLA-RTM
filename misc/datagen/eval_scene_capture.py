"""Ground-truth masks for the EVALUATION scene (sim/scene.py), never used for training.

Builds the evaluation scene unchanged, applies the semantic label class "object" to each of its
objects at runtime (sim/scene.py and sim/scene_objects.py are frozen and not edited), and renders
table-camera frames: RGB, modal instance masks (object uid = index in sim/scene_objects.OBJECTS + 1),
depth, the camera pose and the objects' ground-truth poses.

    .venv/bin/python datagen/eval_scene_capture.py --frames 1        # -> data/perception/eval_layouts/today/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--name", default="today")
ap.add_argument("--layout", type=int, default=None, help="re-arrange the objects with datagen/eval_layouts.py (index)")
args = ap.parse_args()
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import carb  # noqa: E402

for key, value in (("/rtx/materialDb/syncLoads", True), ("/rtx/hydra/materialSyncLoads", True),
                   ("/omni.kit.plugin/syncUsdLoads", True), ("/rtx-transient/resourcemanager/texturestreaming/async", False),
                   ("/rtx-transient/resourcemanager/enableTextureStreaming", False)):
    carb.settings.get_settings().set(key, value)  # no frame before materials are ready (pilot finding)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.experimental.utils import semantics as semantics_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from control.robot import table_camera_pose  # noqa: E402
from datagen.coco import CATEGORIES, annotations_from_ids  # noqa: E402
from sim.scene import OBJECTS, add_table_camera, setup_scene  # noqa: E402

NAMES = list(OBJECTS)


def main() -> int:
    SimulationManager.setup_simulation(dt=1 / 120, device="cuda")
    app.run_coroutine(setup_scene())
    cam_path, _ = add_table_camera()
    stage = stage_utils.get_current_stage()
    layout = None
    if args.layout is not None:  # move the objects (before physics starts); the frozen scene files are unchanged
        from datagen.eval_layouts import base_height, rearranged_layout
        from pxr import Gf, UsdGeom

        layout = rearranged_layout(args.layout)
        for name, p in layout.items():
            o = OBJECTS[name]
            z = base_height(layout, name) + o.height / 2 + 0.001
            half = np.radians(p["yaw_deg"]) / 2
            for op in UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/Objects/{name}")).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(type(op.Get())(p["xy"][0], p["xy"][1], z))
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    op.Set(type(op.Get())(float(np.cos(half)), Gf.Vec3f(0, 0, float(np.sin(half)))))
    for name in NAMES:
        semantics_utils.add_labels(stage.GetPrimAtPath(f"/World/Objects/{name}"), labels="object", taxonomy="class")
    app_utils.play()
    for _ in range(60):  # let the objects settle under physics
        app.update()
    rp = rep.create.render_product(cam_path, (1280, 720))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_ins = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    for a in (a_rgb, a_ins, a_dep):
        a.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    out = ROOT / "data" / "perception" / "eval_layouts" / args.name
    for sub in ("images", "masks", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    objects = RigidPrim([f"/World/Objects/{n}" for n in NAMES])
    coco = {"images": [], "annotations": [], "categories": CATEGORIES,
            "info": {"layout": args.name, "source": "sim/scene.py", "rearranged_index": args.layout, "arrangement": layout}}
    nxt = 1
    for k in range(args.frames):
        for _ in range(2):  # second frame: everything loaded and in sync (pilot finding)
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        rgb = np.asarray(a_rgb.get_data())[..., :3]
        ins = a_ins.get_data()
        ids = np.asarray(ins["data"])
        uid_img = np.zeros(ids.shape, np.uint16)
        for iid, prim_path in ins["info"]["idToLabels"].items():
            p = str(prim_path)
            if "/World/Objects/" in p:
                name = p.split("/World/Objects/")[1].split("/")[0]
                uid_img[ids == int(iid)] = NAMES.index(name) + 1
        pos, quat = objects.get_world_poses()
        name = f"{args.name}_{k:03d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{name}.png"), uid_img)
        depth = np.nan_to_num(np.asarray(a_dep.get_data()), nan=0.0, posinf=0.0)
        cv2.imwrite(str(out / "depth" / f"{name}.png"), np.clip(depth * 1000, 0, 65535).astype(np.uint16))
        coco["images"].append({"id": k + 1, "file_name": f"images/{name}.png", "width": 1280, "height": 720,
                               "camera_pose_world": table_camera_pose().tolist(),
                               "objects": [{"uid": i + 1, "name": n, "position": pos.numpy()[i].tolist(), "quat_wxyz": quat.numpy()[i].tolist()}
                                           for i, n in enumerate(NAMES)]})
        anns, nxt = annotations_from_ids(uid_img, k + 1, nxt, {i + 1: {"name": n} for i, n in enumerate(NAMES)})
        coco["annotations"] += anns
        print(f"[eval-capture] {name}: {len(anns)} visible objects of {len(NAMES)}", flush=True)
    (out / "annotations.json").write_text(json.dumps(coco, indent=1))
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
