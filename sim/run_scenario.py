"""Live simulation of an ACTIVE-PERCEPTION TEST SCENE (sim/scenarios/<name>.json), without changing the
frozen default scene: fills sim.scene_objects.OBJECTS (the dict the frozen sim/scene.py and sim/run_live.py
read) with the scenario's objects before the frozen run_live starts, and replaces placeholder prims by YCB
meshes after the frozen setup_scene. Ground truth for every scenario object comes from run_live's own
/sim/ground_truth/objects publisher (evaluation only).

    python -m sim.run_scenario --scenario ap_occlusion [--headless]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
args, rest = ap.parse_known_args()
SPEC = json.loads((ROOT / "sim" / "scenarios" / f"{args.scenario}.json").read_text())
ASSETS = json.loads((ROOT / "data" / "assets" / "asset_dims.json").read_text())

import sim.scene_objects as so  # noqa: E402  (pure data; no Isaac import)

YCB = {}
so.OBJECTS.clear()
for o in SPEC["objects"]:
    if o["shape"] == "ycb":
        sx, sy_up, sz = ASSETS[o["asset"]]["size"]  # Axis_Aligned YCB: +Y is up
        YCB[o["name"]] = o
        # placeholder (replaced after setup_scene): the upright bounding box, so pure-data helpers stay valid
        so.OBJECTS[o["name"]] = so.SceneObject("scenario", "box", tuple(o["xy"]), (sx, sz, sy_up), (0.5, 0.5, 0.5),
                                               yaw_deg=o.get("yaw_deg", 0.0), mass_kg=0.2)
    else:
        so.OBJECTS[o["name"]] = so.SceneObject("scenario", o["shape"], tuple(o["xy"]), tuple(o["size"]), tuple(o["color"]),
                                               yaw_deg=o.get("yaw_deg", 0.0), mass_kg=o.get("mass_kg"), on=o.get("on"))

sys.argv = [sys.argv[0], *rest]  # run_live parses its own options (--headless, ...)
import sim.run_live as rl  # noqa: E402  (creates the SimulationApp)

_original_setup = rl.setup_scene


async def setup_with_meshes(*a, **k):
    await _original_setup(*a, **k)
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = stage_utils.get_current_stage()
    for name, o in YCB.items():
        info = ASSETS[o["asset"]]
        path = f"/World/Objects/{name}"
        stage.RemovePrim(path)
        sx, sy_up, sz = info["size"]
        yaw = np.radians(o.get("yaw_deg", 0.0))
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # +90 deg about x: the asset's +Y up -> world +Z
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = Rz @ Rx, [o["xy"][0], o["xy"][1], sy_up / 2 + 0.002]
        root = UsdGeom.Xform.Define(stage, path)
        root.AddTransformOp().Set(Gf.Matrix4d(*M.T.ravel()))
        inner = UsdGeom.Xform.Define(stage, f"{path}/asset")
        inner.AddTranslateOp().Set(Gf.Vec3d(*[-float(c) for c in info["centre"]]))
        inner.GetPrim().GetReferences().AddReference(str(ROOT / info["file"]))
        from pxr import Usd
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(0.2)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    print(f"[run_scenario] scene {SPEC['name']}: {len(so.OBJECTS)} objects ({len(YCB)} YCB meshes)", flush=True)


rl.setup_scene = setup_with_meshes

if __name__ == "__main__":
    try:
        rl.main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        rl.simulation_app.close()
