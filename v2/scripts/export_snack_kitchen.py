"""Author a standalone snack-kitchen USD without starting Isaac Sim or requiring a GPU.

From v2: .venv/bin/python scripts/export_snack_kitchen.py --difficulty easy --seed 42
The live simulator uses the same scenario through launch/sim.sh --scenario snack_kitchen.
The export includes standard USD physics for props and a static robot visual at HOME;
launch/sim.sh constructs the controlled robot articulation and ROS cameras separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def robot_visual(stage) -> None:
    """Reference the actual arm and gripper geometry, posed without physics stepping."""
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    from control.collision import CollisionChecker
    from control.kinematics import Kinematics, _axis_angle
    from control.robot import ARM_HOME

    root = UsdGeom.Xform.Define(stage, "/World/RobotPreview")
    root.GetPrim().GetReferences().AddReference(str(ROOT / "assets/vt6l/usd/vt6l/vt6l.usda"))
    root.GetPrim().GetVariantSet("Physics").SetVariantSelection("none")
    checker = CollisionChecker()
    for prim in list(stage.Traverse()):
        name = prim.GetName()
        if name not in checker.joints or not str(prim.GetPath()).startswith("/World/RobotPreview/"):
            continue
        _, origin, axis, joint_name, kind = checker.joints[name]
        transform = origin.copy()
        if kind in ("revolute", "continuous"):
            angle = float(ARM_HOME[int(joint_name[1]) - 1])
            transform[:3, :3] = origin[:3, :3] @ _axis_angle(axis, angle)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="preview").Set(Gf.Matrix4d(*transform.T.ravel()))
    gripper = UsdGeom.Xform.Define(stage, "/World/GripperPreview")
    gripper.GetPrim().GetReferences().AddReference(str(ROOT / "assets/robotiq_2f85/Robotiq_2F_85_edit.usd"))
    gripper.ClearXformOpOrder()
    T = Kinematics(tip="tool0").fk(ARM_HOME)
    a = np.radians(90.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T[:3, :3] = T[:3, :3] @ R
    gripper.AddTransformOp(opSuffix="preview").Set(Gf.Matrix4d(*T.T.ravel()))
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(("/World/RobotPreview", "/World/GripperPreview")):
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)


def export_scene(destination: Path, difficulty: str, seed: int, include_robot=True) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils
    from scenarios import generate
    from scenarios.assets import resolve
    from scenarios.build import _Builder
    from scenarios.checks import static_report
    from scenarios.core import Obj, Part

    sc = generate("snack_kitchen", difficulty, seed)
    missing = sorted({resolve(p.asset) for f in sc.objects + sc.fixtures + sc.decor for p in f.parts
                      if p.shape == "mesh" and not Path(resolve(p.asset)).is_file()})
    manifest = json.loads((ROOT / "assets/snack_kitchen/nvidia_manifest.json").read_text())
    missing += [spec["path"] for spec in manifest["files"]
                if not (ROOT / "data/assets/nvidia_residential" / spec["path"]).is_file()]
    if missing:
        raise FileNotFoundError("Kitchen assets are missing. Run scripts/fetch_snack_assets.py.\n" + "\n".join(missing))
    report = static_report(sc, reach=False)
    if report["problems"]:
        raise ValueError("Invalid scene geometry: " + "; ".join(report["problems"]))
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    builder = _Builder(stage)
    for group, owners, collision in (("Objects", sc.objects, True),
                                     ("Scenario/Fixtures", sc.fixtures, True), ("Scenario/Decor", sc.decor, False)):
        for owner in owners:
            path = f"/World/{group}/{owner.id}"
            xf = UsdGeom.Xform.Define(stage, path)
            builder.xform(xf, owner.pos, owner.yaw_deg)
            friction = builder.friction(*owner.friction) if isinstance(owner, Obj) \
                else builder.friction(0.8, 0.7) if collision else None
            for index, part in enumerate(owner.parts):
                builder.part(path, index, part, collision, friction)
            for index, label in enumerate(owner.labels):
                builder.label(path, index, label)
            if group == "Objects":
                UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
                UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(owner.mass_kg)
    # The live scene's floor plane provides this support; the exported island needs its own finite collider.
    (cx, cy), (length, depth) = sc.view["table"]["centre"], sc.view["table"]["size"]
    support = Part("box", (cx, cy, -0.025), (length, depth, 0.05), (0.9, 0.9, 0.9), visible=False)
    builder.part("/World/CountertopSupport", 0, support, True, builder.friction(0.8, 0.7))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateIntensityAttr(500.0)
    for index, light in enumerate(sc.lights):
        lamp = UsdLux.RectLight.Define(stage, f"/World/Lights/Rect{index}")
        lamp.CreateIntensityAttr(light.intensity)
        lamp.CreateColorAttr(Gf.Vec3f(*light.color))
        lamp.CreateWidthAttr(light.size[0])
        lamp.CreateHeightAttr(light.size[1])
        xf = UsdGeom.Xformable(lamp)
        xf.AddTranslateOp().Set(Gf.Vec3d(*light.pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*light.rotation_deg))
    for name, eye, target in (("Overview", sc.view["eye"], sc.view["target"]),
                              ("TableCamera", *sc.view["table_camera"])):
        camera = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        camera.AddTransformOp().Set(view.GetInverse())
        camera.CreateFocalLengthAttr(18.0 if name == "Overview" else 15.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * 9 / 16)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
    if include_robot:
        robot_visual(stage)
    stage.GetRootLayer().customLayerData = {"scenario": sc.id, "difficulty": difficulty, "seed": seed,
                                          "robotPreviewOnly": bool(include_robot)}
    UsdUtils.ModifyAssetPaths(stage.GetRootLayer(),
                             lambda path: os.path.relpath(Path(path).resolve(), destination.parent)
                             if Path(path).is_absolute() else path)
    stage.GetRootLayer().Save()
    destination.with_suffix(".json").write_text(json.dumps(sc.metadata(), indent=2) + "\n")
    destination.with_suffix(".checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"usd": str(destination), "objects": len(sc.objects), "fixtures": len(sc.fixtures),
            "geometry_problems": report["problems"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--without-robot", action="store_true")
    args = ap.parse_args(argv)
    destination = args.out or ROOT / "out/scenarios/snack_kitchen" / f"snack_kitchen_{args.difficulty}_s{args.seed}.usda"
    print(json.dumps(export_scene(destination, args.difficulty, args.seed, not args.without_robot), indent=2))


if __name__ == "__main__":
    main()
