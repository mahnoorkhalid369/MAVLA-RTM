"""Shared Isaac Sim scene for the VT6L scripts: robot + Robotiq 2F-85, wrist camera, test objects.

Import this only AFTER creating the SimulationApp (Isaac Sim modules require it).
"""

from __future__ import annotations

import os
from pathlib import Path

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.app
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane, Sphere
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.rendering_manager import ViewportManager
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

from control.robot import (  # noqa: F401
    ARM_HOME,
    ARM_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_JOINT,
    GRIPPER_OPEN,
    TABLE_CAMERA_POSITION,
    TABLE_CAMERA_TARGET,
)

ROOT = Path(__file__).resolve().parent.parent
# Epson VT6L built from the manufacturer's manual (assets/vt6l/vt6l.urdf -> python -m sim.import_vt6l).
# The old C8L proxy is still in usd/vt6l_proxy/ for comparison.
ROBOT_USD = ROOT / "assets" / "vt6l" / "usd" / "vt6l" / "vt6l.usda"
ROBOT_PATH = "/World/vt6l"

# Robotiq 2F-85 (Isaac Sim asset, CC-BY 4.0), "mimic" physics variant: one driven joint
# (finger_joint, 0 = open 85 mm ... 0.82 rad = closed); all other linkages follow it.
GRIPPER_USD = ROOT / "assets" / "robotiq_2f85" / "configuration" / "Robotiq_2F_85_config_physics_mimic.usda"
GRIPPER_PATH = "/World/robotiq_2f85"
GRIPPER_ROOT = f"{GRIPPER_PATH}/Robotiq_2F_85"  # articulation root inside the asset
FINGER_JOINTS = [GRIPPER_JOINT]
GRIPPER_MOUNT_YAW_DEG = 90.0  # rotation about the flange axis (see attach_gripper)

# RealSense D435 colour stream: 1280x720, 69.4 x 42.5 deg FOV, usable depth ~0.1-10 m
CAM_RESOLUTION = (720, 1280)  # (height, width)
CAM_HFOV_DEG = 69.4
CAM_CLIP = (0.1, 10.0)



from sim.scene_objects import BLOCKS, OBJECTS, WOOD_DENSITY, SceneObject  # noqa: E402,F401  (test field, pure data)


# Desk microphone, the robot's instruction source. Isaac Sim does not simulate sound: this is
# a placed prop + TF frame (microphone_link, at the capsule); audio comes from the host
# microphone via ros_nodes/microphone_node.py.
MIC_PATH = "/World/Microphone"
from sim.scene_objects import MIC_POSITION, MIC_STEM_HEIGHT  # noqa: E402  (shared with collision checks)


def _matte_material(name: str, rgb_srgb: tuple[float, float, float]) -> UsdShade.Material:
    """Matte paint (like painted wood): fully rough and no specular reflection, so a camera looking
    along the light (the table camera) sees the real colour instead of a white glare."""
    stage = stage_utils.get_current_stage()
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(rgb_srgb)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def srgb_to_linear(rgb: tuple[float, float, float]) -> list[float]:
    """USD displayColor is linear; the colours above are written as normal (sRGB) screen colours."""
    c = np.asarray(rgb, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist()


def find_prim_path(root: str, name: str) -> str:
    for prim in stage_utils.get_current_stage().Traverse():
        path = prim.GetPath().pathString
        if path.startswith(root) and prim.GetName() == name:
            return path
    raise RuntimeError(f"prim '{name}' not found under {root}")


def _grip_material() -> UsdShade.Material:
    """Rubber-like physics material for finger pads and blocks."""
    material = UsdShade.Material.Define(stage_utils.get_current_stage(), "/World/Physics/GripMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(1.2)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)
    return material


def _bind_physics_material(prim_path: str, material: UsdShade.Material) -> None:
    prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def attach_gripper() -> None:
    """Mount the Robotiq 2F-85 on the VT6L flange as part of the arm's articulation.

    Same steps as Isaac Sim's Robot Assembler: drop the gripper's own articulation root and
    join its base_link to j6_link (at tool0, the flange face) with a fixed joint.
    """
    stage = stage_utils.get_current_stage()
    stage_utils.add_reference_to_stage(usd_path=str(GRIPPER_USD), path=GRIPPER_PATH)
    root = stage.GetPrimAtPath(GRIPPER_ROOT)
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    # Rotate the gripper 90 deg about the flange axis so its fingers close along tool0 X
    # (j5 Z at J6 = 0): the side-mounted wrist camera then looks straight into the gap
    # between the pads instead of at the outside of one finger.
    mount = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), GRIPPER_MOUNT_YAW_DEG))

    # Start the gripper at the flange pose so the joint does not have to pull it into place
    tool0 = stage.GetPrimAtPath(find_prim_path(ROBOT_PATH, "tool0"))
    flange = UsdGeom.XformCache().GetLocalToWorldTransform(tool0)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(GRIPPER_PATH))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(mount * flange)

    # tool0 coincides with the j6_link frame (identity tcp offset in the URDF)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{GRIPPER_PATH}/FlangeMountJoint")
    joint.CreateBody0Rel().SetTargets([find_prim_path(ROBOT_PATH, "j6_link")])
    joint.CreateBody1Rel().SetTargets([f"{GRIPPER_ROOT}/base_link"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(mount.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def add_wrist_camera() -> str:
    """Author a D435-like USD camera on the camera_color_optical_frame; return its path."""
    optical = find_prim_path(ROBOT_PATH, "camera_color_optical_frame")
    cam_path = f"{optical}/d435_color"
    cam = UsdGeom.Camera.Define(stage_utils.get_current_stage(), cam_path)
    # ROS optical frame looks along +Z (y down); a USD camera looks along -Z (y up):
    # rotate 180 deg about X.
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    _d435_intrinsics(cam)
    return cam_path


TABLE_CAMERA_PATH = "/World/TableCamera"


def _d435_intrinsics(cam: UsdGeom.Camera) -> None:
    h_aperture = 20.955  # only the focal/aperture ratio matters for the field of view
    cam.GetHorizontalApertureAttr().Set(h_aperture)
    cam.GetVerticalApertureAttr().Set(h_aperture * CAM_RESOLUTION[0] / CAM_RESOLUTION[1])
    cam.GetFocalLengthAttr().Set(h_aperture / (2.0 * np.tan(np.deg2rad(CAM_HFOV_DEG) / 2.0)))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*CAM_CLIP))


def add_table_camera() -> tuple[str, str]:
    """Fixed table camera (eye-to-hand) on a stand: control/robot.py TABLE_CAMERA_POSITION, looking
    at TABLE_CAMERA_TARGET. Returns (USD camera path, path of its ROS optical frame prim).
    """
    stage = stage_utils.get_current_stage()
    cam = UsdGeom.Camera.Define(stage, f"{TABLE_CAMERA_PATH}/d435_color")
    # SetLookAt gives the world->camera (view) matrix in the USD camera convention (-Z forward,
    # +Y up); the prim's transform is its inverse
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*TABLE_CAMERA_POSITION), Gf.Vec3d(*TABLE_CAMERA_TARGET), Gf.Vec3d(0, 0, 1)
    )
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())
    _d435_intrinsics(cam)
    optical = UsdGeom.Xform.Define(stage, f"{cam.GetPath()}/table_camera_optical_frame")
    optical.AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))  # ROS optical: z forward, y down

    # Visible hardware (visual only, no collision): D435 body behind the lens, a pole and a foot.
    # The camera cannot see them: the body is inside the near clip, the pole is straight below it.
    body = UsdGeom.Cube.Define(stage, f"{cam.GetPath()}/body")
    body.GetSizeAttr().Set(1.0)
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.015))  # +Z = behind the lens
    body.AddScaleOp().Set(Gf.Vec3f(0.09, 0.025, 0.025))
    body.CreateDisplayColorAttr([Gf.Vec3f(0.02, 0.02, 0.02)])
    x, y, z = TABLE_CAMERA_POSITION
    pole_h = z - 0.03
    pole = UsdGeom.Cylinder.Define(stage, f"{TABLE_CAMERA_PATH}/pole")
    pole.GetRadiusAttr().Set(0.015)
    pole.GetHeightAttr().Set(pole_h)
    pole.AddTranslateOp().Set(Gf.Vec3d(x, y, pole_h / 2))
    foot = UsdGeom.Cylinder.Define(stage, f"{TABLE_CAMERA_PATH}/foot")
    foot.GetRadiusAttr().Set(0.15)
    foot.GetHeightAttr().Set(0.02)
    foot.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.01))
    for part in (pole, foot):
        part.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.15)])
    return cam.GetPath().pathString, optical.GetPath().pathString


def add_microphone() -> str:
    """Desk microphone prop (visual only); returns the path of its microphone_link frame."""
    stage = stage_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, MIC_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(*MIC_POSITION))
    dark, metal = Gf.Vec3f(0.08, 0.08, 0.09), Gf.Vec3f(0.55, 0.56, 0.58)

    def part(name: str, prim_type, z: float, color: Gf.Vec3f, **attrs) -> None:
        geom = prim_type.Define(stage, f"{MIC_PATH}/{name}")
        geom.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        for attr, value in attrs.items():
            getattr(geom, f"Get{attr}Attr")().Set(value)
        geom.GetDisplayColorAttr().Set([color])

    part("base", UsdGeom.Cylinder, 0.006, dark, Radius=0.055, Height=0.012, Axis="Z")
    part("stem", UsdGeom.Cylinder, 0.012 + MIC_STEM_HEIGHT / 2, metal, Radius=0.006, Height=MIC_STEM_HEIGHT, Axis="Z")
    capsule_z = 0.012 + MIC_STEM_HEIGHT + 0.03
    part("capsule", UsdGeom.Capsule, capsule_z, dark, Radius=0.022, Height=0.03, Axis="Z")
    part("grille", UsdGeom.Sphere, capsule_z + 0.015, metal, Radius=0.023)

    frame = UsdGeom.Xform.Define(stage, f"{MIC_PATH}/microphone_link")
    frame.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, capsule_z))
    return frame.GetPath().pathString


NO_GO_PATH = "/World/NoGoZone"
NO_GO_COLOR = (0.44, 0.34, 0.39)  # sRGB: a little darker than the floor, rose-tinted, where the arm cannot pick


def add_no_go_zone() -> str:
    """Shade the floor outside the arm's measured pick envelope (control/workspace.py): the hole round the base, the
    slice behind it and the band beyond the reach. Visual only: no collision, 0.4 mm above the floor."""
    from control.workspace import no_go_mesh
    stage = stage_utils.get_current_stage()
    points, counts, idx = no_go_mesh()
    mesh = UsdGeom.Mesh.Define(stage, NO_GO_PATH)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*srgb_to_linear(NO_GO_COLOR))])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(_matte_material("NoGoZone", NO_GO_COLOR))
    return NO_GO_PATH


async def setup_scene(block_masses: dict[str, float] | None = None) -> None:
    """Build the scene. block_masses overrides a block's mass (kg) by name."""
    stage_utils.create_new_stage(template="default stage")
    if not ROBOT_USD.exists():
        raise FileNotFoundError(f"{ROBOT_USD} is missing: convert the VT6L URDF first with `python -m sim.import_vt6l`")
    robot_prim = stage_utils.add_reference_to_stage(usd_path=str(ROBOT_USD), path=ROBOT_PATH)
    # The imported asset keeps physics in a variant set; no variant = no physics
    robot_prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    attach_gripper()
    add_microphone()
    GroundPlane("/World/GroundPlane")
    if os.environ.get("NO_GO_ZONE", "1") != "0":  # NO_GO_ZONE=0 leaves the floor plain
        add_no_go_zone()
    DomeLight("/World/DomeLight").set_intensities(np.array([1000.0]))

    material = _grip_material()
    block_masses = block_masses or {}
    for name, obj in OBJECTS.items():
        path = f"/World/Objects/{name}"
        (x, y), (sx, sy, sz) = obj.xy, obj.size
        position = [x, y, obj.base_z + sz / 2]
        yaw = np.radians(obj.yaw_deg) / 2
        orientation = [np.cos(yaw), 0.0, 0.0, np.sin(yaw)]  # wxyz
        colour = [srgb_to_linear(obj.color)]
        if obj.shape == "box":
            Cube(path, sizes=1.0, scales=[sx, sy, sz], positions=position, orientations=orientation, colors=colour)
        elif obj.shape == "cylinder":
            Cylinder(path, radii=sx / 2, heights=sz, axes="Z", positions=position, orientations=orientation,
                     colors=colour)
        else:
            Sphere(path, radii=sx / 2, positions=position, colors=colour)
        GeomPrim(path, apply_collision_apis=True)
        RigidPrim(path, masses=block_masses.get(name, obj.mass))
        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(_matte_material(name, obj.color))
        _bind_physics_material(path, material)
    for pad in ("left_inner_finger", "right_inner_finger"):
        _bind_physics_material(f"{GRIPPER_ROOT}/{pad}", material)

    await omni.kit.app.get_app().next_update_async()
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.3, 1.6, 0.9], target=[0.0, 0.45, 0.2])


def save_frame(out_dir: Path, index: int, rgb: np.ndarray, depth: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb[..., :3]).save(out_dir / f"rgb_{index:04d}.png")
    # RealSense convention: uint16 depth in millimetres (depth_scale = 0.001), 0 = no data
    depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0) * 1000.0
    Image.fromarray(np.clip(depth_mm, 0, 65535).astype(np.uint16)).save(out_dir / f"depth_{index:04d}.png")
