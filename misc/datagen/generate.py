"""Render a synthetic class-agnostic instance-segmentation dataset in Isaac Sim (Replicator annotators).

Scenes come from datagen/layout.py (seeded per image); this script builds them in USD, poses the
VT6L + 2F-85, renders RGB + instance masks (modal) + depth, applies camera effects and writes:
  <out>/<split>/images/<split>_<index>.jpg      RGB with camera effects
  <out>/<split>/masks/<split>_<index>.png       16-bit instance ids = object uid (0 = background)
  <out>/<split>/depth/<split>_<index>.png       16-bit depth in mm (val / test_novel only)
  <out>/<split>/annotations.json                COCO (one category "object"; extra per-object fields)
Labelled: every generated object (incl. one held in the gripper). Unlabelled background: floor,
robot, gripper, camera stand, microphone.

Run with the Isaac Sim venv, simulation NOT running elsewhere (12 GB GPU):
    .venv/bin/python datagen/generate.py --pilot                       # 200 images -> data/perception/synthetic_v1_pilot
    .venv/bin/python datagen/generate.py --split train --start 0 --count 6000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "datagen" / "config_v1.json"))
ap.add_argument("--split", choices=["train", "val", "test_novel"], default="train")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=10)
ap.add_argument("--pilot", action="store_true", help="160 train + 20 val + 20 test_novel into <name>_pilot")
ap.add_argument("--out", default=None)
ap.add_argument("--debug", action="store_true", help="dump the instance-segmentation id maps per image")
ap.add_argument("--align-test", action="store_true", help="re-render each scene 4 more times, saving RGB with mask edges")
ap.add_argument("--check-stale", action="store_true", help="re-render each scene once more and count changed pixels")
args = ap.parse_args()

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import carb  # noqa: E402

# Synthetic data must not render a frame before new materials / textures are ready (pilot, measured:
# stale floor material and missing objects in the RGB with asynchronous loading)
for key, value in (("/rtx/materialDb/syncLoads", True), ("/rtx/hydra/materialSyncLoads", True),
                   ("/omni.kit.plugin/syncUsdLoads", True), ("/rtx-transient/resourcemanager/texturestreaming/async", False),
                   ("/rtx-transient/resourcemanager/enableTextureStreaming", False)):
    carb.settings.get_settings().set(key, value)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.experimental.utils import semantics as semantics_utils  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

from control.robot import ARM_JOINTS, GRIPPER_JOINT  # noqa: E402
from datagen.coco import CATEGORIES, annotations_from_ids  # noqa: E402
from datagen.layout import SceneSampler, apply_rgb_effects, camera_pose, object_rotation, world_extent  # noqa: E402
from datagen.textures import KINDS, make_texture  # noqa: E402
from sim.scene import ROBOT_PATH, ROBOT_USD, _d435_intrinsics, add_microphone, attach_gripper  # noqa: E402

CFG = json.loads(Path(args.config).read_text())
OUT = Path(args.out) if args.out else ROOT / "data" / "perception" / (CFG["name"] + ("_pilot" if args.pilot else ""))
ASSETS = json.loads((ROOT / "data" / "assets" / "asset_dims.json").read_text())
GEN = "/World/Gen"
SHAPE_ASSET = {"box": "cube", "plate": "cube", "cylinder": "cylinder", "sphere": "sphere", "cone": "cone", "torus": "torus"}


# ------------------------------------------------------------------------------------- stage
def build_base_stage():
    stage_utils.create_new_stage(template="default stage")
    stage = stage_utils.get_current_stage()
    robot = stage_utils.add_reference_to_stage(usd_path=str(ROBOT_USD), path=ROBOT_PATH)
    robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    attach_gripper()
    add_microphone()
    # no table-camera hardware: the data camera sits where it is (pilot: its body showed in table views)
    for p in ("/World/defaultLight", "/Environment/defaultLight", "/Environment"):
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)
    cam = UsdGeom.Camera.Define(stage, "/World/DataCam")
    _d435_intrinsics(cam)
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    h = CFG["floor"]["size_m"] / 2
    floor.CreatePointsAttr([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr([(0, 0, 1)] * 4)
    floor.SetNormalsInterpolation("vertex")
    UsdGeom.PrimvarsAPI(floor).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    return stage, cam, floor, dome


def material(stage, path: str, colour, roughness, metallic, texture: Path | None = None, uv_scale=1.0, uv_rot=0.0,
             tint=(1, 1, 1)) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in colour]))
    else:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        xf = UsdShade.Shader.Define(stage, f"{path}/xf")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(uv_scale, uv_scale))
        xf.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(float(uv_rot))
        tex = UsdShade.Shader.Define(stage, f"{path}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(c) for c in tint], 1.0))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def prism_mesh(stage, path: str, sides: int, apex: bool = False) -> UsdGeom.Mesh:
    """Unit-size (1 x 1 x 1, centred) prism with `sides` sides, or a pyramid if apex; with UVs."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / sides
    ring = np.stack([0.5 * np.cos(a), 0.5 * np.sin(a)], axis=1)
    pts = [(x, y, -0.5) for x, y in ring]
    faces, counts = [], []
    if apex:
        pts.append((0.0, 0.0, 0.5))
        for i in range(sides):
            faces += [i, (i + 1) % sides, sides]
            counts.append(3)
    else:
        pts += [(x, y, 0.5) for x, y in ring]
        for i in range(sides):
            j = (i + 1) % sides
            faces += [i, j, sides + j, sides + i]
            counts.append(4)
        faces += list(range(sides, 2 * sides))  # top cap, counter-clockwise seen from above (+z normal)
        counts.append(sides)
    faces += list(range(sides - 1, -1, -1))  # bottom cap, clockwise seen from above (-z normal)
    counts.append(sides)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr("none")
    uv = [(p[0] + 0.5, p[2] + 0.5 if not apex else p[1] + 0.5) for p in pts]
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(uv)
    return m


def asset_ref(stage, path: str, name: str, size, unit_scale: np.ndarray):
    """Reference an asset under path, re-centred so its bbox centre is at the local origin, scaled."""
    info = ASSETS[name]
    outer = UsdGeom.Xform.Define(stage, path)  # our scale + re-centring; the asset keeps its own ops below
    outer.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in unit_scale]))
    outer.AddTranslateOp(opSuffix="centre").Set(Gf.Vec3d(*[-float(c) for c in info["centre"]]))
    inner = stage.DefinePrim(f"{path}/asset", "Xform")
    inner.GetReferences().AddReference(str(ROOT / info["file"]))
    return outer


def build_object(stage, o, tex_dir: Path):
    """One labelled object: an Xform at the bbox centre with the object's rotation; geometry centred."""
    path = f"{GEN}/obj_{o.uid:03d}"
    root = UsdGeom.Xform.Define(stage, path)
    ext = world_extent(o)
    if o.held:
        R = np.asarray(o.extra["R_world"])
        centre = np.asarray(o.position) + R @ [0, 0, o.dims[2] / 2]
    else:
        R = object_rotation(o)
        centre = np.asarray(o.position) + [0, 0, ext[2] / 2]
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, centre
    root.AddTransformOp().Set(Gf.Matrix4d(*M.T.ravel()))
    d = np.asarray(o.dims)
    geo = f"{path}/geo"
    fam = o.family
    if fam in ("box", "plate", "cylinder", "cone", "sphere"):
        a = ASSETS[SHAPE_ASSET[fam]]["size"]
        asset_ref(stage, geo, SHAPE_ASSET[fam], a, d / np.maximum(a, 1e-6))
    elif fam == "torus":
        a = np.asarray(ASSETS["torus"]["size"])
        asset_ref(stage, geo, "torus", a, np.full(3, o.scale))
    elif fam in ("ycb_train", "ycb_novel"):
        asset_ref(stage, geo, o.asset, ASSETS[o.asset]["size"], np.full(3, o.scale))
    elif fam == "capsule":
        c = UsdGeom.Capsule.Define(stage, geo)
        c.CreateRadiusAttr(float(d[0] / 2))
        c.CreateHeightAttr(float(max(d[2] - d[0], 0.001)))
        c.CreateAxisAttr("Z")
    elif fam in ("hex_prism", "tri_prism", "pyramid"):
        m = prism_mesh(stage, geo, {"hex_prism": 6, "tri_prism": 3, "pyramid": 4}[fam], apex=fam == "pyramid")
        UsdGeom.Xformable(m).AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in d]))
    elif fam == "l_shape":
        t = o.extra["thickness"]
        cube = ASSETS["cube"]["size"]
        for k, (size, off) in enumerate((((d[0], t, d[2]), (0, -(d[1] - t) / 2, 0)), ((t, d[1], d[2]), (-(d[0] - t) / 2, 0, 0)))):
            part = UsdGeom.Xform.Define(stage, f"{geo}_{k}")
            part.AddTranslateOp().Set(Gf.Vec3d(*off))
            asset_ref(stage, f"{geo}_{k}/m", "cube", cube, np.asarray(size) / np.maximum(cube, 1e-6))
    else:
        raise ValueError(fam)
    mat = o.material
    if not mat.keep_asset:
        tex = tex_dir / f"object_{mat.texture:03d}.png" if mat.texture is not None else None
        m = material(stage, f"{GEN}/Looks/obj_{o.uid:03d}", mat.colour, mat.roughness, mat.metallic, tex,
                     uv_scale=1.0, tint=mat.colour if tex else (1, 1, 1))
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(m, UsdShade.Tokens.strongerThanDescendants)
    semantics_utils.add_labels(root.GetPrim(), labels="object", taxonomy="class")
    # the object's uid as a second label: the annotator's id -> prim path map is empty for prims created
    # after the first frame (measured), the semantic labels are not
    semantics_utils.add_labels(root.GetPrim(), labels=f"uid_{o.uid:03d}", taxonomy="uid")
    return path


def set_lights(stage, dome, spec):
    dome.CreateIntensityAttr(spec["dome_intensity"])
    dome.CreateColorAttr(Gf.Vec3f(*spec["dome_tint"]))
    for k, L in enumerate(spec["extra"]):
        p = f"{GEN}/light_{k}"
        if L["kind"] == "sphere":
            light = UsdLux.SphereLight.Define(stage, p)
            light.CreateRadiusAttr(0.1)
        elif L["kind"] == "rect":
            light = UsdLux.RectLight.Define(stage, p)
            light.CreateWidthAttr(0.6)
            light.CreateHeightAttr(0.6)
        else:
            light = UsdLux.DistantLight.Define(stage, p)
            light.CreateAngleAttr(1.0)
        light.CreateIntensityAttr(L["intensity"] if L["kind"] != "distant" else L["intensity"] / 20)
        light.CreateColorAttr(Gf.Vec3f(*L["colour"]))
        x = UsdGeom.Xformable(light.GetPrim())
        if L["kind"] == "distant":
            el, az = L["angle_deg"]
            x.AddRotateXYZOp().Set(Gf.Vec3f(90 - el, 0, az))
        else:  # point the rect light at the work area
            x.AddTranslateOp().Set(Gf.Vec3d(*L["position"]))
            eye = np.asarray(L["position"])
            z = eye - np.array([0.0, 0.3, 0.0])
            z /= np.linalg.norm(z)
            xa = np.cross([0, 0, 1], z)
            xa = xa / np.linalg.norm(xa) if np.linalg.norm(xa) > 1e-6 else np.array([1.0, 0, 0])
            R = np.column_stack([xa, np.cross(z, xa), z])
            q = Gf.Matrix3d(*R.T.ravel()).ExtractRotation().GetQuat()
            x.AddOrientOp().Set(Gf.Quatf(q))


def write_readme(out: Path, counts: dict, rate: float | None):
    manifest = json.loads((ROOT / "data" / "assets" / "isaac_6.1" / "ASSETS_MANIFEST.json").read_text())
    used = sorted({m["file"].split("/")[-1].split(".")[0] for m in manifest if m["file"].endswith(".usd")})
    lines = [f"# {CFG['name']} — synthetic class-agnostic instance segmentation", "",
             f"Generated by `datagen/generate.py` with `datagen/config_v1.json` (seed {CFG['seed']}); per-image seeds are "
             "sha256(seed/split/index), so any image can be regenerated.", "",
             "## Splits", *[f"- {k}: {v} images" for k, v in counts.items()], "",
             "Masks are modal (visible pixels only); one category `object`. Unlabelled: floor, robot, gripper, camera stand, microphone.",
             "The evaluation scene's layouts and exact copies of its 21 objects are excluded (datagen/layout.py: _is_eval_copy).",
             f"test_novel uses only families never used in training: {CFG['families']['novel']} (YCB: {CFG['ycb']['novel']}).", "",
             "## Asset sources and licences",
             "- Procedural shapes (capsule, hex / tri prism, pyramid, L-shape from cubes): generated by this script.",
             "- Floor and object textures: generated procedurally by `datagen/textures.py` (no third-party images).",
             "- From NVIDIA's official Isaac Sim asset server "
             "(https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/), full list with URL / size / "
             "ETag / licence note in `data/assets/isaac_6.1/ASSETS_MANIFEST.json`:"]
    lic = {m["file"].split("/")[-1].split(".")[0]: m["licence"] for m in manifest}
    lines += [f"  - `{u}`: {lic[u]}" for u in used]
    if rate:
        lines += ["", f"Pilot render rate (measured): {rate:.2f} s per image."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------- main
def main() -> int:
    stage, cam, floor, dome = build_base_stage()
    out = OUT
    tex_dir = out / "_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    rng_t = np.random.default_rng(CFG["seed"])
    for k in range(CFG["floor"]["texture_pool"]):
        f = tex_dir / f"floor_{k:03d}.png"
        if not f.exists():
            kind = CFG["floor"]["kinds"][k % len(CFG["floor"]["kinds"])]
            cv2.imwrite(str(f), cv2.cvtColor(make_texture(np.random.default_rng([CFG["seed"], 1, k]), kind, 512), cv2.COLOR_RGB2BGR))
    for k in range(CFG["materials"]["texture_pool"]):
        f = tex_dir / f"object_{k:03d}.png"
        if not f.exists():
            kind = KINDS[k % len(KINDS)]
            cv2.imwrite(str(f), cv2.cvtColor(make_texture(np.random.default_rng([CFG["seed"], 2, k]), kind, 256), cv2.COLOR_RGB2BGR))
    del rng_t

    SimulationManager.setup_simulation(dt=1 / 60, device="cuda")
    app_utils.play()
    app.update()
    robot = Articulation(ROBOT_PATH)
    if args.debug:  # label the robot too (debug only), so its mask can be compared with the RGB
        semantics_utils.add_labels(stage.GetPrimAtPath(ROBOT_PATH), labels="robot", taxonomy="class")
    idx = robot.get_dof_indices(ARM_JOINTS + [GRIPPER_JOINT])
    w, h = CFG["render"]["resolution"]
    rp = rep.create.render_product(str(cam.GetPath()), (w, h))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    a_ins = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    for a in (a_rgb, a_ins, a_dep):
        a.attach([rp])
    rep.orchestrator.set_capture_on_play(False)

    sampler = SceneSampler(CFG, {k: tuple(v["size"]) for k, v in ASSETS.items()})
    jobs = ([("train", i) for i in range(160)] + [("val", i) for i in range(20)] + [("test_novel", i) for i in range(20)]) if args.pilot \
        else [(args.split, i) for i in range(args.start, args.start + args.count)]
    coco: dict[str, dict] = {}
    times, stale = [], []
    for n, (split, index) in enumerate(jobs):
        t0 = time.monotonic()
        spec = sampler.sample(split, index)
        d = out / split
        for sub in ("images", "masks") + (("depth",) if split != "train" else ()):
            (d / sub).mkdir(parents=True, exist_ok=True)
        # fresh prim paths for every image: re-using a path (delete + recreate) let the renderer keep the
        # old material in the RGB (pilot, measured: the first image's floor in every frame)
        global GEN
        if stage.GetPrimAtPath(GEN):
            stage.RemovePrim(GEN)
        GEN = f"/World/Gen_{n:05d}"
        stage.DefinePrim(GEN, "Xform")
        stage.DefinePrim(f"{GEN}/Looks", "Scope")
        for o in spec.objects:
            build_object(stage, o, tex_dir)
        fl = spec.floor
        fm = material(stage, f"{GEN}/Looks/floor", (0.5, 0.5, 0.5), fl["roughness"], fl["metallic"],
                      tex_dir / f"floor_{fl['texture']:03d}.png", fl["uv_scale"] * CFG["floor"]["size_m"] / 2, fl["uv_rot_deg"], fl["tint"])
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(fm)
        set_lights(stage, dome, spec.lights)
        T = camera_pose(spec.camera) @ np.diag([1.0, -1.0, -1.0, 1.0])  # ROS optical -> USD camera (-Z forward, +Y up)
        x = UsdGeom.Xformable(cam.GetPrim())
        x.ClearXformOpOrder()
        x.AddTransformOp().Set(Gf.Matrix4d(*T.T.ravel()))
        q = np.concatenate([spec.robot_q, [spec.finger_rad]]).astype(np.float32).reshape(1, -1)
        robot.set_dof_positions(wp.array(q), dof_indices=idx)
        robot.set_dof_position_targets(wp.array(q), dof_indices=idx)
        for _ in range(3):
            app.update()
        posed = robot.get_dof_positions(dof_indices=idx).numpy()[0]
        pose_err = float(np.degrees(np.max(np.abs(posed[:6] - np.asarray(spec.robot_q)))))
        # render each scene twice and keep the second frame: the RGB annotator can lag the instance
        # masks by one frame (measured in the pilot: masks of scene N over the RGB of scene N-1)
        for _ in range(CFG["render"].get("steps_per_image", 2)):
            rep.orchestrator.step(rt_subframes=CFG["render"]["rt_subframes"], delta_time=0.0)
        if args.check_stale:  # one more render of the SAME scene must not change the image or the masks
            rgb_a, ids_a = np.asarray(a_rgb.get_data())[..., :3].astype(int), np.asarray(a_ins.get_data()["data"]).copy()
            rep.orchestrator.step(rt_subframes=CFG["render"]["rt_subframes"], delta_time=0.0)
            rgb_b, ids_b = np.asarray(a_rgb.get_data())[..., :3].astype(int), np.asarray(a_ins.get_data()["data"])
            stale.append({"index": index, "rgb_changed_px": int((np.abs(rgb_a - rgb_b).max(axis=2) > 25).sum()),
                          "mask_changed_px": int((ids_a != ids_b).sum()), "robot_pose_err_deg": round(pose_err, 3)})
        if args.align_test:  # render the same scene more times, save RGB + mask outline per step
            for k in range(4):
                rep.orchestrator.step(rt_subframes=CFG["render"]["rt_subframes"], delta_time=0.0)
                im = np.asarray(a_rgb.get_data())[..., :3].copy()
                idk = np.asarray(a_ins.get_data()["data"])
                edge = cv2.Canny(((idk * 37) % 255).astype(np.uint8), 1, 2) > 0
                im[edge] = (255, 255, 0)
                cv2.imwrite(str(out / f"align_{index}_{k}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        rgb = np.asarray(a_rgb.get_data())[..., :3]
        ins = a_ins.get_data()
        ids, info = np.asarray(ins["data"]), ins["info"]
        if args.debug:
            present = {int(v): int((ids == v).sum()) for v in np.unique(ids)}
            (out / f"debug_{split}_{index:05d}.json").write_text(json.dumps(
                {"present_ids_px": present, "idToLabels": {str(k): str(v) for k, v in info["idToLabels"].items()},
                 "idToSemantics": {str(k): str(v) for k, v in info.get("idToSemantics", {}).items()},
                 "objects": [o.uid for o in spec.objects]}, indent=1))
        uid_img = np.zeros(ids.shape, np.uint16)
        for iid, labels in info.get("idToSemantics", {}).items():
            uid = str(labels.get("uid", "")) if isinstance(labels, dict) else ""
            if uid.startswith("uid_"):
                uid_img[ids == int(iid)] = int(uid[4:].split(",")[0])
        for iid, prim_path in info["idToLabels"].items():  # fallback: prim paths, when present
            p = str(prim_path)
            if "/obj_" in p and not (uid_img[ids == int(iid)] > 0).any():
                uid_img[ids == int(iid)] = int(p.split("/obj_")[1].split("/")[0])
        name = f"{split}_{index:05d}"
        if args.debug:  # RGB with object mask edges (yellow) and robot mask edges (red)
            dbg = rgb.astype(np.uint8).copy()
            robot_px = np.zeros(ids.shape, bool)
            for iid, prim_path in info["idToLabels"].items():
                if str(prim_path).startswith(ROBOT_PATH) or "robotiq" in str(prim_path):
                    robot_px |= ids == int(iid)
            dbg[cv2.Canny(robot_px.astype(np.uint8) * 255, 50, 100) > 0] = (255, 0, 0)
            dbg[cv2.Canny((uid_img * 37 % 251).astype(np.uint8), 1, 2) > 0] = (255, 255, 0)
            cv2.imwrite(str(out / f"debug_{name}.jpg"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
        rgb_fx = apply_rgb_effects(rgb.astype(np.uint8), spec.rgb_effects)
        cv2.imwrite(str(d / "images" / f"{name}.jpg"), cv2.cvtColor(rgb_fx, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, CFG["render"]["jpeg_quality"]])
        cv2.imwrite(str(d / "masks" / f"{name}.png"), uid_img)
        if split != "train":
            depth = np.nan_to_num(np.asarray(a_dep.get_data()), nan=0.0, posinf=0.0)
            cv2.imwrite(str(d / "depth" / f"{name}.png"), np.clip(depth * 1000, 0, 65535).astype(np.uint16))
        c = coco.setdefault(split, {"images": [], "annotations": [], "categories": CATEGORIES, "next": 1,
                                    "info": {"config": CFG["name"], "seed": CFG["seed"]}})
        image_id = len(c["images"]) + 1
        Tcam = camera_pose(spec.camera)
        c["images"].append({"id": image_id, "file_name": f"images/{name}.jpg", "width": w, "height": h, "index": index,
                            "seed": spec.seed, "view": spec.camera["view"], "camera_pose_world": Tcam.tolist(),
                            "robot_q": spec.robot_q, "finger_rad": spec.finger_rad,
                            "objects": [{"uid": o.uid, "family": o.family, "asset": o.asset, "dims": o.dims, "position": o.position,
                                         "yaw_deg": o.yaw_deg, "rest": o.rest, "held": o.held, "stacked_on": o.stacked_on}
                                        for o in spec.objects]})
        info_by_uid = {o.uid: {"family": o.family, "asset": o.asset, "held": o.held, "stacked": o.stacked_on is not None}
                       for o in spec.objects}
        anns, c["next"] = annotations_from_ids(uid_img, image_id, c["next"], info_by_uid, CFG["min_visible_pixels"])
        c["annotations"] += anns
        times.append(time.monotonic() - t0)
        if n % 20 == 0 or n == len(jobs) - 1:
            print(f"[gen] {n + 1}/{len(jobs)} {name}: {len(spec.objects)} objects, {len(anns)} visible masks, "
                  f"{times[-1]:.2f} s (mean {np.mean(times):.2f} s)", flush=True)
    for split, c in coco.items():
        c.pop("next")
        f = out / split / "annotations.json"
        if f.exists() and not args.pilot:  # appending a chunk to an existing split
            old = json.loads(f.read_text())
            off_img, off_ann = len(old["images"]), (max((a["id"] for a in old["annotations"]), default=0))
            for im in c["images"]:
                im["id"] += off_img
            for a in c["annotations"]:
                a["id"] += off_ann
                a["image_id"] += off_img
            old["images"] += c["images"]
            old["annotations"] += c["annotations"]
            c = old
        f.write_text(json.dumps(c))
    if stale:
        (out / "stale_check.json").write_text(json.dumps(stale, indent=1))
        print("[gen] stale check (pixels changed by one more render): rgb max", max(x["rgb_changed_px"] for x in stale),
              "mask max", max(x["mask_changed_px"] for x in stale), "robot pose err max deg",
              max(x["robot_pose_err_deg"] for x in stale), flush=True)
    stats = {"images": len(times), "seconds_per_image_mean": float(np.mean(times)),
             "seconds_per_image_median": float(np.median(times)), "first_image_s": float(times[0])}
    (out / f"render_stats_{args.split if not args.pilot else 'pilot'}.json").write_text(json.dumps(stats, indent=1))
    counts = {s: len(json.loads((out / s / "annotations.json").read_text())["images"]) for s in coco}
    write_readme(out, counts, stats["seconds_per_image_median"] if args.pilot else None)
    print(f"[gen] done: {stats}")
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
