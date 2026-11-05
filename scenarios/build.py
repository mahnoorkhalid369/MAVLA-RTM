"""Build a scenarios Scenario into the Isaac Sim stage (import only after the SimulationApp exists).

    install(sc)                 # BEFORE the frozen setup_scene: fill sim.scene_objects.OBJECTS with placeholders,
                                # so run_live's ground-truth publisher covers every scenario object by name
    await setup_scene()         # frozen: robot + gripper + microphone + floor + dome light + placeholder objects
    populate(sc)                # replace placeholders by compound rigid bodies; add fixtures, decor, labels, lights

Objects: /World/Objects/<id> = one rigid body (root Xform: RigidBodyAPI + MassAPI) with primitive colliders and
visual-only text labels as children. Fixtures: /World/Scenario/Fixtures/<id>, static colliders (no rigid body).
Decor: /World/Scenario/Decor/<id>, visual only. Nothing here touches the robot, cameras or control code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scenarios.core import Label, Part, Scenario

ROOT = Path(__file__).resolve().parents[1]
TEXTURES = ROOT / "out" / "scenario_textures"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# visual material presets: roughness, metallic, specular (UsdPreviewSurface, specular workflow)
LOOKS = {"matte": (1.0, 0.0, 0.0), "plastic": (0.45, 0.0, 0.25), "glass": (0.12, 0.0, 0.5), "metal": (0.32, 1.0, 0.5),
         "cardboard": (0.9, 0.0, 0.05), "painted": (0.6, 0.0, 0.15), "epoxy": (0.35, 0.0, 0.2), "rubber": (0.95, 0.0, 0.02),
         "wood": (0.75, 0.0, 0.1), "foam": (1.0, 0.0, 0.0), "ceramic": (0.25, 0.0, 0.4)}


def install(sc: Scenario) -> None:
    """Placeholders in the frozen scene's OBJECTS dict (pure data): bounding cylinder / box per object."""
    import sim.scene_objects as so
    so.OBJECTS.clear()
    for o in sc.objects:
        fpx, fpy = o.footprint
        shape = "cylinder" if all(p.shape == "cylinder" for p in o.parts if p.collide) else "box"
        top = max((p for p in o.parts if p.collide), key=lambda p: p.pos[2] + p.size[2] / 2)
        so.OBJECTS[o.id] = so.SceneObject(o.role or "scenario", shape, (o.pos[0], o.pos[1]), (fpx, fpy, o.height), top.color,
                                          yaw_deg=o.yaw_deg, mass_kg=o.mass_kg)


# ------------------------------------------------------------------------------------------- USD helpers
class _Builder:
    def __init__(self, stage) -> None:
        self.stage = stage
        self.looks: dict = {}
        self.phys: dict = {}

    def look(self, rgb, kind: str, texture: Path | None = None):
        from pxr import Gf, Sdf, UsdShade
        from sim.scene import srgb_to_linear
        key = (tuple(round(c, 3) for c in rgb), kind, str(texture) if texture else "")
        if key in self.looks:
            return self.looks[key]
        name = "L" + hashlib.md5(repr(key).encode()).hexdigest()[:10]
        path = f"/World/Scenario/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        rough, metal, spec = LOOKS.get(kind, LOOKS["matte"])
        if texture is not None:
            st = UsdShade.Shader.Define(self.stage, f"{path}/st")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(self.stage, f"{path}/tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(rgb)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(spec, spec, spec))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self.looks[key] = mat
        return mat

    def friction(self, static: float, dynamic: float):
        from pxr import UsdPhysics, UsdShade
        key = (round(static, 3), round(dynamic, 3))
        if key not in self.phys:
            m = UsdShade.Material.Define(self.stage, f"/World/Scenario/Physics/F{len(self.phys)}")
            api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
            api.CreateStaticFrictionAttr(static)
            api.CreateDynamicFrictionAttr(dynamic)
            api.CreateRestitutionAttr(0.0)
            self.phys[key] = m
        return self.phys[key]

    @staticmethod
    def xform(prim_xf, pos, yaw_deg: float, scale=None) -> None:
        from pxr import Gf
        prim_xf.ClearXformOpOrder()
        prim_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        h = np.radians(yaw_deg) / 2
        prim_xf.AddOrientOp().Set(Gf.Quatf(float(np.cos(h)), 0.0, 0.0, float(np.sin(h))))
        if scale is not None:
            prim_xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))

    def part(self, parent: str, i: int, p: Part, collide: bool, phys_mat=None) -> None:
        from pxr import UsdGeom, UsdPhysics, UsdShade
        path = f"{parent}/p{i}"
        if p.shape == "cylinder":
            g = UsdGeom.Cylinder.Define(self.stage, path)
            g.GetRadiusAttr().Set(p.size[0] / 2)
            g.GetHeightAttr().Set(p.size[2])
            g.GetAxisAttr().Set("Z")
            g.GetExtentAttr().Set([(-p.size[0] / 2, -p.size[0] / 2, -p.size[2] / 2), (p.size[0] / 2, p.size[0] / 2, p.size[2] / 2)])
            self.xform(g, p.pos, p.yaw_deg)
        else:
            g = UsdGeom.Cube.Define(self.stage, path)
            g.GetSizeAttr().Set(1.0)
            self.xform(g, p.pos, p.yaw_deg, p.size)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(self.look(p.color, p.material))
        if not p.visible:
            g.MakeInvisible()  # collider only (PhysX still uses it)
        if collide and p.collide:
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            if phys_mat is not None:
                UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    def label(self, parent: str, i: int, lb: Label) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        tex = label_texture(lb)
        w, h = lb.size
        a = np.radians(lb.yaw_deg)
        right, up = {"z": ((np.cos(a), np.sin(a), 0), (-np.sin(a), np.cos(a), 0)), "-y": ((1, 0, 0), (0, 0, 1)),
                     "y": ((-1, 0, 0), (0, 0, 1)), "x": ((0, 1, 0), (0, 0, 1)), "-x": ((0, -1, 0), (0, 0, 1))}[lb.normal]
        c, r, u = np.asarray(lb.pos, float), np.asarray(right, float) * w / 2, np.asarray(up, float) * h / 2
        pts = [c - r - u, c + r - u, c + r + u, c - r + u]
        m = UsdGeom.Mesh.Define(self.stage, f"{parent}/label{i}")
        m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
        m.GetFaceVertexCountsAttr().Set([4])
        m.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        m.GetDoubleSidedAttr().Set(False)  # seen only from the front (no mirrored text through open fixtures)
        m.GetExtentAttr().Set([Gf.Vec3f(*map(float, np.min(pts, 0))), Gf.Vec3f(*map(float, np.max(pts, 0)))])
        st = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(self.look(lb.bg, "matte", tex))


def label_texture(lb: Label) -> Path:
    """Render the label text to a PNG (cached by content)."""
    from PIL import Image, ImageDraw, ImageFont
    TEXTURES.mkdir(parents=True, exist_ok=True)
    w, h = lb.size
    H = 128
    W = max(32, int(round(H * w / h)))
    key = hashlib.md5(repr((lb.text, W, H, lb.fg, lb.bg)).encode()).hexdigest()[:16]
    path = TEXTURES / f"{key}.png"
    if path.exists():
        return path
    img = Image.new("RGB", (W, H), tuple(int(c * 255) for c in lb.bg))
    d = ImageDraw.Draw(img)
    size = int(H * 0.72)
    while size > 8:
        font = ImageFont.truetype(FONT, size)
        x0, y0, x1, y1 = d.textbbox((0, 0), lb.text, font=font)
        if x1 - x0 <= W * 0.9 and y1 - y0 <= H * 0.8:
            break
        size -= 2
    d.text(((W - (x1 - x0)) / 2 - x0, (H - (y1 - y0)) / 2 - y0), lb.text, font=font, fill=tuple(int(c * 255) for c in lb.fg))
    img.save(path)
    return path


# ------------------------------------------------------------------------------------------- populate
def populate(sc: Scenario) -> dict:
    """Replace the frozen setup's placeholder objects; add fixtures, decor, labels and lights. Returns prim paths."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = stage_utils.get_current_stage()
    b = _Builder(stage)
    UsdGeom.Scope.Define(stage, "/World/Scenario")
    paths = {"objects": [], "fixtures": [], "decor": []}
    for o in sc.objects:
        path = f"/World/Objects/{o.id}"
        stage.RemovePrim(path)
        root = UsdGeom.Xform.Define(stage, path)
        b.xform(root, o.pos, o.yaw_deg)
        fr = b.friction(*o.friction)
        for i, p in enumerate(o.parts):
            b.part(path, i, p, True, fr)
        for i, lb in enumerate(o.labels):
            b.label(path, i, lb)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(float(o.mass_kg))
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        paths["objects"].append(path)
    for group, items, collide in (("Fixtures", sc.fixtures, True), ("Decor", sc.decor, False)):
        for f in items:
            path = f"/World/Scenario/{group}/{f.id}"
            root = UsdGeom.Xform.Define(stage, path)
            b.xform(root, f.pos, f.yaw_deg)
            fr = b.friction(0.8, 0.7) if collide else None
            for i, p in enumerate(f.parts):
                b.part(path, i, p, collide, fr)
            for i, lb in enumerate(f.labels):
                b.label(path, i, lb)
            paths[group.lower()].append(path)
    if sc.view.get("table"):  # a work cell on a table: the frozen floor plane stays the physics support at z = 0
        # (the table top) but is hidden, the room floor is drawn below; the frozen camera stand is replaced by the
        # scenario's frame (the camera itself stays where control/robot.py TABLE_CAMERA puts it)
        ground = stage.GetPrimAtPath("/World/GroundPlane")
        if ground.IsValid():
            for prim in Usd.PrimRange(ground):
                if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdGeom.Imageable(prim).MakeInvisible()
        for path in ("/World/TableCamera/pole", "/World/TableCamera/foot", "/World/Microphone"):  # not on this table
            prim = stage.GetPrimAtPath(path)  # (the microphone's box stays a planning obstacle: config/perception.py)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        nogo = UsdGeom.Mesh(stage.GetPrimAtPath("/World/NoGoZone"))
        if nogo.GetPrim().IsValid():  # the shaded no-go floor ends at the table edge
            from control.workspace import no_go_clipped
            (cx, cy), (L, W) = sc.view["table"]["centre"], sc.view["table"]["size"]
            pts, counts, idx = no_go_clipped(cx - L / 2, cx + L / 2, cy - W / 2, cy + W / 2, z=0.0009)
            nogo.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in pts])
            nogo.GetFaceVertexCountsAttr().Set(counts)
            nogo.GetFaceVertexIndicesAttr().Set(idx)
            tint = sc.view["table"].get("no_go_color")
            shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/NoGoZone/Shader"))
            if tint and shader.GetPrim().IsValid():  # a tint of the table, not a colour block the detector reads as a thing
                linear = Gf.Vec3f(*[c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in tint])
                shader.GetInput("diffuseColor").Set(linear)
                nogo.GetDisplayColorAttr().Set([linear])
    for i, lt in enumerate(sc.lights):
        path = f"/World/Scenario/Lights/l{i}"
        if lt.kind == "rect":
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateWidthAttr(lt.size[0])
            light.CreateHeightAttr(lt.size[1])
        elif lt.kind == "sphere":
            light = UsdLux.SphereLight.Define(stage, path)
            light.CreateRadiusAttr(lt.size[0])
        else:
            light = UsdLux.DistantLight.Define(stage, path)
        light.CreateIntensityAttr(lt.intensity)
        light.CreateColorAttr(Gf.Vec3f(*lt.color))
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*lt.pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*lt.rotation_deg))
    return paths


def overview_camera(sc: Scenario, path: str = "/World/Scenario/OverviewCamera") -> str:
    """A documentation camera (not a robot sensor) framing the workstation for screenshots."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom
    stage = stage_utils.get_current_stage()
    cam = UsdGeom.Camera.Define(stage, path)
    eye, target = sc.view.get("eye", (1.2, -0.5, 1.1)), sc.view.get("target", (0.0, 0.5, 0.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    cam.GetFocalLengthAttr().Set(18.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * 9 / 16)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 20.0))
    return path

