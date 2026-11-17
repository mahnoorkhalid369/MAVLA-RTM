"""Convert real surgical-instrument scans (3D-COSI, STL in mm) into the scenarios' instrument models.

    uv run --no-project --with trimesh --with fast-simplification --with scipy --with usd-core \
        python scripts/convert_instruments.py --stl-dir data/3d_cosi/stl

(a throwaway environment with the mesh tools; the project's environments stay as they are)

For each instrument in INSTRUMENTS below it writes, into assets/instruments/:
    <name>.usd    the mesh in metres, Z up, lying flat: long axis along +y with the finger rings / handle at +y,
                  centred on its bounding box, lowest point at z = 0; a stainless-steel material
    <name>.json   what the scenarios need without opening the mesh: size, simple collision boxes, mass, source, licence

Collision boxes: the instrument is cut into SLICES pieces along its length; each piece becomes one box covering its
width and height. So the boxes follow the real outline (wide at the rings, narrow at the shaft) and stay simple
enough for the physics engine and the motion planner.

Source: 3D-COSI (Luijten et al., Scientific Data 10, 796, 2023; doi:10.5281/zenodo.10091715), CC BY 4.0.
Get the STL files: download "c_Surgical Instruments.7z" from the Zenodo record and extract the files named in
INSTRUMENTS into --stl-dir (any folder layout; files are found by name).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "instruments"
MAX_FACES = 20000  # enough detail for the cameras, small files
SLICES = 8  # collision boxes per instrument
STEEL = (0.62, 0.63, 0.66)  # linear RGB of brushed stainless steel
DENSITY = 7900.0  # kg/m3
LICENSE = "CC BY 4.0 (3D-COSI, Luijten et al. 2023, doi:10.5281/zenodo.10091715)"

# name: (scan file in 3D-COSI, title as a scrub nurse says it, family)
INSTRUMENTS = {
    "scalpel_no10": ("Inspec Scalpel Blade Number 10 RemoveHighlightsON.stl", "scalpel, No. 3 handle with No. 10 blade",
                     "scalpel"),
    "tissue_forceps": ("Inspec Anatomical Standard Tweezers 1 RemoveHighlightON.stl", "tissue forceps", "thumb_forceps"),
    "debakey_forceps": ("Inspec De Bakey Tweezers 1 RemoveHighlightON.stl", "DeBakey forceps", "thumb_forceps"),
    "straight_scissors": ("Leo Surgical Straight Sharp Scissors 1 SharpFusion.stl", "straight surgical scissors",
                          "scissors"),
    "metzenbaum_scissors": ("Leo Dissecting Scissors Metzenbaum 1 SharpFusion.stl", "Metzenbaum scissors", "scissors"),
    "curved_scissors": ("Inspec Surgical Curved Scissors 3 RemoveHighlightON.stl", "curved surgical scissors", "scissors"),
    "mosquito_hemostat": ("Inspec Arterial Clamp Halsted Mosquito 1 RemoveHighlightON.stl", "Halsted mosquito hemostat",
                          "hemostat"),
    "pean_clamp": ("Inspec Hemostatic Pean Clamp 1 RemoveHighlightON.stl", "Pean clamp", "hemostat"),
    "needle_holder": ("Leo Needle Holder Hegar 1 SharpFusion.stl", "Mayo-Hegar needle holder", "needle_holder"),
}


def lay_flat(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the scan so it lies like an instrument on a tray (see the module docstring)."""
    to_box, extents = trimesh.bounds.oriented_bounds(m)  # aligns the tightest box with the axes, centred
    m.apply_transform(to_box)
    small, mid, long = np.argsort(extents)
    turn = np.zeros((4, 4))
    turn[0, mid], turn[1, long], turn[2, small], turn[3, 3] = 1.0, 1.0, 1.0, 1.0  # x = width, y = length, z = thickness
    if np.linalg.det(turn[:3, :3]) < 0:  # keep it a rotation (a mirrored instrument would be wrong)
        turn[2] *= -1.0
    m.apply_transform(turn)
    v, length = m.vertices, np.ptp(m.vertices[:, 1])
    rings = v[v[:, 1] > v[:, 1].max() - 0.3 * length]
    tips = v[v[:, 1] < v[:, 1].min() + 0.3 * length]
    if np.ptp(rings[:, 0]) < np.ptp(tips[:, 0]):  # the wide end (finger rings, handle) goes to +y
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1)))
    lo, hi = m.bounds
    m.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    return m


def collision_boxes(m: trimesh.Trimesh) -> list[list[float]]:
    """[cx, cy, cz, sx, sy, sz] per slice along y, each from the table (z = 0) to the slice's top."""
    v = m.vertices
    edges = np.linspace(v[:, 1].min(), v[:, 1].max(), SLICES + 1)
    boxes = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = v[(v[:, 1] >= a) & (v[:, 1] <= b)]
        if len(s) < 3:
            continue
        x0, x1, top = s[:, 0].min(), s[:, 0].max(), max(s[:, 2].max(), 0.002)
        boxes.append([round(float(x), 5) for x in ((x0 + x1) / 2, (a + b) / 2, top / 2, max(x1 - x0, 0.003), b - a, top)])
    return boxes


def write_usd(m: trimesh.Trimesh, path: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Instrument")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Instrument/mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(m.vertices.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(m.faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(m.faces.reshape(-1).astype(np.int32)))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(m.vertex_normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float, m.bounds[0])), Gf.Vec3f(*map(float, m.bounds[1]))])
    mat = UsdShade.Material.Define(stage, "/Instrument/Looks/Steel")
    sh = UsdShade.Shader.Define(stage, "/Instrument/Looks/Steel/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*STEEL))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()


def convert(name: str, stl: Path) -> dict:
    scan, title, family = INSTRUMENTS[name]
    m = trimesh.load(stl, force="mesh")  # millimetres
    volume = abs(m.volume) * 1e-9  # m3, of the full scan (watertight), before simplifying
    size = np.sort(m.extents)
    while len(m.faces) > MAX_FACES:  # simplify in steps of at most 10x, and only while the instrument keeps its size
        smaller = m.simplify_quadric_decimation(face_count=max(MAX_FACES, len(m.faces) // 10))
        stalled = len(smaller.faces) > 0.9 * len(m.faces)
        shrunk = np.abs(np.sort(smaller.extents) / size - 1).max() > 0.01  # e.g. a thin blade tip collapsing
        if stalled or shrunk:  # keep the last version that is still the real instrument
            break
        m = smaller
    m.apply_scale(0.001)  # mm -> m
    m = lay_flat(m)
    write_usd(m, OUT / f"{name}.usd")
    spec = {"name": name, "title": title, "family": family,
            "size_m": [round(float(x), 4) for x in np.ptp(m.bounds, axis=0)],
            "collision_boxes_m": collision_boxes(m), "mass_kg": round(volume * DENSITY, 4),
            "faces": int(len(m.faces)), "source": f"3D-COSI: {scan}", "license": LICENSE}
    (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stl-dir", type=Path, required=True, help="folder holding the extracted 3D-COSI STL files")
    ap.add_argument("--only", nargs="+", help="instrument names to convert (default: all)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {p.name: p for p in a.stl_dir.rglob("*.stl")}
    for name in a.only or INSTRUMENTS:
        scan = INSTRUMENTS[name][0]
        if scan not in files:
            print(f"{name}: {scan!r} not found under {a.stl_dir}")
            continue
        s = convert(name, files[scan])
        print(f"{name:20s} {s['size_m'][1] * 1000:5.0f} x {s['size_m'][0] * 1000:4.0f} x {s['size_m'][2] * 1000:4.1f} mm  "
              f"{s['mass_kg'] * 1000:5.1f} g  {s['faces']} faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
