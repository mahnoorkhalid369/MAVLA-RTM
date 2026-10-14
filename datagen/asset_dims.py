#!/usr/bin/env python3
"""Local bounding-box size (m, at scale 1) of every downloaded YCB / Isaac shape asset, cached as
data/assets/asset_dims.json for the scene sampler. Needs pxr: run with the Isaac Sim venv.

    .venv/bin/python datagen/asset_dims.py
"""
import json
from pathlib import Path

from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "assets" / "isaac_6.1" / "Isaac" / "Props"
dims = {}
for f in sorted(list((BASE / "YCB" / "Axis_Aligned").glob("*.usd")) + list((BASE / "Shapes").glob("*.usd"))):
    st = Usd.Stage.Open(str(f))
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    dims[f.stem] = {"size": [round((r.GetMax()[i] - r.GetMin()[i]) * mpu, 5) for i in range(3)],
                    "centre": [round((r.GetMax()[i] + r.GetMin()[i]) / 2 * mpu, 5) for i in range(3)],
                    "file": str(f.relative_to(ROOT))}
(ROOT / "data" / "assets" / "asset_dims.json").write_text(json.dumps(dims, indent=1))
print(len(dims), "assets")
