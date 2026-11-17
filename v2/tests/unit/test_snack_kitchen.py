"""Kitchen scenario regressions: valid USD identifiers, source assets, and task geometry."""

import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest

from scenarios import generate, layout, table_camera
from scenarios.assets import resolve
from scenarios.snack_kitchen import MODEL_SPECS, ROOT


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_household_items_have_valid_usd_names_and_resolvable_visuals(difficulty):
    sc = generate("snack_kitchen", difficulty, 42)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", o.id) for o in sc.objects)
    for owner in sc.objects + sc.fixtures + sc.decor:
        for part in owner.parts:
            if part.shape == "mesh":
                assert Path(resolve(part.asset)).is_file()
    ids = {o.id for o in sc.objects}
    assert all(set(o.attributes.get("look_alikes", [])) <= ids for o in sc.objects)
    assert {"milk", "water bottle", "cracker box", "fruit cup", "apple"} <= {o.attributes["name"] for o in sc.objects}
    assert not {"juice_bottle", "food_004_sugar_box", "food_009_gelatin_box"} & ids
    milk = next(o for o in sc.objects if o.id == "milk")
    water = next(o for o in sc.objects if o.id == "water_bottle")
    assert milk.parts[0].shape == "box" and water.parts[0].shape == "cylinder"
    assert any(lb.text == "MILK" for lb in milk.labels)
    assert any(lb.text == "WATER" for lb in water.labels)
    # The standard grasp must contact the broad bottle body, rather than only surround its narrow neck.
    from control.kinematics import Kinematics
    from control.robot import fingertip_below_tcp
    from control.skills import SkillPlanner
    planner = SkillPlanner(Kinematics())
    width = min(water.footprint)
    tcp_z, _ = planner._grasp_z(water.height, water.height, width)
    body_top = water.parts[0].pos[2] + water.parts[0].size[2] / 2
    assert body_top - (tcp_z - fingertip_below_tcp(width)) >= planner.p.min_contact
    assert min(next(o for o in sc.objects if o.id == "apple").footprint) < 0.080


def test_kitchen_keeps_the_existing_workcell_independent():
    before = generate("workcell", "easy", 42).metadata()
    sc = generate("snack_kitchen", "easy", 42)
    assert generate("workcell", "easy", 42).metadata() == before
    assert sc.view["table"]["height"] == 0.90
    assert not sc.view["show_workspace_overlay"]
    assert not json.loads(layout("snack_kitchen"))["regions"]
    assert table_camera("snack_kitchen") == table_camera("workcell")


def test_hard_tray_starts_empty_and_accepts_a_collision_free_placement():
    from scenarios.checks import placement_feasibility
    scene = generate("snack_kitchen", "hard", 42)
    assert all(o.support == "table" for o in scene.objects)
    result = placement_feasibility(scene)
    assert result["serving_tray"]["feasible_yaws"], result["serving_tray"]["reason"]


def test_normalized_nvidia_models_have_the_declared_centred_bounds():
    pytest.importorskip("pxr", reason="run this check in the USD/Isaac Python environment")
    from pxr import Usd, UsdGeom
    for name, spec in MODEL_SPECS.items():
        stage = Usd.Stage.Open(str(ROOT / spec["wrapper"]))
        assert stage is not None, name
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetDefaultPrim()).ComputeAlignedRange()
        assert np.allclose(bounds.GetSize(), spec["size_m"], atol=1e-5), name
        assert np.allclose(bounds.GetMidpoint(), [0, 0, 0], atol=1e-5), name


def test_exported_usd_has_finite_support_and_independent_prop_physics(tmp_path):
    pytest.importorskip("pxr", reason="run this check in the USD/Isaac Python environment")
    from pxr import Usd, UsdGeom, UsdPhysics
    script = ROOT / "scripts/export_snack_kitchen.py"
    spec = importlib.util.spec_from_file_location("export_snack_kitchen", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "kitchen.usda"
    module.export_scene(path, "easy", 42, include_robot=False)
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert stage.GetPrimAtPath("/World/CountertopSupport/p0").HasAPI(UsdPhysics.CollisionAPI)
    for owner in generate("snack_kitchen", "easy", 42).objects:
        prim = stage.GetPrimAtPath(f"/World/Objects/{owner.id}")
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(owner.mass_kg)
