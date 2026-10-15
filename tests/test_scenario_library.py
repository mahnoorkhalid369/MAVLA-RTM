"""Scenario library (sim/scenario_lib): deterministic generation, spawn geometry, difficulty features,
ground-truth metadata and the easy level's grasp feasibility with the unchanged planner. No simulator."""

import numpy as np
import pytest

from sim.scenario_lib import DIFFICULTIES, SCENARIOS, generate
from sim.scenario_lib.checks import grasp_feasibility, static_report, top_visibility
from sim.scenario_lib.core import REACH_MAX, penetration, world_parts

CASES = [(n, d) for n in SCENARIOS for d in DIFFICULTIES]


@pytest.mark.parametrize("name,difficulty", CASES)
def test_same_seed_same_scene_other_seed_other_layout(name, difficulty):
    a, b, c = generate(name, difficulty, 3), generate(name, difficulty, 3), generate(name, difficulty, 4)
    assert a.metadata() == b.metadata()
    assert a.metadata()["objects"] != c.metadata()["objects"]


@pytest.mark.parametrize("name,difficulty", CASES)
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_spawn_geometry_is_clean(name, difficulty, seed):
    """No interpenetration at spawn, frozen props kept clear, decor out of reach, only marked objects out of reach."""
    rep = static_report(generate(name, difficulty, seed), reach=False)
    assert rep["problems"] == []


@pytest.mark.parametrize("name,difficulty", CASES)
def test_metadata_is_complete_ground_truth(name, difficulty):
    m = generate(name, difficulty, 42).metadata()
    for k in ("scenario_id", "version", "seed", "difficulty", "objects", "fixtures", "regions"):
        assert m[k] not in (None, [], "")
    for o in m["objects"]:
        assert {"id", "category", "spawn_pose", "dimensions", "mass_kg", "attributes", "parts"} <= set(o)
        assert o["mass_kg"] > 0 and o["dimensions"]["height"] > 0
    assert len({o["id"] for o in m["objects"]}) == len(m["objects"])


def test_difficulty_is_configuration_not_code():
    e, m, h = (generate("laboratory", d, 42) for d in DIFFICULTIES)
    containers = lambda sc: [o for o in sc.objects if o.category in ("sample_tube", "sample_vial")]  # noqa: E731
    assert [len(containers(sc)) for sc in (e, m, h)] == [8, 10, 12]
    roles = lambda sc: {o.role for o in sc.objects}  # noqa: E731
    assert not {"occluded", "unreachable", "occupies_destination", "distractor"} & roles(e)
    assert {"occupies_destination", "distractor"} <= roles(m) and not {"occluded", "unreachable"} & roles(m)
    assert {"occluded", "unreachable", "occupies_destination"} <= roles(h)
    assert [o.id for o in h.objects if o.role == "occluded"] == ["vial_hidden"]
    assert [o.id for o in h.objects if o.role == "unreachable"] == ["vial_far"]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_lab_has_six_functional_zones_with_labelled_regions(difficulty):
    sc = generate("laboratory", difficulty, 42)
    assert [z["id"] for z in sc.zones] == ["incoming", "processing", "storage", "inspection", "quarantine", "buffer"]
    labels = {r.label for r in sc.regions}
    assert {"A1", "A4", "B1", "B4", "P1", "P4", "S1", "S4", "Q1", "Q2", "T1", "T3", "INSPECT"} <= labels
    assert {r.zone for r in sc.regions} == {z["id"] for z in sc.zones}
    assert all(r.accepts for r in sc.regions if r.zone != "incoming")  # every zone but incoming receives samples


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_lab_identity_needs_the_label_not_the_cap_colour(difficulty, seed):
    sc = generate("laboratory", difficulty, seed)
    samples = [o for o in sc.objects if "sample_id" in o.attributes]
    ids = [o.attributes["sample_id"] for o in samples]
    assert len(set(ids)) == len(ids) and not any(i[0] in "ABPSQT" for i in ids)  # unique, never a slot letter
    colours = [o.attributes["cap_colour"] for o in samples]
    assert min(colours.count(c) for c in set(colours)) >= 2  # no cap colour identifies a sample
    assert all(o.labels and o.labels[0].text == o.attributes["sample_id"] for o in samples)


def test_lab_visibility_matches_the_design_including_the_robot():
    e, h = generate("laboratory", "easy", 42), generate("laboratory", "hard", 42)
    assert all(v == 1.0 for v in top_visibility(e).values())
    vis = top_visibility(h)
    assert vis["vial_hidden"] == 0.0 and all(v > 0 for k, v in vis.items() if k != "vial_hidden")
    # the robot itself hides part of the bench: the old processing-tray spot is invisible with the robot only
    from sim.scenario_lib.checks import _ray_hits_spheres, _robot_spheres
    from control.robot import table_camera_pose
    c, r = _robot_spheres()
    assert _ray_hits_spheres(np.array([-0.62, 0.08, 0.06]), table_camera_pose()[:3, 3], c, r)


def test_lab_unreachable_object_is_beyond_reach_and_all_others_inside():
    h = generate("laboratory", "hard", 42)
    for o in h.objects:
        assert (np.hypot(*o.pos[:2]) > REACH_MAX) == (o.role == "unreachable"), o.id


def test_lab_containers_fit_the_gripper_and_stand_in_their_holes():
    sc = generate("laboratory", "hard", 7)
    for o in sc.objects:
        if o.category in ("sample_tube", "sample_vial"):
            assert max(o.footprint) <= 0.080
    for rack_id in ("incoming_rack", "processing_rack", "storage_rack"):
        rack = next(f for f in sc.fixtures if f.id == rack_id)
        bars = world_parts(rack.id, rack.pos, rack.yaw_deg, rack.parts)
        for o in sc.objects:
            if o.support == rack_id:
                body = world_parts(o.id, o.pos, o.yaw_deg, o.parts)[0]
                assert all(penetration(body, b) < 1e-6 for b in bars)


def test_easy_lab_every_container_has_a_planned_grasp():
    """Reachable grasp candidates with the project's own planner + collision checker (control/, unchanged)."""
    g = grasp_feasibility(generate("laboratory", "easy", 42), yaws=(0.0, 90.0))
    assert all(v["feasible_yaws"] for v in g.values()), {k: v["reason"] for k, v in g.items() if not v["feasible_yaws"]}


def test_label_textures_render(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from sim.scenario_lib import build
    from sim.scenario_lib.core import Label
    monkeypatch.setattr(build, "TEXTURES", tmp_path)
    p = build.label_texture(Label("A1", (0, 0, 0), (0.03, 0.02)))
    from PIL import Image
    im = np.asarray(Image.open(p))
    assert im.shape[0] == 128 and im.std() > 10  # text drawn on the background
