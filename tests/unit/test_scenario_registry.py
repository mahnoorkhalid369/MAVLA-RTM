"""scenarios/ as the one registry of scenes (scenarios.available()): every scene is selectable the same way, and a
new one is plugged in by adding one file. The default scene refers to the frozen definition (never a copy)."""

import json

import scenarios


def test_every_scene_is_registered():
    assert scenarios.available() == {"default": "frozen", "laboratory": "procedural",
                                     "ap_categories": "json", "ap_occlusion": "json", "graspable_blocks": "json"}


def test_the_default_scene_is_the_frozen_definition_not_a_copy():
    from scenarios import default
    from sim.scene_objects import OBJECTS
    objs = default.objects()
    assert set(objs) == set(OBJECTS) and all(objs[k] is OBJECTS[k] for k in OBJECTS)
    assert not (scenarios.SCENES / "default.json").exists()  # no second definition


def test_infrastructure_modules_are_not_scenarios():
    assert not set(scenarios.available()) & scenarios.INFRASTRUCTURE - {"default"}
    assert scenarios.SCENARIOS == {"laboratory": "scenarios.laboratory"}


def test_a_json_scene_is_plugged_in_by_adding_its_file(tmp_path, monkeypatch):
    (tmp_path / "my_bench.json").write_text(json.dumps({"name": "my_bench", "objects": []}))
    monkeypatch.setattr(scenarios, "SCENES", tmp_path)
    assert scenarios.kind("my_bench") == "json" and scenarios.scene_file("my_bench") == tmp_path / "my_bench.json"


def test_every_registered_scene_is_accepted_by_the_settings():
    from agent import settings
    for name in scenarios.available():
        assert not any("SCENARIO" in p for p in settings.validate({"SCENARIO": name})), name
    assert any("SCENARIO" in p for p in settings.validate({"SCENARIO": "not_plugged_in"}))


def test_graspable_blocks_fit_the_gripper_and_keep_the_default_layout():
    """graspable_blocks: the default scene without objects wider than the gripper holds (control/skills.py
    max_grip_width, 80 mm across the fingers), plus taller blocks narrower than that."""
    import json
    from sim.scene_objects import OBJECTS
    spec = json.loads(scenarios.scene_file("graspable_blocks").read_text())
    objs = {o["name"]: o for o in spec["objects"]}
    assert all(min(o["size"][:2]) <= 0.080 for o in objs.values())
    assert "magenta_box_too_wide" not in objs and set(OBJECTS) - set(objs) == {"magenta_box_too_wide"}
    tall = [o for o in objs.values() if o["name"].endswith("_tall_block")]
    assert len(tall) == 4 and all(o["size"][2] >= 0.07 and max(o["size"][:2]) < 0.080 for o in tall)
