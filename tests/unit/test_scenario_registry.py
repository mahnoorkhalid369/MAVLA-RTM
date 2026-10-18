"""scenarios/ as the one registry of scenes (scenarios.available()): every scene is selectable the same way, and a
new one is plugged in by adding one file. The default scene refers to the frozen definition (never a copy)."""

import json

import scenarios


def test_every_scene_is_registered():
    assert scenarios.available() == {"default": "frozen", "laboratory": "procedural",
                                     "ap_categories": "json", "ap_occlusion": "json"}


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
