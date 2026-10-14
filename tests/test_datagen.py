"""Synthetic-data pieces that run without Isaac Sim: textures, scene sampling (reproducibility, split
separation, evaluation-object exclusion, held objects), COCO / YOLO conversion, camera effects."""

import json
from pathlib import Path

import numpy as np
import pytest

from datagen.coco import annotations_from_ids, coco_to_yolo, mask_polygons
from datagen.layout import ObjectSpec, Material, SceneSampler, apply_rgb_effects, look_at, world_extent
from datagen.textures import KINDS, make_texture

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "datagen" / "config_v1.json").read_text())
FAKE_ASSETS = {name: (0.08, 0.12, 0.06) for name in CFG["ycb"]["train"] + CFG["ycb"]["novel"]} | {"torus": (1.5, 1.5, 0.5)}


@pytest.fixture(scope="module")
def sampler():
    return SceneSampler(CFG, FAKE_ASSETS)


@pytest.mark.parametrize("kind", KINDS)
def test_textures(kind):
    a = make_texture(np.random.default_rng(1), kind, 64)
    b = make_texture(np.random.default_rng(1), kind, 64)
    assert a.shape == (64, 64, 3) and a.dtype == np.uint8 and np.array_equal(a, b)


def test_sampling_is_reproducible(sampler):
    a, b = sampler.sample("train", 7), sampler.sample("train", 7)
    assert a.seed == b.seed and [o.position for o in a.objects] == [o.position for o in b.objects]
    assert sampler.sample("train", 8).seed != a.seed


def test_novel_split_never_uses_training_families(sampler):
    train = {o.family for i in range(30) for o in sampler.sample("train", i).objects}
    novel = {o.family for i in range(30) for o in sampler.sample("test_novel", i).objects}
    assert train and novel and not (train & novel)


def test_eval_object_copies_are_detected(sampler):
    from sim.scene_objects import OBJECTS
    red = OBJECTS["red_cube"]
    assert sampler._is_eval_copy("box", red.size, red.color)
    assert not sampler._is_eval_copy("box", red.size, (0.1, 0.9, 0.9))
    assert not sampler._is_eval_copy("box", (0.03, 0.09, 0.05), red.color)


def test_no_sampled_object_copies_an_eval_object(sampler):
    for i in range(40):
        for o in sampler.sample("train", i).objects:
            assert not sampler._is_eval_copy(o.family, o.dims, o.material.colour)


def test_objects_stand_on_the_floor_or_a_support(sampler):
    for i in range(20):
        sc = sampler.sample("train", i)
        by_uid = {o.uid: o for o in sc.objects}
        for o in sc.objects:
            if o.held:
                continue
            if o.stacked_on is None:
                assert o.position[2] == 0.0
            else:
                s = by_uid[o.stacked_on]
                assert o.position[2] == pytest.approx(s.position[2] + world_extent(s)[2])


def test_held_object_sits_between_the_pads():
    s = SceneSampler(CFG | {"objects": CFG["objects"] | {"held_probability": 1.0}}, FAKE_ASSETS)
    from control.kinematics import Kinematics
    kin = Kinematics()
    found = 0
    for i in range(15):
        sc = s.sample("train", i)
        held = [o for o in sc.objects if o.held]
        if not held:
            continue
        found += 1
        o = held[0]
        tcp = kin.fk(np.asarray(sc.robot_q))[:3, 3]
        assert np.hypot(o.position[0] - tcp[0], o.position[1] - tcp[1]) < 1e-6  # centred under the TCP
        assert o.dims[0] <= 0.08  # fits between the pads
    assert found >= 5


def test_look_at_points_the_optical_axis_at_the_target():
    T = look_at([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    z = T[:3, 2]
    assert np.allclose(z, np.array([-1.0, 0.0, -1.0]) / np.sqrt(2))
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))


def test_masks_to_coco_and_yolo(tmp_path):
    ids = np.zeros((40, 60), np.uint16)
    ids[5:15, 5:25] = 3  # 10 x 20 = 200 px
    ids[20:30, 30:31] = 4  # thin: 10 px (below the minimum)
    anns, nxt = annotations_from_ids(ids, 1, 1, {3: {"family": "box"}, 4: {"family": "box"}}, min_pixels=30)
    assert len(anns) == 1 and anns[0]["area"] == 200 and anns[0]["bbox"] == [5.0, 5.0, 20.0, 10.0] and nxt == 2
    poly = np.asarray(anns[0]["segmentation"][0]).reshape(-1, 2)
    assert poly[:, 0].min() == 5 and poly[:, 0].max() == 24 and poly[:, 1].min() == 5 and poly[:, 1].max() == 14
    coco = {"images": [{"id": 1, "file_name": "images/a_00001.jpg", "width": 60, "height": 40}], "annotations": anns}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    assert coco_to_yolo(tmp_path / "c.json", tmp_path / "labels") == 1
    line = (tmp_path / "labels" / "a_00001.txt").read_text().split()
    assert line[0] == "0" and all(0.0 <= float(v) <= 1.0 for v in line[1:])


def test_mask_polygons_split_parts():
    m = np.zeros((20, 40), bool)
    m[2:8, 2:8] = True
    m[10:18, 25:35] = True  # an occluder cut the object in two
    assert len(mask_polygons(m)) == 2


def test_rgb_effects_are_deterministic_and_bounded():
    img = np.full((32, 32, 3), 128, np.uint8)
    fx = {"noise_sigma": 5.0, "blur_px": 2.0, "motion_blur": True, "motion_angle_deg": 30.0, "exposure": 1.3, "gamma": 0.9, "rng_seed": 4}
    a, b = apply_rgb_effects(img, fx), apply_rgb_effects(img, fx)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.mean() > 128  # brighter with exposure 1.3


def test_world_extent_of_a_tipped_object():
    o = ObjectSpec(1, "box", None, (0.1, 0.04, 0.02), (0, 1), 0.0, (0, 0, 0), Material((1, 1, 1), 0.5, 0.0))
    assert np.allclose(sorted(world_extent(o)), [0.02, 0.04, 0.1]) and world_extent(o)[2] == pytest.approx(0.1)


def test_yolo_merges_mask_parts_into_one_instance(tmp_path):
    """An occluder splitting a mask must not make two YOLO instances."""
    import json

    import cv2
    from datagen.coco import merge_parts

    m = np.zeros((60, 80), np.uint8)
    m[10:30, 5:25] = 1
    m[10:30, 50:70] = 1
    parts = mask_polygons(m)
    assert len(parts) == 2
    poly = merge_parts([np.asarray(p, float).reshape(-1, 2) for p in parts])
    r = np.zeros_like(m)
    cv2.fillPoly(r, [np.round(poly).astype(np.int32)], 1)
    assert (r & m).sum() >= 0.9 * m.sum() and (r & ~m.astype(bool)).sum() < 0.1 * m.sum()
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 80, "height": 60}],
            "annotations": [{"image_id": 1, "segmentation": parts}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    assert coco_to_yolo(tmp_path / "c.json", tmp_path / "labels") == 1
