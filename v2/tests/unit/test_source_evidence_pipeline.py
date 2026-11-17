"""Actual back-of-box failure and negative controls for source confirmation and free placement."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agent import master, workers
from agent.plan_validation import complete_transfers
from agent.tools import Tools
from control.motion import PlanningError
from control.skills import Held, grip_rotation
from perception.geometry import back_project
from perception.source_evidence import foreground, crop_quality
from perception.viewpoints import ViewpointPlanner
from perception.world_model import ObjectRecord, WorldModel
from tests.support.fakes import geom
from tests.unit.test_master_workers import FakeGemma, FakeScene, FakeTools, FakePickPlace, log
from tests.unit.test_source_side_recovery import side_tools


def test_saved_cheezit_body_binds_after_independent_support_filtering():
    root = Path(__file__).parents[1] / "fixtures"
    metadata = json.loads((root / "cheezit_side_foreground.json").read_text())
    data = np.load(root / "cheezit_side_foreground.npz")
    frame = NS(depth=data["depth"], K=data["K"])
    mask, pose = data["labels"] == metadata["source_mark"], data["T_world_cam"]
    world = WorldModel()
    world.objects[3] = ObjectRecord(3, metadata["geometry"])
    for i, g in enumerate(metadata["other_geometry"], 100):
        world.objects[i] = ObjectRecord(i, g)
    raw = back_project(mask, frame.depth, frame.K, pose)
    assert world.associate_partial(raw) == ("new", None, [])  # reproduces the recorded failure
    clean, points, stats = foreground(mask, frame, pose, support_z=metadata["geometry"]["support_z"])
    assert world.associate_partial(points) == ("matched", 3, [3])
    assert clean.sum() == len(points) < len(raw)
    assert stats["foreground_fraction"] > .9
    _, neighbour, _ = foreground(data["labels"] == 1, frame, pose, support_z=metadata["geometry"]["support_z"])
    assert world.associate_partial(np.vstack([points, neighbour]))[0] != "matched"
    assert world.associate_partial(neighbour)[1] != 3
    assert world.objects[3].geometry == metadata["geometry"]  # partial view cannot change grasp geometry


def test_foreground_retains_wrong_foreground_and_points_below_elevated_support():
    # World z varies with pixel row. An independently known elevated plane does not erase objects below it.
    depth = np.ones((40, 40))
    pose = np.eye(4); pose[:3, :3] = [[1,0,0],[0,0,1],[0,-1,0]]; pose[:3, 3] = [0,.2,.05]
    frame = NS(depth=depth, K=(100.,100.,20.,20.))
    mask, points, _ = foreground(np.ones_like(depth, bool), frame, pose, support_z=.1)
    assert np.any((points[:,2] > .003) & (points[:,2] < .09))
    assert np.any(points[:,2] > .12)
    assert not np.any(np.abs(points[:,2]-.1) <= .003)
    assert mask.sum() == len(points)


@pytest.mark.parametrize("verdict,expected", [("yes", True), ("no", False), ("unsure", False)])
def test_gemma_fallback_follows_unique_body_binding_not_siglip_veto(tmp_path, monkeypatch, verdict, expected):
    tools, obs, _ = side_tools(tmp_path, monkeypatch)
    tools.identity_check = lambda *a: {"ok": False, "top": "jello package"}
    calls = []
    result = tools.inspect_source_side(1, judge_identity=lambda view: calls.append(view) or (verdict, "body evidence"))
    assert result["ok"] is expected and len(calls) == 1
    assert calls[0]["object_id"] == 1 and calls[0]["mark"] == 7  # camera mark differs from world ID
    assert calls[0]["quality"]["ok"]
    assert (tools._source_identity_evidence is not None) is expected


@pytest.mark.parametrize("defect", ["merged", "clipped", "scene changes", "joints change", "reset"])
def test_semantic_yes_never_repairs_ambiguous_or_changed_physical_evidence(tmp_path, monkeypatch, defect):
    t, obs, _ = side_tools(tmp_path, monkeypatch)
    t.identity_check = lambda *a: {"ok": False}
    calls = []
    if defect == "merged":
        obs.marks[8] = obs.marks[7].copy(); obs.measured[8] = deepcopy(obs.measured[7])
    if defect == "clipped":
        obs.marks[7][:10, 40:50] = True
    def judge(view):
        calls.append(view)
        if defect == "scene changes": t.scene_revision += 1
        if defect == "reset": t.bridge.epoch = 99
        if defect == "joints change": t.bridge.q = lambda: obs.frame.q + .1
        return "yes", "it looks right"
    result = t.inspect_source_side(1, judge_identity=judge)
    assert not result["ok"] and t._source_identity_evidence is None
    assert len(calls) == (0 if defect in ("merged", "clipped") else 1)


def test_source_judge_uses_frozen_reference_once_and_rejects_expired_instance(tmp_path):
    t = FakeTools(); t.run_dir = tmp_path; t.scene_revision = 0
    obs = FakeScene(t).current()
    obs.rgb = np.full((80,100,3), 100, np.uint8)
    mask = np.zeros((80,100), bool); mask[10:70,30:70] = True
    obs.marks[1] = mask
    g = FakeGemma(["yes: same red package with its back visible"])
    m = workers.Manipulation(t, g, FakeScene(t), log)
    judge = m._source_judge(12, obs)
    view = dict(object_id=12, rgb=obs.rgb.copy(), frame_id="side", mark=9, path="side.jpg")
    assert judge(view)[0] == "yes"
    assert judge(view)[0] == "unsure" and len(g.calls) == 1
    assert g.calls[0]["images"] == 2 and "back, barcode" in g.calls[0]["text"]
    expired = m._source_judge(12, obs)
    t.scene_revision += 1
    assert expired(view)[0] == "unsure" and len(g.calls) == 1


@pytest.mark.parametrize("change", ["approach only", "later disturbance", "source moved", "wrong instance"])
def test_source_reference_is_sealed_after_approach_and_cannot_be_rearmed(tmp_path, change):
    t = FakeTools(); t.run_dir = tmp_path; t.scene_revision = 0
    obs = FakeScene(t).current()
    obs.rgb = np.full((80,100,3),100,np.uint8)
    obs.marks[1] = np.zeros((80,100),bool); obs.marks[1][10:70,30:70] = True
    g = FakeGemma(["yes: same object body"])
    judge = workers.Manipulation(t,g,FakeScene(t),log)._source_judge(12,obs)
    t.scene_revision += 1  # actual execute() behavior for a successful pre-grasp approach
    if change == "source moved": t.world.objects[12].geometry["centre"][0] += .05
    judge.begin_inspection()
    if change == "later disturbance":
        t.scene_revision += 1
        judge.begin_inspection()  # cannot extend the scope after motion
    view = dict(object_id=13 if change == "wrong instance" else 12, rgb=obs.rgb, frame_id="side",mark=9,path="side.jpg")
    assert (judge(view)[0] == "yes") is (change == "approach only")
    assert len(g.calls) == (1 if change == "approach only" else 0)


def test_whole_body_fov_rejects_centered_but_clipped_or_tiny_source():
    K, shape, pose = (500.,500.,320.,240.), (480,640,3), np.eye(4)
    points = np.array([[-.1,-.1,.4],[.1,.1,.4]])
    assert ViewpointPlanner.fits_image(pose, points, K, shape)
    assert not ViewpointPlanner.fits_image(pose, points * [3,3,1], K, shape)
    assert not ViewpointPlanner.fits_image(pose, points * [.01,.01,1], K, shape)
    assert not ViewpointPlanner.fits_image(pose, points * [1,1,-1], K, shape)
    mask = np.zeros((40,40),bool); mask[0:30,5:30] = True
    assert not crop_quality(mask, np.zeros((40,40,3),np.uint8))["ok"]


def test_split_same_object_is_one_full_transfer_but_different_subjects_are_not_merged():
    steps = [("manipulation", "pick up the red cheese crackers"),
             ("manipulation", "place the red cheese crackers on the blue tray"),
             ("verification", "check the tray")]
    normalized = complete_transfers(steps, "place cheezit on tray")
    assert len(normalized) == 2 and "pick up" in normalized[0][1] and "place" in normalized[0][1]
    with pytest.raises(ValueError, match="subjects differ"):
        complete_transfers([steps[0], ("manipulation", "place the milk on the blue tray")])
    with pytest.raises(ValueError, match="unpaired pickup"):
        complete_transfers([steps[0]], "place cheezit on tray")
    assert complete_transfers([steps[0]], "pick up cheezit") == [steps[0]]


def test_master_repairs_incomplete_transfer_before_dispatch():
    g = FakeGemma(["manipulation: pick up the crackers",
                   "manipulation: pick up the crackers and place them on the tray"])
    steps, _ = master.plan(g, "place crackers on tray", "scene", "rgb", log)
    assert "place" in steps[0][1] and len(g.calls) == 2
    assert "unpaired pickup" in g.calls[1]["text"]


def test_grounding_cannot_drop_transfer_destination(monkeypatch):
    FakePickPlace.runs = []
    monkeypatch.setattr(workers, "PickPlace", FakePickPlace)
    t = FakeTools(); g = FakeGemma(["pick #12", "pick #12"])
    m = workers.Manipulation(t, g, FakeScene(t), log)
    assert m.run("pick up the apple and place it on the tray").code == "grounding_invalid"
    assert not FakePickPlace.runs
    other = workers.Manipulation(t,FakeGemma(["pick #12","pick #12"]),FakeScene(t),log)
    other.require_transfer = True  # loop carries the original request's transfer requirement
    assert other.run("handle the selected apple").code == "grounding_invalid"
    assert not FakePickPlace.runs


def placement_tools(tmp_path):
    t = Tools(None,None,None,None,log,tmp_path)
    t.world.objects[7] = ObjectRecord(7, geom(0.,.6,.025,fp=(.36,.25)))
    t.world.objects[6] = ObjectRecord(6, geom(0.,.6,.193,fp=(.074,.074),support=.025))
    t.object_words = {7:"blue serving tray",6:"water bottle"}
    held = Held(.213,.02,name="obj_3",body_size=(.158,.065,.213),width=.065,length=.158)
    return t, held


def test_occupied_tray_has_valid_oriented_free_spots_not_its_occupied_center(tmp_path):
    t, held = placement_tools(tmp_path)
    # A bottle offset to one side leaves real space for a long box on the other side.
    t.world.objects[6].geometry["centre"][0] = -.09
    spots, count = t._free_spots(7,held,yaw_deg=90)
    assert count == 1 and spots
    assert all(np.linalg.norm(p-[0,.6]) > .04 for p in spots)
    assert any(p[0] > .04 for p in spots)
    for xy in spots:
        g = geom(*xy,.213,fp=(.158,.065))
        assert t.footprint_margin(g,t.world.objects[7].geometry,inset=t.CONTAINER_WALL_M) >= 0
    huge = Held(.1,.02,body_size=(.5,.5,.1))
    assert not t._free_spots(7,huge)[0]
    assert t.placement_relation("on",7) == "in"
    t.object_words[7] = "solid blue block"
    assert t.placement_relation("on",7) == "on"


def test_centered_bottle_can_leave_no_room_with_required_payload_and_gap(tmp_path):
    t, held = placement_tools(tmp_path)
    # This is the real failure condition: finding a different point is not permission to reduce margins.
    assert not t._free_spots(7,held,yaw_deg=90)[0]


def test_solid_support_search_tries_another_safe_spot_after_path_rejection(tmp_path):
    t, held = placement_tools(tmp_path)
    t.world.objects.pop(6)
    t.object_words[7] = "solid platform"
    calls = []
    def place(q,held,support,**kwargs):
        calls.append(support.copy())
        if len(calls) == 1: raise PlanningError("carry collision")
        return ["checked carry, lower, release, retreat"], q
    steps, centre, top = t._plan_place(NS(place=place),np.zeros(6),held,geom(.4,.4,.213),"on",7,"robot")
    assert len(calls) == 2 and steps and not np.array_equal(calls[0],calls[1])
    assert np.allclose(centre[:2],calls[1][:2]) and top > .025
