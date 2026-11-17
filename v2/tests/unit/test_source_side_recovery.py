"""Water-bottle regression: change the view of the same source, never repeat grounding or bypass geometry."""

from copy import deepcopy
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from agent import workers
from agent.pick_place import PickPlace
from agent.tools.execution import ExecutionTools
from agent.tools import Tools
from agent.controls import ExecutionControls
from control.robot import ARM_HOME
from tests.support.fakes import geom
from tests.unit.test_master_workers import FakeGemma, FakeScene, FakeTools, log
from tests.unit.test_transfer_engineering import skill_tools, wrist_tools


def recovering_skill(side_ok=True, geometry_ok=True, return_ok=True, home_ok=True):
    t, commands, plans = skill_tools()
    calls, proof = [], {"token": "side identity"}
    def inspect(oid, *, identity_evidence=None):
        calls.append(("top", oid, identity_evidence))
        if identity_evidence is None:
            return {"ok": False, "identity_mismatch": True, "reason": "water bottle ranked below milk carton"}
        assert identity_evidence is proof
        return {"ok": geometry_ok, "reason": "top geometry disagrees"}
    t.inspect_grasp = inspect
    t.inspect_source_side = lambda oid: calls.append(("side", oid)) or {
        "ok": side_ok, "reason": "side identity unconfirmed", "identity_evidence": proof}
    t.return_to_source_inspection = lambda q: calls.append(("return hover",)) or {
        "ok": return_ok, "reason": "hover path blocked"}
    t.return_home = lambda: calls.append(("HOME",)) or {"ok": home_ok, "reason": "HOME path collides"}
    return t, commands, plans, calls


def test_side_acceptance_returns_to_hover_and_rechecks_geometry_before_any_descent():
    t, commands, plans, calls = recovering_skill()
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["status"] == "COMPLETED" and len(plans) == 2
    assert [c[0] for c in calls] == ["top", "side", "return hover", "top", "HOME"]
    assert [c[1] for c in calls if c[0] in ("top", "side")] == [1, 1, 1]
    states = [s["state"] for s in result["timeline"]]
    assert states.index("RECHECKING_GRASP_GEOMETRY") < states.index("DESCENDING")
    assert states.count("APPROACHING") == 1 and states.count("INSPECTING_SOURCE_SIDE") == 1
    assert commands[0][0] == "pre-grasp inspection pose"


@pytest.mark.parametrize("kwargs", [{"side_ok": False}, {"geometry_ok": False}, {"return_ok": False}])
def test_failed_side_recovery_returns_home_without_descending_or_closing(kwargs):
    t, commands, plans, calls = recovering_skill(**kwargs)
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["status"] == "FAILED" and result["failure"]["cls"] == "source_identity"
    assert "returned HOME" in result["failure"]["reason"]
    assert [c[0] for c in calls].count("side") == 1 and [c[0] for c in calls].count("HOME") == 1
    assert len(plans) == 1 and len(commands) == 1
    assert not result["effects"]["grasp_attempted"] and t.held_object is None


def test_unavailable_home_is_reported_as_failure_and_never_claimed_reached():
    t, commands, _, _ = recovering_skill(side_ok=False, home_ok=False)
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert "HOME not verified: HOME path collides" in result["failure"]["reason"]
    assert not any(v.get("check") == "home" for v in result["verifications"])
    assert len(commands) == 1


def test_side_camera_exception_still_attempts_safe_home():
    t, commands, _, calls = recovering_skill()
    def lost_camera(oid):
        raise TimeoutError("wrist camera unavailable")
    t.inspect_source_side = lost_camera
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["status"] == "FAILED" and "wrist camera unavailable" in result["failure"]["reason"]
    assert calls[-1] == ("HOME",) and len(commands) == 1


def test_simulator_reset_during_side_inspection_prevents_home_using_stale_world():
    t, commands, _, calls = recovering_skill()
    t.bridge.epoch = 1
    def reset(oid):
        t.bridge.epoch = 2
        return {"ok": False, "reason": "simulator reset"}
    t.inspect_source_side = reset
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["failure"]["cls"] == "sensor_epoch"
    assert not any(c[0] == "HOME" for c in calls) and len(commands) == 1


def test_source_identity_exhaustion_is_not_sent_back_to_gemma_for_same_grounding(monkeypatch):
    class FailedSourceSkill:
        runs = 0
        def __init__(self, *a):
            pass
        def run(self, mode, oid, *a, **kw):
            self.__class__.runs += 1
            return {"status": "FAILED", "object_id": oid, "stage": "ABORTING_SOURCE_HOME",
                    "effects": {"grasp_attempted": False}, "failure": {"cls": "source_identity"},
                    "message": "source identity unconfirmed; returned HOME"}
    monkeypatch.setattr(workers, "PickPlace", FailedSourceSkill)
    t = FakeTools()
    g = FakeGemma(["pick #12", "pick #13"])
    worker = workers.Manipulation(t, g, FakeScene(t), log)
    report = worker.run("pick the water bottle")
    assert report.code == "source_identity" and report.details["recovery"] == "stop"
    assert len(g.calls) == 1 and FailedSourceSkill.runs == 1 and not worker.context.pending


def side_tools(tmp_path, monkeypatch):
    from perception import viewpoints
    t = Tools(None, None, None, None, log, tmp_path)
    source = geom(0., .55, .16, fp=(.06, .06))
    t.world.observe(source, None)
    t.object_words = {1: "water bottle", 2: "milk carton"}
    q = np.asarray(ARM_HOME, float)
    t.bridge = NS(q=lambda: q.copy(), finger=lambda: 0.)
    camera = np.eye(4); camera[:3, 3] = [0., .55, .45]
    t.wrist_kin = NS(fk=lambda q: camera)
    best = NS(q=q, camera_pos=np.array([.35, .55, .20]))
    plan = NS(ok=True, best=best, summary=lambda: {"ok": True})
    captured = {}
    class Planner:
        def plan(self, target, q, obstacles, **kwargs):
            captured.update(target=target, obstacles=obstacles, **kwargs)
            return plan
        def reached(self, *a):
            return {"pass": True, "camera_pos_error_mm": 0., "axis_error_deg": 0.}
    monkeypatch.setattr(viewpoints, "ViewpointPlanner", Planner)
    mask = np.zeros((100, 100), bool); mask[10:90, 30:70] = True
    measured = dict(source, visible_bottom_z=.01)
    frame = NS(epoch=3, q=q.copy(), finger=0., depth=np.full((100, 100), .3), K=(500.,500.,50.,50.))
    obs = NS(frame_id="wrist_side", frame=frame, rgb=np.ones((100, 100, 3), np.uint8),
             marks={7: mask}, measured={7: measured})
    t.perceive = lambda camera: obs
    pose = np.eye(4)
    pose[:3, :3] = [[1.,0.,0.], [0.,0.,1.], [0.,-1.,0.]]
    pose[:3, 3] = [0.,.25,.055]
    t._camera_pose = lambda frame:pose
    from agent.tools import grasp_inspection
    monkeypatch.setattr(grasp_inspection, "unexplained", lambda *a:[])
    t.is_fresh = lambda obs: True
    t._source_view_move = lambda *a: {"ok": True}
    t.identity_check = lambda oid, obs, mark, role: {
        "ok": True, "object_id": oid, "frame_id": obs.frame_id, "mark": mark, "role": role}
    return t, obs, captured


def test_side_mark_is_associated_by_geometry_and_never_used_as_world_id(tmp_path, monkeypatch):
    t, obs, captured = side_tools(tmp_path, monkeypatch)
    original = deepcopy(t.world.objects[1].geometry)
    result = t.inspect_source_side(1)
    assert result["ok"] and result["mark"] == 7 and result["identity"]["object_id"] == 1
    assert captured["elevations"] == (25., 35.)
    assert captured["min_view_angle_deg"] == 20.
    assert captured["target_box_name"] == "obj_1"  # only line of sight excludes it
    assert "obj_1" in [b.name for b in captured["obstacles"]]  # collision checking retains it
    assert np.allclose(captured["target"], original["centre"])
    assert t.world.objects[1].geometry == original  # oblique geometry never replaces the top geometry


def test_viewpoint_search_rejects_same_direction_before_ik(monkeypatch):
    from perception.viewpoints import ViewpointPlanner
    planner = ViewpointPlanner()
    q = np.asarray(ARM_HOME)
    target = np.array([0., .55, .08])
    ray = planner.kin.fk(q[:planner.n])[:3, 3] - target
    distance = np.linalg.norm(ray)
    elevation = np.degrees(np.arcsin(ray[2]/distance))
    azimuth = np.degrees(np.arctan2(ray[1], ray[0]))
    monkeypatch.setattr(planner, "look_at_ik", lambda *a:pytest.fail("same direction must not reach IK"))
    result = planner.plan(target, q, [], distances=(distance,), elevations=(elevation,),
                          azimuths=(azimuth,), min_view_angle_deg=20.)
    assert not result.ok and result.rejected["view direction too similar to the previous observation"] == 1


@pytest.mark.parametrize("defect", ["duplicate", "clipped", "body hidden", "stale", "identity"])
def test_side_requires_unique_fresh_whole_body_and_positive_identity(tmp_path, monkeypatch, defect):
    t, obs, _ = side_tools(tmp_path, monkeypatch)
    if defect == "duplicate":
        obs.measured[8] = deepcopy(obs.measured[7])
        obs.marks[8] = obs.marks[7].copy()
    elif defect == "clipped":
        obs.marks[7][0:10, 40:50] = True
    elif defect == "body hidden":
        obs.measured[7]["visible_bottom_z"] = .12
    elif defect == "stale":
        t.is_fresh = lambda obs: False
    else:
        t.identity_check = lambda *a: {"ok": False, "top": "milk carton"}
    assert not t.inspect_source_side(1)["ok"]
    assert t._source_identity_evidence is None


@pytest.mark.parametrize("defect", [None, "forged", "object", "scene", "epoch", "age", "name", "geometry", "held"])
def test_side_identity_proof_is_scoped_to_same_source_scene_and_capture(tmp_path, monkeypatch, defect):
    t, obs, _ = side_tools(tmp_path, monkeypatch)
    proof = t.inspect_source_side(1)["identity_evidence"]
    if defect == "forged": proof = dict(proof)
    elif defect == "object": proof["object_id"] = 2
    elif defect == "scene": t.scene_revision += 1
    elif defect == "epoch": obs.frame.epoch += 1
    elif defect == "age": proof["captured_at"] = time.monotonic()-t.MAX_OBS_AGE_S-1
    elif defect == "name": t.object_words[1] = "milk carton"
    elif defect == "geometry": t.world.objects[1].geometry["centre"][0] += .01
    elif defect == "held": t.held_object = 1
    assert t._valid_source_identity(proof, 1, obs) == (defect is None)


def test_side_proof_does_not_bypass_top_geometry_or_unknown_depth_obstacles(tmp_path, monkeypatch):
    from control.collision import Box
    t = wrist_tools(tmp_path, {1: geom(.003, .55, .061, fp=(.05, .05))}, monkeypatch)
    obs = t.perceive("wrist")
    t.bridge.finger = lambda: 0.
    t.object_words[1] = "water bottle"
    proof = {"identity": {"ok": True, "frame_id": "side"}, "object_id": 1, "expected": "water bottle",
             "scene_revision": t.scene_revision, "epoch": None, "captured_at": time.monotonic(),
             "geometry": deepcopy(t.world.objects[1].geometry)}
    t._source_identity_evidence = proof
    t.inspection_obstacles.append(Box("side_only", np.array([.10,.55,.03]), np.eye(3), np.array([.02,.02,.03])))
    t.identity_check = lambda *a: pytest.fail("the top identity vote must not override the confirmed side view")
    assert t.inspect_grasp(1, identity_evidence=proof)["ok"]
    assert t.inspection_obstacles
    assert "side_only" in [b.name for b in t.inspection_obstacles]
    proof["geometry"] = deepcopy(t.world.objects[1].geometry)
    obs.measured[1]["footprint"] = [.025, .025]
    assert not t.inspect_grasp(1, identity_evidence=proof)["ok"]


def test_top_identity_rejection_retains_new_wrist_obstacles_for_side_motion_and_home(tmp_path, monkeypatch):
    t = wrist_tools(tmp_path, {1:geom(0., .55, .06)}, monkeypatch)
    t.identity_check = lambda *a:{"ok":False, "top":"milk carton", "expected":"water bottle"}
    result = t.inspect_grasp(1)
    assert not result["ok"] and result["identity_mismatch"]
    assert t.inspection_obstacles and t.inspection_obstacles[0].name == "wrist_local_0"


@pytest.mark.parametrize("label", ["inspect source from side", "return to source inspection"])
def test_only_successful_empty_hand_view_motion_preserves_side_proof_revision(label):
    result = NS(ok=True, failed_step="", reason="", steps=[])
    t = NS(log=log, controls=ExecutionControls(), held_object=None, bridge=NS(finger=lambda:0.),
           motion_count=0, scene_revision=5, ex=NS(spin=lambda *a:None, execute=lambda *a:result))
    ExecutionTools.execute(t, [], label)
    assert t.scene_revision == 5
    result.ok = False
    ExecutionTools.execute(t, [], label)
    assert t.scene_revision == 6


@pytest.mark.parametrize("defect", ["collision", "forbidden", "held", "reset", "reach"])
def test_source_view_moves_never_bypass_collision_joint_zone_or_empty_hand_checks(tmp_path, monkeypatch, defect):
    from agent.tools import grasp_inspection
    from control import motion
    from safety import joint_limits
    t = Tools(None, None, None, None, log, tmp_path)
    q = np.array(ARM_HOME)
    t.bridge = NS(q=lambda:q.copy(), finger=lambda:0., epoch=1)
    t._sensor_epoch = 1
    collision = NS(ok=defect != "collision", contacts=["source bottle"])
    monkeypatch.setattr(grasp_inspection, "CollisionChecker", lambda obstacles:NS(check_path=lambda path:collision))
    monkeypatch.setattr(motion, "joint_move", lambda *a:NS(positions=np.array([q,q])))
    monkeypatch.setattr(joint_limits, "forbidden_on_path", lambda path:"forbidden joint zone" if defect=="forbidden" else None)
    if defect == "held": t.held_object = 1
    if defect == "reset": t.bridge.epoch = 2
    moves = []
    t.execute = lambda *a:moves.append(a) or NS(ok=True, reason="")
    destination = q + .1 if defect=="reach" else q
    assert not t.return_to_source_inspection(destination)["ok"]
    assert len(moves) == (1 if defect=="reach" else 0)
