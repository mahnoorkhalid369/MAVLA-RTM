"""Transfer policy, image binding, local inspection, payload geometry and failure-stage regressions."""

import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agent import loop, prompts, workers
from agent.contracts import Report
from agent.pick_place import PickPlace
from agent.tools import Tools
from agent.tools.manipulation import PickPlan
from control.robot import ARM_HOME
from control.skills import GripperStep, Held
from control.collision import CollisionChecker
from perception.geometry import central_grasp
from tests.support.fakes import geom
from tests.unit.test_master_workers import FakeGemma, FakeTools, FakeScene, FakePickPlace, log


def test_clear_two_object_request_executes_transfers_without_description_calls(monkeypatch, tmp_path):
    FakePickPlace.fail, FakePickPlace.runs = False, []
    monkeypatch.setattr(loop, "Scene", lambda tools, directory: FakeScene(tools))
    monkeypatch.setattr(workers, "PickPlace", FakePickPlace)
    g = FakeGemma(["manipulation: put the banana on the tray\nmanipulation: put the bottle on the tray\n"
                   "verification: check both objects are on the tray", "pick #12 place on #15", "pick #13 place on #15",
                   "yes: the banana rests on the tray", "yes: the bottle rests on the tray",  # wrist pictures
                   "yes: both requested items are on the tray"])
    result = loop.run_instruction("place banana and water bottle on the serving tray", FakeTools(), g, log, tmp_path)
    assert result["status"] == "COMPLETED"
    assert len(g.calls) == 6 and [r["worker"] for r in result["worker_reports"][:3]] == [
        "manipulation", "manipulation", "verification"]
    policy = json.loads((tmp_path / "prompts.json").read_text())
    assert policy["revision"] == prompts.REVISION and policy["prompts"]["master"] == prompts.MASTER


def test_visual_question_budget_prevents_unlimited_rephrased_descriptions(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "Scene", lambda tools, directory: FakeScene(tools))
    g = FakeGemma(["perception: which banana?", "OK: one banana is visible", "plan: perception: where is the tray?",
                   "OK: tray on the right", "plan: perception: describe surroundings again", "stop: clarification needed"])
    result = loop.run_instruction("move the banana", FakeTools(), g, log, tmp_path)
    assert result["status"] == "STOPPED"
    assert result["worker_reports"][-1]["code"] == "perception_budget"
    assert len(g.calls) == 6  # the refused third visual question never reaches Gemma


def test_wrist_identity_mismatch_grounds_again_with_the_reason(monkeypatch):
    """The wrist close-up (SigLIP 2) shows another object: the skill stops before descent, grounding chooses again
    once with the reason, and only the second choice is executed to the end."""
    runs = []

    class IdentitySkill(FakePickPlace):
        def run(self, mode, object_id, *a, **k):
            runs.append(object_id)
            if object_id == 13:
                return {"status": "FAILED", "failure": {"cls": "identity"}, "stage": "INSPECTING_GRASP",
                        "effects": {"grasp_attempted": False}, "object_id": 13,
                        "message": "up close the chosen object #13 looks like a red cube, not the brown cube"}
            return super().run(mode, object_id, *a, **k)

    monkeypatch.setattr(workers, "PickPlace", IdentitySkill)
    t, entries = FakeTools(), []
    g = FakeGemma(["pick #13", "pick #15"])
    m = workers.Manipulation(t, g, FakeScene(t), lambda kind, **data: entries.append((kind, data)))
    assert m.run("pick up the brown cube").word == "done"
    assert runs == [13, 15]
    assert [kind for kind, _ in entries].count("identity_retry") == 1
    assert "looks like a red cube" in g.calls[1]["text"]


def test_destination_too_small_for_the_object_is_rejected_before_motion(monkeypatch):
    FakePickPlace.fail, FakePickPlace.runs = False, []
    monkeypatch.setattr(workers, "PickPlace", FakePickPlace)
    t = FakeTools()
    g = FakeGemma(["pick #15 place in #12", "pick #15 place on #13"])
    t.world.objects[13].geometry.update(footprint=[0.30, 0.20], extents=[0.30, 0.20])
    m = workers.Manipulation(t, g, FakeScene(t), log)
    assert m.run("put the brown cube on the tray").word == "done"
    assert FakePickPlace.runs == [("pick_place", 15, "on", 13, None)]
    assert "too small" in g.calls[1]["text"]


def test_lost_held_record_can_be_associated_again(tmp_path):
    t = Tools(None, None, None, None, log, tmp_path)
    oid = t.world.observe(geom(0., .55, .06), None)
    t.held_object, t.world.holding = oid, oid
    t.world.objects[oid].status = "held"
    t.clear_held()
    assert t.world.objects[oid].status == "missing" and t.world.holding is None
    assert t.world.observe(geom(.002, .55, .06), None) == oid


def test_central_grasp_of_curved_body_is_on_material_near_its_middle():
    u, width = np.meshgrid(np.linspace(-.08, .08, 120), np.linspace(-.008, .008, 20))
    xy = np.column_stack([u.ravel(), (4 * u ** 2 + width).ravel()])
    centre = np.array([0., .0128])  # bounding centre falls in the curve's empty region
    chosen = central_grasp(xy, centre, 0.)
    assert abs(chosen[0]) < .012
    assert np.min(np.linalg.norm(xy - chosen, axis=1)) <= .003


def test_regular_body_keeps_its_geometrical_centre():
    u, v = np.meshgrid(np.linspace(-.03, .03, 50), np.linspace(-.02, .02, 50))
    xy = np.column_stack([u.ravel(), v.ravel()])
    assert np.array_equal(central_grasp(xy, np.zeros(2), 0.), np.zeros(2))


def test_payload_bounds_include_full_body_and_offcentre_grasp():
    held = Held(.04, .02, width=.02, length=.16, body_size=(.05, .16, .04), body_offset=(.01, -.015))
    T, size = held.collision_body()
    assert np.array_equal(size, [.05, .16, .04]) and np.allclose(T[:2, 3], [.01, -.015])


def test_analytic_robot_rims_are_excluded_without_excluding_nearby_free_space():
    from perception.geometry import RobotVolume
    robot = RobotVolume(CollisionChecker(), np.array(ARM_HOME), .085)
    points = np.array([[-.075, .18, .394], [.075, .18, .394], [.3, .3, .1]])
    assert robot.contains(points).tolist() == [True, True, False]


def wrist_tools(tmp_path, measured, monkeypatch):
    from agent.tools import grasp_inspection
    t = Tools(None, None, None, None, log, tmp_path)
    t.world.observe(geom(0., .55, .06, fp=(.05, .05)), None)
    frame = NS(q=np.array(ARM_HOME), finger=0., stamp=1., K=(100., 100., 10., 10.),
               rgb=np.zeros((20, 20, 3), np.uint8), depth=np.full((20, 20), .3))
    obs = NS(frame_id="wrist_001", frame=frame, measured=measured, object_ids={},
             motion_count=0, taken_at=time.monotonic(), cleared_view=False)
    t.bridge, t.perceive = NS(wait_still=lambda: None, q=lambda: np.array(ARM_HOME)), lambda camera: obs
    t._camera_pose = lambda frame: np.eye(4)
    monkeypatch.setattr(grasp_inspection, "unexplained", lambda *a: [geom(.08, .55, .07)])
    return t


def test_wrist_refines_same_object_and_keeps_unexplained_clearance_obstacle(tmp_path, monkeypatch):
    g = geom(.003, .55, .061, fp=(.05, .05))
    t = wrist_tools(tmp_path, {1: g}, monkeypatch)
    result = t.inspect_grasp(1)
    assert result["ok"] and t.world.objects[1].geometry["centre"][0] == .003
    assert t.inspection_obstacles and t.inspection_obstacles[0].name == "wrist_local_0"
    assert (tmp_path / "images/wrist_001_depth.npz").is_file()


@pytest.mark.parametrize("measured", [{}, {1: geom(0., .55, .06, fp=(.025, .025))},
                                      {1: geom(0., .55, .06), 2: geom(.001, .55, .06)}])
def test_wrist_missing_partial_or_duplicate_target_never_approves_descent(tmp_path, monkeypatch, measured):
    t = wrist_tools(tmp_path, measured, monkeypatch)
    assert not t.inspect_grasp(1)["ok"]


def skill_tools(inspection=True, retention=True):
    commands, plans = [], []
    held = Held(.04, .02, width=.04, length=.04)
    plan = PickPlan(1, ["open", "hover"], ["descend", GripperStep(.4, "close")], ["lift"],
                    ["carry", "lower", "release", "retreat"], held, {}, [0., .55, .02], .04, "on", 2, home=["home"])
    t = NS(held_object=None, held_close_target=.4, last_referent=None, tol=NS(empty_deg=.5))
    t.prepare_pick = lambda: {"ok": True}
    t.is_fresh, t.update_world = lambda obs: True, lambda obs: None
    t.perceive = lambda camera: NS(marks={1: True, 2: True}, object_ids={1: 1, 2: 2}, frame_id="table_test")
    def plan_transfer(*a, **k):
        plans.append(1)
        return plan
    t.bridge, t.plan_pick_place = NS(q=lambda: np.array(ARM_HOME)), plan_transfer
    t.inspect_destination = lambda reference_id: None  # no identity matcher in these tests
    t.execute = lambda steps, label: commands.append((label, list(steps))) or NS(ok=True)
    t.inspect_grasp = lambda oid: {"ok": inspection, "reason": "missing wrist view"}
    t.verify_contact = lambda target: {"check": "contact", "pass": True}
    t.set_held = lambda *a: setattr(t, "held_object", 1)
    t.confirm_held = lambda *a: {"check": "retention", "pass": retention}
    t.clear_held = lambda: setattr(t, "held_object", None)
    t.assume_placed, t.park_for_view = lambda plan: t.clear_held(), lambda *a: {"ok": True}
    t.verify_placement = lambda plan: {"check": "placement", "pass": True}
    t.looks = []
    t.look_at_placement = lambda plan, farther=False: t.looks.append(farther) or {
        "ok": True, "rgb": None, "frame_id": "wrist_look", "path": None}
    t.return_home, t.verify_home = lambda: {"ok": True}, lambda: {"check": "home", "pass": True, "max_error_deg": 0.}
    return t, commands, plans


def test_skill_inspects_and_replans_before_descent_then_checks_lift_before_carry():
    t, commands, plans = skill_tools()
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["status"] == "COMPLETED" and len(plans) == 2
    labels = [label for label, _ in commands]
    assert labels == ["pre-grasp inspection pose", "refined approach", "close", "lift",
                      "carry via validated waypoint", "lower", "release", "retreat"]
    assert commands[0][1] == ["open", "hover"]  # no descent before inspection


@pytest.mark.parametrize("answers, status, looks", [
    (["yes"], "COMPLETED", [False]),
    (["no"], "FAILED", [False]),
    (["unsure", "yes"], "COMPLETED", [False, True]),
    (["unsure", "unsure"], "FAILED", [False, True])])
def test_placement_is_judged_on_the_wrist_picture_without_table_matching(answers, status, looks):
    """After release the wrist camera looks down and Gemma's word decides; unsure -> one look from farther.
    The table re-measurement (verify_placement) is not used."""
    t, commands, _ = skill_tools()
    t.verify_placement = lambda plan: pytest.fail("placement must not be decided by table matching")
    skill, said = PickPlace(t, log), list(answers)
    skill.judge_placement = lambda view: (said.pop(0), "the cube on its support")
    result = skill.run("pick_place", 1, "on", 2)
    assert result["status"] == status and t.looks == looks and not said
    if status == "FAILED":
        assert result["failure"]["cls"] == "placement"


def test_missing_wrist_evidence_stops_before_descent_or_grasp():
    t, commands, plans = skill_tools(inspection=False)
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["status"] == "FAILED" and result["failure"]["cls"] == "inspection"
    assert len(commands) == 1 and len(plans) == 1


def test_new_wrist_obstacle_rejects_replanned_route_before_descent():
    from control.motion import PlanningError
    t, commands, _ = skill_tools()
    original = t.plan_pick_place
    calls = []
    def refine(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise PlanningError("descend: collision with wrist_local_0")
        return original(*a, **k)
    t.plan_pick_place = refine
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["failure"]["cls"] == "blocked" and len(commands) == 1


def test_same_failed_grasp_is_not_reissued_with_only_reworded_message(monkeypatch):
    class FailedSkill:
        calls = 0
        def __init__(self, *a, **k):
            pass
        def run(self, *a, **k):
            FailedSkill.calls += 1
            return {"status": "FAILED", "failure": {"cls": "retention"}, "message": "retention failed"}
    monkeypatch.setattr(workers, "PickPlace", FailedSkill)
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12", "pick #12"]), FakeScene(t), log)
    assert m.run("pick the banana").code == "retention"
    assert m.run("try picking the banana again please").code == "unchanged_action"
    assert FailedSkill.calls == 1


def test_failed_lift_retention_stops_before_carry_and_invalidates_held_state():
    t, commands, _ = skill_tools(retention=False)
    result = PickPlace(t, log).run("pick_place", 1, "on", 2)
    assert result["failure"]["cls"] == "retention" and t.held_object is None
    assert [label for label, _ in commands][-1] == "lift"
