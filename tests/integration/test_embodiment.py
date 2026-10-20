"""The robot's embodiment: the shared identity in front of the embodied Gemma roles (agent/vlm_roles), the
directional look capability (agent/capabilities/inspection.py) and its measured verification (agent/goals.py).
Recorded Gemma answers and mocked tools (no simulator, no GPU, no motion); no scenario knowledge."""

import json

import numpy as np
import pytest

pytest.importorskip("pydantic")

from agent.vlm_roles import EMBODIED_ROLES, IDENTITY_SYSTEM, Roles  # noqa: E402


class SystemRecorder:
    """A VLM that records the system prompt of every call and replays answers."""

    def __init__(self, answers):
        self.answers, self.systems = list(answers), []

    def generate(self, text, image=None, system=None, max_new_tokens=None, thinking=None):
        self.systems.append(system)
        return self.answers.pop(0)


# ------------------------------------------------------------------------------------------ shared identity
def test_the_identity_is_put_in_front_of_every_embodied_role_once():
    vlm = SystemRecorder([json.dumps({"kind": "description", "evidence": "look"}),
                          json.dumps({"goal": "g", "steps": [{"do": "observe"}], "reason": "r"}),
                          json.dumps({"action": "stop", "reason": "r"}),
                          json.dumps({"answer": "two", "object_ids": [1]})])
    r = Roles(vlm)
    r.classify_request("look")
    r.plan_goal("look", "menu")
    r.decide("g", "", "e", "menu", has_pending=False)
    r.answer("how many?", "facts", [1])
    assert all(s.startswith(IDENTITY_SYSTEM) and s.count(IDENTITY_SYSTEM) == 1 for s in vlm.systems)
    assert vlm.systems[1] != IDENTITY_SYSTEM  # the role's own instructions follow the identity
    assert {"request", "plan", "decide", "decompose", "answer"} == EMBODIED_ROLES


def test_perception_roles_judge_images_without_the_identity():
    vlm = SystemRecorder([json.dumps({"match": "yes", "reason": "r"})])
    Roles(vlm).confirm(np.zeros((4, 4, 3), np.uint8), "the <A>")
    assert vlm.systems == [None]


def test_the_identity_defines_the_robot_frame_and_separates_looking_from_moving_the_hand():
    text = IDENTITY_SYSTEM.lower()
    assert '"you" means the robot' in text and "gripper" in text and "wrist" in text
    assert '"robot" frame' in text and "x = your right, y = forward" in text and '"camera" frame' in text
    assert "not moving your hand" in text and "never guess a frame" in text


# ---------------------------------------------------------------------------------------- look (direction)
pytest.importorskip("langgraph")

from types import SimpleNamespace  # noqa: E402

from agent import capabilities as caps  # noqa: E402
from agent.capabilities.specs import SPECS, planner_menu  # noqa: E402
from agent.directions import DIRECTIONS, direction_vector, view_error  # noqa: E402
from agent.vlm_roles import RecordedVLM, VLMFailure  # noqa: E402
from control.robot import ARM_HOME  # noqa: E402
from tests.support.fakes import GoalTools  # noqa: E402


class ArmTools(GoalTools):
    """GoalTools whose joint state follows the executed motion (like joint feedback after a reached move)."""

    def __init__(self, plans=(), move=True, ok=True, reason="", **kw):
        super().__init__(list(plans), **kw)
        self.q_now, self.move, self.ok, self.reason = np.array(ARM_HOME, float), move, ok, reason
        self.bridge = SimpleNamespace(q=lambda: self.q_now.copy(), wait_still=lambda: None, finger=lambda: 0.0)
        self.OPEN_FINGER_DEG = 2.0

    def execute(self, steps, label):
        self.executed.append(label)
        if not self.ok:
            return SimpleNamespace(ok=False, failed_step=label, reason=self.reason, steps=[])
        self.motion_count += 1
        if self.move:
            self.q_now = np.asarray(steps[-1].trajectory.positions[-1], float)
        return SimpleNamespace(ok=True, failed_step="", reason="", steps=[])

    def perceive(self, camera):
        o = super().perceive(camera)
        o.camera = camera
        return o

    def describe(self, obs, marks=None):
        pass


def context(t):
    from agent.task_state import TaskState
    return caps.Context(t, lambda *a, **k: None, ".", None, None, TaskState("look"))


def test_look_is_a_sensing_motion_of_the_inspection_worker_in_the_menu():
    s = SPECS["look"]
    assert (s.safety, s.effect) == ("motion", "sensing") and "- look:" in planner_menu()
    assert '"frame": "robot"' in s.args and '"camera" only when' in s.summary and all(d in s.args for d in DIRECTIONS)


def test_look_steps_carry_an_explicit_direction_and_frame():
    from agent.vlm_roles import Roles
    vlm = RecordedVLM([json.dumps({"goal": "g", "steps": [{"do": "look"}]}),
                       json.dumps({"goal": "g", "steps": [{"do": "look", "direction": "Left", "frame": "robot"}]})])
    p = Roles(vlm).plan_goal("look at your left side", "menu")
    assert p.steps[0].args == {"direction": "left", "frame": "robot"} and "look needs direction" in vlm.calls[1]
    with pytest.raises(VLMFailure):  # a direction outside the vocabulary is not guessed: schema error, then failure
        Roles(RecordedVLM([json.dumps({"goal": "g", "steps": [{"do": "look", "direction": "over there"}]})] * 2)).plan_goal("x", "m")


def test_frames_are_explicit_and_differ():
    assert np.allclose(direction_vector("left", "robot"), [-1, 0, 0]) and np.allclose(direction_vector("down", "camera"), [0, 0, -1])
    assert not np.allclose(direction_vector("left", "camera"), direction_vector("left", "robot"))


@pytest.mark.parametrize("direction", ["left", "right", "forward", "down"])
def test_look_moves_the_wrist_camera_and_measures_where_it_points(direction):
    t = ArmTools()
    ctx = context(t)
    r = caps.run_look(ctx, {"direction": direction, "frame": "robot"})
    assert r.outcome == "completed" and t.executed == ["look view"], r.summary
    v = view_error(t.q_now, direction, "robot")  # measured from the (fake) joint feedback, not the plan
    assert v["pointing"] and r.data["view"]["error_deg"] == v["error_deg"] and "The arm stays here" in r.summary
    assert ctx.obs.camera == "wrist" and r.data["objects"] == 3  # what the robot now sees: the wrist view


def test_a_blocked_move_is_not_reported_as_looking():
    t = ArmTools(ok=False, reason="motion blocked by the execution controls")
    r = caps.run_look(context(t), {"direction": "left", "frame": "robot"})
    assert r.outcome == "failed" and "not executed" in r.summary and "view" not in r.data


def test_a_viewpoint_not_reached_fails():
    t = ArmTools(move=False)  # the command returned ok but the joints did not get there
    r = caps.run_look(context(t), {"direction": "left", "frame": "robot"})
    assert r.outcome == "failed" and "not reached" in r.summary


def test_no_safe_viewpoint_is_infeasible_not_success(monkeypatch):
    from control.collision import Box
    t = ArmTools()
    monkeypatch.setattr(t, "obstacles", lambda: [Box("wall", np.array([0.0, 0.0, 0.8]), np.eye(3), np.array([2.0, 2.0, 0.05]))])
    r = caps.run_look(context(t), {"direction": "left", "frame": "robot"})
    assert r.outcome == "infeasible" and "no safe viewpoint" in r.summary and t.executed == []


def test_no_looking_around_while_holding_an_object():
    t = ArmTools()
    t.world.holding = 1
    r = caps.run_look(context(t), {"direction": "left", "frame": "robot"})
    assert r.outcome == "infeasible" and t.executed == []


# ------------------------------------------------------------------------------- measured goal verification
from agent.goals import GoalEvaluator, derive_conditions  # noqa: E402
from tests.support.runners import plan, run_supervisor as run  # noqa: E402

LOOK_LEFT = ("look", {"direction": "left", "frame": "robot"})


def test_look_sets_a_measured_viewing_condition_that_a_later_motion_ends():
    assert derive_conditions([("look", {"direction": "left", "frame": "robot"})]) == [
        {"pred": "viewing", "args": ["left"], "frame": "robot"}]
    assert derive_conditions([("look", {"direction": "left", "frame": "robot"}), ("return_home", {})]) == [
        {"pred": "at_home", "args": []}]
    assert derive_conditions([("observe", {})]) == [{"pred": "seen", "args": [], "camera": "table"}]


def test_viewing_is_evaluated_from_the_measured_joint_state():
    t = ArmTools()
    caps.run_look(context(t), {"direction": "right", "frame": "robot"})
    ev = GoalEvaluator(t, None)
    yes = ev.evaluate({"pred": "viewing", "args": ["right"], "frame": "robot"}, {})
    no = ev.evaluate({"pred": "viewing", "args": ["left"], "frame": "robot"}, {})
    assert (yes.satisfied, yes.confidence) == (True, "measured") and "deg from right" in yes.evidence
    assert no.satisfied is False


def test_look_left_is_delegated_to_the_inspection_worker_verified_and_completed(tmp_path):
    t = ArmTools([plan(LOOK_LEFT)])
    t.request_kind = "description"  # "look at your left side": sensing is authorized, object changes are not
    r, ev = run(tmp_path, t, instruction="Look at your left side.")
    assert r["status"] == "COMPLETED" and t.executed == ["look view"]
    assert [e["worker"] for e in ev if e["kind"] == "delegate"][:2] == ["perception", "inspection"]  # observe first
    sub = r["subgoal_checks"][0]
    assert sub["outcome"] == "completed" and sub["results"][0]["condition"]["pred"] == "viewing"
    check = r["goal_check"]
    assert any(c["condition"]["pred"] == "viewing" and c["satisfied"] and c["confidence"] == "measured" for c in check)


def test_a_blocked_look_is_never_completed(tmp_path):
    t = ArmTools([plan(LOOK_LEFT)], ok=False, reason="motion blocked by the execution controls")
    t.request_kind = "description"
    t.decisions = [{"action": "stop", "reason": "the motion is blocked"}]
    r, _ = run(tmp_path, t, instruction="Look at your left side.")
    assert r["status"] == "FAILED" and "not executed" in r["message"] and "Gemma stopped" in r["message"]


def test_a_camera_that_ends_up_pointing_elsewhere_fails_the_goal(tmp_path, monkeypatch):
    """The worker reports success but the measured camera points elsewhere (e.g. the arm moved afterwards): the
    verification worker's measurement wins."""
    t = ArmTools([plan(LOOK_LEFT)])
    t.request_kind = "description"
    t.decisions = [{"action": "stop", "reason": "cannot confirm the view"}]
    real = caps.run_look

    def look_then_drift(ctx, args):
        res = real(ctx, args)
        t.q_now = np.array(ARM_HOME, float)  # the joints are back at HOME before the check
        return res
    monkeypatch.setattr(caps, "run_look", look_then_drift)
    r, _ = run(tmp_path, t, instruction="Look at your left side.")
    assert r["status"] != "COMPLETED"
    assert r["subgoal_checks"][0]["outcome"] == "failed" and "deg from left" in r["subgoal_checks"][0]["summary"]


def test_a_plan_with_nothing_verifiable_is_never_completed(tmp_path, monkeypatch):
    """E.g. Gemma answers "look left" with an explore that finds nothing: no step sets a checkable effect, so the
    goal is not shown; Gemma is asked, and without a better plan the task fails instead of completing."""
    monkeypatch.setattr(caps, "run_explore", lambda ctx, args: caps.StepResult("completed", "looked around"))
    t = ArmTools([plan(("explore", {"object": "thing"}))])
    t.decisions = [{"action": "stop", "reason": "nothing to check"}]
    r, _ = run(tmp_path, t, instruction="Look around.")
    assert r["status"] == "FAILED" and r["conditions"] == [] and "no step of the plan sets an effect" in t.decide_calls[0]["event"]


def test_moving_the_hand_is_not_turned_into_looking(tmp_path):
    """No capability moves the gripper to a free position: Gemma's empty plan is reported as unsupported, nothing moves."""
    t = ArmTools([plan(reason="I cannot move my hand to a free position; I can point my camera or pick and place")])
    r, _ = run(tmp_path, t, instruction="Move your hand to the left.")
    assert r["status"] == "UNSUPPORTED" and t.executed == []


def test_home_is_not_counted_as_looking_anywhere_in_particular():
    """At HOME the canted wrist camera looks down and to the front-left; it is not 'looking left' or 'looking down'."""
    errs = {d: view_error(ARM_HOME, d) for d in DIRECTIONS}
    assert not any(v["pointing"] for v in errs.values()), {d: v["error_deg"] for d, v in errs.items()}
