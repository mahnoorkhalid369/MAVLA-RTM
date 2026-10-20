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
    assert '"frame": "robot" | "camera"' in s.args and all(d in s.args for d in DIRECTIONS)


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
