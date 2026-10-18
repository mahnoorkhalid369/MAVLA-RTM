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
