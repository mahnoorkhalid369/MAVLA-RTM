"""Held-object evidence from the wrist camera (agent/tools/verification.py wrist_sees_held), 2026-10-26: the table
camera could not see a thin block held in the gripper, the robot concluded "lost while lifting", and the next
instruction opened the gripper on it. The wrist camera rides on J5 and looks straight between the fingers."""

from types import SimpleNamespace

import numpy as np
import pytest

from control.kinematics import Kinematics
from control.robot import ARM_HOME
from perception.viewpoints import CAM_TIP

FX = 640 / np.tan(np.radians(69.4 / 2))
K = (FX, FX, 640.0, 360.0)


def _tools(depth, finger_deg=30.5):
    from agent.tools.verification import VerificationTools
    t = VerificationTools.__new__(VerificationTools)
    t.kin, t.wrist_kin, t.log = Kinematics(), Kinematics(tip=CAM_TIP), (lambda *a, **k: None)
    frame = SimpleNamespace(camera="wrist", depth=depth, K=K, q=np.array(ARM_HOME, float), finger=np.radians(finger_deg),
                            rgb=np.zeros((*depth.shape, 3), np.uint8), stamp=0.0)
    t.bridge = SimpleNamespace(snapshot=lambda cam: frame)
    return t


def _depth(between_fingers=None):
    d = np.full((720, 1280), 0.60)  # the table far below the gripper
    if between_fingers is not None:
        d[380:560, 590:690] = between_fingers  # a surface right between the fingers
    return d


def test_an_object_between_the_fingers_is_seen():
    r = _tools(_depth(0.15)).wrist_sees_held()
    assert r["held"] is True and r["fraction"] == 1.0
    assert all(560 > v > 380 and 690 > u > 590 for u, v in r["pixels"])  # the gripper axis, centre-bottom of the image


def test_seeing_through_to_the_table_is_empty():
    assert _tools(_depth()).wrist_sees_held()["held"] is False


def test_no_depth_there_is_no_evidence():
    assert _tools(np.zeros((720, 1280))).wrist_sees_held()["held"] is None


def test_the_axis_projects_to_the_same_pixels_whatever_j6_does():
    a = _tools(_depth(0.15))
    b = _tools(_depth(0.15))
    b.bridge.snapshot("wrist").q[5] += np.radians(90)
    assert a.wrist_sees_held()["pixels"] == b.wrist_sees_held()["pixels"]


def test_borderline_fingers_are_settled_by_the_wrist_camera():
    from config.agent import Tolerances
    t = _tools(_depth(0.15))
    t.tol = Tolerances()
    t.verify_contact = lambda target: {"check": "contact", "pass": False, "stopped_short_deg": 1.26}
    t.look_in_gripper = lambda oid: (_ for _ in ()).throw(AssertionError("the table camera is not needed"))
    v = t.confirm_held(0.5, 1, "retention_after_lift")
    assert v["pass"] and v["by"] == "fingers borderline -> wrist camera"


def test_a_pick_never_opens_the_gripper_on_an_unrecorded_object():
    pytest.importorskip("pydantic")  # agent.capabilities imports the Gemma roles
    from agent.capabilities.manipulation import run_pick, run_pick_place
    ctx = SimpleNamespace(tools=SimpleNamespace(unknown_held={"finger_deg": 30.5}, held_object=None))
    assert run_pick(ctx, {"object": {"text": "a"}}, 1).outcome == "unsafe"
    assert run_pick_place(ctx, {"object": {"text": "a"}, "relation": "in", "reference": {"text": "b"}}, True, 1).outcome == "unsafe"
