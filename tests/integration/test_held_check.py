"""Is the object still held? Fingers first, then the table camera when the fingers are borderline
(agent/tools/verification.py confirm_held / look_in_gripper), at the lift and carry checks and at the start of the next
instruction (agent/tools/held.py revalidate_held). Live 2026-10-21 (out/demo/2026-10-21/live_092646/01): after the lift
the fingers stopped 1.45 deg short (threshold 1.5) with the purple object still in them; the task was reported as
"object lost", and the next instruction did not know it was holding anything. Stub sensors (no sim)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")

from agent.tools import Tools  # noqa: E402
from tests.support.fakes import geom  # noqa: E402

TARGET = np.radians(25.14)  # the close target of the live grasp
TCP = np.eye(4)
TCP[:3, 3] = [0.452, 0.447, 0.19]  # fingertip centre after the lift


def tools(short_deg, measured, excluded=()):
    t = Tools(bridge=None, detector=None, roles=None, executor=None, log=lambda *a, **k: None, run_dir=Path("."))
    t.bridge = SimpleNamespace(finger=lambda: TARGET - np.radians(short_deg), q=lambda: np.zeros(6))
    t.kin = SimpleNamespace(fk=lambda q: TCP)
    t.looks = 0

    def perceive(camera):
        t.looks += 1
        return SimpleNamespace(frame_id="table_002", measured=dict(enumerate(measured, 1)), excluded=list(excluded))
    t.perceive = perceive
    oid = t.world.observe(geom(0.4516, 0.4475, 0.0707, fp=(0.052, 0.052)), None)  # the object, before the pick
    return t, oid


IN_FINGERS = geom(0.466, 0.431, 0.205, fp=(0.018, 0.008), support=0.135)  # what the camera saw in the gripper
ON_SPOT = geom(0.4516, 0.4475, 0.0707, fp=(0.052, 0.052))


def test_clear_contact_needs_no_look():
    t, oid = tools(1.84, [])
    v = t.confirm_held(TARGET, oid, "retention_after_lift")
    assert v["pass"] and v["by"] == "fingers" and t.looks == 0


def test_closed_on_nothing_needs_no_look():
    t, oid = tools(0.1, [])
    v = t.confirm_held(TARGET, oid, "retention_after_lift")
    assert not v["pass"] and t.looks == 0


def test_borderline_fingers_the_camera_sees_it_in_the_gripper():
    t, oid = tools(1.45, [IN_FINGERS])
    v = t.confirm_held(TARGET, oid, "retention_after_lift")
    assert v["pass"] and t.looks == 1 and v["camera"]["in_gripper"] and v["camera"]["old_spot"] == "empty"


def test_borderline_fingers_and_the_object_still_on_the_table_is_not_held():
    t, oid = tools(1.45, [ON_SPOT])
    v = t.confirm_held(TARGET, oid, "retention_after_lift")
    assert not v["pass"] and v["camera"]["old_spot"] == "occupied"


def test_the_held_object_dropped_by_perception_counts_as_seen_in_the_gripper():
    t, oid = tools(1.45, [], excluded=[{"reason": "the held object"}])
    assert t.confirm_held(TARGET, oid, "retention_after_lift")["pass"]


def test_the_next_instruction_recovers_a_grasp_the_lift_check_could_not_confirm():
    t, oid = tools(1.45, [IN_FINGERS])
    from control.skills import Held
    plan = SimpleNamespace(object_id=oid, held=Held(0.052, 0.035, 0, f"obj_{oid}", 0.052, 0.052))
    t.unconfirmed_grasp = (plan, TARGET)
    ev = t.revalidate_held()
    assert ev["held"] == "recovered" and t.held_object == oid and t.world.holding == oid
    assert t.unconfirmed_grasp is None
