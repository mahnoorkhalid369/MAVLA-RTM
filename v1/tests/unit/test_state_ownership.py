"""Shared-state ownership (agent/tools/held.py): the held object is recorded twice, Tools.held_object and
WorldModel.holding. Until they are merged into one record, every Tools operation that changes either must keep
them equal. Real Tools; the bridge is a stub with a settable finger angle (no robot)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pydantic")  # tests.support.fakes uses the agent schemas

from agent.tools import Tools  # noqa: E402
from tests.support.fakes import geom  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    finger = {"rad": 0.0}
    bridge = SimpleNamespace(finger=lambda: finger["rad"], q=lambda: np.zeros(6))
    t = Tools(bridge, None, None, None, lambda *a, **k: None, tmp_path)
    oid = t.world.observe(geom(0.5, 0.2, 0.05), None)
    held = SimpleNamespace(collision_body=lambda: (np.eye(4), (0.04, 0.04, 0.05)), height=0.05, width=0.04, yaw_deg=0.0)
    plan = SimpleNamespace(held=held, object_id=oid, target_centre=[0.6, 0.1, 0.025], target_top=0.05)
    return t, plan, finger


def agree(t) -> bool:
    return t.world.holding == t.held_object and (t.held_object is None) == (t.held_box is None) == (t.held is None)


def test_set_and_clear_keep_both_records_equal(setup):
    t, plan, _ = setup
    assert agree(t) and t.held_object is None
    t.set_held(plan, close_target=0.5)
    assert agree(t) and t.held_object == plan.object_id and t.world.objects[plan.object_id].status == "held"
    t.clear_held()
    assert agree(t) and t.held_object is None


def test_release_keeps_both_records_equal(setup):
    t, plan, _ = setup
    t.set_held(plan, close_target=0.5)
    t.assume_placed(plan)
    assert agree(t) and t.held_object is None and t.world.objects[plan.object_id].status == "on_surface"


def test_revalidation_keeps_both_records_equal(setup):
    t, plan, finger = setup
    t.set_held(plan, close_target=0.5)
    finger["rad"] = 0.3  # stalled short of the close target: still holding
    assert t.revalidate_held()["held"] == "verified" and agree(t) and t.held_object == plan.object_id
    finger["rad"] = 0.5  # closed all the way: nothing between the fingers
    assert t.revalidate_held()["held"] == "invalidated" and agree(t) and t.held_object is None
    assert t.world.objects[plan.object_id].status == "missing"
