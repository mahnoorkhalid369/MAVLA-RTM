"""World model under occlusion (perception/world_model.py WorldModel.observe): an object covered by another keeps its
remembered geometry, and an object whose lower part is hidden keeps its remembered support. Replays the live run of
2026-10-21 (purple cylinder placed on the 71 mm maroon block; the placement check passed, the goal check said "not on"
from the partial views). Measured numbers from out/demo/2026-10-21/live_081925/02 (table_003)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")  # tests.support.fakes uses the agent's schemas

from agent.goals import GoalEvaluator  # noqa: E402
from perception.world_model import WorldModel  # noqa: E402
from tests.support.fakes import geom  # noqa: E402


def placed_scene():
    w = WorldModel()
    block = w.observe(geom(0.2816, 0.7973, 0.0707, fp=(0.0395, 0.0395)), None)  # the maroon block, before the place
    cyl = w.observe(geom(0.4516, 0.4476, 0.0707, fp=(0.051, 0.051)), None)  # the purple cylinder, before the pick
    g = dict(w.objects[cyl].geometry)  # Tools.assume_placed: the released object at its planned pose
    g.update(centre=[0.2816, 0.7973, 0.1061], support_z=0.0707, top_z=0.1414)
    w.objects[cyl].geometry = g
    # the re-observation: the cylinder's lower part is hidden (its "support" appears at 89 mm), and the block is
    # seen only as a sliver of its side below it (top 60 mm, 43 x 17 mm)
    seen_cyl = geom(0.2842, 0.7962, 0.1414, fp=(0.052, 0.052), support=0.0894)
    sliver = geom(0.2985, 0.7943, 0.0601, fp=(0.043, 0.017))
    frame = [sliver, seen_cyl]
    ids = [w.observe(m, None, others=frame) for m in frame]
    return w, block, cyl, ids


def test_the_covered_support_and_the_placed_object_keep_their_geometry():
    w, block, cyl, ids = placed_scene()
    assert ids == [block, cyl]  # the same physical objects (data association unchanged)
    b, c = w.objects[block].geometry, w.objects[cyl].geometry
    assert abs(b["top_z"] - 0.0707) < 1e-6 and abs(b["centre"][0] - 0.2816) < 1e-6  # not the sliver
    assert abs(c["support_z"] - 0.0707) < 1e-6 and abs(c["height"] - 0.0707) < 1e-3  # not the hidden-bottom estimate
    assert abs(c["centre"][0] - 0.2842) < 1e-6  # the new top position is used


def test_the_goal_check_now_sees_the_placement():
    w, block, cyl, _ = placed_scene()
    ev = GoalEvaluator(SimpleNamespace(world=w), None)
    ok, conf, why = ev._on([cyl, block], {"pred": "on"})
    assert ok is True and conf == "measured", why


def test_an_uncovered_object_is_updated_as_before():
    w = WorldModel()
    oid = w.observe(geom(0.30, 0.50, 0.050), None)
    moved = geom(0.31, 0.50, 0.050)
    assert w.observe(moved, None, others=[moved]) == oid and w.objects[oid].geometry is moved


# ------------------------------------------------------------------ a stack seen as one object
def stacked_scene():
    """Live 2026-10-21 (out/demo/2026-10-21/live_084858/02): a 70.7 mm object placed on the 101 mm block; the table
    camera then measured ONE object: top 171 mm, bottom on the floor."""
    w = WorldModel()
    block = w.observe(geom(-0.4984, 0.5464, 0.101, fp=(0.05, 0.05)), None)
    held = w.observe(geom(0.4516, 0.4475, 0.0707, fp=(0.052, 0.052)), None)
    g = dict(w.objects[held].geometry)  # Tools.assume_placed
    g.update(centre=[-0.4984, 0.5464, 0.1363], support_z=0.101, top_z=0.1716)
    w.objects[held].geometry = g
    merged = geom(-0.495, 0.546, 0.171, fp=(0.045, 0.045))  # support 0: the whole stack
    return w, block, held, merged


def test_a_merged_stack_does_not_move_the_placed_objects_bottom_to_the_floor():
    w, block, held, merged = stacked_scene()
    assert w.observe(merged, None, others=[merged]) == held
    g = w.objects[held].geometry
    assert abs(g["support_z"] - 0.101) < 1e-6 and abs(w.objects[block].geometry["top_z"] - 0.101) < 1e-6
    ok, _, why = GoalEvaluator(SimpleNamespace(world=w), None)._on([held, block], {"pred": "on"})
    assert ok is True, why


def test_a_lower_bottom_without_anything_below_is_taken_as_measured():
    w = WorldModel()
    oid = w.observe(geom(0.3, 0.5, 0.06, support=0.02), None)  # remembered 20 mm up, nothing under it
    g = geom(0.3, 0.5, 0.06)
    w.observe(g, None, others=[g])
    assert w.objects[oid].geometry["support_z"] == 0.0


def test_the_goal_check_accepts_the_workers_passed_placement_check():
    """The world model's bottom is wrong (merged view), but the worker's camera check of this very placement passed."""
    w, block, held, merged = stacked_scene()
    w.objects[held].geometry = dict(merged)  # as the old placement check stored it
    rec = {"check": "placement", "object_id": held, "relation": "on", "reference_id": block, "pass": True,
           "xy_error_mm": 2.9, "z_error_mm": -0.7, "frame_id": "table_003"}
    t = SimpleNamespace(world=w)
    ok, conf, why = GoalEvaluator(t, SimpleNamespace(verified=[rec]))._on([held, block], {"pred": "on"})
    assert ok is True and conf == "measured" and "placement check passed" in why
    other = dict(rec, reference_id=999)  # a check of another destination proves nothing about this one
    assert GoalEvaluator(t, SimpleNamespace(verified=[other]))._on([held, block], {"pred": "on"})[0] is False
    later_grasp = [rec, {"check": "contact", "pass": True}]  # picked up again since: the record is stale
    assert GoalEvaluator(t, SimpleNamespace(verified=later_grasp))._on([held, block], {"pred": "on"})[0] is False
