"""Placing INSIDE a container (agent/tools/manipulation.py _plan_in, agent/goals.py in(x, y)), 2026-10-26: the work
cell's trays are containers; "on" a tray meant its 40 mm rim. Real kinematics and collision checks, stub world."""

from types import SimpleNamespace

import numpy as np
import pytest

from control.collision import Box
from control.kinematics import Kinematics
from control.motion import PlanningError
from control.robot import ARM_HOME
from control.skills import Held

pytest.importorskip("pydantic")


def _tray(k, xy, yaw, size=(0.20, 0.14), rim=0.04, floor=0.004):
    g = {"centre": [xy[0], xy[1], rim / 2], "top_z": rim, "height": rim, "support_z": 0.0, "visible_bottom_z": floor,
         "extents": list(size), "footprint": list(size), "yaw_deg": yaw, "tilt_deg": 0.5}
    a = np.radians(yaw)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    o = SimpleNamespace(id=k, status="on_surface", geometry=g)
    o.box = lambda: Box(f"obj_{k}", np.array(g["centre"]), R, np.array([size[0] / 2, size[1] / 2, rim / 2]))
    return o


def _tools(objects):
    from agent.tools.manipulation import ManipulationTools
    t = ManipulationTools.__new__(ManipulationTools)
    from config.agent import Relations
    t.kin, t.log, t.rel = Kinematics(), (lambda *a, **k: None), Relations()
    t.world = SimpleNamespace(objects={o.id: o for o in objects})
    return t


def _block(k, xy, edge=0.05, floor=0.004):
    g = {"centre": [xy[0], xy[1], floor + edge / 2], "top_z": floor + edge, "support_z": floor, "footprint": [edge, edge],
         "extents": [edge, edge], "yaw_deg": 0.0}
    o = SimpleNamespace(id=k, status="on_surface", geometry=g)
    o.box = lambda: Box(f"obj_{k}", np.array(g["centre"]), np.eye(3), np.array([edge / 2, edge / 2, edge / 2]))
    return o


def test_a_held_block_goes_inside_each_workcell_tray_onto_its_floor():
    from scenarios.workcell import TRAY_SIZE, TRAYS
    trays = [_tray(k, xy, float(np.degrees(np.arctan2(xy[0], -xy[1]))), TRAY_SIZE)
             for k, (_, _, _, _, xy) in enumerate(TRAYS, 1)]
    t = _tools(trays)
    held = Held(height=0.05, grip_depth=0.025, width=0.05, length=0.05)
    for tray in trays:
        steps, centre, top = t._plan_in(np.array(ARM_HOME, float), held, tray.id)
        g = tray.geometry
        assert np.allclose(centre[:2], g["centre"][:2]) and top == pytest.approx(g["visible_bottom_z"] + 0.05)
        assert [getattr(s, "label", "") for s in steps][-2:] == ["open gripper", "retreat"]


def _red_tray(contents=()):
    from scenarios.workcell import TRAY_SIZE, TRAYS
    xy = TRAYS[0][4]
    tray = _tray(1, xy, float(np.degrees(np.arctan2(xy[0], -xy[1]))), TRAY_SIZE)
    a = np.radians(tray.geometry["yaw_deg"])
    blocks = [_block(k, (xy[0] + u * np.cos(a) - v * np.sin(a), xy[1] + u * np.sin(a) + v * np.cos(a)))
              for k, (u, v) in enumerate(contents, 2)]
    return _tools([tray] + blocks), tray, blocks


def test_a_block_goes_beside_what_is_already_in_the_tray():
    """Live 2026-10-26: the pink block sat at the red tray's centre; placing the red block there "found no collision-free
    way to lower" (three identical retries). The tray holds 3-4 blocks: the next one goes on a free part of its floor."""
    t, tray, (pink,) = _red_tray([(0.0, 0.0)])
    held = Held(height=0.05, grip_depth=0.025, width=0.05, length=0.05)
    steps, centre, _ = t._plan_in(np.array(ARM_HOME, float), held, tray.id)
    assert np.hypot(*(np.asarray(centre[:2]) - pink.geometry["centre"][:2])) >= 0.05 + 0.02


def test_four_blocks_fit_in_a_workcell_tray():
    t, tray, _ = _red_tray([(-0.07, -0.045), (0.07, -0.045), (-0.07, 0.045)])  # three corners taken
    held = Held(height=0.05, grip_depth=0.025, width=0.05, length=0.05)
    _, centre, _ = t._plan_in(np.array(ARM_HOME, float), held, tray.id)
    assert centre is not None


def test_a_full_tray_is_refused_once_and_not_retried():
    from agent.tools.manipulation import ManipulationTools
    t, tray, _ = _red_tray([(u, v) for u in (-0.09, 0.0, 0.09) for v in (-0.045, 0.045)])
    with pytest.raises(PlanningError, match="no free spot inside the container") as e:
        t._plan_in(np.array(ARM_HOME, float), Held(height=0.05, grip_depth=0.025, width=0.05, length=0.05), tray.id)
    assert "6 object(s) already in it" in str(e.value) and ManipulationTools.deterministic(str(e.value))


def test_the_container_is_walls_and_a_floor_not_a_solid_block():
    t = _tools([_tray(1, (0.0, 0.6), 0.0)])
    boxes = t._container_boxes(1)
    assert len(boxes) == 5 and all(b.name == "obj_1" for b in boxes)
    floor = boxes[-1]
    assert floor.centre[2] + floor.half[2] == pytest.approx(0.004)  # the floor slab ends at the measured floor
    walls = boxes[:4]
    assert all(w.centre[2] + w.half[2] == pytest.approx(0.04) for w in walls)  # walls up to the rim


def test_an_object_that_does_not_fit_is_refused_with_both_sizes():
    t = _tools([_tray(1, (0.0, 0.6), 0.0)])
    with pytest.raises(PlanningError, match="does not fit inside the container"):
        t._plan_in(np.array(ARM_HOME, float), Held(height=0.05, grip_depth=0.02, width=0.07, length=0.21), 1)


def test_in_holds_for_an_object_on_the_container_floor_and_not_beside_it():
    from agent.goals import GoalEvaluator
    tray = _tray(1, (0.0, 0.6), 0.0).geometry
    inside = {"centre": [0.01, 0.6, 0.03], "support_z": 0.004, "top_z": 0.054, "footprint": [0.05, 0.05]}
    beside = {**inside, "centre": [0.0, 0.75, 0.025], "support_z": 0.0}
    from agent.tools import Tools
    tools = SimpleNamespace(world=SimpleNamespace(holding=None, objects={1: SimpleNamespace(geometry=tray), 2: SimpleNamespace(geometry=inside),
                                                           3: SimpleNamespace(geometry=beside)}),
                            relation_holds=lambda a, b, rel, frame="robot": Tools.relation_holds(None, a, b, rel, frame))
    ev = GoalEvaluator.__new__(GoalEvaluator)
    ev.t, ev.s = tools, SimpleNamespace(verified=[])
    assert ev.evaluate({"pred": "in", "args": ["#2", "#1"]}, {}).satisfied is True
    assert ev.evaluate({"pred": "in", "args": ["#3", "#1"]}, {}).satisfied is False


def _verify(measured, relation="in"):
    """VerificationTools.verify_placement on a stub world: a 50 mm block planned into a tray (floor 4 mm, rim 40 mm)."""
    from agent.tools.manipulation import PickPlan
    from agent.tools.verification import VerificationTools
    from config.agent import Tolerances
    t = VerificationTools.__new__(VerificationTools)
    tray = _tray(1, (0.0, 0.6), 0.0)
    block = SimpleNamespace(id=2, status="on_surface", geometry={"centre": [0.0, 0.6, 0.029], "footprint": [0.05, 0.05],
                                                                 "top_z": 0.054, "support_z": 0.004})  # assume_placed
    t.world = SimpleNamespace(objects={1: tray, 2: block}, holding=2)
    t.tol, t.log = Tolerances(), (lambda *a, **k: None)
    geoms = {m: {"centre": [0.0, 0.6, top / 2], "top_z": top, "tilt_deg": 0.0, "support_z": 0.0, "height": top,
                 "footprint": [0.05, 0.05]} for m, top in measured.items()}
    t.perceive = lambda cam: SimpleNamespace(measured=geoms, frame_id="table_009", object_ids={})  # associated later
    geoms[7]["footprint"] = [0.20, 0.14]  # mark 7 is the tray: its rim height and size
    t.update_world = lambda obs: []
    held = Held(height=0.05, grip_depth=0.025, width=0.05, length=0.05)
    plan = PickPlan(2, [], [], [], [], held, {}, [0.0, 0.6, 0.029], 0.054, relation, 1)
    v = t.verify_placement(plan)
    v["record"] = block.geometry
    return v


def test_a_placement_inside_a_container_is_checked_on_the_object_not_the_container():
    """Live 2026-10-26: block and tray share a centre; the tray's 40 mm rim was measured as the placed object
    ("z -14.5 mm, out of tolerance") although the block was in the tray at the planned height."""
    assert _verify({7: 0.040, 8: 0.054})["pass"]  # mark 7 is the tray: skipped; mark 8, the block, is checked
    assert not _verify({7: 0.040, 8: 0.080})["pass"]  # the block is 26 mm too high: not where it was put


def test_the_destination_is_never_taken_for_the_placed_object_and_never_written_into_its_record():
    """Live 2026-10-26: "place it on the red tray": the tray's rim was measured as the block ("z -120.5 mm") and the
    block's record took the tray's size and place, so the world held the tray twice and lost the block."""
    for relation in ("on", "in"):
        v = _verify({7: 0.040}, relation=relation)  # the camera sees only the tray there
        assert not v["pass"] and v["reason"].startswith("no object measured")
        assert v["record"]["footprint"] == [0.05, 0.05]  # still the block's own record (the planned pose)


def test_a_failed_check_keeps_the_measurement_only_when_it_is_the_placed_object():
    v = _verify({7: 0.040, 8: 0.080})  # the block, 26 mm too high: where it really is
    assert not v["pass"] and v["record"]["top_z"] == 0.080
    from agent.tools.verification import VerificationTools
    assert not VerificationTools._same_size({"footprint": [0.20, 0.14]}, {"footprint": [0.05, 0.05]})
