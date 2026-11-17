"""Collision checking (control/collision.py) and its use in the skill planner.

Cases: known collision-free poses, deliberate obstacle intersections, floor contact (allowed for the
fingertips only within the grasp margin), self-collision within the joint limits, swept paths whose
end points are free but whose middle is not, and the planner rejecting unsafe pick plans.
"""

import numpy as np
import pytest

from control.collision import Box, CollisionChecker
from control.kinematics import R_DOWN, Kinematics
from control.motion import PlanningError, joint_move
from control.robot import ARM_HOME, ARM_SPEED
from control.skills import SkillPlanner
from sim.scene_objects import OBJECTS, obstacle_boxes

KIN = Kinematics()
SCENE = CollisionChecker(obstacle_boxes())
EMPTY = CollisionChecker()


def pairs(contacts) -> set:
    return {frozenset((c.a, c.b)) for c in contacts}


def tcp_down(x, y, z):
    q, ok = KIN.ik_best(np.array([x, y, z]), R_DOWN, ARM_HOME)
    assert ok
    return q


def test_known_free_poses():
    assert SCENE.check(ARM_HOME) == []
    assert SCENE.check(np.radians([0, -5, -50, 45, -5, 0])) == []  # observe pose
    assert EMPTY.check(np.zeros(6)) == []  # stretched straight up


def test_obstacle_at_the_gripper_is_detected():
    T = KIN.fk(ARM_HOME)
    palm = T[:3, 3] - 0.09 * T[:3, 2]  # 90 mm back from the TCP towards the flange (the TCP itself is the empty finger gap)
    checker = CollisionChecker([Box("probe", palm, np.eye(3), np.full(3, 0.02))])
    assert any(c.b == "probe" and c.a in ("pads", "palm", "linkages") for c in checker.check(ARM_HOME))


def test_floor_contact_only_below_the_grasp_height():
    assert EMPTY.check(tcp_down(0.0, 0.6, 0.0225)) == []  # 45 mm cube grasp: tips ~8 mm above the floor
    hits = EMPTY.check(tcp_down(0.0, 0.6, 0.005))  # tips 10 mm below the floor
    assert frozenset(("pads", "floor")) in pairs(hits)


@pytest.mark.parametrize("deg, pair", [
    ([111, -146, -29, 185, 63, -117], ("base_link", "palm")),  # gripper driven into the base
    ([149, -156, 158, 185, 11, 337], ("base_link", "j2_link")),  # upper arm swung down into the base
    ([25, -58, -49, 8, -93, -356], ("j1_link", "pads")),  # fingers folded back into the shoulder
])
def test_self_collision_within_joint_limits(deg, pair):
    q = np.radians(deg)
    assert np.all(q >= KIN.lower) and np.all(q <= KIN.upper)
    assert frozenset(pair) in pairs(EMPTY.check(q))


def test_adjacent_links_are_excluded_and_listed():
    assert frozenset(("j2_link", "j3_link")) in EMPTY.excluded_pairs
    assert frozenset(("j2_link", "j4_link")) in EMPTY.excluded_pairs  # J3 fold: Epson's limits
    for real in (("base_link", "palm"), ("j1_link", "pads"), ("base_link", "j2_link"), ("j4_link", "pads")):
        assert frozenset(real) not in EMPTY.excluded_pairs  # real collision pairs stay checked


def test_swept_path_is_checked_between_free_end_points():
    a, b = tcp_down(-0.3, 0.5, 0.15), tcp_down(0.3, 0.5, 0.15)
    path = joint_move(KIN, a, b, ARM_SPEED).positions
    T_mid = KIN.fk(path[len(path) // 2])
    palm_mid = T_mid[:3, 3] - 0.09 * T_mid[:3, 2]  # where the palm passes half-way
    wall = CollisionChecker([Box("wall", palm_mid, np.eye(3), np.array([0.01, 0.01, 0.03]))])
    assert wall.check(a) == [] and wall.check(b) == []
    res = wall.check_path(path)
    assert not res.ok and any(c.b == "wall" for c in res.contacts)


def test_planner_accepts_a_clear_pick_and_rejects_a_blocked_one():
    red = OBJECTS["red_cube"]
    top = np.array([*red.xy, red.height])
    SkillPlanner(KIN, collision=SCENE).pick(ARM_HOME, top, width=0.06, target="red_cube")
    lid = Box("lid", top + [0, 0, 0.07], np.eye(3), np.array([0.05, 0.05, 0.005]))  # plate above the cube
    with pytest.raises(PlanningError, match="collision"):
        SkillPlanner(KIN, collision=CollisionChecker(obstacle_boxes() + [lid])).pick(ARM_HOME, top, width=0.06, target="red_cube")


def test_grasp_yaw_matters_for_a_rotated_block():
    """A 70 mm cube turned 45 deg is 99 mm across its corners: the open pads (87 mm) only clear it
    when the gripper is turned to its faces."""
    a = np.radians(45)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    cube = Box("big", np.array([0.0, 0.6, 0.035]), R, np.full(3, 0.035))
    planner = SkillPlanner(KIN, collision=CollisionChecker([cube]))
    with pytest.raises(PlanningError, match="collision"):
        planner.pick(ARM_HOME, np.array([0.0, 0.6, 0.07]), width=0.07, yaw_deg=0.0, target="big")
    planner.pick(ARM_HOME, np.array([0.0, 0.6, 0.07]), width=0.07, yaw_deg=45.0, target="big")


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_tiers_with_collisions(name):
    """With every object and fixture as an obstacle: easy..hard objects plan unless something is
    stacked on them (you cannot pick the bottom of a stack); impossible ones are refused."""
    o = OBJECTS[name]
    blocked = any(other.on == name for other in OBJECTS.values())
    try:
        SkillPlanner(KIN, collision=SCENE).pick(
            ARM_HOME, np.array([*o.xy, o.base_z + o.height]), width=min(o.size[:2]), yaw_deg=o.yaw_deg,
            target=name, length=max(o.size[:2]), support=o.on, height=o.height)
        feasible = True
    except PlanningError:
        feasible = False
    assert feasible == (o.tier != "impossible" and not blocked)


def test_conservative_path_catches_a_thin_wall_between_samples(monkeypatch):
    """With samples 10 deg apart, a 2 mm wall half-way along the palm's path lies between them:
    checking the samples alone misses it; conservative advancement subdivides and finds it."""
    import control.collision as cc
    monkeypatch.setattr(cc, "MAX_STEP_DEG", 10.0)
    a = tcp_down(-0.10, 0.60, 0.25)
    b = a.copy()
    b[0] += np.radians(10.0)  # J1 only
    T_mid = KIN.fk((a + b) / 2)
    palm = T_mid[:3, 3] - 0.09 * T_mid[:3, 2]
    radial = palm[:2] / np.linalg.norm(palm[:2])
    R = np.array([[radial[0], -radial[1], 0], [radial[1], radial[0], 0], [0, 0, 1]])  # wall across the sweep
    wall = CollisionChecker([Box("thin_wall", palm, R, np.array([0.05, 0.001, 0.05]))])
    assert wall.check(a) == [] and wall.check(b) == []  # both samples are clear
    res = wall.check_path(np.array([a, b]))
    assert not res.ok and any(c.b == "thin_wall" for c in res.contacts)


REVIEWED_EXCLUSIONS = {  # changing this list must be a deliberate, reviewed decision
    ("base_link", "j1_link"), ("j1_link", "j2_link"), ("j2_link", "j3_link"), ("j3_link", "j4_link"),
    ("j4_link", "j5_link"), ("j5_link", "j6_link"),  # joined by a joint
    ("camera_link", "j5_link"),  # camera bolted to the J5 housing
    ("j5_link", "palm"), ("j5_link", "pads"), ("j5_link", "linkages"), ("held", "j5_link"),  # turn about J5's own axis inside its round housing
    ("j6_link", "palm"), ("j6_link", "pads"), ("j6_link", "linkages"), ("held", "j6_link"),  # gripper bolted to the flange
    ("palm", "pads"), ("palm", "linkages"), ("linkages", "pads"),  # parts of the gripper
    ("held", "palm"), ("held", "pads"), ("held", "linkages"),  # the held object touches the gripper by design
    ("j2_link", "j4_link"),  # ASSUMPTION: only J3 moves them; Epson's J3 range taken as the fold limit (verify on CAD / robot)
}


def test_excluded_pairs_are_the_reviewed_list():
    assert {tuple(sorted(p)) for p in EMPTY.excluded_pairs} == {tuple(sorted(p)) for p in REVIEWED_EXCLUSIONS}
