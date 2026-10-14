"""Every test-field object's difficulty tier against the VT6L model's real reach.

Plans the top-down pick (control/skills.py) for each object in sim/scene_objects.py:
easy / medium / hard must plan (their difficulty is physical: size, rotation, rolling, weight),
impossible must be refused. Re-run after any change to the robot model or the skills.
"""

import numpy as np
import pytest

from control.kinematics import Kinematics
from control.motion import PlanningError
from control.robot import ARM_HOME
from control.skills import SkillPlanner
from sim.scene_objects import OBJECTS

SKILLS = SkillPlanner(Kinematics())


def plans(top, width, yaw=0.0) -> bool:
    try:
        SKILLS.pick(ARM_HOME, np.asarray(top, float), width=width, yaw_deg=yaw)
        return True
    except PlanningError:
        return False


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_tier_matches_reach(name):
    o = OBJECTS[name]
    feasible = plans((*o.xy, o.base_z + o.height), min(o.size[:2]), o.yaw_deg)
    assert feasible == (o.tier != "impossible"), f"{name} ({o.tier}) at {np.hypot(*o.xy):.3f} m"


def test_edge_cubes_sit_at_the_edges():
    navy, olive = OBJECTS["navy_cube_far"], OBJECTS["olive_cube_near_base"]
    r_far = np.hypot(*navy.xy)
    assert not plans((0.0, r_far + 0.02, navy.height), 0.045)  # 20 mm further out is beyond reach
    assert 0.415 <= np.hypot(*olive.xy) <= 0.44  # just outside the reliable near edge
