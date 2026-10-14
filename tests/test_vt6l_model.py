"""The VT6L model (assets/vt6l/vt6l.urdf) against numbers printed in the Epson VT manipulator manual
("VT series Rev.6": Appendix A p.175, Outer Dimensions p.42, Standard Motion Range p.44-45).

Every expected value below is copied from the manual, not derived from the URDF.
"""

import numpy as np
import pytest

from control.kinematics import URDF_PATH, Kinematics

P = Kinematics(tip="p_point")  # wrist reference point: intersection of the J4, J5, J6 axes
FLANGE = Kinematics(tip="tool0")
J2_CENTRE = np.array([0.0, 0.100, 0.412])  # J2 axis: 100 mm in front of J1, 412 mm above the base


def p_point(j2=0.0, j3=0.0):
    return P.fk(np.radians([0, j2, j3, 0, 0, 0]))[:3, 3]


def test_the_project_uses_the_vt6l_model():
    assert URDF_PATH.name == "vt6l.urdf"


def test_zero_pose_matches_the_drawing():
    # 0 pulse position: upper arm vertical (420), forearm horizontal (400), flange 80 beyond P
    assert np.allclose(p_point(), [0.0, 0.500, 0.832], atol=1e-6)
    assert np.allclose(FLANGE.fk(np.zeros(6))[:3, 3], [0.0, 0.580, 0.832], atol=1e-6)


def test_reach_to_p_point_and_flange():
    # arm stretched forward horizontally: J2 -90 (upper arm forward), J3 +90 (forearm in line)
    q = np.radians([0, -90, 90, 0, 0, 0])
    assert np.hypot(*P.fk(q)[:2, 3]) == pytest.approx(0.920, abs=1e-6)  # "arm length 920 mm"
    assert np.hypot(*FLANGE.fk(q)[:2, 3]) == pytest.approx(1.000, abs=1e-6)  # 1000 mm to the flange
    assert P.fk(np.radians([0, 0, 90, 0, 0, 0]))[2, 3] == pytest.approx(1.232, abs=1e-6)  # top of range


@pytest.mark.parametrize("j3, mm", [(-51, 274.4), (190, 527.3)])  # footnotes *1, *2: J2 centre -> P point
def test_motion_range_radii_from_j2(j3, mm):
    assert np.linalg.norm(p_point(j3=j3) - J2_CENTRE) * 1000 == pytest.approx(mm, abs=0.1)


@pytest.mark.parametrize("j3, mm", [(-51, 351.7), (190, 293.9)])  # footnotes *3, *4: J1 centre -> P point
def test_motion_range_radii_from_j1(j3, mm):
    assert np.hypot(*p_point(j3=j3)[:2]) * 1000 == pytest.approx(mm, abs=0.1)


def test_joint_ranges_and_speeds_from_appendix_a():
    lo, hi = np.degrees(FLANGE.lower), np.degrees(FLANGE.upper)
    assert np.allclose(lo, [-170, -160, -51, -200, -125, -360], atol=0.01)
    assert np.allclose(hi, [170, 65, 190, 200, 125, 360], atol=0.01)
    speeds = np.degrees(FLANGE.max_velocity)
    assert np.allclose(speeds, [166.2, 122.5, 141.2, 268.7, 296.8, 293.2], atol=0.05)  # AC standard model
