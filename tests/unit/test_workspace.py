"""The measured pick envelope (control/workspace.py) and the safety layer's Epson joint-combination zones
(safety/joint_limits.py): what the simulator shades as no-go, and what no motion may enter."""

import numpy as np
import pytest

from control.workspace import PICK_ENVELOPE, SHADE_TO_M, envelope, no_go_mesh, pickable
from safety.joint_limits import JointLimits, check_trajectory, forbidden_combination, forbidden_on_path


def test_the_envelope_matches_the_epson_shape():
    for b in (0.0, 45.0, 90.0, -90.0):  # in front and to the sides: inner hole ~0.25 m, outer edge ~1 m
        inner, outer = envelope(b)
        assert 0.2 <= inner <= 0.3 and 0.95 <= outer <= 1.05
    inner, outer = envelope(180.0)  # straight behind: J1 cannot face it; only a short tilted band
    assert inner >= 0.4 and outer <= 0.8
    assert envelope(-180.0) == envelope(180.0) and envelope(30.0) == pytest.approx(envelope(30.0 + 360.0))


def test_known_spots():
    assert pickable(0.0, 0.6)  # in front, top-down ring
    assert pickable(0.002, -0.621)  # the live block straight behind (tilted 40 deg)
    assert not pickable(0.0, 0.15)  # inside the hole round the base
    assert not pickable(0.0, 1.2)  # beyond the reach
    assert not pickable(0.0, -0.9)  # far behind


def test_the_no_go_mesh_covers_exactly_the_floor_outside_the_envelope():
    pts, counts, idx = no_go_mesh(step_deg=2.0)
    assert len(idx) == sum(counts) and max(idx) < len(pts) and np.all(pts[:, 2] > 0)  # just above the floor
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert r.max() == pytest.approx(SHADE_TO_M)
    for quad in np.array(idx).reshape(-1, 4):  # every quad is the hole or the outer band of its slice, never the ring
        c = pts[quad].mean(axis=0)
        assert not pickable(c[0], c[1])
    edge = {tuple(np.round(p, 9)) for p in pts}  # neighbouring slices share their corner points: a smooth boundary
    assert len(edge) < len(pts) * 0.6


def test_epson_forbidden_combinations():
    deg = np.radians
    assert forbidden_combination(deg([0, 0, 0, 0, -90, 0])) is None  # HOME
    assert forbidden_combination(deg([0, -150, -40, 0, 0, 0]))  # front chart: J2 far forward, forearm down
    assert forbidden_combination(deg([150, 0, -45, 0, 0, 0]))  # back chart: forearm below -40 with |J1| > 110
    assert forbidden_combination(deg([150, 0, -36, 0, 0, 0])) is None
    assert forbidden_combination(deg([150, 0, -39, 0, 0, 0]))  # within the 2 deg margin of the boundary
    assert forbidden_combination(deg([0, 0, -45, 0, 0, 0])) is None  # the same J2/J3 is allowed facing forward


def test_the_safety_layer_rejects_a_trajectory_through_a_forbidden_zone():
    lim = JointLimits.from_urdf()
    path = np.radians([[150, 0, -30, 0, 0, 0], [150, 0, -45, 0, 0, 0], [150, 0, -30, 0, 0, 0]])
    problems = check_trajectory(lim, lim.names, [0.0, 2.0, 4.0], path)
    assert any("Epson manual forbids" in p for p in problems)
    assert forbidden_on_path(path).startswith("point 1:")


def test_envelope_bearings_are_measured_both_sides_or_mirrored():
    keys = set(PICK_ENVELOPE)
    assert all(-k in keys or k in (0.0, 180.0) for k in keys)


def test_on_a_table_the_shading_stops_at_the_table_edge():
    """scenarios/workcell.py: the no-go shading is clipped to the table top (nothing hangs in the air beyond it)."""
    from control.workspace import no_go_clipped
    pts, counts, idx = no_go_clipped(-1.15, 1.15, -0.40, 1.10)
    assert len(idx) == sum(counts) and min(counts) >= 3
    assert pts[:, 0].min() >= -1.15 - 1e-9 and pts[:, 0].max() <= 1.15 + 1e-9
    assert pts[:, 1].min() >= -0.40 - 1e-9 and pts[:, 1].max() <= 1.10 + 1e-9
    k = 0
    for n in counts:  # every clipped polygon is outside the pick envelope (its centroid)
        c = pts[idx[k:k + n]].mean(axis=0)
        assert not pickable(c[0], c[1])
        k += n
