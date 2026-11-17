"""Regressions for invalid visual IDs, bad RGB-D bundles, hidden supports and complete grasp candidates."""

from types import SimpleNamespace as NS
import threading

import cv2
import numpy as np
import pytest

from agent import workers
from agent.contracts import Report
from agent.tools import Tools
from agent.tools.manipulation import PickPlan
from control.robot import ARM_HOME
from control.skills import Held, SkillPlanner
from perception.capture import CaptureBuffer, ObservationError, check_table_plane
from perception.geometry import snap_supports, round_section_grasp, measure
from perception.instances import parent_of_parts
from perception.somark import layout
from perception.world_model import WorldModel
from tests.support.fakes import geom
from tests.unit.test_master_workers import FakeGemma, FakeTools, FakeScene, FakePickPlace, log


def capture():
    c = CaptureBuffer()
    rgb, depth = np.zeros((24, 32, 3), np.uint8), np.ones((24, 32), np.float32)
    c.images['table', 'rgb'].append((rgb, 1., 'optical'))
    for kind in ('rgb', 'depth'):
        c.intrinsics['table', kind].append(((100, 100, 16, 12), (24, 32), 'optical', 1.))
    c.joints.append((1., np.array(ARM_HOME), 0.))
    c.poses['table'].append((1., np.eye(4)))
    return c, depth


def test_capture_waits_for_same_exposure_and_rejects_mismatched_calibration():
    c, depth = capture()
    c.images['table', 'depth'].append((depth, 1.08, 'optical'))
    assert c.select('table', .9) is None
    c.images['table', 'depth'].append((depth, 1., 'optical'))
    f = c.select('table', .9)
    assert f.stamp == f.depth_stamp == f.joint_stamp == f.pose_stamp
    c.intrinsics['table', 'depth'].clear()
    c.intrinsics['table', 'depth'].append(((110, 110, 16, 12), (24, 32), 'optical', 1.))
    with pytest.raises(ObservationError, match='calibration'):
        c.select('table', .9)


def test_capture_rejects_moving_joints_and_outdated_extrinsics():
    c, depth = capture()
    c.images['table', 'depth'].append((depth, 1., 'optical'))
    c.joints.append((1.02, np.array(ARM_HOME) + .02, 0.))
    assert c.select('table', .9) is None
    c.joints.pop()
    c.poses['table'].clear()
    c.poses['table'].append((.5, np.eye(4)))
    assert c.select('table', .9) is None


def test_ros_decoders_honor_pixel_encoding_padding_endianness_and_camera_quaternion():
    from agent.ros_bridge import RosBridge
    b = RosBridge.__new__(RosBridge)
    b.lock, b.latest, b.capture = threading.Lock(), {}, CaptureBuffer()
    header = NS(stamp=NS(sec=1, nanosec=0), frame_id='optical')
    rows = np.zeros((2, 12), np.uint8)
    rows[:, :9] = np.tile([0, 0, 255], (2, 3))
    b._img('table', 'rgb', NS(data=rows.tobytes(), height=2, width=3, step=12, encoding='bgr8', header=header))
    assert b.latest['table', 'rgb'][0][0, 0].tolist() == [255, 0, 0]
    rows = np.zeros((2, 8), np.uint8)
    rows[:, :6] = np.full((2, 3), 1200, dtype='>u2').view(np.uint8).reshape(2, 6)
    b._img('table', 'depth', NS(data=rows.tobytes(), height=2, width=3, step=8, encoding='16UC1',
                               is_bigendian=True, header=header))
    assert np.allclose(b.latest['table', 'depth'][0], 1.2)
    b._pose('table', NS(header=NS(stamp=header.stamp, frame_id='world'), pose=NS(
        position=NS(x=.1, y=.2, z=1.), orientation=NS(x=0., y=0., z=np.sqrt(.5), w=np.sqrt(.5)))))
    assert np.allclose(b.capture.poses['table'][0][1][:3, :3] @ [1., 0., 0.], [0., 1., 0.])


def test_broad_table_plane_rejects_the_logged_155mm_coordinate_jump():
    x, y = np.meshgrid(np.linspace(.2, .8, 30), np.linspace(.1, .7, 30))
    p = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    assert check_table_plane(p) == pytest.approx(0.)
    with pytest.raises(ObservationError, match='155.0 mm'):
        check_table_plane(p + [0, .094, .155])


def test_world_rejects_coherent_coordinate_jump_without_allocating_new_ids():
    w = WorldModel()
    masks, measured = {}, {}
    for k in range(1, 5):
        m = np.zeros((80, 200), bool)
        m[20:40, k * 35:k * 35 + 15] = True
        masks[k], measured[k] = m, geom(k * .15, .5, .07)
    ids = w.observe_frame(measured, masks)
    count = w.next_id
    assert w.observe_frame({k: geom(k * .15 + .001, .5, .071) for k in masks}, masks) == ids
    with pytest.raises(ObservationError, match='coordinate jump'):
        w.observe_frame({k: geom(k * .15, .594, .225) for k in masks}, masks)
    assert w.next_id == count and len(w.objects) == 4


def test_a_different_appearance_cannot_inherit_a_confirmed_objects_id():
    w = WorldModel()
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    first, second = geom(.3, .5, .06), geom(.3, .5, .06)
    first['colour_signature'] = ([1.] + [0.] * 7) * 3
    second['colour_signature'] = ([0.] * 7 + [1.]) * 3
    old = w.observe_frame({1: first}, {1: mask})[1]
    assert w.observe_frame({1: second}, {1: mask})[1] != old


def test_crowded_label_boxes_do_not_overlap_or_merge_numbers():
    masks = {}
    for k, x, y in [(18, 80, 80), (31, 90, 60), (32, 100, 60), (33, 110, 60)]:
        m = np.zeros((200, 320), bool)
        m[y:y + 25, x:x + 20] = True
        masks[k] = m
    rectangles = []
    for k, _, (u, v) in layout(masks):
        (w, h), _ = cv2.getTextSize(f'#{k}', cv2.FONT_HERSHEY_SIMPLEX, .8, 2)
        rectangles.append((u - w // 2 - 2, v - h - 4, u - w // 2 + w + 2, v + 3))
    for i, a in enumerate(rectangles):
        for b in rectangles[i + 1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def test_bottle_cap_joins_whole_body_but_a_stacked_box_remains_separate():
    body = np.zeros((160, 160), bool)
    body[50:140, 55:110] = True
    cap = np.zeros_like(body)
    cap[35:51, 70:95] = True
    a = geom(.3, .5, .15, fp=(.06, .06), yaw=None)
    b = geom(.3, .5, .17, fp=(.03, .03), support=.15, yaw=None)
    b['visible_bottom_z'] = .15
    assert parent_of_parts({1: body, 2: cap}, {1: a, 2: b}) == {2: 1}
    a['yaw_deg'], b['yaw_deg'] = 0., 0.
    assert parent_of_parts({1: body, 2: cap}, {1: a, 2: b}) == {}


def test_a_lower_milk_label_is_consolidated_with_the_complete_pack():
    body = np.zeros((160, 160), bool)
    body[20:140, 50:110] = True
    label = np.zeros_like(body)
    label[100:135, 55:105] = True
    a, b = geom(.3, .5, .145, fp=(.055, .06)), geom(.3, .5, .064, fp=(.04, .01))
    assert parent_of_parts({1: body, 2: label}, {1: a, 2: b}) == {2: 1}


def test_hidden_apple_bottom_uses_visible_support_plane_not_visible_fruit_surface():
    apple = geom(.3, .5, .070, fp=(.067, .067), support=.040)
    apple['visible_bottom_z'] = .040
    x, y = np.meshgrid(np.linspace(.24, .36, 25), np.linspace(.44, .56, 25))
    p = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    snap_supports([apple], p)
    assert apple['support_confident'] and apple['support_z'] == pytest.approx(0.)
    assert apple['height'] == pytest.approx(.070)
    apple['support_z'] = .040
    snap_supports([apple], np.empty((0, 3)))
    assert not apple['support_confident'] and apple['height'] == pytest.approx(.070)


def test_a_trays_own_interior_floor_is_not_its_support():
    tray = geom(.3, .5, .025, fp=(.36, .26), support=.004)
    tray['visible_bottom_z'] = .004
    x, y = np.meshgrid(np.linspace(.08, .52, 60), np.linspace(.30, .70, 60))
    inside = ((abs(x - .3) < .18) & (abs(y - .5) < .13)).ravel()
    cloud = np.column_stack([x.ravel(), y.ravel(), np.where(inside, .004, 0.)])
    snap_supports([tray], cloud, [inside])
    assert tray['support_confident'] and tray['support_z'] == pytest.approx(0.)
    assert tray['height'] == pytest.approx(.025)


def test_half_visible_cylinder_has_a_body_diameter_not_a_thin_front_slab():
    angle, height = np.meshgrid(np.linspace(-1.2, 1.2, 90), np.linspace(.07, .145, 80))
    points = np.column_stack([.3 + .0275 * np.sin(angle.ravel()),
                              .5 + .0275 * np.cos(angle.ravel()), height.ravel()])
    centre, width = round_section_grasp(points, .168, np.array([.3, .51]))
    assert centre == pytest.approx([.3, .5], abs=.001)
    assert width == pytest.approx(.055, abs=.001)


def test_carton_grasp_is_centered_and_spans_both_faces():
    from tests.unit.test_perception import render_box, K
    from control.robot import table_camera_pose
    from config.agent import Tolerances
    pose = table_camera_pose()
    depth, rgb = render_box(pose, (.3, .45), (.055, .060, .145), 0.)
    g, why = measure(rgb[..., 0] == 200, depth, K, pose, None)
    assert g is not None, why
    assert .050 < g.grasp_width < .070
    assert np.linalg.norm(np.array(g.grasp_centre) - [.3, .45]) <= Tolerances().xy_m


def pixel_scene(tools):
    s = FakeScene(tools)
    s.obs.rgb = np.zeros((120, 240, 3), np.uint8)
    for i, oid in enumerate([12, 13, 15], 1):
        m = np.zeros((120, 240), bool)
        m[40:80, i * 50:i * 50 + 30] = True
        s.obs.marks[i] = m
    return s


def test_invalid_visual_id_is_repaired_on_the_numbered_image_before_motion(monkeypatch):
    monkeypatch.setattr(workers, 'PickPlace', FakePickPlace)
    FakePickPlace.fail, FakePickPlace.runs = False, []
    tools = FakeTools()
    g = FakeGemma(['pick #316', 'pick #12'])
    result = workers.Manipulation(tools, g, pixel_scene(tools), log).run('pick the water bottle')
    assert result.ok and g.calls[1]['images'] == 1  # the whole numbered image only: no crop boards
    assert 'invalid source ID #316' in g.calls[1]['text'] and 'Allowed current object IDs: #12, #13, #15' in g.calls[1]['text']
    assert len(FakePickPlace.runs) == 1


def test_persistent_invalid_id_stops_without_claiming_object_absence(monkeypatch):
    monkeypatch.setattr(workers, 'PickPlace', FakePickPlace)
    FakePickPlace.runs = []
    tools = FakeTools()
    g = FakeGemma(['pick #316', 'pick #16'])
    result = workers.Manipulation(tools, g, pixel_scene(tools), log).run('pick the water bottle')
    assert result.code == 'grounding_invalid' and result.details['recovery'] == 'stop'
    assert 'absence has not been established' in result.details['required_change']
    assert FakePickPlace.runs == []


def test_moving_selected_object_during_grounding_never_reaches_motion(monkeypatch):
    monkeypatch.setattr(workers, 'PickPlace', FakePickPlace)
    FakePickPlace.runs = []
    tools = FakeTools()
    scene = pixel_scene(tools)
    def observe(force=False):
        tools.world.objects[12].geometry['centre'][0] += .025
        return scene.obs
    scene.current = observe
    g = FakeGemma(['pick #12', 'pick #12'])
    result = workers.Manipulation(tools, g, scene, log).run('pick the cube')
    assert result.code == 'grounding_invalid' and 'changed during visual grounding' in result.text
    assert FakePickPlace.runs == []


def test_viewer_uses_matching_capture_and_hides_old_masks_during_motion(tmp_path):
    import time
    from agent.live import LiveChannel
    from agent.tools.perception import Observation
    from agent.viewer import Viewer
    rgb = np.zeros((80, 160, 3), np.uint8)
    rgb[..., 0] = 250
    mask = np.zeros((80, 160), bool)
    mask[30:55, 50:80] = True
    obs = Observation('table_test', 'table', rgb, None, {1: mask}, {1: geom(0., .5, .05)}, object_ids={1: 42})
    live = LiveChannel(tmp_path)
    live.start_run(tmp_path / 'run', 'pick the cube')
    live.perception(obs)
    viewer = Viewer(tmp_path)
    current = np.zeros_like(rgb)
    current[..., 0] = 250  # BGR blue live frame, different from the captured RGB red frame
    viewer.frames['table'] = (current, time.time())
    viewer.poll_files()
    assert viewer.annotated('table')[70, 150].tolist() == [0, 0, 250]
    viewer.rs.moving = True
    assert np.array_equal(viewer.annotated('table'), current)


def placement_tools(tmp_path, moving=False):
    t = Tools(None, None, None, None, log, tmp_path)
    a = geom(.0, .55, .0748, fp=(.05, .05), support=.0044, yaw=None)
    a.update(top_planar=False, tilt_deg=7.14, support_confident=True)
    ref = geom(0., .55, .0254, fp=(.36, .26))
    ref['visible_bottom_z'] = .0044
    oid = t.world.observe(a, None)
    rid = t.world.observe(ref, None)
    plan = PickPlan(oid, [], [], [], [], Held(.0704, .02, width=.05), {}, [0., .55, .0396], .0748, 'in', rid)
    t.bridge = NS(finger=lambda: 0.)
    n = []
    def observe(camera):
        n.append(1)
        g = dict(a, centre=list(a['centre']))
        if moving and len(n) > 1:
            g['centre'][0] += .015
        return NS(frame_id=f'table_{len(n)}', camera='table', measured={1: g, 2: ref}, marks={},
                  object_ids={}, descriptions={})
    t.perceive = observe
    return t, plan, a, ref


def test_corrected_apple_height_and_curved_tilt_do_not_reject_supported_containment(tmp_path):
    t, plan, a, ref = placement_tools(tmp_path)
    assert t.verify_placement(plan)['pass']
    a['centre'][0] = .025  # ordinary tray containment does not require the exact planned centre
    assert t.placement_goal(a, plan, ref)['pass']
    plan.strict_pose = True
    assert not t.placement_goal(a, plan, ref)['pass']  # explicit exact pose keeps tight tolerances


def test_containment_requires_whole_footprint_support_and_stability(tmp_path):
    t, plan, a, ref = placement_tools(tmp_path, moving=True)
    assert not t.verify_placement(plan)['pass']
    a['centre'][0] = .175  # centre inside, but the apple overhangs the tray
    assert not t.placement_goal(a, plan, ref)['pass']


def test_a_missing_or_moving_destination_cannot_pass_using_old_geometry(tmp_path):
    t, plan, a, ref = placement_tools(tmp_path)
    original = t.perceive
    def missing(camera):
        obs = original(camera)
        del obs.measured[2]
        return obs
    t.perceive = missing
    assert not t.verify_placement(plan)['pass']
    t, plan, a, ref = placement_tools(tmp_path)
    ref['centre'][0] = .25
    assert not t.verify_placement(plan)['pass']


def test_verified_delivery_reconciles_a_new_detection_alias_but_not_an_existing_item(tmp_path):
    t, plan, a, ref = placement_tools(tmp_path)
    a['centre'][0] = .125  # still entirely inside the tray, beyond the generic 100 mm tracking gate
    result = t.verify_placement(plan)
    assert result['pass'] and 'reconciled_alias' in result
    assert t.world.objects[result['reconciled_alias']].status == 'merged'
    obs = t.perceive('table')
    t.update_world(obs)
    assert obs.object_ids[1] == plan.object_id
    t, plan, a, ref = placement_tools(tmp_path)
    a['centre'][0] = .125
    t.world.observe(dict(a), None)  # this matching item was already present before delivery
    assert not t.verify_placement(plan)['pass']
    a['centre'][0], a['support_z'] = 0., .04
    assert not t.placement_goal(a, plan, ref)['pass']
    a['support_z'], a['top_planar'], a['tilt_deg'] = .0044, True, 15.
    assert not t.placement_goal(a, plan, ref)['pass']


def test_full_grasp_search_continues_after_first_descent_collision(tmp_path, monkeypatch):
    from control.motion import PlanningError
    t = Tools(None, None, None, None, log, tmp_path)
    t.world.observe(geom(0., .55, .06), None)
    original, yaws = SkillPlanner.grasp, []
    def grasp(self, q, top, yaw, *args, **kwargs):
        yaws.append(yaw)
        if len(yaws) == 1:
            raise PlanningError('descend: collision with a neighbouring carton')
        return original(self, q, top, yaw, *args, **kwargs)
    monkeypatch.setattr(SkillPlanner, 'grasp', grasp)
    plan = t.plan_pick(1, np.array(ARM_HOME))
    assert len(yaws) >= 2 and plan.approach and plan.lift


def test_full_transfer_search_continues_after_first_placement_collision(tmp_path, monkeypatch):
    from control.motion import PlanningError
    t = Tools(None, None, None, None, log, tmp_path)
    t.world.observe(geom(0., .55, .06), None)
    t.world.observe(geom(.2, .55, .03, fp=(.14, .14)), None)
    original, calls = t._plan_place, []
    def place(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PlanningError('lower: collision at the destination')
        return original(*args, **kwargs)
    monkeypatch.setattr(t, '_plan_place', place)
    plan = t.plan_pick_place(1, 'on', 2, 'robot', np.array(ARM_HOME))
    assert len(calls) >= 2 and plan.place and plan.home
