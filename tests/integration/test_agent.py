"""Offline tests of the agent slice (agent/): geometry, world model, VLM roles, planning tools and the
supervisor's state machine, with synthetic depth, recorded VLM answers and stub tools (no sim, no GPU).

Needs pydantic + langgraph (the agent environment, .venv-seg); skipped where they are missing.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langgraph")

from tests.support.fakes import StubTools, geom  # noqa: E402,F401  (shared fakes; other test modules import them from tests.support)
from tests.support.runners import TASK, run_workflow as run  # noqa: E402,F401

from perception.geometry import measure  # noqa: E402
from agent.eventlog import EventLog  # noqa: E402
from agent.pick_place import PickPlaceWorkflow  # noqa: E402
from agent.tools import Tools  # noqa: E402
from agent.vlm_roles import Roles, RecordedVLM, VLMFailure  # noqa: E402
from perception.world_model import WorldModel  # noqa: E402
from control.motion import PlanningError  # noqa: E402
from control.robot import table_camera_pose  # noqa: E402

AGENT = Path(__file__).resolve().parents[2] / "agent"

# ------------------------------------------------------------------------------- synthetic depth
W, H, F = 1280, 960, 1100.0
K = (F, F, W / 2, H / 2)


def render(boxes):
    """Depth + per-box masks seen by the table camera: axis-aligned boxes (centre xy, size xyz) on the floor."""
    T = table_camera_pose()
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(u - K[2]) / F, (v - K[3]) / F, np.ones_like(u, float)], -1)
    d = d_cam @ T[:3, :3].T
    o = T[:3, 3]
    t_best = -o[2] / d[..., 2]  # floor
    owner = np.full((H, W), -1)
    for i, (cxy, size) in enumerate(boxes):
        lo = np.array([cxy[0] - size[0] / 2, cxy[1] - size[1] / 2, 0.0])
        hi = np.array([cxy[0] + size[0] / 2, cxy[1] + size[1] / 2, size[2]])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tn <= tf) & (tn > 0) & (tn < t_best)
        t_best = np.where(hit, tn, t_best)
        owner[hit] = i
    depth = (t_best[..., None] * d_cam)[..., 2].astype(np.float32)  # metres along the optical axis
    return depth, [owner == i for i in range(len(boxes))]


def test_measure_box_from_mask_and_depth():
    depth, masks = render([((0.60, 0.30), (0.040, 0.060, 0.050))])
    g, why = measure(masks[0], depth, K, table_camera_pose(), robot=None)
    assert g is not None, why
    assert np.hypot(g.centre[0] - 0.60, g.centre[1] - 0.30) < 0.004
    assert abs(g.top_z - 0.050) < 0.002
    assert abs(g.height - 0.050) < 0.004
    assert g.support_z == 0.0
    assert abs(g.grasp_width - 0.040) < 0.004  # across the narrow side
    assert g.tilt_deg < 2.0


def test_measure_stacked_box_reports_support_height():
    depth, masks = render([((0.60, 0.30), (0.08, 0.08, 0.04)), ((0.60, 0.30), (0.04, 0.04, 0.09))])
    g, _ = measure(masks[1], depth, K, table_camera_pose(), robot=None)
    assert g is not None and abs(g.top_z - 0.09) < 0.003


def test_measure_rejects_tiny_mask():
    depth, _ = render([])
    m = np.zeros((H, W), bool)
    m[10:13, 10:13] = True
    g, why = measure(m, depth, K, table_camera_pose(), robot=None)
    assert g is None and "few" in why


# ------------------------------------------------------------------------------------ world model


def test_world_model_keeps_ids_by_geometry_only():
    w = WorldModel()
    a = w.observe(geom(0.6, 0.3, 0.04), {"name": "thing"})
    b = w.observe(geom(0.4, 0.5, 0.04), None)
    assert a != b
    assert w.observe(geom(0.605, 0.302, 0.041), None) == a  # same object, small measurement noise
    assert w.observe(geom(0.6, 0.3, 0.09), None) not in (a, b)  # something taller at the same spot


# ------------------------------------------------------------------------------------- VLM roles
def test_describe_requires_exactly_the_drawn_ids():
    img = np.zeros((32, 32, 3), np.uint8)
    bad = json.dumps({"objects": [{"id": 1, "name": "x"}]})
    good = json.dumps({"objects": [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}]})
    d = Roles(RecordedVLM([bad, good])).describe(img, [1, 2])
    assert sorted(o.id for o in d.objects) == [1, 2]


def test_resolve_rejects_unknown_id_and_null_without_ambiguity():
    img = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(VLMFailure):
        Roles(RecordedVLM(['{"id": 7}', '{"id": null, "ambiguous": false}'])).resolve(img, [{"id": 1}], "x", [1])
    r = Roles(RecordedVLM(['{"id": null, "ambiguous": true, "candidates": [1, 2]}'])).resolve(img, [], "x", [1, 2])
    assert r.ambiguous and r.candidates == [1, 2]


def test_no_object_knowledge_or_ground_truth_in_agent_code():
    """Rules: no colour/shape words or object lists in prompts or code; no simulator ground truth."""
    words = re.compile(r"\b(red|green|blue|yellow|orange|purple|pink|white|black|tan|cube|cylinder|sphere|prism|box_\w+)\b", re.I)
    current_perception = [AGENT.parent / "perception" / f"{m}.py" for m in ("detector", "geometry", "somark", "world_model", "viewpoints")]
    for f in [*AGENT.rglob("*.py"), *current_perception, *(AGENT.parent / "config").glob("*.py")]:  # agent code wherever it lives
        src = f.read_text()
        code = "\n".join(line.split("#")[0] for line in src.splitlines() if not line.strip().startswith(("#", '"""')))
        assert "ground_truth" not in code and ".truth" not in code, f.name
        hits = [m.group(0) for m in words.finditer(code)]
        assert not hits, f"{f.name}: {hits}"


# ------------------------------------------------------------------------------------ planning tools
def tools_with(objects):
    t = Tools(bridge=None, detector=None, roles=None, executor=None, log=lambda *a, **k: None, run_dir=Path("."))
    ids = [t.world.observe(g, None) for g in objects]
    return t, ids


def test_plan_on_places_on_measured_top():
    t, (a, b) = tools_with([geom(0.55, 0.15, 0.04), geom(0.55, 0.45, 0.05, fp=(0.06, 0.06))])
    from control.robot import ARM_HOME
    p = t.plan_pick_place(a, "on", b, "robot", np.array(ARM_HOME, float))
    assert np.allclose(p.target_centre[:2], [0.55, 0.45])
    assert abs(p.target_top - (0.05 + 0.04)) < 1e-6
    assert p.held.width == pytest.approx(0.04)


def test_plan_on_refuses_narrow_support():
    t, (a, b) = tools_with([geom(0.55, 0.15, 0.04), geom(0.55, 0.45, 0.10, fp=(0.02, 0.02))])
    from control.robot import ARM_HOME
    with pytest.raises(PlanningError):
        t.plan_pick_place(a, "on", b, "robot", np.array(ARM_HOME, float))


def test_plan_next_to_finds_a_free_spot_with_gap():
    t, (a, b) = tools_with([geom(0.55, 0.15, 0.04), geom(0.0, 0.65, 0.08, fp=(0.05, 0.05))])
    from control.robot import ARM_HOME
    p = t.plan_pick_place(a, "next_to", b, "robot", np.array(ARM_HOME, float))
    d = np.hypot(p.target_centre[0] - 0.0, p.target_centre[1] - 0.65)
    assert 0.045 + 0.02 - 1e-6 <= d <= 0.045 + 0.02 + 0.10 + 1e-6
    assert p.target_top == pytest.approx(0.04)


def test_relation_check_is_deterministic():
    t, _ = tools_with([])
    anchor, right, far = geom(0.0, 0.6, 0.04), geom(0.08, 0.6, 0.04), geom(0.5, 0.6, 0.04)
    assert t.relation_holds(right, anchor, "right_of")
    assert not t.relation_holds(right, anchor, "left_of")
    assert not t.relation_holds(far, anchor, "next_to")  # beyond the 0.15 m relation gap


# ------------------------------------------------------------------------------------- supervisor






def test_supervisor_happy_path_completes_only_after_verification(tmp_path):
    r, ev = run(tmp_path, StubTools())
    assert r["status"] == "COMPLETED"
    states = [e["state"] for e in ev if e["kind"] == "state"]
    assert states == ["OBSERVING", "LOCALIZING", "PLANNING_GRASP", "APPROACHING", "GRASPING",
                      "VERIFYING_GRASP", "TRANSPORTING", "PLACING", "RETREATING", "VERIFYING_PLACEMENT",
                      "RETURNING_HOME", "VERIFYING_HOME", "COMPLETED"]
    assert "back at HOME" in r["message"]
    assert (tmp_path / "checkpoints.sqlite").exists()


def test_supervisor_placement_failure_is_reported_not_done(tmp_path):
    r, _ = run(tmp_path, StubTools(placement=(False,)))
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "placement"
    assert r["retries"]["placement"] == 1  # budget 1: one re-observation, then an honest failure
    assert "did not report success" in r["message"]


def test_supervisor_grasp_recovery_then_success(tmp_path):
    t = StubTools(contact=(False, True, True))
    r, _ = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and r["retries"]["grasp"] == 1
    assert "grasp recovery" in t.executed


def test_supervisor_execution_failure_has_no_retry(tmp_path):
    r, _ = run(tmp_path, StubTools(exec_fail="lift and carry"))
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "execution"


def test_supervisor_unattended_ambiguity_is_scripted_cancel(tmp_path):
    r, ev = run(tmp_path, StubTools(ambiguous={"a": [1, 3]}))
    assert r["status"] == "CANCELLED"
    asks = [e for e in ev if e["kind"] == "ask_user"]
    assert len(asks) == 1 and asks[0]["answer"] == "cancel" and asks[0]["source"] == "scripted"


def test_park_pose_choice_counts_blocked_camera_rays():
    t, _ = tools_with([])
    from control.robot import ARM_HOME
    executed = []
    t.bridge = SimpleNamespace(q=lambda: np.array(ARM_HOME, float))
    t.execute = lambda steps, label: executed.append(steps[0].label) or SimpleNamespace(ok=True)
    r = t.park_for_view(None)
    assert r["ok"] and r["blocked"] == min(x["blocked"] for x in r["tried"] if x["path_free"])
    r = t.park_for_view((0.6, 0.25))  # a point near the camera's view centre must end unblocked
    assert r["ok"] and r["blocked"] == 0


def test_describe_one_per_isolated_crop():
    from perception.somark import isolate
    rgb = np.full((100, 100, 3), 200, np.uint8)
    m = np.zeros((100, 100), bool)
    m[40:60, 40:60] = True
    c = isolate(rgb, m)
    h, w = c.shape[:2]
    assert tuple(c[h // 2, w // 2]) == (200, 200, 200) and tuple(c[2, 2]) == (60, 60, 60)  # object untouched, rest dimmed
    d = Roles(RecordedVLM(['{"name": "thing", "colour": "c"}'])).describe_one(c)
    assert d.name == "thing"


def test_park_poses_stay_inside_joint_limits():
    from control.kinematics import Kinematics
    k = Kinematics()
    for j1 in Tools.PARK_J1_DEG:
        assert k.lower[0] + np.radians(5) < np.radians(j1) < k.upper[0] - np.radians(5)


def test_resolve_retry_names_the_wrong_field():
    img = np.zeros((32, 32, 3), np.uint8)
    vlm = RecordedVLM(['{"id": [1, 2], "ambiguous": true, "candidates": [1, 2]}', '{"id": null, "ambiguous": true, "candidates": [1, 2]}'])
    r = Roles(vlm).resolve(img, [], "x", [1, 2])
    assert r.ambiguous and "id" in vlm.calls[1].split("not valid")[1]


def test_square_footprint_width_is_the_mean_of_both_extents():
    depth, masks = render([((0.0, 0.70), (0.035, 0.035, 0.035))])
    g, _ = measure(masks[0], depth, K, table_camera_pose(), robot=None)
    assert g.symmetry_deg == 90.0 and g.extents[0] == g.extents[1] == g.grasp_width
    assert abs(g.grasp_width - 0.035) < 0.003



# ----------------------------------------------------------------------------------- return to HOME
def test_home_once_after_verified_placement(tmp_path):
    t = StubTools()
    r, _ = run(tmp_path, t)
    assert r["status"] == "COMPLETED" and t.home_calls == 1 and t.executed.count("return HOME") == 1
    assert [v["check"] for v in r["verifications"]][-1] == "home"


def test_failed_placement_never_moves_home(tmp_path):
    t = StubTools(placement=(False,))
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and t.home_calls == 0 and "return HOME" not in t.executed


def test_home_refused_is_a_clear_failure(tmp_path):
    t = StubTools(home_ok="the robot is still holding an object")
    r, _ = run(tmp_path, t)
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "home" and "still holding" in r["failure"]["reason"]
    assert r["message"].startswith("Placement verified") and "did not report success" in r["message"]


def test_home_not_reached_is_not_completed(tmp_path):
    r, _ = run(tmp_path, StubTools(home_error_deg=2.4))
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "home" and "2.4 deg" in r["failure"]["reason"]


def test_home_execution_failure_has_no_retry(tmp_path):
    r, _ = run(tmp_path, StubTools(exec_fail="return HOME"))
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "execution" and "return HOME" in r["failure"]["reason"]


def _home_tools(q, finger_deg=0.0, holding=None):
    from control.robot import ARM_HOME
    t, _ = tools_with([])
    t.bridge = SimpleNamespace(q=lambda: np.array(q, float), finger=lambda: np.radians(finger_deg), wait_still=lambda: None)
    t.world.holding = holding
    t.executed = []
    t.execute = lambda steps, label: t.executed.append((label, steps)) or SimpleNamespace(ok=True, reason="")
    return t, np.array(ARM_HOME, float)


def test_return_home_plans_a_collision_checked_move_to_the_validated_pose():
    from control.robot import ARM_HOME
    q = np.array(ARM_HOME, float)
    q[0] = np.radians(135)
    t, home = _home_tools(q)
    r = t.return_home()
    assert r["ok"] and t.executed[0][0] == "return HOME"
    traj = t.executed[0][1][0].trajectory
    assert np.allclose(traj.positions[-1], home) and np.allclose(traj.positions[0], q)


def test_return_home_refuses_while_holding_or_closed():
    from control.robot import ARM_HOME
    t, _ = _home_tools(ARM_HOME, holding=3)
    assert not t.return_home()["ok"] and not t.executed
    t, _ = _home_tools(ARM_HOME, finger_deg=17.0)
    r = t.return_home()
    assert not r["ok"] and "gripper not open" in r["reason"] and not t.executed


def test_verify_home_uses_joint_feedback_and_the_executor_tolerance():
    from control.robot import ARM_HOME
    q = np.array(ARM_HOME, float)
    q[2] += np.radians(1.0)
    t, _ = _home_tools(q)
    v = t.verify_home()
    assert v["pass"] and abs(v["max_error_deg"] - 1.0) < 1e-6 and v["tolerance_deg"] == 1.5 and v["tcp_error_mm"] > 0
    q[2] += np.radians(1.0)
    assert not _home_tools(q)[0].verify_home()["pass"]


def test_live_channel_writes_masks_ids_and_current_run(tmp_path):
    import cv2
    from agent.live import LiveChannel
    from agent.tools import Observation
    m1, m2 = np.zeros((40, 60), bool), np.zeros((40, 60), bool)
    m1[5:10, 5:10], m2[20:30, 30:40] = True, True
    obs = Observation("table_001", "table", np.zeros((40, 60, 3), np.uint8), None, {1: m1, 2: m2},
                      {1: geom(0.5, 0.1, 0.04), 2: geom(0.6, 0.2, 0.05)}, object_ids={1: 7, 2: 8},
                      descriptions={1: {"name": "cube"}}, scores={1: 0.91, 2: 0.88})
    live = LiveChannel(tmp_path)
    live.start_run(tmp_path / "run", "put a on b")
    live.perception(obs)
    lab = cv2.imread(str(tmp_path / "labels.png"), cv2.IMREAD_UNCHANGED)
    p = json.loads((tmp_path / "perception.json").read_text())
    assert lab[7, 7] == 1 and lab[25, 35] == 2 and lab[0, 0] == 0
    assert p["marks"]["1"]["object_id"] == 7 and p["marks"]["1"]["score"] == 0.91 and p["camera"] == "table"
    assert json.loads((tmp_path / "current.json").read_text())["instruction"] == "put a on b"



# ------------------------------------------------------------------- latency: fewer VLM calls / moves
def _obs_with(n):
    from agent.tools import Observation
    rgb = np.zeros((80, 200, 3), np.uint8)
    marks, measured = {}, {}
    for k in range(1, n + 1):
        m = np.zeros((80, 200), bool)
        m[10:20, 15 * k:15 * k + 8] = True
        marks[k], measured[k] = m, geom(0.3 + 0.1 * k, 0.3, 0.04)
    return Observation("table_001", "table", rgb, rgb, marks, measured)


def test_shortlist_describes_only_candidates(tmp_path):
    t, _ = tools_with([])
    t.run_dir = tmp_path
    (tmp_path / "images").mkdir()
    obs = _obs_with(6)
    vlm = RecordedVLM(['{"lists": {"A": [2, 5], "B": [4]}}', '{"name": "cube", "colour": "red"}',
                       '{"name": "cube", "colour": "orange"}', '{"name": "box", "colour": "tan"}'])
    t.roles = Roles(vlm)
    t.shortlist(obs, ["red cube", "tan box", None])
    assert len(vlm.calls) == 4  # 1 batched shortlist + 3 crops instead of 6 crops
    assert obs.shortlists == {"red cube": [2, 5], "tan box": [4]} and sorted(obs.descriptions) == [2, 4, 5]
    assert t.pool(obs, "red cube") == [2, 5] and t.pool(obs, "something else") == list(obs.marks)


def test_shortlist_empty_or_failed_falls_back_to_all(tmp_path):
    t, _ = tools_with([])
    t.run_dir = tmp_path
    (tmp_path / "images").mkdir()
    obs = _obs_with(3)
    d = '{"name": "x"}'
    t.roles = Roles(RecordedVLM(['{"lists": {"A": []}}', d, d, d]))
    t.shortlist(obs, ["odd thing"])
    assert sorted(obs.descriptions) == [1, 2, 3]
    obs = _obs_with(3)
    t.roles = Roles(RecordedVLM(['nope', '{"lists": {"A": [9]}}', d, d, d]))  # invalid twice -> fallback
    t.shortlist(obs, ["odd thing"])
    assert sorted(obs.descriptions) == [1, 2, 3]


def test_resolution_only_sees_the_shortlist(tmp_path):
    t, _ = tools_with([])
    t.run_dir = tmp_path
    (tmp_path / "images").mkdir()
    obs = _obs_with(6)
    obs.shortlists = {"red cube": [2, 5]}
    obs.descriptions = {2: {"name": "cube", "colour": "red"}, 5: {"name": "cube", "colour": "dark red"}}
    vlm = RecordedVLM(['{"id": 2}', '{"id": 2, "ambiguous": false, "candidates": [2]}'])
    t.roles = Roles(vlm)
    r = t.resolve(obs, "red cube")
    assert r.id == 2 and "1:" not in vlm.calls[0] and "2:" in vlm.calls[0] and "5:" in vlm.calls[0]
    vlm2 = RecordedVLM(['{"id": 4}', '{"id": null, "ambiguous": true, "candidates": [2]}',
                        '{"id": null, "ambiguous": true, "candidates": [2, 5]}'])
    t.roles = Roles(vlm2)
    assert t.resolve(obs, "red cube").id is None  # an object outside the shortlist is rejected (retry asked)


def test_describe_many_batches_and_retries_invalid_items_once():
    class BatchVLM(RecordedVLM):
        def __init__(self, batch_answers, single):
            super().__init__(single)
            self.batches = list(batch_answers)

        def generate_batch(self, text, images, max_new_tokens):
            return [self.batches.pop(0) for _ in images]
    img = np.zeros((10, 10, 3), np.uint8)
    vlm = BatchVLM(['{"name": "a"}', 'garbage', '{"name": "c"}'], ['{"name": "b"}'])
    d = Roles(vlm).describe_many({1: img, 2: img, 3: img})
    assert {k: v.name for k, v in d.items()} == {1: "a", 2: "b", 3: "c"} and len(vlm.calls) == 1  # one retry for #2
    vlm = BatchVLM(['bad'], ['still bad'])
    with pytest.raises(VLMFailure):
        Roles(vlm).describe_many({1: img})


def test_placement_view_prefers_home_and_home_then_skips_the_move():
    from control.robot import ARM_HOME
    t, _ = _home_tools(np.array(ARM_HOME, float))
    r = t.park_for_view((0.72, -0.08))  # in front of the camera: HOME does not block the view
    assert r["ok"] and r["j1"] == 0.0
    r = t.return_home()
    assert r["ok"] and r["reason"] == "already at HOME" and not [x for x in t.executed if x[0] == "return HOME"]


def test_description_memory_reuses_only_unchanged_objects():
    from agent.tools import DescriptionMemory
    m = DescriptionMemory()
    g, c = geom(0.5, 0.3, 0.04), np.array([200.0, 30.0, 30.0])
    m.store(g, c, {"name": "cube", "colour": "red"})
    assert m.lookup(geom(0.505, 0.302, 0.041), c + 5)["colour"] == "red"      # same object, measurement noise
    assert m.lookup(geom(0.6, 0.3, 0.04), c) is None                           # moved
    assert m.lookup(g, np.array([30.0, 200.0, 30.0])) is None                  # different appearance at the same place
    assert m.lookup(geom(0.5, 0.3, 0.09), c) is None                           # something stacked / taller


def test_describe_uses_session_memory_and_stores_new(tmp_path):
    from agent.tools import DescriptionMemory
    t, _ = tools_with([])
    t.run_dir = tmp_path
    (tmp_path / "images").mkdir()
    t.memory = DescriptionMemory()
    obs = _obs_with(3)
    t.roles = Roles(RecordedVLM(['{"name": "a"}', '{"name": "b"}', '{"name": "c"}']))
    t.describe(obs)
    obs2 = _obs_with(3)
    t2, _ = tools_with([])
    t2.run_dir, t2.memory = tmp_path, t.memory
    vlm = RecordedVLM([])
    t2.roles = Roles(vlm)
    t2.describe(obs2)  # next instruction, same scene: no VLM call
    assert not vlm.calls and [obs2.descriptions[k]["name"] for k in (1, 2, 3)] == ["a", "b", "c"]



def test_intermediate_step_skips_home_and_reuses_a_fresh_observation(tmp_path):
    t = StubTools()
    r, ev = run(tmp_path, t, final_home=False)
    states = [e["state"] for e in ev if e["kind"] == "state"]
    assert r["status"] == "COMPLETED" and "RETURNING_HOME" not in states and t.home_calls == 0
    assert "back at HOME" not in r["message"]


# ------------------------------------------------------------------------------ cross-camera world model
def test_cross_camera_association_and_fusion_rule():
    w = WorldModel()
    a = w.observe(geom(0.6, 0.3, 0.04), {"name": "cube"})
    w.observe(geom(0.4, 0.5, 0.04), None)
    # wrist sees the same cube, full top face, consistent size: refines geometry
    st, oid = w.add_observation(geom(0.602, 0.301, 0.0405), "wrist", "wrist_001", 1, {"name": "red cube"})
    assert (st, oid) == ("matched", a) and w.objects[a].geometry["centre"][0] == 0.602
    assert w.objects[a].observations[-1]["used_for_geometry"] and w.objects[a].descriptions_by_camera["wrist"]["name"] == "red cube"
    # partial view (smaller footprint): evidence only, geometry kept
    st, oid = w.add_observation(geom(0.615, 0.30, 0.04, fp=(0.02, 0.02)), "wrist", "wrist_002", 1)
    assert st == "matched" and w.objects[a].geometry["centre"][0] == 0.602 and not w.objects[a].observations[-1]["used_for_geometry"]
    # something only the wrist camera saw: new object
    st, oid = w.add_observation(geom(0.9, -0.2, 0.05), "wrist", "wrist_002", 2)
    assert st == "new" and w.objects[oid].observations[0]["camera"] == "wrist"
    # two stored objects equally close: ambiguous, not merged
    w2 = WorldModel()
    w2.observe(geom(0.50, 0.30, 0.04), None)
    w2.observe(geom(0.52, 0.30, 0.04, fp=(0.07, 0.07)), None)
    assert w2.add_observation(geom(0.51, 0.30, 0.04), "wrist", "w", 1) == ("ambiguous", None)


def test_partial_wrist_view_of_a_large_object_is_associated_not_new():
    w = WorldModel()
    big = w.observe(geom(0.02, 1.02, 0.20, fp=(0.10, 0.12)), None)
    st, oid = w.add_observation(geom(-0.014, 1.016, 0.20, fp=(0.03, 0.02)), "wrist", "w", 2)  # 34 mm off, inside the box
    assert (st, oid) == ("matched", big) and not w.objects[big].observations[-1]["used_for_geometry"]


# ------------------------------------------------------------------------------- compositional manipulation
def _wf(tmp_path, tools, mode, task, responder=None, final_home=True):
    from agent.ask import ScriptedResponder
    log = EventLog(tmp_path / f"{mode}.jsonl")
    r = PickPlaceWorkflow(tools, log, tmp_path, responder or ScriptedResponder(["cancel"]), final_home=final_home,
                          mode=mode).run("x", task)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / f"{mode}.jsonl").read_text().splitlines()]


def test_pick_only_ends_holding_without_placement_or_home(tmp_path):
    t = StubTools()
    r, ev = _wf(tmp_path, t, "pick", {"object": {"text": "a"}, "relation": None, "reference": None, "frame": "robot"})
    states = [e["state"] for e in ev if e["kind"] == "state"]
    assert r["status"] == "COMPLETED" and "PLACING" not in states and "RETURNING_HOME" not in states and t.home_calls == 0
    assert t.held_object == t.ids[1] and t.world.holding == t.ids[1]
    assert [v["check"] for v in r["verifications"]] == ["contact", "retention_after_lift"]
    assert t.executed == ["approach", "close", "lift"]


def test_pick_only_reports_a_lost_object(tmp_path):
    t = StubTools(contact=(True, False))
    r, _ = _wf(tmp_path, t, "pick", {"object": {"text": "a"}, "relation": None, "reference": None, "frame": "robot"})
    assert r["status"] == "FAILED" and t.held_object is None


def test_place_mode_uses_the_held_object_without_regrasp(tmp_path):
    t = StubTools()
    _wf(tmp_path, t, "pick", {"object": {"text": "a"}, "relation": None, "reference": None, "frame": "robot"})
    t.executed.clear()
    r, ev = _wf(tmp_path, t, "place", {"object": {"text": "#held"}, "relation": "next_to", "reference": {"text": "b"}, "frame": "robot"})
    assert r["status"] == "COMPLETED" and "approach" not in t.executed and "close" not in t.executed
    assert t.executed[0] == "carry" and t.held_object is None and t.home_calls == 1


def test_id_reference_and_remembered_clarification_skip_the_model(tmp_path):
    t = StubTools(ambiguous={"a": [1, 3]})
    from agent.ask import ScriptedResponder
    r, _ = _wf(tmp_path, t, "pick_place", {"object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}, "frame": "robot"},
               responder=ScriptedResponder(["3"]))
    assert r["status"] == "COMPLETED" and t.resolved_refs["a"] == t.ids[3]
    (tmp_path / "2").mkdir()
    r, ev = _wf(tmp_path / "2", t, "pick_place", {"object": {"text": "a"}, "relation": "on", "reference": {"text": "#%d" % t.ids[2]}, "frame": "robot"})
    assert not [e for e in ev if e["kind"] == "question"]  # clarified before: not asked again


def test_deterministic_infeasibility_is_not_retried(tmp_path, monkeypatch):
    from control.motion import PlanningError
    t = StubTools()
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise PlanningError("carry above target: [302, 148, 206] mm is out of reach")
    monkeypatch.setattr(t, "plan_pick_place", boom)
    r, ev = _wf(tmp_path, t, "pick_place", {"object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}, "frame": "robot"})
    assert r["status"] == "FAILED" and r["failure"]["cls"] == "infeasible" and len(calls) == 1
    assert not [e for e in ev if e["kind"] == "recovery"]
