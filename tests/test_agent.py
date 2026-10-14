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

from agent.geometry import measure  # noqa: E402
from agent.graph import EventLog, Supervisor  # noqa: E402
from agent.tools import Tools  # noqa: E402
from agent.vlm_roles import Roles, RecordedVLM, VLMFailure  # noqa: E402
from agent.world import WorldModel  # noqa: E402
from control.motion import PlanningError  # noqa: E402
from control.robot import table_camera_pose  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"

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
def geom(x, y, top, fp=(0.04, 0.04), support=0.0, yaw=0.0):
    h = top - support
    return {"centre": [x, y, support + h / 2], "top_z": top, "height": h, "support_z": support, "yaw_deg": yaw,
            "extents": list(fp), "footprint": [max(fp), min(fp)], "grasp_yaw_deg": yaw, "grasp_width": min(fp),
            "symmetry_deg": 90.0 if abs(fp[0] - fp[1]) < 0.008 else 180.0, "tilt_deg": 0.5}


def test_world_model_keeps_ids_by_geometry_only():
    w = WorldModel()
    a = w.observe(geom(0.6, 0.3, 0.04), {"name": "thing"})
    b = w.observe(geom(0.4, 0.5, 0.04), None)
    assert a != b
    assert w.observe(geom(0.605, 0.302, 0.041), None) == a  # same object, small measurement noise
    assert w.observe(geom(0.6, 0.3, 0.09), None) not in (a, b)  # something taller at the same spot


# ------------------------------------------------------------------------------------- VLM roles
def test_parse_valid_and_retry_then_fail():
    good = json.dumps({"task": "pick_place", "object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}})
    events = []
    roles = Roles(RecordedVLM(["not json", good]), log=lambda k, **f: events.append(f))
    t = roles.parse("put a on b")
    assert t.relation == "on" and t.object.text == "a"
    assert [e["valid"] for e in events] == [False, True]
    with pytest.raises(VLMFailure):
        Roles(RecordedVLM(["{}", "nope"])).parse("x")


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
    for f in AGENT.glob("*.py"):
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
class StubTools:
    """Scripted outcomes for each tool: tests the state machine, budgets and honesty of the verdict."""

    def __init__(self, contact=(True,), placement=(True,), resolve_ambiguous=False, exec_fail=None, task="pick_place"):
        self.contact, self.placement = list(contact), list(placement)
        self.resolve_ambiguous, self.exec_fail, self.task = resolve_ambiguous, exec_fail, task
        self.world = WorldModel()
        self.ex = SimpleNamespace(safety_events=[], spin=lambda s: None)
        self.bridge = SimpleNamespace(q=lambda: np.zeros(6))
        self.held_box = None
        self.executed = []
        from agent.vlm_roles import TaskSpec
        self.roles = SimpleNamespace(parse=lambda s: TaskSpec.model_validate(
            {"task": self.task, "object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}}))

    def park_for_view(self, xy):
        return {"ok": True}

    def assume_placed(self, plan):
        pass

    def perceive(self, camera):
        ids = {1: self.world.observe(geom(0.55, 0.15, 0.04), None), 2: self.world.observe(geom(0.55, 0.45, 0.05), None)}
        return SimpleNamespace(marks={1: None, 2: None}, measured={1: geom(0.55, 0.15, 0.04), 2: geom(0.55, 0.45, 0.05)},
                               object_ids=ids, descriptions={})

    def describe(self, obs):
        pass

    def update_world(self, obs):
        return []

    def resolve(self, obs, text):
        from agent.vlm_roles import Resolution
        if self.resolve_ambiguous:
            return Resolution(id=None, ambiguous=True, candidates=[1, 2])
        return Resolution(id=1 if text == "a" else 2)

    def confirm(self, obs, mark, text):
        from agent.vlm_roles import Confirmation
        return Confirmation(match="yes")

    def relation_holds(self, *a):
        return True

    def plan_pick_place(self, oid, rel, rid, frame, q):
        from control.skills import GripperStep, Held
        return SimpleNamespace(object_id=oid, hover=["hover"], approach=["down", GripperStep(0.5, "close")], lift=["lift"],
                               place=["carry", "lower", "open", "retreat"], held=Held(0.04, 0.02, 0, "obj_1", 0.04, 0.04),
                               grasp={}, target_centre=[0.55, 0.45, 0.07], target_top=0.09, relation=rel, reference_id=rid)

    def execute(self, steps, label):
        self.executed.append(label)
        ok = label != self.exec_fail
        return SimpleNamespace(ok=ok, failed_step="x" if not ok else "", reason="safety layer rejected" if not ok else "")

    def verify_contact(self, target):
        return {"check": "contact", "pass": self.contact.pop(0) if len(self.contact) > 1 else self.contact[0]}

    def verify_placement(self, plan):
        ok = self.placement.pop(0) if len(self.placement) > 1 else self.placement[0]
        return {"check": "placement", "pass": ok, "xy_error_mm": 3.0, "z_error_mm": 1.0, "tilt_deg": 0.4}


def run(tmp_path, tools, instruction="put a on b"):
    log = EventLog(tmp_path / "events.jsonl")
    sup = Supervisor(tools, log, tmp_path)
    r = sup.run(instruction)
    log.close()
    return r, [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def test_supervisor_happy_path_completes_only_after_verification(tmp_path):
    r, ev = run(tmp_path, StubTools())
    assert r["status"] == "COMPLETED"
    states = [e["state"] for e in ev if e["kind"] == "state"]
    assert states == ["UNDERSTANDING", "OBSERVING", "LOCALIZING", "PLANNING_GRASP", "APPROACHING", "GRASPING",
                      "VERIFYING_GRASP", "TRANSPORTING", "PLACING", "VERIFYING_PLACEMENT", "COMPLETED"]
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


def test_supervisor_ambiguity_asks_and_cancels(tmp_path):
    r, ev = run(tmp_path, StubTools(resolve_ambiguous=True))
    assert r["status"] == "CANCELLED"
    asks = [e for e in ev if e["kind"] == "ask_user"]
    assert len(asks) == 1 and asks[0]["answer"] == "cancel" and asks[0]["scripted"]


def test_supervisor_unsupported_task_asks(tmp_path):
    r, _ = run(tmp_path, StubTools(task="unsupported"))
    assert r["status"] == "CANCELLED"


def test_parse_inconsistent_task_gets_one_retry():
    bad = json.dumps({"task": "pick", "object": {"text": "a", "relation": "on", "anchor": "b"}, "relation": "on",
                      "reference": {"text": "b"}})
    good = json.dumps({"task": "pick_place", "object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}})
    vlm = RecordedVLM([bad, good])
    t = Roles(vlm).parse("put a on b")
    assert t.task == "pick_place" and t.object.anchor is None
    assert "pick_place" in vlm.calls[1]  # the retry carries the validation error


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
    from agent.somark import isolate
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


def test_lexical_cross_check_triggers_comparison():
    """The resolver picks a partial match (as in rehearsal cycle 3, i3 run 2): the comparison decides."""
    from agent.tools import Observation
    t, _ = tools_with([])
    t.run_dir = Path(__file__).parent  # images are written next to the test only if it fails before cleanup
    rgb = np.zeros((60, 60, 3), np.uint8)
    m = np.zeros((60, 60), bool)
    m[20:30, 20:30] = True
    obs = Observation("t", "table", rgb, rgb, {1: m, 19: m}, {}, descriptions={
        1: {"name": "Cube", "colour": "brown/orange-brown", "shape": "cube"},
        19: {"name": "Cube", "colour": "Orange", "shape": "Cube"}})
    assert t.lexical_candidates(obs, "the orange cube") == [1, 19]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        t.run_dir = Path(d)
        t.roles = Roles(RecordedVLM(['{"id": 1}', '{"id": 19, "ambiguous": false, "candidates": [19]}']))
        assert t.resolve(obs, "orange cube").id == 19  # visual choice = most specific description: accepted
        t.roles = Roles(RecordedVLM(['{"id": 1}', '{"id": 1, "ambiguous": false, "candidates": [1]}']))
        r = t.resolve(obs, "orange cube")  # visual choice disagrees with the descriptions: ask, never guess
        assert r.id is None and r.ambiguous and r.candidates == [1, 19]


def test_square_footprint_width_is_the_mean_of_both_extents():
    depth, masks = render([((0.0, 0.70), (0.035, 0.035, 0.035))])
    g, _ = measure(masks[0], depth, K, table_camera_pose(), robot=None)
    assert g.symmetry_deg == 90.0 and g.extents[0] == g.extents[1] == g.grasp_width
    assert abs(g.grasp_width - 0.035) < 0.003
