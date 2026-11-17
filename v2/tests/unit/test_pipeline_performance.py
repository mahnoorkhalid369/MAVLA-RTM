"""Optimization contracts: actual evidence, invalidation, independent final checking, and evaluator negatives."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent import loop, workers
from agent.contracts import Outcome, TaskContext
from agent.performance import Performance
from agent.timing import span
from evaluation.benchmark import cases, compare, in_destination, score, timing_summary
from tests.unit.test_master_workers import FakeGemma, FakeTools, FakeScene, FakePickPlace, log, stub_arm


def receipt(t, c):
    return {"outcome": c, "word": "yes", "text": "the red cube rests on the brown cube",
            "view": t.wrist_picture("release"), "scene_revision": t.scene_revision, "epoch": None}


def setup_verifier():
    t, ctx = FakeTools(), TaskContext()
    t.scene_revision = 3
    t.performance = Performance(reuse_placement=True, finish_plan=True)
    c = Outcome("placed", 12, "on", 15)
    ctx.record(c, "placed")
    ctx.placements[12] = receipt(t, c)
    g = FakeGemma(["yes: request satisfied"])
    return t, ctx, g, workers.Verification(t, ctx, FakeScene(t), log, g)


def test_release_receipt_eliminates_repeated_placement_question_but_not_final_check():
    t, ctx, g, v = setup_verifier()
    before = len(t.pictures)
    assert v.run().word == "yes"
    assert not g.calls and len(t.pictures) == before
    assert v.final_check("put red on brown").word == "yes"
    assert len(g.calls) == 1 and g.calls[0]["images"] == 1
    assert len(t.pictures) == before


@pytest.mark.parametrize("change", ["motion", "reset", "destination", "negative"])
def test_receipt_invalidated_by_disturbance_reset_changed_goal_or_negative_verdict(change):
    t, ctx, g, v = setup_verifier()
    if change == "motion":
        t.scene_revision += 1
    elif change == "reset":
        t.bridge = SimpleNamespace(epoch=1)
    elif change == "destination":
        ctx.outcomes[12] = replace(ctx.outcomes[12], reference_id=13)
    else:
        ctx.placements[12]["word"] = "no"
    assert v._receipt(ctx.outcomes[12]) is None
    t.held_object = 13  # cannot acquire a new view while holding
    assert v.run().word == "no"


def test_baseline_does_not_use_release_receipt():
    t, ctx, g, v = setup_verifier()
    t.performance = Performance(reuse_placement=False)
    assert v.run().word == "yes"
    assert len(g.calls) == 1 and len(t.pictures) == 2


class ReceiptedSkill(FakePickPlace):
    def run(self, mode, object_id, relation=None, reference_id=None, **kwargs):
        r = super().run(mode, object_id, relation, reference_id, **kwargs)
        if r["status"] == "COMPLETED" and relation is not None:
            self.placement_evidence = {"view": self.tools.wrist_picture("release"), "text": "red rests on brown",
                                       "word": "yes", "scene_revision": self.tools.scene_revision, "epoch": None}
            self.on_effect("placement_verified", r["object_id"])
        return r


def test_exhausted_successful_transfer_plan_skips_done_call_and_keeps_original_goal_check(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "PickPlace", ReceiptedSkill)
    t = FakeTools()
    t.scene_revision = 1
    t.performance = Performance(reuse_placement=True, finish_plan=True)
    g = FakeGemma(["manipulation: put red on brown", "pick #12 place on #15", "yes: red is on brown"])
    r = loop.run_instruction("put red on brown", t, g, log, tmp_path)
    assert r["status"] == "COMPLETED" and len(g.calls) == 3
    assert r["planned_actions"] == [{"worker": "manipulation", "message": "put red on brown", "status": "done"}]
    assert r["worker_reports"][-1]["message"] == "original goal check"
    assert "put red on brown" in g.calls[-1]["text"]


def test_incomplete_model_plan_still_fails_original_request_check(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "PickPlace", ReceiptedSkill)
    t = FakeTools()
    t.scene_revision = 1
    t.performance = Performance(reuse_placement=True, finish_plan=True)
    g = FakeGemma(["manipulation: put red on brown", "pick #12 place on #15",
                   "no: the other requested object is missing", "stop: incomplete request"])
    r = loop.run_instruction("put red AND the other cube on brown", t, g, log, tmp_path)
    assert r["status"] == "STOPPED" and len(g.calls) == 4
    assert "AND the other cube" in g.calls[2]["text"]


def test_failure_span_is_logged():
    records = []
    with pytest.raises(ValueError), span(lambda k, **v: records.append(v), "capture", "test"):
        raise ValueError("missing frame")
    assert records[0]["ok"] is False and records[0]["seconds"] >= 0


def test_exclusive_timings_do_not_double_count_nested_work_or_overlap():
    events = [{"kind": "timing", "category": c, "start_s": a, "end_s": b, "t": b}
              for c,a,b in [("skill", 0, 10), ("planning", 0, 10), ("motion", 1, 4),
                            ("gemma", 2, 6), ("capture", 6, 9), ("settling", 6, 7)]]
    r = timing_summary(events)
    assert r["exclusive_s"] == {"planning": 2., "motion": 1., "gemma+motion": 2., "gemma": 2., "settling": 1., "capture": 2.}
    assert sum(r["exclusive_s"].values()) == 10


def truth_for(sc):
    return {o.id: {"position": list(o.pos), "quat_wxyz": [np.cos(np.radians(o.yaw_deg)/2), 0., 0., np.sin(np.radians(o.yaw_deg)/2)]}
            for o in sc.objects}


def positioned(region):
    return {"position": list(region.centre), "quat_wxyz": [np.cos(np.radians(region.yaw_deg)/2), 0., 0., np.sin(np.radians(region.yaw_deg)/2)]}


def test_fixed_suite_and_ground_truth_scoring_include_hard_negatives():
    import copy
    import scenarios
    suite = cases()
    assert len(suite) == 63 and len({c["id"] for c in suite}) == 63
    case = suite[0]
    sc = scenarios.generate("snack_kitchen", "easy", 7)
    initial = truth_for(sc)
    final = copy.deepcopy(initial)
    tray = next(r for r in sc.regions if r.id == "serving_tray")
    basket = next(r for r in sc.regions if r.id == "storage_basket")
    final["apple"] = positioned(tray)
    r = score(case, sc, initial, final, {"status": "COMPLETED"})
    assert r["task_success"] and not r["false_accept"]
    assert score(case, sc, initial, final, {"status": "STOPPED"})["false_reject"]
    final["apple"] = positioned(basket)
    r = score(case, sc, initial, final, {"status": "COMPLETED"})
    assert r["wrong_destination"] and r["false_accept"]
    final["apple"] = positioned(tray)
    final["milk"] = positioned(basket)
    assert score(case, sc, initial, final, {"status": "COMPLETED"})["wrong_object"]
    assert not score(case, sc, {}, final, {"status": "COMPLETED"})["scored"]
    apple = next(o for o in sc.objects if o.id == "apple")
    edge = positioned(tray)
    edge["position"][0] += tray.size[0]/2
    assert not in_destination(apple, edge, tray)
    impossible = next(c for c in suite if c["id"] == "easy_s7_impossible")
    assert score(impossible, sc, initial, initial, {"status": "STOPPED"})["task_success"]
    assert score(impossible, sc, initial, initial, {"status": "COMPLETED"})["false_accept"]


def test_comparison_requires_paired_scored_runs():
    assert not compare([{"profile": "headless", "case_id": "a", "score": {"scored": False}}])["headless"]["no_observed_paired_regression"]


def test_system_prefix_cache_clones_and_rejects_nonprefix():
    from agent.prefix_cache import SystemPrefixCache
    class Inputs(dict):
        def to(self, device):
            return self
    proc = SimpleNamespace(apply_chat_template=lambda *a, **k: Inputs(input_ids=np.array([[1,2]])))
    calls = []
    def prefill(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(past_key_values={"mutable": []})
    model = SimpleNamespace(device="cpu", __call__=prefill)
    class Model:
        device = "cpu"
        def __call__(self, **kwargs):
            return prefill(**kwargs)
    cache = SystemPrefixCache()
    args, info = cache.prepare(proc, Model(), {"input_ids": np.array([[1,2,3]])}, "system", True)
    args["past_key_values"]["mutable"].append(99)
    args2, info2 = cache.prepare(proc, Model(), {"input_ids": np.array([[1,2,4]])}, "system", True)
    assert not args2["past_key_values"]["mutable"] and len(calls) == 1 and info2["hit"]
    args3, info3 = cache.prepare(proc, Model(), {"input_ids": np.array([[9,2,4]])}, "changed", True)
    assert not args3 and info3["tokens"] == 0


def test_parallel_home_is_joined_and_failure_cannot_become_completion():
    import threading
    from agent.pick_place import PickPlace
    started = threading.Event()
    judged = threading.Event()
    calls = []
    def home():
        calls.append("HOME")
        started.set()
        assert judged.wait(2)
        return {"ok": False, "reason": "trajectory rejected", "result": object()}
    t = SimpleNamespace(performance=Performance(overlap_home=True), held_object=None, scene_revision=3,
                        return_home=home, last_referent=None,
                        look_at_placement=lambda *a, **k: {"ok": True, "rgb": "image", "frame_id": "f", "path": None})
    skill = PickPlace(t, log)
    skill.final_home, skill.object_id, skill.verifications, skill.effects, skill.retries = True, 12, [], {}, {}
    skill.plan = SimpleNamespace(object_id=12, reference_id=15, relation="on")
    def judge(view):
        assert started.wait(2)
        judged.set()
        return "yes", "red rests on brown"
    assert skill._placement_by_image(judge) == "RETURNING_HOME"
    assert calls == ["HOME"] and skill.effects["placement_verified"]
    assert skill._returning_home() != "VERIFYING_HOME"
    assert calls == ["HOME"]  # consume the joined result, never repeat the motion


def test_camera_motion_preserves_receipts_but_manipulation_and_failures_invalidate():
    from agent.tools.execution import ExecutionTools
    from agent.controls import ExecutionControls
    res = SimpleNamespace(ok=True, failed_step="", reason="", steps=[])
    t = SimpleNamespace(log=log, controls=ExecutionControls(motion="real"), held_object=None,
                        bridge=SimpleNamespace(finger=lambda:0.),
                        motion_count=0, scene_revision=5, ex=SimpleNamespace(spin=lambda *a:None, execute=lambda *a:res))
    ExecutionTools.execute(t, [], "return HOME")
    assert t.scene_revision == 5
    ExecutionTools.execute(t, [], "close")
    assert t.scene_revision == 6
    res.ok = False
    ExecutionTools.execute(t, [], "final check")
    assert t.scene_revision == 7


def test_carry_speedup_retains_waypoint_and_complete_payload_path_check():
    from control.kinematics import Kinematics
    from control.robot import ARM_HOME
    from control.skills import Held, SkillParams, SkillPlanner
    kin = Kinematics()
    held = Held(.04, .02, width=.03, length=.03)
    q, goal = ARM_HOME.copy(), ARM_HOME.copy()
    q[0] -= .3
    goal[0] += .3
    paths = []
    durations = []
    for scale in (1., 1.25):
        planner = SkillPlanner(kin, SkillParams(carry_speed_scale=scale))
        planner._check = lambda step, opening, payload, ignored: paths.append((step, payload))
        step = planner._joint_step(q, goal, "carry above target", held=held)
        durations.append(step.trajectory.duration)
        assert np.any(np.all(np.isclose(step.trajectory.positions, ARM_HOME), axis=1))
        peak = np.max(np.abs(np.diff(step.trajectory.positions, axis=0)/np.diff(step.trajectory.times)[:,None]), axis=0)
        assert np.all(peak <= kin.max_velocity + 1e-6)
        assert planner.p.place_speed == SkillParams().place_speed
    assert durations[1] < durations[0] and len(paths) == 2 and all(p is held for _,p in paths)


def test_final_home_failure_is_not_silently_accepted():
    t, ctx, g, v = setup_verifier()
    ctx.placements.clear()
    t.return_home = lambda: {"ok": False, "reason": "blocked"}
    assert v.run().code == "home"


def test_crash_is_not_a_successful_refusal_of_an_impossible_request():
    import scenarios
    sc = scenarios.generate("snack_kitchen", "easy", 7)
    initial = truth_for(sc)
    case = next(c for c in cases() if c["id"] == "easy_s7_impossible")
    assert not score(case, sc, initial, initial, {"status": "FAILED"})["task_success"]


def test_noninteractive_clarification_stops_without_claiming_completion(tmp_path):
    g = FakeGemma(["perception: is the banana present?", "NOT FOUND: no banana",
                   "ask: Can you supply the banana?"])
    def no_input(question):
        raise EOFError()
    r = loop.run_instruction("put the banana in the tray", FakeTools(), g, log, tmp_path, ask_user=no_input)
    assert r["status"] == "STOPPED" and "clarification" in r["message"]
    assert not r["motion_attempted"]


def test_camera_motion_does_not_reopen_a_rejected_original_goal_check(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "PickPlace", ReceiptedSkill)
    t = FakeTools()
    t.scene_revision, t.motion_count = 1, 0
    t.performance = Performance(reuse_placement=True, finish_plan=True)
    g = FakeGemma(["manipulation: put red on brown", "pick #12 place on #15",
                   "no: the other requested item is missing", "done: everything is complete"])
    # Camera movement alone cannot justify another claim of completion.
    ask = g.ask
    def advancing_camera(*a, **k):
        t.motion_count += 1
        return ask(*a, **k)
    g.ask = advancing_camera
    r = loop.run_instruction("put BOTH cubes on brown", t, g, log, tmp_path)
    assert r["status"] == "STOPPED" and len(g.calls) == 4
    assert "already rejected" in r["message"]


def test_moving_an_excluded_item_and_putting_it_back_is_still_scored_wrong():
    import copy
    import scenarios
    sc = scenarios.generate("snack_kitchen", "easy", 7)
    initial = truth_for(sc)
    final = copy.deepcopy(initial)
    final["apple"] = positioned(next(r for r in sc.regions if r.id == "serving_tray"))
    case = cases()[0]
    r = score(case, sc, initial, final, {"status":"COMPLETED"}, moved_during={"milk"})
    assert r["wrong_object"] and r["false_accept"] and not r["task_success"]


def test_placement_labels_require_unique_binding_and_stationary_capture():
    import copy
    import scenarios
    from evaluation.benchmark import placement_labels
    sc = scenarios.generate("snack_kitchen", "easy", 7)
    initial = truth_for(sc)
    apple = next(o for o in sc.objects if o.id == "apple")
    tray = next(r for r in sc.regions if r.id == "serving_tray")
    final = copy.deepcopy(initial)
    final["apple"] = positioned(tray)
    events = [{"kind":"grounded_action", "monotonic_s":10., "object_id":4, "destination":"#7",
               "object_geometry":{"centre":list(apple.pos)}, "destination_geometry":{"centre":list(tray.centre)}},
              {"kind":"placement_check", "frame_id":"wrist_1", "object_id":4, "reference_id":7,
               "sim_time":5., "relation":"in", "word":"yes"}]
    samples = [{"wall_s":10., "sim_time":1., "objects":initial}]+[
        {"wall_s":20.+i, "sim_time":5.+i*.1, "objects":final} for i in (-2,-1,0,1,2)]
    labels = placement_labels(events, samples, sc)
    assert labels[0]["label"] is True
    assert placement_labels(events, samples[:1], sc)[0]["label"] is None
    samples[0]["objects"] = copy.deepcopy(initial)
    samples[0]["objects"]["milk"]["position"] = list(apple.pos)
    assert placement_labels(events, samples, sc)[0]["label"] is None


def test_reset_while_gemma_judges_cannot_certify_a_pre_reset_placement():
    from agent.pick_place import PickPlace
    t = SimpleNamespace(performance=Performance(), bridge=SimpleNamespace(epoch=0),
                        held_object=None, scene_revision=3,
                        look_at_placement=lambda *a, **k: {"ok":True, "frame_id":"f", "path":None, "epoch":0})
    skill = PickPlace(t, log)
    skill.final_home, skill.object_id, skill.verifications, skill.effects, skill.retries = True, 12, [], {"released":True}, {}
    skill.plan = SimpleNamespace(object_id=12, reference_id=15, relation="on")
    def judge(view):
        t.bridge.epoch = 1
        return "yes", "it was in the tray"
    assert skill._placement_by_image(judge) == "FAILED"
    assert skill.failure["cls"] == "sensor_epoch" and not skill.effects.get("placement_verified")


def test_reset_during_original_goal_check_rejects_the_answer():
    t, ctx, g, v = setup_verifier()
    t.bridge = SimpleNamespace(epoch=0)
    ctx.placements[12]["epoch"] = 0
    ask = g.ask
    def reset(*a, **k):
        answer = ask(*a, **k)
        t.bridge.epoch = 1
        return answer
    g.ask = reset
    result = v.final_check("put red on brown")
    assert result.word == "no" and result.code == "sensor_epoch"
