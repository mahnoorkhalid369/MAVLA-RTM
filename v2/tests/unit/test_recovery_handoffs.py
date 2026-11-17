"""Regressions for the real tuna run's stale state and master/worker ping-pong."""

from types import SimpleNamespace as NS

import pytest

from agent import loop, workers
from agent.contracts import Outcome, Report
from agent.pick_place import PickPlace
from tests.unit.test_master_workers import FakeGemma, FakeScene, FakeTools, log, stub_arm
from tests.unit.test_transfer_engineering import skill_tools


def test_failed_pregrasp_a_does_not_block_successful_pick_b(monkeypatch):
    class Skill:
        def __init__(self, tools, logger):
            self.tools = tools
        def run(self, mode, oid, *a, **k):
            if oid == 12:
                return {"status": "FAILED", "object_id": oid, "stage": "INSPECTING_GRASP",
                        "effects": {"grasp_attempted": False, "released": False},
                        "failure": {"cls": "inspection"}, "message": "missing wrist view"}
            self.tools.held_object = oid
            return {"status": "COMPLETED", "object_id": oid}
    monkeypatch.setattr(workers, "PickPlace", Skill)
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12", "pick #13"]), FakeScene(t), log)
    first = m.run("pick a cube")
    assert first.details["stage"] == "INSPECTING_GRASP" and not m.context.pending
    assert m.run("pick the corrected cube").ok
    assert workers.Verification(t, m.context, m.scene, log).run().word == "yes"
    assert set(m.context.outcomes) == {13}


def test_release_survives_failed_placement_without_stale_holding(monkeypatch):
    class Skill:
        def __init__(self, tools, logger):
            self.tools = tools
        def run(self, mode, oid, *a, **k):
            self.tools.held_object = None
            self.on_effect("released", 12)
            return {"status": "FAILED", "object_id": 12, "stage": "VERIFYING_PLACEMENT",
                    "effects": {"grasp_attempted": False, "released": True, "placement_verified": False},
                    "failure": {"cls": "placement"}, "message": "height check failed"}
    monkeypatch.setattr(workers, "PickPlace", Skill)
    t = FakeTools()
    t.held_object = 12
    m = workers.Manipulation(t, FakeGemma(["place on #15"]), FakeScene(t), log)
    m.context.record(Outcome("holding", 12), "Holding old object.")
    report = m.run("place the held cube on its support")
    assert report.details["held_object"] is None and report.details["recovery"] == "stop"
    assert m.context.outcomes[12].kind == "released" and not m.context.pending
    assert "Holding old object" not in m.context.summary()
    assert workers.Verification(t, m.context, m.scene, log).run().code == "placement_unverified"


def test_interrupted_place_transport_requires_reconciliation_even_without_new_pick(monkeypatch):
    class Skill:
        def __init__(self, tools, logger):
            pass
        def run(self, *a, **k):
            return {"status": "FAILED", "object_id": 12, "stage": "CARRYING",
                    "effects": {"grasp_attempted": False, "released": False},
                    "failure": {"cls": "execution"}, "message": "carry interrupted"}
    monkeypatch.setattr(workers, "PickPlace", Skill)
    t = FakeTools()
    t.held_object = 12
    m = workers.Manipulation(t, FakeGemma(["place on #15"]), FakeScene(t), log)
    m.context.record(Outcome("holding", 12), "Old verified holding state.")
    report = m.run("place the held object")
    assert report.details["recovery"] == "stop" and m.context.pending == {12}
    assert 12 not in m.context.outcomes and "Old verified holding state" not in m.context.summary()


def test_exception_after_release_persists_released_state(monkeypatch, tmp_path):
    class Skill:
        def __init__(self, tools, logger):
            self.tools = tools
        def run(self, mode, oid, *a, **k):
            self.tools.held_object = None
            self.on_effect("released", 12)
            raise TimeoutError("camera lost after release")
    monkeypatch.setattr(workers, "PickPlace", Skill)
    t = FakeTools()
    t.held_object = 12
    g = FakeGemma(["manipulation: place the held cube on its support", "place on #15"])
    result = loop.run_instruction("place it on its support", t, g, log, tmp_path)
    assert result["status"] == "FAILED" and result["outcomes"][0]["kind"] == "released"
    assert not result["unresolved_actions"]


@pytest.mark.parametrize("recover_ok", [False, True])
def test_wrist_recovery_stays_local_and_is_bounded(monkeypatch, tmp_path, recover_ok):
    class Skill:
        resumes = 0
        def __init__(self, tools, logger):
            self.tools = tools
        def run(self, mode, oid, *a, **k):
            self.oid = oid
            return self.failed()
        def failed(self):
            return {"status": "FAILED", "object_id": self.oid, "stage": "INSPECTING_GRASP",
                    "effects": {"grasp_attempted": False, "released": False},
                    "failure": {"cls": "inspection"}, "message": "wrist association failed"}
        def resume(self, state):
            assert state == "INSPECTING_GRASP"
            Skill.resumes += 1
            if not recover_ok:
                return self.failed()
            self.tools.held_object = self.oid
            return {"status": "COMPLETED", "object_id": self.oid}
    monkeypatch.setattr(workers, "PickPlace", Skill)
    answers = ["manipulation: pick the cube", "pick #12"]
    if recover_ok:
        answers += ["done: picked up the cube", "yes: the cube is held"]
    g = FakeGemma(answers)
    result = loop.run_instruction("pick the cube", FakeTools(), g, log, tmp_path)
    assert Skill.resumes == 1 and not result["unresolved_actions"]
    assert result["status"] == ("COMPLETED" if recover_ok else "STOPPED")
    assert len(g.calls) == (4 if recover_ok else 2)
    assert result["worker_reports"][1]["message"] == "measured local recovery"


def test_real_skill_reinspection_does_not_repeat_prepare_home_or_hover():
    t, commands, _ = skill_tools(inspection=False)
    skill = PickPlace(t, log)
    first = skill.run("pick_place", 1, "on", 2)
    assert first["failure"]["cls"] == "inspection"
    t.inspect_grasp = lambda oid: {"ok": True}
    result = skill.resume("INSPECTING_GRASP")
    assert result["status"] == "COMPLETED"
    assert sum(label == "pre-grasp inspection pose" for label, _ in commands) == 1
    stages = [s["state"] for s in result["timeline"]]
    assert stages.count("PREPARING") == 1 and stages.count("INSPECTING_GRASP") == 2
    assert result["effects"]["released"] and result["effects"]["placement_verified"]


@pytest.mark.parametrize("recover_held", [False, True])
def test_retention_recovery_uses_actual_sensors_and_never_repeats_pick(monkeypatch, recover_held):
    t, commands, _ = skill_tools(retention=False)
    objects = FakeTools().world.objects
    t.world, t.object_words = NS(objects={1: objects[12], 2: objects[15]}), {1: "cube", 2: "support"}
    t.inspect_destination = lambda reference_id: None  # no identity matcher in these tests
    def held_state():
        if hasattr(t, "unconfirmed_grasp") and recover_held:
            t.held_object = 1
            t.confirm_held = lambda *a: {"check": "retention", "pass": True}
            return {"held": "recovered"}
        return {"held": "none"}
    t.revalidate_held = held_state
    monkeypatch.setattr(workers, "PickPlace", PickPlace)
    m = workers.Manipulation(t, FakeGemma(["pick #1 place on #2", "yes: the cube rests on its support"]),
                             FakeScene(t), log)
    first = m.run("put the cube on its support")
    assert first.code == "retention" and first.details["recovery"] == "reconcile_held"
    result = m.recover(first)
    assert result.ok == recover_held
    assert m.context.outcomes[1].kind == ("placed" if recover_held else "unknown")
    assert not m.context.pending
    assert sum(label == "close" for label, _ in commands) == 1


def test_failed_completion_gate_is_not_sent_to_master_repeatedly(monkeypatch, tmp_path):
    monkeypatch.setattr(workers.Verification, "run", lambda *a: Report("no", "physical state unresolved", code="unresolved_action"))
    g = FakeGemma(["manipulation: pick the cube", "pick #12", "done: finished", "done: finished again"])
    result = loop.run_instruction("pick the cube", FakeTools(), g, log, tmp_path)
    assert result["status"] == "STOPPED" and len(g.calls) == 3
    assert sum(r["message"] == "completion gate" for r in result["worker_reports"]) == 1


def test_repeated_done_does_not_repeat_failed_visual_goal_check(tmp_path):
    g = FakeGemma(["manipulation: pick the cube", "pick #12", "done: finished", "no: wrong object held", "done: finished again"])
    result = loop.run_instruction("pick the cube", FakeTools(), g, log, tmp_path)
    assert result["status"] == "STOPPED" and "already rejected" in result["message"]
    assert sum(r["message"] == "original goal check" for r in result["worker_reports"]) == 1


def test_user_clarification_reaches_final_goal_check(tmp_path):
    g = FakeGemma(["perception: which bottle?", "UNSURE: two bottles", "ask: Which bottle?",
                   "plan: manipulation: pick the white bottle", "pick #12", "done: white bottle held", "yes: white bottle is held"])
    result = loop.run_instruction("pick a bottle", FakeTools(), g, log, tmp_path, ask_user=lambda q: "the white bottle")
    assert result["status"] == "COMPLETED"
    assert "User: the white bottle" in g.calls[-1]["text"]
    assert "the white bottle" in result["effective_request"]
