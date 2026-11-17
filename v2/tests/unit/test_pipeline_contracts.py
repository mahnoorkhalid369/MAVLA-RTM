"""Regressions for the captured dispatch/image failures, without model weights or ROS."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agent import loop, master, workers
from agent.contracts import Outcome, Report
from agent.gemma import Gemma
from tests.unit.test_master_workers import FakeGemma, FakePickPlace, FakeScene, FakeTools, log, stub_arm


def test_multiline_master_plan_retains_every_step():
    action, steps, memory = master.parse_decision(
        "plan: perception: locate the bottle\nplan: manipulation: put it in the tray\nmemory: not executed yet")
    assert action == "plan" and [w for w, _ in steps] == ["perception", "manipulation"]
    assert memory == "not executed yet"


@pytest.mark.parametrize("answer", [
    "plan: perception: find the bottle -> ok: it is safe\nplan: manipulation: pick it",
    "plan: manipulation: pick the bottle\nmemory: done<channel|>plan: perception: inspect it",
    "done: finished\nplan: manipulation: put it down",
    "plan: manipulation: put it down\nPlan still to do:\npick up juice",
])
def test_master_rejects_the_entire_malformed_response(answer):
    with pytest.raises(ValueError):
        master.parse_decision(answer)


def test_echoed_plan_is_repaired_before_a_worker_can_be_dispatched():
    g = FakeGemma(["plan: perception: find water -> ok: found\nplan: manipulation: pick water",
                   "plan: manipulation: pick water\nmemory: pickup pending"])
    action, steps, _ = master.decide(g, "pick water", "scene", "", "", [], "budget", log)
    assert action == "plan" and steps == [("manipulation", "pick water")]
    assert len(g.calls) == 2


def test_split_pick_place_is_one_complete_skill_not_pick_only():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12\nplace on #15\nnames: #12 = red cube; #15 = brown cube"]),
                             FakeScene(t), log)
    assert m.run("put the red cube on the brown cube").word == "done"
    assert FakePickPlace.runs == [("pick_place", 12, "on", 15, None)]
    assert m.context.outcomes[12] == Outcome("placed", 12, "on", 15)


def test_extra_grounding_action_cannot_be_silently_ignored():
    t = FakeTools()
    answer = "pick #12\nplace on #15\npick #13"
    m = workers.Manipulation(t, FakeGemma([answer, answer]), FakeScene(t), log)
    r = m.run("move the cube")
    assert r.word == "failed" and not FakePickPlace.runs


def test_remembered_but_unmarked_destination_is_repaired_against_the_image():
    t = FakeTools()
    t.held_object = 12
    scene = FakeScene(t)
    del scene.obs.object_ids[3]  # #15 remains in world memory but is absent from the supplied image
    g = FakeGemma(["place on #15", "place on #13"])
    m = workers.Manipulation(t, g, scene, log)
    assert "#15" not in scene.text(numbers=True)
    assert m.run("place it on the other cube").word == "done"
    assert "not marked in the supplied image" in g.calls[1]["text"]
    assert FakePickPlace.runs == [("place", None, "on", 13, None)]


def test_expired_grounding_observation_is_refreshed_before_motion(monkeypatch):
    t = FakeTools()
    fresh = iter([False, True, True])
    t.is_fresh = lambda obs: next(fresh)
    scene = FakeScene(t)
    refreshes = []
    monkeypatch.setattr(scene, "current", lambda force=False: refreshes.append(force) or scene.obs)
    g = FakeGemma(["pick #12", "pick #12"])
    assert workers.Manipulation(t, g, scene, log).run("pick the cube").word == "done"
    assert refreshes == [True, True, True] and len(FakePickPlace.runs) == 1  # includes the post-inference geometry refresh
    assert "image expired" in g.calls[1]["text"]


def test_report_prose_cannot_override_its_unsure_status(monkeypatch, tmp_path):
    monkeypatch.setattr(workers.Perception, "run", lambda self, message: Report("unsure", "old report -> done: found"))
    g = FakeGemma(["perception: find a cube\nmanipulation: pick the cube", "stop: Please identify the cube."])
    r = loop.run_instruction("pick a cube", FakeTools(), g, log, tmp_path)
    assert r["status"] == "STOPPED" and not FakePickPlace.runs


def test_unknown_worker_output_is_not_success():
    assert workers._status("I guess so", {"OK": "ok"}).word == "unsure"


def test_home_is_a_verifiable_outcome(tmp_path):
    t = FakeTools()
    t.return_home = lambda: {"ok": True}
    t.verify_home = lambda: {"pass": True}
    g = FakeGemma(["manipulation: go home", "home", "done: Returned home.", "yes: Joint feedback confirms HOME."])
    r = loop.run_instruction("go home", t, g, log, tmp_path)
    assert r["status"] == "COMPLETED" and r["outcomes"][0]["kind"] == "home"
    assert "Current joint feedback: HOME reached." in g.calls[-1]["text"]


def test_home_cannot_bypass_checking_an_unfulfilled_placement_request(tmp_path):
    t = FakeTools()
    t.return_home = lambda: {"ok": True}
    t.verify_home = lambda: {"pass": True}
    g = FakeGemma(["manipulation: go home", "home", "done: Finished.",
                   "no: The bottle is still outside the tray.", "stop: The bottle was not placed."])
    r = loop.run_instruction("put the bottle in the tray", t, g, log, tmp_path)
    assert r["status"] == "STOPPED"
    assert 'Original user goal: "put the bottle in the tray"' in g.calls[3]["text"]


def test_omitting_manipulation_cannot_bypass_the_original_goal_check(tmp_path):
    g = FakeGemma(["perception: find the bottle", "OK: The bottle stands on the table.", "done: Finished.",
                   "no: The bottle has not been picked up.", "stop: No pickup executed."])
    r = loop.run_instruction("pick up the bottle", FakeTools(), g, log, tmp_path)
    assert r["status"] == "STOPPED" and not FakePickPlace.runs


def test_multiple_worker_status_lines_are_not_partially_accepted():
    r = workers._status("yes: in place\nno: actually outside the tray", {"YES": "yes", "NO": "no"})
    assert r.word == "unsure" and r.code == "invalid_response"


def test_final_model_check_uses_wrist_pictures_only():
    """The final check never uses the table camera: the gripper's view while holding, and one picture looking down at
    each placed object (moving the wrist camera there)."""
    t = FakeTools()
    t.held_object = 12
    entries = []
    g = FakeGemma(["yes: a cube is held", "yes: the cube is on the brown cube"])
    log_ = lambda kind, **data: entries.append((kind, data))
    v = workers.Verification(t, workers.TaskContext(), FakeScene(t), log_, g)
    assert v.final_check("pick up the cube").word == "yes" and g.calls[0]["images"] == 1
    assert t.pictures == ["final gripper"]
    t.held_object = None
    v.context.record(Outcome("placed", 12, "on", 15), "put the red cube on the brown cube")
    assert v.final_check("place the cube on the brown cube").word == "yes" and g.calls[1]["images"] == 1
    assert t.pictures[-1] == "final check"
    assert all(i["camera"] == "wrist" for kind, data in entries if kind == "model_input" for i in data["images"])


def test_failed_action_remains_unresolved_until_same_object_succeeds():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12", "pick #12"]), FakeScene(t), log)
    FakePickPlace.fail = True
    assert m.run("pick the cube").word == "failed"
    assert workers.Verification(t, m.context, m.scene, log).run().code == "unresolved_action"
    FakePickPlace.fail = False
    assert m.run("pick the cube").word == "done" and not m.context.pending
    assert workers.Verification(t, m.context, m.scene, log).run().word == "yes"


def test_later_placement_supersedes_the_same_objects_old_effect():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12 place on #13", "pick #12 place on #15"]), FakeScene(t), log)
    m.run("move it to the first cube")
    m.run("then move it to the second cube")
    g = FakeGemma(["yes: the red cube rests on the brown cube"])
    assert workers.Verification(t, m.context, m.scene, log, g).run().word == "yes"
    assert len(g.calls) == 1 and "on brown cube" in g.calls[0]["text"]  # only the latest placement is looked at
    assert len(m.context.history) == 2 and len(m.context.outcomes) == 1


def test_verification_reports_what_the_wrist_camera_saw():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12 place on #15"]), FakeScene(t), log)
    m.run("put the cube on its target")
    r = workers.Verification(t, m.context, m.scene, log, FakeGemma(["no: the red cube lies on the table beside it"])).run()
    assert r.word == "no" and "wrist camera: the red cube lies on the table beside it" in r.text


def test_false_model_memory_does_not_replace_execution_state(tmp_path):
    g = FakeGemma(["manipulation: pick the red cube\nmemory: already placed", "pick #12", "stop: Still holding it."])
    r = loop.run_instruction("pick a cube", FakeTools(), g, log, tmp_path)
    assert "In the gripper: red cube" in g.calls[2]["text"]
    assert "Execution record (kept by the system" in g.calls[2]["text"] and "picked up" in g.calls[2]["text"]
    assert r["outcomes"][0]["kind"] == "holding"


def test_repeated_unchanged_worker_failure_stops_instead_of_resetting_forever(tmp_path):
    FakePickPlace.fail = True
    g = FakeGemma(["manipulation: pick the red cube", "pick #12",
                   "plan: manipulation: pick the red cube", "pick #12"])
    r = loop.run_instruction("pick the red cube", FakeTools(), g, log, tmp_path)
    assert r["status"] == "STOPPED" and len(FakePickPlace.runs) == 1
    assert len(g.calls) == 2
    assert r["unresolved_actions"] == [12] and r["motion_attempted"]


def test_camera_exception_still_persists_a_result(monkeypatch, tmp_path):
    def broken(self, force=False):
        raise TimeoutError("no camera frame")
    monkeypatch.setattr(FakeScene, "current", broken)
    r = loop.run_instruction("look", FakeTools(), FakeGemma([]), log, tmp_path)
    assert r["status"] == "FAILED"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "FAILED"


def test_parking_failure_records_contacts_and_joint_zone_reason(monkeypatch):
    from agent.tools import manipulation
    from control.collision import Contact, Path_Result

    checker = SimpleNamespace(
        spheres=lambda q: (np.zeros((1, 3)), np.ones(1), None, None),
        check_path=lambda qs: Path_Result(False, 7, [Contact("j1_link", "unseen_4", 0.003)]))
    monkeypatch.setattr(manipulation, "CollisionChecker", lambda obstacles: checker)
    monkeypatch.setattr(manipulation, "joint_move", lambda *a: SimpleNamespace(positions=np.zeros((2, 6))))
    monkeypatch.setattr(manipulation, "forbidden_on_path", lambda positions: "joint combination forbidden")
    t = manipulation.ManipulationTools()
    t.bridge, t.kin, t.held = SimpleNamespace(q=lambda: np.zeros(6)), None, None
    t.PARK_J1_DEG, t.obstacles, t.workspace_points = (0.,), lambda: [], lambda: [(0., 0.5)]
    t._rays_blocked, t.log = lambda *a: 0, log
    r = t.park_for_view()
    assert not r["ok"]
    candidate = r["tried"][0]
    assert candidate["collision_sample"] == 7 and "unseen_4" in candidate["contacts"][0]
    assert candidate["forbidden_reason"] == "joint combination forbidden"


def test_model_adapter_saves_the_actual_resized_rgb_input(tmp_path):
    class Inputs(dict):
        def to(self, device):
            return self
    captured, records = [], []
    def template(messages, **kwargs):
        captured.append(messages)
        return Inputs(input_ids=np.array([[1, 2]]))
    g = Gemma.__new__(Gemma)
    g.vlm = SimpleNamespace(
        processor=SimpleNamespace(apply_chat_template=template, decode=lambda *a, **k: "OK",
                                  tokenizer=SimpleNamespace(encode=lambda *a, **k: []),
                                  parse_response=lambda *a, **k: {"content": "OK"}),
        model=SimpleNamespace(device="cpu", generate=lambda **k: np.array([[1, 2, 3]])),
        config=SimpleNamespace(thinking_tokens=10, image_tokens=1120), _image=lambda image: image.resize((64, 36)))
    g.torch = SimpleNamespace(inference_mode=nullcontext)
    g.log, g.run_dir = lambda kind, **data: records.append((kind, data)), tmp_path
    assert g.ask("look", images=[np.zeros((72, 128, 3), np.uint8)]) == "OK"
    assert captured[0][-1]["content"][0]["image"].size == (64, 36)
    assert records[0][1]["images"][0]["shape"] == [36, 64, 3]
    assert (tmp_path / "images/model_inputs/001_0.png").is_file()
