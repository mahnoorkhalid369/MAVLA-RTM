"""The v2 agent without a GPU or a simulator: Gemma's answers are scripted, the arm is a stub.

The master and the workers talk in plain sentences, objects described in words; internal numbers appear only in the
manipulation worker's grounding. These tests check the connections: the plan's worker names, the workers' status words,
grounding (objects, calibrated places, the held object), measured verification, no numbers outside grounding, and the
loop end to end (live requests of 2026-11-10: "pick the small red block", "can you see the mat on the table",
"place a yellow block on the brown block").
"""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from agent import loop, master, workers
from agent.gemma import VLMFailure
from agent.contracts import Outcome
from config import layout
from tests.support.fakes import geom

MAT = layout.Region("R2", "arrangement mat", "", (0.0, 0.6, 0.0), (0.3, 0.3))


class FakeGemma:
    def __init__(self, answers):
        self.answers, self.calls, self.last_thinking = list(answers), [], ""

    def ask(self, text, images=(), system=None, thinking=False, max_new_tokens=400):
        self.calls.append({"text": text, "images": len(images), "thinking": thinking})
        return self.answers.pop(0)


def obj(x, y, top, desc, fp=(0.04, 0.04)):
    return SimpleNamespace(geometry=geom(x, y, top, fp=fp), description=desc, status="on_surface")


class FakeTools:
    CONTAINER_WALL_M = 0.008

    def __init__(self):
        self.held_object = None
        self.object_words = {12: "red cube", 13: "red cube", 15: "brown cube"}  # as grounding named them
        self.world = SimpleNamespace(objects={
            12: obj(0.64, 0.41, 0.04, {"colour": "red", "shape": "cube"}),
            13: obj(0.35, 0.28, 0.05, {"colour": "red", "shape": "cube"}),
            15: obj(0.34, 0.51, 0.06, {"colour": "brown", "shape": "cube"}, fp=(0.06, 0.06))})

    def is_fresh(self, obs):
        return True

    def wrist_picture(self, label="wrist"):
        self.pictures = getattr(self, "pictures", []) + [label]
        return {"ok": True, "rgb": "wrist view", "frame_id": "wrist_test", "path": None}

    def look_at(self, point, box_name=None, farther=False, label="look"):  # the wrist camera looks down at a point
        return self.wrist_picture(label)

    def return_home(self):
        return {"ok": True}

    def verify_home(self):
        return {"check": "home", "pass": True, "max_error_deg": 0.0}

    def revalidate_held(self):
        return {"held": "verified" if self.held_object is not None else "none", "finger_deg": 20.0}

    def relation_holds(self, target, anchor, relation, frame="robot"):
        return True

    def in_region(self, g, region):
        return {"inside": True}


class FakeScene(workers.Scene):
    def __init__(self, tools):
        super().__init__(tools, None)
        self.obs = SimpleNamespace(marks={1: None, 2: None, 3: None}, rgb="view", id_image="numbered view",
                                   frame_id="table_test", taken_at=time.monotonic(), motion_count=0,
                                   object_ids={i: oid for i, oid in enumerate(tools.world.objects, 1)})

    def current(self, force=False):
        return self.obs


class FakePickPlace:
    fail, runs = False, []

    def __init__(self, tools, log, consult=None):
        self.tools = tools

    def run(self, mode, object_id, relation=None, reference_id=None, final_home=True, observation=None, region=None):
        FakePickPlace.runs.append((mode, object_id, relation, reference_id, region))
        if self.fail:
            return {"status": "FAILED", "object_id": object_id, "message": "grasp: no contact"}
        if mode == "pick":
            self.tools.held_object = object_id
            return {"status": "COMPLETED", "object_id": object_id, "message": "finger contact confirmed"}
        held, self.tools.held_object = self.tools.held_object, None
        return {"status": "COMPLETED", "object_id": object_id or held, "message": "placed 2 mm from the planned spot"}


def log(*a, **k):
    pass


@pytest.fixture(autouse=True)
def stub_arm(monkeypatch):
    FakePickPlace.fail, FakePickPlace.runs = False, []
    monkeypatch.setattr(workers, "PickPlace", FakePickPlace)
    monkeypatch.setattr(loop, "Scene", lambda tools, run_dir: FakeScene(tools))
    monkeypatch.setattr(layout, "REGIONS", {"R2": MAT})


def test_plan_lines_are_read_by_worker_name_only():
    steps = master.parse_steps("1. perception: is there a mat on the table?\n- **manipulation**: put the cube on the mat\n"
                               "verification: is it on the mat?\nreason: ...")
    assert steps == [("perception", "is there a mat on the table?"), ("manipulation", "put the cube on the mat"),
                     ("verification", "is it on the mat?")]
    with pytest.raises(ValueError):
        master.parse_steps("observe: look at the table")


def test_a_decision_in_another_form_is_sent_back_once():
    g = FakeGemma(["I think we are finished.", "done: Yes, I can see the black mat in the middle of the table."])
    assert master.decide(g, "can you see the mat", "scene", "", "", [], "40 steps", log) == \
        ("done", "Yes, I can see the black mat in the middle of the table.", "")
    assert "not one of" in g.calls[1]["text"] and all(c["thinking"] for c in g.calls)
    with pytest.raises(VLMFailure):
        master.decide(FakeGemma(["hmm", "still hmm"]), "x", "scene", "", "", [], "40 steps", log)


def test_perception_answers_in_sentences_with_a_status_word():
    t = FakeTools()
    p = workers.Perception(FakeGemma(["UNSURE: two red cubes fit: a small one (41 mm) and a larger one (51 mm)."]),
                           FakeScene(t), log)
    r = p.run("find the red block")
    assert r.word == "unsure" and p.gemma.calls[0]["images"] == 1 and "#" not in p.gemma.calls[0]["text"]


def test_the_scene_is_in_words_and_only_grounding_sees_numbers_including_the_held_object():
    t = FakeTools()
    t.held_object = 12
    words, numbered = FakeScene(t).text(), FakeScene(t).text(numbers=True)
    assert "#" not in words and "R2" not in words and "mm" not in words and "x +" not in words
    assert "arrangement mat" in words and "In the gripper: red cube (in the gripper)" in words
    # the catalogue (Gemma's names from the session's first look) in words, without numbers; the held one is not listed
    assert "Objects on the table (catalogue names): red cube, brown cube" in words
    assert "#12: red cube" in numbered and "without a number" in numbered and "R2: arrangement mat" in numbered
    assert "#13: red cube" in numbered and "#15: brown cube" in numbered


def test_manipulation_turns_a_sentence_into_an_arm_action():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12", "place on #15"]), FakeScene(t), log)
    assert m.run("pick up the small red block").word == "done" and t.held_object == 12
    r = m.run("put the block you are holding on the brown block")
    assert r.word == "done" and FakePickPlace.runs[-1] == ("place", None, "on", 15, None)
    assert list(m.context.outcomes.values()) == [Outcome("placed", 12, "on", 15)] and "#" not in r.text


def test_naming_the_held_object_in_a_put_down_is_the_put_down():
    t = FakeTools()
    t.held_object = 12
    m = workers.Manipulation(t, FakeGemma(["pick #12 place on #15"]), FakeScene(t), log)
    assert m.run("place the yellow block on the brown block").word == "done"
    assert FakePickPlace.runs[-1] == ("place", None, "on", 15, None)


def test_a_place_on_the_table_is_a_calibrated_region():
    t = FakeTools()
    t.held_object = 12
    m = workers.Manipulation(t, FakeGemma(["place on R2"]), FakeScene(t), log)
    r = m.run("put it on the arrangement mat")
    assert r.word == "done" and FakePickPlace.runs[-1] == ("place", None, "in", None, "R2")
    assert "arrangement mat" in r.text


def test_manipulation_refuses_what_cannot_be_done():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["place on #15", "pick #99", "pick #99", "cannot: nothing like that is on the table",
                                           "wave hello", "still waving"]), FakeScene(t), log)
    assert "empty" in m.run("put it down").text
    assert "invalid source ID" in m.run("pick the blue one").text
    assert m.run("pick the unicorn").text == "nothing like that is on the table"
    assert m.run("wave").word == "failed" and not FakePickPlace.runs


def test_verification_measures_what_the_arm_did():
    t = FakeTools()
    m = workers.Manipulation(t, FakeGemma(["pick #12"]), FakeScene(t), log)
    v = workers.Verification(t, m.context, m.scene, log)
    assert v.run().word == "no"
    m.run("pick it")
    assert str(v.run()) == "yes: holding the red cube (in the gripper)"
    t.held_object = None
    assert v.run().word == "no"


def test_pick_the_small_red_block_ends_on_a_measured_yes_without_numbers_for_the_user(tmp_path):
    g = FakeGemma([
        "perception: find the small red block\nmanipulation: pick up the small red block\nverification: is it held?",
        "OK: the small red block is the 41 mm red cube; the other red cube is 51 mm.",
        "pick #12",
        "yes: the gripper holds a small red cube",
    ])
    t = FakeTools()
    r = loop.run_instruction("pick the small red block", t, g, log, tmp_path, ask_user=lambda q: "")
    assert r["status"] == "COMPLETED" and t.held_object == 12 and r["message"] == "holding the red cube (in the gripper)"
    assert [c["thinking"] for c in g.calls] == [True, False, True, True]  # master, brief perception, grounding, final check
    assert g.calls[-1]["text"].startswith('Original user goal: "pick the small red block"')
    assert "red cube" not in g.calls[-1]["text"]  # the final check gets the user's words only, no earlier names


def test_a_question_ends_with_the_masters_answer(tmp_path):
    g = FakeGemma([
        "perception: is there a mat on the table?",
        "OK: yes, a black arrangement mat with nine white squares lies in the middle of the table.",
        "done: Yes, I can see the black arrangement mat in the middle of the table, with nine squares.",
        "yes: the arrangement mat is visible on the table.",
    ])
    r = loop.run_instruction("can you see the mat on the table", FakeTools(), g, log, tmp_path, ask_user=lambda q: "")
    assert r["status"] == "COMPLETED" and r["message"].startswith("Yes, I can see the black arrangement mat")


def test_questions_to_the_user_carry_no_numbers(tmp_path):
    asked = []
    g = FakeGemma([
        "perception: find the yellow block",
        "UNSURE: two yellow cubes fit: one on the left (#6) and one in front (#14).",
        "ask: Which yellow block do you mean: the one on the left (#6) or the one in front (#14)?",
        "stop: I will wait for you to tell me which one.",
    ])
    loop.run_instruction("pick the yellow block", FakeTools(), g, log, tmp_path, ask_user=lambda q: asked.append(q) or "")
    assert "the one on the left or the one in front" in asked[0] and "#" not in asked[0]


def test_failed_arm_action_without_supported_recovery_stops_before_another_master_call(tmp_path):
    FakePickPlace.fail = True
    g = FakeGemma([
        "manipulation: pick up the red cube #12",
        "pick #12",
        "done: I picked it up.",
        "stop: The grasp failed.",
    ])
    r = loop.run_instruction("pick the red cube", FakeTools(), g, log, tmp_path, ask_user=lambda q: "")
    assert r["status"] == "STOPPED"
    assert len(g.calls) == 2 and len(r["worker_reports"]) == 1
    assert r["worker_reports"][0]["details"]["recovery"] == "stop"


def test_a_step_by_step_master_is_not_stopped_by_its_own_progress(tmp_path):
    g = FakeGemma([
        "perception: which yellow block is larger?",
        "OK: the larger yellow cube is 60 mm, in the middle of the pickup area.",
        "plan: manipulation: pick up the larger yellow cube",
        "plan: manipulation: pick up the larger yellow cube and put it on the brown cube",  # bounded repair
        "pick #12 place on #15",
        "plan: verification: is it on the brown cube?",
        "yes: the yellow cube rests on the brown cube",  # verification: the wrist camera's picture of the placement
        "yes: a yellow cube sits on the brown cube",     # final check against the user's words (same wrist picture)
    ])
    t = FakeTools()
    r = loop.run_instruction("place a yellow block on the brown block", t, g, log, tmp_path, ask_user=lambda q: "")
    assert r["status"] == "COMPLETED" and FakePickPlace.runs[-1] == ("pick_place", 12, "on", 15, None)


def test_a_grounding_answer_in_words_is_sent_back_once():
    t = FakeTools()
    g = FakeGemma(["pick #5 place in red tray", "pick #12 place on #15"])
    m = workers.Manipulation(t, g, FakeScene(t), log)
    assert m.run("put the small red cube in the red tray").word == "done"
    assert "not one of the action lines" in g.calls[1]["text"]


def test_skill_messages_reach_the_master_without_numbers():
    assert workers._words("No gripper orientation can pick up object #12 at x +638, y +413 mm (0.92 m from the base): "
                          "it is outside the arm's range.") == "No gripper orientation can pick up the object: it is outside the arm's range."
    assert workers._words("placement outside tolerance (xy 10 mm, z 5 mm, tilt 5 deg)") == "placement outside tolerance"


def test_the_master_keeps_its_memory_and_sees_the_state_and_budget(tmp_path):
    g = FakeGemma([
        "manipulation: put the small red cube on the larger red cube\nmemory: stacking the reds first",
        "cannot: the larger red cube is not reachable",
        "stop: the larger red cube cannot be reached.\nmemory: gave up on the reds",
    ])
    r = loop.run_instruction("stack the red blocks", FakeTools(), g, log, tmp_path, ask_user=lambda q: "")
    first_decision = g.calls[2]["text"]
    assert "Memory: stacking the reds first" in first_decision and "Budget: 39 steps and 19 decisions" in first_decision
    assert "Current table RGB and measured state:" in first_decision and g.calls[2]["images"] == 1
    assert r["status"] == "STOPPED" and r["memory"] == "gave up on the reds"


def test_anything_measured_at_the_gripper_is_the_held_object():
    from agent.tools.perception import PerceptionTools
    tcp = [0.37, 0.62, 0.20]
    assert PerceptionTools.at_gripper({"centre": [0.369, 0.626, 0.097], "support_z": 0.0 + 0.03}, tcp)  # live 2026-11-10
    assert not PerceptionTools.at_gripper({"centre": [0.369, 0.626, 0.03], "support_z": 0.0}, tcp)  # on the table


def test_collision_is_reported_without_visual_override_or_deleted_records():
    from agent.pick_place import PickPlace
    from control.motion import PlanningError
    calls = []
    def blocked(*a, **k):
        calls.append(1)
        raise PlanningError("carry above target: collision, held with obj_16 (13.2 mm)")
    t = SimpleNamespace(bridge=SimpleNamespace(q=lambda: None), plan_place_held=blocked, held_object=10,
                        world=SimpleNamespace(objects={1: object(), 10: object(), 16: object()}))
    skill = PickPlace(t, log)
    skill.mode, skill.relation, skill.reference_id, skill.region, skill.object_id = "place", "in", 1, None, 10
    skill.retries, skill.failure = {}, None
    assert skill._planning_grasp() == "FAILED" and skill.failure["cls"] == "blocked"
    assert set(t.world.objects) == {1, 10, 16} and len(calls) == 1


def test_grounding_names_the_objects_it_uses_and_the_reports_use_those_words():
    t = FakeTools()
    t.object_words = {}
    g = FakeGemma(["pick #5\nnames: #5 = tall pink block"])
    t.world.objects[5] = obj(0.6, 0.26, 0.12, {}, fp=(0.046, 0.046))
    m = workers.Manipulation(t, g, FakeScene(t), log)
    r = m.run("pick up the tall pink block")
    assert t.object_words == {5: "tall pink block"} and "the tall pink block" in r.text and "#" not in r.text


def test_a_measured_result_that_does_not_match_the_users_words_is_not_done(tmp_path):
    g = FakeGemma([
        "manipulation: put the tall pink block on the red cube\nverification: is it on the red cube?",  # misread "red tray"
        "pick #12 place on #13",
        "yes: the pink block rests on the red cube",                  # verification: what the arm did, by the wrist picture
        "no: the pink block is on a red cube, not in the red tray",   # final check with the user's words only
        "stop: I put it on the wrong place.",
    ])
    r = loop.run_instruction("place the tall pink block on the red tray", FakeTools(), g, log, tmp_path,
                             ask_user=lambda q: "")
    assert r["status"] == "STOPPED"
    assert any("final check against the user's words: no: the pink block is on a red cube" in x for x in r["reports"])


def test_depth_on_the_table_that_no_measured_object_explains_becomes_an_obstacle():
    from agent.tools.perception import unexplained
    K = (600.0, 600.0, 320.0, 240.0)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # camera 1 m above the table, looking straight down
    T[:3, 3] = [0.5, 0.0, 1.0]
    depth = np.full((480, 640), 1.0, np.float32)  # the table top
    depth[220:260, 300:340] = 0.95  # something 50 mm tall right under the camera, which the detector missed
    robot = SimpleNamespace(contains=lambda P: np.zeros(len(P), bool))
    found = unexplained(SimpleNamespace(depth=depth, K=K), T, robot, [])
    assert found and all(abs(g["top_z"] - 0.05) < 0.01 for g in found)
    assert all(abs(g["centre"][0] - 0.5) < 0.06 for g in found)
    measured = [{"centre": [0.5, 0.0, 0.025], "extents": [0.04, 0.04], "footprint": [0.04, 0.04], "top_z": 0.05,
                 "yaw_deg": 0.0}]
    assert unexplained(SimpleNamespace(depth=depth, K=K), T, robot, measured) == []  # a measured object explains it
