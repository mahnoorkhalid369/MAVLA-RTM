"""The shared object matcher (agent/tools/resolution.py ResolutionTools.resolve): appearance judged by Gemma on the
candidates' crops, side and size judged on measured geometry with its uncertainty, clarification only when the
measurements cannot decide, answers limited to objects of the current observation, and the chosen physical object
kept across camera updates (world ids, not per-frame marks). Selection only: no grasp feasibility here.
Scripted Gemma answers (no sim, no GPU)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic")

from agent.ask import Candidate, Question, ScriptedResponder, parse_answer  # noqa: E402
from agent.tools.resolution import ResolutionTools, rank_by_size, side_split  # noqa: E402
from agent.vlm_roles import AppearanceMatch, Confirmation, ReferenceSpec, Resolution  # noqa: E402
from tests.support.fakes import StubTools, geom  # noqa: E402


def cube(x, y, side_mm, height_mm=None, top_seen=True):
    s, h = side_mm / 1000, (height_mm or side_mm) / 1000
    g = geom(x, y, h, fp=(s, s))
    if not top_seen:
        g["tilt_deg"] = None
    return g


# ------------------------------------------------------------------ size: measured, with uncertainty
def test_41_vs_45_mm_is_decisive_without_asking():
    r = rank_by_size({12: cube(0.3, 0.5, 41), 14: cube(0.4, 0.5, 45)}, "volume", "largest")
    assert r["winner"] == 14 and r["measure"] == "volume" and not r["approximate"]


def test_44_vs_45_mm_is_within_the_measurement_error():
    r = rank_by_size({12: cube(0.3, 0.5, 44), 14: cube(0.4, 0.5, 45)}, "volume", "largest")
    assert r["winner"] is None and r["tied"] == [12, 14]


def test_without_every_height_volume_falls_back_to_footprint_area_flagged_approximate():
    r = rank_by_size({1: cube(0.3, 0.5, 40, height_mm=100, top_seen=False), 2: cube(0.4, 0.5, 50)}, "volume", "largest")
    assert r["measure"] == "footprint" and r["approximate"] and r["winner"] == 2  # the tall one is not judged by height


def test_smallest_and_height():
    gs = {1: cube(0.3, 0.5, 40, height_mm=90), 2: cube(0.4, 0.5, 40, height_mm=50)}
    assert rank_by_size(gs, "height", "largest")["winner"] == 1
    assert rank_by_size(gs, "volume", "smallest")["winner"] == 2


# ------------------------------------------------------------------ side: explicit frame
def test_side_in_the_robot_frame_with_a_borderline_band():
    s = side_split({1: [0.30, 0.5], 2: [-0.30, 0.5], 3: [0.01, 0.6]}, [1, 0, 0], "robot")  # x = the robot's right
    assert (s["inside"], s["outside"], s["borderline"]) == ([1], [2], [3])
    assert side_split({1: [0.3, 0.5]}, [0, 0, 1], "robot")["ignored"] == "vertical direction"


# ------------------------------------------------------------------ the whole matcher
class Matcher(ResolutionTools):
    """ResolutionTools over a scripted scene; Gemma's parts are scripted (split and appearance)."""

    def __init__(self, tmp_path, measured, spec, looks_like, unsure=(), best=None, final=None):
        """final: Gemma's final choice when several remain (2026-10-26); None = it cannot decide either."""
        self.run_dir, self.events, self.shown, self.compared, self.finals = tmp_path, [], [], [], []
        (tmp_path / "images").mkdir(exist_ok=True)

        def each(crops, words):
            self.shown.append(sorted(crops))
            return {k: Confirmation(match="yes" if k in looks_like else "unsure" if k in unsure else "no") for k in crops}

        def joint(img, words, ids):
            self.compared.append(sorted(ids))
            return AppearanceMatch(matches=[k for k in ids if best is None or k in best])
        def resolve(som, descs, text, ids, allowed=None):
            from agent.vlm_roles import Resolution
            self.finals.append(sorted(ids))
            return Resolution(id=final, candidates=[final]) if final is not None else \
                Resolution(id=None, ambiguous=True, candidates=sorted(ids), reason="cannot tell")
        self.roles = SimpleNamespace(parse_reference=lambda text: spec, appearance_each=each, match_appearance=joint,
                                     resolve=resolve)
        rgb = np.zeros((60, 200, 3), np.uint8)
        marks = {}
        for i, k in enumerate(measured):
            m = np.zeros((60, 200), bool)
            m[10:30, 10 + 20 * i:25 + 20 * i] = True
            marks[k] = m
        self.obs = SimpleNamespace(frame_id="table_001", rgb=rgb, som=rgb, marks=marks, measured=measured,
                                   object_ids={k: 100 + k for k in measured}, descriptions={}, shortlists={})

    def pool(self, obs, text):
        return list(obs.marks)

    def describe(self, obs, marks=None):  # Gemma's descriptions: not needed by the scripted final choice
        pass

    def log(self, kind, **kw):
        self.events.append({"kind": kind, **kw})


SCENE = {9: cube(-0.30, 0.55, 98), 12: cube(0.30, 0.50, 41), 14: cube(0.45, 0.30, 45), 5: cube(0.40, 0.60, 60)}
BIGGEST_PINK_RIGHT = ReferenceSpec(appearance="pink block", size={"measure": "volume", "order": "largest"},
                                   side={"direction": [1, 0, 0], "frame": "robot"})


def test_biggest_pink_block_on_your_right(tmp_path):
    """Live 2026-10-21: the biggest pink object overall (#9, left) was chosen and 'on your right' was dropped. Side
    first (measured), then appearance on the remaining crops (Gemma), then size (measured): #14, nobody asked."""
    t = Matcher(tmp_path, SCENE, BIGGEST_PINK_RIGHT, looks_like={9, 12, 14})
    r = t.resolve(t.obs, "the biggest pink block on your right")
    assert r.id == 14 and not r.ambiguous
    assert t.shown == [[5, 12, 14]]  # the left object was never shown to Gemma
    m = next(e for e in t.events if e["kind"] == "match")
    assert [s["stage"] for s in m["stages"]] == ["side", "appearance", "size"]
    assert m["world_id"] == 114 and m["world_ids"][14] == 114  # mark of this frame vs persistent identity


def test_indistinguishable_sizes_go_to_the_user(tmp_path):
    scene = {12: cube(0.30, 0.50, 44), 14: cube(0.45, 0.30, 45)}
    t = Matcher(tmp_path, scene, BIGGEST_PINK_RIGHT, looks_like={12, 14})
    r = t.resolve(t.obs, "the biggest pink block on your right")
    assert r.id is None and r.ambiguous and r.candidates == [12, 14]


def test_an_appearance_gemma_cannot_judge_goes_to_the_user_when_nothing_clearly_fits(tmp_path):
    t = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="pink block"), looks_like=set(), unsure={12, 14})
    r = t.resolve(t.obs, "the pink block")
    assert r.id is None and r.ambiguous and r.candidates == [12, 14]


def test_unsure_objects_do_not_block_a_clear_match(tmp_path):
    """Live 2026-10-21: Gemma 4 E2B called almost every non-matching object "unsure"."""
    t = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="pink block"), looks_like={12}, unsure={5, 9, 14})
    assert t.resolve(t.obs, "the pink block").id == 12


def test_several_clear_matches_are_compared_jointly(tmp_path):
    t = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="maroon block"), looks_like={5, 12}, best={12})
    assert t.resolve(t.obs, "the maroon block").id == 12 and t.compared == [[5, 12]]
    t2 = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="maroon block"), looks_like={5, 12})  # no difference
    r = t2.resolve(t2.obs, "the maroon block")
    assert r.id is None and sorted(r.candidates) == [5, 12] and t2.finals == [[5, 12]]  # Gemma could not decide either
    t3 = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="maroon block"), looks_like={5, 12}, final=5)
    assert t3.resolve(t3.obs, "the maroon block").id == 5  # 2026-10-26: Gemma's final choice, nobody asked


def test_no_joint_comparison_when_size_decides(tmp_path):
    t = Matcher(tmp_path, SCENE, BIGGEST_PINK_RIGHT, looks_like={9, 12, 14})
    t.resolve(t.obs, "the biggest pink block on your right")
    assert t.compared == []


def test_one_clear_appearance_match_needs_no_question(tmp_path):
    t = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="pink block"), looks_like={12})
    assert t.resolve(t.obs, "the pink block").id == 12


def test_no_supported_match_is_reported_as_none(tmp_path):
    t = Matcher(tmp_path, SCENE, ReferenceSpec(appearance="pink block"), looks_like=set())
    r = t.resolve(t.obs, "the pink block")
    assert r.id is None and not r.ambiguous and r.candidates == []


def test_a_rejected_candidate_is_left_out(tmp_path):
    t = Matcher(tmp_path, SCENE, BIGGEST_PINK_RIGHT, looks_like={9, 12, 14})
    assert t.resolve(t.obs, "the biggest pink block on your right", exclude={14}).id == 12


# ------------------------------------------------------------------ clarification: any object of the current view
Q = Question("choose", "Which one?", "object",
             [Candidate(12, "cube", (300, 500), 0.58, (44, 44, 44)), Candidate(14, "cube", (450, 300), 0.54, (45, 45, 45))],
             visible=[5, 9, 12, 14])


def test_the_user_may_name_a_visible_object_that_was_not_proposed():
    a, _ = parse_answer("9", Q)
    assert a.kind == "choice" and a.mark == 9


def test_a_number_that_is_not_in_the_current_view_is_refused():
    a, why = parse_answer("7", Q)
    assert a is None and "not an object in the current view" in why


# ------------------------------------------------------------------ in the pick workflow
def run(tmp_path, tools, task, answers=("cancel",)):
    from agent.eventlog import EventLog
    from agent.pick_place import PickPlaceWorkflow
    log = EventLog(tmp_path / "events.jsonl")
    r = PickPlaceWorkflow(tools, log, tmp_path, ScriptedResponder(list(answers)), final_home=False, mode="pick").run("t", task)
    log.close()
    return r, [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


def pick(text, **extra):
    return {"object": {"text": text, **extra}, "relation": None, "reference": None, "frame": "robot"}


def test_the_users_answer_may_be_any_visible_object(tmp_path):
    t = StubTools(ambiguous={"a": [1, 2]})
    r, ev = run(tmp_path, t, pick("a"), answers=["3"])
    assert r["status"] == "COMPLETED" and t.planned[0][0] == t.ids[3]
    q = next(e for e in ev if e["kind"] == "question")
    assert q["candidates"] == [1, 2] and q["visible"] == [1, 2, 3]


def test_an_answer_outside_the_current_view_is_not_accepted(tmp_path):
    t = StubTools(ambiguous={"a": [1, 2]})
    r, _ = run(tmp_path, t, pick("a"), answers=["9"])
    assert r["status"] == "CANCELLED" and t.planned == []


class RenumberingTools(StubTools):
    """Every observation numbers the objects differently (marks are per-frame labels); Gemma's resolve is
    inconsistent (another object each call); the first grasp misses, so the object is observed and localized again."""

    def __init__(self):
        super().__init__(contact=(False, True))
        self.frames, self.resolves = 0, []

    def perceive(self, camera):
        self.frames += 1
        order = [1, 2, 3] if self.frames % 2 else [3, 1, 2]  # mark -> OBJECTS key
        ids = {m: self.world.observe(self.OBJECTS[k][0], self.OBJECTS[k][1]) for m, k in enumerate(order, 1)}
        self.ids = ids
        return SimpleNamespace(marks={m: None for m in ids}, measured={m: self.OBJECTS[k][0] for m, k in enumerate(order, 1)},
                               object_ids=ids, descriptions={}, frame_id=f"table_{self.frames:03d}")

    def resolve(self, obs, text, exclude=(), choose=True):
        self.resolves.append(obs.frame_id)
        return Resolution(id=len(self.resolves))  # a different mark each time


def test_the_chosen_object_is_kept_across_camera_updates(tmp_path):
    t = RenumberingTools()
    r, ev = run(tmp_path, t, pick("a"))
    first, second = [p[0] for p in t.planned[:2]]
    assert first == second  # the same physical object, although its mark changed (1 -> 2)
    assert t.resolves == ["table_001"]  # matched once; the next frame found it by its world id
    loc = [e for e in ev if e["kind"] == "localized"]
    assert [e["object_mark"] for e in loc[:2]] == [1, 2] and loc[0]["object_id"] == loc[1]["object_id"]


def test_a_task_binding_is_revalidated_in_the_current_frame(tmp_path):
    t = RenumberingTools()
    t.perceive("table")  # the world knows the objects (world ids 1..3)
    wid = t.ids[3]
    r, ev = run(tmp_path, t, pick("a", world_id=wid))
    assert t.planned[0][0] == wid and t.resolves == []  # bound earlier in the task: no new matching
    assert any("bound earlier in this task" in e.get("reason", "") for e in ev if e["kind"] == "resolution")


def test_a_task_binding_not_in_view_is_matched_again(tmp_path):
    t = RenumberingTools()
    r, ev = run(tmp_path, t, pick("a", world_id=999))
    assert t.resolves and any(e["kind"] == "identity" for e in ev)


def test_the_match_line_shows_every_stage(tmp_path):
    from agent.events import format_event
    t = Matcher(tmp_path, SCENE, BIGGEST_PINK_RIGHT, looks_like={9, 12, 14})
    t.resolve(t.obs, "the biggest pink block on your right")
    line = format_event(next(e for e in t.events if e["kind"] == "match"))
    assert line.text == ("match 'the biggest pink block on your right' in table_001: side [1.0, 0.0, 0.0] (robot): "
                         "[12, 14, 5]; looks like 'pink block': [12, 14]; largest volume [12=68921, 14=91125 mm3]: 14; "
                         "-> object 14 (world #114)")


def test_the_reference_split_accepts_words_as_a_list_or_null():
    """Live 2026-10-21: Gemma wrote "appearance": ["pink", "block"] and "other": null."""
    s = ReferenceSpec.model_validate({"appearance": ["pink", "block"], "size": None, "side": None, "other": None})
    assert (s.appearance, s.other) == ("pink block", "")
