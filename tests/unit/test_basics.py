"""The basic operations made dependable (2026-10-26 evening, docs/history/ROBUSTNESS_FIXES_2026-10-26.md section 10): the
gripper state in Gemma's menu, the held object's words across instructions, the clarify answer, the work cell's props."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")


def test_the_menu_notes_what_the_gripper_state_allows_as_a_condition():
    """Live 2026-10-26 ("place red block in red tray"): place was planned with an empty gripper."""
    from agent.capabilities.specs import planner_menu
    line = lambda menu, op: next(x for x in menu.splitlines() if x.startswith(f"  - {op}:"))  # noqa: E731
    empty, holding = planner_menu(held=None), planner_menu(held=4)
    assert "EMPTY now: only after a pick" in line(empty, "place") and "[needs" not in line(empty, "pick")
    assert "HOLDS #4 now: only after placing it" in line(holding, "pick") and "[needs" not in line(holding, "place")
    assert "HOLDS #4" in line(holding, "pick_place")
    assert "[needs" not in planner_menu()  # not stated (a decomposition): no note


def test_a_clarify_question_may_leave_the_request_null():
    """Live 2026-10-26: {"status": "ask", "question": "...", "request": null} failed validation twice."""
    from agent.vlm_roles.schemas import Clarification
    c = Clarification.model_validate({"status": "ask", "question": "Which one?", "request": None, "reason": None})
    assert c.request == "" and c.question == "Which one?"


def test_the_held_object_keeps_the_words_that_named_it(tmp_path):
    """Live 2026-10-26: "pick the pink block" bound it to #4; the next instruction, "place it on the red tray", looked
    "pink block" up again in a frame without the held block and took a red cube 934 mm away."""
    pytest.importorskip("langgraph")
    from tests.support.fakes import GoalTools
    from tests.support.runners import plan, run_supervisor
    t = GoalTools([plan(("observe", {}))])
    t.held_object, t.held_words = 4, {"pink block": 4, "red cube": 9}
    r, ev = run_supervisor(tmp_path, t)
    assert r["bindings"].get("pink block") == 4 and "red cube" not in r["bindings"]
    assert any(e["kind"] == "binding" and e["text"] == "pink block" for e in ev)
    assert t.held_words == {"pink block": 4}  # still held: kept for the next instruction
    t.held_object = None
    (tmp_path / "next").mkdir()
    run_supervisor(tmp_path / "next", t)
    assert t.held_words == {}  # put down: the words are free again


@pytest.mark.parametrize("cell", ["", "0,1.30,1.45,0,0.42,0"])
def test_the_frozen_props_are_obstacles_only_in_the_frozen_cell(cell):
    """The work cell hides the microphone and the camera stand (scenarios/build.py); the agent excluded depth points
    inside the microphone's box, which the work cell's red tray now covers."""
    env = {**os.environ, "TABLE_CAMERA": cell}
    out = subprocess.run([sys.executable, "-c", "from config.perception import FIXTURES; print(len(FIXTURES))"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == ("0" if cell else "4")


def test_workcell_trays_hold_four_blocks_clear_of_the_mat_and_of_each_other():
    import numpy as np
    from control.workspace import pickable
    from scenarios.workcell import MAT, TRAY_SIZE, TRAYS
    (L, W), half_mat = TRAY_SIZE, 1.5 * MAT["cell"] + 0.02
    assert (L - 0.012) * (W - 0.012) >= 4 * 0.06 ** 2 * 2  # four 60 mm blocks with room for the fingers
    for _, _, _, _, xy in TRAYS:
        a = np.arctan2(xy[0], -xy[1])
        corners = [np.asarray(xy) + sx * L / 2 * np.array([np.cos(a), np.sin(a)]) + sy * W / 2 * np.array([-np.sin(a), np.cos(a)])
                   for sx in (-1, 1) for sy in (-1, 1)]
        assert all(pickable(*c) for c in corners)
        assert all(not (abs(c[0] - MAT["centre"][0]) < half_mat and abs(c[1] - MAT["centre"][1]) < half_mat) for c in corners)
