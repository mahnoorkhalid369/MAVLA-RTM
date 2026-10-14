"""Rehearsal scoring against ground truth (evaluation/demo_rehearsal.py): false done / false failed."""

from evaluation.demo_rehearsal import INSTRUCTIONS, score


def truth(**pos):
    base = {"red_cube": [0.0, 0.58, 0.03], "green_cube": [0.15, 0.5, 0.025], "pink_cube": [0.62, 0.1, 0.02],
            "white_tall_cylinder": [0.25, -0.55, 0.06]}
    base.update(pos)
    return {"sim_time": 1.0, "objects": {n: {"position": p, "quat_wxyz": [1, 0, 0, 0]} for n, p in base.items()}}


def test_on_success_and_false_done():
    before = truth()
    good = truth(red_cube=[0.153, 0.502, 0.05 + 0.03])
    s = score(INSTRUCTIONS[1], before, good, {"status": "COMPLETED"})
    assert s["correct_object"] and s["correct_place"] and s["truth_success"] and not s["false_done"]
    off = truth(red_cube=[0.17, 0.5, 0.08])  # 20 mm off: truth fails
    s = score(INSTRUCTIONS[1], before, off, {"status": "COMPLETED"})
    assert not s["truth_success"] and s["false_done"]
    s = score(INSTRUCTIONS[1], before, good, {"status": "FAILED"})
    assert s["false_failed"]


def test_wrong_object_is_not_success():
    s = score(INSTRUCTIONS[1], truth(), truth(pink_cube=[0.15, 0.5, 0.07]), {"status": "COMPLETED"})
    assert s["moved_most"] == "pink_cube" and not s["correct_object"] and s["false_done"]


def test_next_to_gap():
    s = score(INSTRUCTIONS[4], truth(), truth(pink_cube=[0.28, -0.61, 0.02]), {"status": "COMPLETED", "plan": {"target_centre_mm": [278, -605, 21]}})
    assert s["correct_place"] and s["xy_error_mm"] < 10
