"""agent/goals.py derive_conditions: the desired final state follows the steps' order (one gripper)."""

from agent.goals import derive_conditions

PICK = ("pick", {"object": {"text": "a"}})


def test_a_place_after_a_pick_releases_the_held_object_even_when_bound_to_an_id():
    place = ("place", {"object": {"text": "#4"}, "relation": "next_to", "reference": {"text": "c"}})
    conds = derive_conditions([PICK, place])
    assert {"pred": "holding", "args": ["a"]} not in conds
    assert {"pred": "next_to", "args": ["#4", "c"]} in conds and {"pred": "not_holding", "args": []} in conds


def test_one_gripper_a_later_pick_ends_the_earlier_holding():
    conds = derive_conditions([PICK, ("pick", {"object": {"text": "b"}})])
    assert conds == [{"pred": "holding", "args": ["b"]}]


def test_a_pick_alone_still_means_holding():
    assert derive_conditions([PICK]) == [{"pred": "holding", "args": ["a"]}]
