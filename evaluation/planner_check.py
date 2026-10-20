"""Offline check of the goal planner (real Gemma, no sim): per instruction, the request kind (authorization) and
whether the plan is one of the acceptable capability sequences with the expected arguments; validity after the one
retry; latency; and the old parse call's latency for comparison. The same cases are run before and after a prompt
change (the script runs unchanged on an older checkout: capabilities it does not know simply fail there).

    .venv-seg/bin/python -m evaluation.planner_check --out out/perf/planner_check.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIND = [[], ["explore"], ["observe", "explore"], ["observe", "locate"], ["locate"], ["observe", "explore", "locate"]]
LOOK_AT = [["observe", "inspect"], ["inspect"], ["observe", "locate"], ["locate"], ["observe", "locate", "inspect"]]


def look(direction, *then):
    """Acceptable plans that point the wrist camera in `direction` (robot frame), optionally followed by `then`."""
    return [[("look", direction)] + list(then), [("observe", None), ("look", direction)] + list(then)]


# (instruction, acceptable step sequences, expected pick_place (object, relation, reference) words, group)
# a step is a capability name, or (capability, direction) for look (checked with frame "robot")
CASES = [
    ("Put the red cube on the green cube.", [["pick_place"]], [("red cube", "on", "green cube")], "tabletop"),
    ("Pick up the yellow cube and put it on the tan box.", [["pick_place"]], [("yellow cube", "on", "tan box")], "tabletop"),
    ("Put the orange cube on the purple cylinder.", [["pick_place"]], [("orange cube", "on", "purple cylinder")], "tabletop"),
    ("Put the pink cube next to the white tall cylinder.", [["pick_place"]], [("pink cube", "next_to", "white tall cylinder")], "tabletop"),
    ("Take a look at your surroundings.", [["observe", "describe_scene"], ["describe_scene"], ["observe"]], [], "tabletop"),
    ("What objects do you see on the table?", [["observe", "describe_scene"], ["describe_scene"]], [], "tabletop"),
    ("How many cubes are on the table?", [["observe", "answer"], ["answer"]], [], "tabletop"),
    ("Put the red cube on the green cube, then put the orange cube on the purple cylinder.", [["pick_place", "pick_place"]],
     [("red cube", "on", "green cube"), ("orange cube", "on", "purple cylinder")], "tabletop"),
    ("Go back to your home position.", [["return_home"]], [], "tabletop"),
    ("Where is the red cube?", [["observe", "locate"], ["locate"]], [], "tabletop"),
    ("Find the banana.", FIND, [], "tabletop"),
    ("Sing me a song.", [[]], [], "tabletop"),
    # embodied: directions in the robot's own frame; camera orientation vs moving the hand
    ("Look at your left side.", look("left"), [], "embodied"),
    ("Turn your camera to the right.", look("right"), [], "embodied"),
    ("Look down at the floor.", look("down"), [], "embodied"),
    ("Look behind you.", look("backward"), [], "embodied"),
    ("What is on your left?", look("left", ("describe_scene", None)) + look("left", ("answer", None)), [], "embodied"),
    ("Look at the red cube.", LOOK_AT, [], "embodied"),
    ("Move your hand to the left.", [[]], [], "embodied"),  # no capability moves the gripper to a free position
]


def norm(t):
    return " ".join(str(t or "").lower().replace("the ", "").split())


def matches(steps, acceptable) -> bool:
    """steps: [(capability, args)] from Gemma; acceptable: sequences of names or (name, direction)."""
    for seq in acceptable:
        if len(seq) != len(steps):
            continue
        ok = True
        for want, (cap, args) in zip(seq, steps):
            name, direction = want if isinstance(want, tuple) else (want, None)
            frame = "robot" if direction == "down" else args.get("frame", "robot")  # down is the same in every frame
            if cap != name or (direction is not None and (args.get("direction"), frame) != (direction, "robot")):
                ok = False
                break
        if ok:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agent.capabilities import planner_menu
    from agent.vlm_roles import GemmaVLM, Roles, VLMFailure
    events = []
    roles = Roles(GemmaVLM(), lambda k, **f: events.append({"kind": k, **f}))
    rows = []
    for text, acceptable, pps, group in CASES:
        events.clear()
        t0 = time.monotonic()
        try:
            kind = roles.classify_request(text).kind
        except VLMFailure as exc:
            kind = f"invalid: {exc}"
        t1 = time.monotonic()
        try:
            p = roles.plan_goal(text, planner_menu())
            steps = [(s.capability, s.args) for s in p.steps]
            got = [s.capability for s in p.steps]
            got_pp = [(norm(s.args["object"]["text"]), s.args["relation"], norm(s.args["reference"]["text"]))
                      for s in p.steps if s.capability == "pick_place"]
            err = None
        except VLMFailure as exc:
            p, steps, got, got_pp, err = None, None, None, [], str(exc)
        dt = time.monotonic() - t1
        ok = got is not None and matches(steps, acceptable) and got_pp == [(norm(o), r, norm(f)) for o, r, f in pps]
        rows.append({"instruction": text, "group": group, "request_kind": kind, "acceptable": acceptable, "got": got,
                     "args": [a for _, a in steps] if steps else None, "pick_place": got_pp, "ok": ok, "valid": p is not None,
                     "seconds": round(dt, 2), "request_seconds": round(t1 - t0, 2),
                     "attempts": len([e for e in events if e["kind"] == "vlm" and e.get("role") == "plan"]), "error": err,
                     "reason": p.reason if p else None})
        extra = [a for c, a in steps if c == "look"] if steps else ""
        print(("OK  " if ok else "BAD ") + f"{dt:5.1f}s [{kind}] {text!r} -> {got} {got_pp or ''} {extra or ''} "
              f"{(p.reason or '')[:70] if p and not got else (err or '')[:90]}", flush=True)
    t_parse = []
    for text, _, pps, _ in CASES[:4]:
        t0 = time.monotonic()
        roles.parse(text)
        t_parse.append(round(time.monotonic() - t0, 2))
    groups = {g: {"ok": sum(r["ok"] for r in rows if r["group"] == g), "valid": sum(r["valid"] for r in rows if r["group"] == g),
                  "n": sum(r["group"] == g for r in rows)} for g in ("tabletop", "embodied")}
    rep = {"rows": rows, "ok": sum(r["ok"] for r in rows), "valid": sum(r["valid"] for r in rows), "n": len(rows), "groups": groups,
           "plan_seconds_pick_place": [r["seconds"] for r in rows[:4]], "old_parse_seconds_pick_place": t_parse}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("ok", rep["ok"], "/", rep["n"], "| valid JSON plans", rep["valid"], "| by group", groups, "| plan s (4 pick_place)", rep["plan_seconds_pick_place"], "| old parse s", t_parse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
