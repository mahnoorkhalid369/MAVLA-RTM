"""Desired final-state conditions and their evaluation over MEASURED state (verification / critic worker).

A goal is a list of conditions {"pred": name, "args": [refs...], ...}. Refs are bound to world object ids by
the supervisor (from the workers' results: what was located, grasped, used as destination). Each condition
evaluates to satisfied True / False / None (not decidable from the evidence) with a confidence:
  measured  - from depth geometry, joint / finger feedback (independent of the language model)
  model     - from Gemma's output (e.g. an answer): reported, never called verified
  none      - no evidence

Predicates (reusable building blocks, not a list of supported sentences):
  holding(x)              the gripper holds x: world says so AND the fingers are stalled on it now
  not_holding()           nothing in the gripper (finger feedback)
  on(x, y)                x rests on y's top (measured after the action)
  next_to / left_of / right_of / in_front_of / behind (x, y)   deterministic relation check (agent.tools)
  located(x)              x is bound to a measured object
  observed(x, camera)     x has a measurement from that camera (e.g. wrist) in this task
  ordered(xs, by, axis)   the objects' measured `by` (size | height | width) increases along `axis` (x | y)
  answered(question)      an answer was produced (confidence: model)
  reported()              a scene report was produced
  at_home()               HOME joint check passed
Anything else -> satisfied None, "no evaluator for <pred>" (reported as unverifiable, never as done).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PREDICATES = ("holding", "not_holding", "on", "next_to", "left_of", "right_of", "in_front_of", "behind", "located",
              "observed", "ordered", "answered", "reported", "at_home")
RELATIONS = ("next_to", "left_of", "right_of", "in_front_of", "behind")


@dataclass
class ConditionResult:
    condition: dict
    satisfied: bool | None
    confidence: str  # measured | model | none
    evidence: str = ""
    bound: dict = field(default_factory=dict)


def derive_conditions(steps) -> list[dict]:
    """Default desired final state from a plan's steps (when the planner gives none). steps: [(capability, args)]."""
    out = []
    for cap, a in steps:
        obj = ((a.get("object") or {}).get("text") if isinstance(a.get("object"), dict) else a.get("target")) or None
        ref = (a.get("reference") or {}).get("text") if isinstance(a.get("reference"), dict) else None
        if cap == "pick":
            out.append({"pred": "holding", "args": [obj]})
        elif cap == "pick_place":
            out.append({"pred": a["relation"], "args": [obj, ref]})
        elif cap == "place":
            out.append({"pred": a["relation"], "args": [obj or "#held", ref]})
            out.append({"pred": "not_holding", "args": []})
        elif cap == "locate":
            out.append({"pred": "located", "args": [a.get("target")]})
        elif cap == "inspect":
            out.append({"pred": "observed", "args": [a.get("target")], "camera": "wrist"})
        elif cap == "answer":
            out.append({"pred": "answered", "args": [a.get("question")]})
        elif cap == "describe_scene":
            out.append({"pred": "reported", "args": []})
        elif cap == "return_home":
            out.append({"pred": "at_home", "args": []})
    # a later pick of the same object supersedes nothing; a later place supersedes an earlier holding(x)
    placed = {c["args"][0] for c in out if c["pred"] in RELATIONS + ("on",)}
    return [c for c in out if not (c["pred"] == "holding" and c["args"][0] in placed)]


class GoalEvaluator:
    def __init__(self, tools, state) -> None:
        self.t, self.s = tools, state  # agent.tools.Tools, agent.world_state.TaskState

    def _geom(self, oid):
        o = self.t.world.objects.get(oid)
        return None if o is None else o.geometry

    @staticmethod
    def _id(ref, bindings: dict):
        """A condition argument -> world id: an int, a "#<id>" reference, or a text bound by a worker's result."""
        if isinstance(ref, int) or ref is None:
            return ref
        if isinstance(ref, str) and ref.startswith("#") and ref[1:].isdigit():
            return int(ref[1:])
        return bindings.get(ref)

    def evaluate(self, cond: dict, bindings: dict) -> ConditionResult:
        p, args = cond.get("pred"), list(cond.get("args") or [])
        ids = [self._id(a, bindings) for a in args]
        bound = {str(a): i for a, i in zip(args, ids)}
        fn = getattr(self, f"_{p}", None)
        if fn is None or p not in PREDICATES:
            return ConditionResult(cond, None, "none", f"no evaluator for {p!r}", bound)
        try:
            ok, conf, why = fn(ids, cond)
        except Exception as exc:  # an evaluation error is "unknown", never "true"
            ok, conf, why = None, "none", f"evaluation error: {exc!r}"
        return ConditionResult(cond, ok, conf, why, bound)

    # ------------------------------------------------------------------------------------ predicates
    def _holding(self, ids, c):
        if ids[0] is None:
            return None, "none", "object not identified"
        if self.t.world.holding != ids[0]:
            return False, "measured", f"the world model says the gripper holds {self.t.world.holding}"
        v = self.t.verify_contact(self.t.held_close_target)
        return bool(v["pass"]), "measured", f"fingers {v.get('stopped_short_deg')} deg short of the close target (need >= 1.5)"

    def _not_holding(self, ids, c):
        f = float(np.degrees(self.t.bridge.finger()))
        return (self.t.world.holding is None and f <= self.t.OPEN_FINGER_DEG), "measured", f"finger {f:.1f} deg"

    def _on(self, ids, c):
        a, b = (self._geom(i) for i in ids[:2])
        if a is None or b is None:
            return None, "none", "objects not identified"
        dz = abs(a["support_z"] - b["top_z"])
        dxy = float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))
        ok = dz <= 0.010 and dxy <= max(b["footprint"]) / 2 + 0.010
        return ok, "measured", f"support {dz * 1000:.0f} mm from the top, centre {dxy * 1000:.0f} mm from its centre"

    def _relation(self, ids, c):
        a, b = (self._geom(i) for i in ids[:2])
        if a is None or b is None:
            return None, "none", "objects not identified"
        ok = bool(self.t.relation_holds(a, b, c["pred"], c.get("frame", "robot")) and a["support_z"] <= 0.010)
        d = float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))
        return ok, "measured", f"centres {d * 1000:.0f} mm apart, on the floor: {a['support_z'] <= 0.010}"

    _next_to = _left_of = _right_of = _in_front_of = _behind = _relation

    def _located(self, ids, c):
        return (ids[0] is not None and ids[0] in self.t.world.objects), "measured", "bound to a measured object" if ids[0] else "not found"

    def _observed(self, ids, c):
        o = self.t.world.objects.get(ids[0]) if ids[0] is not None else None
        cam = c.get("camera", "wrist")
        n = 0 if o is None else sum(1 for r in o.observations if r["camera"] == cam)
        return (n > 0), "measured", f"{n} {cam} observation(s)"

    def _ordered(self, ids, c):
        gs = [self._geom(i) for i in ids]
        if any(g is None for g in gs) or len(gs) < 2:
            return None, "none", "objects not identified"
        by, axis = c.get("by", "size"), {"x": 0, "y": 1}[c.get("axis", "x")]
        key = {"size": lambda g: g["footprint"][0] * g["footprint"][1] * g["height"], "height": lambda g: g["height"],
               "width": lambda g: g["footprint"][1]}[by]
        order_pos = [i for _, i in sorted((g["centre"][axis], i) for g, i in zip(gs, ids))]
        order_key = [i for _, i in sorted((key(g), i) for g, i in zip(gs, ids))]
        return order_pos == order_key, "measured", f"by position {order_pos}, by {by} {order_key}"

    def _answered(self, ids, c):
        a = self.s.hypotheses.get("answers") or []
        return (bool(a) or None), "model", "answer produced by Gemma from the measured object list (not verified)"

    def _reported(self, ids, c):
        return bool(self.s.facts.get("objects")), "measured", "report lists measured objects"

    def _at_home(self, ids, c):
        v = [x for x in self.s.verified if x.get("check") == "home"]
        return (bool(v) and bool(v[-1]["pass"])) or None, "measured", "HOME joint check" if v else "no HOME check this task"
