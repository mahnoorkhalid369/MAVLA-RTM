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
  in(x, y)                x is inside container y (tray, bin): within its outline, no higher than its rim
  next_to / left_of / right_of / in_front_of / behind (x, y)   deterministic relation check (agent.tools)
  located(x)              x is bound to a measured object
  observed(x, camera)     x has a measurement from that camera (e.g. wrist) in this task
  ordered(xs, by, axis)   the objects' measured `by` (size | height | width) increases along `axis` (x | y)
  answered(question)      an answer was produced (confidence: model)
  reported()              a scene report was produced
  at_home()               HOME joint check passed
  viewing()               the wrist camera points along "direction" ([x, y, z] in "frame": robot | camera) now:
                          forward kinematics of the measured joint state (agent/directions.py tolerances)
  seen()                  a frame from that "camera" (table | wrist) was taken and measured in this task
Anything else -> satisfied None, "no evaluator for <pred>" (reported as unverifiable, never as done).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PREDICATES = ("holding", "not_holding", "on", "in", "next_to", "left_of", "right_of", "in_front_of", "behind", "located",
              "observed", "ordered", "answered", "reported", "at_home", "viewing", "seen")
RELATIONS = ("in", "next_to", "left_of", "right_of", "in_front_of", "behind")  # "in": inside a container's outline
OBJECT_ARGS = ("holding", "on", "located", "observed") + RELATIONS  # predicates whose arguments name objects
# objects of these may be grounded for the goal check: binding alone never satisfies them (located would be circular)
GROUNDABLE = ("holding", "on", "observed") + RELATIONS
# the fewest arguments each predicate needs (live 2026-11-04: "holding()" from a decomposition -> IndexError)
MIN_ARGS = {"holding": 1, "on": 2, "located": 1, "observed": 1, "ordered": 1, **{r: 2 for r in RELATIONS}}
# how many arguments each predicate takes (None: any number); the line parser splits a condition's text by it, so the
# user's words may contain commas (live 2026-11-04: holding(<words>, <more words>) became two arguments)
ARITY = {**{p: 0 for p in ("not_holding", "seen", "viewing", "reported", "at_home")},
         **{p: 1 for p in ("holding", "located", "observed", "answered")}, "on": 2, **{r: 2 for r in RELATIONS},
         "ordered": None}
# conditions on where objects are: only an operation that changes an object (pick / place / ...) can make them true
OBJECT_STATE = ("holding", "on", "ordered") + RELATIONS
# capabilities that move the arm (= agent.capabilities.specs MOTION; kept here so this module needs no pydantic;
# tests/unit/test_structure.py checks they agree)
ARM_MOTION = frozenset({"pick", "place", "pick_place", "return_home", "inspect", "explore", "look"})


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
    last_pick = None  # the holding(x) condition of the most recent pick: a later place / pick_place releases it
    for cap, a in steps:
        if cap in ARM_MOTION:  # the camera's direction is a state of the arm: a later motion ends an earlier look
            out = [c for c in out if c["pred"] != "viewing"]
        obj = ((a.get("object") or {}).get("text") if isinstance(a.get("object"), dict) else a.get("target")) or None
        ref = (a.get("reference") or {}).get("text") if isinstance(a.get("reference"), dict) else None
        if cap in ("place", "pick_place") and last_pick is not None and last_pick in out:
            out.remove(last_pick)  # by step order: the object a place puts down is the one picked before it (often "#id")
            last_pick = None
        if cap == "pick":
            if last_pick is not None and last_pick in out:  # one gripper: a later pick ends the earlier holding
                out.remove(last_pick)
            last_pick = {"pred": "holding", "args": [obj]}
            out.append(last_pick)
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
        elif cap == "look":
            out.append({"pred": "viewing", "args": [], "direction": a.get("direction"), "frame": a.get("frame") or "robot"})
        elif cap == "observe":
            out.append({"pred": "seen", "args": [], "camera": "table"})
    # a later pick of the same object supersedes nothing; a later place supersedes an earlier holding(x)
    placed = {c["args"][0] for c in out if c["pred"] in RELATIONS + ("on",)}
    return [c for c in out if not (c["pred"] == "holding" and c["args"][0] in placed)]


class GoalEvaluator:
    def __init__(self, tools, state) -> None:
        self.t, self.s = tools, state  # agent.tools.Tools, agent.task_state.TaskState

    def _geom(self, oid):
        o = self.t.world.objects.get(oid)
        return None if o is None else o.geometry

    @staticmethod
    def _id(ref, bindings: dict):
        """A condition argument -> world id: an int, a "#<id>" reference, or a text bound by a worker's result; a fixed
        region's id ("R<n>", config/layout.py) stays that id (its own namespace: never a world id)."""
        from config import layout
        if isinstance(ref, int) or ref is None:
            return ref
        if isinstance(ref, str) and ref.startswith("#") and ref[1:].isdigit():
            return int(ref[1:])
        r = layout.region(ref)
        if r is not None:
            return r.id
        return bindings.get(ref)

    @staticmethod
    def _region(i):
        from config import layout
        return layout.region(i) if isinstance(i, str) else None

    def evaluate(self, cond: dict, bindings: dict) -> ConditionResult:
        p, args = cond.get("pred"), list(cond.get("args") or [])
        ids = [self._id(a, bindings) for a in args]
        bound = {str(a): i for a, i in zip(args, ids)}
        fn = getattr(self, f"_{p}", None)
        if fn is None or p not in PREDICATES:
            return ConditionResult(cond, None, "none", f"no evaluator for {p!r}", bound)
        if len(args) < MIN_ARGS.get(p, 0):
            return ConditionResult(cond, None, "none", f"malformed: {p} needs {MIN_ARGS[p]} argument(s), got {len(args)}", bound)
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
        if any(self._region(i) for i in ids[:2]):
            return None, "none", "on a fixed region is not measured (only in(<object>, <region>))"
        a, b = (self._geom(i) for i in ids[:2])
        if a is None or b is None:
            return None, "none", "objects not identified"
        dz = abs(a["support_z"] - b["top_z"])
        dxy = float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))
        ok = dz <= 0.010 and dxy <= max(b["footprint"]) / 2 + 0.010
        why = f"support {dz * 1000:.0f} mm from the top, centre {dxy * 1000:.0f} mm from its centre"
        return self._worker_placed(ids, c, ok, why)

    def _worker_placed(self, ids, c, ok, why):
        """The manipulation worker's response is accepted as measured evidence: its own camera check of placing THIS
        object with THIS relation on/at THIS destination passed (the measured pose where it was planned). The world
        model can disagree when the camera sees a stack as one object (live 2026-10-21). A check of another object,
        relation or destination proves nothing, and a later grasp makes it stale."""
        p = None if ok else self._placed(ids[0])
        dest = None if p is None else (p.get("reference_region") or p.get("reference_id"))
        if p is not None and dest == ids[1] and p.get("relation") == c.get("pred"):
            return True, "measured", (f"the worker's placement check passed: {p.get('xy_error_mm')} mm from the planned "
                                      f"spot, top {p.get('z_error_mm')} mm (frame {p.get('frame_id')}); world model: {why}")
        return ok, "measured", why

    def _placed(self, oid):
        """The latest placement check of this object in this task, if it passed and the object was not picked up
        again since (a later grasp would supersede it)."""
        if oid is None or self.s is None or self.t.world.holding == oid:
            return None
        vs = list(getattr(self.s, "verified", []))
        idx = [i for i, v in enumerate(vs) if v.get("check") == "placement" and v.get("object_id") == oid]
        if not idx or not vs[idx[-1]].get("pass"):
            return None
        if any(v.get("check", "").startswith("contact") for v in vs[idx[-1] + 1:]):  # a grasp since: may have moved
            return None
        return vs[idx[-1]]

    def _relation(self, ids, c):
        if any(self._region(i) for i in ids[:2]):
            return None, "none", f"{c['pred']} a fixed region is not measured (only in(<object>, <region>))"
        a, b = (self._geom(i) for i in ids[:2])
        if a is None or b is None:
            return None, "none", "objects not identified"
        ok = bool(self.t.relation_holds(a, b, c["pred"], c.get("frame", "robot")) and a["support_z"] <= 0.010)
        d = float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))
        return self._worker_placed(ids, c, ok, f"centres {d * 1000:.0f} mm apart, on the floor: {a['support_z'] <= 0.010}")

    _next_to = _left_of = _right_of = _in_front_of = _behind = _relation

    def _in(self, ids, c):
        """x inside container y: x's centre within y's measured outline and x resting no higher than y's rim
        (agent.tools relation_holds "in"); not required to stand on the table (it stands on the container's floor).
        y a fixed region (config/layout.py): x's whole measured footprint inside its calibrated outline and x resting on
        its surface (Tools.in_region)."""
        region = self._region(ids[1])
        if region is not None:
            a = self._geom(ids[0])
            if a is None:
                return None, "none", "object not identified"
            r = self.t.in_region(a, region)
            why = (f"footprint {r['margin_mm']:+.0f} mm inside the outline of {region.id} ({region.name}) at its worst "
                   f"point, resting {r['resting_mm']:.0f} mm above the region's surface")
            return self._worker_placed(ids, c, r["inside"], why)
        a, b = (self._geom(i) for i in ids[:2])
        if a is None or b is None:
            return None, "none", "objects not identified"
        ok = bool(self.t.relation_holds(a, b, "in"))
        d = float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))
        why = f"centre {d * 1000:.0f} mm from the container's centre, resting {a['support_z'] * 1000:.0f} mm up (rim {b['top_z'] * 1000:.0f} mm)"
        return self._worker_placed(ids, c, ok, why)

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
        """The question got an answer: an answer step's (Gemma, unverified) certain, non-empty answer, or a completed
        locate (measured). A scene report is NOT an answer; an uncertain answer is not one either."""
        answers = [a for a in (self.s.hypotheses.get("answers") or []) if str(a.get("answer", "")).strip()]
        located = [a for a in self.s.actions if a.get("capability") == "locate" and a.get("status") == "completed"]
        if answers:
            last = answers[-1]
            if last.get("certain") is False:
                return False, "model", f"the answer is not certain: {last.get('answer')!r}"
            return True, "model", "answer produced by Gemma from the measured object list (not verified)"
        if located:
            return True, "measured", "answered by locating the object (measured position)"
        return False, "measured", "no answer was produced for this question (a scene report is not an answer)"

    def _reported(self, ids, c):
        reports = getattr(self.s, "reports", []) or []
        if reports and all("reply" in r for r in reports):  # only Gemma's own reply (a finish with ANSWER)
            return True, "model", "Gemma's reply to the user (from the measured object list; not verified)"
        n = len(reports)
        return n > 0, "measured", (f"{n} scene report(s) of measured objects given this task" if n
                                   else "no scene report was given this task")

    def _viewing(self, ids, c):
        from agent.directions import view_error
        direction, frame = c.get("direction"), c.get("frame", "robot")
        if direction is None:
            return None, "none", "no direction given"
        v = view_error(self.t.bridge.q(), direction, frame)  # measured joint state -> forward kinematics
        return v["pointing"], "measured", (f"wrist camera axis {v['error_deg']} deg from {direction} ({frame} frame; "
                                           f"limit {v['tolerance_deg']}), camera at {v['camera_position_mm']} mm")

    def _seen(self, ids, c):
        cam = c.get("camera", "table")
        frames = [f for f in self.s.frames if f.get("camera") == cam]
        return bool(frames), "measured", (f"{len(frames)} {cam} frame(s) measured this task" if frames
                                          else f"no {cam} frame this task")

    def _at_home(self, ids, c):
        v = [x for x in self.s.verified if x.get("check") == "home"]
        return (bool(v) and bool(v[-1]["pass"])) or None, "measured", "HOME joint check" if v else "no HOME check this task"
