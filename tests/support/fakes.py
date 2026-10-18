"""Fakes of agent.tools.Tools for offline tests (no simulator, no GPU, no ROS).

    geom(...)    a measured-geometry dict like perception.geometry.Measured.to_dict()
    StubTools    scripted outcomes for each Tools method used by the pick-and-place workflow
    GoalTools    StubTools + what the goal supervisor needs (scripted plans, real relation geometry,
                 the real session hooks)

Moved verbatim from tests/integration/test_agent.py (geom, StubTools) and tests/integration/test_supervisor_goal.py (GoalTools).
"""

from types import SimpleNamespace

import numpy as np

from agent.vlm_roles import SceneAnswer
from perception.world_model import WorldModel


def geom(x, y, top, fp=(0.04, 0.04), support=0.0, yaw=0.0):
    h = top - support
    return {"centre": [x, y, support + h / 2], "top_z": top, "height": h, "support_z": support, "yaw_deg": yaw,
            "extents": list(fp), "footprint": [max(fp), min(fp)], "grasp_yaw_deg": yaw, "grasp_width": min(fp),
            "symmetry_deg": 90.0 if abs(fp[0] - fp[1]) < 0.008 else 180.0, "tilt_deg": 0.5}


class StubTools:
    """Scripted outcomes for each tool: tests the state machine, budgets and honesty of the verdict."""

    OBJECTS = {1: (geom(0.55, 0.15, 0.04), {"name": "cube", "colour": "yellow"}),
               2: (geom(0.55, 0.45, 0.05), {"name": "box", "colour": "tan"}),
               3: (geom(0.0, 0.85, 0.04, fp=(0.05, 0.05)), {"name": "cube", "colour": "gold yellow"})}

    def __init__(self, contact=(True,), placement=(True,), ambiguous=None, exec_fail=None, task="pick_place",
                 home_ok=True, home_error_deg=0.1):
        self.contact, self.placement = list(contact), list(placement)
        self.home_ok, self.home_error_deg, self.home_calls = home_ok, home_error_deg, 0
        self.resolved_refs, self.last_referent = {}, None
        self.held = self.held_object = self.held_geom = self.held_close_target = None
        self.ambiguous, self.exec_fail, self.task = ambiguous or {}, exec_fail, task
        self.confirmed, self.planned = [], []
        self.world = WorldModel()
        self.ex = SimpleNamespace(safety_events=[], spin=lambda s: None)
        self.bridge = SimpleNamespace(q=lambda: np.zeros(6))
        self.held_box = None
        self.executed = []
        from agent.vlm_roles import TaskSpec
        self.roles = SimpleNamespace(parse=lambda s: TaskSpec.model_validate(
            {"task": self.task, "object": {"text": "a"}, "relation": "on", "reference": {"text": "b"}}))

    def park_for_view(self, xy):
        return {"ok": True}

    def assume_placed(self, plan):
        pass

    def perceive(self, camera):
        ids = self.ids = {k: self.world.observe(g, d) for k, (g, d) in self.OBJECTS.items()}
        return SimpleNamespace(marks={k: None for k in self.OBJECTS}, measured={k: g for k, (g, _) in self.OBJECTS.items()},
                               object_ids=ids, descriptions={k: d for k, (_, d) in self.OBJECTS.items()}, frame_id="t")

    def describe(self, obs, marks=None):
        pass

    def shortlist(self, obs, texts):
        self.shortlisted = texts

    def update_world(self, obs):
        return []

    def resolve(self, obs, text):
        from agent.vlm_roles import Resolution
        if text in self.ambiguous:
            return Resolution(id=None, ambiguous=True, candidates=self.ambiguous[text])
        return Resolution(id=1 if text == "a" else 2)

    def confirm(self, obs, mark, text):
        from agent.vlm_roles import Confirmation
        self.confirmed.append((mark, text))
        return Confirmation(match="yes")

    def relation_holds(self, *a):
        return True

    def plan_pick_place(self, oid, rel, rid, frame, q):
        from control.skills import GripperStep, Held
        self.planned.append((oid, rid))
        return SimpleNamespace(object_id=oid, hover=["hover"], approach=["down", GripperStep(0.5, "close")], lift=["lift"],
                               place=["carry", "lower", "open", "retreat"], held=Held(0.04, 0.02, 0, "obj_1", 0.04, 0.04),
                               grasp={}, target_centre=[0.55, 0.45, 0.07], target_top=0.09, relation=rel, reference_id=rid)

    def execute(self, steps, label):
        self.executed.append(label)
        ok = label != self.exec_fail
        return SimpleNamespace(ok=ok, failed_step="x" if not ok else "", reason="safety layer rejected" if not ok else "")

    def verify_contact(self, target):
        return {"check": "contact", "pass": self.contact.pop(0) if len(self.contact) > 1 else self.contact[0]}

    deterministic = staticmethod(lambda reason: "out of reach" in reason)

    def set_held(self, plan, close_target):
        self.held, self.held_object, self.held_close_target = plan.held, plan.object_id, close_target
        self.held_geom = dict(self.world.objects[plan.object_id].geometry) if plan.object_id in self.world.objects else geom(0.5, 0.2, 0.04)
        self.world.holding = plan.object_id

    def clear_held(self):
        self.held = self.held_object = self.held_geom = self.held_close_target = None
        self.world.holding = None

    def plan_pick(self, oid, q):
        p = self.plan_pick_place(oid, "hold", None, "robot", q)
        p.place = []
        return p

    def plan_place_held(self, rel, rid, frame, q):
        p = self.plan_pick_place(self.held_object, rel, rid, frame, q)
        p.hover, p.approach, p.lift = [], [], []
        return p

    def return_home(self):
        self.home_calls += 1
        if self.home_ok is True:
            res = self.execute(["home"], "return HOME")
            return {"ok": res.ok, "reason": res.reason, "result": res}
        return {"ok": False, "reason": self.home_ok}

    def verify_home(self):
        return {"check": "home", "max_error_deg": self.home_error_deg, "tolerance_deg": 1.5, "tcp_error_mm": 0.5,
                "pass": self.home_error_deg <= 1.5}

    def verify_placement(self, plan):
        ok = self.placement.pop(0) if len(self.placement) > 1 else self.placement[0]
        return {"check": "placement", "pass": ok, "xy_error_mm": 3.0, "z_error_mm": 1.0, "tilt_deg": 0.4}


class GoalTools(StubTools):
    """StubTools + what the goal supervisor needs; plans are scripted. Relations are judged with the real
    Tools.relation_holds geometry; placing a held object follows the requested relation and reference
    (misplace=True keeps the old fixed spot on top of object 2: a wrong placement for negative tests)."""

    from agent.tools import Tools as _T
    relation_holds = _T.relation_holds
    _direction = _T._direction
    begin_instruction = _T.begin_instruction  # session hooks: the real code
    revalidate_held = _T.revalidate_held
    OPEN_FINGER_DEG = _T.OPEN_FINGER_DEG
    del _T

    def log(self, *a, **k):  # replaced by the instruction's event log in sessions
        pass

    def __init__(self, plans, answer=None, misplace=False, **kw):
        super().__init__(**kw)
        from config.agent import Relations
        self.rel, self.misplace = Relations(), misplace
        self.plans, self.plan_calls, self.perceived, self.parks = list(plans), [], 0, 0
        self.plan_contexts: list = []  # the context Gemma was given with each planning call
        self.motion_count, self.view_cleared_at = 0, -1
        tools = self

        def plan_goal(text, menu, context=""):
            tools.plan_calls.append(text)
            tools.plan_contexts.append(context)
            if not tools.plans:  # the scripted planner has nothing more to say: like Gemma failing twice
                from agent.vlm_roles import VLMFailure
                raise VLMFailure("plan: no scripted plan left")
            p = tools.plans.pop(0)
            if isinstance(p, Exception):
                raise p
            return p
        from agent.vlm_roles import RequestKind
        # the request classifier (agent/authority.py): tests set request_kind; "action" grants every effect
        self.request_kind = "action"
        self.roles = SimpleNamespace(plan_goal=plan_goal, answer=lambda q, facts, ids: answer or SceneAnswer(answer="two", object_ids=[1]),
                                     classify_request=lambda text: RequestKind(kind=tools.request_kind, evidence=text))

    def execute(self, steps, label):
        self.motion_count += 1
        return super().execute(steps, label)

    def plan_place_held(self, rel, rid, frame, q):
        """The held object goes where the relation says, from the reference's measured geometry: on its top, or
        beside it (3 cm gap) in the relation's direction (next_to: the reference's right)."""
        p = super().plan_place_held(rel, rid, frame, q)
        if self.misplace:
            return p
        g, h = self.world.objects[rid].geometry, self.held_geom
        if rel == "on":
            xy, top = np.asarray(g["centre"][:2]), g["top_z"] + h["height"]
        else:
            d = np.array([1.0, 0.0]) if rel == "next_to" else self._direction(rel, np.asarray(g["centre"][:2]), frame)
            xy = np.asarray(g["centre"][:2]) + d * (max(g["footprint"]) / 2 + max(h["footprint"]) / 2 + 0.03)
            top = h["height"]
        p.target_centre, p.target_top = [float(xy[0]), float(xy[1]), top - h["height"] / 2], float(top)
        return p

    def verify_placement(self, plan):
        v = super().verify_placement(plan)
        o = self.world.objects.get(plan.object_id)
        if v["pass"] and o is not None:  # like Tools.verify_placement: the re-measured pose replaces the old one
            h = o.geometry["height"]
            o.geometry = geom(plan.target_centre[0], plan.target_centre[1], plan.target_top, support=plan.target_top - h)
        return v

    def park_for_view(self, xy):
        self.parks += 1
        self.view_cleared_at = self.motion_count
        return {"ok": True}

    def perceive(self, camera):
        self.perceived += 1
        o = super().perceive(camera)
        o.camera, o.scores, o.excluded = "table", {k: 0.9 for k in o.marks}, []
        o.taken_at, o.motion_count = 0.0, self.motion_count
        o.cleared_view = self.view_cleared_at == self.motion_count
        return o

    def is_fresh(self, obs, need_clear_view=False):
        return obs is not None and obs.motion_count == self.motion_count and (obs.cleared_view or not need_clear_view)

    def view_blockage(self, q=None):
        return 1, 36

    def obstacles(self):
        return []

    def lexical_candidates(self, obs, text):
        return []

    def resolve(self, obs, text):
        from agent.vlm_roles import Resolution
        if text in self.ambiguous:
            return Resolution(id=None, ambiguous=True, candidates=self.ambiguous[text])
        return Resolution(id={"a": 1, "b": 2, "c": 3}.get(text, 2))
