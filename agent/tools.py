"""Agent tools: the only way the agents touch the world. Deterministic except the VLM roles.

perceive -> describe -> resolve / confirm (Perception); plan_pick / plan_place (Grasp + Motion
planner: SkillPlanner + CollisionChecker with obstacles from MEASURED geometry); execute
(RosExecutor); verify_contact / verify_placement (sensor-only); park_for_view (arm out of the table
camera's line of sight before re-observing). No ground truth, no object knowledge.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from agent.config import FIXTURES, Relations, Tolerances, unit
from agent.geometry import RobotVolume, measure, snap_supports
from agent.somark import draw_marks, draw_names, grid, isolate, montage
from agent.world import WorldModel
from control.collision import Box, CollisionChecker
from control.kinematics import Kinematics
from control.motion import PlanningError, joint_move
from control.robot import ARM_HOME, ARM_SPEED, gripper_gap, table_camera_pose
from control.skills import Held, SkillPlanner


def mean_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return rgb[mask.astype(bool)].reshape(-1, 3).mean(axis=0) if mask.any() else np.zeros(3)


class DescriptionMemory:
    """VLM descriptions kept across the instructions of one session. A description is reused only for a
    measurement at the same place with the same size (WorldModel gates: 30 mm, top 20 mm, footprint 20 mm)
    AND the same mean colour inside the mask (every channel within COLOUR_TOL of 255): anything that
    moved, was covered or looks different is described again."""

    COLOUR_TOL = 20.0

    def __init__(self) -> None:
        self.entries: list[tuple[dict, np.ndarray, dict]] = []

    @staticmethod
    def _same_place(a: dict, b: dict) -> bool:
        from agent.world import SAME_FOOTPRINT, SAME_TOP, SAME_XY
        return (np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]) < SAME_XY
                and abs(a["top_z"] - b["top_z"]) < SAME_TOP and abs(a["footprint"][0] - b["footprint"][0]) < SAME_FOOTPRINT)

    def lookup(self, g: dict, colour: np.ndarray) -> dict | None:
        for eg, ec, d in self.entries:
            if self._same_place(eg, g) and np.all(np.abs(ec - colour) <= self.COLOUR_TOL):
                return dict(d)
        return None

    def store(self, g: dict, colour: np.ndarray, description: dict) -> None:
        colour = np.asarray(colour, float)
        self.entries = [e for e in self.entries if not (self._same_place(e[0], g) and np.all(np.abs(e[1] - colour) <= self.COLOUR_TOL))]
        self.entries.append((dict(g), colour, dict(description)))


@dataclass
class Observation:
    frame_id: str
    camera: str
    rgb: np.ndarray
    som: np.ndarray | None
    marks: dict[int, np.ndarray]  # mark number -> mask
    measured: dict[int, dict]  # mark number -> geometry dict
    object_ids: dict[int, int] = field(default_factory=dict)  # mark number -> world object id
    descriptions: dict[int, dict] = field(default_factory=dict)  # mark number -> VLM description
    excluded: list[dict] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)  # mark number -> detector confidence
    shortlists: dict[str, list[int]] = field(default_factory=dict)  # reference text -> candidate marks
    taken_at: float = 0.0  # time.monotonic() when the frame was taken
    motion_count: int = -1  # Tools.motion_count when the frame was taken (robot not moved since if equal)
    cleared_view: bool = False  # taken after park_for_view (arm out of the camera's view)


@dataclass
class PickPlan:
    object_id: int
    hover: list
    approach: list  # [descend, close]
    lift: list
    place: list  # [carry, lower, open, retreat]
    held: Held
    grasp: dict
    target_centre: list  # planned centre of the placed object (world)
    target_top: float
    relation: str
    reference_id: int | None


class Tools:
    HOME_TOLERANCE_DEG = 1.5  # per joint: RosExecutor's validated reach tolerance (a step fails beyond it)
    OPEN_FINGER_DEG = 2.0  # finger_joint <= this = gripper open (open = 0 deg; 2 deg = 83 mm gap)

    def __init__(self, bridge, detector, roles, executor, log, run_dir, live=None, memory=None) -> None:
        self.bridge, self.detector, self.roles, self.ex, self.log, self.run_dir = bridge, detector, roles, executor, log, run_dir
        self.live = live  # agent.live.LiveChannel or None (viewer is optional)
        self.memory = memory  # agent.tools.DescriptionMemory shared by the instructions of one session, or None
        self.motion_count = 0  # every executed robot / gripper command increments it (observation freshness)
        # held object (persists across instructions of a session): Held body, world id, measured geometry at the
        # grasp, finger close target (for retention checks)
        self.held, self.held_object, self.held_geom, self.held_close_target = None, None, None, None
        self.resolved_refs: dict[str, int] = {}  # reference text -> world id chosen by the user (clarifications)
        self.last_referent: int | None = None  # world id of the object the last instruction acted on
        self.view_cleared_at = -1  # motion_count right after the last successful park_for_view
        self.world = WorldModel()
        self.kin = Kinematics()
        self.wrist_kin = Kinematics(tip="camera_color_optical_frame")
        self.tol, self.rel = Tolerances(), Relations()
        self.held_box = None  # (pose in tool0, size) while holding, from measured geometry
        self._frames = 0

    # ------------------------------------------------------------------ perception
    def _camera_pose(self, frame) -> np.ndarray:
        return table_camera_pose() if frame.camera == "table" else self.wrist_kin.fk(frame.q)

    MAX_OBS_AGE_S = 120.0

    def is_fresh(self, obs: Observation | None, need_clear_view: bool = False) -> bool:
        """An observation still describes the scene: the robot has not moved since it was taken, it is younger
        than MAX_OBS_AGE_S and (if required) it was taken with the arm parked out of the camera's view."""
        return (obs is not None and obs.motion_count == self.motion_count
                and time.monotonic() - obs.taken_at < self.MAX_OBS_AGE_S and (obs.cleared_view or not need_clear_view))

    def perceive(self, camera: str = "table") -> Observation:
        frame = self.bridge.snapshot(camera)
        T = self._camera_pose(frame)
        dets = self.detector.detect(frame.rgb)
        checker = CollisionChecker()
        tool = Kinematics(tip="tool0").fk(frame.q)
        robot = RobotVolume(checker, frame.q, gripper_gap(frame.finger), self.held_box, tool)
        marks, measured, excluded, scores = {}, {}, [], {}
        for d in sorted(dets, key=lambda d: -d.score):
            g, why = measure(d.mask, frame.depth, frame.K, T, robot)
            if g is None:
                excluded.append({"score": round(d.score, 3), "pixels": int(d.mask.sum()), "reason": why})
                continue
            k = len(marks) + 1
            marks[k], measured[k], scores[k] = d.mask, g.to_dict(), round(float(d.score), 3)
        if self.held_box is not None:  # the held object is not on the table: drop its measurement
            Th, size = self.held_box
            Tw = tool @ Th
            half = np.asarray(size) / 2 + 0.02
            for k in [k for k, g in measured.items() if np.all(np.abs((np.asarray(g["centre"]) - Tw[:3, 3]) @ Tw[:3, :3]) <= half)]:
                excluded.append({"score": scores[k], "pixels": int(marks[k].sum()), "reason": "the held object"})
                del marks[k], measured[k], scores[k]
            order = sorted(marks)  # renumber 1..n consistently
            marks, measured, scores = ({i + 1: d[k] for i, k in enumerate(order)} for d in (marks, measured, scores))
        snap_supports(list(measured.values()))
        self._frames += 1
        fid = f"{camera}_{self._frames:03d}"
        som = draw_marks(frame.rgb, marks) if marks else frame.rgb.copy()
        cv2.imwrite(str(self.run_dir / "images" / f"{fid}_som.jpg"), cv2.cvtColor(som, cv2.COLOR_RGB2BGR))
        self.log("tool", tool="perceive", camera=camera, frame_id=fid, detections=len(dets), objects=len(marks),
                 excluded=excluded, measured={k: {kk: v[kk] for kk in ("centre", "top_z", "height", "footprint", "grasp_yaw_deg", "grasp_width")}
                                              for k, v in measured.items()})
        obs = Observation(fid, camera, frame.rgb, som, marks, measured, excluded=excluded, scores=scores,
                          taken_at=time.monotonic(), motion_count=self.motion_count,
                          cleared_view=self.view_cleared_at == self.motion_count)
        if self.live:
            self.live.perception(obs)
        return obs

    def describe(self, obs: Observation, marks=None) -> None:
        """The VLM's own words for the given marks (default: all), each from its own isolated crop.
        Objects already described in this task (same place and size) keep their description."""
        if not obs.marks:
            return
        todo = {}
        for k in (list(obs.marks) if marks is None else marks):
            if k in obs.descriptions:
                continue
            known = self.world.match(obs.measured[k])
            if known is not None and self.world.objects[known].description:
                obs.descriptions[k] = dict(self.world.objects[known].description)
                continue
            if self.memory is not None:
                d = self.memory.lookup(obs.measured[k], mean_rgb(obs.rgb, obs.marks[k]))
                if d is not None:
                    obs.descriptions[k] = d
                    self.log("tool", tool="describe_reused", mark=k, description=d)
                    continue
            c = isolate(obs.rgb, obs.marks[k])
            cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_obj_{k}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            todo[k] = c
        if todo:  # one call per crop (same prompt as validated)
            for k, d in self.roles.describe_many(todo).items():
                obs.descriptions[k] = d.model_dump()
                if self.memory is not None:
                    self.memory.store(obs.measured[k], mean_rgb(obs.rgb, obs.marks[k]), obs.descriptions[k])
        img = draw_names(obs.som, {k: obs.marks[k] for k in obs.descriptions},
                         {k: f"{v.get('colour', '')} {v.get('name', '')}" for k, v in obs.descriptions.items()})
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_names.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if self.live:
            self.live.perception(obs)

    def shortlist(self, obs: Observation, texts: list[str]) -> None:
        """One batched VLM call over a grid of every object's isolated crop: per reference text, the
        objects that could match. Only those are then described. A failed call, or an empty list for a
        text, falls back to describing every object (open-ended references still work)."""
        texts = list(dict.fromkeys(t for t in texts if t))
        if not obs.marks or not texts:
            return
        phrases = {chr(ord("A") + i): t for i, t in enumerate(texts)}
        img = grid({k: isolate(obs.rgb, m) for k, m in obs.marks.items()})
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_grid.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        t0 = time.monotonic()
        try:
            lists = self.roles.shortlist(img, phrases, list(obs.marks))
        except Exception as exc:  # VLMFailure: describe everything instead (previous behaviour)
            self.log("tool", tool="shortlist", ok=False, reason=str(exc))
            self.describe(obs)
            return
        obs.shortlists = {phrases[k]: sorted(set(v)) for k, v in lists.items()}
        empty = [t for t, v in obs.shortlists.items() if not v]
        self.log("tool", tool="shortlist", ok=True, lists=obs.shortlists, objects=len(obs.marks),
                 seconds=round(time.monotonic() - t0, 2), fallback_all=empty)
        if empty:
            self.describe(obs)
        else:
            self.describe(obs, sorted({k for v in obs.shortlists.values() for k in v}))

    def pool(self, obs: Observation, text: str) -> list[int]:
        """Candidate marks for a reference: its shortlist, or every object when there is none."""
        return obs.shortlists.get(text) or list(obs.marks)

    def update_world(self, obs: Observation) -> list[dict]:
        """Store every measured object; returns neighbours that moved (whole-scene re-perception)."""
        before = {oid: list(o.geometry["centre"]) for oid, o in self.world.objects.items()}
        for k, g in obs.measured.items():
            obs.object_ids[k] = self.world.observe(g, obs.descriptions.get(k))
            if k not in obs.descriptions and self.world.objects[obs.object_ids[k]].description:  # known: no VLM call
                obs.descriptions[k] = dict(self.world.objects[obs.object_ids[k]].description)
        moved = []
        visible_top = {oid for oid, o in self.world.objects.items() if o.geometry.get("tilt_deg") is not None}
        for oid, c in before.items():
            now = self.world.objects[oid].geometry["centre"]
            # only objects whose top face is seen now: a newly covered object's centre is not measurable
            if oid in obs.object_ids.values() and oid in visible_top and np.hypot(now[0] - c[0], now[1] - c[1]) > self.tol.xy_m:
                moved.append({"id": oid, "from_mm": [round(v * 1000) for v in c], "to_mm": [round(v * 1000) for v in now]})
        if moved:
            self.log("world", event="disturbed", moved=moved)
        if self.live:
            self.live.perception(obs)
        return moved

    STOPWORDS = {"the", "a", "an", "of", "and", "that", "this", "is", "one", "with"}  # English function words

    @classmethod
    def words(cls, text: str) -> set[str]:
        ws = {w for w in re.split(r"[^a-z]+", text.lower()) if w and w not in cls.STOPWORDS}
        return ws | {w[:-1] for w in ws if len(w) > 3 and w.endswith("s")}  # plural -> singular

    def lexical_candidates(self, obs: Observation, text: str) -> list[int]:
        """Marks (among the reference's candidates) whose VLM description contains every word of the
        user's reference (the VLM's own words; deterministic, no word lists)."""
        need = self.words(text)
        pool = set(self.pool(obs, text)) if obs.marks else set(obs.descriptions)
        return [k for k, d in obs.descriptions.items() if k in pool and
                need and need <= self.words(" ".join(str(d.get(f, "")) for f in ("name", "colour", "shape", "material")))]

    def resolve(self, obs: Observation, text: str):
        """VLM resolution over the descriptions, cross-checked: if the pick is not among the objects whose
        description contains every word of the reference, or several objects do, the VLM compares those
        objects' isolated crops side by side and must pick one (or say ambiguous -> ask the user)."""
        pool = self.pool(obs, text)
        self.describe(obs, pool)  # no-op for objects already described
        descs = [{"id": k, **obs.descriptions.get(k, {})} for k in pool]
        r = self.roles.resolve(obs.som, descs, text, pool)
        lex = self.lexical_candidates(obs, text)
        need_compare = bool(lex) and (len(lex) > 1 or (r.id is not None and r.id not in lex))
        self.log("tool", tool="lexical_check", text=text, resolver_id=r.id, lexical=lex, compare=need_compare)
        if not need_compare:
            return r
        ids = sorted(set(lex) | ({r.id} if r.id is not None else set()))
        img = montage({k: isolate(obs.rgb, obs.marks[k]) for k in ids})
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_compare_{'_'.join(map(str, ids))}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        c = self.roles.compare(img, text, ids)
        # second, independent cue from the VLM's own descriptions: the candidate whose description adds
        # the fewest words beyond the reference is the most specific match. Accept the visual choice only
        # if it is the single most specific candidate; if the cues disagree or tie -> ambiguous (ask).
        need = self.words(text)
        extra = {k: len(self.words(" ".join(str(obs.descriptions.get(k, {}).get(f, "")) for f in ("name", "colour", "shape"))) - need)
                 for k in ids}
        best = [k for k in ids if extra[k] == min(extra.values())]
        agree = c.id is not None and best == [c.id]
        self.log("tool", tool="compare", text=text, ids=ids, extra_words=extra, most_specific=best, agree=agree, **c.model_dump())
        if agree:
            return c
        return type(c)(id=None, ambiguous=True, candidates=ids,
                       reason=f"visual choice {c.id} vs most specific description {best}: cues disagree")

    def candidate_image(self, obs: Observation, marks: list[int]) -> str:
        """Picture for a clarification question: the candidates' isolated crops side by side, numbered."""
        path = self.run_dir / "images" / f"{obs.frame_id}_question_{'_'.join(map(str, marks))}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(montage({k: isolate(obs.rgb, obs.marks[k]) for k in marks}), cv2.COLOR_RGB2BGR))
        return str(path)

    def confirm(self, obs: Observation, mark: int, text: str):
        c = isolate(obs.rgb, obs.marks[mark])
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_crop_{mark}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
        return self.roles.confirm(c, text)

    def relation_holds(self, target: dict, anchor: dict, relation: str, frame: str = "robot") -> bool:
        """Deterministic check of a relational REFERENCE from measured positions (ARCHITECTURE §3.8)."""
        t, a = np.asarray(target["centre"]), np.asarray(anchor["centre"])
        if relation == "on":
            return abs(target["support_z"] - anchor["top_z"]) < 0.01 and np.hypot(*(t[:2] - a[:2])) < max(anchor["footprint"]) / 2
        gap = np.hypot(*(t[:2] - a[:2])) - (max(target["footprint"]) + max(anchor["footprint"])) / 2
        if gap > self.rel.relation_check_max_gap_m:
            return False
        if relation == "next_to":
            return True
        axis = self._direction(relation, a[:2], frame)
        v = unit(t[:2] - a[:2])
        return float(np.dot(v, axis)) >= np.cos(np.radians(45))

    # -------------------------------------------------------------- planning
    def _direction(self, relation: str, ref_xy, frame: str) -> np.ndarray:
        if frame == "camera":
            R = table_camera_pose()[:3, :3]
            right, forward = unit(R[:2, 0]), unit(R[:2, 2])  # image right; viewing direction on the floor
            dirs = {"left_of": -right, "right_of": right, "in_front_of": -forward, "behind": forward}
        else:  # robot frame: +x right, +y forward; "in front of" = between the reference and the robot
            radial = unit(ref_xy)
            dirs = {"left_of": np.array([-1.0, 0.0]), "right_of": np.array([1.0, 0.0]), "in_front_of": -radial, "behind": radial}
        return dirs[relation]

    def obstacles(self) -> list[Box]:
        boxes = [o.box() for o in self.world.objects.values() if o.status in ("on_surface", "occluded")]
        boxes += [Box(f.name, np.asarray(f.centre), np.eye(3), np.asarray(f.half)) for f in FIXTURES]
        return boxes

    def _support_name(self, g: dict) -> str | None:
        if g["support_z"] <= 0.0:
            return None
        for o in self.world.objects.values():
            og = o.geometry
            if abs(og["top_z"] - g["support_z"]) < 0.01 and np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]) < max(og["footprint"]) / 2 + 0.01:
                return f"obj_{o.id}"
        return None

    def plan_pick_place(self, object_id: int, relation: str, reference_id: int | None, frame: str,
                        q_start: np.ndarray, yaw_offset_deg: float = 0.0) -> PickPlan:
        o = self.world.objects[object_id]
        g = o.geometry
        for oid in (object_id, reference_id if relation == "on" else None):
            if oid is not None and self.world.objects[oid].geometry["tilt_deg"] is None:
                raise PlanningError(f"object {oid}: its top face is not visible, so its pose is not measured well enough")
        planner = SkillPlanner(self.kin, collision=CollisionChecker(self.obstacles()))
        width = g["grasp_width"]
        # extent along the fingers' faces = the footprint side that is not the grasp width
        fp = g["footprint"]
        length = fp[0] if abs(fp[1] - width) <= abs(fp[0] - width) else fp[1]
        planner.check_width(width)
        top = np.array([g["centre"][0], g["centre"][1], g["top_z"]])
        name = f"obj_{object_id}"
        hover, q, yaw = planner.hover(q_start, top, g["grasp_yaw_deg"] + yaw_offset_deg, g["symmetry_deg"])
        approach, lift, q, held = planner.grasp(q, top, yaw, name, width, length, self._support_name(g), g["height"])
        place, target_centre, target_top = self._plan_place(planner, q, held, g, relation, reference_id, frame)
        return PickPlan(object_id, hover, approach, lift, place, held,
                        {"yaw_deg": round(yaw, 1), "width_mm": round(width * 1000, 1), "height_mm": round(g["height"] * 1000, 1)},
                        list(map(float, target_centre)), float(target_top), relation, reference_id)

    DETERMINISTIC = ("out of reach", "too narrow", "fits at most", "no free, reachable", "top face is not visible",
                     "only ", "mm of the pads")  # PlanningError texts that re-observing cannot change

    @classmethod
    def deterministic(cls, reason: str) -> bool:
        return any(k in reason for k in cls.DETERMINISTIC)

    def plan_pick(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float = 0.0) -> PickPlan:
        """Grasp-only plan (hover, descend + close, lift) with the same checks as plan_pick_place."""
        o = self.world.objects[object_id]
        g = o.geometry
        if g["tilt_deg"] is None:
            raise PlanningError(f"object {object_id}: its top face is not visible, so its pose is not measured well enough")
        planner = SkillPlanner(self.kin, collision=CollisionChecker(self.obstacles()))
        width = g["grasp_width"]
        fp = g["footprint"]
        length = fp[0] if abs(fp[1] - width) <= abs(fp[0] - width) else fp[1]
        planner.check_width(width)
        top = np.array([g["centre"][0], g["centre"][1], g["top_z"]])
        hover, q, yaw = planner.hover(q_start, top, g["grasp_yaw_deg"] + yaw_offset_deg, g["symmetry_deg"])
        approach, lift, q, held = planner.grasp(q, top, yaw, f"obj_{object_id}", width, length, self._support_name(g), g["height"])
        return PickPlan(object_id, hover, approach, lift, [], held,
                        {"yaw_deg": round(yaw, 1), "width_mm": round(width * 1000, 1), "height_mm": round(g["height"] * 1000, 1)},
                        [float(g["centre"][0]), float(g["centre"][1]), float(g["centre"][2])], float(g["top_z"]), "hold", None)

    def plan_place_held(self, relation: str, reference_id: int | None, frame: str, q_start: np.ndarray) -> PickPlan:
        """Placement plan for the object already in the gripper (carry, lower, open, retreat)."""
        if self.held is None or self.held_object is None:
            raise PlanningError("nothing is held")
        planner = SkillPlanner(self.kin, collision=CollisionChecker(self.obstacles()))
        place, target_centre, target_top = self._plan_place(planner, q_start, self.held, self.held_geom, relation, reference_id, frame)
        return PickPlan(self.held_object, [], [], [], place, self.held,
                        {"yaw_deg": round(self.held.yaw_deg, 1), "width_mm": round(self.held.width * 1000, 1),
                         "height_mm": round(self.held.height * 1000, 1)},
                        list(map(float, target_centre)), float(target_top), relation, reference_id)

    def set_held(self, plan: PickPlan, close_target: float) -> None:
        self.held, self.held_object, self.held_close_target = plan.held, plan.object_id, close_target
        self.held_geom = dict(self.world.objects[plan.object_id].geometry)
        self.held_box = plan.held.collision_body()
        self.world.holding = plan.object_id
        self.world.objects[plan.object_id].status = "held"

    def clear_held(self) -> None:
        self.held, self.held_object, self.held_geom, self.held_close_target = None, None, None, None
        self.held_box, self.world.holding = None, None

    def _plan_place(self, planner, q, held: Held, g: dict, relation: str, reference_id: int | None, frame: str):
        if relation == "on":
            r = self.world.objects[reference_id].geometry
            if min(r["extents"]) < self.tol.min_support_width_m:
                raise PlanningError(f"the reference's top ({min(r['extents']) * 1000:.0f} mm) is too narrow to place on")
            support = np.array([r["centre"][0], r["centre"][1], r["top_z"]])
            place, _ = planner.place(q, held, support)
            return place, [support[0], support[1], support[2] + held.height / 2], support[2] + held.height
        r = self.world.objects[reference_id].geometry
        ref_xy = np.asarray(r["centre"][:2])
        reach = max(r["footprint"]) / 2 + max(g["footprint"]) / 2 + self.rel.gap_m
        if relation == "next_to":
            cands = [self._direction(k, ref_xy, frame) for k in ("in_front_of", "left_of", "right_of", "behind")]
            cands.sort(key=lambda d: abs(np.hypot(*(ref_xy + d * reach)) - 0.65))  # closest to mid reach first
        else:
            cands = [self._direction(relation, ref_xy, frame)]
        last = None
        for d in cands:
            for ang in self.rel.search_angles_deg:
                a = np.radians(ang)
                dd = np.array([d[0] * np.cos(a) - d[1] * np.sin(a), d[0] * np.sin(a) + d[1] * np.cos(a)])
                for extra in np.arange(0.0, self.rel.search_max_m + 1e-9, self.rel.search_step_m):
                    xy = ref_xy + dd * (reach + extra)
                    if not 0.45 <= np.hypot(*xy) <= 0.90:  # reliable pick/place ring (feasibility map, measured)
                        continue
                    if self._too_close(xy, g, exclude=None):
                        continue
                    try:
                        place, _ = planner.place(q, held, np.array([xy[0], xy[1], 0.0]))
                        return place, [xy[0], xy[1], held.height / 2], held.height
                    except PlanningError as exc:
                        last = exc
        raise PlanningError(f"no free, reachable, collision-free spot {relation} the reference ({last})")

    def _too_close(self, xy, g: dict, exclude) -> bool:
        r = max(g["footprint"]) / 2
        for o in self.world.objects.values():
            if o.status not in ("on_surface", "occluded") or o.id == exclude:
                continue
            og = o.geometry
            if og["centre"] is not None and np.hypot(og["centre"][0] - xy[0], og["centre"][1] - xy[1]) < r + max(og["footprint"]) / 2 + self.rel.gap_m:
                if np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]) > 1e-6:  # not the moving object itself
                    return True
        return False

    # ------------------------------------------------------------------- acting
    def execute(self, steps, label: str):
        t0 = time.monotonic()
        self.log("tool", tool="execute_start", label=label, steps=[getattr(x, "label", str(x)) for x in steps])
        # RosExecutor (frozen) reads its sim clock only while spinning: refresh it after the VLM pauses,
        # otherwise its first step starts from a stale clock (wrong tracking numbers, early step end)
        self.ex.spin(0.3)
        self.motion_count += 1  # counted before executing: a failed command may still have moved the robot
        res = self.ex.execute(steps)
        self.log("tool", tool="execute", label=label, ok=res.ok, failed_step=res.failed_step, reason=res.reason,
                 steps=res.steps, seconds=round(time.monotonic() - t0, 2))
        return res

    def verify_contact(self, close_target: float) -> dict:
        finger = self.bridge.finger()
        short = float(np.degrees(close_target - finger))
        v = {"check": "contact", "finger_deg": round(float(np.degrees(finger)), 2), "target_deg": round(float(np.degrees(close_target)), 2),
             "stopped_short_deg": round(short, 2), "gap_mm": round(gripper_gap(finger) * 1000, 1), "pass": short >= self.tol.contact_deg}
        self.log("verification", **v)
        return v

    # ARM_HOME turned about J1; kept >= 10 deg inside the J1 limits (+-170): a goal AT the limit can be
    # measured a hair beyond it and the safety layer then (correctly) rejects the next trajectory
    PARK_J1_DEG = (160.0, 135.0, -135.0, 90.0, -90.0, -160.0, 45.0, -45.0, 0.0)

    @staticmethod
    def workspace_points() -> list:
        """Floor points on the reliable reach ring (0.45-0.90 m), every 30 deg, for the first look."""
        return [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for r in (0.5, 0.7, 0.9) for a in range(0, 360, 30)]

    def view_blockage(self, q=None) -> tuple[int, int]:
        """(workspace points whose camera ray the arm blocks at joint state q (default: now), total points)."""
        q = self.bridge.q() if q is None else q
        cam = table_camera_pose()[:3, 3]
        c, r, idx, names = CollisionChecker().spheres(q)
        pts = self.workspace_points()
        blocked = 0
        for x, y in pts:
            seg = np.array([x, y, 0.03]) - cam
            t = np.clip(((c - cam) @ seg) / (seg @ seg), 0, 1)
            blocked += int(np.any(np.linalg.norm(cam + t[:, None] * seg - c, axis=1) < r + 0.02))
        return blocked, len(pts)

    def park_for_view(self, target_xy=None) -> dict:
        """Move the arm to a collision-free pose (ARM_HOME turned about J1) from which the table camera
        sees target_xy (must be unblocked), or, with target_xy None, the pose blocking the fewest
        workspace points. The camera ray test uses the robot's collision spheres (+2 cm)."""
        q0 = self.bridge.q()
        cam = table_camera_pose()[:3, 3]
        checker = CollisionChecker(self.obstacles())
        targets = [tuple(target_xy)] if target_xy is not None else self.workspace_points()
        tried, best = [], None
        # for a single target try HOME (J1 0) first: if it sees the target, the view move IS the return HOME
        order = ((0.0,) + tuple(j for j in self.PARK_J1_DEG if j != 0.0)) if target_xy is not None else self.PARK_J1_DEG
        for j1 in order:
            q = np.asarray(ARM_HOME, float).copy()
            q[0] = np.radians(j1)
            c, r, idx, names = checker.spheres(q)
            blocked = 0
            for x, y in targets:
                seg = np.array([x, y, 0.03]) - cam
                t = np.clip(((c - cam) @ seg) / (seg @ seg), 0, 1)
                blocked += int(np.any(np.linalg.norm(cam + t[:, None] * seg - c, axis=1) < r + 0.02))
            traj = joint_move(self.kin, q0, q, ARM_SPEED)
            if self.held is not None:  # carrying: the held object moves with the gripper (closed opening)
                free = checker.check_path(traj.positions, self.held.width or 0.01, self.held.collision_body(),
                                          {(p, f"obj_{self.held_object}") for p in [x.name for x in checker.parts] + ["palm", "pads", "linkages", "held"]}).ok
            else:
                free = checker.check_path(traj.positions).ok
            tried.append({"j1": j1, "blocked": blocked, "path_free": free})
            if free and (best is None or blocked < best[0]):
                best = (blocked, j1, traj)
        if best is None or (target_xy is not None and best[0] > 0):
            self.log("tool", tool="park_for_view", ok=False, tried=tried)
            return {"ok": False, "tried": tried}
        from control.skills import MoveStep
        blocked, j1, traj = best
        cur = np.degrees(np.abs(q0 - traj.positions[-1])).max()
        if cur < 0.1:  # already there: no motion needed
            self.view_cleared_at = self.motion_count
            return {"ok": True, "j1": j1, "blocked": blocked, "tried": tried, "moved": False}
        res = self.execute([MoveStep(traj, f"park J1 {j1:+.0f} (view blocked for {blocked}/{len(targets)} points)")], "park")
        if res.ok:
            self.view_cleared_at = self.motion_count
        return {"ok": res.ok, "j1": j1, "blocked": blocked, "tried": tried, "moved": True}

    def assume_placed(self, plan: PickPlan) -> None:
        """Before re-observing: count the released object as an obstacle at its PLANNED pose (it is no
        longer in the gripper; its measured pose follows from the re-observation)."""
        o = self.world.objects[plan.object_id]
        g = dict(o.geometry)
        g["centre"] = list(plan.target_centre)
        g["support_z"] = plan.target_top - plan.held.height
        g["top_z"] = plan.target_top
        o.geometry, o.status = g, "on_surface"
        self.clear_held()  # released: nothing is held any more, whatever the verification says

    def return_home(self) -> dict:
        """Collision-checked joint move to the validated ARM_HOME, only when nothing is held and the
        gripper is open. Returns {"ok", "reason", "result"}; executes through RosExecutor."""
        from control.skills import MoveStep
        finger = float(np.degrees(self.bridge.finger()))
        if self.world.holding is not None or self.held_box is not None:
            out = {"ok": False, "reason": "the robot is still holding an object"}
        elif finger > self.OPEN_FINGER_DEG:
            out = {"ok": False, "reason": f"gripper not open (finger {finger:.1f} deg > {self.OPEN_FINGER_DEG} deg)"}
        elif float(np.degrees(np.abs(self.bridge.q() - np.array(ARM_HOME, float))).max()) < 0.1:
            self.log("tool", tool="return_home", ok=True, planned_s=0.0, note="already at HOME (the placement view pose)")
            return {"ok": True, "reason": "already at HOME", "result": None}
        else:
            q0 = self.bridge.q()
            traj = joint_move(self.kin, q0, np.array(ARM_HOME, float), ARM_SPEED)
            res = CollisionChecker(self.obstacles()).check_path(traj.positions)
            if not res.ok:
                out = {"ok": False, "reason": "HOME path collides: " + "; ".join(map(str, res.contacts[:3]))}
            else:
                self.log("tool", tool="return_home", ok=True, planned_s=round(float(traj.duration), 1))
                r = self.execute([MoveStep(traj, "return HOME")], "return HOME")
                return {"ok": r.ok, "reason": r.reason, "result": r}
        self.log("tool", tool="return_home", ok=False, reason=out["reason"])
        return out

    def verify_home(self) -> dict:
        """HOME from joint-state feedback: every joint within HOME_TOLERANCE_DEG of ARM_HOME."""
        self.bridge.wait_still()
        q = self.bridge.q()
        home = np.array(ARM_HOME, float)
        err = np.degrees(np.abs(q - home))
        tcp = float(np.linalg.norm(self.kin.fk(q)[:3, 3] - self.kin.fk(home)[:3, 3]))
        v = {"check": "home", "joint_error_deg": [round(float(x), 3) for x in err], "max_error_deg": round(float(err.max()), 3),
             "tolerance_deg": self.HOME_TOLERANCE_DEG, "tcp_error_mm": round(tcp * 1000, 2),
             "pass": bool(err.max() <= self.HOME_TOLERANCE_DEG)}
        self.log("verification", **v)
        return v

    def verify_placement(self, plan: PickPlan) -> dict:
        """Re-observe the whole table (whole-scene re-perception) and check the placed object's pose."""
        obs = self.perceive("table")
        self.held_box = None
        self.world.holding = None
        o = self.world.objects[plan.object_id]
        best, best_d = None, np.inf
        for k, g in obs.measured.items():
            d = float(np.hypot(g["centre"][0] - plan.target_centre[0], g["centre"][1] - plan.target_centre[1]))
            if d < best_d:
                best, best_d = k, d
        v = {"check": "placement", "planned_centre_mm": [round(x * 1000, 1) for x in plan.target_centre],
             "planned_top_mm": round(plan.target_top * 1000, 1), "frame_id": obs.frame_id}
        if best is None or best_d > 0.05:
            v.update({"pass": False, "reason": "no object measured within 50 mm of the planned place"})
        else:
            g = obs.measured[best]
            xy, dz, tilt = best_d, g["top_z"] - plan.target_top, g["tilt_deg"]
            if tilt is None:  # top face not visible: cannot verify the pose
                tilt = float("inf")
            v.update({"measured_centre_mm": [round(x * 1000, 1) for x in g["centre"]], "xy_error_mm": round(xy * 1000, 1),
                      "z_error_mm": round(dz * 1000, 1), "tilt_deg": None if tilt == float("inf") else round(tilt, 2),
                      "pass": xy <= self.tol.xy_m and abs(dz) <= self.tol.z_m and tilt <= self.tol.tilt_deg})
            o.geometry, o.status = g, "on_surface"
        moved = self.update_world(obs)  # geometric check: no VLM calls (known objects keep their description)
        v["disturbed_neighbours"] = [m for m in moved if m["id"] != plan.object_id]
        self.log("verification", **v)
        return v
