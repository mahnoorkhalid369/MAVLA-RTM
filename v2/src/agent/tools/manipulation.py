"""Manipulation planning on measured geometry: grasp / place plans (SkillPlanner + CollisionChecker with
obstacles from the world model), parking the arm out of the table camera's view, returning HOME. Every motion
is executed through Tools.execute (agent/tools/execution.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from agent.timing import timed

import numpy as np

from config import layout
from config.agent import REACH_MAX_M, REACH_MIN_M
from config.perception import FIXTURES
from control.collision import Box, CollisionChecker
from control.motion import PlanningError, joint_move
from control.robot import ARM_HOME, ARM_SPEED, table_camera_pose
from control.skills import Held, SkillPlanner, SkillParams, GripperStep, grip_rotation
from agent.performance import options
from control.robot import GRIPPER_OPEN
from safety.joint_limits import forbidden_on_path


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
    reference_region: str | None = None  # a fixed region of the calibrated layout (config/layout.py) as the destination
    home: list = field(default_factory=list)  # validated post-release return, planned before pickup
    strict_pose: bool = False  # explicit exact-pose tasks; ordinary tray transfers use supported containment


class ManipulationTools:
    def placement_relation(self, relation, reference_id):
        """Ordinary 'on a tray' means its interior; a solid support remains 'on'.

        Require an explicit container noun in the grounded catalogue, never infer a cavity from
        colour, footprint or a nearby object's name. Geometry/path feasibility is still mandatory.
        """
        name = self.object_words.get(reference_id, "")
        if (relation == "on" and re.search(r"\b(?:tray|basket|bin|bowl|container)\b", name, re.I)
                and not re.search(r"\b(?:closed|sealed|lidded)\b", name, re.I)):
            return "in"
        return relation

    def obstacles(self) -> list[Box]:
        boxes = [o.box() for o in self.world.objects.values() if o.status in ("on_surface", "occluded")]
        boxes += [Box(f.name, np.asarray(f.centre), np.eye(3), np.asarray(f.half)) for f in FIXTURES]
        boxes += [Box(f"unseen_{i}", np.asarray(g["centre"]), np.eye(3), np.array([g["extents"][0] / 2,
                      g["extents"][1] / 2, g["height"] / 2])) for i, g in getattr(self, "unexplained", {}).items()]
        boxes += getattr(self, "inspection_obstacles", [])
        return boxes

    def _support_name(self, g: dict) -> str | None:
        if g["support_z"] <= 0.0:
            return None
        for o in self.world.objects.values():
            og = o.geometry
            if abs(og["top_z"] - g["support_z"]) < 0.01 and np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]) < max(og["footprint"]) / 2 + 0.01:
                return f"obj_{o.id}"
        return None

    def _grasp_inputs(self, object_id: int):
        """(planner, geometry, top centre, width, length) for a grasp on the measured geometry (top face visible)."""
        g = self.world.objects[object_id].geometry
        if g.get('support_confident') is False:
            raise PlanningError('object support surface is unconfirmed; obtain a calibrated unobstructed view before picking')
        if g["tilt_deg"] is None:
            raise PlanningError(f"object {object_id}: its top face is not visible, so its pose is not measured well enough")
        planner = SkillPlanner(self.kin, params=SkillParams(carry_speed_scale=options(self).carry_speed_scale), collision=CollisionChecker(self.obstacles()))
        width = g["grasp_width"]
        # extent along the fingers' faces = the footprint side that is not the grasp width
        fp = g["footprint"]
        length = fp[0] if abs(fp[1] - width) <= abs(fp[0] - width) else fp[1]
        planner.check_width(width)
        xy = g.get("grasp_centre") or g["centre"][:2]
        top = np.array([*xy, g["top_z"]])
        return planner, g, top, width, length

    @staticmethod
    def _body_bounds(g, top, yaw):
        a = np.radians(g.get("yaw_deg") or 0.)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        grip = grip_rotation(yaw)[:2, :2]
        ext = np.abs(grip.T @ R) @ np.array(g["extents"][:2])
        offset = grip.T @ (np.array(g["centre"][:2]) - top[:2])
        return (*map(float, ext), float(g["height"])), tuple(map(float, offset))

    def _plan_grasp(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float, accept=None):
        """The top-down grasp shared by plan_pick and plan_pick_place: hover, descend + close, lift, on the measured
        geometry (its top face must be visible). Returns (planner, geometry, hover, approach, lift, q, held, grasp)."""
        planner, g, top, width, length = self._grasp_inputs(object_id)
        base = g['grasp_yaw_deg'] + yaw_offset_deg
        angles = list(np.arange(0., 360., g['symmetry_deg']))
        if g.get('yaw_deg') is None:
            angles = list(np.arange(0., 360., 45.))
        elif g.get('top_planar', True) and length <= planner.p.max_grip_width:
            angles += [90., 270.]
        candidates, error = [], None
        for angle in dict.fromkeys(angles):
            gap, along = (length, width) if g.get('yaw_deg') is not None and angle % 180 == 90 else (width, length)
            try:
                planner.check_width(gap)
                hover, q, yaw = planner.hover(q_start, top, float(base + angle), 360.)
                body, offset = self._body_bounds(g, top, yaw)
                approach, lift, q, held = planner.grasp(q, top, yaw, f"obj_{object_id}", gap, along,
                                                      self._support_name(g), g['height'], body_size=body, body_offset=offset)
                grasp = {"method": "top_down", "yaw_deg": round(yaw, 1), "tilt_deg": 0.0, "width_mm": round(gap * 1000, 1),
                 "height_mm": round(g["height"] * 1000, 1),
                 "centre_x_m": float(top[0]), "centre_y_m": float(top[1]), "centre_z_m": float(top[2]),
                 "body_width_m": body[0], "body_length_m": body[1],
                 "clearance": "open approach and full payload lift collision-checked"}
                result = (planner, g, hover, approach, lift, q, held, grasp)
                if accept is not None:
                    result = accept(result)
                candidates.append({'yaw_deg': round(yaw, 1), 'ok': True})
                self.log('grasp_candidates', object_id=object_id, candidates=candidates)
                return result
            except PlanningError as exc:
                error = exc
                candidates.append({'yaw_deg': round(base + angle, 1), 'ok': False, 'reason': str(exc)})
        self.log('grasp_candidates', object_id=object_id, candidates=candidates)
        reason = next((c['reason'] for c in candidates if 'collision' in c.get('reason', '')), str(error))
        raise PlanningError(f'no complete collision-checked grasp/transfer among {len(candidates)} top-down candidates: {reason}')

    def _plan_tilted_grasp(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float):
        """The bounded tilted grasp (SkillPlanner.tilted_pick): same object, same pick, another physical way to do
        it. Returns (g, hover, approach, lift, held, grasp, candidates); PlanningError carries .candidates."""
        planner, g, top, width, length = self._grasp_inputs(object_id)
        if np.linalg.norm(top[:2] - np.array(g["centre"][:2])) > 0.003:
            raise PlanningError("a curved off-centre grasp requires a top-down approach")
        hover, approach, lift, _, held, cands = planner.tilted_pick(
            q_start, top, g["grasp_yaw_deg"] + yaw_offset_deg, g["symmetry_deg"], f"obj_{object_id}", width, length,
            self._support_name(g), g["height"], body_size=(max(g["footprint"]), max(g["footprint"]), g["height"]))
        grasp = {"method": "tilted", "yaw_deg": round(held.yaw_deg, 1), "tilt_deg": round(held.tilt_deg, 1),
                 "width_mm": round(width * 1000, 1), "height_mm": round(g["height"] * 1000, 1)}
        return g, hover, approach, lift, held, grasp, cands

    @timed("planning")
    def plan_pick_place(self, object_id: int, relation: str, reference_id: int | None, frame: str,
                        q_start: np.ndarray, yaw_offset_deg: float = 0.0, region: str | None = None) -> PickPlan:
        for oid in (object_id, reference_id if relation in ("on", "in") else None):
            if oid is not None and self.world.objects[oid].geometry["tilt_deg"] is None:
                raise PlanningError(f"object {oid}: its top face is not visible, so its pose is not measured well enough")
        def complete(candidate):
            planner, g, hover, approach, lift, q, held, grasp = candidate
            place, target_centre, target_top = self._plan_place(planner, q, held, g, relation, reference_id, frame, region)
            plan = PickPlan(object_id, hover, approach, lift, place, held, grasp,
                        list(map(float, target_centre)), float(target_top), relation, reference_id, region)
            plan.home = self.plan_transfer_home(plan)
            return plan
        return self._plan_grasp(object_id, q_start, yaw_offset_deg, accept=complete)

    def grasp_reach(self, object_id: int) -> dict:
        """After an out-of-reach grasp: can ANY gripper orientation reach this object? Kinematics only (joint limits,
        no collisions), searched from seeds over the whole joint range: (1) the skill's grasp, gripper pointing down,
        at every yaw, above the object and at its top; (2) the object's top in any orientation (position-only IK),
        reporting the tilt that would be needed. {"from_above", "any_orientation", "tilt_deg", "distance_m",
        "bearing_deg", "centre"}"""
        from control.skills import grip_rotation
        g = self.world.objects[object_id].geometry
        c = np.asarray(g["centre"], float)
        top = np.array([c[0], c[1], g["top_z"]])
        kin, lo, hi = self.kin, self.kin.lower, self.kin.upper
        seeds = [np.clip(np.array([j1, j2, j3, 0.0, j5, 0.0]), lo, hi)
                 for j1 in np.linspace(lo[0], hi[0], 9) for j2, j3, j5 in ((-0.5, 0.5, -1.2), (0.6, 1.2, -1.2),
                                                                            (-1.0, 2.0, 1.2), (-2.4, 2.5, -2.0))]
        from_above = False
        for yaw in (0.0, 45.0, 90.0, 135.0):
            for s in seeds:
                ok = all(kin.ik(p, grip_rotation(yaw), s, iterations=120)[1] for p in (top + [0, 0, 0.12], top))
                if ok:
                    from_above = True
                    break
            if from_above:
                break
        best = None
        for s in seeds:  # position only: does the hand get there at all, and with which tilt?
            q = s.copy()
            for _ in range(150):
                T = kin.fk(q)
                e = top - T[:3, 3]
                if np.linalg.norm(e) < 0.003:
                    break
                J = np.zeros((3, len(q)))
                for i in range(len(q)):
                    dq = q.copy()
                    dq[i] += 1e-6
                    J[:, i] = (kin.fk(dq)[:3, 3] - T[:3, 3]) / 1e-6
                q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e), lo, hi)
            T = kin.fk(q)
            if np.linalg.norm(top - T[:3, 3]) < 0.005:
                tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0))))  # 0 = pointing straight down
                best = tilt if best is None else min(best, tilt)
        out = {"from_above": from_above, "any_orientation": best is not None,
               "tilt_deg": None if best is None else round(best), "distance_m": round(float(np.hypot(*c[:2])), 2),
               "bearing_deg": round(float(np.degrees(np.arctan2(c[0], c[1])))), "centre": [round(float(v), 3) for v in c]}
        self.log("tool", tool="grasp_reach", object_id=object_id, **out)
        return out

    DETERMINISTIC = ("out of reach", "too narrow", "fits at most", "no free, reachable", "top face is not visible",
                     "only ", "mm of the pads", "held with a tilted grasp", "does not fit inside",
                     "no free spot inside", "at any free spot", "is not a region of", "a fixed planar region")  # PlanningError texts that re-observing cannot change

    @classmethod
    def deterministic(cls, reason: str) -> bool:
        return any(k in reason for k in cls.DETERMINISTIC)

    @timed("planning")
    def plan_pick(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float = 0.0) -> PickPlan:
        """Grasp-only plan (hover, descend + close, lift) with the same checks as plan_pick_place.

        Top-down first (the default). Only if it fails: the bounded tilted grasp of the SAME object, an alternative
        physical way to do the same delegated pick, with the same checks (IK, joint limits, collisions, table and
        finger clearance, approach, grasp, lift). Nothing else changes: the object, the operation and what happens
        next stay the supervisor's. Both failing raises the top-down PlanningError, with a note that the tilted
        grasps failed too (their reasons are in the grasp_select log event)."""
        top_down = {"method": "top_down", "ok": True, "reason": ""}
        try:
            _, g, hover, approach, lift, _, held, grasp = self._plan_grasp(object_id, q_start, yaw_offset_deg)
        except PlanningError as exc:
            if any(k in str(exc) for k in ("fits at most", "top face is not visible")):
                raise  # tilting cannot change the object's width or make its top face measured
            if not any(k in str(exc) for k in ("out of reach", "no IK solution")):
                # the tilted grasp is for REACH only: dodging a neighbour by tilting gave a hold that cannot be placed
                # (live 2026-10-26: top-down collided with a neighbour by 0.7 mm, the block was taken tilted, and the
                # next "put it in the tray" was refused)
                raise
            top_down = {"method": "top_down", "ok": False, "reason": str(exc)}
            try:
                g, hover, approach, lift, held, grasp, cands = self._plan_tilted_grasp(object_id, q_start, yaw_offset_deg)
            except PlanningError as tilted:
                cands = getattr(tilted, "candidates", [])
                self.log("tool", tool="grasp_select", object_id=object_id, selected=None, top_down=top_down,
                         tilted=cands, max_tilt_deg=SkillPlanner(self.kin).p.max_tilt_deg)
                raise PlanningError(f"{exc}; {tilted}") from None
            self.log("tool", tool="grasp_select", object_id=object_id, selected="tilted", top_down=top_down, tilted=cands,
                     tilt_deg=grasp["tilt_deg"], yaw_deg=grasp["yaw_deg"], max_tilt_deg=SkillPlanner(self.kin).p.max_tilt_deg)
        else:
            self.log("tool", tool="grasp_select", object_id=object_id, selected="top_down", top_down=top_down, tilted=[],
                     yaw_deg=grasp["yaw_deg"])
        return PickPlan(object_id, hover, approach, lift, [], held, grasp,
                        [float(g["centre"][0]), float(g["centre"][1]), float(g["centre"][2])], float(g["top_z"]), "hold", None)

    @timed("planning")
    def plan_place_held(self, relation: str, reference_id: int | None, frame: str, q_start: np.ndarray,
                        region: str | None = None) -> PickPlan:
        """Placement plan for the object already in the gripper (carry, lower, open, retreat)."""
        if self.held is None or self.held_object is None:
            raise PlanningError("nothing is held")
        if self.held.tilt_deg:  # placement plans only top-down holds; this change does not modify placement
            raise PlanningError(f"the object is held with a tilted grasp ({self.held.tilt_deg:+.0f} deg); "
                                "placement is planned only for objects picked top-down")
        planner = SkillPlanner(self.kin, params=SkillParams(carry_speed_scale=options(self).carry_speed_scale), collision=CollisionChecker(self.obstacles()))
        place, target_centre, target_top = self._plan_place(planner, q_start, self.held, self.held_geom, relation,
                                                            reference_id, frame, region)
        plan = PickPlan(self.held_object, [], [], [], place, self.held,
                        {"yaw_deg": round(self.held.yaw_deg, 1), "width_mm": round(self.held.width * 1000, 1),
                         "height_mm": round(self.held.height * 1000, 1)},
                        list(map(float, target_centre)), float(target_top), relation, reference_id, region)
        plan.home = self.plan_transfer_home(plan)
        return plan

    def plan_transfer_home(self, plan):
        """Validate retreat-to-HOME with the released payload at its intended support location."""
        obstacles = [b for b in self.obstacles() if b.name != f"obj_{plan.object_id}"]
        size = plan.held.body_size or (plan.held.width, plan.held.length or plan.held.width, plan.held.height)
        radius = float(np.hypot(*size[:2]) / 2)
        obstacles.append(Box(f"obj_{plan.object_id}", np.array(plan.target_centre), np.eye(3),
                             np.array([radius, radius, plan.held.height / 2])))
        planner = SkillPlanner(self.kin, params=SkillParams(carry_speed_scale=options(self).carry_speed_scale), collision=CollisionChecker(obstacles))
        home, _ = planner.home(plan.place[-1].trajectory.positions[-1])
        return home

    def prepare_pick(self):
        evidence = self.revalidate_held()
        if self.held_object is not None or evidence.get("held") == "unknown":
            return {"ok": False, "reason": "resolve or place the held object before another pick"}
        if np.degrees(self.bridge.finger()) > self.OPEN_FINGER_DEG:
            result = self.execute([GripperStep(GRIPPER_OPEN, "open empty gripper", 0.8)], "prepare pick")
            if not result.ok:
                return {"ok": False, "reason": result.reason}
        chained = getattr(self, "chain_ready", False)
        self.chain_ready = False
        if chained and getattr(self, "chain_ready_revision", None) == getattr(self, "scene_revision", None):
            self.log("tool", tool="prepare_pick", ok=True, note="continue from verified placement; plan from measured joints")
            return {"ok": True}
        result = self.return_home()
        if not result["ok"]:
            return result
        return {"ok": self.verify_home()["pass"], "reason": "initial HOME joint check"}

    CONTAINER_WALL_M = 0.008  # container wall thickness assumed inside its measured outline (the planner's walls)

    def _container_floor(self, g: dict) -> float:
        """A container's inside floor: its lowest visible point, when the camera sees down into it (between what it
        stands on and 10 mm below its rim); otherwise 5 mm above what it stands on."""
        b, base, top = g.get("visible_bottom_z"), g["support_z"], g["top_z"]
        return float(b) if b is not None and base < b < top - 0.01 else base + 0.005

    def _container_boxes(self, rid: int) -> list[Box]:
        """A container (the destination of "in") as four walls and a floor slab inside its measured outline, so the
        gripper may go down inside it; every part keeps the container's obstacle name."""
        g = self.world.objects[rid].geometry
        a = np.radians(g["yaw_deg"] or 0.0)
        R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        (ex, ey), t = g["extents"], self.CONTAINER_WALL_M
        base, top, floor = g["support_z"], g["top_z"], self._container_floor(g)
        c, h, name = np.array([g["centre"][0], g["centre"][1], (base + top) / 2]), (top - base) / 2, f"obj_{rid}"
        boxes = [Box(name, c + R @ [s * (ex / 2 - t / 2), 0.0, 0.0], R, np.array([t / 2, ey / 2, h])) for s in (-1, 1)]
        boxes += [Box(name, c + R @ [0.0, s * (ey / 2 - t / 2), 0.0], R, np.array([ex / 2, t / 2, h])) for s in (-1, 1)]
        if floor - base > 0.001:
            boxes.append(Box(name, np.array([c[0], c[1], (base + floor) / 2]), R, np.array([ex / 2, ey / 2, (floor - base) / 2])))
        return boxes

    def _container_contents(self, rid: int, g: dict) -> list:
        """(centre xy, radius) of the measured objects standing inside a container's outline (not the container itself)."""
        a = np.radians(g["yaw_deg"] or 0.0)
        out = []
        for o in self.world.objects.values():
            og = o.geometry
            if o.id == rid or o.status not in ("on_surface", "occluded") or og["centre"] is None:
                continue
            d = np.asarray(og["centre"][:2]) - np.asarray(g["centre"][:2])
            u, v = d[0] * np.cos(a) + d[1] * np.sin(a), -d[0] * np.sin(a) + d[1] * np.cos(a)
            if abs(u) < g["extents"][0] / 2 and abs(v) < g["extents"][1] / 2:
                out.append((np.asarray(og["centre"][:2]), max(og["footprint"]) / 2))
        return out

    def _free_spots(self, rid: int, held: Held, *, yaw_deg=None, inset=None) -> tuple[list, int]:
        """(candidate centres, objects inside) on a container's floor for the held object, nearest its centre first: inside
        the walls, and clear of the objects already in it (their footprint plus the placement gap)."""
        g = self.world.objects[rid].geometry
        dims = held.body_size or (held.width or 0., held.length or held.width or 0.)
        a = np.radians(g.get("yaw_deg") or 0.)
        axes = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        moving = grip_rotation(held.yaw_deg if yaw_deg is None else yaw_deg)[:2, :2]
        half = np.asarray(dims[:2]) / 2
        reach = np.abs(axes.T @ moving) @ half
        edge = self.CONTAINER_WALL_M if inset is None else inset
        lim = np.asarray(g["extents"]) / 2 - edge - .004 - reach
        step = self.rel.search_step_m * 2
        us, vs = (step * np.arange(-np.floor(m / step), np.floor(m / step) + 1)
                  if m >= 0 else np.array([]) for m in lim)
        contents = [o.geometry for o in self.world.objects.values()
                    if o.id != rid and f"obj_{o.id}" != held.name and o.status in ("on_surface", "occluded")]
        c = np.asarray(g["centre"][:2])
        spots = []
        def clear(xy, other):
            b = np.radians(other.get("yaw_deg") or 0.)
            other_axes = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
            other_half = np.asarray(other["extents"]) / 2
            delta = np.asarray(other["centre"][:2]) - xy
            # Separating-axis test: preserve the full oriented payload, including narrow boxes.
            return any(abs(delta @ axis) >= np.abs(axis @ moving) @ half
                       + np.abs(axis @ other_axes) @ other_half + self.rel.gap_m
                       for axis in (*moving.T, *other_axes.T))
        for u in us:
            for v in vs:
                xy = c + np.array([u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)])
                if all(clear(xy, other) for other in contents):
                    spots.append(xy)
        spots.sort(key=lambda xy: np.hypot(*(xy - c)))
        return spots, len(self._container_contents(rid, g))

    MAX_CONTAINER_SPOTS = 12  # free spots tried (nearest the centre first) before reporting no collision-free way

    def _plan_in(self, q, held: Held, reference_id: int):
        """Put the held object down INSIDE a container (tray, bin): on a free spot of the container's floor (the
        centre when it is empty; beside what is already in it otherwise), the gripper turned so its open fingers fit
        between the walls and the contents; every motion collision-checked against the walls, the floor and the
        contents."""
        r = self.world.objects[reference_id].geometry
        inside = [e - 2 * self.CONTAINER_WALL_M for e in r["extents"]]
        size = sorted((held.body_size or (held.width or 0., held.length or held.width or 0.))[:2], reverse=True)
        if size[0] > max(inside) or size[1] > min(inside):
            raise PlanningError(f"the held object ({size[0] * 1000:.0f} x {size[1] * 1000:.0f} mm) does not fit inside the "
                                f"container ({max(inside) * 1000:.0f} x {min(inside) * 1000:.0f} mm inside)")
        obstacles = [b for b in self.obstacles() if b.name != f"obj_{reference_id}"] + self._container_boxes(reference_id)
        planner = SkillPlanner(self.kin, params=SkillParams(carry_speed_scale=options(self).carry_speed_scale), collision=CollisionChecker(obstacles))
        floor = self._container_floor(r)
        yaw_c, last = float(r["yaw_deg"] or 0.0), None
        n_inside, tried, available = 0, [], 0
        for yaw in dict.fromkeys((yaw_c - 90.0, yaw_c, held.yaw_deg)):
            spots, n_inside = self._free_spots(reference_id, held, yaw_deg=yaw)
            available += len(spots)
            for xy in spots[:self.MAX_CONTAINER_SPOTS]:
                support = np.array([xy[0], xy[1], floor])
                try:
                    place, _ = planner.place(q, held, support, yaw_deg=yaw)
                    self.log("tool", tool="placement_search", reference_id=reference_id, relation="in",
                             chosen_mm=[round(float(v)*1000, 1) for v in support], yaw_deg=yaw,
                             occupants=n_inside, tried=tried, collision_free=True)
                    return place, [support[0], support[1], floor + held.height / 2], floor + held.height
                except PlanningError as exc:
                    last = exc
                    tried.append({"xy_mm": [round(float(v)*1000, 1) for v in xy], "yaw_deg": yaw, "reason": str(exc)})
        self.log("tool", tool="placement_search", reference_id=reference_id, relation="in", ok=False,
                 candidates=available, occupants=n_inside, tried=tried)
        if not available:
            raise PlanningError(f"no free spot inside the container with the full payload and placement gap ({n_inside} object(s) already in it)")
        raise PlanningError(f"bounded collision-checked placement search exhausted ({len(tried)} attempts; {last})")

    REGION_EDGE_M = 0.010  # a spot in a fixed region keeps the placed object this far inside the outline
    MAX_REGION_SPOTS = 20  # free spots tried (nearest the region's centre first) before reporting none

    def region_spots(self, region, held: Held, g: dict) -> list:
        """Candidate centres for the held object inside a fixed planar region, nearest the region's centre first: the
        object's circumscribed circle (any gripper yaw) at least REGION_EDGE_M inside the outline, on the reliable
        reach ring, and clear of every measured object (its footprint plus the placement gap)."""
        dims = held.body_size or (held.width or 0., held.length or held.width or 0.)
        r = float(np.hypot(*dims[:2])) / 2
        lim = [e / 2 - r - self.REGION_EDGE_M for e in region.size]
        step = self.rel.search_step_m * 2
        us, vs = (step * np.arange(-np.floor(m / step), np.floor(m / step) + 1) if m >= 0 else np.array([]) for m in lim)
        spots = []
        for u in us:
            for v in vs:
                xy = region.world((u, v))
                if REACH_MIN_M <= np.hypot(*xy) <= REACH_MAX_M and not self._too_close(xy, g, exclude=None):
                    spots.append(xy)
        c = np.asarray(region.centre[:2], float)
        spots.sort(key=lambda xy: float(np.hypot(*(xy - c))))
        return spots

    def _plan_region(self, planner, q, held: Held, g: dict, relation: str, region_id: str):
        """Put the held object down INSIDE a fixed planar region of the calibrated layout (config/layout.py): on a free
        spot of the table within its outline, the centre first when it is free; each candidate's carry, lowering and
        retreat collision-checked by the planner. Only "in" is computed for a region; nothing else is substituted."""
        region = layout.region(region_id)
        if region is None:
            raise PlanningError(f"{region_id} is not a region of this work cell")
        if relation != "in":
            raise PlanningError(f"{region.id} ({region.name}) is a fixed planar region: placing {relation!r} it is not "
                                f"computed (only 'in')")
        spots = self.region_spots(region, held, g)
        tried, last = [], None
        for xy in spots[:self.MAX_REGION_SPOTS]:
            support = np.array([xy[0], xy[1], float(region.centre[2])])
            for yaw in (None, held.yaw_deg + 90.0):  # the yaw it was picked with first
                try:
                    place, _ = planner.place(q, held, support, yaw_deg=yaw)
                except PlanningError as exc:
                    last = exc
                    tried.append({"xy_mm": [round(float(v) * 1000) for v in xy], "ok": False, "reason": str(exc)[:120]})
                    continue
                self.log("tool", tool="region_place", region=region.id, name=region.name,
                         region_centre_mm=[round(float(v) * 1000) for v in region.centre[:2]],
                         candidate_mm=[round(float(v) * 1000) for v in xy], inside=True, reachable=True,
                         collision_free=True, edge_margin_mm=round(self.REGION_EDGE_M * 1000), candidates=len(spots),
                         tried=tried)
                return place, [support[0], support[1], support[2] + held.height / 2], support[2] + held.height
        self.log("tool", tool="region_place", region=region.id, name=region.name, candidates=len(spots), tried=tried, ok=False)
        if not spots:
            raise PlanningError(f"no free spot inside {region.id} ({region.name}) for the held object: none is clear of "
                                f"the objects there, {self.REGION_EDGE_M * 1000:.0f} mm inside the outline and in reach")
        raise PlanningError(f"no free, reachable, collision-free spot inside {region.id} ({region.name}) ({last})")

    def _plan_place(self, planner, q, held: Held, g: dict, relation: str, reference_id: int | None, frame: str,
                    region: str | None = None):
        if region is not None:
            return self._plan_region(planner, q, held, g, relation, region)
        if relation == "in":
            return self._plan_in(q, held, reference_id)
        if relation == "on":
            r = self.world.objects[reference_id].geometry
            if min(r["extents"]) < self.tol.min_support_width_m:
                raise PlanningError(f"the reference's top ({min(r['extents']) * 1000:.0f} mm) is too narrow to place on")
            last, tried, available = None, [], 0
            for yaw in dict.fromkeys((held.yaw_deg, held.yaw_deg + 90.)):
                spots, occupants = self._free_spots(reference_id, held, yaw_deg=yaw, inset=.004)
                available += len(spots)
                for xy in spots[:self.MAX_CONTAINER_SPOTS]:
                    support = np.array([*xy, r["top_z"]])
                    try:
                        place, _ = planner.place(q, held, support, yaw_deg=yaw)
                        self.log("tool", tool="placement_search", reference_id=reference_id, relation="on",
                                 chosen_mm=[round(float(v)*1000, 1) for v in support], yaw_deg=yaw,
                                 occupants=occupants, tried=tried, collision_free=True)
                        return place, [*xy, support[2] + held.height / 2], support[2] + held.height
                    except PlanningError as exc:
                        last = exc
                        tried.append({"xy_mm": [round(float(v)*1000, 1) for v in xy], "yaw_deg": yaw, "reason": str(exc)})
            self.log("tool", tool="placement_search", reference_id=reference_id, relation="on", ok=False,
                     candidates=available, tried=tried)
            if not available:
                raise PlanningError("no free spot on the support with the full payload and placement gap")
            raise PlanningError(f"bounded collision-checked placement search exhausted ({len(tried)} attempts; {last})")
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
                    if not REACH_MIN_M <= np.hypot(*xy) <= REACH_MAX_M:  # reliable pick/place ring (feasibility map, measured)
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

    # ARM_HOME turned about J1, out to 2 deg inside the J1 limits (+-170): a goal AT the limit can be measured a hair
    # beyond it (up to the 1.5 deg reach tolerance) and the safety layer then (correctly) rejects the next trajectory
    PARK_J1_DEG = (168.0, 135.0, -135.0, 90.0, -90.0, -168.0, 45.0, -45.0, 0.0)

    @staticmethod
    def workspace_points() -> list:
        """Floor points on the reliable reach ring (0.45-0.90 m), every 30 deg, for the first look."""
        return [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for r in (0.5, 0.7, 0.9) for a in range(0, 360, 30)]

    def view_blockage(self, q=None) -> tuple[int, int]:
        """(workspace points whose camera ray the arm blocks at joint state q (default: now), total points)."""
        q = self.bridge.q() if q is None else q
        cam = getattr(self, 'last_table_pose', table_camera_pose())[:3, 3]
        c, r, idx, names = CollisionChecker().spheres(q)
        pts = self.workspace_points()
        return self._rays_blocked(cam, c, r, pts), len(pts)

    @staticmethod
    def _rays_blocked(cam, centres, radii, points) -> int:
        """How many floor points (x, y) have their table-camera ray pass within 2 cm of the robot's collision
        spheres (centres, radii)."""
        blocked = 0
        for x, y in points:
            seg = np.array([x, y, 0.03]) - cam
            t = np.clip(((centres - cam) @ seg) / (seg @ seg), 0, 1)
            blocked += int(np.any(np.linalg.norm(cam + t[:, None] * seg - centres, axis=1) < radii + 0.02))
        return blocked

    def park_for_view(self, target_xy=None) -> dict:
        """Move the arm to a collision-free pose (ARM_HOME turned about J1) from which the table camera
        sees target_xy (must be unblocked), or, with target_xy None, the pose blocking the fewest
        workspace points. The camera ray test uses the robot's collision spheres (+2 cm)."""
        q0 = self.bridge.q()
        cam = getattr(self, 'last_table_pose', table_camera_pose())[:3, 3]
        checker = CollisionChecker(self.obstacles())
        targets = [tuple(target_xy)] if target_xy is not None else self.workspace_points()
        tried, best = [], None
        # for a single target try HOME (J1 0) first: if it sees the target, the view move IS the return HOME
        order = ((0.0,) + tuple(j for j in self.PARK_J1_DEG if j != 0.0)) if target_xy is not None else self.PARK_J1_DEG
        for j1 in order:
            q = np.asarray(ARM_HOME, float).copy()
            q[0] = np.radians(j1)
            c, r, idx, names = checker.spheres(q)
            blocked = self._rays_blocked(cam, c, r, targets)
            traj = joint_move(self.kin, q0, q, ARM_SPEED)
            if self.held is not None:  # carrying: the held object moves with the gripper (closed opening)
                check = checker.check_path(traj.positions, self.held.width or 0.01, self.held.collision_body(),
                                           {(p, f"obj_{self.held_object}") for p in [x.name for x in checker.parts] + ["palm", "pads", "linkages", "held"]})
            else:
                check = checker.check_path(traj.positions)
            zone = forbidden_on_path(traj.positions)  # Epson's J1/J2/J3 combinations, every sample
            free = check.ok and zone is None
            tried.append({"j1": j1, "blocked": blocked, "path_free": free, "collision_sample": check.index,
                          "contacts": [str(c) for c in check.contacts[:3]], "forbidden_reason": zone})
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

    @timed("planning")
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
            zone = forbidden_on_path(traj.positions)
            if not res.ok:
                out = {"ok": False, "reason": "HOME path collides: " + "; ".join(map(str, res.contacts[:3]))}
            elif zone:
                out = {"ok": False, "reason": f"HOME path: {zone}"}
            else:
                self.log("tool", tool="return_home", ok=True, planned_s=round(float(traj.duration), 1))
                r = self.execute([MoveStep(traj, "return HOME")], "return HOME")
                return {"ok": r.ok, "reason": r.reason, "result": r}
        self.log("tool", tool="return_home", ok=False, reason=out["reason"])
        return out
