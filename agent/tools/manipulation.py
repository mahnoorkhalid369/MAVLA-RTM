"""Manipulation planning on measured geometry: grasp / place plans (frozen SkillPlanner + CollisionChecker with
obstacles from the world model), parking the arm out of the table camera's view, returning HOME. Every motion
is executed through Tools.execute (agent/tools/execution.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.agent import REACH_MAX_M, REACH_MIN_M
from config.perception import FIXTURES
from control.collision import Box, CollisionChecker
from control.motion import PlanningError, joint_move
from control.robot import ARM_HOME, ARM_SPEED, table_camera_pose
from control.skills import Held, SkillPlanner


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


class ManipulationTools:
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

    def _plan_grasp(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float):
        """The grasp part shared by plan_pick and plan_pick_place: hover, descend + close, lift, on the measured
        geometry (its top face must be visible). Returns (planner, geometry, hover, approach, lift, q, held, grasp)."""
        g = self.world.objects[object_id].geometry
        if g["tilt_deg"] is None:
            raise PlanningError(f"object {object_id}: its top face is not visible, so its pose is not measured well enough")
        planner = SkillPlanner(self.kin, collision=CollisionChecker(self.obstacles()))
        width = g["grasp_width"]
        # extent along the fingers' faces = the footprint side that is not the grasp width
        fp = g["footprint"]
        length = fp[0] if abs(fp[1] - width) <= abs(fp[0] - width) else fp[1]
        planner.check_width(width)
        top = np.array([g["centre"][0], g["centre"][1], g["top_z"]])
        hover, q, yaw = planner.hover(q_start, top, g["grasp_yaw_deg"] + yaw_offset_deg, g["symmetry_deg"])
        approach, lift, q, held = planner.grasp(q, top, yaw, f"obj_{object_id}", width, length, self._support_name(g), g["height"])
        grasp = {"yaw_deg": round(yaw, 1), "width_mm": round(width * 1000, 1), "height_mm": round(g["height"] * 1000, 1)}
        return planner, g, hover, approach, lift, q, held, grasp

    def plan_pick_place(self, object_id: int, relation: str, reference_id: int | None, frame: str,
                        q_start: np.ndarray, yaw_offset_deg: float = 0.0) -> PickPlan:
        for oid in (object_id, reference_id if relation == "on" else None):
            if oid is not None and self.world.objects[oid].geometry["tilt_deg"] is None:
                raise PlanningError(f"object {oid}: its top face is not visible, so its pose is not measured well enough")
        planner, g, hover, approach, lift, q, held, grasp = self._plan_grasp(object_id, q_start, yaw_offset_deg)
        place, target_centre, target_top = self._plan_place(planner, q, held, g, relation, reference_id, frame)
        return PickPlan(object_id, hover, approach, lift, place, held, grasp,
                        list(map(float, target_centre)), float(target_top), relation, reference_id)

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
                     "only ", "mm of the pads")  # PlanningError texts that re-observing cannot change

    @classmethod
    def deterministic(cls, reason: str) -> bool:
        return any(k in reason for k in cls.DETERMINISTIC)

    def plan_pick(self, object_id: int, q_start: np.ndarray, yaw_offset_deg: float = 0.0) -> PickPlan:
        """Grasp-only plan (hover, descend + close, lift) with the same checks as plan_pick_place."""
        _, g, hover, approach, lift, _, held, grasp = self._plan_grasp(object_id, q_start, yaw_offset_deg)
        return PickPlan(object_id, hover, approach, lift, [], held, grasp,
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
            blocked = self._rays_blocked(cam, c, r, targets)
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
