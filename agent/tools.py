"""Agent tools: the only way the agents touch the world. Deterministic except the VLM roles.

perceive -> describe -> resolve / confirm (Perception); plan_pick / plan_place (Grasp + Motion
planner: SkillPlanner + CollisionChecker with obstacles from MEASURED geometry); execute
(RosExecutor); verify_contact / verify_placement (sensor-only); park_for_view (arm out of the table
camera's line of sight before re-observing). No ground truth, no object knowledge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from agent.config import FIXTURES, Relations, Tolerances, unit
from agent.geometry import RobotVolume, measure, snap_supports
from agent.somark import crop, draw_marks, draw_names
from agent.world import WorldModel
from control.collision import Box, CollisionChecker
from control.kinematics import Kinematics
from control.motion import PlanningError, joint_move
from control.robot import ARM_HOME, ARM_SPEED, gripper_gap, table_camera_pose
from control.skills import Held, SkillPlanner


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
    def __init__(self, bridge, detector, roles, executor, log, run_dir) -> None:
        self.bridge, self.detector, self.roles, self.ex, self.log, self.run_dir = bridge, detector, roles, executor, log, run_dir
        self.world = WorldModel()
        self.kin = Kinematics()
        self.wrist_kin = Kinematics(tip="camera_color_optical_frame")
        self.tol, self.rel = Tolerances(), Relations()
        self.held_box = None  # (pose in tool0, size) while holding, from measured geometry
        self._frames = 0

    # ------------------------------------------------------------------ perception
    def _camera_pose(self, frame) -> np.ndarray:
        return table_camera_pose() if frame.camera == "table" else self.wrist_kin.fk(frame.q)

    def perceive(self, camera: str = "table") -> Observation:
        frame = self.bridge.snapshot(camera)
        T = self._camera_pose(frame)
        dets = self.detector.detect(frame.rgb)
        checker = CollisionChecker()
        tool = Kinematics(tip="tool0").fk(frame.q)
        robot = RobotVolume(checker, frame.q, gripper_gap(frame.finger), self.held_box, tool)
        marks, measured, excluded = {}, {}, []
        for d in sorted(dets, key=lambda d: -d.score):
            g, why = measure(d.mask, frame.depth, frame.K, T, robot)
            if g is None:
                excluded.append({"score": round(d.score, 3), "pixels": int(d.mask.sum()), "reason": why})
                continue
            k = len(marks) + 1
            marks[k], measured[k] = d.mask, g.to_dict()
        snap_supports(list(measured.values()))
        self._frames += 1
        fid = f"{camera}_{self._frames:03d}"
        som = draw_marks(frame.rgb, marks) if marks else frame.rgb.copy()
        cv2.imwrite(str(self.run_dir / "images" / f"{fid}_som.jpg"), cv2.cvtColor(som, cv2.COLOR_RGB2BGR))
        self.log("tool", tool="perceive", camera=camera, frame_id=fid, detections=len(dets), objects=len(marks),
                 excluded=excluded, measured={k: {kk: v[kk] for kk in ("centre", "top_z", "height", "footprint", "grasp_yaw_deg", "grasp_width")}
                                              for k, v in measured.items()})
        return Observation(fid, camera, frame.rgb, som, marks, measured, excluded=excluded)

    def describe(self, obs: Observation) -> None:
        if not obs.marks:
            return
        d = self.roles.describe(obs.som, list(obs.marks))
        obs.descriptions = {o.id: o.model_dump(exclude={"id"}) for o in d.objects}
        img = draw_names(obs.som, obs.marks, {k: v.get("name", "") for k, v in obs.descriptions.items()})
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_names.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def update_world(self, obs: Observation) -> list[dict]:
        """Store every measured object; returns neighbours that moved (whole-scene re-perception)."""
        before = {oid: list(o.geometry["centre"]) for oid, o in self.world.objects.items()}
        for k, g in obs.measured.items():
            obs.object_ids[k] = self.world.observe(g, obs.descriptions.get(k))
        moved = []
        for oid, c in before.items():
            now = self.world.objects[oid].geometry["centre"]
            if oid in obs.object_ids.values() and np.hypot(now[0] - c[0], now[1] - c[1]) > self.tol.xy_m:
                moved.append({"id": oid, "from_mm": [round(v * 1000) for v in c], "to_mm": [round(v * 1000) for v in now]})
        if moved:
            self.log("world", event="disturbed", moved=moved)
        return moved

    def resolve(self, obs: Observation, text: str):
        descs = [{"id": k, **obs.descriptions.get(k, {})} for k in obs.marks]
        return self.roles.resolve(obs.som, descs, text, list(obs.marks))

    def confirm(self, obs: Observation, mark: int, text: str):
        c = crop(obs.rgb, obs.marks[mark])
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

    PARK_J1_DEG = (170.0, 135.0, -135.0, 90.0, -90.0, -170.0, 45.0, -45.0, 0.0)  # ARM_HOME turned about J1

    @staticmethod
    def workspace_points() -> list:
        """Floor points on the reliable reach ring (0.45-0.90 m), every 30 deg, for the first look."""
        return [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for r in (0.5, 0.7, 0.9) for a in range(0, 360, 30)]

    def park_for_view(self, target_xy=None) -> dict:
        """Move the arm to a collision-free pose (ARM_HOME turned about J1) from which the table camera
        sees target_xy (must be unblocked), or, with target_xy None, the pose blocking the fewest
        workspace points. The camera ray test uses the robot's collision spheres (+2 cm)."""
        q0 = self.bridge.q()
        cam = table_camera_pose()[:3, 3]
        checker = CollisionChecker(self.obstacles())
        targets = [tuple(target_xy)] if target_xy is not None else self.workspace_points()
        tried, best = [], None
        for j1 in self.PARK_J1_DEG:
            q = np.asarray(ARM_HOME, float).copy()
            q[0] = np.radians(j1)
            c, r, idx, names = checker.spheres(q)
            blocked = 0
            for x, y in targets:
                seg = np.array([x, y, 0.03]) - cam
                t = np.clip(((c - cam) @ seg) / (seg @ seg), 0, 1)
                blocked += int(np.any(np.linalg.norm(cam + t[:, None] * seg - c, axis=1) < r + 0.02))
            traj = joint_move(self.kin, q0, q, ARM_SPEED)
            free = checker.check_path(traj.positions).ok
            tried.append({"j1": j1, "blocked": blocked, "path_free": free})
            if free and (best is None or blocked < best[0]):
                best = (blocked, j1, traj)
        if best is None or (target_xy is not None and best[0] > 0):
            self.log("tool", tool="park_for_view", ok=False, tried=tried)
            return {"ok": False, "tried": tried}
        from control.skills import MoveStep
        blocked, j1, traj = best
        res = self.execute([MoveStep(traj, f"park J1 {j1:+.0f} (view blocked for {blocked}/{len(targets)} points)")], "park")
        return {"ok": res.ok, "j1": j1, "blocked": blocked, "tried": tried}

    def assume_placed(self, plan: PickPlan) -> None:
        """Before re-observing: count the released object as an obstacle at its PLANNED pose (it is no
        longer in the gripper; its measured pose follows from the re-observation)."""
        o = self.world.objects[plan.object_id]
        g = dict(o.geometry)
        g["centre"] = list(plan.target_centre)
        g["support_z"] = plan.target_top - plan.held.height
        g["top_z"] = plan.target_top
        o.geometry, o.status = g, "on_surface"
        self.held_box, self.world.holding = None, None

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
        self.describe(obs)
        moved = self.update_world(obs)
        v["disturbed_neighbours"] = [m for m in moved if m["id"] != plan.object_id]
        self.log("verification", **v)
        return v
