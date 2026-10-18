"""Exploration agent: choose a wrist-camera viewpoint that gives a closer / unobstructed look at a target point,
validate it geometrically and for safety, and move there through the validated execution layer.

A viewpoint is a camera POSITION + LOOK-AT point. The wrist camera is the URDF frame
`camera_color_optical_frame` on the J5 housing (validated against simulator ground truth, Checkpoint 2):
its optical axis is NOT the gripper axis, so every candidate is solved for the camera frame itself.

Candidate generation is deterministic (no random search): camera positions on spheres around the target
(distance x elevation x azimuth). Each candidate must pass, in order:
  1. look-at IK for the camera frame converges (position <= 5 mm, axis <= 2 deg), joints >= 5 deg inside limits
  2. the camera ray to the target is clear of every measured object box and of the robot itself
  3. the target lies within the image and at a usable depth (>= 0.20 m)
  4. the joint path from the current state is collision-free (CollisionChecker.check_path, measured obstacles)
Survivors are scored: distance error from the preferred working distance + joint travel; the expected gain
is reported as the pixel-area ratio against the table camera ((d_table / d_wrist)^2).
Execution: one MoveStep through Tools.execute (RosExecutor + simulator safety layer); reach is verified from
joint feedback (every joint within 1.5 deg, the executor's validated reach tolerance) and camera-pose FK.
Nothing here accepts a pose or joint value from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.agent import REACH_TOLERANCE_DEG
from control.collision import CollisionChecker
from control.kinematics import Kinematics
from control.motion import joint_move
from control.robot import ARM_HOME, ARM_SPEED, table_camera_pose

CAM_TIP = "camera_color_optical_frame"
PREFERRED_DISTANCE = 0.35  # m: D435 colour + depth both sharp; ~4x the table camera's pixels on the object
DISTANCES = (0.35, 0.45)
ELEVATIONS_DEG = (60.0, 40.0)  # camera above the target, from the horizontal
AZIMUTH_STEP_DEG = 45.0
MIN_DEPTH = 0.20
LIMIT_MARGIN = np.radians(5.0)
REACH_TOL_DEG = REACH_TOLERANCE_DEG  # config/agent.py: the executor's validated reach tolerance
MAX_CANDIDATES_EXECUTED = 2
VIEW_PAD = 0.010  # m: line-of-sight clearance from measured objects
REGION_RADIUS = 0.03  # m: a hidden region is seen only if points this far around its centre are all visible
REGION_HEIGHT = 0.04  # m: ... at the floor and this high


@dataclass
class Viewpoint:
    q: np.ndarray
    camera_pos: np.ndarray
    target: np.ndarray
    distance: float
    elevation_deg: float
    azimuth_deg: float
    travel_rad: float
    gain: float  # expected pixel-area ratio vs the table camera
    cost: float = 0.0


@dataclass
class ViewPlan:
    ok: bool
    best: Viewpoint | None = None
    evaluated: int = 0
    rejected: dict = field(default_factory=dict)  # reason -> count
    reason: str = ""

    def summary(self) -> dict:
        b = self.best
        return {"ok": self.ok, "evaluated": self.evaluated, "rejected": self.rejected, "reason": self.reason,
                "best": None if b is None else {"camera_pos_mm": [round(v * 1000) for v in b.camera_pos],
                                                "distance_m": b.distance, "elevation_deg": b.elevation_deg,
                                                "azimuth_deg": b.azimuth_deg, "travel_deg": round(np.degrees(b.travel_rad), 1),
                                                "expected_gain": round(b.gain, 1)}}


class ViewpointPlanner:
    def __init__(self, kin_cam: Kinematics | None = None) -> None:
        self.kin = kin_cam or Kinematics(tip=CAM_TIP)  # chain base -> J5 -> camera (J6 does not move the camera)
        self.n = len(self.kin.lower)
        full = Kinematics()
        self.lo, self.hi = full.lower + LIMIT_MARGIN, full.upper - LIMIT_MARGIN

    # ------------------------------------------------------------------ look-at IK for the camera frame
    def look_at_ik(self, cam_pos: np.ndarray, target: np.ndarray, seed: np.ndarray, iters: int = 150):
        """Full joint state (J6 kept from the seed) putting the camera at cam_pos with its optical axis (+z)
        through target. Damped least squares over the camera chain's joints on position (3) + axis direction
        (cross product, rank 2). Returns (q, ok)."""
        seed = np.asarray(seed, float)
        q = seed[:self.n].copy()
        d = (target - cam_pos) / np.linalg.norm(target - cam_pos)

        def err(qq):
            T = self.kin.fk(qq)
            return np.concatenate([cam_pos - T[:3, 3], np.cross(T[:3, 2], d)])
        for _ in range(iters):
            e = err(q)
            if np.linalg.norm(e[:3]) < 0.005 and np.linalg.norm(e[3:]) < np.sin(np.radians(2.0)):
                break
            J = np.zeros((6, self.n))
            for j in range(self.n):
                dq = np.zeros(self.n)
                dq[j] = 1e-4
                J[:, j] = (e - err(q + dq)) / 1e-4  # = d(achieved)/dq
            step = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q = q + np.clip(step, -0.15, 0.15)
        T = self.kin.fk(q)
        ok = (np.linalg.norm(cam_pos - T[:3, 3]) < 0.005 and np.degrees(np.arccos(np.clip(T[:3, 2] @ d, -1, 1))) < 2.0)
        return np.concatenate([q, seed[self.n:]]), bool(ok)

    # ---------------------------------------------------------------------------- candidate evaluation
    @staticmethod
    def _segment_hits_box(p0, p1, box, pad=0.005) -> bool:
        R, c, h = box.R, box.centre, box.half + pad
        a, b = R.T @ (p0 - c), R.T @ (p1 - c)
        t0, t1 = 0.0, 1.0
        d = b - a
        for i in range(3):
            if abs(d[i]) < 1e-12:
                if abs(a[i]) > h[i]:
                    return False
                continue
            u, v = (-h[i] - a[i]) / d[i], (h[i] - a[i]) / d[i]
            t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
            if t0 > t1:
                return False
        return True

    def plan(self, target: np.ndarray, q_now: np.ndarray, obstacles: list, target_box_name: str | None = None,
             distances=DISTANCES, elevations=ELEVATIONS_DEG, azimuths=None, region_radius: float = 0.0,
             region_height: float = 0.0) -> ViewPlan:
        """All candidates around `target` (world point to look at); the best safe one (or a reason)."""
        checker = CollisionChecker(obstacles)
        others = [b for b in obstacles if b.name != target_box_name]
        kin_tcp = Kinematics()
        cam_table = table_camera_pose()[:3, 3]
        d_table = float(np.linalg.norm(target - cam_table))
        azimuths = azimuths if azimuths is not None else np.arange(0.0, 360.0, AZIMUTH_STEP_DEG)
        rej: dict = {}
        n = 0

        def reject(why):
            rej[why] = rej.get(why, 0) + 1
        seeds = [np.asarray(q_now, float), np.asarray(ARM_HOME, float)]
        # pass 1 (cheap): look-at IK + joint limits for every candidate, cost known
        cands = []
        for dist in distances:
            for el in elevations:
                for az in azimuths:
                    n += 1
                    e, a = np.radians(el), np.radians(az)
                    cam = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                    if cam[2] < 0.15 or np.hypot(*cam[:2]) < 0.20:
                        reject("camera too low / too close to the base")
                        continue
                    seed = seeds + [np.array([np.arctan2(-cam[0], cam[1]), *ARM_HOME[1:5], q_now[5]])]
                    sol = None
                    for s0 in seed:
                        q, ok = self.look_at_ik(cam, target, s0)
                        if ok:
                            sol = q
                            break
                    if sol is None:
                        reject("unreachable (camera look-at IK)")
                        continue
                    if np.any(sol < self.lo) or np.any(sol > self.hi):
                        reject("joint limits (5 deg margin)")
                        continue
                    travel = float(np.abs(sol - q_now).sum())
                    cands.append(Viewpoint(sol, cam, target, dist, el, float(az), travel, (d_table / dist) ** 2,
                                           cost=abs(dist - PREFERRED_DISTANCE) * 10 + travel))
        # pass 2 (expensive), in cost order, first that passes everything wins
        cands.sort(key=lambda v: v.cost)
        for v in cands:
            cam, sol = v.camera_pos, v.q
            # the whole region must be in clear view: rays to points around the look-at point (+-region_radius,
            # two heights) clear every measured object by VIEW_PAD (a grazing view past an edge is not enough)
            pts = [target + [dx, dy, dz] for dx in (-region_radius, 0, region_radius) for dy in (-region_radius, 0, region_radius)
                   for dz in (0.0, region_height)] if region_radius > 0 else [target]
            if any(self._segment_hits_box(cam, p, b, pad=VIEW_PAD) for p in pts for b in others):
                reject("view blocked by a measured object")
                continue
            c, r, idx, names = checker.spheres(sol)
            body = np.array([names[i] != "camera_link" for i in idx])
            seg = target - cam
            tt = np.clip(((c[body] - cam) @ seg) / (seg @ seg), 0.05, 0.95)
            if np.any(np.linalg.norm(cam + tt[:, None] * seg - c[body], axis=1) < r[body]):
                reject("view blocked by the robot itself")
                continue
            if checker.check(sol):
                reject("pose in collision")
                continue
            if not checker.check_path(joint_move(kin_tcp, np.asarray(q_now, float), sol, ARM_SPEED).positions).ok:
                reject("path to the pose collides")
                continue
            return ViewPlan(True, v, n, rej)
        why = max(rej, key=rej.get) if rej else "no candidates"
        return ViewPlan(False, None, n, rej, f"no safe viewpoint: most candidates failed on '{why}'")

    def reached(self, q_meas: np.ndarray, vp: Viewpoint) -> dict:
        err = np.degrees(np.abs(np.asarray(q_meas) - vp.q))
        T = self.kin.fk(np.asarray(q_meas)[:self.n])
        d = (vp.target - T[:3, 3]) / np.linalg.norm(vp.target - T[:3, 3])
        return {"max_joint_error_deg": round(float(err.max()), 3), "camera_pos_error_mm": round(float(np.linalg.norm(T[:3, 3] - vp.camera_pos)) * 1000, 1),
                "axis_error_deg": round(float(np.degrees(np.arccos(np.clip(T[:3, 2] @ d, -1, 1)))), 2),
                "pass": bool(err.max() <= REACH_TOL_DEG)}


def shadow_points(objects: list[dict], n_max: int = 3) -> list[tuple[np.ndarray, int]]:
    """Floor points the TABLE camera cannot see: just behind each measured object along the camera's line of
    sight (shadow length = height x horizontal distance / camera height). Largest shadows first.
    Returns [(point, object id)]."""
    cam = table_camera_pose()[:3, 3]
    pts = []
    for o in objects:
        g = o["geometry"]
        c = np.array(g["centre"])
        horiz = c[:2] - cam[:2]
        dist = float(np.linalg.norm(horiz))
        length = g["top_z"] * dist / max(cam[2] - g["top_z"], 0.1)
        u = horiz / dist
        p = np.array([*(c[:2] + u * (max(g["footprint"]) / 2 + length / 2)), 0.02])
        pts.append((length, p, o["id"]))
    pts.sort(key=lambda x: -x[0])
    return [(p, i) for _, p, i in pts[:n_max]]
