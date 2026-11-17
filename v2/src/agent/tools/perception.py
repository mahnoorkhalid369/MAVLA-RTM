"""Perception tools: camera frame -> YOLO masks -> measured geometry -> world model (no language model here).

perceive (one camera frame, measured objects, set-of-mark image), update_world (persistent ids), is_fresh, at_gripper
(something measured at the fingertips while holding = the held object)."""

from __future__ import annotations

import time
from agent.timing import timed
from dataclasses import dataclass, field

import cv2
import numpy as np

from control.collision import CollisionChecker
from control.kinematics import Kinematics
from control.robot import gripper_gap, table_camera_pose
from config.agent import REACH_MAX_M
from perception.geometry import RobotVolume, in_fixtures, measure, snap_supports
from perception.somark import draw_marks
from perception.capture import ObservationError, validate_frame, surface_points, check_table_plane
from perception.instances import clean_mask, parent_of_parts, colour_signature


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
    taken_at: float = 0.0  # time.monotonic() when the frame was taken
    motion_count: int = -1  # Tools.motion_count when the frame was taken (robot not moved since if equal)
    cleared_view: bool = False  # taken after park_for_view (arm out of the camera's view)
    frame: object | None = field(default=None, repr=False)  # source RGB-D and capture joint state for local inspection
    grouped_parts: list = field(default_factory=list)


STAND_M = 0.015  # above the table top: a point higher than this stands on the table (the mat and markings are lower)
CELL_M = 0.03  # unexplained points are grouped in square columns of this size
MIN_POINTS = 6  # per column (depth sampled every 3rd pixel), so single noisy pixels are no obstacle
MARGIN_M = 0.025  # around a measured object's outline: its own side faces and edges


def unexplained(frame, T, robot, measured: list[dict], tcp=None, stride: int = 3) -> list[dict]:
    # Depth points standing on the table that are not the robot, a fixture, a measured object or (while holding) the
    # held object at the gripper: something the detector did not outline (live 2026-11-10: a green cube on the dark
    # mat). Grouped into columns from the table top to their highest point, in the measured objects' geometry format.
    d = frame.depth[::stride, ::stride].astype(np.float64)
    v, u = np.mgrid[0:frame.depth.shape[0]:stride, 0:frame.depth.shape[1]:stride]
    ok = np.isfinite(d) & (d > 0.05) & (d < 6.0)
    fx, fy, cx, cy = frame.K
    P = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=-1)[ok] @ T[:3, :3].T + T[:3, 3]
    P = P[(P[:, 2] > STAND_M) & (P[:, 2] < 0.40) & (np.hypot(P[:, 0], P[:, 1]) < REACH_MAX_M + 0.15)]
    for g in measured:
        a = np.radians(g.get("yaw_deg") or 0.0)
        dx, dy = P[:, 0] - g["centre"][0], P[:, 1] - g["centre"][1]
        lx, ly = np.cos(a) * dx + np.sin(a) * dy, -np.sin(a) * dx + np.cos(a) * dy
        ex = g.get("extents") or [max(g["footprint"]), min(g["footprint"])]
        P = P[~((np.abs(lx) <= ex[0] / 2 + MARGIN_M) & (np.abs(ly) <= ex[1] / 2 + MARGIN_M)
                & (P[:, 2] <= g["top_z"] + MARGIN_M))]
    P = P[~in_fixtures(P)]
    P = P[~robot.contains(P)] if len(P) else P
    if tcp is not None:
        P = P[~((np.hypot(P[:, 0] - tcp[0], P[:, 1] - tcp[1]) < 0.08) & (P[:, 2] - tcp[2] > -0.15))]
    out = []
    if len(P):
        cells, inv, counts = np.unique(np.floor(P[:, :2] / CELL_M).astype(int), axis=0, return_inverse=True,
                                       return_counts=True)
        for i, (c, n) in enumerate(zip(cells, counts)):
            if n < MIN_POINTS:
                continue
            top = float(P[inv.ravel() == i, 2].max())
            xy = (c + 0.5) * CELL_M
            out.append({"centre": [float(xy[0]), float(xy[1]), top / 2], "extents": [CELL_M, CELL_M],
                        "footprint": [CELL_M, CELL_M], "height": top, "top_z": top, "support_z": 0.0, "yaw_deg": 0.0})
    return out


class PerceptionTools:
    def _camera_pose(self, frame) -> np.ndarray:
        if getattr(frame, 'T_world_camera', None) is not None:
            return np.asarray(frame.T_world_camera, float).copy()
        return table_camera_pose() if getattr(frame, 'camera', 'table') == "table" else self.wrist_kin.fk(frame.q)

    def _check_camera_pose(self, frame, pose):
        validate_frame(frame, pose)
        if frame.camera == 'wrist' and getattr(frame, 'T_world_camera', None) is not None:
            expected = self.wrist_kin.fk(frame.q)
            angle = np.degrees(np.arccos(np.clip((np.trace(expected[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
            if np.linalg.norm(expected[:3, 3] - pose[:3, 3]) > .015 or angle > 2.:
                raise ObservationError('published wrist pose disagrees with captured robot joints/mount calibration')

    MAX_OBS_AGE_S = 120.0

    def is_fresh(self, obs: Observation | None, need_clear_view: bool = False) -> bool:
        """An observation still describes the scene: the robot has not moved since it was taken, it is younger
        than MAX_OBS_AGE_S and (if required) it was taken with the arm parked out of the camera's view."""
        frame = getattr(obs, 'frame', None)
        still = (frame is None or self.bridge is None or
                 np.max(np.abs(self.bridge.q() - frame.q)) <= np.radians(1.5))
        return (obs is not None and still and obs.motion_count == self.motion_count
                and time.monotonic() - obs.taken_at < self.MAX_OBS_AGE_S and (obs.cleared_view or not need_clear_view))

    @timed("perception")
    def perceive(self, camera: str = "table") -> Observation:
        frame = self.bridge.snapshot(camera)
        if frame.camera != camera:
            raise ObservationError(f'requested {camera} camera but received {frame.camera} capture')
        if hasattr(self, '_sensor_epoch') and self._sensor_epoch != getattr(frame, 'epoch', 0):
            raise ObservationError('simulator clock reset; restart the session before reusing object or held-state records')
        self._sensor_epoch = getattr(frame, 'epoch', 0)
        T = self._camera_pose(frame)
        self._frames += 1
        fid = f"{camera}_{self._frames:03d}"
        directory = self.run_dir / 'images'
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f'{fid}_depth.npz', depth=frame.depth, K=np.asarray(frame.K),
                            T_world_cam=T, q=frame.q, finger=frame.finger, stamp=frame.stamp)
        cv2.imwrite(str(directory / f'{fid}_rgb.png'), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        self.log('sensor_bundle', frame_id=fid, camera=camera, stamp=frame.stamp,
                 depth_stamp=getattr(frame, 'depth_stamp', None), pose_stamp=getattr(frame, 'pose_stamp', None),
                 joint_stamp=getattr(frame, 'joint_stamp', None), K=list(frame.K), T_world_camera=T.tolist(),
                 q=np.asarray(frame.q).tolist(), pose_source='published' if getattr(frame, 'T_world_camera', None) is not None else 'configured_replay')
        self._check_camera_pose(frame, T)
        checker = CollisionChecker()
        tool = Kinematics(tip="tool0").fk(frame.q)
        robot = RobotVolume(checker, frame.q, gripper_gap(frame.finger), self.held_box, tool)
        horizontal = surface_points(frame, T)
        horizontal = horizontal[~robot.contains(horizontal)]
        if camera == 'table':
            table_z = check_table_plane(horizontal)
            self.last_table_pose = T.copy()
            self.log('calibration', frame_id=fid, table_plane_mm=round(table_z * 1000, 2), ok=True)
        dets = self.detector.detect(frame.rgb)
        marks, measured, excluded, scores = {}, {}, [], {}
        for d in sorted(dets, key=lambda d: -d.score):
            mask, fraction = clean_mask(d.mask, frame, T, robot)
            g, why = (None, f'mostly robot ({fraction:.0%})') if fraction > .5 else measure(mask, frame.depth, frame.K, T, robot)
            if g is None:
                excluded.append({"score": round(d.score, 3), "pixels": int(d.mask.sum()), "reason": why})
                continue
            k = len(marks) + 1
            marks[k], measured[k], scores[k] = mask, g.to_dict(), round(float(d.score), 3)
        parents = parent_of_parts(marks, measured)
        grouped = [{'part': k, 'body': j} for k, j in parents.items()]
        for k, j in parents.items():
            marks[j] |= marks[k]
        for k in parents:
            del marks[k], measured[k], scores[k]
        for j in set(parents.values()):
            g, why = measure(marks[j], frame.depth, frame.K, T, robot)
            if g is not None:
                measured[j] = g.to_dict()
        order = sorted(marks)
        marks, measured, scores = ({i + 1: data[k] for i, k in enumerate(order)} for data in (marks, measured, scores))
        if self.held_box is not None:  # the held object is not on the table: drop its measurement
            Th, size = self.held_box
            Tw = tool @ Th
            half = np.asarray(size) / 2 + 0.02
            tcp = self.kin.fk(frame.q)[:3, 3]
            for k in [k for k, g in measured.items() if np.all(np.abs((np.asarray(g["centre"]) - Tw[:3, 3]) @ Tw[:3, :3]) <= half)
                      or self.at_gripper(g, tcp)]:
                excluded.append({"score": scores[k], "pixels": int(marks[k].sum()), "reason": "the held object"})
                del marks[k], measured[k], scores[k]
            order = sorted(marks)  # renumber 1..n consistently
            marks, measured, scores = ({i + 1: d[k] for i, k in enumerate(order)} for d in (marks, measured, scores))
        # A tray's interior floor is part of the tray, not what the tray itself stands on.
        # Exclude each object's own measured pixels when looking for its supporting surface.
        pc = (horizontal - T[:3, 3]) @ T[:3, :3]
        fx, fy, cx, cy = frame.K
        u = np.clip(np.rint(fx * pc[:, 0] / pc[:, 2] + cx).astype(int), 0, frame.depth.shape[1] - 1)
        v = np.clip(np.rint(fy * pc[:, 1] / pc[:, 2] + cy).astype(int), 0, frame.depth.shape[0] - 1)
        snap_supports(list(measured.values()), horizontal, [marks[k][v, u] for k in measured])
        for k, geometry in measured.items():
            geometry['colour_signature'] = colour_signature(frame.rgb, marks[k])
        if camera == "table":  # what depth sees standing on the table that no measured object explains: obstacles too
            self.inspection_obstacles = []
            tcp = self.kin.fk(frame.q)[:3, 3] if self.held_object is not None else None
            self.unexplained = dict(enumerate(unexplained(frame, T, robot, list(measured.values()), tcp)))
        som = draw_marks(frame.rgb, marks) if marks else frame.rgb.copy()
        cv2.imwrite(str(self.run_dir / "images" / f"{fid}_som.jpg"), cv2.cvtColor(som, cv2.COLOR_RGB2BGR))
        self.log("tool", tool="perceive", camera=camera, frame_id=fid, detections=len(dets), objects=len(marks),
                 excluded=excluded, grouped_parts=grouped, measured={k: {kk: v[kk] for kk in ("centre", "top_z", "height", "support_z", "support_confident", "visible_bottom_z", "footprint", "grasp_yaw_deg", "grasp_width", "top_planar", "tilt_deg", "top_plane_fraction")}
                                              for k, v in measured.items()},
                 unexplained=[[round(c, 3) for c in g["centre"][:2]] + [round(g["top_z"], 3)]
                              for g in self.unexplained.values()] if camera == "table" else None)
        obs = Observation(fid, camera, frame.rgb, som, marks, measured, excluded=excluded, scores=scores,
                          taken_at=time.monotonic(), motion_count=self.motion_count,
                          cleared_view=self.view_cleared_at == self.motion_count, frame=frame, grouped_parts=grouped)
        if self.live:
            self.live.perception(obs)
        return obs

    @staticmethod
    def at_gripper(g: dict, tcp) -> bool:
        # off the table and around the fingertips: the held object, often outlined together with the fingers (live
        # 2026-11-10: fingers + green cube measured as one 65 x 17 mm object 20 cm up, then kept as an obstacle)
        return (np.hypot(g["centre"][0] - tcp[0], g["centre"][1] - tcp[1]) < 0.08
                and -0.15 < g["centre"][2] - tcp[2] < 0.08 and g.get("support_z", 0.0) > 0.02)

    def update_world(self, obs: Observation) -> list[dict]:
        """Store every measured object; returns neighbours that moved (whole-scene re-perception)."""
        before = {oid: list(o.geometry["centre"]) for oid, o in self.world.objects.items()}
        if obs.camera == "table":  # the whole frame at once, by outline first: ids stay the same between looks
            obs.object_ids.update(self.world.observe_frame(obs.measured, obs.marks, obs.descriptions))
        else:
            for k, g in obs.measured.items():
                obs.object_ids[k] = self.world.observe(g, obs.descriptions.get(k), others=list(obs.measured.values()))
        for k in obs.measured:
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
            self.scene_revision = getattr(self, "scene_revision", 0) + 1
            self.log("world", event="disturbed", moved=moved)
        if self.live:
            self.live.perception(obs)
        return moved
