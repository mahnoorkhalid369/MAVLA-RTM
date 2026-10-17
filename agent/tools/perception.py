"""Perception tools: camera frame -> YOLO masks -> measured geometry -> Gemma descriptions -> world model.

perceive (one camera frame, measured objects, set-of-mark image), describe (Gemma's words per object crop, reused
from the world model / description memory when unchanged), update_world (persistent ids), is_fresh."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from agent.tools.memory import mean_rgb
from control.collision import CollisionChecker
from control.kinematics import Kinematics
from control.robot import gripper_gap, table_camera_pose
from perception.geometry import RobotVolume, measure, snap_supports
from perception.somark import draw_marks, draw_names, grid, isolate


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


class PerceptionTools:
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
