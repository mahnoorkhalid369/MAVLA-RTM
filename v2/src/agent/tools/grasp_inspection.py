"""Local RGB-D evidence before descent. Missing or ambiguous evidence stops the grasp."""

from __future__ import annotations

import cv2
import numpy as np
import time
from copy import copy, deepcopy

from control.collision import Box, CollisionChecker
from control.kinematics import Kinematics
from control.robot import gripper_gap
from config.agent import REACH_TOLERANCE_DEG
from perception.geometry import RobotVolume
from agent.tools.perception import unexplained
from perception.somark import isolate
from agent.contracts import sensor_epoch
from perception.source_evidence import foreground, crop_quality


class GraspInspectionTools:
    SOURCE_SIDE_DISTANCES = (0.35, 0.45)
    SOURCE_SIDE_ELEVATIONS_DEG = (25.0, 35.0)

    def inspect_grasp(self, object_id: int, *, identity_evidence=None) -> dict:
        # The side view may reveal hazards hidden from the subsequent top view.
        if identity_evidence is None:
            self.inspection_obstacles = []
        try:
            obs = self.perceive("wrist")  # snapshot enforces settling
        except Exception as exc:
            return self._inspection_result(False, object_id, reason=f"wrist RGB-D unavailable: {exc}")
        if not self.is_fresh(obs) or obs.frame is None:
            return self._inspection_result(False, object_id, frame_id=obs.frame_id, reason="wrist frame is stale or lacks depth")
        if np.degrees(np.max(np.abs(np.array(self.bridge.q()) - obs.frame.q))) > REACH_TOLERANCE_DEG:
            return self._inspection_result(False, object_id, frame_id=obs.frame_id,
                                           reason="joint feedback changed since the wrist exposure")
        target = self.world.objects[object_id].geometry
        rgb = obs.rgb if hasattr(obs, "rgb") else obs.frame.rgb
        self._source_camera = (obs.frame.K, rgb.shape)
        candidates = []
        associations = {}
        for mark, g in obs.measured.items():
            status, oid, alternatives = self.world.associate(g)
            associations[mark] = {"status": status, "object_id": oid, "candidates": alternatives}
            if status == "matched" and oid == object_id:
                candidates.append((mark, g))
        if len(candidates) != 1:
            return self._inspection_result(False, object_id, frame_id=obs.frame_id,
                                           reason="the wrist view does not uniquely measure the selected object",
                                           matched_marks=[k for k, _ in candidates], associations=associations)
        mark, g = candidates[0]
        # Retain newly seen neighbours/depth even when the identity vote rejects the source:
        # the side recovery and an abort-to-HOME must also avoid these obstacles.
        self._add_inspection_obstacles(obs, mark, target)
        if identity_evidence is not None:
            # Only the tool's short-lived, same-object side observation can replace the top identity vote.
            # The fresh top RGB-D geometry and collision checks below are still mandatory.
            if not self._valid_source_identity(identity_evidence, object_id, obs):
                return self._inspection_result(False, object_id, frame_id=obs.frame_id,
                                               reason="side-view identity evidence expired or its object/scene changed")
            identity = identity_evidence["identity"]
            self.log("source_identity_reused", object_id=object_id, frame_id=obs.frame_id,
                     identity_frame=identity["frame_id"], mark=mark)
        else:
            if hasattr(obs, "marks"):
                quality = crop_quality(obs.marks[mark], obs.rgb)
                self.log("source_crop_quality", object_id=object_id, frame_id=obs.frame_id, mark=mark, **quality)
                if not quality["ok"]:
                    return self._inspection_result(False, object_id, frame_id=obs.frame_id, mark=mark,
                                                   identity_mismatch=True, quality=quality,
                                                   reason="source identity unobservable in the top crop: " + quality["reason"])
            identity = self.identity_check(object_id, obs, mark, "source")
        if identity is not None and not identity["ok"]:
            return self._inspection_result(False, object_id, frame_id=obs.frame_id, mark=mark, identity=identity,
                                           identity_mismatch=True,
                                           reason=f"source identity unconfirmed for object #{object_id}: "
                                                  f"{identity['top']} ranked above {identity['expected']}")
        differences = {"top_mm": round(abs(g["top_z"] - target["top_z"]) * 1000, 2),
                       "centre_mm": round(float(np.linalg.norm(np.array(g["centre"][:2]) - target["centre"][:2])) * 1000, 2),
                       "footprint_mm": [round(abs(a - b) * 1000, 2) for a, b in zip(g["footprint"], target["footprint"])],
                       "top_visible": g.get("tilt_deg") is not None}
        if (g.get("tilt_deg") is None or abs(g["top_z"] - target["top_z"]) > 0.010
                or np.linalg.norm(np.array(g["centre"][:2]) - target["centre"][:2]) > 0.015
                or max(abs(a - b) for a, b in zip(g["footprint"], target["footprint"])) > 0.010):
            return self._inspection_result(False, object_id, frame_id=obs.frame_id,
                                           reason="wrist geometry is partial or disagrees with the selected table object",
                                           mark=mark, differences=differences)
        # Keep the table's measured support: an oblique wrist view can hide the bottom.
        refined = dict(g, support_z=target["support_z"], height=g["top_z"] - target["support_z"],
                       support_confident=target.get('support_confident', True))
        refined["centre"] = [g["centre"][0], g["centre"][1], target["support_z"] + refined["height"] / 2]
        a = np.radians((target.get("yaw_deg") or 0.) - (g.get("yaw_deg") or 0.))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        b = np.radians(g.get("yaw_deg") or 0.)
        axes = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
        offset = axes.T @ (np.array(target["centre"][:2]) - g["centre"][:2])
        # A local refinement must not silently shrink the full payload bounds.
        extents = np.maximum(g["extents"][:2], np.abs(R) @ np.array(target["extents"][:2]) + 2 * np.abs(offset))
        refined["extents"], refined["footprint"] = extents.tolist(), sorted(extents.tolist(), reverse=True)
        self.world.objects[object_id].geometry = refined
        obs.object_ids[mark] = object_id
        frame, T = obs.frame, self._camera_pose(obs.frame)
        self.run_dir.joinpath("images").mkdir(parents=True, exist_ok=True)
        # perceive() saves the complete capture, including failed inspections. Retain support for replay stubs.
        path = self.run_dir / 'images' / f'{obs.frame_id}_depth.npz'
        if not path.exists():
            np.savez_compressed(path, depth=frame.depth, K=np.array(frame.K), q=frame.q, stamp=frame.stamp, T_world_cam=T)
        return self._inspection_result(True, object_id, frame_id=obs.frame_id, mark=mark,
                                       obstacles=len(self.inspection_obstacles), geometry=refined,
                                       note="wrist pose refined; full approach and transport must be replanned before descent")

    def _add_inspection_obstacles(self, obs, mark, target):
        frame, T = obs.frame, self._camera_pose(obs.frame)
        tool = Kinematics(tip="tool0").fk(frame.q)
        robot = RobotVolume(CollisionChecker(), frame.q, gripper_gap(frame.finger), None, tool)
        # Supplement table obstacles with local wrist measurements and unexplained depth.
        local = [v for k, v in obs.measured.items() if k != mark]
        local += unexplained(frame, T, robot, list(obs.measured.values()))
        for geometry in local:
            if np.linalg.norm(np.array(geometry["centre"][:2]) - target["centre"][:2]) > 0.20:
                continue
            a = np.radians(geometry.get("yaw_deg") or 0.)
            R = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
            half = np.array([*geometry["extents"][:2], geometry["height"]]) / 2
            name = f"wrist_local_{len(self.inspection_obstacles)}"
            self.inspection_obstacles.append(Box(name, np.array(geometry["centre"]), R, half))

    def _empty_source_inspection(self):
        return (self.held_object is None and self.held_box is None and self.world.holding is None
                and getattr(self, "unknown_held", None) is None and getattr(self, "unconfirmed_grasp", None) is None
                and np.degrees(self.bridge.finger()) <= self.OPEN_FINGER_DEG)

    def _source_view_move(self, q, label):
        """Both the side move and its return retain every object as a collision obstacle."""
        from control.motion import joint_move
        from control.robot import ARM_SPEED
        from control.skills import MoveStep
        from safety.joint_limits import forbidden_on_path

        if not self._empty_source_inspection():
            return {"ok": False, "reason": "source inspection requires a verified empty, open gripper"}
        if hasattr(self, "_sensor_epoch") and sensor_epoch(self) != self._sensor_epoch:
            return {"ok": False, "reason": "simulator reset; no source-view motion with stale geometry"}
        q0 = np.asarray(self.bridge.q(), float)
        trajectory = joint_move(self.kin, q0, np.asarray(q, float), ARM_SPEED)
        collision = CollisionChecker(self.obstacles()).check_path(trajectory.positions)
        forbidden = forbidden_on_path(trajectory.positions)
        if not collision.ok or forbidden:
            return {"ok": False, "reason": forbidden or f"source-view path collides: {collision.contacts[:3]}"}
        result = self.execute([MoveStep(trajectory, label)], label)
        reached = np.degrees(np.max(np.abs(np.asarray(self.bridge.q()) - q))) <= REACH_TOLERANCE_DEG
        return {"ok": bool(result.ok and reached),
                "reason": result.reason if not result.ok else ("" if reached else "source-view joint reach check failed")}

    def return_to_source_inspection(self, q):
        return self._source_view_move(q, "return to source inspection")

    def inspect_source_side(self, object_id, *, judge_identity=None):
        """One side observation of the frozen source, for identity only; never refines grasp geometry."""
        from perception.viewpoints import ViewpointPlanner

        self._source_identity_evidence = None
        if not self._empty_source_inspection():
            return {"ok": False, "reason": "source inspection requires a verified empty, open gripper"}
        target = deepcopy(self.world.objects[object_id].geometry)
        revision, epoch = self.scene_revision, sensor_epoch(self)
        planner = ViewpointPlanner()
        q0 = np.asarray(self.bridge.q(), float)
        box = self.world.objects[object_id].box()
        corners = [box.centre + box.R @ (box.half * [x, y, z])
                   for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        camera = getattr(self, "_source_camera", None)
        vp = planner.plan(np.asarray(target["centre"], float), q0, self.obstacles(),
                          target_box_name=f"obj_{object_id}", distances=self.SOURCE_SIDE_DISTANCES,
                          elevations=self.SOURCE_SIDE_ELEVATIONS_DEG, min_view_angle_deg=20.0,
                          target_points=corners, camera_K=camera[0] if camera else None,
                          image_shape=camera[1] if camera else None, body_visibility_fraction=2/3)
        self.log("tool", tool="source_side_view", object_id=object_id, **vp.summary())
        if not vp.ok:
            return {"ok": False, "reason": vp.reason}
        # A retry must provide a genuinely different direction, rather than the same cap view.
        old_ray = self.wrist_kin.fk(q0)[:3, 3] - np.asarray(target["centre"])
        new_ray = vp.best.camera_pos - np.asarray(target["centre"])
        angle = float(np.degrees(np.arccos(np.clip(old_ray @ new_ray /
                      (np.linalg.norm(old_ray) * np.linalg.norm(new_ray)), -1., 1.))))
        if not np.isfinite(angle) or angle < 20.0:
            return {"ok": False, "reason": "no sufficiently different side viewpoint (less than 20 degrees)"}
        move = self._source_view_move(vp.best.q, "inspect source from side")
        if not move["ok"]:
            return move
        reach = planner.reached(self.bridge.q(), vp.best)
        if not reach["pass"] or reach["camera_pos_error_mm"] > 5.0 or reach["axis_error_deg"] > 2.0:
            return {"ok": False, "reason": "side-view camera pose was not reached", "reach": reach}
        obs = self.perceive("wrist")  # settled, calibrated RGB-D; local mark numbers are not object IDs
        if not self.is_fresh(obs) or obs.frame is None:
            return {"ok": False, "reason": "side RGB-D observation is stale or missing"}
        matches, associations, clean_masks = [], {}, {}
        T = self._camera_pose(obs.frame)
        for k, g in obs.measured.items():
            # A side mask can show the labelled body while the gripper hides the cap. Its measured
            # top is then LOWER than the stored object's top; do not use the top-view association gate.
            clean, points, stats = foreground(obs.marks[k], obs.frame, T,
                                              support_z=target.get("support_z")
                                              if target.get("support_confident", True) else None)
            clean_masks[k] = clean
            binding = {}
            status, oid, candidates = self.world.associate_partial(points, diagnostics=binding)
            if stats["foreground_fraction"] < .5:
                status, oid, candidates = "insufficient_foreground", None, []
            associations[k] = {"status": status, "object_id": oid, "candidates": candidates,
                               **stats, "binding": binding, "depth_points": len(points),
                               "visible_top_mm": round(g["top_z"] * 1000, 1)}
            if status == "matched" and oid == object_id:
                matches.append(k)
        self.log("tool", tool="source_side_binding", object_id=object_id, frame_id=obs.frame_id,
                 associations=associations, matched_marks=matches)
        if len(matches) != 1:
            return {"ok": False, "reason": "side view does not uniquely associate the selected source",
                    "frame_id": obs.frame_id, "matched_marks": matches, "associations": associations}
        mark = matches[0]
        self._add_inspection_obstacles(obs, mark, target)
        # Reject cropped-off silhouettes and views hiding most of the body's height.
        quality = crop_quality(clean_masks[mark], obs.rgb)
        bottom = obs.measured[mark].get("visible_bottom_z")
        if (not quality["ok"]
                or bottom is None or bottom > target["support_z"] + 0.35 * target["height"]):
            return {"ok": False, "reason": "side view does not show an unclipped source body",
                    "frame_id": obs.frame_id, "mark": mark, "quality": quality}
        semantic_obs = copy(obs)
        semantic_obs.marks = {**obs.marks, mark: clean_masks[mark]}
        identity = self.identity_check(object_id, semantic_obs, mark, "source_side")
        if (identity is None or not identity["ok"]) and judge_identity is not None:
            crop = isolate(obs.rgb, clean_masks[mark], margin=.15)
            path = self.run_dir / "images" / f"{obs.frame_id}_source_confirmation.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            verdict, reason = judge_identity({"rgb": crop, "object_id": object_id, "frame_id": obs.frame_id,
                                               "path": str(path), "mark": mark, "quality": quality})
            identity = {"ok": verdict == "yes", "verdict": verdict, "reason": reason,
                        "object_id": object_id, "frame_id": obs.frame_id, "mark": mark,
                        "expected": self.object_words.get(object_id), "method": "gemma_reference"}
            self.log("source_confirmation", **identity)
        if (revision != self.scene_revision or epoch != sensor_epoch(self)
                or target != self.world.objects[object_id].geometry or not self.is_fresh(obs)
                or not self._empty_source_inspection()
                or np.degrees(np.max(np.abs(np.asarray(self.bridge.q()) - obs.frame.q))) > REACH_TOLERANCE_DEG):
            return {"ok": False, "reason": "source/scene/camera changed during identity inspection"}
        if identity is None or not identity["ok"]:
            return {"ok": False, "reason": "side-view source identity remains unconfirmed", "identity": identity,
                    "frame_id": obs.frame_id, "mark": mark}
        proof = {"identity": identity, "object_id": object_id, "expected": self.object_words.get(object_id),
                 "geometry": deepcopy(target), "epoch": getattr(obs.frame, "epoch", None),
                 "scene_revision": self.scene_revision, "captured_at": time.monotonic()}
        self._source_identity_evidence = proof
        return {"ok": True, "frame_id": obs.frame_id, "mark": mark, "identity": identity,
                "angle_change_deg": round(angle, 1), "identity_evidence": proof}

    def _valid_source_identity(self, proof, object_id, obs):
        return (proof is getattr(self, "_source_identity_evidence", None)
                and proof["object_id"] == object_id and proof["identity"]["ok"]
                and proof["expected"] == self.object_words.get(object_id)
                and proof["scene_revision"] == self.scene_revision
                and proof["epoch"] == getattr(obs.frame, "epoch", None)
                and time.monotonic() - proof["captured_at"] <= self.MAX_OBS_AGE_S
                and proof["geometry"] == self.world.objects[object_id].geometry
                and self._empty_source_inspection())

    def identity_check(self, object_id: int, obs, mark: int, role: str) -> dict | None:
        """The wrist close-up of one object against every catalogue name (SigLIP 2): is it what the catalogue calls
        it? None when there is nothing to check (no matcher, no name, a one-name catalogue). Gemma chose the object;
        this only confirms what the camera sees up close before the gripper descends (source) or opens (destination)."""
        matcher, expected = getattr(self, "matcher", None), self.object_words.get(object_id)
        every = list(dict.fromkeys(w for w in self.object_words.values() if w and w != "object"))
        if matcher is None or not expected or expected == "object" or len(every) < 2:
            return None
        crop = isolate(obs.rgb, obs.marks[mark], margin=0.15)
        path = self.run_dir / "images" / f"{obs.frame_id}_identity_{role}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        result = {**matcher.check(crop, expected, every), "object_id": object_id, "mark": mark, "role": role,
                  "frame_id": obs.frame_id, "crop": str(path)}
        self.log("identity_check", **result)
        return result

    def inspect_destination(self, reference_id: int) -> dict | None:
        """Before release, above the destination: the wrist close-up of the destination object, checked like the
        source. None (no check, logged) when the wrist camera does not uniquely see it; placement still proceeds."""
        if getattr(self, "matcher", None) is None or reference_id not in self.object_words:
            return None
        try:
            obs = self.perceive("wrist")  # snapshot enforces settling
        except Exception as exc:
            self.log("identity_check", role="destination", object_id=reference_id, ok=None, reason=f"wrist view: {exc!r}")
            return None
        marks = [k for k, g in obs.measured.items() if self.world.associate(g)[:2] == ("matched", reference_id)]
        if len(marks) != 1:
            self.log("identity_check", role="destination", object_id=reference_id, ok=None, frame_id=obs.frame_id,
                     reason=f"the wrist camera sees the destination in {len(marks)} measured objects, not one")
            return None
        return self.identity_check(reference_id, obs, marks[0], "destination")

    def _inspection_result(self, ok, object_id, **evidence):
        result = {"ok": ok, "object_id": object_id, **evidence}
        self.log("tool", tool="inspect_grasp", **result)
        return result
