"""Sensor-only verification: finger contact, HOME from joint feedback, the wrist camera's picture of a placement (judged
by Gemma in agent/workers.py, not here), and relations between two measured objects (in, on, next_to, left_of,
right_of, in_front_of, behind). verify_placement (table re-measurement) is kept for runs without a language model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from copy import deepcopy
from agent.timing import timed
from perception.capture import ObservationError

import cv2
import numpy as np

from config import layout
from config.agent import REACH_TOLERANCE_DEG
from control.robot import ARM_HOME, gripper_gap, table_camera_pose
from perception.world_model import SAME_FOOTPRINT, appearance_agrees

if TYPE_CHECKING:
    from agent.tools.manipulation import PickPlan


class VerificationTools:
    HOME_TOLERANCE_DEG = REACH_TOLERANCE_DEG  # per joint: RosExecutor's validated reach tolerance (config/agent.py)

    PLACEMENT_VIEW = {"near": (0.30, 0.35), "far": (0.45,)}  # camera distance from the placed object (m)
    PLACEMENT_VIEW_ELEVATIONS_DEG = (80.0, 65.0, 50.0)  # looking down at it, steepest first

    def look_at_placement(self, plan: PickPlan, farther: bool = False) -> dict:
        """After release: the wrist camera looks down at the placed object and the destination around it."""
        return self.look_at(plan.target_centre, f"obj_{plan.object_id}", farther, "look at the placement")

    @timed("planning")
    def look_at(self, point, box_name: str | None = None, farther: bool = False, label: str = "look") -> dict:
        """Move the wrist camera to look down at a world point and return its picture. The pose is a collision-checked
        look-at viewpoint (perception/viewpoints.py; the object box `box_name` may be in the line of sight); nothing
        is measured or matched: the picture goes to Gemma, who judges it in words. Not while holding something (the
        viewpoint path is checked for the robot alone)."""
        from control.motion import joint_move
        from control.robot import ARM_SPEED
        from control.skills import MoveStep
        from perception.viewpoints import ViewpointPlanner

        if self.held_object is not None or self.held_box is not None:
            return {"ok": False, "reason": "the gripper holds an object: no viewpoint move"}
        q0 = self.bridge.q()
        vp = ViewpointPlanner().plan(np.asarray(point, float), q0, self.obstacles(), target_box_name=box_name,
                                     distances=self.PLACEMENT_VIEW["far" if farther else "near"],
                                     elevations=self.PLACEMENT_VIEW_ELEVATIONS_DEG, region_radius=0.04)
        self.log("tool", tool="look_at", label=label, farther=farther, **vp.summary())
        if not vp.ok:
            return {"ok": False, "reason": vp.reason}
        res = self.execute([MoveStep(joint_move(self.kin, q0, vp.best.q, ARM_SPEED), label)], label)
        if not res.ok:
            return {"ok": False, "reason": f"{label}: {res.reason}"}
        return self.wrist_picture(label)

    def wrist_picture(self, label: str = "wrist") -> dict:
        """The wrist camera's picture from where the arm is now (saved as images/wrist_NNN_<label>.jpg)."""
        frame = self.bridge.snapshot("wrist")  # snapshot already enforces settling
        epoch = getattr(frame, "epoch", None)
        if hasattr(self, "_sensor_epoch") and epoch != self._sensor_epoch:
            raise ObservationError("simulator reset; restart before using wrist verification")
        self._frames += 1
        fid = f"wrist_{self._frames:03d}"
        path = self.run_dir / "images" / f"{fid}_{label.replace(' ', '_')}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        return {"ok": True, "rgb": frame.rgb, "frame_id": fid, "path": str(path), "sim_time": frame.stamp, "epoch": epoch}

    def verify_contact(self, close_target: float) -> dict:
        finger = self.bridge.finger()
        short = float(np.degrees(close_target - finger))
        v = {"check": "contact", "finger_deg": round(float(np.degrees(finger)), 2), "target_deg": round(float(np.degrees(close_target)), 2),
             "stopped_short_deg": round(short, 2), "gap_mm": round(gripper_gap(finger) * 1000, 1), "pass": short >= self.tol.contact_deg}
        self.log("verification", **v)
        return v

    IN_GRIPPER_XY_M, IN_GRIPPER_Z_M = 0.06, (-0.15, 0.08)  # around the fingertip centre (TCP): a held object's centre

    def confirm_held(self, close_target: float, object_id: int | None, check: str) -> dict:
        """Is the object still in the gripper? Fingers first: clearly stalled -> held; closed on nothing -> not held.
        In between (live 2026-10-21: 1.45 deg short after a lift, threshold 1.5, object still in the fingers) the table
        camera looks (look_in_gripper): held if it measures something in the gripper and the object's old spot on the
        table is empty; anything else stays not confirmed (never assumed)."""
        v = self.verify_contact(close_target)
        v["check"], v["by"] = check, "fingers"
        if v["pass"] or v["stopped_short_deg"] < self.tol.empty_deg:
            return v
        wrist = self.wrist_sees_held()  # the wrist camera looks straight between the fingers (2026-10-26)
        if wrist.get("held"):
            v.update(by="fingers borderline -> wrist camera", wrist=wrist, **{"pass": True})
            self.log("verification", **{k: v[k] for k in ("check", "by", "pass", "stopped_short_deg")}, wrist=wrist)
            return v
        look = self.look_in_gripper(object_id)
        v.update(by="fingers borderline -> wrist camera, table camera", wrist=wrist, camera=look,
                 **{"pass": bool(look["in_gripper"] and look["old_spot"] != "occupied")})
        self.log("verification", **{k: v[k] for k in ("check", "by", "pass", "stopped_short_deg")}, wrist=wrist, camera=look)
        return v

    # gripper axis points (m from the TCP towards the fingertips, tool z) where a held object is: palm side to tips
    WRIST_AXIS_POINTS_M = (-0.020, -0.012, -0.004, 0.004)
    WRIST_WINDOW_PX = 4  # +- pixels sampled around each projected point
    WRIST_HELD_MARGIN_M = 0.04  # depth up to this far beyond the gripper axis point = a surface between the fingers

    def wrist_sees_held(self) -> dict:
        """Is something between the fingers? The wrist camera sits on the J5 housing, so the gripper axis (through the
        TCP, on the J6 axis) projects to the same pixels whatever J6 does: its DEPTH there shows a surface at the
        fingers (held) or the table far below (empty). Geometry only, no detector, no Gemma (live 2026-10-26: the
        table camera could not see the held block in the gripper; the wrist camera sees it plainly).
        -> {"held": True | False | None (no valid depth at those pixels), "fraction", "pixels", ...}"""
        try:
            frame = self.bridge.snapshot("wrist")
        except Exception as exc:  # no wrist frame: no evidence either way
            return {"held": None, "reason": f"no wrist frame ({exc})"}
        Tc = self._camera_pose(frame) if getattr(frame, 'T_world_camera', None) is not None else self.wrist_kin.fk(frame.q)
        if getattr(frame, 'T_world_camera', None) is not None:
            self._check_camera_pose(frame, Tc)
        tcp = self.kin.fk(frame.q)
        fx, fy, cx, cy = frame.K
        h, w = frame.depth.shape[:2]
        near, total, pixels, expected = 0, 0, [], []
        for s in self.WRIST_AXIS_POINTS_M:
            p = tcp[:3, 3] + tcp[:3, 2] * s
            pc = Tc[:3, :3].T @ (p - Tc[:3, 3])
            if pc[2] <= 0.02:
                continue
            u, v = int(round(fx * pc[0] / pc[2] + cx)), int(round(fy * pc[1] / pc[2] + cy))
            if not (0 <= u < w and 0 <= v < h):
                continue
            k = self.WRIST_WINDOW_PX
            z = frame.depth[max(v - k, 0):v + k + 1, max(u - k, 0):u + k + 1].astype(float).ravel()
            z = z[np.isfinite(z) & (z > 0.02)]
            total += len(z)
            near += int(np.sum(z <= pc[2] + self.WRIST_HELD_MARGIN_M))
            pixels.append([u, v])
            expected.append(round(float(pc[2]), 3))
        if total == 0:
            return {"held": None, "reason": "the gripper axis is not in the wrist image or has no depth", "pixels": pixels}
        frac = near / total
        out = {"held": bool(frac >= 0.5), "fraction": round(frac, 2), "pixels": pixels, "expected_m": expected,
               "finger_deg": round(float(np.degrees(frame.finger)), 2)}
        self.log("tool", tool="wrist_held_check", **out)
        return out

    def look_in_gripper(self, object_id: int | None) -> dict:
        """One table-camera observation (no Gemma): is something measured in the gripper (around the fingertip centre,
        off the table), and is the object's remembered spot on the table empty?"""
        obs = self.perceive("table")
        tool = self.kin.fk(self.bridge.q())[:3, 3]
        near = [k for k, g in obs.measured.items()
                if np.hypot(g["centre"][0] - tool[0], g["centre"][1] - tool[1]) < self.IN_GRIPPER_XY_M
                and self.IN_GRIPPER_Z_M[0] < g["centre"][2] - tool[2] < self.IN_GRIPPER_Z_M[1] and g.get("support_z", 0.0) > 0.02]
        removed = [e for e in obs.excluded if e.get("reason") == "the held object"]  # dropped by perceive (held box)
        spot = "unknown"
        o = self.world.objects.get(object_id) if object_id is not None else None
        g0 = (self.held_geom if object_id == self.held_object and self.held_geom else None) or (o.geometry if o else None)
        if g0 is not None:
            on_spot = any(np.hypot(g["centre"][0] - g0["centre"][0], g["centre"][1] - g0["centre"][1]) < 0.03
                          and abs(g["top_z"] - g0["top_z"]) < 0.02 for g in obs.measured.values())
            spot = "occupied" if on_spot else "empty"
        return {"frame_id": obs.frame_id, "in_gripper": bool(near or removed), "marks_in_gripper": near,
                "old_spot": spot, "tcp_mm": [round(float(x) * 1000) for x in tool]}

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

    REGION_INSIDE_TOL_M = 0.005  # a footprint may reach this far past a region's outline (measurement noise)
    REGION_REST_TOL_M = 0.010  # resting on the region's surface: the support within this of it (as for "on")

    def in_region(self, g: dict, region) -> dict:
        """Is a measured object inside a fixed planar region (config/layout.py)? Its whole footprint within the
        calibrated outline (worst corner, or the edge of a round one) and resting on the region's surface.
        -> {"inside", "margin_mm" (>= 0: inside by that much at the worst point), "resting_mm" (support above it)}"""
        m = region.footprint_margin(g["centre"], g.get("extents") or g["footprint"], g.get("yaw_deg"))
        rest = float(g["support_z"]) - float(region.centre[2])
        return {"inside": bool(m >= -self.REGION_INSIDE_TOL_M and abs(rest) <= self.REGION_REST_TOL_M),
                "margin_mm": round(m * 1000, 1), "resting_mm": round(rest * 1000, 1)}

    @staticmethod
    def _same_size(g: dict, known: dict) -> bool:
        """A measurement has the footprint of a known object (both sides within the world model's footprint gate)."""
        return all(abs(a - b) < SAME_FOOTPRINT for a, b in zip(sorted(g["footprint"]), sorted(known["footprint"])))

    @staticmethod
    def footprint_margin(target, anchor, inset=0.):
        """Worst footprint edge inside a rotated supporting rectangle; round bodies use their radius."""
        a = np.radians(anchor.get('yaw_deg') or 0.)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        local = R.T @ (np.array(target['centre'][:2]) - anchor['centre'][:2])
        half = np.array(anchor.get('extents') or anchor['footprint']) / 2 - inset
        if target.get('yaw_deg') is None:
            reach = np.full(2, max(target['footprint']) / 2)
        else:
            b = np.radians(target['yaw_deg'] - (anchor.get('yaw_deg') or 0.))
            reach = np.abs([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]]) @ (np.array(target.get('extents') or target['footprint']) / 2)
        return float(np.min(half - np.abs(local) - reach))

    @staticmethod
    def placement_identity(g, known):
        return (appearance_agrees(g, known) and all(abs(a - b) <= max(.015, a * .25)
                for a, b in zip(sorted(known['footprint']), sorted(g['footprint']))))

    def placement_goal(self, g, plan, ref):
        xy = float(np.linalg.norm(np.array(g['centre'][:2]) - plan.target_centre[:2]))
        dz = g['top_z'] - plan.target_top
        tilt = g.get('tilt_deg')
        v = {'xy_error_mm': round(xy * 1000, 1), 'z_error_mm': round(dz * 1000, 1),
             'tilt_deg': tilt, 'tilt_applied': g.get('top_planar', True),
             'measured_centre_mm': [round(x * 1000, 1) for x in g['centre']]}
        support = plan.target_top - plan.held.height
        resting = abs(g['support_z'] - support)
        v['resting_error_mm'] = round(resting * 1000, 1)
        angle_ok = not v['tilt_applied'] or (tilt is not None and tilt <= self.tol.tilt_deg)
        if getattr(plan, 'strict_pose', False) and not v['tilt_applied']:
            angle_ok = False  # a curved surface plane cannot establish a requested rigid orientation
        # Full body height is obtained from a confirmed support before pickup. A floating/stacked object
        # must not pass just because a patch of tray floor is visible beside it.
        v['geometry_consistent'] = abs(dz) <= self.tol.z_m
        if plan.relation in ('in', 'on') and ref is not None and not getattr(plan, 'strict_pose', False):
            margin = self.footprint_margin(g, ref, self.CONTAINER_WALL_M if plan.relation == 'in' else 0.)
            v['footprint_margin_mm'] = round(margin * 1000, 1)
            v['criterion'] = 'whole footprint inside destination, supported, stable and released'
            v['pass'] = bool(margin >= -self.REGION_INSIDE_TOL_M and resting <= self.REGION_REST_TOL_M
                             and g.get('support_confident', True) and angle_ok and v['geometry_consistent'])
        else:
            v['criterion'] = 'planned pose'
            v['pass'] = bool(xy <= self.tol.xy_m and abs(dz) <= self.tol.z_m and angle_ok)
        region = layout.region(plan.reference_region)
        if region is not None:
            v['region'] = {'id': region.id, **self.in_region(g, region)}
            v['pass'] = v['pass'] and v['region']['inside']
        return v

    def verify_placement(self, plan: PickPlan) -> dict:
        """Re-observe the whole table (whole-scene re-perception) and check the placed object's pose."""
        obs = self.perceive("table")
        self.held_box = None
        self.world.holding = None
        o = self.world.objects[plan.object_id]
        known = deepcopy(o.geometry)
        known_ids = set(self.world.objects)
        moved = self.update_world(obs)
        ref = self.world.objects[plan.reference_id].geometry if plan.reference_id in self.world.objects else None
        v = {"check": "placement", "object_id": plan.object_id, "relation": plan.relation,
             "reference_id": plan.reference_id, "reference_region": plan.reference_region,
             "planned_centre_mm": [round(x * 1000, 1) for x in plan.target_centre],
             "planned_top_mm": round(plan.target_top * 1000, 1), "frame_id": obs.frame_id}

        def candidates(observation):
            out = []
            for mark, g in observation.measured.items():
                observed_id = observation.object_ids.get(mark)
                if observed_id in known_ids and observed_id != plan.object_id:
                    continue  # another previously known item is not evidence of the delivered item
                if observation.object_ids.get(mark) == plan.reference_id or not self.placement_identity(g, known):
                    continue
                d = np.linalg.norm(np.array(g['centre'][:2]) - plan.target_centre[:2])
                in_destination = ref is not None and plan.relation in ('in', 'on') and self.footprint_margin(g, ref) >= -.005
                if d <= .05 or in_destination:
                    out.append((mark, g))
            return out

        found = candidates(obs)
        v['candidate_marks'] = [mark for mark, _ in found]
        released = self.held_object is None and float(np.degrees(self.bridge.finger())) <= self.OPEN_FINGER_DEG
        v['release_confirmed'] = released
        if plan.reference_id is not None and plan.reference_id not in obs.object_ids.values():
            v.update({'pass': False, 'reason': 'the destination is not measured in the current table observation'})
        elif len(found) != 1:
            v.update({'pass': False, 'reason': 'no unique size/appearance-compatible released object at the destination'})
            v['depth'] = self.top_by_depth(plan)  # diagnostic only: depth alone cannot establish object identity
        else:
            mark, g = found[0]
            v.update(self.placement_goal(g, plan, ref))
            if v['pass'] and released:
                # Re-observe after the robot has settled. No new arm action or VLM call.
                second = self.perceive('table')
                moved += self.update_world(second)
                previous_ref = ref
                ref = self.world.objects[plan.reference_id].geometry if plan.reference_id in self.world.objects else None
                reference_seen = plan.reference_id is None or plan.reference_id in second.object_ids.values()
                reference_stable = (previous_ref is None or (ref is not None
                                    and np.linalg.norm(np.array(ref['centre'][:2]) - previous_ref['centre'][:2]) <= .005
                                    and abs(ref['top_z'] - previous_ref['top_z']) <= .003))
                again = candidates(second) if reference_seen else []
                stable = len(again) == 1 and reference_stable
                if stable:
                    next_mark, h = again[0]
                    stable = (np.linalg.norm(np.array(g['centre'][:2]) - h['centre'][:2]) <= .005
                              and abs(g['top_z'] - h['top_z']) <= .003
                              and abs(g['support_z'] - h['support_z']) <= .005
                              and self.placement_goal(h, plan, ref)['pass'])
                    if stable:
                        g = h
                        mark = next_mark
                        v.update(self.placement_goal(g, plan, ref))
                v['stability'] = {'pass': bool(stable), 'frames': [obs.frame_id, second.frame_id]}
                v['release_confirmed'] = self.held_object is None and float(np.degrees(self.bridge.finger())) <= self.OPEN_FINGER_DEG
                v['pass'] = bool(stable and v['release_confirmed'])
                v['frame_id'] = second.frame_id
                obs = second
            else:
                v['pass'] = False
            if v['pass']:
                # Store measured support/height, never overwrite evidence with the intended pose.
                o.geometry, o.status = dict(g), 'on_surface'
                observed_id = obs.object_ids.get(mark)
                if observed_id is not None and observed_id != plan.object_id:
                    # A uniquely verified physical handoff can reconcile a newly allocated detection alias.
                    # Never merge a different object that was already known before verification.
                    self.world.objects[observed_id].status = 'merged'
                    v['reconciled_alias'] = observed_id
                obs.object_ids[mark] = plan.object_id
                if mark in obs.marks:
                    o.mask = obs.marks[mark].copy()
            if not v['pass']:
                v['reason'] = 'released-object identity, support, containment, orientation or stability is unconfirmed'
        v["disturbed_neighbours"] = [m for m in moved if m["id"] != plan.object_id]
        self.log("verification", **v)
        return v

    def relation_holds(self, target: dict, anchor: dict, relation: str, frame: str = "robot") -> bool:
        t, a = np.asarray(target["centre"]), np.asarray(anchor["centre"])
        if relation in ('in', 'on'):
            inset = self.CONTAINER_WALL_M if relation == 'in' else 0.
            support = self._container_floor(anchor) if relation == 'in' else anchor['top_z']
            return bool(target.get('support_confident', True)
                        and self.footprint_margin(target, anchor, inset) >= -self.REGION_INSIDE_TOL_M
                        and abs(target['support_z'] - support) <= self.REGION_REST_TOL_M)
        gap = np.hypot(*(t[:2] - a[:2])) - (max(target["footprint"]) + max(anchor["footprint"])) / 2
        if gap > self.rel.relation_check_max_gap_m:
            return False
        if relation == "next_to":
            return True
        if frame == "camera":
            R = getattr(self, 'last_table_pose', table_camera_pose())[:3, :3]
            right, forward = _unit(R[:2, 0]), _unit(R[:2, 2])  # image right; viewing direction on the floor
            dirs = {"left_of": -right, "right_of": right, "in_front_of": -forward, "behind": forward}
        else:  # robot frame: +x right, +y forward; "in front of" = between the anchor and the robot
            radial = _unit(a[:2])
            dirs = {"left_of": np.array([-1.0, 0.0]), "right_of": np.array([1.0, 0.0]), "in_front_of": -radial, "behind": radial}
        return float(np.dot(_unit(t[:2] - a[:2]), dirs[relation])) >= np.cos(np.radians(45))

    DEPTH_TOP_TOL_M = 0.012  # the planned top surface, along the camera ray

    # Without the detector: is the placed object's top surface where the plan put it? The table camera's depth at a
    # few points of the planned top face (projected into the image) against the planned surface's depth.
    def top_by_depth(self, plan) -> dict:
        try:
            frame = self.bridge.snapshot("table")
        except Exception as exc:
            return {"present": None, "reason": f"no table frame ({exc})"}
        Tc = self._camera_pose(frame)
        fx, fy, cx, cy = frame.K
        h, w = frame.depth.shape[:2]
        c = np.asarray(plan.target_centre[:2], float)
        r = 0.3 * min(plan.held.width, plan.held.length)
        hit = total = 0
        for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
            p = np.array([c[0] + dx, c[1] + dy, plan.target_top])
            pc = Tc[:3, :3].T @ (p - Tc[:3, 3])
            if pc[2] <= 0.05:
                continue
            u, v = int(round(fx * pc[0] / pc[2] + cx)), int(round(fy * pc[1] / pc[2] + cy))
            if not (0 <= u < w and 0 <= v < h):
                continue
            z = frame.depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].astype(float).ravel()
            z = z[np.isfinite(z) & (z > 0.05)]
            total += len(z)
            hit += int(np.sum(np.abs(z - pc[2]) <= self.DEPTH_TOP_TOL_M))
        if total == 0:
            return {"present": None, "reason": "the planned place is not in the table image or has no depth"}
        out = {"present": bool(hit / total >= 0.6), "fraction": round(hit / total, 2)}
        self.log("tool", tool="placement_depth_check", **out)
        return out


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
