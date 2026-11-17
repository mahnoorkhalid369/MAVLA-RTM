"""Sensor-only verification: finger contact, HOME from joint feedback, placement from a fresh table observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from config import layout
from config.agent import REACH_TOLERANCE_DEG
from control.robot import ARM_HOME, gripper_gap
from perception.world_model import SAME_FOOTPRINT

if TYPE_CHECKING:
    from agent.tools.manipulation import PickPlan


class VerificationTools:
    HOME_TOLERANCE_DEG = REACH_TOLERANCE_DEG  # per joint: RosExecutor's validated reach tolerance (config/agent.py)

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
        Tc = self.wrist_kin.fk(frame.q)
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

    def verify_placement(self, plan: PickPlan) -> dict:
        """Re-observe the whole table (whole-scene re-perception) and check the placed object's pose."""
        obs = self.perceive("table")
        self.held_box = None
        self.world.holding = None
        o = self.world.objects[plan.object_id]
        best, best_d = None, np.inf
        # the destination is not the placed object: placed IN a container the object shares its centre, placed ON a
        # wide one (a tray's rim) it is next to it (live 2026-10-26: the tray's rim, 40 mm, was measured instead of the
        # block, "z -120.5 mm, out of tolerance"). Recognised by its measured geometry (this frame's labels are
        # associated with world ids only afterwards): the destination's top height, most of its size, over its centre
        ref = self.world.objects[plan.reference_id].geometry if plan.reference_id in self.world.objects else None

        def is_container(g):
            return (ref is not None and abs(g["top_z"] - ref["top_z"]) < 0.01 and max(g["footprint"]) > 0.7 * max(ref["footprint"])
                    and np.hypot(g["centre"][0] - ref["centre"][0], g["centre"][1] - ref["centre"][1]) < max(ref["footprint"]) / 2)
        for k, g in obs.measured.items():
            if is_container(g):
                continue
            d = float(np.hypot(g["centre"][0] - plan.target_centre[0], g["centre"][1] - plan.target_centre[1]))
            if d < best_d:
                best, best_d = k, d
        v = {"check": "placement", "object_id": plan.object_id, "relation": plan.relation, "reference_id": plan.reference_id,
             "reference_region": plan.reference_region,
             "planned_centre_mm": [round(x * 1000, 1) for x in plan.target_centre],
             "planned_top_mm": round(plan.target_top * 1000, 1), "frame_id": obs.frame_id}
        if best is None or best_d > 0.05:
            v.update({"pass": False, "reason": "no object measured within 50 mm of the planned place"})
        else:
            g = obs.measured[best]
            xy, dz, tilt = best_d, g["top_z"] - plan.target_top, g["tilt_deg"]
            if tilt is None:  # top face not visible: cannot verify the pose
                tilt = float("inf")
            z_ok = abs(dz) <= self.tol.z_m
            v.update({"measured_centre_mm": [round(x * 1000, 1) for x in g["centre"]], "xy_error_mm": round(xy * 1000, 1),
                      "z_error_mm": round(dz * 1000, 1), "tilt_deg": None if tilt == float("inf") else round(tilt, 2),
                      "pass": xy <= self.tol.xy_m and z_ok and tilt <= self.tol.tilt_deg})
            region = layout.region(plan.reference_region)
            if region is not None:  # into a fixed region: the measured footprint must also be inside its outline
                v["region"] = {"id": region.id, **self.in_region(g, region)}
                v["pass"] = v["pass"] and v["region"]["inside"]
                if not v["region"]["inside"]:
                    v["reason"] = (f"the placed object is not inside {region.id} ({region.name}): worst point "
                                   f"{v['region']['margin_mm']} mm inside the outline, resting {v['region']['resting_mm']} mm up")
            if v["pass"]:  # the measured top confirms the planned pose: the bottom is where it was put (the camera
                # may see a stack as one object, whose bottom is the floor; live 2026-10-21)
                support = plan.target_top - plan.held.height
                g = dict(g, support_z=support, height=g["top_z"] - support)
                g["centre"] = [g["centre"][0], g["centre"][1], support + g["height"] / 2]
                o.geometry = g
            elif self._same_size(g, o.geometry):  # off the planned pose, but the placed object by its size: where it is
                o.geometry = g
            else:  # not evidence about the placed object: its record keeps the planned pose (assume_placed), not
                # another object's size and place (live 2026-10-26: the block's record took the tray's 202 x 142 mm)
                v["record"] = "kept the planned pose: the nearest measurement is not the placed object's size"
            o.status = "on_surface"
        moved = self.update_world(obs)  # geometric check: no VLM calls (known objects keep their description)
        v["disturbed_neighbours"] = [m for m in moved if m["id"] != plan.object_id]
        self.log("verification", **v)
        return v
