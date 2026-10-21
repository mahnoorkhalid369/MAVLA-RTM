"""Sensor-only verification: finger contact, HOME from joint feedback, placement from a fresh table observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from config.agent import REACH_TOLERANCE_DEG
from control.robot import ARM_HOME, gripper_gap

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
        look = self.look_in_gripper(object_id)
        v.update(by="fingers borderline -> table camera", camera=look,
                 **{"pass": bool(look["in_gripper"] and look["old_spot"] != "occupied")})
        self.log("verification", **{k: v[k] for k in ("check", "by", "pass", "stopped_short_deg")}, camera=look)
        return v

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
        v = {"check": "placement", "object_id": plan.object_id, "relation": plan.relation, "reference_id": plan.reference_id,
             "planned_centre_mm": [round(x * 1000, 1) for x in plan.target_centre],
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
            if v["pass"]:  # the measured top confirms the planned pose: the bottom is where it was put (the camera
                # may see a stack as one object, whose bottom is the floor; live 2026-10-21)
                support = plan.target_top - plan.held.height
                g = dict(g, support_z=support, height=g["top_z"] - support)
                g["centre"] = [g["centre"][0], g["centre"][1], support + g["height"] / 2]
            o.geometry, o.status = g, "on_surface"
        moved = self.update_world(obs)  # geometric check: no VLM calls (known objects keep their description)
        v["disturbed_neighbours"] = [m for m in moved if m["id"] != plan.object_id]
        self.log("verification", **v)
        return v
