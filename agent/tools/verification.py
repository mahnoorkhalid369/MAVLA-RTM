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
