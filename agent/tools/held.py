"""Session and held-object state.

Owner: Tools (one per session, agent/session.py). What the robot holds is recorded here and nowhere else:
    held               control.skills.Held body of the grasped object (None = nothing held)
    held_object        its world id
    held_geom          its measured geometry at the grasp
    held_close_target  finger close target (retention checks)
    held_box           its collision body in tool0 (perception drops it, planning carries it)
WorldModel.holding mirrors held_object for the world model (WorldModel.match skips the held object). set_held and
clear_held change both (assume_placed and revalidate_held go through clear_held); verify_placement additionally
resets held_box and world.holding after a release. A remembered held object is a claim: revalidate_held checks it
against finger feedback before anything relies on it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent.tools.manipulation import PickPlan


class HeldObjectState:
    OPEN_FINGER_DEG = 2.0  # finger_joint <= this = gripper open (open = 0 deg; 2 deg = 83 mm gap)

    def begin_instruction(self, log, roles, run_dir) -> dict:
        """Next instruction of the same session: new outputs (event log, VLM roles with their counters, images);
        the session state (world model, held object, last referent, clarifications) is kept, and the physical
        part of it is re-verified from the sensors before anything relies on it."""
        self.log, self.roles, self.run_dir = log, roles, run_dir
        return self.revalidate_held()

    def revalidate_held(self) -> dict:
        """A remembered held object is a claim, not evidence: the fingers must still be stalled on something
        (finger feedback, the same check as the retention check after a lift). If not, the claim is dropped and
        the object's pose is unknown (status "missing") until it is measured again."""
        finger = float(np.degrees(self.bridge.finger()))
        ev = {"held_object": self.held_object, "finger_deg": round(finger, 2), "last_referent": self.last_referent}
        if self.held_object is None:
            ev["held"] = "none"
            if finger > self.OPEN_FINGER_DEG:
                ev["note"] = "gripper not open, but nothing is recorded as held"
        else:
            v = self.verify_contact(self.held_close_target)
            if v["pass"]:
                ev["held"] = "verified"
            else:
                ev["held"] = "invalidated"
                lost = self.world.objects.get(self.held_object)
                if lost is not None:
                    lost.status = "missing"
                self.clear_held()
        self.log("session_state", **ev)
        return ev

    def set_held(self, plan: PickPlan, close_target: float) -> None:
        self.held, self.held_object, self.held_close_target = plan.held, plan.object_id, close_target
        self.held_geom = dict(self.world.objects[plan.object_id].geometry)
        self.held_box = plan.held.collision_body()
        self.world.holding = plan.object_id
        self.world.objects[plan.object_id].status = "held"

    def clear_held(self) -> None:
        self.held, self.held_object, self.held_geom, self.held_close_target = None, None, None, None
        self.held_box, self.world.holding = None, None
