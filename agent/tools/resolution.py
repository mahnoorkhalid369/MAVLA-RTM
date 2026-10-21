"""Target resolution: which measured object the user's words mean (Gemma resolve over its own descriptions,
confirmation on the isolated crop; ambiguity goes to the user) and deterministic relation checks on measured
geometry (relation_holds). No word matching in Python."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from control.robot import table_camera_pose
from perception.somark import isolate, montage

if TYPE_CHECKING:
    from agent.tools.perception import Observation


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


class ResolutionTools:
    def resolve(self, obs: Observation, text: str, exclude=()):
        """Gemma's resolution over the descriptions of the candidate objects (its shortlist, if any): one object, or
        ambiguous with candidates (the user is asked; the model never guesses). The chosen object is then confirmed
        on its crop by the caller. exclude: marks already rejected for this reference in this task."""
        pool = [k for k in self.pool(obs, text) if k not in set(exclude)]
        if not pool:
            from agent.vlm_roles import Resolution
            return Resolution(id=None, ambiguous=False, candidates=[], reason="every candidate was rejected")
        self.describe(obs, pool)  # no-op for objects already described
        descs = [{"id": k, **obs.descriptions.get(k, {}), "measured": self.measured_facts(obs, k)} for k in pool]
        r = self.roles.resolve(obs.som, descs, text, pool)
        self.log("tool", tool="resolve", text=text, id=r.id, ambiguous=r.ambiguous, candidates=r.candidates)
        return r

    @staticmethod
    def measured_facts(obs: Observation, mark: int) -> str:
        """What the depth camera measured about one object, as text for Gemma: height, footprint, position in the
        robot frame and the camera's average colour inside its mask (measurements, not a model's words)."""
        g = (obs.measured or {}).get(mark) or {}
        parts = []
        if g.get("height") is not None:
            parts.append(f"{g['height'] * 1000:.0f} mm tall")
        if g.get("footprint"):
            parts.append("footprint " + " x ".join(f"{v * 1000:.0f}" for v in g["footprint"][:2]) + " mm")
        if g.get("centre"):
            parts.append(f"at x {g['centre'][0] * 1000:+.0f}, y {g['centre'][1] * 1000:+.0f} mm")
        mask, rgb = (obs.marks or {}).get(mark), getattr(obs, "rgb", None)
        if mask is not None and rgb is not None and np.asarray(mask).any():
            r, gr, b = np.median(np.asarray(rgb)[np.asarray(mask, bool)], axis=0)[:3]
            parts.append(f"camera colour rgb({r:.0f}, {gr:.0f}, {b:.0f})")
        return ", ".join(parts)

    def candidate_image(self, obs: Observation, marks: list[int]) -> str:
        """Picture for a clarification question: the candidates' isolated crops side by side, numbered."""
        path = self.run_dir / "images" / f"{obs.frame_id}_question_{'_'.join(map(str, marks))}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(montage({k: isolate(obs.rgb, obs.marks[k]) for k in marks}), cv2.COLOR_RGB2BGR))
        return str(path)

    def confirm(self, obs: Observation, mark: int, text: str):
        c = isolate(obs.rgb, obs.marks[mark])
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_crop_{mark}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
        return self.roles.confirm(c, text)

    def relation_holds(self, target: dict, anchor: dict, relation: str, frame: str = "robot") -> bool:
        """Deterministic check of a relational REFERENCE from measured positions (ARCHITECTURE §3.8)."""
        t, a = np.asarray(target["centre"]), np.asarray(anchor["centre"])
        if relation == "in":  # inside the anchor's measured top-face rectangle (rack slot, tray), not above its top
            yaw = np.radians(anchor.get("yaw_deg") or 0.0)
            d = t[:2] - a[:2]
            local = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
            ext = anchor.get("extents") or [anchor["footprint"][0], anchor["footprint"][0]]
            inside = abs(local[0]) <= ext[0] / 2 + 0.01 and abs(local[1]) <= ext[1] / 2 + 0.01
            return bool(inside and target["support_z"] <= anchor["top_z"] + 0.01)
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

    def _direction(self, relation: str, ref_xy, frame: str) -> np.ndarray:
        if frame == "camera":
            R = table_camera_pose()[:3, :3]
            right, forward = unit(R[:2, 0]), unit(R[:2, 2])  # image right; viewing direction on the floor
            dirs = {"left_of": -right, "right_of": right, "in_front_of": -forward, "behind": forward}
        else:  # robot frame: +x right, +y forward; "in front of" = between the reference and the robot
            radial = unit(ref_xy)
            dirs = {"left_of": np.array([-1.0, 0.0]), "right_of": np.array([1.0, 0.0]), "in_front_of": -radial, "behind": radial}
        return dirs[relation]
