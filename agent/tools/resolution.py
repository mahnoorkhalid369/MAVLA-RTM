"""Target resolution: which measured object the user's words mean (Gemma resolve, lexical cross-check over
Gemma's own descriptions, side-by-side comparison, confirmation on the isolated crop) and deterministic
relation checks on measured geometry (relation_holds)."""

from __future__ import annotations

import re
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
    STOPWORDS = {"the", "a", "an", "of", "and", "that", "this", "is", "one", "with"}  # English function words

    @classmethod
    def words(cls, text: str) -> set[str]:
        ws = {w for w in re.split(r"[^a-z]+", text.lower()) if w and w not in cls.STOPWORDS}
        return ws | {w[:-1] for w in ws if len(w) > 3 and w.endswith("s")}  # plural -> singular

    def lexical_candidates(self, obs: Observation, text: str) -> list[int]:
        """Marks (among the reference's candidates) whose VLM description contains every word of the
        user's reference (the VLM's own words; deterministic, no word lists)."""
        need = self.words(text)
        pool = set(self.pool(obs, text)) if obs.marks else set(obs.descriptions)
        return [k for k, d in obs.descriptions.items() if k in pool and
                need and need <= self.words(" ".join(str(d.get(f, "")) for f in ("name", "colour", "shape", "material")))]

    def resolve(self, obs: Observation, text: str):
        """VLM resolution over the descriptions, cross-checked: if the pick is not among the objects whose
        description contains every word of the reference, or several objects do, the VLM compares those
        objects' isolated crops side by side and must pick one (or say ambiguous -> ask the user)."""
        pool = self.pool(obs, text)
        self.describe(obs, pool)  # no-op for objects already described
        descs = [{"id": k, **obs.descriptions.get(k, {})} for k in pool]
        r = self.roles.resolve(obs.som, descs, text, pool)
        lex = self.lexical_candidates(obs, text)
        need_compare = bool(lex) and (len(lex) > 1 or (r.id is not None and r.id not in lex))
        self.log("tool", tool="lexical_check", text=text, resolver_id=r.id, lexical=lex, compare=need_compare)
        if not need_compare:
            return r
        ids = sorted(set(lex) | ({r.id} if r.id is not None else set()))
        img = montage({k: isolate(obs.rgb, obs.marks[k]) for k in ids})
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_compare_{'_'.join(map(str, ids))}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        c = self.roles.compare(img, text, ids)
        # second, independent cue from the VLM's own descriptions: the candidate whose description adds
        # the fewest words beyond the reference is the most specific match. Accept the visual choice only
        # if it is the single most specific candidate; if the cues disagree or tie -> ambiguous (ask).
        need = self.words(text)
        extra = {k: len(self.words(" ".join(str(obs.descriptions.get(k, {}).get(f, "")) for f in ("name", "colour", "shape"))) - need)
                 for k in ids}
        best = [k for k in ids if extra[k] == min(extra.values())]
        agree = c.id is not None and best == [c.id]
        self.log("tool", tool="compare", text=text, ids=ids, extra_words=extra, most_specific=best, agree=agree, **c.model_dump())
        if agree:
            return c
        return type(c)(id=None, ambiguous=True, candidates=ids,
                       reason=f"visual choice {c.id} vs most specific description {best}: cues disagree")

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
