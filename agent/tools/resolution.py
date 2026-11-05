"""Target resolution: which measured object the user's words mean, and deterministic relation checks on measured
geometry (relation_holds). No word matching in Python.

resolve() is the one shared matcher (pick/place, locate, inspect, goal grounding). It splits the user's words by how
each part can be judged (Gemma, ReferenceSpec) and judges each part where it is reliable:
    appearance  Gemma looks at the candidates' isolated crops together in one picture (joint comparison)
    side        measured positions in an explicit frame (robot base, or the fixed camera's picture)
    size        measured geometry with its uncertainty: a winner only when it is ahead by more than the measurement
                error; volume when every candidate's height is measured, otherwise the footprint area (approximate)
Several candidates the measurements cannot tell apart, an appearance Gemma cannot judge, or no match -> the
Resolution is ambiguous / empty and the caller asks the user (the model never guesses). Words that are none of the
three (e.g. a relation to another object) go to Gemma's resolve over its descriptions, as before. Selection only:
whether the chosen object can be grasped is decided later by the planner (width, reach), never here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from control.robot import table_camera_pose
from perception.somark import grid, isolate, montage

if TYPE_CHECKING:
    from agent.tools.perception import Observation

SIZE_SIGMA_M = 0.002  # one standard deviation of a measured dimension (depth camera, top face / height)
SIDE_MARGIN_M = 0.03  # within this distance of the dividing line, the side is not decided by the measurement
APPEARANCE_TILES = 8  # candidates per appearance picture (one Gemma call each)


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def size_value(g: dict, measure: str, use_volume: bool) -> tuple[float, float] | None:
    """(value, standard deviation) of one size measure from measured geometry; None if not measured.
    "volume" falls back to the footprint area when use_volume is False (some height is not measured)."""
    fp, h = g.get("footprint") or [], g.get("height")
    if len(fp) < 2:
        return None
    a, b, s = float(fp[0]), float(fp[1]), SIZE_SIGMA_M
    if measure == "height":
        return (float(h), s) if h is not None else None
    if measure == "width":
        return a, s
    if measure == "volume" and use_volume:
        v = a * b * float(h)
        return v, v * np.sqrt((s / a) ** 2 + (s / b) ** 2 + (s / max(float(h), 1e-3)) ** 2)
    area = a * b  # footprint area (also the documented proxy for "volume" when a height is missing)
    return area, area * np.sqrt((s / a) ** 2 + (s / b) ** 2)


def rank_by_size(geoms: dict[int, dict], measure: str, order: str) -> dict:
    """Uncertainty-aware ranking. -> {"winner": mark | None, "tied": [marks within 2 combined sigma of the best],
    "measure": what was compared, "approximate": bool, "values": {mark: value}}. Volume only when every candidate's
    height is measured from a visible top face; otherwise the footprint area, flagged approximate."""
    use_volume = all(g.get("height") is not None and g.get("tilt_deg", 0.0) is not None for g in geoms.values())
    used = measure if (measure != "volume" or use_volume) else "footprint"
    vals = {k: size_value(g, measure, use_volume) for k, g in geoms.items()}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return {"winner": None, "tied": sorted(geoms), "measure": used, "approximate": True, "values": {}}
    sign = 1.0 if order == "largest" else -1.0
    best = max(vals, key=lambda k: sign * vals[k][0])
    tied = sorted(k for k, (v, sd) in vals.items()
                  if sign * (vals[best][0] - v) <= 2 * np.hypot(vals[best][1], sd))  # includes best itself
    unmeasured = sorted(set(geoms) - set(vals))
    return {"winner": best if len(tied) == 1 and not unmeasured else None, "tied": sorted(set(tied) | set(unmeasured)),
            "measure": used, "approximate": used != measure, "values": {k: v[0] for k, v in vals.items()}}


def side_split(centres: dict[int, list], direction, frame: str) -> dict:
    """Which measured objects lie on the given side: {"inside": [...], "borderline": [...], "outside": [...]}.
    Robot frame: the dividing line passes through the robot base. Camera frame: through the point the fixed camera's
    optical axis meets the floor (the middle of its picture)."""
    from agent.directions import direction_vector
    d = direction_vector(direction, frame)[:2]
    if np.linalg.norm(d) < 0.3:  # a vertical direction names no side on the table
        return {"inside": sorted(centres), "borderline": [], "outside": [], "ignored": "vertical direction"}
    d = unit(d)
    origin = np.zeros(2)
    if frame == "camera":
        T = table_camera_pose()
        axis, pos = T[:3, 2], T[:3, 3]
        if axis[2] < -1e-3:
            origin = (pos + (-pos[2] / axis[2]) * axis)[:2]
    out = {"inside": [], "borderline": [], "outside": []}
    for k, c in centres.items():
        proj = float(np.dot(np.asarray(c[:2], float) - origin, d))
        out["inside" if proj > SIDE_MARGIN_M else "outside" if proj < -SIDE_MARGIN_M else "borderline"].append(k)
    return out


class ResolutionTools:
    def resolve(self, obs: Observation, text: str, exclude=(), choose: bool = True):
        """The shared matcher (see the module docstring): one object, or ambiguous with candidates (the caller asks
        the user), or none. exclude: marks already rejected for this reference in this task. Every stage is logged
        ("match") with marks (this frame's labels) and world ids (persistent identities). choose=False: when several
        candidates fit the words equally, they are returned as candidates instead of a final choice among them (the
        goal check: Gemma, as the master, chooses with BIND, 2026-11-04)."""
        from agent.vlm_roles import Resolution, VLMFailure
        pool = [k for k in self.pool(obs, text) if k not in set(exclude)]
        if not pool:
            return Resolution(id=None, ambiguous=False, candidates=[], reason="every candidate was rejected")
        try:
            spec = self.reference_spec(text)
        except VLMFailure as exc:  # the words could not be split: Gemma's resolve over its descriptions, as before
            self.log("match", text=text, spec=None, reason=f"reference not parsed: {exc}")
            return self._resolve_described(obs, text, pool)
        if not (spec.appearance.strip() or spec.size or spec.side or spec.other.strip()):  # nothing to judge by
            self.log("match", text=text, frame_id=getattr(obs, "frame_id", None), spec=spec.model_dump(), stages=[], pool=pool,
                     reason="no appearance, size or side words: Gemma's resolve over its descriptions")
            return self._resolve_described(obs, text, pool)
        stages, sure, unsure = [], list(pool), []
        if spec.side is not None:
            sd = side_split({k: obs.measured[k]["centre"] for k in sure}, spec.side.direction, spec.side.frame)
            stages.append({"stage": "side", "frame": spec.side.frame, "direction": spec.side.direction, **sd})
            sure, unsure = sd["inside"], sd["borderline"]
        if spec.appearance.strip() and (sure or unsure):
            m, u = self.appearance(obs, spec.appearance, sure + unsure)
            stages.append({"stage": "appearance", "words": spec.appearance, "matches": m, "unsure": u,
                           "rejected": sorted(set(sure + unsure) - set(m) - set(u))})
            if m:  # an object Gemma could not judge matters only when none clearly fits
                u = []
            sure, unsure = [k for k in sure if k in m], sorted(set(u) | {k for k in unsure if k in m})
            # joint comparison, when needed; not when other words remain: it judges the appearance words alone and would
            # narrow out the object the remaining words point to (live 2026-10-26: "red" kept two cubes, the tray was lost)
            if len(sure) > 1 and spec.size is None and not spec.other.strip() and len(sure) <= APPEARANCE_TILES:
                best = self.compare_appearance(obs, spec.appearance, sure)
                stages.append({"stage": "joint appearance", "words": spec.appearance, "compared": sure, "matches": best})
                if best:
                    sure = [k for k in sure if k in best]
        if spec.size is not None and len(sure) + len(unsure) > 1:
            rk = rank_by_size({k: obs.measured[k] for k in sure + unsure}, spec.size.measure, spec.size.order)
            stages.append({"stage": "size", "order": spec.size.order, "requested": spec.size.measure,
                           **{k: v for k, v in rk.items() if k != "values"},
                           "values": {k: round(v * (1e9 if rk["measure"] == "volume" else 1e6 if rk["measure"] == "footprint"
                                                    else 1e3)) for k, v in rk["values"].items()},
                           "unit": {"volume": "mm3", "footprint": "mm2"}.get(rk["measure"], "mm")})
            if rk["winner"] is not None and rk["winner"] in sure:
                sure, unsure = [rk["winner"]], []
            else:
                sure, unsure = [k for k in rk["tied"] if k in sure], [k for k in rk["tied"] if k in unsure]
        rest = sure + unsure
        if len(rest) > 1 and (choose or spec.other.strip()):  # several remain: Gemma makes the final choice from the whole numbered picture, its
            # descriptions and the measured facts (any numbered object may be its answer); only if Gemma itself cannot
            # decide is the user asked (live 2026-10-26: two trays left after the look check went straight to the user)
            r = self._resolve_described(obs, text, rest)
            stages.append({"stage": "final choice" if not spec.other.strip() else "other words", "words": spec.other or text,
                           "id": r.id, "candidates": r.candidates})
            if r.id is not None:
                sure, unsure, rest = [r.id], [], [r.id]
            elif r.candidates:
                sure, unsure, rest = list(r.candidates), [], list(r.candidates)
        if len(sure) == 1 and not unsure:
            r = Resolution(id=sure[0], reason="; ".join(_stage_text(s) for s in stages) or "the only candidate")
        elif rest:
            why = ("the measurements cannot tell these apart" if len(sure) > 1 else
                   "the picture does not show clearly which of these fits")
            r = Resolution(id=None, ambiguous=True, candidates=rest, reason=why)
        else:
            r = Resolution(id=None, ambiguous=False, candidates=[], reason="no candidate fits every part of the reference")
        ids = getattr(obs, "object_ids", {}) or {}
        self.log("match", text=text, frame_id=getattr(obs, "frame_id", None), spec=spec.model_dump(), pool=pool, excluded=sorted(set(exclude)),
                 stages=stages, id=r.id, world_id=ids.get(r.id), ambiguous=r.ambiguous, candidates=r.candidates,
                 world_ids={k: ids.get(k) for k in pool}, reason=r.reason)
        self.log("tool", tool="resolve", text=text, id=r.id, ambiguous=r.ambiguous, candidates=r.candidates)
        return r

    def reference_spec(self, text: str):
        """Gemma's split of the words (appearance / size / side / other); the same words give the same split for the
        whole session (one call per distinct text)."""
        cache = self.__dict__.setdefault("_reference_specs", {})
        key = text.strip().lower()
        if key not in cache:
            cache[key] = self.roles.parse_reference(text)
        return cache[key]

    def appearance(self, obs: Observation, words: str, marks: list[int]) -> tuple[list[int], list[int]]:
        """Gemma judges each candidate's isolated crop on its own against the appearance words (batched).
        -> (matches, unsure). Live 2026-10-21: judging eight tiles in one picture, Gemma 4 E2B missed clear matches
        and called nearly every other object "unsure"; one crop at a time is the reliable form."""
        a = self.roles.appearance_each({k: isolate(obs.rgb, obs.marks[k]) for k in marks}, words)
        return [k for k in marks if a[k].match == "yes"], [k for k in marks if a[k].match == "unsure"]

    def compare_appearance(self, obs: Observation, words: str, marks: list[int]) -> list[int]:
        """Several crops each fit on their own: Gemma compares them side by side in one picture and keeps those that
        fit best (a subset of marks; [] = no narrowing)."""
        img = grid({k: isolate(obs.rgb, obs.marks[k]) for k in marks}, cols=min(4, len(marks)))
        cv2.imwrite(str(self.run_dir / "images" / f"{obs.frame_id}_appearance_{'_'.join(map(str, marks))}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        a = self.roles.match_appearance(img, words, marks)
        best = [k for k in marks if k in a.matches]
        return best if 0 < len(best) < len(marks) else []

    def _resolve_described(self, obs: Observation, text: str, pool: list[int]):
        """Gemma's resolve over its descriptions and the measured facts of the pool: it chooses AMONG the pool (live
        2026-10-26: allowed to name any object in the whole picture, it picked neighbours of the right object; measured
        4 of 13 correct on the whole picture). The pool keeps every object the words could mean: the earlier look
        check judges the whole words (kind included) and no longer narrows by part of them."""
        self.describe(obs, pool)  # no-op for objects already described
        descs = [{"id": k, **obs.descriptions.get(k, {}), "measured": self.measured_facts(obs, k)} for k in pool]
        return self.roles.resolve(obs.som, descs, text, pool)

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


def _stage_text(s: dict) -> str:
    if s["stage"] == "side":
        return f"{len(s['inside'])} on that side ({s['frame']} frame)"
    if s["stage"] == "appearance":
        return f"appearance fits {s['matches']}"
    if s["stage"] == "joint appearance":
        return f"fits best {s['matches'] or s['compared']}"
    if s["stage"] == "size":
        return f"{s['order']} by {s['measure']}" + (" (approximate)" if s["approximate"] else "")
    return f"{s['stage']}: {s.get('words', '')}"
