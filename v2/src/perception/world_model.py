"""Shared world model: objects with persistent ids, measured geometry and the VLM's own descriptions.

Ids are given once, at the first look, and kept: a table-camera frame is matched to the remembered objects as a
whole and one to one (observe_frame), first by the object's outline in the image (the camera is fixed, so an object
that did not move keeps its outline even when its measured size wobbles: the serving tray with a mug in it came out
0.37 or 0.27 m long, live 2026-11-11, and got a new id at every look), then by the nearest remembered object for one
that moved a little (knocked, dropped). Only a detection that matches nothing is a new object. Geometry only: no
names, colours or object lists. Visibility reasoning decides "occluded" vs "absent".
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from perception.capture import ObservationError

SAME_XY, SAME_TOP, SAME_FOOTPRINT = 0.03, 0.02, 0.02  # m: gates for "the same object"
HIDDEN_TOL = 0.010  # m: a support or top this much off the remembered one is a hidden part, not a change
MASK_IOU = 0.30  # table camera: outlines overlapping this much (intersection over union) are the same object
NEAR_M, MOVED_TOP = 0.06, 0.04  # ... and its centre within NEAR_M (or half its long side, if larger); a moved one:
MOVED_M = 0.10  # the nearest remembered object within MOVED_M (or half its long side), top height within MOVED_TOP


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return None if not len(rows) else (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ba, bb = _box(a), _box(b)
    if ba is None or bb is None or ba[1] <= bb[0] or bb[1] <= ba[0] or ba[3] <= bb[2] or bb[3] <= ba[2]:
        return 0.0
    r0, r1, c0, c1 = min(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), max(ba[3], bb[3])
    a, b = a[r0:r1, c0:c1], b[r0:r1, c0:c1]
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _apart(a: dict, b: dict) -> float:
    return float(np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]))


def appearance_agrees(a, b):
    x, y = a.get('colour_signature'), b.get('colour_signature')
    return x is None or y is None or float(np.sqrt(np.array(x) * np.array(y)).sum() / 3) >= .65


def _reach(g: dict, base: float) -> float:
    return max(base, max(g["footprint"]) / 2)


def covered_by(og: dict, others) -> bool:
    """Another measurement of the same frame rests on the remembered object og: it stands over og's footprint, its
    bottom at or above og's top. og's top face is then hidden, and a measurement of og is a partial (side) view."""
    reach = max(og["footprint"]) / 2 + HIDDEN_TOL
    return any(np.hypot(o["centre"][0] - og["centre"][0], o["centre"][1] - og["centre"][1]) < reach
               and o.get("support_z", 0.0) >= og["top_z"] - 1.5 * HIDDEN_TOL and o["top_z"] > og["top_z"] + HIDDEN_TOL
               for o in others)


def same_place(a: dict, b: dict) -> bool:
    """Two measured geometries (perception.geometry.Measured dicts) are the same object at the same place:
    centre within SAME_XY, top height within SAME_TOP, long footprint side within SAME_FOOTPRINT. Used by the
    world model (WorldModel.match) and by the session's description memory (agent.tools.DescriptionMemory)."""
    return (np.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1]) < SAME_XY
            and abs(a["top_z"] - b["top_z"]) < SAME_TOP and abs(a["footprint"][0] - b["footprint"][0]) < SAME_FOOTPRINT)


@dataclass
class ObjectRecord:
    id: int
    geometry: dict  # perception.geometry.Measured as a dict (latest)
    description: dict | None = None  # VLM: {name, colour, shape, material}
    status: str = "on_surface"  # on_surface | held | occluded | missing | merged (verified alias)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 1
    absent_count: int = 0
    # per-camera evidence: {"camera", "frame_id", "mark", "centre", "footprint", "top_z", "top_visible",
    # "time", "used_for_geometry"}; descriptions per camera (Gemma, hypotheses)
    observations: list = field(default_factory=list)
    descriptions_by_camera: dict = field(default_factory=dict)
    mask: np.ndarray | None = field(default=None, repr=False)  # its outline in the latest table-camera frame

    def box(self):
        """Collision box (control.collision.Box) from the measured geometry, named obj_<id>."""
        from control.collision import Box

        g = self.geometry
        a = np.radians(g["yaw_deg"] or 0.0)
        R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        half = np.array([g["extents"][0] / 2, g["extents"][1] / 2, g["height"] / 2])
        return Box(f"obj_{self.id}", np.asarray(g["centre"], float), R, half)


class WorldModel:
    def __init__(self) -> None:
        self.objects: dict[int, ObjectRecord] = {}
        self.next_id = 1
        self.holding: int | None = None

    def match(self, g: dict) -> int | None:
        best, best_d = None, np.inf
        for o in self.objects.values():
            if o.status in ("held", "merged"):
                continue
            og = o.geometry
            if not same_place(og, g):
                continue
            d = float(np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]))
            if d < best_d:
                best, best_d = o.id, d
        return best

    def observe(self, g: dict, description: dict | None, others=()) -> int:
        """Fold one table-camera measurement in. others: the frame's other measurements. Two occlusion cases keep
        what was measured before instead of a partial view (live 2026-10-21: an object placed on a 71 mm support;
        the support was then seen as a 60 mm sliver and the placed object's bottom 18 mm too high, so on() failed):
          covered     another object rests on the remembered one: its geometry is kept as it was
          bottom hid  same top, but the support appears higher than remembered: the lower part is hidden; the
                      remembered support (and so the height) is kept, the new top and position are used
          merged      same top, but the support appears lower than remembered while another remembered object fills
                      the gap below (its top at the remembered support): the camera saw the stack as one object;
                      the remembered support is kept (live 2026-10-21: an object placed on a 101 mm support)"""
        oid = self.match(g)
        if oid is None:
            return self._new(g, description)
        self._update(oid, g, description, others)
        return oid

    def observe_frame(self, measured: dict, masks: dict, descriptions: dict | None = None) -> dict:
        """One whole table-camera frame -> {mark: id}, one to one. measured / masks: mark -> geometry / outline.
        1. same outline: mask overlap >= MASK_IOU with a remembered object near it (best overlaps first)
        2. moved a little: the nearest remaining remembered object within MOVED_M at about the same top height
        3. anything else is a new object. A remembered object not matched keeps its id (hidden, or out of view)."""
        descriptions, others = descriptions or {}, list(measured.values())
        # Identical image silhouettes must not all acquire a new world position because of bad extrinsics/depth.
        shifts = []
        for k, mask in masks.items():
            if k not in measured:
                continue
            candidates = [(mask_iou(o.mask, mask), o) for o in self.objects.values()
                          if o.mask is not None and o.mask.shape == mask.shape and o.status not in ('held', 'merged')]
            if candidates:
                score, old = max(candidates, key=lambda item: item[0])
                if score > .6:
                    shifts.append(np.array([*measured[k]['centre'][:2], measured[k]['top_z']])
                                  - np.array([*old.geometry['centre'][:2], old.geometry['top_z']]))
        if len(shifts) >= 3:
            shifts = np.array(shifts)
            shift = np.median(shifts, axis=0)
            coherent = np.linalg.norm(shifts - shift, axis=1) < .012
            if coherent.sum() >= max(3, len(shifts) * .6) and np.linalg.norm(shift) > .025:
                raise ObservationError('scene-wide coordinate jump for unchanged silhouettes: '
                                       + str(np.round(shift * 1000, 1).tolist()) + ' mm; recheck camera/depth calibration')
        free = {o.id for o in self.objects.values() if o.status not in ("held", "merged")}
        pairs: dict = {}

        def assign(cands) -> None:
            for _, k, oid in sorted(cands, key=lambda c: c[0]):
                if k not in pairs and oid in free:
                    pairs[k] = oid
                    free.discard(oid)

        assign([(-iou, k, oid) for k, m in masks.items() if k in measured for oid in free
                if (og := self.objects[oid].geometry) is not None and self.objects[oid].mask is not None
                and self.objects[oid].mask.shape == m.shape and _apart(og, measured[k]) < _reach(og, NEAR_M)
                and abs(og['top_z'] - measured[k]['top_z']) < MOVED_TOP
                and appearance_agrees(og, measured[k])
                and (iou := mask_iou(self.objects[oid].mask, m)) >= MASK_IOU])
        assign([(d, k, oid) for k, g in measured.items() if k not in pairs for oid in free
                if (d := _apart(og := self.objects[oid].geometry, g)) < _reach(og, MOVED_M)
                and abs(og["top_z"] - g["top_z"]) < MOVED_TOP and appearance_agrees(og, g)
                and all(abs(a - b) < max(.025, a * .4) for a, b in zip(og['footprint'], g['footprint']))])
        out = {}
        for k, g in measured.items():
            if k in pairs:
                self._update(pairs[k], g, descriptions.get(k), others)
            else:
                pairs[k] = self._new(g, descriptions.get(k))
            self.objects[pairs[k]].mask = masks[k].copy() if k in masks else None
            out[k] = pairs[k]
        return out

    def _new(self, g: dict, description: dict | None) -> int:
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = ObjectRecord(oid, deepcopy(g), description)
        return oid

    def _update(self, oid: int, g: dict, description: dict | None, others) -> None:
        original = g
        g = deepcopy(g)
        o = self.objects[oid]
        og = o.geometry
        if covered_by(og, [x for x in others if x is not original]):
            g = og
        elif (_apart(og, g) < SAME_XY and abs(og['top_z'] - g['top_z']) < SAME_TOP
              and not g.get('support_confident', False) and og.get('support_confident', True)
              and og.get("support_z") is not None and g.get("support_z") is not None) and (
                g["support_z"] > og["support_z"] + HIDDEN_TOL
                or (g["support_z"] < og["support_z"] - HIDDEN_TOL and self._fills_gap(oid, og))):
            g = dict(g, support_z=og["support_z"], height=g["top_z"] - og["support_z"],
                     support_confident=og.get('support_confident', True))
            g["centre"] = [g["centre"][0], g["centre"][1], og["support_z"] + g["height"] / 2]
        o.geometry, o.last_seen, o.times_seen, o.absent_count = g, time.time(), o.times_seen + 1, 0
        o.status = "on_surface" if o.status != "held" else o.status
        if description:
            o.description = description

    def _fills_gap(self, oid: int, og: dict) -> bool:
        """Another remembered object stands right under og: its top at og's support, within og's footprint."""
        reach = max(og["footprint"]) / 2 + HIDDEN_TOL
        return any(o.id != oid and o.status not in ("held", "merged")
                   and abs(o.geometry["top_z"] - og["support_z"]) <= 1.5 * HIDDEN_TOL
                   and np.hypot(o.geometry["centre"][0] - og["centre"][0], o.geometry["centre"][1] - og["centre"][1]) < reach
                   for o in self.objects.values())

    # cross-camera association (different viewpoints: no footprint gate, partial views differ in size)
    ASSOC_XY, ASSOC_TOP, AMBIGUOUS_RATIO = 0.03, 0.02, 1.5
    REFINE_FOOTPRINT = 0.010  # a wrist measurement replaces geometry only if its footprint agrees this well

    def associate(self, g: dict) -> tuple[str, int | None, list[int]]:
        """("matched", id, [id]) | ("new", None, []) | ("ambiguous", None, candidates): geometry only."""
        c = []
        for o in self.objects.values():
            if o.status in ("held", "merged"):
                continue
            og = o.geometry
            d = float(np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]))
            if d < self.ASSOC_XY and abs(og["top_z"] - g["top_z"]) < self.ASSOC_TOP:
                c.append((d, o.id))
        c.sort()
        if not c:  # a partial view of a larger object: centre inside a stored footprint at the same top height
            for o in self.objects.values():
                og = o.geometry
                if o.status in ("held", "merged") or abs(og["top_z"] - g["top_z"]) >= self.ASSOC_TOP:
                    continue
                a = np.radians(og.get("yaw_deg") or 0.0)
                d = np.array(g["centre"][:2]) - np.array(og["centre"][:2])
                local = np.array([d @ [np.cos(a), np.sin(a)], d @ [-np.sin(a), np.cos(a)]])
                if np.all(np.abs(local) <= np.array(og["extents"]) / 2 + 0.015):
                    c.append((float(np.hypot(*d)), o.id))
            if len(c) == 1:
                return "matched", c[0][1], [c[0][1]]
            if len(c) > 1:
                return "ambiguous", None, [i for _, i in sorted(c)]
            return "new", None, []
        if len(c) > 1 and c[1][0] < self.AMBIGUOUS_RATIO * max(c[0][0], 0.005):
            return "ambiguous", None, [i for _, i in c]
        return "matched", c[0][1], [c[0][1]]

    def associate_partial(self, points: np.ndarray, *, diagnostics=None) -> tuple[str, int | None, list[int]]:
        """Identity-only binding of an oblique visible surface, without inventing its hidden top/centre.

        A substantial vertical body patch must lie almost entirely inside exactly one remembered object
        volume. This never updates stored geometry and never authorizes a grasp; the top-view association
        and full geometry gates remain mandatory before descent.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 30:
            return "new", None, []
        candidates = []
        for o in self.objects.values():
            if o.status not in ("on_surface", "occluded"):
                continue
            g = o.geometry
            a = np.radians(g.get("yaw_deg") or 0.0)
            R = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
            local = (points - np.asarray(g["centre"])) @ R
            half = np.asarray([*g["extents"], g["height"]]) / 2
            inside = np.all(np.abs(local) <= half + .010, axis=1)
            if diagnostics is not None:
                diagnostics[o.id] = {"inside_fraction": round(float(inside.mean()), 4),
                                     "inside_points": int(inside.sum()), "required_fraction": .90}
            if inside.mean() < .90 or inside.sum() < 30:
                continue
            lo, hi = np.percentile(points[inside, 2], [2, 98])
            if diagnostics is not None:
                diagnostics[o.id]["vertical_span_mm"] = round(float(hi-lo)*1000, 1)
            if hi-lo < min(.030, g["height"] * .25):
                continue  # a cap/label sliver or tabletop patch is insufficient body evidence
            candidates.append(o.id)
        if len(candidates) == 1:
            return "matched", candidates[0], candidates
        return ("ambiguous", None, candidates) if candidates else ("new", None, [])

    def add_observation(self, g: dict, camera: str, frame_id: str, mark: int, description: dict | None = None) -> tuple[str, int | None]:
        """Fold one camera's measurement into the world model. Measurements (depth) and descriptions (Gemma)
        are kept apart; a wrist measurement refines an object's geometry only if its top face was visible and
        its footprint agrees with the stored one (otherwise it is evidence only, e.g. a partial view)."""
        status, oid, cands = self.associate(g)
        rec = {"camera": camera, "frame_id": frame_id, "mark": mark, "centre": list(g["centre"]), "footprint": list(g["footprint"]),
               "top_z": g["top_z"], "top_visible": g.get("tilt_deg") is not None, "time": time.time(), "used_for_geometry": False}
        if status == "ambiguous":
            return status, None  # not merged: the caller reports the candidates
        if status == "new":
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = ObjectRecord(oid, g, description)
            rec["used_for_geometry"] = True
        else:
            o = self.objects[oid]
            og = o.geometry
            consistent = rec["top_visible"] and all(abs(a - b) <= self.REFINE_FOOTPRINT for a, b in zip(g["footprint"], og["footprint"]))
            if consistent:
                o.geometry = g
                rec["used_for_geometry"] = True
            o.last_seen, o.times_seen, o.absent_count = time.time(), o.times_seen + 1, 0
        o = self.objects[oid]
        o.observations.append(rec)
        if description:
            o.descriptions_by_camera[camera] = description
            if o.description is None:
                o.description = description
        return status, oid

    def summary(self) -> list[dict]:
        return [{"id": o.id, "status": o.status, "description": o.description,
                 "centre_mm": [round(v * 1000) for v in o.geometry["centre"]],
                 "height_mm": round(o.geometry["height"] * 1000), "footprint_mm": [round(v * 1000) for v in o.geometry["footprint"]]}
                for o in self.objects.values()]
