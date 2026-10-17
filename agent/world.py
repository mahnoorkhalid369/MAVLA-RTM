"""Shared world model: objects with persistent ids, measured geometry and the VLM's own descriptions.

Matching a new measurement to a remembered object uses 3-D gates only (centre distance, top height,
footprint); no names, colours or object lists. Visibility reasoning decides "occluded" vs "absent".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

SAME_XY, SAME_TOP, SAME_FOOTPRINT = 0.03, 0.02, 0.02  # m: gates for "the same object"


@dataclass
class ObjectRecord:
    id: int
    geometry: dict  # agent.geometry.Measured as a dict (latest)
    description: dict | None = None  # VLM: {name, colour, shape, material}
    status: str = "on_surface"  # on_surface | held | occluded | missing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 1
    absent_count: int = 0
    # per-camera evidence: {"camera", "frame_id", "mark", "centre", "footprint", "top_z", "top_visible",
    # "time", "used_for_geometry"}; descriptions per camera (Gemma, hypotheses)
    observations: list = field(default_factory=list)
    descriptions_by_camera: dict = field(default_factory=dict)

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
            if o.status == "held":
                continue
            og = o.geometry
            d = float(np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]))
            if d < SAME_XY and abs(og["top_z"] - g["top_z"]) < SAME_TOP and \
                    abs(og["footprint"][0] - g["footprint"][0]) < SAME_FOOTPRINT and d < best_d:
                best, best_d = o.id, d
        return best

    def observe(self, g: dict, description: dict | None) -> int:
        oid = self.match(g)
        now = time.time()
        if oid is None:
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = ObjectRecord(oid, g, description)
        else:
            o = self.objects[oid]
            o.geometry, o.last_seen, o.times_seen, o.absent_count = g, now, o.times_seen + 1, 0
            o.status = "on_surface" if o.status != "held" else o.status
            if description:
                o.description = description
        return oid

    # cross-camera association (different viewpoints: no footprint gate, partial views differ in size)
    ASSOC_XY, ASSOC_TOP, AMBIGUOUS_RATIO = 0.03, 0.02, 1.5
    REFINE_FOOTPRINT = 0.010  # a wrist measurement replaces geometry only if its footprint agrees this well

    def associate(self, g: dict) -> tuple[str, int | None, list[int]]:
        """("matched", id, [id]) | ("new", None, []) | ("ambiguous", None, candidates): geometry only."""
        c = []
        for o in self.objects.values():
            if o.status == "held":
                continue
            og = o.geometry
            d = float(np.hypot(og["centre"][0] - g["centre"][0], og["centre"][1] - g["centre"][1]))
            if d < self.ASSOC_XY and abs(og["top_z"] - g["top_z"]) < self.ASSOC_TOP:
                c.append((d, o.id))
        c.sort()
        if not c:  # a partial view of a larger object: centre inside a stored footprint at the same top height
            for o in self.objects.values():
                og = o.geometry
                if o.status == "held" or abs(og["top_z"] - g["top_z"]) >= self.ASSOC_TOP:
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
