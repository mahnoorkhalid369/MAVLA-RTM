"""Shared world model: objects with persistent ids, measured geometry and the VLM's own descriptions.

Matching a new measurement to a remembered object uses 3-D gates only (centre distance, top height,
footprint); no names, colours or object lists. Visibility reasoning decides "occluded" vs "absent".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

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

    def mark_unseen(self, oid: int, visible_but_absent: bool) -> None:
        """Visibility reasoning: occluded / out of view gives no evidence; 2 'visible but absent' -> missing."""
        o = self.objects[oid]
        if visible_but_absent:
            o.absent_count += 1
            if o.absent_count >= 2:
                o.status = "missing"
        elif o.status == "on_surface":
            o.status = "occluded"

    def summary(self) -> list[dict]:
        return [{"id": o.id, "status": o.status, "description": o.description,
                 "centre_mm": [round(v * 1000) for v in o.geometry["centre"]],
                 "height_mm": round(o.geometry["height"] * 1000), "footprint_mm": [round(v * 1000) for v in o.geometry["footprint"]]}
                for o in self.objects.values()]

    def to_dict(self) -> dict:
        return {"objects": {k: asdict(v) for k, v in self.objects.items()}, "next_id": self.next_id, "holding": self.holding}
