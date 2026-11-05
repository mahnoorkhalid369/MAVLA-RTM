"""The work cell's calibrated layout: fixed operational regions (flat markings on the table no detector sees), with
ids R1, R2, ... in their own namespace, never world-object ids (#n).

Movable scene entities are perceived through the cameras. Fixed work-cell geometry and explicitly calibrated regions
come from this layout: the scenario declares them (scenarios/<name>.py layout()) and the launcher exports them as
WORKCELL_LAYOUT, before anything imports this module, as it exports the table camera's mount (control/robot.py
TABLE_CAMERA). Unset or empty = no regions.

This module gives ids and geometry only. Which region the user's words mean is Gemma's reading: Python never compares
a region's name with words; it accepts or rejects an id ("R1") and computes geometry for it."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

REGION_ID = re.compile(r"R\d+")  # the id form, structurally (a region is referred to only by its id)


@dataclass(frozen=True)
class Region:
    id: str
    name: str  # the marking's printed label (shown to Gemma; never matched in Python)
    purpose: str
    centre: tuple  # (x, y, z) m, robot base frame; z = the surface objects rest on
    size: tuple  # (along its x, along its y) m
    yaw_deg: float = 0.0
    frame: str = "robot"
    kind: str = "planar_region"

    def local(self, xy) -> np.ndarray:
        """Robot-frame (x, y) -> the region's own axes, from its centre."""
        a = np.radians(self.yaw_deg)
        d = np.asarray(xy, float)[:2] - np.asarray(self.centre[:2], float)
        return np.array([np.cos(a) * d[0] + np.sin(a) * d[1], -np.sin(a) * d[0] + np.cos(a) * d[1]])

    def world(self, uv) -> np.ndarray:
        a = np.radians(self.yaw_deg)
        u, v = uv
        return np.asarray(self.centre[:2], float) + np.array([u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)])

    def footprint_margin(self, centre, extents, yaw_deg: float | None) -> float:
        """How far (m) a footprint is inside the region's boundary at its worst point: >= 0 entirely inside, < 0
        sticking out by that much. A rectangle (centre xy, extents along its own yaw) by its corners; a round one
        (yaw None, extents = its diameter) as a circle."""
        half = np.asarray(self.size, float) / 2
        if yaw_deg is None:
            return float(np.min(half - np.abs(self.local(centre))) - max(extents[:2]) / 2)
        a = np.radians(yaw_deg)
        ex, ey = (float(e) / 2 for e in extents[:2])
        corners = [np.asarray(centre[:2], float) + np.array([sx * ex * np.cos(a) - sy * ey * np.sin(a),
                                                             sx * ex * np.sin(a) + sy * ey * np.cos(a)])
                   for sx in (-1, 1) for sy in (-1, 1)]
        return float(min(np.min(half - np.abs(self.local(c))) for c in corners))

    def line(self) -> str:
        """The region as Gemma sees it: its id and its name only (the geometry stays here, for Python)."""
        return f"{self.id}: {self.name}"


def load(text: str | None) -> dict[str, Region]:
    """WORKCELL_LAYOUT (JSON {"scenario", "regions": [...]}) -> {id: Region}; "" / None = no regions."""
    if not text:
        return {}
    out = {}
    for r in json.loads(text)["regions"]:
        if not REGION_ID.fullmatch(r["id"]) or r.get("shape", "rectangle") != "rectangle":
            raise ValueError(f"WORKCELL_LAYOUT: region {r.get('id')!r}: ids are R<n> and only rectangles are supported")
        out[r["id"]] = Region(r["id"], r["name"], r.get("purpose", ""), tuple(r["centre"]), tuple(r["size"]),
                              float(r.get("yaw_deg", 0.0)), r.get("frame", "robot"), r.get("type", "planar_region"))
    return out


REGIONS: dict[str, Region] = load(os.environ.get("WORKCELL_LAYOUT"))


def region(ref) -> Region | None:
    """The region an id refers to ("R1", or "r1"), else None. Structural: an id, never a name."""
    t = str(ref).strip().upper() if isinstance(ref, str) else ""
    return REGIONS.get(t) if REGION_ID.fullmatch(t) else None


def is_region_id(ref) -> bool:
    """Has the form of a region id (whether or not that region exists)."""
    return isinstance(ref, str) and REGION_ID.fullmatch(ref.strip().upper()) is not None


def regions_text() -> str:
    """The layout's regions for Gemma's context ("" when there are none)."""
    if not REGIONS:
        return ""
    return "Fixed regions (refer to a region only by its id):\n" + "\n".join(r.line() for r in REGIONS.values())
