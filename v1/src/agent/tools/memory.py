"""Description memory of one session: Gemma's descriptions reused for an unchanged object (same place, same size,
same mean colour), so the same object is not described again by every instruction."""

from __future__ import annotations

import numpy as np

from perception.world_model import same_place


def mean_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return rgb[mask.astype(bool)].reshape(-1, 3).mean(axis=0) if mask.any() else np.zeros(3)


class DescriptionMemory:
    """VLM descriptions kept across the instructions of one session. A description is reused only for a
    measurement at the same place with the same size (WorldModel gates: 30 mm, top 20 mm, footprint 20 mm)
    AND the same mean colour inside the mask (every channel within COLOUR_TOL of 255): anything that
    moved, was covered or looks different is described again."""

    COLOUR_TOL = 20.0

    def __init__(self) -> None:
        self.entries: list[tuple[dict, np.ndarray, dict]] = []

    @staticmethod
    def _same_place(a: dict, b: dict) -> bool:
        return same_place(a, b)

    def lookup(self, g: dict, colour: np.ndarray) -> dict | None:
        for eg, ec, d in self.entries:
            if self._same_place(eg, g) and np.all(np.abs(ec - colour) <= self.COLOUR_TOL):
                return dict(d)
        return None

    def store(self, g: dict, colour: np.ndarray, description: dict) -> None:
        colour = np.asarray(colour, float)
        self.entries = [e for e in self.entries if not (self._same_place(e[0], g) and np.all(np.abs(e[1] - colour) <= self.COLOUR_TOL))]
        self.entries.append((dict(g), colour, dict(description)))
