"""Where the scenarios' 3D models come from (pure data: no Isaac Sim imports).

A scenario names a model by a short key; resolve() turns it into what USD loads:

    "instruments/metzenbaum_scissors"   -> assets/instruments/metzenbaum_scissors.usd   (in this repository)
    "i4h:Surgery Room/SurgicalTable_A/sm_surgicaltable_a01_01.usd"
                                         -> NVIDIA Isaac for Healthcare operating-room scene, version pinned below

Sources and licences (also written into each scene's metadata through the parts that use them):
  instruments   3D-COSI, 3D scans of real clinical instruments (Luijten et al., Scientific Data 10, 796, 2023;
                doi:10.5281/zenodo.10091715), CC BY 4.0; converted by scripts/convert_instruments.py
  i4h:          NVIDIA Isaac for Healthcare asset bundle v0.9.0, operating-room scene "shared_OR_without_Mark"
                (Z up, centimetres, origin on the floor); check its terms before publishing renders
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from scenarios.core import Obj, Part

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is in src/<package>/)
INSTRUMENTS = ROOT / "assets" / "instruments"
I4H_OR = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/v0.9.0/7cf86a1/"
          "Props/shared_OR_without_Mark/Collected_surgery_room_movie_with_heart_adjusted/Assets/")
CM = 0.01  # the i4h models are in centimetres
STEEL_DENSITY = 7900.0  # kg/m3, surgical stainless steel


def resolve(key: str) -> str:
    """The file or URL that USD references for a model key."""
    if key.startswith("i4h:"):
        return I4H_OR + key[4:].replace(" ", "%20")
    return str(ROOT / "assets" / f"{key}.usd")


@lru_cache(maxsize=None)
def instrument_spec(name: str) -> dict:
    """What the converter measured on one instrument: size, collision boxes, mass, source (assets/instruments/*.json)."""
    return json.loads((INSTRUMENTS / f"{name}.json").read_text())


def available_instruments() -> list[str]:
    return sorted(p.stem for p in INSTRUMENTS.glob("*.json"))


def instrument(oid: str, name: str, x: float, y: float, z: float, yaw: float, role: str = "instrument",
               support: str = "floor") -> Obj:
    """A real instrument lying flat: the scanned mesh for its look, invisible boxes for physics and planning.
    Frame: long axis along +y (finger rings or handle at +y), origin at the middle of its bounding box, on its
    lowest point (z = the surface it lies on)."""
    s = instrument_spec(name)
    parts = [Part("mesh", (0.0, 0.0, 0.0), tuple(s["size_m"]), (0.7, 0.7, 0.7), asset=f"instruments/{name}",
                  collide=False, name=f"{oid}/mesh")]
    parts += [Part("box", tuple(b[:3]), tuple(b[3:]), (0.7, 0.7, 0.7), material="metal", visible=False,
                   name=f"{oid}/collider{k}") for k, b in enumerate(s["collision_boxes_m"])]
    attrs = {"name": s["title"], "family": s["family"], "length_m": s["size_m"][1], "ring_span_m": s["size_m"][0],
             "source": s["source"], "license": s["license"]}
    return Obj(oid, s["family"], (x, y, z), yaw, parts, s["mass_kg"], [], attrs, role, support, (1.0, 0.9))
