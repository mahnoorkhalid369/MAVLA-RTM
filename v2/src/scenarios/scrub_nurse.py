"""Scenario: robotic scrub nurse, surgical instrument handover (v2.0).

Modelled on the RoboNurse-VLA setup (IEEE 2025, document 11246030): an arm with a parallel gripper beside a
blue-draped back table on which surgical instruments lie side by side, ring handles towards the surgeon; spoken
requests; a depth camera; the instrument is handed into the surgeon's open hand. Here the Epson VT6L stands at the
back of the draped table. The instruments are 3D scans of real clinical instruments (3D-COSI). They lie across two
low silicone rests, as a scrub nurse lays them on a rolled towel, so the 2F-85's pads close around a shaft with room
under it. The surgeon stands at the table's front-left corner with an open gloved hand over the drape: the handover
point. Beyond the arm's reach: the operating table with a draped patient, the ceiling surgical light, the anaesthesia
machine with the anaesthetist, an instrument trolley (NVIDIA Isaac for Healthcare models).

Robot frame: +y = forward (towards the table camera), +x = the robot's right, so the camera sees +x on its LEFT. The
table top is z = 0. Every instrument and the surgeon's hand are in the arm's top-down ring; the patient is not.

Difficulty: easy = 5 instruments of different kinds, square to the robot; medium = 7 instruments with look-alikes
(straight vs Metzenbaum scissors, mosquito vs Pean clamps) and one empty rest position for returns; hard = medium at
small random angles plus a folded gauze sponge on the drape.

What the agent knows: instruments, the rests and the surgeon's hand through the cameras only. The fixed rest positions
1-8 (layout() below) are exported as WORKCELL_LAYOUT, like the work cell's mat cells. Ground truth (instrument names,
look-alikes) is evaluation-only.
"""

from __future__ import annotations

import numpy as np

from scenarios.assets import CM, instrument
from scenarios.core import Fixture, Light, Part, Region, Scenario
from scenarios.parts import box_obj, surface

ID = "scrub_nurse"
TITLE = "Robotic scrub nurse: surgical instrument handover"
VERSION = "2.0"

# ----------------------------------------------------------------------------------------------- layout (m)
TABLE_CAMERA = ((0.0, 1.30, 1.45), (0.0, 0.42, 0.0))  # overhead, on a stand at the table's front
TABLE = dict(centre=(0.0, 0.35), size=(1.60, 1.30), thickness=0.05, height=0.80, no_go_color=(0.10, 0.30, 0.60))
FLOOR = -TABLE["height"]
ROW_Y = 0.52  # the instruments' centres, forward of the robot base
PITCH = 0.11  # rest positions along x: the open pads (85 mm) clear the neighbours' finger-ring shanks
SLOTS = 8
REST = dict(offset=0.045, width=0.016, height=0.020)  # two ridges at ROW_Y +- offset
HAND = dict(xy=(-0.52, 0.66), palm=(0.085, 0.10, 0.022), z=0.095)  # surgeon's open hand (palm bottom z)

DRAPE, DRAPE_EDGE = (0.16, 0.42, 0.78), (0.12, 0.34, 0.66)  # sterile blue
GLOVE, GOWN, CAP = (0.92, 0.83, 0.68), (0.38, 0.58, 0.52), (0.20, 0.45, 0.80)
SILICONE, STEEL_GREY = (0.26, 0.28, 0.31), (0.55, 0.57, 0.60)

# ------------------------------------------------------------------------------------------ difficulty
EASY = ["scalpel_no10", "tissue_forceps", "straight_scissors", "mosquito_hemostat", "needle_holder"]
FULL = EASY + ["metzenbaum_scissors", "pean_clamp"]
CONFIG = {
    "easy": dict(instruments=EASY, slots=[0, 2, 3, 5, 7], yaw_jitter=0.0, gauze=False),
    "medium": dict(instruments=FULL, slots=None, yaw_jitter=0.0, gauze=False),  # None: 7 of 8, one left free
    "hard": dict(instruments=FULL, slots=None, yaw_jitter=7.0, gauze=True),
}
FEATURES = {
    "easy": "5 instruments of different kinds, a free rest position between most of them, square to the robot",
    "medium": "7 instruments with look-alikes (straight vs Metzenbaum scissors, mosquito vs Pean clamps), one free position",
    "hard": "medium at random angles up to 7 deg, plus a folded gauze sponge on the drape",
}

CEILING = FLOOR + 3.0
# NVIDIA Isaac for Healthcare models (scenarios/assets.py): id, model key, where its origin goes (x, y, z), yaw, size.
# The models' origin is on the floor, except the ceiling light's (it hangs from the ceiling).
OR_MODELS = [
    ("operating_table", "Surgery Room/SurgicalTable_A/sm_surgicaltable_a01_01.usd", (-1.80, 0.30, FLOOR), 90.0,
     (2.21, 0.69, 0.94)),
    ("surgical_light", "Surgery Room/SurgicalLightCeilingArms_A/sm_surgicallightceilingarms_a01_01.usd",
     (-2.90, 0.60, CEILING), 90.0, (3.96, 1.49, 2.05)),
    ("anaesthesia_machine", "Surgery Room/AnesthesiaMachine_A/sm_anesthesiamachine_a01_01.usd", (-1.85, -1.35, FLOOR),
     0.0, (2.26, 0.85, 1.70)),
    ("anaesthetist", "Library/Misc/CharacterWorker/sm_maledoctor_a01_01.usd", (-1.80, -0.75, FLOOR), 0.0,
     (0.55, 0.64, 1.71)),
    ("instrument_trolley", "Surgery Room/InstrumentTrolley_A/sm_instrumenttrolley_a01_01.usd", (1.30, -1.20, FLOOR),
     0.0, (0.80, 0.54, 1.07)),
]


def _slot_x(k: int) -> float:
    return (k - (SLOTS - 1) / 2) * PITCH


def _box(centre, size, color, material="painted", name="") -> Part:
    """A visual-only box (decor)."""
    return Part("box", tuple(centre), tuple(size), color, material=material, collide=False, name=name)


# ---------------------------------------------------------------------------------------------- fixtures
def _rests() -> Fixture:
    length = SLOTS * PITCH + 0.05
    parts = [Part("box", (0.0, ROW_Y + s * REST["offset"], REST["height"] / 2), (length, REST["width"], REST["height"]),
                  SILICONE, material="rubber", name=f"instrument_rest/ridge{s:+d}") for s in (-1, 1)]
    return Fixture("instrument_rest", "instrument_rest", (0.0, 0.0, 0.0), 0.0, parts,
                   description="two dark silicone ridges the instruments lie across")


def _hand_yaw() -> float:
    hx, hy = HAND["xy"]
    return float(np.degrees(np.arctan2(-hx, hy)))  # the hand's local +y points away from the robot base


def _hand() -> Fixture:
    """The surgeon's open gloved hand over the drape, palm up, fingers towards the robot; forearm and gown sleeve
    run outwards to the surgeon (colliders: the arm plans around them)."""
    pw, pl, pt = HAND["palm"]
    z0 = HAND["z"]

    def box(c, s, col, name, mat="rubber"):
        return Part("box", (c[0], c[1], z0 + c[2]), s, col, material=mat, name=f"surgeon_hand/{name}")

    parts = [box((0.0, 0.0, pt / 2), (pw, pl, pt), GLOVE, "palm")]
    for k, (dx, length) in enumerate(zip((-0.0285, -0.0095, 0.0095, 0.0285), (0.062, 0.072, 0.070, 0.056))):
        parts.append(box((dx, -pl / 2 - length / 2, 0.008), (0.017, length, 0.016), GLOVE, f"finger{k}"))
    parts += [box((pw / 2 + 0.012, -0.012, 0.009), (0.020, 0.055, 0.018), GLOVE, "thumb"),
              box((0.0, pl / 2 + 0.150, 0.026), (0.062, 0.30, 0.055), GLOVE, "forearm"),
              box((0.0, pl / 2 + 0.36, 0.035), (0.10, 0.14, 0.09), GOWN, "sleeve", "cardboard")]
    return Fixture("surgeon_hand", "surgeon_hand", (*HAND["xy"], 0.0), _hand_yaw(), parts,
                   description="the surgeon's open gloved hand, palm up: the handover point")


# ------------------------------------------------------------------------------------------------ decor
def _surgeon() -> Fixture:
    """The surgeon whose hand is the handover point: simple shapes joined to that hand, beyond the arm's reach."""
    u = np.asarray(HAND["xy"]) / np.hypot(*HAND["xy"])
    c = np.asarray(HAND["xy"]) + u * 0.85
    parts = [_box((0, 0, (FLOOR + 0.10) / 2), (0.36, 0.22, 0.10 - FLOOR), GOWN, "cardboard", "legs"),
             _box((0, 0, 0.36), (0.46, 0.26, 0.56), GOWN, "cardboard", "torso"),
             _box((-0.25, -0.12, 0.42), (0.10, 0.30, 0.10), GOWN, "cardboard", "upper_arm"),
             Part("cylinder", (0, 0, 0.78), (0.19, 0.19, 0.24), (0.86, 0.70, 0.58), material="rubber", collide=False,
                  name="head"),
             Part("cylinder", (0, 0, 0.92), (0.21, 0.21, 0.06), CAP, material="cardboard", collide=False, name="cap"),
             _box((0, -0.095, 0.75), (0.15, 0.02, 0.09), (0.55, 0.75, 0.90), "cardboard", "mask")]
    return Fixture("surgeon", "person", (float(c[0]), float(c[1]), 0.0), _hand_yaw(), parts)


def _patient() -> Fixture:
    """A draped patient on the operating table (the table's top is 0.94 m above the floor)."""
    x, y, _ = OR_MODELS[0][2]
    top = FLOOR + 0.94
    parts = [_box((0, -0.05, top + 0.10), (0.50, 1.55, 0.20), DRAPE, "cardboard", "patient_drape"),
             _box((0, 0.10, top + 0.205), (0.16, 0.14, 0.01), (0.88, 0.70, 0.62), "rubber", "surgical_site"),
             Part("cylinder", (0, 0.86, top + 0.09), (0.19, 0.19, 0.17), (0.88, 0.72, 0.60), material="rubber",
                  collide=False, name="patient_head"),
             _box((0, 0.70, top + 0.30), (0.62, 0.02, 0.40), DRAPE, "cardboard", "anaesthesia_screen")]
    return Fixture("patient", "patient", (x, y, 0.0), 0.0, parts)


def _or_models() -> list[Fixture]:
    """NVIDIA's operating-room models, visual only."""
    return [Fixture(fid, fid, pos, yaw, [Part("mesh", (0, 0, 0), size, STEEL_GREY, asset=f"i4h:{key}", asset_scale=CM,
                                              collide=False, name=fid)])
            for fid, key, pos, yaw, size in OR_MODELS]


def _room() -> list:
    """The back table under the drape, the camera stand, floor and walls (simple shapes), then the OR models."""
    (cx, cy), (L, W), t = TABLE["centre"], TABLE["size"], TABLE["thickness"]
    table = [_box((cx, cy, -t / 2 - 0.0004), (L, W, t - 0.0008), STEEL_GREY, "metal", "slab")]
    table += [_box((cx + sx * (L / 2 - 0.05), cy + sy * (W / 2 - 0.05), FLOOR + (TABLE["height"] - t) / 2),
                   (0.04, 0.04, TABLE["height"] - t), (0.70, 0.72, 0.75), "metal", f"leg{sx:+d}{sy:+d}")
              for sx in (-1, 1) for sy in (-1, 1)]
    y0, y1 = 0.15, cy + W / 2  # the drape covers the front of the table; the robot stands on bare steel behind it
    skirt = 0.42
    drape = [_box((cx, y1 + 0.002, -skirt / 2 - 0.001), (L + 0.008, 0.004, skirt), DRAPE_EDGE, "cardboard", "skirt_front")]
    drape += [_box((s * (L / 2 + 0.002), (y0 + y1) / 2, -skirt / 2 - 0.001), (0.004, y1 - y0, skirt), DRAPE_EDGE,
                   "cardboard", f"skirt{s:+d}") for s in (-1, 1)]
    cam_x, cam_y, cam_z = TABLE_CAMERA[0]  # camera stand: pole from the floor, boom to the camera
    stand = [_box((cam_x, cam_y + 0.12, (FLOOR + cam_z + 0.10) / 2), (0.04, 0.04, cam_z + 0.10 - FLOOR),
                  (0.25, 0.26, 0.28), "metal", "pole"),
             _box((cam_x, cam_y + 0.06, cam_z + 0.08), (0.03, 0.15, 0.03), (0.25, 0.26, 0.28), "metal", "boom")]
    wall = (0.62, 0.76, 0.74)
    walls = [_box((0.0, -1.85, FLOOR + 1.5), (7.0, 0.06, 3.0), wall, "painted", "back_wall"),
             _box((2.6, 0.2, FLOOR + 1.5), (0.06, 4.2, 3.0), wall, "painted", "side_wall"),
             _box((0.0, 0.2, CEILING + 0.03), (7.0, 4.2, 0.06), (0.93, 0.94, 0.95), "painted", "ceiling")]
    return [surface("room_floor", (0.0, 0.0), (8.0, 8.0), (0.55, 0.60, 0.60), z=FLOOR + 0.0003, material="epoxy"),
            Fixture("table", "table", (0.0, 0.0, 0.0), 0.0, table),
            Fixture("drape_skirts", "drape", (0.0, 0.0, 0.0), 0.0, drape),
            Fixture("camera_stand", "camera_stand", (0.0, 0.0, 0.0), 0.0, stand),
            Fixture("walls", "room", (0.0, 0.0, 0.0), 0.0, walls),
            surface("table_top", TABLE["centre"], TABLE["size"], STEEL_GREY, z=0.0003, material="metal"),
            surface("drape", (cx, (y0 + y1) / 2), (L, y1 - y0), DRAPE, z=0.0006, material="cardboard"),
            _surgeon(), _patient(), *_or_models()]


def layout() -> list[dict]:
    """Calibrated fixed regions (robot base frame, m; the table top is z = 0): the instrument area and its eight rest
    positions, numbered 1 = the robot's left (-x). Geometry and ids only."""
    regions = [{"name": "instrument area", "purpose": "where the instruments lie across the two rests",
                "centre": [0.0, ROW_Y, REST["height"]], "size": [SLOTS * PITCH, 2 * REST["offset"] + REST["width"]]}]
    regions += [{"name": f"instrument position {k + 1}", "purpose": "one instrument's place on the rests "
                 "(1 = the robot's left)", "centre": [_slot_x(k), ROW_Y, REST["height"]], "size": [PITCH, 0.16]}
                for k in range(SLOTS)]
    return [{"id": f"R{k}", "frame": "robot", "type": "planar_region", "shape": "rectangle", "yaw_deg": 0.0, **r}
            for k, r in enumerate(regions, 1)]


# ------------------------------------------------------------------------------------------------ build
def build(difficulty: str, seed: int) -> Scenario:
    cfg = CONFIG[difficulty]
    rng = np.random.default_rng(seed)
    names = [cfg["instruments"][k] for k in rng.permutation(len(cfg["instruments"]))]
    if cfg["slots"] is not None:
        slots = list(cfg["slots"])
    else:  # 7 of the 8 positions; the free one is chosen by the seed (a place to return an instrument to)
        free = int(rng.integers(SLOTS))
        slots = [k for k in range(SLOTS) if k != free]

    objects = []
    for name, k in zip(names, slots):
        yaw = float(rng.uniform(-cfg["yaw_jitter"], cfg["yaw_jitter"])) if cfg["yaw_jitter"] else 0.0
        o = instrument(name, name, _slot_x(k), ROW_Y, REST["height"], yaw, support="instrument_rest")
        o.attributes.update(position=k + 1, lies_on_rests=True)
        objects.append(o)
    for o in objects:  # look-alikes: same kind of instrument (ground truth for ambiguity experiments)
        o.attributes["look_alikes"] = [p.id for p in objects if p is not o and p.category == o.category]
    if cfg["gauze"]:
        objects.append(box_obj("gauze_sponge", "gauze", 0.30, 0.30, 0.0, float(rng.uniform(0, 30)), (0.10, 0.07, 0.018),
                               (0.97, 0.97, 0.95), material="foam", attributes={"name": "gauze sponge"},
                               role="distractor", support="table"))

    occupied = {k: o.id for k, o in zip(slots, objects)}
    regions = [Region(f"position_{k + 1}", "instrument_slot", (_slot_x(k), ROW_Y, REST["height"]), (PITCH - 0.01, 0.16),
                      0.0, f"POSITION {k + 1}", "instrument place on the rests", occupied_by=occupied.get(k),
                      zone="instrument_area", accepts="instrument") for k in range(SLOTS)]
    hx, hy = HAND["xy"]
    regions.append(Region("surgeon_hand", "hand", (hx, hy, HAND["z"] + HAND["palm"][2]), HAND["palm"][:2], _hand_yaw(),
                          "SURGEON'S HAND", "hand the instrument over here", zone="handover", accepts="instrument"))
    zones = [{"id": "instrument_area", "title": "INSTRUMENT AREA", "centre": [0.0, ROW_Y],
              "size": [SLOTS * PITCH, 0.20], "purpose": "instruments on two rests, positions 1-8 (1 = the robot's left)"},
             {"id": "handover", "title": "SURGEON'S HAND", "centre": [hx, hy], "size": [0.12, 0.20],
              "purpose": "the surgeon's open hand: hand instruments over here"}]
    lights = [Light("rect", 4200.0, (1.0, 0.98, 0.95), pos=(-0.2, 0.45, 1.70), size=(0.9, 0.9))]
    description = " ".join(__doc__.split("\n\n")[1].split())
    return Scenario(ID, TITLE, difficulty, seed, description, [_rests(), _hand()], objects, _room(), regions, lights,
                    zones=zones, view={"eye": (1.35, 1.65, 1.15), "target": (-0.35, 0.40, 0.0), "table": TABLE,
                                       "table_camera": TABLE_CAMERA, "wrist": ["surgeon_hand"]},
                    notes={"difficulty_features": FEATURES[difficulty], "instruments": len(slots)}, version=VERSION)
