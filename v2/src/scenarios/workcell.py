"""Scenario: snack tray preparation at home (v2.0, 2026-11-11).

The Epson VT6L stands at the back of a wooden table inside an aluminium frame. Seen from the front (the overhead table
camera on the frame's front beam), household items stand on the left of the table: snack packets, drinks, a fruit cup,
a banana and pantry items. On the right are an empty SERVING TRAY (where a person's snack is prepared) and a deeper
STORAGE BASKET. Nothing on the table is printed with words for the robot: what an item is, is seen. The use case: a
person asks for a personalised snack ("water, crackers and a fruit cup"), may change the order midway, and some
requested items may be missing or look alike.

Items: real scanned household objects from the YCB set (scenarios/assets.py; CC BY 4.0) for their look, each with an
invisible box or cylinder collider of its measured size for physics and planning; the water bottle, juice bottle and
fruit cup are built from primitives (YCB has none). Only items the 2F-85 can hold across one side (<= 72 mm) are
graspable; the mug (with its handle) and the wide tuna can are things to leave alone.

Difficulty: which items are on the table is FIXED per difficulty (the experiment's task lists rely on it); the seed
only moves them.
    easy    6 items, wide spacing, square to the robot
    medium  9 items, closer: adds look-alike pairs (water / juice bottle, pudding / gelatin box)
    hard    11 items at random angles, the tuna can (too wide to grasp), and the person's mug already on the tray

Robot frame: +y = forward (towards the camera), +x = the robot's right, so the camera sees +x on its LEFT. The table
top is z = 0 (the planner's support surface); the room floor is drawn 0.75 m below it. Every item is in the arm's
top-down ring (control/workspace.py, config/agent.py REACH_MIN_M..REACH_MAX_M); the tray and basket sit where a
top-down placement reaches. The scene has no calibrated places (layout() is empty): the tray and basket are seen
and measured like the items.
"""

from __future__ import annotations

import numpy as np

from scenarios.assets import ycb_spec
from scenarios.core import Fixture, Light, Obj, Part, Placer, Region, Scenario
from scenarios.parts import capped_container, surface, tray

ID = "workcell"
TITLE = "Snack tray preparation"
VERSION = "2.0"

# Overhead table camera on the frame's front beam, looking down and back at the robot (control/robot.py TABLE_CAMERA)
TABLE_CAMERA = ((0.0, 1.30, 1.45), (0.0, 0.42, 0.0))

TABLE = dict(centre=(0.0, 0.35), size=(2.30, 1.50), thickness=0.05, height=0.75,
             no_go_color=(0.50, 0.33, 0.20))  # the shaded no-go floor on this table: a darker wood tone
WOOD, WOOD_EDGE = (0.62, 0.43, 0.26), (0.50, 0.33, 0.19)
ALU = (0.72, 0.74, 0.77)
ITEMS_AREA = dict(centre=(0.48, 0.36), size=(0.56, 0.72))  # robot's right = the camera's left
RING = (0.42, 0.80)  # items stay this far from the base (m): inside the top-down ring with margin
LOW_M, LOW_MIN_M = 0.06, 0.46  # an item lower than LOW_M stays at least LOW_MIN_M away (2026-11-11 design checks: a
# 39 mm banana at 0.43 m was out of reach for the descent)
# robot's left = the camera's right; long sides tangent to the reach ring (as the old trays: outlines stay apart)
CONTAINERS = (("serving_tray", "serving tray", (0.40, 0.28), 0.025, (0.80, 0.64, 0.42), (-0.50, 0.50)),
              ("storage_basket", "storage basket", (0.30, 0.22), 0.08, (0.50, 0.34, 0.18), (-0.66, -0.06)))

# YCB items: key -> (what it is, category, collider, mass kg (YCB), graspable)
YCB = {"003_cracker_box": ("cracker box", "snack", "box", 0.411, True),
       "008_pudding_box": ("chocolate pudding box", "snack", "box", 0.187, True),
       "009_gelatin_box": ("gelatin dessert box", "snack", "box", 0.097, True),
       "011_banana": ("banana", "fruit", "box", 0.066, True),
       "005_tomato_soup_can": ("tomato soup can", "pantry", "cylinder", 0.349, True),
       "004_sugar_box": ("sugar box", "pantry", "box", 0.514, True),
       "006_mustard_bottle": ("mustard bottle", "condiment", "box", 0.603, True),
       "010_potted_meat_can": ("potted meat can", "pantry", "box", 0.370, True),
       "007_tuna_fish_can": ("tuna can", "pantry", "cylinder", 0.171, False),  # 86 mm across: wider than the gripper
       "025_mug": ("mug", "personal", "box", 0.118, False)}  # the person's mug: not a snack, leave it
# built from primitives: key -> (what it is, category, body d, body h, cap d, cap h, body colour, cap colour)
BUILT = {"water_bottle": ("water bottle", "drink", 0.062, 0.150, 0.034, 0.022, (0.78, 0.90, 0.98), (0.10, 0.35, 0.85)),
         "juice_bottle": ("orange juice bottle", "drink", 0.062, 0.150, 0.034, 0.022, (0.98, 0.58, 0.10), (0.95, 0.95, 0.95)),
         "fruit_cup": ("fruit cup", "snack", 0.066, 0.060, 0.068, 0.004, (0.99, 0.80, 0.40), (0.85, 0.86, 0.88))}

ON_TABLE = {  # fixed per difficulty, in this order
    "easy": ["water_bottle", "003_cracker_box", "fruit_cup", "011_banana", "008_pudding_box", "005_tomato_soup_can"],
    "medium": ["water_bottle", "juice_bottle", "003_cracker_box", "fruit_cup", "011_banana", "008_pudding_box",
               "009_gelatin_box", "005_tomato_soup_can", "004_sugar_box"],
    "hard": ["water_bottle", "juice_bottle", "003_cracker_box", "fruit_cup", "011_banana", "008_pudding_box",
             "009_gelatin_box", "005_tomato_soup_can", "004_sugar_box", "006_mustard_bottle", "007_tuna_fish_can"],
}
LOOK_ALIKES = ({"water_bottle", "juice_bottle"}, {"008_pudding_box", "009_gelatin_box"})  # same shape, other item
ON_TRAY = {"easy": [], "medium": [], "hard": ["025_mug"]}  # already on the serving tray (not part of the order)
TALL_M, TALL_GAP_M = 0.12, 0.10  # a short item keeps this gap (bounding circles) from a taller-than-TALL_M one: the wrist
# camera beside the gripper passes it on the way down (2026-11-11 design checks: 48-65 mm let it touch a cracker box)
CONFIG = {  # clearance: free space between neighbours: the open 2F-85 pads (~45 mm each side) on one axis
    "easy": dict(clearance=0.060, random_yaw=False),
    "medium": dict(clearance=0.060, random_yaw=False),
    "hard": dict(clearance=0.056, random_yaw=True),
}
FEATURES = {
    "easy": "6 items, wide spacing, square to the robot",
    "medium": "9 items, closer; look-alike pairs (water / juice bottle, pudding / gelatin box)",
    "hard": "11 items at random angles; an ungraspable tuna can; the person's mug already on the serving tray",
}


def _visual(id: str, category: str, parts: list, labels=None) -> Fixture:
    return Fixture(id, category, (0.0, 0.0, 0.0), 0.0, parts, labels or [])


def _box(centre, size, color, material="painted", name="") -> Part:
    return Part("box", tuple(centre), tuple(size), color, material=material, collide=False, name=name)


def _room() -> list:
    """Table body and legs, room floor, aluminium frame, lab walls and benches: visual only (the physics support is
    the simulator's floor plane at z = 0, hidden by scenarios/build.py)."""
    (cx, cy), (L, W), t, h = TABLE["centre"], TABLE["size"], TABLE["thickness"], TABLE["height"]
    floor = -h
    table = [_box((cx, cy, -t / 2 - 0.0004), (L, W, t - 0.0008), WOOD_EDGE, "wood", "slab")]
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.append(_box((cx + sx * (L / 2 - 0.06), cy + sy * (W / 2 - 0.06), floor + (h - t) / 2),
                              (0.06, 0.06, h - t), (0.30, 0.31, 0.33), "metal", f"leg{sx:+d}{sy:+d}"))
    for sy in (-1, 1):  # stretchers between the legs
        table.append(_box((cx, cy + sy * (W / 2 - 0.06), floor + 0.15), (L - 0.12, 0.03, 0.03), (0.30, 0.31, 0.33), "metal"))
    # aluminium frame: four posts outside the table corners, top beams, the camera's front beam and mount
    px, back, front, top = 1.60, -0.45, 1.30, 1.75  # posts outside the table camera's view (it would measure them)
    frame = [_box((s * px, y, (floor + top) / 2), (0.045, 0.045, top - floor), ALU, "metal", f"post{s:+d}{y:+.2f}")
             for s in (-1, 1) for y in (back, front)]
    frame += [_box((s * px, (back + front) / 2, top), (0.045, front - back, 0.045), ALU, "metal") for s in (-1, 1)]
    frame += [_box((0.0, y, top), (2 * px, 0.045, 0.045), ALU, "metal") for y in (back, front)]
    cam_x, cam_y, cam_z = TABLE_CAMERA[0]
    frame.append(_box((cam_x, cam_y, (cam_z + top) / 2 + 0.01), (0.03, 0.03, top - cam_z - 0.02), ALU, "metal", "camera_mount"))
    room = [_box((0.0, -1.70, floor + 1.4), (5.0, 0.06, 2.8), (0.86, 0.88, 0.90), "painted", "back_wall"),
            _box((-2.4, 0.2, floor + 1.4), (0.06, 3.8, 2.8), (0.84, 0.86, 0.89), "painted", "side_wall"),
            _box((1.9, -1.30, floor + 0.45), (1.2, 0.60, 0.90), (0.88, 0.89, 0.90), "painted", "bench_right"),
            _box((1.9, -1.30, floor + 0.92), (1.24, 0.64, 0.04), (0.35, 0.36, 0.38), "epoxy", "bench_right_top"),
            _box((-1.6, -1.30, floor + 0.45), (1.4, 0.60, 0.90), (0.88, 0.89, 0.90), "painted", "bench_left"),
            _box((-1.6, -1.30, floor + 0.92), (1.44, 0.64, 0.04), (0.35, 0.36, 0.38), "epoxy", "bench_left_top"),
            _box((-1.6, -1.45, floor + 1.55), (1.4, 0.30, 0.03), (0.80, 0.82, 0.85), "painted", "shelf_left"),
            _box((-1.9, -1.45, floor + 1.65), (0.10, 0.10, 0.18), (0.20, 0.45, 0.80), "plastic", "bottle_1"),
            _box((-1.5, -1.45, floor + 1.64), (0.14, 0.10, 0.16), (0.90, 0.90, 0.88), "plastic", "box_1"),
            _box((0.6, -1.66, floor + 1.55), (0.60, 0.02, 0.40), (0.95, 0.95, 0.96), "painted", "whiteboard"),
            _box((2.3, 0.6, floor + 0.95), (0.60, 0.60, 1.90), (0.80, 0.82, 0.84), "painted", "cabinet")]
    return [surface("room_floor", (0.0, 0.0), (8.0, 8.0), (0.62, 0.64, 0.66), z=floor + 0.0003, material="epoxy"),
            _visual("table", "table", table), _visual("frame", "aluminium_frame", frame), _visual("room", "room", room),
            surface("table_top", TABLE["centre"], TABLE["size"], WOOD, z=0.0003, material="wood")]


def _slug(key: str) -> str:
    """The item's id stem from what it is ("cracker_box"): USD prim names must not start with a digit (a YCB key
    like "003_cracker_box" stopped the simulator, 2026-11-11)."""
    return (BUILT[key] if key in BUILT else YCB[key])[0].replace(" ", "_")


def item(key: str, oid: str, x: float, y: float, z: float, yaw: float, role: str, support: str = "table") -> Obj:
    """One household item standing upright at (x, y) on a surface at height z."""
    if key in BUILT:
        name, cat, d, h, cd, ch, body, cap = BUILT[key]
        return capped_container(oid, cat, x, y, z, yaw, d, h, cd, ch, body, cap, fill=0.9,
                                attributes={"key": key, "name": name, "category": cat, "graspable": True,
                                            "source": "primitive"},
                                role=role, support=support)
    name, cat, collider, mass, graspable = YCB[key]
    s = ycb_spec(key)
    w, h, dpt = s["size"]  # +y up in the model: upright, the footprint is x by z and the height is y
    parts = [Part("mesh", (0.0, 0.0, h / 2), (w, dpt, h), (0.7, 0.7, 0.7), asset=f"ycb:{key}", collide=False,
                  name=f"{oid}/mesh", tilt_x_deg=90.0)]
    parts.append(Part(collider, (0.0, 0.0, h / 2), (w, dpt, h) if collider == "box" else (w, w, h), (0.7, 0.7, 0.7),
                      material="cardboard", visible=False, name=f"{oid}/collider"))
    attrs = {"key": key, "name": name, "category": cat, "graspable": graspable, "source": "YCB " + key,
             "license": "CC BY 4.0"}
    return Obj(oid, cat, (x, y, z), yaw, parts, mass, [], attrs, role if graspable else "unreachable", support, (0.8, 0.7))


def _place(keys, rng, fixtures, cfg) -> list | None:
    """Items in the items area inside the reach ring, largest first, cfg["clearance"] apart; None if one does not fit."""
    placer = Placer(rng, fixtures, clearance=cfg["clearance"])
    (cx, cy), (sx, sy) = ITEMS_AREA["centre"], ITEMS_AREA["size"]
    probe = {k: item(k, "probe", 0.0, 0.0, 0.0, 0.0, "") for k in keys}
    size = {k: float(np.prod(o.footprint)) for k, o in probe.items()}
    placed = {}
    for k in sorted(range(len(keys)), key=lambda k: -size[keys[k]]):
        for _ in range(300):
            x, y = np.asarray((cx, cy)) + (rng.random(2) - 0.5) * (np.asarray((sx, sy)) - 0.06)
            if not (LOW_MIN_M if probe[keys[k]].height < LOW_M else RING[0]) <= np.hypot(x, y) <= RING[1]:
                continue
            yaw = float(rng.uniform(0.0, 180.0)) if cfg["random_yaw"] else 0.0
            o = item(keys[k], f"{_slug(keys[k])}_{k + 1}", float(x), float(y), 0.0, yaw, "item")
            if placer.free(o) and _apart_from_tall(o, placed.values()):
                placed[k] = placer.add(o)
                break
        else:
            return None
    return [placed[k] for k in range(len(keys))]


def _apart_from_tall(o: Obj, others) -> bool:
    r = np.hypot(*o.footprint) / 2
    for p in others:
        if (o.height > TALL_M) != (p.height > TALL_M):
            if np.hypot(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) - r - np.hypot(*p.footprint) / 2 < TALL_GAP_M:
                return False
    return True


def layout() -> list[dict]:
    """The work cell's calibrated fixed regions: none (no markings on this table; the tray and basket are seen)."""
    return []


def _yaw_to_ring(xy) -> float:
    return float(np.degrees(np.arctan2(xy[0], -xy[1])))  # long side tangent to the reach ring


def build(difficulty: str, seed: int) -> Scenario:
    cfg = CONFIG[difficulty]
    fixtures, regions = [], []
    for cid, name, size, rim, rgb, xy in CONTAINERS:
        yaw = _yaw_to_ring(xy)
        fixtures.append(tray(cid, (xy[0], xy[1], 0.0), yaw, size=size, rim=rim, color=rgb, wall=0.006,
                             category=cid, description=f"{name} (open, {round(rim * 1000)} mm rim)"))
        regions.append(Region(cid, "tray", (xy[0], xy[1], 0.004), (size[0] - 0.012, size[1] - 0.012), yaw, name,
                              "the person's snack" if cid == "serving_tray" else "storage", zone="containers",
                              accepts="any"))

    keys = ON_TABLE[difficulty]
    for attempt in range(300):  # a jammed layout is redrawn from a sub-seed: still deterministic for (difficulty, seed)
        objects = _place(keys, np.random.default_rng([seed, attempt]), fixtures, cfg)
        if objects is not None:
            break
    else:
        raise RuntimeError(f"workcell {difficulty} seed {seed}: the items area cannot hold {len(keys)} items")
    tray_xy, tray_yaw = np.asarray(CONTAINERS[0][5]), _yaw_to_ring(CONTAINERS[0][5])
    along = np.asarray((np.cos(np.radians(tray_yaw)), np.sin(np.radians(tray_yaw))))  # the tray's long side
    for k, key in enumerate(ON_TRAY[difficulty]):  # already on the serving tray, at one end: the middle stays free
        x, y = tray_xy + along * (0.13 - 0.26 * k)
        objects.append(item(key, f"{_slug(key)}_on_tray", float(x), float(y), 0.004, tray_yaw, "on_tray",
                            support="serving_tray"))
    for o in objects:  # look-alikes (ground truth for ambiguity experiments): the designed pairs present
        pair = next((q for q in LOOK_ALIKES if o.attributes["key"] in q), ())
        o.attributes["look_alikes"] = [p.id for p in objects if p is not o and p.attributes["key"] in pair]

    decor = _room()
    lights = [Light("rect", 3500.0, (1.0, 0.97, 0.92), pos=(0.0, 0.45, 2.1), size=(1.4, 0.8))]
    zones = [{"id": "items", "title": "ITEMS", "centre": list(ITEMS_AREA["centre"]), "size": list(ITEMS_AREA["size"]),
              "purpose": "household items to choose from (no marking on the table)"},
             {"id": "containers", "title": "TRAY AND BASKET", "centre": [-0.58, 0.22], "size": [0.45, 0.9],
              "purpose": "serving tray (the snack) and storage basket"}]
    description = " ".join(__doc__.split("\n\n")[1].split())
    return Scenario(ID, TITLE, difficulty, seed, description, fixtures, objects, decor, regions, lights, zones=zones,
                    view={"eye": (0.0, 1.75, 1.35), "target": (0.0, 0.35, 0.0), "table": TABLE,
                          "table_camera": TABLE_CAMERA, "wrist": ["serving_tray"]},
                    notes={"difficulty_features": FEATURES[difficulty], "objects": len(objects)}, version=VERSION)
