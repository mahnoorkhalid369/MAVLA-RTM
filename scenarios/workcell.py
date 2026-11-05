"""Scenario: tabletop sorting work cell (v1.1).

The Epson VT6L stands at the back of a wooden table inside an aluminium frame. Seen from the front (the overhead table
camera on the frame's front beam), the table holds a PICKUP AREA of coloured blocks, cylinders and capped tubes on the
left, a dark 3 x 3 ARRANGEMENT MAT in the middle and three sorting trays on the right: TRAY 1 (red), TRAY 2 (blue),
TRAY 3 (green). The wrist RealSense on the gripper stays for close inspection. Behind the robot is a lab room
(visual only). Easy / medium / hard put more objects in the pickup area with less space between them; hard turns them
to random angles and adds look-alikes (same colour, other size).

Robot frame: +y = forward (towards the camera), +x = the robot's right, so the camera sees +x on its LEFT. The table
top is z = 0 (the planner's support surface); the room floor is drawn 0.75 m below it. Every object is in the arm's
top-down ring (control/workspace.py, config/agent.py REACH_MIN_M..REACH_MAX_M); the trays and the mat sit where a top-down
placement reaches.

What the agent knows (2026-11-04): movable scene entities (the objects, and the trays, which are movable bins) are
perceived through the cameras only. Fixed work-cell geometry and its explicitly calibrated operational regions (the
pickup area, the arrangement mat and its nine cells: flat markings no detector sees) come from the work cell's
calibrated layout, layout() below, exported to the agent as WORKCELL_LAYOUT (config/layout.py), as the table camera's
mount is (TABLE_CAMERA). The layout gives geometry and ids only; what the user's words refer to stays Gemma's reading.
"""

from __future__ import annotations

import numpy as np

from scenarios.core import Fixture, Label, Light, Part, Placer, Region, Scenario
from scenarios.parts import box_obj, capped_container, surface, tray, zone_outline

ID = "workcell"
TITLE = "Tabletop sorting work cell"
VERSION = "1.1"

# Overhead table camera on the frame's front beam, looking down and back at the robot (control/robot.py TABLE_CAMERA)
TABLE_CAMERA = ((0.0, 1.30, 1.45), (0.0, 0.42, 0.0))

TABLE = dict(centre=(0.0, 0.35), size=(2.30, 1.50), thickness=0.05, height=0.75,
             no_go_color=(0.50, 0.33, 0.20))  # the shaded no-go floor on this table: a darker wood tone
WOOD, WOOD_EDGE = (0.62, 0.43, 0.26), (0.50, 0.33, 0.19)
ALU = (0.72, 0.74, 0.77)
PICKUP = dict(centre=(0.47, 0.44), size=(0.40, 0.46))  # robot's right = the camera's left
MAT = dict(centre=(0.0, 0.62), cell=0.12, border=0.04)  # 3 x 3 cells; the rubber mat is 3 cells + border wide
# robot's left = the camera's right. Each tray holds 3-4 blocks (inside 0.288 x 0.188 m); long side tangent to the reach
# ring at 0.70 m, 36 deg apart: ~8.6 cm between trays and 5 cm from the mat, so their outlines stay apart in the table
# camera's view (live 2026-10-26: 4 cm apart, the blue tray's outline took in the red tray's corner)
TRAY_SIZE, TRAY_RIM = (0.30, 0.20), 0.04
TRAYS = (("tray_1", "TRAY 1", "red", (0.85, 0.12, 0.10), (-0.431, 0.552)),
         ("tray_2", "TRAY 2", "blue", (0.12, 0.30, 0.85), (-0.673, 0.193)),
         ("tray_3", "TRAY 3", "green", (0.12, 0.60, 0.22), (-0.658, -0.239)))
RING = (0.42, 0.85)  # objects stay this far from the base (m): inside the top-down ring with margin

COLOURS = {"red": (0.85, 0.12, 0.10), "blue": (0.12, 0.30, 0.85), "yellow": (0.95, 0.82, 0.10),
           "green": (0.12, 0.60, 0.22), "cyan": (0.10, 0.75, 0.80), "orange": (0.95, 0.50, 0.08),
           "brown": (0.45, 0.28, 0.14), "purple": (0.48, 0.20, 0.75), "pink": (0.95, 0.35, 0.70)}
# (kind, colour, size): cube (edge), tall (w, w, h), cylinder (d, h), tube (body d, body h)
CATALOGUE = [("cube", "red", 0.05), ("cube", "blue", 0.06), ("cube", "yellow", 0.05), ("cube", "green", 0.05),
             ("cube", "cyan", 0.045), ("cube", "orange", 0.04), ("cube", "brown", 0.06), ("tall", "pink", (0.045, 0.12)),
             ("cylinder", "blue", (0.05, 0.05)), ("cylinder", "purple", (0.05, 0.06)), ("tube", "white", (0.03, 0.09)),
             ("cube", "red", 0.04), ("cube", "yellow", 0.06), ("tall", "pink", (0.04, 0.09)), ("tube", "white", (0.03, 0.07)),
             ("cube", "blue", 0.045), ("cube", "green", 0.06), ("cube", "purple", 0.05)]

CONFIG = {  # clearance: free space between neighbours, enough for the open 2F-85 pads (~45 mm each side) on one axis
    "easy": dict(objects=10, clearance=0.050, random_yaw=False),
    "medium": dict(objects=12, clearance=0.052, random_yaw=False),
    "hard": dict(objects=12, clearance=0.048, random_yaw=True),
}
FEATURES = {
    "easy": "10 objects, wide spacing, square to the robot",
    "medium": "12 objects, closer together",
    "hard": "12 objects at random angles, look-alikes (same colour, other size)",
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


def _object(kind: str, colour: str, size, oid: str, x: float, y: float, yaw: float, role: str):
    attrs = {"colour": colour, "kind": kind}
    if kind == "tube":
        d, h = size
        return capped_container(oid, "capped_tube", x, y, 0.0, yaw, d, h, d + 0.006, 0.016, (0.95, 0.95, 0.95),
                                (0.80, 0.80, 0.82), attributes=attrs, role=role, support="table")
    rgb = COLOURS[colour]
    if kind == "cylinder":
        d, h = size
        o = box_obj(oid, "cylinder", x, y, 0.0, yaw, (d, d, h), rgb, material="wood", attributes=attrs, role=role,
                    support="table")
        o.parts = [Part("cylinder", (0, 0, h / 2), (d, d, h), rgb, material="wood", name=f"{oid}/body")]
        return o
    if kind == "tall":
        w, h = size
        return box_obj(oid, "tall_block", x, y, 0.0, yaw, (w, w, h), rgb, material="wood", attributes=attrs, role=role,
                       support="table")
    return box_obj(oid, "cube", x, y, 0.0, yaw, (size, size, size), rgb, material="wood", attributes=attrs, role=role,
                   support="table")


def _footprint(kind: str, size) -> float:
    return size if kind == "cube" else size[0]


def _place(picks, rng, fixtures, cfg) -> list | None:
    """Objects in the pickup area inside the reach ring, largest first, cfg["clearance"] apart; None if one does not fit."""
    placer = Placer(rng, fixtures, clearance=cfg["clearance"])
    (pcx, pcy), (psx, psy) = PICKUP["centre"], PICKUP["size"]
    order = sorted(range(len(picks)), key=lambda k: -_footprint(picks[k][0], picks[k][2]))
    placed = {}
    for k in order:
        kind, colour, size = picks[k]
        for _ in range(300):
            x, y = np.asarray((pcx, pcy)) + (rng.random(2) - 0.5) * (np.asarray((psx, psy)) - 0.06)
            if not RING[0] <= np.hypot(x, y) <= RING[1]:
                continue
            yaw = float(rng.uniform(0.0, 90.0)) if cfg["random_yaw"] else 0.0
            o = _object(kind, colour, size, f"{colour}_{kind}_{k + 1}", float(x), float(y), yaw, "pickup")
            if placer.free(o):
                placed[k] = placer.add(o)
                break
        else:
            return None
    return [placed[k] for k in range(len(picks))]


def _cells():
    """The mat's 3 x 3 cells: (label, centre xy); A = nearest the robot, 1 = the robot's left."""
    (mx, my), c = MAT["centre"], MAT["cell"]
    return [(f"{'ABC'[i]}{j + 1}", (mx + dx, my + dy)) for i, dy in enumerate((-c, 0.0, c)) for j, dx in enumerate((-c, 0.0, c))]


def layout() -> list[dict]:
    """The work cell's calibrated fixed regions (robot base frame, m; the table top is z = 0): flat markings on the
    table that the cameras cannot detect as objects. Built from the same constants as the scene (one source). Ids are
    R1, R2, ... in this order; "name" is the marking's printed label, as a person reads it."""
    (mx, my), c = MAT["centre"], MAT["cell"]
    regions = [{"name": "pickup area", "purpose": "where objects to be sorted or arranged start",
                "centre": [*PICKUP["centre"], 0.0], "size": list(PICKUP["size"])},
               {"name": "arrangement mat", "purpose": "dark rubber mat with a 3 x 3 grid of cells A1-C3",
                "centre": [mx, my, 0.0], "size": [3 * c + MAT["border"]] * 2}]
    regions += [{"name": f"arrangement mat cell {label}", "purpose": "one cell of the mat (A = the row nearest the "
                 "robot, 1 = the column on the robot's left)", "centre": [x, y, 0.0], "size": [c, c]}
                for label, (x, y) in _cells()]
    return [{"id": f"R{k}", "frame": "robot", "type": "planar_region", "shape": "rectangle", "yaw_deg": 0.0, **r}
            for k, r in enumerate(regions, 1)]


def build(difficulty: str, seed: int) -> Scenario:
    cfg = CONFIG[difficulty]
    rng = np.random.default_rng(seed)
    fixtures, regions = [], []

    for tid, title, colour, rgb, xy in TRAYS:
        yaw = float(np.degrees(np.arctan2(xy[0], -xy[1])))  # long side tangent to the reach ring
        t = tray(tid, (xy[0], xy[1], 0.0), yaw, size=TRAY_SIZE, rim=TRAY_RIM, color=rgb, wall=0.006,
                 description=f"{colour} sorting tray (open, 40 mm rim)")
        fixtures.append(t)
        regions.append(Region(tid, "tray", (xy[0], xy[1], 0.004), (TRAY_SIZE[0] - 0.012, TRAY_SIZE[1] - 0.012), yaw, title, f"sort {colour} objects",
                              zone="trays", accepts="any"))
    (mx, my), c = MAT["centre"], MAT["cell"]
    for label, (x, y) in _cells():
        regions.append(Region(f"mat_{label}", "mat_cell", (x, y, 0.0006), (c, c), 0.0, label, "arrangement cell",
                              zone="mat", accepts="any"))

    # ---------------------------------------------------------------------------------------- pickup objects
    picks = [CATALOGUE[k] for k in rng.permutation(len(CATALOGUE))[:cfg["objects"]]]
    for attempt in range(300):  # a jammed layout is redrawn from a sub-seed: still deterministic for (difficulty, seed)
        objects = _place(picks, np.random.default_rng([seed, attempt]), fixtures, cfg)
        if objects is not None:
            break
    else:
        raise RuntimeError(f"workcell {difficulty} seed {seed}: the pickup area cannot hold {len(picks)} objects")
    for o in objects:  # look-alikes: same colour and kind (ground truth for ambiguity experiments)
        o.attributes["look_alikes"] = [p.id for p in objects if p is not o and p.attributes["colour"] == o.attributes["colour"]
                                       and p.attributes["kind"] == o.attributes["kind"]]

    # ------------------------------------------------------------------------------------- markings + room
    # labels lie on the table and read from the camera's side (turned 180 deg: the camera looks back at the robot)
    mat = [surface("arrangement_mat", (mx, my), (3 * c + MAT["border"], 3 * c + MAT["border"]), (0.20, 0.21, 0.23), z=0.0006, material="rubber",
                   labels=[Label("ARRANGEMENT MAT", (0.0, (3 * c) / 2 + 0.040, 0.0012), (0.22, 0.028), fg=(1, 1, 1),
                                 bg=(0.10, 0.20, 0.40), yaw_deg=180.0)])]
    for tid, title, colour, rgb, xy in TRAYS:  # a label in front of each tray (camera side)
        out = np.asarray(xy) / np.hypot(*xy)
        p = np.asarray(xy) + out * (TRAY_SIZE[1] / 2 + 0.035)
        yaw = float(np.degrees(np.arctan2(xy[0], -xy[1])))
        mat.append(Fixture(f"{tid}_label", "label", (float(p[0]), float(p[1]), 0.0), 0.0, [],
                           [Label(f"{title} ({colour.upper()})", (0.0, 0.0, 0.0012), (0.13, 0.026), fg=(1, 1, 1), bg=rgb,
                                  yaw_deg=yaw)]))
    for label, (x, y) in _cells():
        mat.append(zone_outline(f"mat_cell_{label}", (x, y), (c - 0.012, c - 0.012), (0.80, 0.80, 0.82), width=0.003, z=0.0010))
    pickup = zone_outline("pickup_area", PICKUP["centre"], PICKUP["size"], (0.30, 0.65, 0.95), width=0.008, z=0.0008,
                          title="PICKUP AREA", title_side="y", title_fg=(1, 1, 1))
    for lb in pickup.labels:
        lb.yaw_deg += 180.0
    decor = _room() + mat + [pickup]
    lights = [Light("rect", 3500.0, (1.0, 0.97, 0.92), pos=(0.0, 0.45, 2.1), size=(1.4, 0.8))]
    zones = [{"id": "pickup", "title": "PICKUP AREA", "centre": list(PICKUP["centre"]), "size": list(PICKUP["size"]),
              "purpose": "objects to be sorted or arranged"},
             {"id": "mat", "title": "ARRANGEMENT MAT", "centre": [mx, my], "size": [3 * c, 3 * c],
              "purpose": "3 x 3 cells A1-C3 (A = nearest the robot, 1 = the robot's left)"},
             {"id": "trays", "title": "TRAYS", "centre": [-0.58, 0.16], "size": [0.45, 1.0],
              "purpose": "TRAY 1 red, TRAY 2 blue, TRAY 3 green"}]
    description = " ".join(__doc__.split("\n\n")[1].split())
    return Scenario(ID, TITLE, difficulty, seed, description, fixtures, objects, decor, regions, lights, zones=zones,
                    view={"eye": (0.0, 1.75, 1.35), "target": (0.0, 0.35, 0.0), "table": TABLE,
                          "table_camera": TABLE_CAMERA, "wrist": ["tray_1"]},
                    notes={"difficulty_features": FEATURES[difficulty], "objects": len(objects)}, version=VERSION)
