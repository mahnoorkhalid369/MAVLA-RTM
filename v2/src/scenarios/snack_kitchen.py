"""Home kitchen island for preparing a personalised snack tray.

Uses seeded snack layouts, the project's robot-frame convention and camera mount,
and collision-backed serving/storage containers. Replaces the laboratory dressing
with a kitchen island, cabinets, tiled backsplash, window, and NVIDIA Residential
USD appliances/props. All tall background decor stays outside the arm's workspace.

The countertop is z=0; the room floor is 0.90 m below it. Decorative cabinetry is
not a task destination. Item identities and example orders are evaluation metadata
only: the agent gets RGB-D observations. There are no calibrated R<n> mat cells.

NVIDIA geometry is referenced through metre-scale, centred USD wrappers in
assets/snack_kitchen/models. Fetch textures/models with scripts/fetch_snack_assets.py.
"""

from __future__ import annotations

import json
from math import atan2, degrees
from pathlib import Path

import numpy as np

from scenarios.assets import ycb_spec
from scenarios.core import Fixture, Label, Light, Obj, Part, Placer, Region, Scenario
from scenarios.parts import capped_container, surface, tray

ID = "snack_kitchen"
TITLE = "Home kitchen: personalised snack tray"
VERSION = "1.1"
TABLE_CAMERA = ((0.0, 1.30, 1.45), (0.0, 0.42, 0.0))
TABLE = dict(centre=(0.0, 0.35), size=(2.30, 1.50), thickness=0.05, height=0.90,
             no_go_color=(0.78, 0.67, 0.55))
FLOOR = -TABLE["height"]
ROOT = Path(__file__).resolve().parents[2]
MODEL_SPECS = json.loads((ROOT / "assets/snack_kitchen/nvidia_manifest.json").read_text())["models"]
ITEMS_AREA = dict(centre=(0.48, 0.44), size=(0.74, 0.78))
CONTAINERS = (("serving_tray", "serving tray", (0.36, 0.26), 0.025, (0.22, 0.57, 0.55), (-0.50, 0.50)),
              ("storage_basket", "storage basket", (0.30, 0.22), 0.08, (0.50, 0.34, 0.18), (-0.66, -0.06)))
YCB = {"003_cracker_box": ("cracker box", "snack", "box", 0.411),
       "008_pudding_box": ("chocolate pudding box", "snack", "box", 0.187),
       "009_gelatin_box": ("gelatin dessert box", "snack", "box", 0.097),
       "005_tomato_soup_can": ("tomato soup can", "pantry", "cylinder", 0.349),
       "004_sugar_box": ("sugar box", "pantry", "box", 0.514),
       "006_mustard_bottle": ("mustard bottle", "condiment", "box", 0.603),
       "010_potted_meat_can": ("potted meat can", "pantry", "box", 0.370),
       "007_tuna_fish_can": ("tuna can", "pantry", "cylinder", 0.171),
       "025_mug": ("mug", "personal", "box", 0.118)}
BUILT = {"fruit_cup": ("fruit cup", "snack", 0.066, 0.060, 0.068, 0.004, (0.99, 0.80, 0.40), (0.85, 0.86, 0.88))}
EASY_ITEMS = ["milk", "water_bottle", "003_cracker_box", "fruit_cup", "apple", "008_pudding_box"]
ITEMS = {"easy": EASY_ITEMS,
         "medium": EASY_ITEMS + ["005_tomato_soup_can"],
         "hard": EASY_ITEMS + ["005_tomato_soup_can", "006_mustard_bottle"]}
# Counter-edge spacing, in addition to the larger tall/short-object separation below.
# Difficulty adds distinct items and rotations, rather than deliberately ambiguous packaging.
CONFIG = {"easy": (0.100, False), "medium": (0.090, False), "hard": (0.080, True)}

EXAMPLE_REQUESTS = (
    "Put the milk and the crackers in the blue serving tray.",
    "Put the apple and the water bottle in the blue serving tray. Leave the pudding on the counter.",
    "Prepare the blue serving tray with water, crackers, and a fruit cup.",
)


def _box(name, centre, size, color, material="painted") -> Part:
    return Part("box", tuple(centre), tuple(size), color, material=material, collide=False, name=name)


def _group(name: str, category: str, parts: list[Part]) -> Fixture:
    return Fixture(name, category, (0.0, 0.0, 0.0), 0.0, parts)


def _model(name: str, centre, scale=1.0, yaw=0.0) -> Part:
    size = tuple(scale * v for v in MODEL_SPECS[name]["size_m"])
    return Part("mesh", tuple(centre), size, (0.7, 0.7, 0.7), yaw_deg=yaw,
                collide=False, name=name, asset=f"snack_kitchen/models/{name}", asset_scale=scale)


def _kitchen() -> list[Fixture]:
    """Background is visual only and kept beyond the shared planner's reach."""
    (cx, cy), (length, depth) = TABLE["centre"], TABLE["size"]
    cabinet, door = (0.16, 0.24, 0.31), (0.22, 0.33, 0.42)
    oak, stone, countertop = (0.57, 0.39, 0.24), (0.60, 0.65, 0.62), (0.88, 0.78, 0.66)
    island = [_box("island_slab", (cx, cy, -0.027), (length, depth, 0.050), countertop, "matte"),
              _box("island_cabinet", (cx, cy, FLOOR / 2 - 0.025),
                   (length - 0.10, depth - 0.12, -FLOOR - 0.10), cabinet),
              _box("island_toe_kick", (cx, cy, FLOOR + 0.05), (length - 0.22, depth - 0.24, 0.10), (0.09, 0.14, 0.20))]
    for k in range(4):
        x = cx + (k - 1.5) * (length - 0.14) / 4
        island += [_box(f"island_door_{k}", (x, cy + depth / 2 - 0.055, -0.445),
                        ((length - 0.18) / 4, 0.025, 0.72), door),
                   _box(f"island_handle_{k}", (x, cy + depth / 2 - 0.035, -0.18), (0.15, 0.018, 0.014),
                        (0.68, 0.53, 0.32), "metal")]
    rear = [_box("rear_cabinet", (0.50, -1.60, FLOOR / 2), (3.40, 0.60, -FLOOR - 0.04), oak, "wood"),
            _box("rear_worktop", (0.50, -1.60, 0.02), (3.48, 0.68, 0.04), stone, "ceramic"),
            _box("upper_cabinets", (0.50, -2.04, 1.00), (3.40, 0.34, 0.64), (0.88, 0.88, 0.82))]
    for k in range(6):
        x = -0.92 + k * 0.568
        rear += [_box(f"rear_door_{k}", (x, -1.286, -0.44), (0.55, 0.025, 0.75), oak, "wood"),
                 _box(f"rear_handle_{k}", (x, -1.268, -0.18), (0.15, 0.018, 0.012), (0.20, 0.22, 0.22), "metal"),
                 _box(f"upper_door_{k}", (x, -1.858, 1.00), (0.55, 0.025, 0.61), (0.93, 0.93, 0.88))]
    walls = [_box("back_wall", (0.0, -2.32, FLOOR + 1.40), (6.20, 0.06, 2.80), (0.91, 0.92, 0.88)),
             _box("side_wall", (-3.10, 0.0, FLOOR + 1.40), (0.06, 4.60, 2.80), (0.91, 0.92, 0.88)),
             _box("window_frame", (-3.055, 0.20, 0.67), (0.025, 1.64, 1.24), (0.96, 0.96, 0.92)),
             _box("window_glass", (-3.035, 0.20, 0.67), (0.012, 1.50, 1.10), (0.55, 0.74, 0.83), "glass"),
             _box("window_mullion", (-3.020, 0.20, 0.67), (0.012, 0.035, 1.10), (0.96, 0.96, 0.92))]
    for row in range(3):
        for col in range(12):
            walls.append(_box(f"backsplash_{row}_{col}", (-1.13 + col * 0.29, -2.278, 0.16 + row * 0.16),
                              (0.282, 0.012, 0.152), (0.80, 0.85, 0.80), "ceramic"))
    fridge_h = MODEL_SPECS["fridge"]["size_m"][2]
    microwave_h = MODEL_SPECS["microwave"]["size_m"][2]
    board_h = MODEL_SPECS["cutting_board"]["size_m"][2] * 0.5
    tray_h = MODEL_SPECS["white_tray"]["size_m"][2]
    props = [_model("fridge", (-2.10, -1.55, FLOOR + fridge_h / 2)),
             _model("microwave", (1.52, -1.57, 0.04 + microwave_h / 2)),
             _model("cutting_board", (0.65, -1.55, 0.04 + board_h / 2), scale=0.5, yaw=12),
             _model("white_tray", (-0.20, -1.55, 0.04 + tray_h / 2), yaw=5)]
    return [surface("kitchen_floor", (0.0, 0.0), (8.0, 8.0), (0.64, 0.57, 0.46), z=FLOOR + 0.0003, material="wood"),
            _group("kitchen_island", "kitchen_island", island),
            surface("countertop", TABLE["centre"], TABLE["size"], countertop, z=0.0003, material="matte"),
            _group("kitchen_cabinets", "kitchen_cabinets", rear),
            _group("kitchen_room", "kitchen_room", walls), _group("nvidia_kitchen_props", "kitchen_props", props)]


def _apple(pos) -> Obj:
    """A uniformly scaled small apple; cylindrical collision proxy, not fruit deformation."""
    size = tuple(0.75 * v for v in MODEL_SPECS["apple"]["size_m"])
    diameter, height = max(size[:2]), size[2]
    return Obj("apple", "fruit", pos, 0.0,
               [Part("mesh", (0.0, 0.0, height / 2), size, (0.7, 0.2, 0.1), collide=False,
                     asset="snack_kitchen/models/apple", asset_scale=0.75, name="apple/visual"),
                Part("cylinder", (0.0, 0.0, height / 2), (diameter, diameter, height), (0.7, 0.2, 0.1),
                     visible=False, name="apple/collider")],
               0.15, attributes={"name": "apple", "graspable": True, "source": "NVIDIA Residential",
                                 "asset_scale": 0.75}, role="item", support="table", friction=(0.8, 0.7))


def _milk(pos, yaw) -> Obj:
    """Small rectangular milk pack, with broad flat gripping faces and visible product labels."""
    w, d, h = 0.055, 0.060, 0.145
    blue, white = (0.08, 0.30, 0.68), (0.96, 0.97, 0.93)
    parts = [Part("box", (0, 0, h / 2), (w, d, h), white, material="cardboard", name="milk/body"),
             Part("box", (0, 0, 0.014), (w + 0.0004, d + 0.0004, 0.028), blue,
                  material="cardboard", collide=False, name="milk/blue_band")]
    labels = [Label("MILK", (0, 0, h + 0.0006), (0.049, 0.031), fg=white, bg=blue, yaw_deg=180.0 - yaw)]
    for normal, x, y, width in (("x", w / 2, 0, d), ("-x", -w / 2, 0, d),
                                 ("y", 0, d / 2, w), ("-y", 0, -d / 2, w)):
        labels.append(Label("MILK", (x * 1.025, y * 1.025, h * 0.57), (width * 0.91, 0.040),
                            normal=normal, fg=white, bg=blue))
    return Obj("milk", "drink", pos, yaw, parts, 0.22, labels,
               {"name": "milk", "graspable": True, "source": "procedural", "item_key": "milk"},
               role="item", support="table")


def _water(pos, yaw) -> Obj:
    """Blue water bottle with a broad body, narrower shoulder/neck, blue cap and WATER labels."""
    blue, body = (0.05, 0.30, 0.82), (0.49, 0.77, 0.94)
    # A single rigid object. Labels are visible product packaging, not identity metadata for the agent.
    # Keep the broad body under the pads at the standard grasp depth; a long skinny neck leaves no body contact.
    parts = [Part("cylinder", (0, 0, 0.0725), (0.055, 0.055, 0.145), body,
                  material="plastic", name="water/body"),
             Part("cylinder", (0, 0, 0.149), (0.044, 0.044, 0.008), body,
                  material="plastic", name="water/shoulder"),
             Part("cylinder", (0, 0, 0.1565), (0.026, 0.026, 0.007), body,
                  material="plastic", name="water/neck"),
             Part("cylinder", (0, 0, 0.164), (0.030, 0.030, 0.008), blue,
                  material="plastic", name="water/cap")]
    labels = [Label("WATER", (0, s * 0.0279, 0.083), (0.044, 0.031),
                    normal="y" if s > 0 else "-y", fg=blue) for s in (-1, 1)]
    return Obj("water_bottle", "drink", pos, yaw, parts, 0.18, labels,
               {"name": "water bottle", "graspable": True, "source": "procedural", "item_key": "water_bottle"},
               role="item", support="table")


def _item(key, pos, yaw) -> Obj:
    if key == "apple":
        return _apple(pos)
    if key == "milk":
        return _milk(pos, yaw)
    if key == "water_bottle":
        return _water(pos, yaw)
    if key in BUILT:
        name, category, d, h, cd, ch, body, cap = BUILT[key]
        obj = capped_container(key, category, *pos, yaw, d, h, cd, ch, body, cap, fill=0.9,
                                attributes={"name": name, "graspable": True, "source": "procedural", "item_key": key},
                                role="item", support="table")
        obj.labels.append(Label("FRUIT", (0, 0, h + ch + 0.0006), (0.054, 0.025),
                                fg=(0.20, 0.31, 0.06), bg=(0.99, 0.89, 0.44), yaw_deg=180.0 - yaw))
        return obj
    name, category, shape, mass = YCB[key]
    width, height, depth = ycb_spec(key)["size"]
    size = (width, depth, height) if shape == "box" else (width, width, height)
    graspable = key not in ("007_tuna_fish_can", "025_mug")
    return Obj(f"food_{key}", category, pos, yaw,
               [Part("mesh", (0, 0, height / 2), (width, depth, height), (0.7, 0.7, 0.7),
                     asset=f"ycb:{key}", tilt_x_deg=90, collide=False),
                Part(shape, (0, 0, height / 2), size, (0.7, 0.7, 0.7), visible=False)], mass,
               attributes={"name": name, "graspable": graspable, "source": f"YCB {key}", "license": "CC BY 4.0",
                           "item_key": key}, role="item" if graspable else "unreachable", support="table")


def _items(difficulty, seed, fixtures) -> list[Obj]:
    keys = ITEMS[difficulty]
    clearance, random_yaw = CONFIG[difficulty]
    sizes = {key: float(np.prod(_item(key, (0, 0, 0), 0).footprint)) for key in keys}
    for attempt in range(150):
        rng = np.random.default_rng([seed, attempt])
        placer, placed = Placer(rng, fixtures, clearance=clearance), {}
        for key in sorted(keys, key=lambda k: -sizes[k]):
            for _ in range(400):
                xy = np.asarray(ITEMS_AREA["centre"]) + (rng.random(2) - 0.5) * (np.asarray(ITEMS_AREA["size"]) - 0.07)
                if not 0.42 <= np.linalg.norm(xy) <= 0.85:
                    continue
                # In the baseline, point package fronts towards the table camera rather than nutrition panels.
                yaw = float(rng.uniform(0, 180)) if random_yaw else (180.0 if key in YCB else 0.0)
                obj = _item(key, (*map(float, xy), 0.0), yaw)
                # The wrist camera extends beside the fingers: short food needs extra clearance from tall cartons.
                apart = all((obj.height > 0.12) == (other.height > 0.12) or
                            np.linalg.norm(xy - np.asarray(other.pos[:2])) >=
                            (np.hypot(*obj.footprint) + np.hypot(*other.footprint)) / 2 + 0.10
                            for other in placed.values())
                if placer.free(obj) and apart:
                    placed[key] = placer.add(obj)
                    break
            else:
                break
        if len(placed) == len(keys):
            return [placed[key] for key in keys]
    raise RuntimeError(f"snack_kitchen {difficulty} seed {seed}: could not generate a separated food layout")


def layout() -> list[dict]:
    return []  # tray and basket are physical objects; no privileged placement-region IDs


def build(difficulty: str, seed: int) -> Scenario:
    fixtures, regions = [], []
    for cid, name, size, rim, color, xy in CONTAINERS:
        yaw = degrees(atan2(xy[0], -xy[1]))
        fixtures.append(tray(cid, (*xy, 0.0), yaw, size=size, rim=rim, color=color, wall=0.006,
                             category=cid, description=f"{name} (open, {round(rim * 1000)} mm rim)"))
        regions.append(Region(cid, "tray", (*xy, 0.004), (size[0] - 0.012, size[1] - 0.012), yaw, name,
                              "serving" if cid == "serving_tray" else "storage", accepts="any"))
    objects = _items(difficulty, seed, fixtures)
    notes = {"objects": len(objects), "example_requests": list(EXAMPLE_REQUESTS),
             "minimum_counter_clearance_m": CONFIG[difficulty][0],
             "receiving_tray_initially_empty": True,
             "task": "Collect the requested snack items, preserve exclusions, ask before substituting missing items.",
             "asset_source": "NVIDIA Residential_NVD@10012 and NVIDIA Isaac Sim YCB",
             "asset_manifest": "assets/snack_kitchen/nvidia_manifest.json",
             "validation_scope": "Static geometry/visibility and offline planner; live RGB-D and grasps require Isaac Sim."}
    return Scenario(ID, TITLE, difficulty, seed,
                   description="A home kitchen island with snacks on the left, a teal serving tray and a storage "
                               "basket on the right; cabinetry, refrigerator, and microwave in the background.",
                   objects=objects, fixtures=fixtures, decor=_kitchen(), regions=regions, notes=notes, version=VERSION,
                   lights=[Light("rect", 1800.0, (1.0, 0.97, 0.91), pos=(0.0, 0.35, 2.30), size=(2.0, 1.5)),
                           Light("rect", 450.0, (1.0, 0.90, 0.76), pos=(0.6, -1.60, 0.65), size=(2.5, 0.2))],
                   view={"table": dict(TABLE), "eye": (3.2, 3.6, 2.40), "table_camera": TABLE_CAMERA,
                         "target": (0.0, -0.25, 0.05), "show_workspace_overlay": False})
