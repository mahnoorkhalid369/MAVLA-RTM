"""Scenario library core: pure data + numpy (no Isaac Sim), so layouts can be generated, checked and tested
offline and built identically in the simulator by scenarios/build.py.

A scenario is a function (difficulty config, seeded rng) -> Scenario:
  fixtures  static rigid parts (racks, trays, bins): boxes / cylinders with colliders, never moved by physics
  objects   dynamic rigid bodies the robot may manipulate; each is one rigid body made of primitive parts
            (e.g. a tube = body cylinder + cap cylinder) plus optional flat text labels (visual only)
  decor     visual-only parts far outside the robot's reach (walls, cabinets, zone markings): no colliders
  regions   named placement / storage regions (rack slots, tray areas, zones) with a surface height
  lights    extra lights on top of the default dome light

Conventions (same as sim/scene_objects.py): metres, world = robot base frame, +y = the arm's front at J1 = 0,
+x its right, z up, floor at z = 0 (objects rest on the floor or on fixtures). Colours are sRGB 0-1.
Everything written into Scenario.metadata() is GROUND TRUTH for development and evaluation; nothing here is
read by the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from config.agent import REACH_MAX_M, REACH_MIN_M

LIBRARY_VERSION = "1.0"
DIFFICULTIES = ("easy", "medium", "hard")
REACH_MIN, REACH_MAX = REACH_MIN_M, REACH_MAX_M  # m from the base: reliable top-down picks (config/agent.py)
MAX_GRIP_WIDTH = 0.080  # m (2F-85 opens 85 mm; skills keep 5 mm margin)
# Kept clear for the frozen scene's props (sim/scene_objects.py): microphone and the table-camera stand
KEEP_CLEAR = [((-0.45, 0.35), 0.09), ((1.5, 0.25), 0.20)]
DECOR_MIN_RADIUS = 1.15  # m: decor must be beyond the arm's reach (with gripper) everywhere


@dataclass
class Part:
    """A primitive in its owner's frame (an object's or a fixture's): box (size = x, y, z edges) or cylinder
    (size = diameter, diameter, height), centred at `pos`, rotated `yaw_deg` about z."""
    shape: str
    pos: tuple
    size: tuple
    color: tuple
    yaw_deg: float = 0.0
    material: str = "matte"  # matte | plastic | glass | metal | rubber | cardboard | painted
    collide: bool = True
    visible: bool = True  # False: collider only (e.g. one wall behind several visual-only coloured segments)
    name: str = ""


@dataclass
class Label:
    """Flat text label (visual only), in its owner's frame. normal: 'z' (lying face up) or 'x' / '-x' / 'y' /
    '-y' (standing, facing that way). size = (width, height) of the label face in m."""
    text: str
    pos: tuple
    size: tuple
    normal: str = "z"
    fg: tuple = (0.05, 0.05, 0.05)
    bg: tuple = (0.97, 0.97, 0.95)
    yaw_deg: float = 0.0


@dataclass
class Fixture:
    id: str
    category: str
    pos: tuple  # frame origin on the floor (z of the frame = 0 unless raised)
    yaw_deg: float
    parts: list
    labels: list = field(default_factory=list)
    description: str = ""


@dataclass
class Obj:
    id: str
    category: str
    pos: tuple  # x, y of the body frame; z = the surface it stands on (parts are relative to that)
    yaw_deg: float
    parts: list
    mass_kg: float
    labels: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)  # cap colour, sample code, ... (ground truth)
    role: str = ""  # target | distractor | occluded | unreachable | similar | ...
    support: str = "floor"  # fixture / region the object rests on
    friction: tuple = (0.9, 0.8)  # static, dynamic

    @property
    def height(self) -> float:
        return max(p.pos[2] + p.size[2] / 2 for p in self.parts if p.collide)

    @property
    def footprint(self) -> tuple:
        """Bounding box of the collision parts in the object frame (x, y)."""
        xs = [(p.pos[0] - _half_xy(p)[0], p.pos[0] + _half_xy(p)[0]) for p in self.parts if p.collide]
        ys = [(p.pos[1] - _half_xy(p)[1], p.pos[1] + _half_xy(p)[1]) for p in self.parts if p.collide]
        return (max(b for _, b in xs) - min(a for a, _ in xs), max(b for _, b in ys) - min(a for a, _ in ys))


@dataclass
class Region:
    id: str
    kind: str  # rack_slot | tray | zone | bin
    centre: tuple  # x, y, surface z
    size: tuple  # x, y extent (m)
    yaw_deg: float = 0.0
    label: str = ""
    purpose: str = ""
    occupied_by: str | None = None
    zone: str = ""  # functional zone id
    accepts: str = ""  # receiving regions: container type it is meant to receive ("tube" | "vial" | "any"); "" = pick only


@dataclass
class Light:
    kind: str  # distant | rect | sphere
    intensity: float
    color: tuple = (1.0, 1.0, 1.0)
    pos: tuple = (0.0, 0.0, 2.0)
    rotation_deg: tuple = (0.0, 0.0, 0.0)  # xyz Euler (deg)
    size: tuple = (1.0, 1.0)


@dataclass
class Scenario:
    id: str
    title: str
    difficulty: str
    seed: int
    description: str
    fixtures: list
    objects: list
    decor: list  # Fixture-like visual-only groups
    regions: list
    lights: list = field(default_factory=list)
    zones: list = field(default_factory=list)  # functional zones [{id, title, centre, size, purpose}]
    view: dict = field(default_factory=dict)  # overview camera {eye, target} + wrist inspection targets for screenshots
    notes: dict = field(default_factory=dict)  # difficulty-specific notes (what is hard and why)
    version: str = LIBRARY_VERSION

    def metadata(self) -> dict:
        """Ground-truth description (spawn poses, dimensions, materials, regions). Evaluation only."""
        return {
            "scenario_id": self.id, "title": self.title, "version": self.version, "library_version": LIBRARY_VERSION,
            "difficulty": self.difficulty, "seed": self.seed, "description": self.description, "notes": self.notes,
            "units": "m, deg; world = robot base frame (+y front, +x right, z up)",
            "objects": [{"id": o.id, "category": o.category, "role": o.role, "support": o.support,
                         "spawn_pose": {"xyz": [round(o.pos[0], 4), round(o.pos[1], 4), round(o.pos[2], 4)],
                                        "yaw_deg": round(o.yaw_deg, 2)},
                         "dimensions": {"footprint": [round(v, 4) for v in o.footprint], "height": round(o.height, 4)},
                         "distance_from_base": round(float(np.hypot(o.pos[0], o.pos[1])), 3),
                         "mass_kg": round(o.mass_kg, 4), "friction": list(o.friction), "attributes": o.attributes,
                         "parts": [asdict(p) for p in o.parts], "labels": [asdict(lb) for lb in o.labels]}
                        for o in self.objects],
            "fixtures": [{"id": f.id, "category": f.category, "pos": list(f.pos), "yaw_deg": f.yaw_deg,
                          "description": f.description, "parts": [asdict(p) for p in f.parts],
                          "labels": [asdict(lb) for lb in f.labels]} for f in self.fixtures],
            "zones": self.zones,
            "regions": [asdict(r) for r in self.regions],
            "decor": [{"id": d.id, "category": d.category, "pos": list(d.pos)} for d in self.decor],
            "lights": [asdict(lt) for lt in self.lights],
        }


# ---------------------------------------------------------------------------------------------- geometry
def _half_xy(p: Part) -> tuple:
    if p.shape == "cylinder":
        return (p.size[0] / 2, p.size[0] / 2)
    a = np.radians(p.yaw_deg)
    hx, hy = p.size[0] / 2, p.size[1] / 2
    return (abs(np.cos(a)) * hx + abs(np.sin(a)) * hy, abs(np.sin(a)) * hx + abs(np.cos(a)) * hy)


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@dataclass
class WorldPart:
    owner: str
    name: str
    shape: str
    centre: np.ndarray
    R: np.ndarray
    half: np.ndarray  # half extents (cylinder: radius, radius, half height)


def world_parts(owner_id: str, pos, yaw_deg: float, parts, collide_only: bool = True) -> list[WorldPart]:
    Rb = rot_z(yaw_deg)
    out = []
    for i, p in enumerate(parts):
        if collide_only and not p.collide:
            continue
        c = np.asarray(pos, float) + Rb @ np.asarray(p.pos, float)
        R = Rb @ rot_z(p.yaw_deg)
        h = np.array([p.size[0] / 2, (p.size[0] if p.shape == "cylinder" else p.size[1]) / 2, p.size[2] / 2])
        out.append(WorldPart(owner_id, p.name or f"{owner_id}/{i}", p.shape, c, R, h))
    return out


def scenario_parts(sc: Scenario) -> tuple[list[WorldPart], list[WorldPart]]:
    """(fixture parts, object parts) in world coordinates, colliders only."""
    fx = [wp for f in sc.fixtures for wp in world_parts(f.id, f.pos, f.yaw_deg, f.parts)]
    ob = [wp for o in sc.objects for wp in world_parts(o.id, o.pos, o.yaw_deg, o.parts)]
    return fx, ob


def penetration(a: WorldPart, b: WorldPart) -> float:
    """Overlap depth (m) of two z-rotated parts (boxes; cylinders tested as circles in xy), 0 if apart.
    Separating-axis test in xy plus the z interval."""
    dz = min(a.centre[2] + a.half[2], b.centre[2] + b.half[2]) - max(a.centre[2] - a.half[2], b.centre[2] - b.half[2])
    if dz <= 0:
        return 0.0
    d = b.centre[:2] - a.centre[:2]
    if a.shape == "cylinder" and b.shape == "cylinder":
        return max(0.0, min(dz, a.half[0] + b.half[0] - float(np.linalg.norm(d))))
    if a.shape == "cylinder" or b.shape == "cylinder":
        box, cyl = (b, a) if a.shape == "cylinder" else (a, b)
        local = box.R[:2, :2].T @ (cyl.centre[:2] - box.centre[:2])
        q = np.clip(local, -box.half[:2], box.half[:2])
        dist = float(np.linalg.norm(local - q))
        if dist > 0:
            return max(0.0, min(dz, cyl.half[0] - dist))
        inside = float(min(box.half[0] - abs(local[0]), box.half[1] - abs(local[1])))
        return min(dz, cyl.half[0] + inside)
    depth = dz
    for axis in (a.R[:2, 0], a.R[:2, 1], b.R[:2, 0], b.R[:2, 1]):
        ra = abs(a.R[:2, 0] @ axis) * a.half[0] + abs(a.R[:2, 1] @ axis) * a.half[1]
        rb = abs(b.R[:2, 0] @ axis) * b.half[0] + abs(b.R[:2, 1] @ axis) * b.half[1]
        ov = ra + rb - abs(d @ axis)
        if ov <= 0:
            return 0.0
        depth = min(depth, ov)
    return float(depth)


def overlaps(sc: Scenario, tol: float = 0.0005) -> list[tuple[str, str, float]]:
    """Every pair of colliding parts that interpenetrate by more than tol (m): object-object and
    object-fixture (fixture-fixture pairs are rigidly joined parts of one static fixture and are skipped)."""
    fx, ob = scenario_parts(sc)
    out = []
    for i, a in enumerate(ob):
        for b in ob[i + 1:] + fx:
            if a.owner == b.owner:
                continue
            d = penetration(a, b)
            if d > tol:
                out.append((a.name, b.name, round(d * 1000, 2)))
    return out


# --------------------------------------------------------------------------------------- placement helpers
class Placer:
    """Deterministic placement: candidate positions are drawn from the seeded rng and accepted only if the
    object's colliders do not overlap anything placed so far (plus a clearance) and keep-clear zones."""

    def __init__(self, rng: np.random.Generator, fixtures: list, clearance: float = 0.004) -> None:
        self.rng, self.clearance = rng, clearance
        self.parts = [wp for f in fixtures for wp in world_parts(f.id, f.pos, f.yaw_deg, f.parts)]

    def free(self, o: Obj) -> bool:
        for wp in world_parts(o.id, o.pos, o.yaw_deg, o.parts):
            grown = WorldPart(wp.owner, wp.name, wp.shape, wp.centre, wp.R, wp.half + [self.clearance, self.clearance, 0.0])
            if any(penetration(grown, q) > 0 for q in self.parts):
                return False
            if any(np.hypot(*(wp.centre[:2] - np.asarray(c))) < r + max(wp.half[:2]) for c, r in KEEP_CLEAR):
                return False
        return True

    def add(self, o: Obj) -> Obj:
        self.parts += world_parts(o.id, o.pos, o.yaw_deg, o.parts)
        return o

    def place_in_area(self, make, centre, size, z: float, yaw_range=(0.0, 0.0), tries: int = 200,
                      area_yaw: float = 0.0) -> Obj:
        """make(x, y, z, yaw) -> Obj; sample uniformly in a (possibly rotated) rectangle until it fits."""
        R = rot_z(area_yaw)[:2, :2]
        for _ in range(tries):
            u = (self.rng.random(2) - 0.5) * np.asarray(size)
            x, y = np.asarray(centre[:2]) + R @ u
            yaw = float(self.rng.uniform(*yaw_range)) if yaw_range[1] > yaw_range[0] else yaw_range[0]
            o = make(float(x), float(y), z, yaw)
            if self.free(o):
                return self.add(o)
        raise RuntimeError(f"could not place an object in the area at {centre} after {tries} tries")


def jitter(rng: np.random.Generator, amount: float) -> np.ndarray:
    return (rng.random(2) - 0.5) * 2 * amount
