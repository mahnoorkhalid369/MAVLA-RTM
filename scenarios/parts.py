"""Reusable procedural building blocks (pure data): fixtures (racks, trays, bins, plinths), visual-only decor
(bench surface, zone markings, back panels, cabinets) and compound rigid objects (capped containers, boxes).
Every builder returns core.Fixture / core.Obj in its own frame; scenarios place them."""

from __future__ import annotations

import numpy as np

from scenarios.core import Fixture, Label, Obj, Part

DENSITY = {"plastic": 950.0, "glass": 2500.0, "wood": 600.0, "cardboard": 250.0, "metal": 7800.0, "foam": 60.0}


# ------------------------------------------------------------------------------------------- fixtures
def rack(id: str, pos, yaw_deg: float, rows: int, cols: int, pitch=(0.06, 0.08), hole: float = 0.036,
         height: float = 0.055, color=(0.20, 0.45, 0.80), slot_labels: list | None = None, title: str = "",
         category: str = "sample_rack", description: str = "") -> tuple[Fixture, list]:
    """Open-ended tube rack: base plate, end walls, a top grid of bars leaving square holes.
    Columns along the rack's x (pitch[0]), rows along y (pitch[1]); row 0 is nearest the rack's -y edge.
    Returns (fixture, slots) with slots = [(row, col, (x, y) in the rack frame)]; the hole floor is the base
    plate top (BASE_T). Slot labels lie on the bar in front (-y side) of each hole."""
    px, py = pitch
    t_base, t_top, wall = 0.006, 0.006, 0.006
    W, D = cols * px + (px - hole), rows * py + (py - hole)
    xs = [(c - (cols - 1) / 2) * px for c in range(cols)]
    ys = [(r - (rows - 1) / 2) * py for r in range(rows)]
    parts = [Part("box", (0, 0, t_base / 2), (W, D, t_base), color, material="plastic", name=f"{id}/base")]
    ztop = height - t_top / 2
    for k in range(cols + 1):  # bars along y between columns
        x = (k - cols / 2) * px
        parts.append(Part("box", (x, 0, ztop), (px - hole, D, t_top), color, material="plastic", name=f"{id}/bar_y{k}"))
    for k in range(rows + 1):  # bars along x between rows
        y = (k - rows / 2) * py
        parts.append(Part("box", (0, y, ztop), (W, py - hole, t_top), color, material="plastic", name=f"{id}/bar_x{k}"))
    for s in (-1, 1):  # end walls carry the top grid
        parts.append(Part("box", (s * (W / 2 - wall / 2), 0, height / 2), (wall, D, height), color, material="plastic",
                          name=f"{id}/end{s:+d}"))
    labels, slots = [], []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            slots.append((r, c, (x, y)))
            if slot_labels:
                labels.append(Label(slot_labels[r * cols + c], (x, y - py / 2, height + 0.0006),
                                    (min(px - 0.008, 0.034), min(py - hole - 0.006, 0.020))))
    if title:
        labels.append(Label(title, (0, -D / 2 - 0.0005, height / 2), (min(W * 0.8, 0.16), min(height * 0.55, 0.03)),
                            normal="-y", fg=(1, 1, 1), bg=color))
    return Fixture(id, category, tuple(pos), yaw_deg, parts, labels, description), slots


def tray(id: str, pos, yaw_deg: float, size=(0.22, 0.15), rim: float = 0.018, color=(0.85, 0.87, 0.90),
         wall: float = 0.005, base: float = 0.004, category: str = "tray", stripes=None, title: str = "",
         title_fg=(0.05, 0.05, 0.05), title_bg=None, description: str = "", rim_color=None) -> Fixture:
    """Shallow open tray (also bins with a tall rim). stripes = (colour a, colour b, n segments per wall)
    paints the rim in alternating segments (hazard / quarantine marking)."""
    L, Wd = size
    parts = [Part("box", (0, 0, base / 2), (L, Wd, base), color, material="plastic", name=f"{id}/base")]
    rc = rim_color or color
    for axis, length, offs in (("x", L, Wd / 2 - wall / 2), ("y", Wd, L / 2 - wall / 2)):
        for s in (-1, 1):
            p, sz = ((0, s * offs, rim / 2), (length, wall, rim)) if axis == "x" else ((s * offs, 0, rim / 2), (wall, length, rim))
            # one collider per wall; painted stripes are visual-only segments over it (collision cost scales with
            # the number of collider boxes, and the stripes are colour, not geometry)
            parts.append(Part("box", p, sz, rc, material="plastic", name=f"{id}/rim_{axis}{s:+d}", visible=not stripes))
            if stripes:
                n = stripes[2]
                seg = length / n
                for k in range(n):
                    mid = -length / 2 + seg * (k + 0.5)
                    q, qs = (((mid, s * offs, rim / 2), (seg, wall + 0.0004, rim + 0.0004)) if axis == "x"
                             else ((s * offs, mid, rim / 2), (wall + 0.0004, seg, rim + 0.0004)))
                    parts.append(Part("box", q, qs, stripes[k % 2], material="plastic", collide=False,
                                      name=f"{id}/stripe_{axis}{s:+d}_{k}"))
    labels = []
    if title:
        labels.append(Label(title, (0, -Wd / 2 - 0.0006, rim / 2), (min(L * 0.85, 0.20), rim * 0.75), normal="-y",
                            fg=title_fg, bg=title_bg or rc))
    return Fixture(id, category, tuple(pos), yaw_deg, parts, labels, description)


def plinth(id: str, pos, yaw_deg: float, size, color, category: str, description: str = "", top_color=None,
           labels=None, material: str = "painted") -> Fixture:
    """A solid static block (instrument, cooler, shelf base). top_color adds a contrasting lid."""
    parts = [Part("box", (0, 0, size[2] / 2), size, color, material=material, name=f"{id}/body")]
    if top_color:
        parts.append(Part("box", (0, 0, size[2] + 0.003), (size[0] * 0.92, size[1] * 0.92, 0.006), top_color,
                          material="plastic", name=f"{id}/lid"))
    return Fixture(id, category, tuple(pos), yaw_deg, parts, labels or [], description)


# ----------------------------------------------------------------------------------------- decor (visual)
def surface(id: str, centre, size, color, z: float = 0.0003, material: str = "matte", labels=None) -> Fixture:
    """Visual-only work-surface slab / floor marking (no collider: objects rest on the physics floor)."""
    return Fixture(id, "surface", (centre[0], centre[1], 0.0), 0.0,
                   [Part("box", (0, 0, z - 0.0002), (size[0], size[1], 0.0004), color, material=material, collide=False)],
                   labels or [])


def zone_outline(id: str, centre, size, color, width: float = 0.012, z: float = 0.0006, title: str = "",
                 title_side: str = "-y", title_fg=(0.05, 0.05, 0.05)) -> Fixture:
    """Floor tape rectangle (visual only) with an optional floor text label outside its edge."""
    L, W = size
    parts = [Part("box", (0, s * W / 2, z), (L + width, width, 0.0004), color, collide=False) for s in (-1, 1)]
    parts += [Part("box", (s * L / 2, 0, z), (width, W + width, 0.0004), color, collide=False) for s in (-1, 1)]
    labels = []
    if title:
        off = {"-y": (0, -W / 2 - 0.022), "y": (0, W / 2 + 0.022), "x": (L / 2 + 0.022, 0), "-x": (-L / 2 - 0.022, 0)}[title_side]
        yaw = {"-y": 0.0, "y": 0.0, "x": 90.0, "-x": 90.0}[title_side]
        labels.append(Label(title, (off[0], off[1], z + 0.0004), (min(max(L, W) * 0.9, 0.024 * len(title) + 0.02), 0.030),
                            fg=title_fg, bg=color, yaw_deg=yaw))
    return Fixture(id, "zone_marking", (centre[0], centre[1], 0.0), 0.0, parts, labels)


def panel(id: str, centre, size, color, yaw_deg: float = 0.0, labels=None, material: str = "painted",
          category: str = "wall") -> Fixture:
    """Visual-only vertical panel / cabinet block standing on the floor (size = x, y, z in its frame)."""
    return Fixture(id, category, (centre[0], centre[1], 0.0), yaw_deg,
                   [Part("box", (0, 0, size[2] / 2), size, color, material=material, collide=False)], labels or [])


def sign(id: str, centre, z: float, size, text: str, bg=(0.98, 0.98, 0.98), fg=(0.05, 0.05, 0.05)) -> Fixture:
    """Visual-only wall sign facing -y (towards the robot) at height z, size = (width, height)."""
    return Fixture(id, "sign", (centre[0], centre[1], 0.0), 0.0,
                   [Part("box", (0, 0, z), (size[0], 0.006, size[1]), bg, collide=False)],
                   [Label(text, (0, -0.0035, z), (size[0] * 0.92, size[1] * 0.45), normal="-y", fg=fg, bg=bg)])


def shelf_unit(id: str, centre, yaw_deg: float, width: float, depth: float, levels, color, items=()) -> Fixture:
    """Visual-only open shelf (frame + boards) with simple items [(level, x, shape, size, colour)]."""
    parts, H = [], max(levels) + 0.03
    for s in (-1, 1):
        parts.append(Part("box", (s * width / 2, 0, H / 2), (0.02, depth, H), color, collide=False))
    parts.append(Part("box", (0, depth / 2, H / 2), (width, 0.01, H), tuple(c * 0.85 for c in color), collide=False))
    for z in levels:
        parts.append(Part("box", (0, 0, z), (width, depth, 0.018), color, collide=False))
    for lv, x, shape, size, col in items:
        z = levels[lv] + 0.009 + size[2] / 2
        parts.append(Part(shape, (x, 0, z), size, col, material="plastic", collide=False))
    return Fixture(id, "shelf", (centre[0], centre[1], 0.0), yaw_deg, parts)


# ------------------------------------------------------------------------------------------- objects
def capped_container(id: str, category: str, x: float, y: float, z: float, yaw: float, body_d: float, body_h: float,
                     cap_d: float, cap_h: float, body_color, cap_color, body_material: str = "plastic",
                     cap_material: str = "plastic", fill: float = 0.4, top_text: str = "", side_text: str = "",
                     side_normal: str = "x", attributes=None, role: str = "", support: str = "floor",
                     friction=(0.9, 0.8), wall_density: float | None = None) -> Obj:
    """Upright cylindrical container with a cap: one rigid body (2 colliders), mass from a thin wall +
    contents (fill = fraction of the volume at water density). top_text on the cap, side_text facing
    side_normal (visual labels)."""
    parts = [Part("cylinder", (0, 0, body_h / 2), (body_d, body_d, body_h), body_color, material=body_material, name=f"{id}/body"),
             Part("cylinder", (0, 0, body_h + cap_h / 2), (cap_d, cap_d, cap_h), cap_color, material=cap_material, name=f"{id}/cap")]
    r = body_d / 2
    shell = (np.pi * body_d * body_h + np.pi * r * r) * 0.0012 * (wall_density or DENSITY[body_material])
    contents = np.pi * r * r * body_h * fill * 1000.0
    cap = np.pi * (cap_d / 2) ** 2 * cap_h * 0.3 * DENSITY["plastic"]
    labels = []
    if top_text:
        s = cap_d * 0.62
        labels.append(Label(top_text, (0, 0, body_h + cap_h + 0.0005), (s, s * 0.6), bg=cap_color,
                            fg=_contrast(cap_color)))
    if side_text:
        n = {"x": (1, 0), "-x": (-1, 0), "y": (0, 1), "-y": (0, -1)}[side_normal]
        w = min(body_d * 0.9, 0.04)
        labels.append(Label(side_text, (n[0] * (r + 0.0006), n[1] * (r + 0.0006), body_h * 0.55), (w, min(body_h * 0.4, w * 0.6)),
                            normal=side_normal))
    return Obj(id, category, (x, y, z), yaw, parts, float(shell + contents + cap), labels, dict(attributes or {}), role,
               support, friction)


def box_obj(id: str, category: str, x: float, y: float, z: float, yaw: float, size, color, material: str = "cardboard",
            mass: float | None = None, top_text: str = "", side_text: str = "", side_normal: str = "x", attributes=None,
            role: str = "", support: str = "floor", friction=(0.8, 0.7), label_bg=(0.97, 0.97, 0.95)) -> Obj:
    parts = [Part("box", (0, 0, size[2] / 2), tuple(size), color, material=material, name=f"{id}/body")]
    m = mass if mass is not None else size[0] * size[1] * size[2] * DENSITY.get(material, 500.0)
    labels = []
    if top_text:
        labels.append(Label(top_text, (0, 0, size[2] + 0.0005), (size[0] * 0.7, min(size[1] * 0.45, size[0] * 0.35)), bg=label_bg))
    if side_text:
        n = {"x": (1, 0), "-x": (-1, 0), "y": (0, 1), "-y": (0, -1)}[side_normal]
        half = size[0] / 2 if side_normal in ("x", "-x") else size[1] / 2
        face_w = size[1] if side_normal in ("x", "-x") else size[0]
        labels.append(Label(side_text, (n[0] * (half + 0.0006), n[1] * (half + 0.0006), size[2] * 0.55),
                            (face_w * 0.75, min(size[2] * 0.45, face_w * 0.3)), normal=side_normal, bg=label_bg))
    return Obj(id, category, (x, y, z), yaw, parts, float(m), labels, dict(attributes or {}), role, support, friction)


def _contrast(rgb) -> tuple:
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (0.02, 0.02, 0.02) if lum > 0.45 else (0.98, 0.98, 0.98)
