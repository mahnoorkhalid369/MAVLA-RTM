"""Scenario 1: biomedical laboratory sample handling (v2.0).

A sample-processing workstation on a dark epoxy bench with six functional zones: INCOMING rack (A1-A4 / B1-B4),
PROCESSING rack (P1-P4), STORAGE rack (S1-S4), INSPECTION station (white pad with a target ring), QUARANTINE tray
(Q1, Q2, hazard-striped rim) and BUFFER tray (T1-T3). Self-standing 50 mL-style tubes and amber glass vials carry
unique sample IDs on the cap top and on a side label; cap colours repeat, so identity needs the label. Medium /
hard add a centrifuge beside the incoming rack and (hard) a sample cooler that hides one vial from the table
camera. Back wall, reagent shelf, fume hood and sign are visual-only decor beyond the arm's reach.

Rigid bodies only (no liquids). Tubes 30 x 100 mm + 34 x 14 mm cap, vials 24 x 48 mm + 26 x 12 mm cap: within
the 2F-85's 80 mm grip. Every zone is placed where the table camera sees it past the robot at HOME and where the
project's planner reaches it (scenarios/checks.py). Spacing follows the gripper: the open pads reach ~75 mm
from the object centre along the closing direction, so neighbours in that direction are >= 90-100 mm apart;
receiving rack holes are 56 mm so a held tube keeps the planner's 10 mm clearance while it is lowered.
"""

from __future__ import annotations

import numpy as np

from scenarios.core import Label, Light, Part, Placer, Region, Scenario, jitter, rot_z
from scenarios.parts import (box_obj, capped_container, panel, plinth, rack, shelf_unit, sign, surface, tray,
                                    zone_outline)

ID = "laboratory"
TITLE = "Biomedical laboratory sample handling"
VERSION = "2.0"

CAPS = {  # cap colour code -> sRGB
    "red": (0.85, 0.12, 0.12), "blue": (0.12, 0.32, 0.85), "green": (0.15, 0.62, 0.25), "yellow": (0.95, 0.82, 0.10),
    "purple": (0.55, 0.25, 0.75), "white": (0.94, 0.94, 0.94),
}
TUBE = dict(body_d=0.030, body_h=0.100, cap_d=0.034, cap_h=0.014, body_color=(0.88, 0.90, 0.93))
VIAL = dict(body_d=0.024, body_h=0.048, cap_d=0.026, cap_h=0.012, body_color=(0.58, 0.33, 0.10), body_material="glass")
ID_LETTERS = "HKMNRX"  # sample IDs never use the slot letters A B P S Q T
BASE_T, TRAY_T, PAD_T = 0.006, 0.004, 0.012  # support heights: rack base plate, tray base, inspection pad

# Difficulty = configuration (counts per zone, colours, yaw of the labels, occluders), not separate code.
CONFIG = {
    "easy": dict(incoming=6, processing=0, storage=0, quarantine=0, buffer=2, hidden=False, unreachable=False,
                 cap_colours=4, random_yaw=False, centrifuge=False, cooler=False, tip_box=False),
    "medium": dict(incoming=6, processing=1, storage=1, quarantine=0, buffer=2, hidden=False, unreachable=False,
                   cap_colours=3, random_yaw=True, centrifuge=True, cooler=False, tip_box=True),
    "hard": dict(incoming=5, processing=1, storage=2, quarantine=1, buffer=1, hidden=True, unreachable=True,
                 cap_colours=3, random_yaw=True, centrifuge=True, cooler=True, tip_box=True),
}
FEATURES = {
    "easy": ["8 containers (6 tubes, 2 vials), every cap colour used exactly twice: identity needs the label",
             "side labels face the table camera; every container visible and graspable; every receiving region free"],
    "medium": ["10 containers, 3 cap colours (3-4 containers per colour): look-alikes",
               "random container yaw: side labels may face away (cap-top IDs need the wrist camera)",
               "centrifuge behind the incoming rack (clutter near the grasp approaches)",
               "pipette-tip box distractor; P and S slots partly occupied"],
    "hard": ["12 containers, 3 cap colours (4 per colour)",
             "vial_hidden: hidden from the table camera behind the sample cooler (wrist viewpoint exists, graspable)",
             "vial_far: beyond the arm's reach (0.99 m), the only unreachable container",
             "occupied locations: S1, S2, a P slot, Q1, a buffer spot",
             "random yaw, centrifuge, tip box"],
}


def _sample_ids(rng, n):
    nums = rng.choice(np.arange(100, 1000), size=n, replace=False)
    return [f"{ID_LETTERS[int(rng.integers(len(ID_LETTERS)))]}{k}" for k in nums]


def inspection_station(pos) -> tuple:
    """Low white pad (static) with a printed target ring: put a sample here for a wrist-camera close look."""
    from scenarios.core import Fixture
    parts = [Part("box", (0, 0, PAD_T / 2), (0.14, 0.14, PAD_T), (0.95, 0.95, 0.95), material="matte",
                  name="inspection_station/base"),
             Part("cylinder", (0, 0, PAD_T + 0.0002), (0.070, 0.070, 0.0004), (0.08, 0.08, 0.08), collide=False),
             Part("cylinder", (0, 0, PAD_T + 0.0004), (0.056, 0.056, 0.0004), (0.95, 0.95, 0.95), collide=False)]
    labels = [Label("INSPECT", (0, -0.056, PAD_T + 0.0006), (0.10, 0.018)),
              Label("INSPECTION", (0.0706, 0, PAD_T / 2), (0.12, 0.010), normal="x", fg=(1, 1, 1), bg=(0.10, 0.10, 0.12))]
    return Fixture("inspection_station", "inspection_station", tuple(pos), 0.0, parts, labels,
                   "white 140 x 140 x 12 mm pad with a target ring for close wrist-camera inspection")


def build(difficulty: str, seed: int) -> Scenario:
    cfg = CONFIG[difficulty]
    rng = np.random.default_rng(seed)
    fixtures, objects, regions, zones = [], [], [], []

    def jit(xy, a=0.006):
        return (*(np.asarray(xy) + jitter(rng, a)), 0.0)

    def frame(fx, local):
        p = np.asarray(fx.pos[:2]) + (rot_z(fx.yaw_deg) @ np.array([*local, 0.0]))[:2]
        return float(p[0]), float(p[1])

    # --------------------------------------------------------------------------- fixtures + receiving regions
    inc_pos = jit((-0.02, 0.66))
    inc_labels = [f"{r}{c}" for r in "AB" for c in range(1, 5)]
    inc, inc_slots = rack("incoming_rack", inc_pos, 0.0, 2, 4, pitch=(0.06, 0.10), slot_labels=inc_labels, title="INCOMING",
                          color=(0.20, 0.42, 0.78), description="2 x 4 tube rack (36 mm holes), A1-A4 front row, B1-B4")
    proc_pos = jit((-0.62, 0.32))
    proc, proc_slots = rack("processing_rack", proc_pos, 90.0, 1, 4, pitch=(0.075, 0.10), hole=0.056,
                            slot_labels=[f"P{i}" for i in range(1, 5)], title="PROCESSING", color=(0.93, 0.55, 0.15),
                            description="1 x 4 receiving rack (56 mm holes), slots P1-P4")
    sto_pos = jit((0.60, 0.30))
    sto, sto_slots = rack("storage_rack", sto_pos, 90.0, 1, 4, pitch=(0.075, 0.10), hole=0.056,
                          slot_labels=[f"S{i}" for i in range(1, 5)], title="STORAGE", color=(0.15, 0.55, 0.35),
                          description="1 x 4 receiving rack (56 mm holes), slots S1-S4")
    insp_pos = jit((0.22, 0.42), 0.004)
    insp = inspection_station(insp_pos)
    q_pos = jit((0.45, 0.64), 0.004)
    quar = tray("quarantine_tray", q_pos, 0.0, size=(0.20, 0.22), rim=0.035, color=(0.20, 0.20, 0.20),
                stripes=((0.98, 0.80, 0.05), (0.08, 0.08, 0.08), 4), category="quarantine_tray",
                description="tray with a hazard-striped rim, spots Q1 / Q2; anything placed here is isolated")
    buf_pos = jit((0.64, -0.10), 0.004)
    buf = tray("buffer_tray", buf_pos, 0.0, size=(0.30, 0.19), rim=0.018, color=(0.80, 0.83, 0.86), category="buffer_tray",
               description="shallow tray for temporary set-down, spots T1-T3 in one row")
    for k, dx in enumerate((-0.05, 0.05)):
        quar.labels.append(Label(f"Q{k + 1}", (dx, -0.07, TRAY_T + 0.0006), (0.03, 0.018), fg=(0.05, 0.05, 0.05),
                                 bg=(0.98, 0.80, 0.05)))
    for k, dx in enumerate((-0.09, 0.0, 0.09)):
        buf.labels.append(Label(f"T{k + 1}", (dx, -0.055, TRAY_T + 0.0006), (0.03, 0.018)))
    fixtures += [inc, proc, sto, insp, quar, buf]
    if cfg["centrifuge"]:
        fixtures.append(plinth("centrifuge", jit((0.25, 0.82), 0.003), 0.0, (0.17, 0.18, 0.19), (0.93, 0.93, 0.95),
                               "instrument", "benchtop centrifuge (static), behind the incoming rack: clutter beside "
                               "the grasp approaches (kept 90 mm clear of the nearest tube column)",
                               top_color=(0.35, 0.37, 0.42)))
    if cfg["cooler"]:
        fixtures.append(plinth("sample_cooler", (-0.30, 0.56, 0.0), 0.0, (0.10, 0.18, 0.24), (0.20, 0.45, 0.85),
                               "instrument", "sample cooler (static): hides the vial behind it from the table camera",
                               top_color=(0.92, 0.92, 0.94)))

    for (r, c, xy), lab in zip(inc_slots, inc_labels):
        regions.append(Region(f"incoming_{lab}", "rack_slot", (*frame(inc, xy), BASE_T), (0.036, 0.036), 0.0, lab,
                              "incoming sample slot (pick only: 36 mm holes)", zone="incoming"))
    for name, fx, slots, z in (("processing", proc, proc_slots, "P"), ("storage", sto, sto_slots, "S")):
        for (r, c, xy), k in zip(slots, range(1, 5)):
            regions.append(Region(f"{name}_{z}{k}", "rack_slot", (*frame(fx, xy), BASE_T), (0.056, 0.056), 90.0, f"{z}{k}",
                                  f"{name} slot", zone=name, accepts="tube"))
    regions.append(Region("inspection_pad", "inspection", (insp_pos[0], insp_pos[1], PAD_T), (0.056, 0.056), 0.0, "INSPECT",
                          "close wrist-camera inspection", zone="inspection", accepts="any"))
    for k, dx in enumerate((-0.05, 0.05)):
        regions.append(Region(f"quarantine_Q{k + 1}", "tray_spot", (q_pos[0] + dx, q_pos[1], TRAY_T), (0.05, 0.05), 0.0,
                              f"Q{k + 1}", "isolation of suspect samples", zone="quarantine", accepts="any"))
    for k, dx in enumerate((-0.09, 0.0, 0.09)):
        regions.append(Region(f"buffer_T{k + 1}", "tray_spot", (buf_pos[0] + dx, buf_pos[1], TRAY_T), (0.04, 0.04), 0.0,
                              f"T{k + 1}", "temporary set-down", zone="buffer", accepts="vial"))
    zones += [
        {"id": "incoming", "title": "INCOMING", "fixture": "incoming_rack", "centre": list(inc_pos[:2]), "size": [0.33, 0.30],
         "purpose": "samples arrive here; pick only"},
        {"id": "processing", "title": "PROCESSING", "fixture": "processing_rack", "centre": list(proc_pos[:2]),
         "size": [0.20, 0.38], "purpose": "samples being processed"},
        {"id": "storage", "title": "STORAGE", "fixture": "storage_rack", "centre": list(sto_pos[:2]), "size": [0.20, 0.38],
         "purpose": "finished samples"},
        {"id": "inspection", "title": "INSPECTION", "fixture": "inspection_station", "centre": list(insp_pos[:2]),
         "size": [0.18, 0.18], "purpose": "close look with the wrist camera (labels, caps, damage)"},
        {"id": "quarantine", "title": "QUARANTINE", "fixture": "quarantine_tray", "centre": list(q_pos[:2]),
         "size": [0.26, 0.28], "purpose": "isolation of suspect or mislabelled samples"},
        {"id": "buffer", "title": "BUFFER", "fixture": "buffer_tray", "centre": list(buf_pos[:2]), "size": [0.36, 0.25],
         "purpose": "temporary set-down during rearrangement"},
    ]

    # ------------------------------------------------------------------------------------------- samples
    n_total = (cfg["incoming"] + cfg["processing"] + cfg["storage"] + cfg["quarantine"] + cfg["buffer"]
               + cfg["hidden"] + cfg["unreachable"])
    palette = list(rng.permutation(list(CAPS)))[:cfg["cap_colours"]]
    caps = iter(rng.permutation([palette[k % len(palette)] for k in range(n_total)]))  # each colour >= 2 times
    ids = iter(_sample_ids(rng, n_total))
    placer = Placer(rng, fixtures)
    reg = {r.id: r for r in regions}

    def yaw():
        return float(rng.uniform(0.0, 360.0)) if cfg["random_yaw"] else 0.0

    def add(kind, oid, x, y, z, role, support, region=None, **attrs):
        cap, sid = str(next(caps)), next(ids)
        spec = TUBE if kind == "tube" else VIAL
        o = capped_container(oid, "sample_tube" if kind == "tube" else "sample_vial", float(x), float(y), z, yaw(), **spec,
                             cap_color=CAPS[cap], top_text=sid, side_text=sid,
                             attributes={"sample_id": sid, "cap_colour": cap, "container": kind, "label_text": [sid, sid],
                                         "zone": region.zone if region else None, "slot": region.label if region else None,
                                         **attrs}, role=role, support=support)
        if region is not None:
            region.occupied_by = oid
        objects.append(placer.add(o))
        return o

    def in_slot(region):
        return np.asarray(region.centre[:2]) + jitter(rng, 0.0015)

    for i, s in enumerate(sorted(rng.choice(8, size=cfg["incoming"], replace=False))):
        r = reg[f"incoming_{inc_labels[s]}"]
        add("tube", f"tube_{i + 1}", *in_slot(r), BASE_T, "sample", "incoming_rack", r)
    for i, k in enumerate(sorted(rng.choice(4, size=cfg["processing"], replace=False))):
        r = reg[f"processing_P{k + 1}"]
        add("tube", f"processing_tube_{i + 1}", *in_slot(r), BASE_T, "occupies_destination", "processing_rack", r)
    for k in range(cfg["storage"]):
        r = reg[f"storage_S{k + 1}"]
        add("tube", f"stored_tube_{k + 1}", *in_slot(r), BASE_T, "occupies_destination", "storage_rack", r)
    for k in range(cfg["quarantine"]):
        r = reg[f"quarantine_Q{k + 1}"]
        add("vial", "quarantined_vial", *in_slot(r), TRAY_T, "occupies_destination", "quarantine_tray", r, status="quarantined")
    for i, k in enumerate(sorted(rng.choice(3, size=cfg["buffer"], replace=False))):
        r = reg[f"buffer_T{k + 1}"]
        add("vial", f"vial_{i + 1}", *in_slot(r), TRAY_T, "sample", "buffer_tray", r)
    if cfg["hidden"]:  # on the -x side of the cooler; the table camera is on +x
        add("vial", "vial_hidden", -0.47, 0.56 + float(jitter(rng, 0.01)[0]), 0.0, "occluded", "floor")
    if cfg["unreachable"]:
        add("vial", "vial_far", -0.26, 0.955, 0.0, "unreachable", "floor")
    for o in objects:  # look-alikes: same container type and cap colour (ground truth for ambiguity experiments)
        o.attributes["look_alikes"] = [p.id for p in objects if p is not o and p.attributes["cap_colour"] == o.attributes["cap_colour"]
                                       and p.category == o.category]
    if cfg["tip_box"]:
        def make_tips(x, y, z, yaw_deg):
            return box_obj("pipette_tip_box", "consumable_box", x, y, z, yaw_deg, (0.11, 0.075, 0.055), (0.20, 0.55, 0.90),
                           material="plastic", mass=0.15, top_text="TIPS 200uL", attributes={"contents": "pipette tips"},
                           role="distractor")
        objects.append(placer.place_in_area(make_tips, (-0.37, 0.77), (0.04, 0.03), 0.0, yaw_range=(-20.0, 20.0)))

    # --------------------------------------------------------------------------------------- decor + zones
    decor = [
        surface("floor", (0.0, 0.3), (6.0, 6.0), (0.74, 0.76, 0.78), z=0.0001, material="painted"),
        surface("bench", (0.0, 0.33), (1.90, 1.36), (0.15, 0.16, 0.18), z=0.0004, material="epoxy"),
        zone_outline("zone_incoming", inc_pos[:2], (0.33, 0.30), (0.95, 0.95, 0.95), title="INCOMING"),
        zone_outline("zone_processing", proc_pos[:2], (0.20, 0.38), (0.95, 0.60, 0.20), title="PROCESSING", title_side="-x"),
        zone_outline("zone_storage", sto_pos[:2], (0.20, 0.38), (0.30, 0.80, 0.45), title="STORAGE", title_side="x"),
        zone_outline("zone_inspection", insp_pos[:2], (0.18, 0.18), (0.75, 0.75, 0.78), title="INSPECTION"),
        zone_outline("zone_quarantine", q_pos[:2], (0.26, 0.28), (0.95, 0.20, 0.15), title="QUARANTINE", title_side="y"),
        zone_outline("zone_buffer", buf_pos[:2], (0.36, 0.25), (0.40, 0.70, 0.95), title="BUFFER"),
        panel("back_wall", (0.0, 1.42), (2.6, 0.04, 1.1), (0.88, 0.90, 0.92)),
        panel("side_wall", (-1.32, 0.45), (0.04, 2.0, 1.1), (0.86, 0.89, 0.91)),
        panel("fume_hood", (0.70, 1.30), (0.70, 0.20, 0.95), (0.80, 0.82, 0.85), category="fume_hood"),
        panel("fume_hood_sash", (0.70, 1.195), (0.62, 0.01, 0.50), (0.55, 0.70, 0.78), material="glass", category="fume_hood"),
        shelf_unit("reagent_shelf", (-0.45, 1.30), 0.0, 0.80, 0.20, (0.25, 0.60, 0.95), (0.93, 0.93, 0.93),
                   items=[(0, -0.30, "cylinder", (0.07, 0.07, 0.16), (0.55, 0.30, 0.10)),
                          (0, -0.18, "cylinder", (0.07, 0.07, 0.16), (0.55, 0.30, 0.10)),
                          (0, 0.05, "box", (0.12, 0.09, 0.10), (0.20, 0.45, 0.80)),
                          (1, -0.25, "box", (0.20, 0.12, 0.14), (0.90, 0.90, 0.88)),
                          (1, 0.10, "cylinder", (0.09, 0.09, 0.20), (0.85, 0.85, 0.90)),
                          (2, 0.0, "box", (0.40, 0.15, 0.10), (0.95, 0.60, 0.15))]),
        sign("lab_sign", (-0.05, 1.395), 0.85, (0.24, 0.16), "BSL-2  LAB 4", bg=(0.98, 0.82, 0.10)),
    ]
    lights = [Light("rect", 9000.0, (1.0, 0.98, 0.95), pos=(0.0, 0.40, 2.2), size=(1.2, 0.6))]
    description = " ".join(__doc__.split("\n\n")[1].split())
    return Scenario(ID, TITLE, difficulty, seed, description, fixtures, objects, decor, regions, lights, zones=zones,
                    view={"eye": (-0.55, 1.30, 1.95), "target": (0.06, 0.30, 0.0),
                          "wrist": ["incoming_rack", "inspection_station"]},
                    notes={"difficulty_features": FEATURES[difficulty], "containers": n_total}, version=VERSION)
