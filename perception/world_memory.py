"""Long-term object memory: what the robot has seen, where, and when. Saved to JSON.

No ROS, no model. The agent adds sightings (label + 3-D estimate) during scans and tasks,
and moves objects in memory after it picks / places them.

    memory = WorldMemory(path)
    obj = memory.observe("red cube", position_xyz, size_xy)   # merges with a known object nearby
    memory.place(obj.id, new_top_centre)                      # after the robot moved it
    print(memory.describe())                                  # for the model's prompt / the user

Run `python3 -m perception.world_memory` for a self-test.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SAME_PLACE = 0.06  # m: a sighting this close to a remembered object (and similar label) is that object
SAME_SPOT = 0.04  # m: this close is the same object whatever the label says
SAME_HEIGHT = 0.02  # m: top heights must match too (a cube stacked on another is a different object)


@dataclass
class MemoryObject:
    id: int
    label: str  # e.g. "red cube"
    position: list[float]  # top-centre (m, world); last known
    size_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])  # footprint (m)
    status: str = "on_table"  # "on_table" | "held"
    times_seen: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    moved_by_robot: bool = False
    yaw_deg: float | None = None  # rotation about vertical, -45..45 deg; None = round / unknown

    def pose_text(self) -> str:
        theta = "n/a (round or not measured)" if self.yaw_deg is None else f"{self.yaw_deg:+.0f} deg"
        return f"x {self.position[0] * 1000:.0f} mm, y {self.position[1] * 1000:.0f} mm, theta {theta}"


COLOURS = {"red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray",
           "grey", "teal", "lime", "navy", "olive", "tan", "gold", "silver", "beige", "magenta", "cyan"}
SHAPES = {"cube": "cube", "box": "cube", "block": "cube", "cylinder": "cylinder", "can": "cylinder",
          "sphere": "sphere", "ball": "sphere"}


# Neighbouring colour names that a camera under scene lighting easily confuses
COLOUR_FAMILIES = [{"yellow", "gold"}, {"gray", "grey", "silver"}, {"tan", "beige"}, {"navy", "blue"}]


def _with_family(colours: set[str]) -> set[str]:
    out = set(colours)
    for family in COLOUR_FAMILIES:
        if colours & family:
            out |= family
    return out


def matches(query: str, label: str) -> bool:
    """Does a seen label fit what the user asked for? Colours must agree (or be close relatives,
    e.g. yellow / gold); shapes too if both are given."""
    q, l = _words(query), _words(label)
    q_col, l_col = _with_family(q & COLOURS), l & COLOURS
    if q_col and not q_col & l_col:
        return False
    q_shape = {SHAPES[w] for w in q if w in SHAPES}
    l_shape = {SHAPES[w] for w in l if w in SHAPES}
    if q_shape and l_shape and not q_shape & l_shape:
        return False
    return bool(q_col or q_shape)


def _words(label: str) -> set[str]:
    return {w for w in label.lower().replace("-", " ").split() if w not in ("a", "the", "small", "big", "large")}


class WorldMemory:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.objects: dict[int, MemoryObject] = {}
        self.next_id = 1
        # where the robot looked for something and did NOT find it: {"query", "heading", "view", "time"}
        self.searched: list[dict] = []
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            self.next_id = data.get("next_id", 1)
            self.objects = {o["id"]: MemoryObject(**o) for o in data.get("objects", [])}
            self.searched = data.get("searched", [])

    # ------------------------------------------------------------------ updates
    def observe(self, label: str, position, size_xy=(0.0, 0.0), yaw_deg: float | None = None) -> MemoryObject:
        """Record a sighting; merge with the closest matching remembered object if there is one."""
        p = np.asarray(position, float)
        best, best_d = None, np.inf
        for obj in self.objects.values():
            if obj.status != "on_table":
                continue
            d = float(np.linalg.norm(np.asarray(obj.position[:2]) - p[:2]))
            if abs(obj.position[2] - p[2]) > SAME_HEIGHT:
                continue
            similar = bool(_words(obj.label) & _words(label))
            if (d < SAME_PLACE and similar) or d < SAME_SPOT:
                if d < best_d:
                    best, best_d = obj, d
        now = time.time()
        if best is None:
            best = MemoryObject(self.next_id, label, p.tolist(), list(map(float, size_xy)), yaw_deg=yaw_deg)
            self.objects[best.id] = best
            self.next_id += 1
        else:  # running average smooths depth noise between sightings
            w = 1.0 / (best.times_seen + 1)
            best.position = ((1 - w) * np.asarray(best.position) + w * p).tolist()
            if max(size_xy) > 0:
                best.size_xy = list(map(float, size_xy))
            if yaw_deg is not None:
                best.yaw_deg = yaw_deg
            best.times_seen += 1
            best.last_seen = now
        self.save()
        return best

    def pick(self, obj_id: int) -> None:
        self.objects[obj_id].status = "held"
        self.save()

    def place(self, obj_id: int, top_centre) -> None:
        obj = self.objects[obj_id]
        obj.position, obj.status, obj.moved_by_robot, obj.last_seen = list(map(float, top_centre)), "on_table", True, time.time()
        self.save()

    def clear(self) -> None:
        self.objects, self.next_id, self.searched = {}, 1, []
        self.save()

    # ------------------------------------------------------- where it looked
    def mark_not_found(self, query: str, heading_deg: float, view: str) -> None:
        self.searched.append({"query": query, "heading": round(float(heading_deg)), "view": view, "time": time.time()})
        self.save()

    def was_searched(self, query: str, heading_deg: float, view: str, max_age: float = 900.0) -> bool:
        now = time.time()
        return any(
            r["view"] == view and abs(r["heading"] - heading_deg) < 10 and now - r["time"] < max_age
            and matches(query, r["query"])
            for r in self.searched
        )

    def found(self, query: str) -> None:
        """Forget the 'not found here' notes for something that has now been found."""
        self.searched = [r for r in self.searched if not matches(query, r["query"])]
        self.save()

    def describe_searched(self) -> str:
        if not self.searched:
            return ""
        by_query: dict[str, list[str]] = {}
        for r in self.searched:
            by_query.setdefault(r["query"], []).append(f"J1 {r['heading']:+d} deg {r['view']}")
        return "\n".join(f"looked for {q} without finding it at: {', '.join(v)}" for q, v in by_query.items())

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"next_id": self.next_id, "objects": [asdict(o) for o in self.objects.values()], "searched": self.searched}
        self.path.write_text(json.dumps(data, indent=2))

    # -------------------------------------------------------------------- views
    def get(self, obj_id: int) -> MemoryObject | None:
        return self.objects.get(obj_id)

    def find(self, description: str) -> MemoryObject | None:
        """The most recently seen object on the table that matches a description ("red cube")."""
        hits = [o for o in self.objects.values() if o.status == "on_table" and matches(description, o.label)]
        return max(hits, key=lambda o: o.last_seen) if hits else None

    def describe(self, now: float | None = None) -> str:
        """One line per object, for the model's prompt and for the user."""
        if not self.objects:
            return "(nothing remembered yet)"
        now = now or time.time()
        lines = []
        for o in sorted(self.objects.values(), key=lambda o: o.id):
            x, y, z = (v * 1000 for v in o.position)
            age = now - o.last_seen
            ago = f"{age:.0f} s" if age < 120 else f"{age / 60:.0f} min"
            theta = "theta n/a" if o.yaw_deg is None else f"theta {o.yaw_deg:+.0f} deg"
            where = "in the gripper" if o.status == "held" else f"at ({x:.0f}, {y:.0f}) mm, top {z:.0f} mm, {theta}"
            moved = ", moved by you" if o.moved_by_robot else ""
            lines.append(f"#{o.id} {o.label} {where} (seen {o.times_seen}x, last {ago} ago{moved})")
        return "\n".join(lines)


if __name__ == "__main__":
    m = WorldMemory()
    a = m.observe("red cube", [0.0, 0.58, 0.06], [0.06, 0.06])
    m.observe("red block", [0.004, 0.583, 0.061])  # same object seen again
    b = m.observe("green cube", [0.15, 0.50, 0.05])
    m.observe("green", [0.151, 0.502, 0.05])  # 2 mm away, different wording: same object (same spot)
    m.observe("lime cube", [0.15, 0.50, 0.09])  # stacked on the green one: different height, new object
    m.observe("black cube", [-0.60, 0.05, 0.045])
    m.pick(a.id)
    m.place(a.id, [0.30, 0.40, 0.06])
    print(m.describe())
