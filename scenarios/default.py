"""The default scene: the cube scene of the frozen manipulation baseline (manipulation_A_2026-10-14).

Its definition stays where the baseline froze it: the objects are sim/scene_objects.py OBJECTS (21 primitives in
tiers easy / medium / hard / impossible) and the frozen sim/scene.py builds them with the robot, cameras and
fixtures. Both files are checksummed (docs/baselines/manipulation_A_2026-10-14/MANIFEST.sha256), so this entry
REFERS to them instead of copying them: one definition, selectable like every other scenario.

    ./launch/sim.sh --scenario default        or   SCENARIO = "default" in agent/settings.py
"""

from __future__ import annotations

KIND = "frozen"
DESCRIPTION = "frozen cube scene of the manipulation baseline (21 primitives, tiers easy / medium / hard / impossible)"


def objects() -> dict:
    """The scene's objects (read-only view of the frozen sim/scene_objects.py OBJECTS: name -> SceneObject)."""
    from sim.scene_objects import OBJECTS
    return dict(OBJECTS)
