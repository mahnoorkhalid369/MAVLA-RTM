"""The capability registry: what the planner may ask for (CapabilitySpec: inputs, preconditions, safety class,
effect class, failures, verification), the planner's menu, the typed step result and the per-instruction
context the executors use.

Safety classes: "read_only" (cameras / models only, no robot command), "motion" (robot commands, each one
collision-checked and verified). Effect classes (agent/authority.py): none | sensing | object_change."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    summary: str  # shown to the planner
    args: str  # JSON shape shown to the planner
    safety: str  # read_only | motion
    preconditions: str
    effects: str
    failures: str
    verification: str
    implemented: bool = True
    effect: str = "none"  # none | sensing | object_change (agent/authority.py): what running it may change


@dataclass
class StepResult:
    outcome: str  # agent.task_state.OUTCOMES
    summary: str
    data: dict = field(default_factory=dict)


_SPECS = [
    CapabilitySpec("observe", "look at the table now and measure every object", '{"do": "observe"}',
                   "read_only", "cameras streaming", "fresh observation in the task state (facts + Gemma descriptions)",
                   "no frame; no objects measured", "at least one object measured, or reported as none"),
    CapabilitySpec("describe_scene", "tell the user which objects are on the table and where", '{"do": "describe_scene"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "a grounded scene report",
                   "no fresh observation", "the report lists only measured objects; names marked as Gemma's"),
    CapabilitySpec("answer", "answer a question about the objects on the table", '{"do": "answer", "question": "<the question>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "an answer (a hypothesis)",
                   "no observation; invalid model answer", "refers only to measured objects; marked unverified"),
    CapabilitySpec("locate", "find one described object on the table and say where it is", '{"do": "locate", "object": "<words>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "object id and measured position",
                   "not visible / not confirmed / ambiguous", "same resolution, cross-check and confirmation as pick_place"),
    CapabilitySpec("pick", "grasp ONE object and hold it (no destination)", '{"do": "pick", "object": "<words>"}', "motion",
                   "gripper empty; object resolvable and graspable", "the object is held (lifted)",
                   "not found / ambiguous / too wide / out of reach / no contact / lost while lifting",
                   "finger contact at the grasp and retention after the lift (finger stall)"),
    CapabilitySpec("place", "put the object the robot is HOLDING onto or next to another object",
                   '{"do": "place", "object": "it", "relation": "on" | "next_to" | "left_of" | "right_of" | "in_front_of" | '
                   '"behind", "target": "<destination object>"}', "motion",
                   "an object is held; destination resolvable", "the held object stands at the destination; arm at HOME",
                   "nothing held / destination not found / no free reachable spot / collision / placement out of tolerance",
                   "contact during the carry, camera placement check, HOME joints"),
    CapabilitySpec("decompose", "plan a multi-object goal over the measured objects", '{"do": "decompose", "goal": "<the goal>"}',
                   "read_only", "a fresh observation (taken automatically)", "concrete steps on #ids + final conditions",
                   "invalid plan / not achievable with the available steps", "references must be measured objects"),
    CapabilitySpec("pick_place", "move ONE object onto or next to another object",
                   '{"do": "pick_place", "object": "<object to move>", "relation": "on" | "next_to" | "left_of" | '
                   '"right_of" | "in_front_of" | "behind", "target": "<destination object>"}',
                   "motion", "gripper empty; object and destination resolvable (asks the user if ambiguous)",
                   "the object stands at the destination; arm at HOME after the last motion step",
                   "not found / ambiguous / out of reach / collision / no contact / placement out of tolerance / safety reject",
                   "finger contact, camera placement check (xy 10 mm, z 5 mm, tilt 5 deg), HOME joints (1.5 deg)"),
    CapabilitySpec("return_home", "move the arm to its safe home pose", '{"do": "return_home"}', "motion",
                   "nothing held; gripper open", "arm at HOME", "holding; path collides; execution failure",
                   "HOME joints within 1.5 deg (joint feedback)"),
    CapabilitySpec("inspect", "move the wrist camera closer to look at one visible object in detail",
                   '{"do": "inspect", "object": "<words>"}', "motion",
                   "object visible and confirmed; nothing held", "close-up position / size / description (wrist camera); arm back at HOME",
                   "object not found / no safe viewpoint / move failed", "viewpoint reached (joint feedback), wrist measurement associated"),
    CapabilitySpec("explore", "look for something the table camera cannot see (e.g. behind another object)",
                   '{"do": "explore", "object": "<words>", "near": "<words for an object it may be behind, or null>"}', "motion",
                   "nothing held", "wrist observations of hidden regions in the world model; arm back at HOME",
                   "no hidden region / no safe viewpoint / not found", "finding is decided by locate, not by moving"),
]

EFFECT = {"pick": "object_change", "place": "object_change", "pick_place": "object_change", "inspect": "sensing",
          "explore": "sensing", "return_home": "sensing"}  # everything else: none (read-only)

SPECS: dict[str, CapabilitySpec] = {s.name: replace(s, effect=EFFECT.get(s.name, "none")) for s in _SPECS}

MOTION = {n for n, s in SPECS.items() if s.safety == "motion"}


def planner_menu(specs: dict[str, CapabilitySpec] | None = None) -> str:
    """The capability list as the planner sees it (default: SPECS)."""
    return "\n".join(f"- {s.name}: {s.summary}. args {s.args}" + ("" if s.implemented else " (not available yet)")
                     for s in (SPECS if specs is None else specs).values())


def available(disabled=()) -> dict[str, CapabilitySpec]:
    """SPECS with the named capabilities marked not available (implemented=False); SPECS itself is never changed."""
    return {n: (replace(s, implemented=False) if n in disabled else s) for n, s in SPECS.items()}


@dataclass
class Context:
    """What executors may use. One instance per instruction, owned by the supervisor."""
    tools: object
    log: Callable
    run_dir: object
    responder: object
    budgets: object
    state: object  # agent.task_state.TaskState
    obs: object = None  # latest agent.tools.Observation (table camera)
    extra_obs: list = field(default_factory=list)  # wrist observations taken during this instruction
    vlm_calls: Callable = lambda: 0  # Gemma calls so far (set by the supervisor; WorkerRegistry metrics)
