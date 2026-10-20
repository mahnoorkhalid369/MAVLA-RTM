"""The operations this software implements (CapabilitySpec: inputs, preconditions, safety class, effect class,
failures, verification, argument interface), shown to Gemma as what is available NOW, the typed step result and the
per-instruction context the executors use.

This is not the vocabulary of the robot's tasks: Gemma may name any operation in a plan. An operation listed here
validates its own arguments (its interface); an operation not listed here has no implementation and is reported as
unavailable when the plan reaches it (nothing is substituted for it).

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
    adapt: Callable | None = None  # Gemma's flat step fields -> the executor's arguments; ValueError = invalid
    worker: str = ""  # the worker that implements it
    needs_view: bool = False  # needs a fresh measured observation first (the supervisor takes one)
    verify_after: bool = False  # its effect is measurable: the verification worker checks it right after the step


@dataclass
class StepResult:
    outcome: str  # agent.task_state.OUTCOMES
    summary: str
    data: dict = field(default_factory=dict)


# ----------------------------------------------------------------------- argument interfaces (validation)
NULLS = {"", "null", "none"}  # "not given", as a small model sometimes writes JSON null as text
FRAMES = ("robot", "camera")  # coordinate frames the geometry can express (agent/directions.py)
# relations the validated placement planner and the measured relation check compute (agent/tools/manipulation.py,
# agent/tools/resolution.py relation_holds); "in" only as a location constraint on a reference
PLACE_RELATIONS = ("on", "next_to", "left_of", "right_of", "in_front_of", "behind")
LOCATION_RELATIONS = PLACE_RELATIONS + ("in",)


def _text(v):
    return None if v is None or str(v).strip().lower() in NULLS else str(v).strip()


def _need(op: str, raw: dict, *keys) -> None:
    missing = [k for k in keys if _text(raw.get(k)) is None]
    if missing:
        raise ValueError(f"{op} needs {', '.join(missing)}")


def _frame(op: str, raw: dict) -> str:
    f = (_text(raw.get("frame")) or "robot").lower()
    if f not in FRAMES:
        raise ValueError(f"{op}: frame must be one of {FRAMES}, not {f!r}")
    return f


def _relation(op: str, v, allowed) -> str:
    r = (_text(v) or "").lower()
    if r not in allowed:
        raise ValueError(f"{op}: the relation must be one the geometry can compute {allowed}, not {r!r}")
    return r


def _ref(op: str, raw: dict, key: str) -> dict:
    rel, anchor = _text(raw.get("object_relation")), _text(raw.get("object_anchor"))
    if bool(rel) != bool(anchor):
        raise ValueError(f"{op}: object_relation and object_anchor go together")
    return {"text": _text(raw.get(key)), "relation": _relation(op, rel, LOCATION_RELATIONS) if rel else None, "anchor": anchor}


def _none(op):
    return lambda raw: {}


def _target(op):
    def f(raw):
        _need(op, raw, "object")
        r = _ref(op, raw, "object")
        return {"target": r["text"], "relation": r["relation"], "anchor": r["anchor"]}
    return f


def _pick(raw):
    _need("pick", raw, "object")
    return {"object": _ref("pick", raw, "object")}


def _place(raw):
    _need("place", raw, "relation", "target")
    return {"object": {"text": _text(raw.get("object")), "relation": None, "anchor": None},
            "relation": _relation("place", raw.get("relation"), PLACE_RELATIONS),
            "reference": {"text": _text(raw.get("target")), "relation": None, "anchor": None}, "frame": _frame("place", raw)}


def _pick_place(raw):
    _need("pick_place", raw, "object", "relation", "target")
    return {"object": _ref("pick_place", raw, "object"), "relation": _relation("pick_place", raw.get("relation"), PLACE_RELATIONS),
            "reference": {"text": _text(raw.get("target")), "relation": None, "anchor": None}, "frame": _frame("pick_place", raw)}


def _answer(raw):
    _need("answer", raw, "question")
    return {"question": _text(raw.get("question"))}


def _explore(raw):
    _need("explore", raw, "object")
    return {"target": _text(raw.get("object")), "near": _text(raw.get("near"))}


def _decompose(raw):
    _need("decompose", raw, "goal")
    return {"goal": _text(raw.get("goal"))}


def _look(raw):
    from agent.directions import direction_vector
    frame = _frame("look", raw)
    d = raw.get("direction")
    try:
        direction_vector(d, frame)
    except ValueError as exc:
        raise ValueError(f"look: {exc}") from None
    return {"direction": [float(v) for v in d], "frame": frame}


_SPECS = [
    CapabilitySpec("observe", "look at your workspace with the fixed camera now and measure every object", '{"do": "observe"}',
                   "read_only", "cameras streaming", "fresh observation in the task state (facts + Gemma descriptions)",
                   "no frame; no objects measured", "at least one object measured, or reported as none"),
    CapabilitySpec("describe_scene", "tell the user which objects you see and where", '{"do": "describe_scene"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "a grounded scene report",
                   "no fresh observation", "the report lists only measured objects; names marked as Gemma's"),
    CapabilitySpec("answer", "answer a question about the objects you see", '{"do": "answer", "question": "<the question>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "an answer (a hypothesis)",
                   "no observation; invalid model answer", "refers only to measured objects; marked unverified"),
    CapabilitySpec("locate", "find one described object and say where it is", '{"do": "locate", "object": "<words>"}',
                   "read_only", "a fresh observation (taken automatically if missing)", "object id and measured position",
                   "not visible / not confirmed / ambiguous", "same resolution, cross-check and confirmation as pick_place"),
    CapabilitySpec("pick", "grasp ONE object and hold it (no destination)", '{"do": "pick", "object": "<words>"}', "motion",
                   "gripper empty; object resolvable and graspable", "the object is held (lifted)",
                   "not found / ambiguous / too wide / out of reach / no contact / lost while lifting",
                   "finger contact at the grasp and retention after the lift (finger stall)"),
    CapabilitySpec("place", "put the object the robot is HOLDING onto or next to another object",
                   '{"do": "place", "object": "#<held id>", "relation": "on" | "next_to" | "left_of" | "right_of" | "in_front_of" | '
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
    CapabilitySpec("look", "point your wrist camera along a direction and see what is there (the arm moves only to aim "
                   "the camera, and stays there). direction = [x, y, z] in the frame: \"robot\" = your own axes; \"camera\" "
                   "only when the user means directions in the fixed camera's picture",
                   '{"do": "look", "direction": [<x>, <y>, <z>], "frame": "robot"}',
                   "motion", "nothing held", "the wrist camera points that way (arm stays there); wrist observation in the world model",
                   "no safe viewpoint that way / move not executed / viewpoint not reached / holding an object",
                   "camera direction measured from joint feedback (forward kinematics): the looked-at floor point within 30 deg of "
                   "the direction (a vertical direction: axis within 15 deg)"),
    CapabilitySpec("explore", "look for something the fixed camera cannot see (e.g. behind another object)",
                   '{"do": "explore", "object": "<words>", "near": "<words for an object it may be behind, or null>"}', "motion",
                   "nothing held", "wrist observations of hidden regions in the world model; arm back at HOME",
                   "no hidden region / no safe viewpoint / not found", "finding is decided by locate, not by moving"),
]

EFFECT = {"pick": "object_change", "place": "object_change", "pick_place": "object_change", "inspect": "sensing",
          "explore": "sensing", "look": "sensing", "return_home": "sensing"}  # everything else: none (read-only)

ADAPT = {"observe": _none("observe"), "describe_scene": _none("describe_scene"), "answer": _answer,
         "locate": _target("locate"), "pick": _pick, "place": _place, "decompose": _decompose, "pick_place": _pick_place,
         "return_home": _none("return_home"), "inspect": _target("inspect"), "look": _look, "explore": _explore}
WORKER = {"observe": "perception", "describe_scene": "perception", "answer": "perception", "locate": "perception",
          "inspect": "inspection", "explore": "inspection", "look": "inspection", "pick": "manipulation",
          "place": "manipulation", "pick_place": "manipulation", "return_home": "manipulation", "decompose": "planning"}
NEEDS_VIEW = {"describe_scene", "answer", "locate", "inspect", "explore", "look", "decompose"}  # measured view first
VERIFY_AFTER = {"pick", "place", "pick_place", "return_home", "locate", "inspect", "look"}  # measurable effect

SPECS: dict[str, CapabilitySpec] = {s.name: replace(s, effect=EFFECT.get(s.name, "none"), adapt=ADAPT[s.name], worker=WORKER[s.name],
                                                    needs_view=s.name in NEEDS_VIEW, verify_after=s.name in VERIFY_AFTER)
                                    for s in _SPECS}

MOTION = {n for n, s in SPECS.items() if s.safety == "motion"}


# Operations the supervisor delegates but Gemma cannot plan (not in the planner's menu or schema)
INTERNAL: dict[str, CapabilitySpec] = {
    "verify": CapabilitySpec("verify", "compare measured state with a goal or sub-goal's conditions", '{"conditions": [...]}',
                             "read_only", "conditions bound to world objects", "evidence: satisfied / not / undecidable",
                             "no evidence for a condition", "goal predicates over measured state (agent/goals.py)",
                             effect="none", worker="verification"),
}


def adapt_args(name: str, raw: dict) -> dict:
    """Gemma's flat step fields -> arguments: through the operation's own interface if it is implemented
    (ValueError if invalid), otherwise kept as given (the operation is unavailable; it is never replaced)."""
    spec = SPECS.get(name) or INTERNAL.get(name)
    if spec is not None and spec.adapt is not None:
        return spec.adapt(raw)
    return {k: v for k, v in raw.items() if not (isinstance(v, str) and _text(v) is None) and v is not None}


def planner_menu(specs: dict[str, CapabilitySpec] | None = None) -> str:
    """The workers, their responsibilities and the operations each one implements now, as Gemma sees them."""
    from agent.workers import RESPONSIBILITY
    specs = SPECS if specs is None else specs
    out = []
    for w in dict.fromkeys(s.worker for s in specs.values()):
        out.append(f"{w} worker: {RESPONSIBILITY.get(w, '')}")
        out += [f"  - {s.name}: {s.summary}. args {s.args}" + ("" if s.implemented else " (not available yet)")
                for s in specs.values() if s.worker == w]
    return "\n".join(out)


def check_worker(name: str, worker: str | None) -> None:
    """Interface check of Gemma's worker choice: the named worker must implement the operation. An operation nobody
    implements is not an error here (it is reported unavailable when reached)."""
    spec = SPECS.get(name)
    if spec is None:
        return
    if not worker:
        raise ValueError(f"{name}: name the worker for this subtask (\"worker\")")
    if worker.strip().lower() != spec.worker:
        raise ValueError(f"{name}: the {worker} worker does not implement {name}; it is an operation of the {spec.worker} worker")


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
