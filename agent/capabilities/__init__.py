"""Capabilities: the operations this software implements, and how each one runs.

Gemma may name any operation in a plan; the ones implemented here are offered to it as available now, each with its
own argument interface (specs.py adapt_args). An operation without an implementation is reported unavailable, never
replaced. Gemma cannot pass code and never produces joint targets. Executors call the existing, validated tools:
motions still go through SkillPlanner / ViewpointPlanner + CollisionChecker + RosExecutor + the safety layer.

    specs.py         SPECS (CapabilitySpec per capability), available(), planner_menu(), StepResult, Context
    perception.py    observe, describe_scene, answer, locate                  (effect: none)
    inspection.py    inspect, explore, look (wrist camera; arm moves, objects do not) (effect: sensing)
    manipulation.py  pick, place, pick_place (PickPlaceWorkflow), return_home (effect: object_change / sensing)
    planning.py      decompose                                                 (effect: none)
    verification.py  verify: goal / sub-goal conditions over measured state     (internal; effect: none)

EXECUTORS maps each implemented operation to the worker that owns it and its executor fn(ctx, args, req). The
supervisor does not know operation names: it reads the step's request context (step index, the later steps' safety
and effect classes) and the spec's properties. Executors are looked up through this module at call time, so tests can
replace `capabilities.run_<name>`.
"""

import sys as _sys


from agent.capabilities.inspection import MAX_EXPLORE_VIEWPOINTS, _wrist_observation, run_explore, run_inspect, run_look
from agent.capabilities.manipulation import _failure_outcome, _skill, run_pick, run_pick_place, run_place, run_return_home
from agent.capabilities.perception import (_ensure_described, _find_in, _location_check, _target_mark,
                                           facts_text, run_answer, run_describe_scene, run_locate, run_observe,
                                           scene_report)
from agent.capabilities.planning import run_decompose, world_facts
from agent.capabilities.specs import (EFFECT, INTERNAL, MOTION, SPECS, CapabilitySpec, Context, StepResult, adapt_args,
                                      available, check_worker, planner_menu)
from agent.capabilities.verification import run_verify

__all__ = ["MAX_EXPLORE_VIEWPOINTS", "_wrist_observation", "run_explore", "run_inspect", "run_look", "_failure_outcome", "_skill",
           "run_pick", "run_pick_place", "run_place", "run_return_home", "_ensure_described", "_find_in",
           "_location_check", "_target_mark", "facts_text", "run_answer", "run_describe_scene", "run_locate",
           "run_observe", "scene_report", "run_decompose", "world_facts", "EFFECT", "MOTION", "SPECS", "CapabilitySpec",
           "Context", "StepResult", "available", "planner_menu", "INTERNAL", "run_verify", "adapt_args", "EXECUTORS", "check_worker"]

_me = _sys.modules[__name__]


def _later(req) -> list:
    """The steps after this one, as the supervisor describes them (safety, effect, authorized)."""
    return (req.context or {}).get("later_steps") or []


def _index(req) -> int:
    return (req.context or {}).get("step_index") or 0


def _last_motion(req) -> bool:  # no later step moves the arm: this one ends at HOME
    return not any(s.get("safety") == "motion" for s in _later(req))


def _clear_view(args, req) -> bool:  # an authorized object change follows: park the arm so it can reuse this view
    return args["clear_view"] if "clear_view" in args else any(
        s.get("effect") == "object_change" and s.get("authorized") for s in _later(req))


def _worker(name: str) -> str:
    return (SPECS.get(name) or INTERNAL[name]).worker


EXECUTORS = {  # implemented operation -> (worker from its spec, executor(ctx, args, req))
    "observe": (_worker("observe"), lambda c, a, r: _me.run_observe(c, a, clear_view=_clear_view(a, r))),
    "describe_scene": (_worker("describe_scene"), lambda c, a, r: _me.run_describe_scene(c, a)),
    "answer": (_worker("answer"), lambda c, a, r: _me.run_answer(c, a)),
    "locate": (_worker("locate"), lambda c, a, r: _me.run_locate(c, a)),
    "inspect": (_worker("inspect"), lambda c, a, r: _me.run_inspect(c, a)),
    "explore": (_worker("explore"), lambda c, a, r: _me.run_explore(c, a)),
    "look": (_worker("look"), lambda c, a, r: _me.run_look(c, a)),
    "pick": (_worker("pick"), lambda c, a, r: _me.run_pick(c, a, _index(r))),
    "place": (_worker("place"), lambda c, a, r: _me.run_place(c, a, _last_motion(r), _index(r))),
    "pick_place": (_worker("pick_place"), lambda c, a, r: _me.run_pick_place(c, a, _last_motion(r), _index(r))),
    "return_home": (_worker("return_home"), lambda c, a, r: _me.run_return_home(c, a)),
    "decompose": (_worker("decompose"), lambda c, a, r: _me.run_decompose(c, a)),
    "verify": (_worker("verify"), lambda c, a, r: _me.run_verify(c, a)),  # internal: not offered to Gemma
}
