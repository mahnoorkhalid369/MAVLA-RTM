"""Capabilities: the only things the planner can ask for, and how each one runs.

The planner (Gemma) composes these into a plan; it cannot name anything else (the plan schema is an enum of the
names in SPECS), cannot pass code, and never produces joint targets. Executors call the existing, validated
tools: motions still go through SkillPlanner + CollisionChecker + RosExecutor + the safety layer.

    specs.py         SPECS (CapabilitySpec per capability), available(), planner_menu(), StepResult, Context
    perception.py    observe, describe_scene, answer, locate                  (effect: none)
    inspection.py    inspect, explore (wrist camera; arm moves, objects do not) (effect: sensing)
    manipulation.py  pick, place, pick_place (PickPlaceWorkflow), return_home (effect: object_change / sensing)
    planning.py      decompose                                                 (effect: none)
    verification.py  verify: goal / sub-goal conditions over measured state     (internal; effect: none)

The supervisor calls each executor through this package (`capabilities.run_<name>`), so every name below stays
importable from agent.capabilities.
"""

from agent.capabilities.inspection import MAX_EXPLORE_VIEWPOINTS, _wrist_observation, run_explore, run_inspect, run_look
from agent.capabilities.manipulation import _failure_outcome, _skill, run_pick, run_pick_place, run_place, run_return_home
from agent.capabilities.perception import (GENERIC_WORDS, _ensure_described, _find_in, _location_check, _target_mark,
                                           facts_text, run_answer, run_describe_scene, run_locate, run_observe,
                                           scene_report)
from agent.capabilities.planning import run_decompose, world_facts
from agent.capabilities.specs import (EFFECT, INTERNAL, MOTION, SPECS, CapabilitySpec, Context, StepResult, available,
                                      planner_menu)
from agent.capabilities.verification import run_verify

__all__ = ["MAX_EXPLORE_VIEWPOINTS", "_wrist_observation", "run_explore", "run_inspect", "run_look", "_failure_outcome", "_skill",
           "run_pick", "run_pick_place", "run_place", "run_return_home", "GENERIC_WORDS", "_ensure_described", "_find_in",
           "_location_check", "_target_mark", "facts_text", "run_answer", "run_describe_scene", "run_locate",
           "run_observe", "scene_report", "run_decompose", "world_facts", "EFFECT", "MOTION", "SPECS", "CapabilitySpec",
           "Context", "StepResult", "available", "planner_menu", "INTERNAL", "run_verify"]
