"""Agent tools: the only way the agent touches the world. Geometry and motion are measured/deterministic;
semantic confirmation callbacks are owned by the manipulation worker, not motion planning.

One object, Tools, built from these parts (one file each):

    held.py          HeldObjectState     the held object, re-verified from finger feedback and the wrist camera
    perception.py    PerceptionTools     perceive, update_world, is_fresh, at_gripper   (+ Observation)
    grasp_inspection.py  GraspInspectionTools  local wrist RGB-D association and obstacle evidence before descent
    manipulation.py  ManipulationTools   plan_pick / plan_place_held / plan_pick_place, park_for_view, return_home
    execution.py     ExecutionTools      execute: the only path to robot motion (gated by Tools.controls)
    verification.py  VerificationTools   verify_contact, confirm_held, verify_home, verify_placement, relation_holds

    tools = Tools(bridge, detector, gemma, executor, log, run_dir)
    obs = tools.perceive("table"); tools.update_world(obs)

Tools lives as long as the session: the world model (persistent object ids) and the held object carry over from one
instruction to the next, and so do the words grounding used for each object (object_words, for the reports).
"""

from __future__ import annotations

from agent.controls import ExecutionControls
from agent.performance import Performance
from agent.tools.execution import ExecutionTools
from agent.tools.held import HeldObjectState
from agent.tools.grasp_inspection import GraspInspectionTools
from agent.tools.manipulation import ManipulationTools, PickPlan
from agent.tools.perception import Observation, PerceptionTools
from agent.tools.verification import VerificationTools
from config.agent import Relations, Tolerances
from control.kinematics import Kinematics
from perception.viewpoints import CAM_TIP
from perception.world_model import WorldModel

__all__ = ["Tools", "Observation", "PickPlan"]


class Tools(HeldObjectState, PerceptionTools, GraspInspectionTools, ManipulationTools, ExecutionTools, VerificationTools):
    def __init__(self, bridge, detector, gemma, executor, log, run_dir, live=None, controls=None) -> None:
        self.bridge, self.detector, self.gemma, self.ex, self.log, self.run_dir = bridge, detector, gemma, executor, log, run_dir
        self.live = live  # agent.live.LiveChannel or None (the dashboard is optional)
        self.controls = controls or ExecutionControls()  # motion real / dry_run / blocked
        self.performance = Performance.from_env()
        self.scene_revision = 0  # potential physical disturbance, separate from camera/HOME motion
        if self.bridge is not None:
            self.bridge.log = log
        self.motion_count = 0  # every executed robot / gripper command increments it (observation freshness)
        self.held, self.held_object, self.held_geom, self.held_close_target, self.held_box = None, None, None, None, None
        self.unconfirmed_grasp = None  # (plan, close target) of a grasp the lift check could not confirm
        self.unknown_held = None  # something between the fingers that no grasp recorded
        self.last_referent = None  # world id of the last object handled
        self.object_words: dict[int, str] = {}  # world id -> the words grounding used for it ("tall pink block")
        self.unexplained: dict[int, dict] = {}  # what depth sees standing on the table that no measured object explains
        self.inspection_obstacles = []  # local wrist geometry, cleared by each new table observation
        self.view_cleared_at = -1  # motion_count right after the last park_for_view
        self.world = WorldModel()
        self.kin = Kinematics()
        self.wrist_kin = Kinematics(tip=CAM_TIP)  # the wrist camera's optical frame
        self.tol, self.rel = Tolerances(), Relations()
        self._frames = 0
