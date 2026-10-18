"""Agent tools: the only way the agent touches the world. Deterministic except the Gemma roles. No ground truth,
no object knowledge.

The supervisor and the capabilities use one object, `Tools`, built from these parts (one file each):

    held.py          HeldObjectState     session / held-object state; re-verified from finger feedback
    perception.py    PerceptionTools     perceive, describe, update_world, is_fresh   (+ Observation)
    memory.py        DescriptionMemory   Gemma descriptions reused within a session
    resolution.py    ResolutionTools     resolve, lexical check, compare, confirm, relation_holds
    manipulation.py  ManipulationTools   plan_pick / plan_place_held / plan_pick_place, park_for_view,
                                         return_home, obstacles   (+ PickPlan)
    execution.py     ExecutionTools      execute: the only path to robot motion (gated by Tools.controls)
    verification.py  VerificationTools   verify_contact, verify_home, verify_placement

    tools = Tools(bridge, detector, roles, executor, log, run_dir)
    obs = tools.perceive("table"); tools.describe(obs); tools.update_world(obs)

State owned by Tools (lives as long as the session, agent/session.py): the world model (`world`), the held
object (held.py), the description memory (`memory`), user clarifications (`resolved_refs`), the last handled
object (`last_referent`) and the motion counter used for observation freshness (`motion_count`).
"""

from __future__ import annotations

from agent.controls import ExecutionControls
from agent.tools.execution import ExecutionTools
from agent.tools.held import HeldObjectState
from agent.tools.manipulation import ManipulationTools, PickPlan
from agent.tools.memory import DescriptionMemory, mean_rgb
from agent.tools.perception import Observation, PerceptionTools
from agent.tools.resolution import ResolutionTools, unit
from agent.tools.verification import VerificationTools
from config.agent import Relations, Tolerances
from control.kinematics import Kinematics
from perception.viewpoints import CAM_TIP
from perception.world_model import WorldModel

__all__ = ["Tools", "Observation", "PickPlan", "DescriptionMemory", "mean_rgb", "unit"]


class Tools(HeldObjectState, PerceptionTools, ResolutionTools, ManipulationTools, ExecutionTools, VerificationTools):
    # what the tools can sense (perceive(camera) takes "table" or "wrist"); recorded in the task ledger
    SENSORS = ("table camera (RGB-D, fixed)", "wrist camera (RGB-D, on the arm)", "joint states", "gripper finger position")

    def __init__(self, bridge, detector, roles, executor, log, run_dir, live=None, memory=None, controls=None) -> None:
        self.bridge, self.detector, self.roles, self.ex, self.log, self.run_dir = bridge, detector, roles, executor, log, run_dir
        self.live = live  # agent.live.LiveChannel or None (viewer is optional)
        self.memory = memory  # DescriptionMemory shared by the instructions of one session, or None
        self.motion_count = 0  # every executed robot / gripper command increments it (observation freshness)
        # held object (persists across instructions of a session): Held body, world id, measured geometry at the
        # grasp, finger close target (for retention checks)
        self.held, self.held_object, self.held_geom, self.held_close_target = None, None, None, None
        self.resolved_refs: dict[str, int] = {}  # reference text -> world id chosen by the user (clarifications)
        self.last_referent: int | None = None  # world id of the object the last instruction acted on
        self.view_cleared_at = -1  # motion_count right after the last successful park_for_view
        self.world = WorldModel()
        self.kin = Kinematics()
        self.wrist_kin = Kinematics(tip=CAM_TIP)  # the wrist camera's optical frame
        self.tol, self.rel = Tolerances(), Relations()
        self.held_box = None  # (pose in tool0, size) while holding, from measured geometry
        self._frames = 0
        self.controls = controls or ExecutionControls()  # motion real / dry_run / blocked (agent/controls.py)
