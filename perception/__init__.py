"""Perception: from camera frames to measured, described objects with persistent ids.

Current pipeline (used by the agent; see docs/architecture/EXECUTION_FLOW.md section 4):
    detector.py      YoloDetector: class-agnostic instance masks (YOLO11s-seg)
    geometry.py      measure(): mask + depth -> 3-D geometry (robot / fixture / floor removed); RobotVolume
    somark.py        numbered mark images and isolated crops shown to Gemma
    world_model.py   WorldModel: persistent object ids, association across frames and cameras, fusion rule
    viewpoints.py    ViewpointPlanner: collision-checked wrist-camera viewpoints (inspect / explore; also
                     used by the scenario design checks)

Shared with older code (kept here because frozen or protected files import them):
    grounding.py           footprint / grasp-pose geometry (used by geometry.py, the frozen known-pose
                           experiments and the legacy node)
    classical_detector.py  depth + colour detector (frozen experiments/known_pose_suite.py, scripts)
    world_memory.py        legacy long-term memory (ros_nodes/vlm_agent_node.py)

Gemma's visual interpretation (describe / resolve / confirm prompts) is in agent/vlm_roles/; the tools that run
this pipeline for the agent are in agent/tools/.
"""
