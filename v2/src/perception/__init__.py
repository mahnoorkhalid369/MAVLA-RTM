"""Perception: from camera frames to measured, described objects with persistent ids.

    detector.py      YoloDetector: class-agnostic instance masks (YOLO11s-seg)
    geometry.py      measure(): mask + depth -> 3-D geometry (robot / fixture / floor removed); RobotVolume
    grounding.py     footprint / grasp-pose geometry (used by geometry.py)
    somark.py        numbered mark images and isolated crops shown to Gemma
    world_model.py   WorldModel: persistent object ids, association across frames and cameras
    viewpoints.py    collision-checked wrist-camera viewpoints (wrist camera pose; scenario design checks)

Gemma's words for the objects come from agent/gemma.py (describe_grid); the tools that run this pipeline for the
agent are in agent/tools/.
"""
