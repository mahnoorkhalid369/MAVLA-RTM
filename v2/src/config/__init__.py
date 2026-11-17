"""Configuration of the project: fixed physical, safety and model settings. No object knowledge lives here:
objects are discovered (masks), measured (depth) and described (Gemma) at runtime.

    config/perception.py   depth-measurement thresholds, fixtures excluded from perception (camera stand,
                           microphone, robot base column)
    config/agent.py        placement tolerances, recovery budgets, placement-relation search
    config/models.py       YOLO-seg weights and detector settings

Kept where they are (frozen or shared, documented here so there is one place to look):
    control/robot.py       robot joints, HOME pose, speeds, table-camera pose (frozen manipulation core)
    vlm/config.py          Gemma 4 E2B model settings (shared with the legacy stack)
    agent/controls.py      execution controls of one agent process (real / dry-run / blocked motion,
                           disabled capabilities, replanning limit)
"""
