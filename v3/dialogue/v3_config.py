"""Settings of v3/dialogue.

The arm speeds are this version's own copy. v2 keeps its values (30 deg/s, 0.08 m/s, 0.025 m/s) in control/robot.py and
control/skills.py; arm.py hands the values below to v2's planner at start-up, so no v2 file is changed.
The robot's own limits are 122 to 166 deg/s for the three large joints; v2's safety check rejects any faster motion.
"""
import math

SPEED_UP = 1.5  # times v2's arm speeds
JOINT_SPEED = math.radians(30.0) * SPEED_UP  # rad/s on joint moves: from HOME to the object, the carry, back to HOME
LINE_SPEED = 0.08 * SPEED_UP  # m/s at the peak of straight lines: down to the object, the lift, the retreat
LOWER_SPEED = 0.025 * SPEED_UP  # m/s for the final lowering before the release

# The grip: the fingers step in until they stall on the object, then close this much past the gap they felt.
# The finger drive is soft; 16 mm past contact is what holds a 0.2 kg object (8 mm let the fruit cup slip, measured).
GRIP_STEP_M = 0.004  # m the fingers close per step while feeling for the object
GRIP_SQUEEZE_M = 0.016  # m closed further once the object is felt
GRIP_OPEN_M = 0.082  # m: the widest gap the first step starts from (the 2F-85 opens to 85 mm)
