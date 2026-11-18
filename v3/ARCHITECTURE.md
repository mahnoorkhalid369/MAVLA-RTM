# MAVLA-RTM v3: architecture

v3 is a small rewrite of the agent layer. Gemma decides what to do; code only measures and moves.
v1 and v2 are untouched. v3 imports v2's tested motion and camera code instead of copying it.

## Roles

| Part | Does | Decided by |
|---|---|---|
| Master | Reads the request and the table, writes the plan, splits a multi-object request into one object at a time, decides after every report, asks the user when unsure | Gemma |
| Perception worker | Names the objects on the table and answers visual questions | Gemma |
| Manipulation worker | Chooses which object to pick and where to put it | Gemma |
| Verification worker | Looks at the table afterwards and answers yes or no | Gemma |
| Robot | Finds object outlines, measures position from depth, plans and executes a collision-free motion, reads finger and joint sensors | Code |

## Flow of one instruction

1. `look`: YOLO outlines each object, depth gives its position and size, Gemma names it from a zoomed crop.
   This happens at the table camera only; the wrist camera is not used in v3.
2. Master plans: one worker step per line. Several objects become several pick-and-place steps.
3. Each step goes to its worker. The manipulation worker answers with object numbers; the robot does the motion.
4. After a successful step the next planned step runs. Every failure goes back to the master with the
   report; it re-plans, asks the user, or stops. The arm returns to HOME after every action.
5. When the master says done, the verification worker checks the table against the user's own words.

## What code is allowed to decide

Only physical facts:
- Is a motion collision-free and within joint limits (v2 planner and safety layer).
- Did the fingers stop on something (finger feedback).
- Did the arm reach HOME (joint feedback).
- Budgets: at most 12 steps and 8 master decisions per instruction.

Everything else is Gemma's decision. There are no keyword rules for "on", "in", "behind" or "next to",
no name matching, no identity model and no failure lookup table. The relation the user asked for is
passed to Gemma in words and checked by Gemma in words.

## Files

| File | Contents |
|---|---|
| `prompts.py` | Every prompt (master and workers, taken from v2 and shortened) |
| `agents.py` | Master and the three workers: a prompt plus the tools each may use, and the loop |
| `robot.py` | What agents can call: `look`, `transfer`, `pick`, `home`, `holding`; wraps v2's planner, executor and segmentation |
| `run.py` | Start the simulator and models, read instructions, save logs |
| `launch.sh` | The command that starts it |

## Reused from v2 (imported, not copied)

Gemma loader, YOLO segmentation, depth geometry, motion planner, collision checks, ROS executor,
simulator launch and the safety layer.

## Left out on purpose

Wrist re-identification (SigLIP and the side view), the put/place keyword rules, the failure-to-advice
table, the container keyword rule and the dashboard.

## Known limits

- Top-down grasps of upright objects only.
- Names come from a 4-bit Gemma and can be wrong for look-alike objects; the master is told to ask.
- Gemma's yes/no on a picture is not proof of a physical state; holding and HOME are read from sensors.
