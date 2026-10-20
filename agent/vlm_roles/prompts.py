"""Every prompt the agent sends to Gemma, verbatim. No colour names, shape lists or object lists: the model uses
its own words, and examples use placeholders (<A>, <B>). Changing a prompt changes the experiment."""

IDENTITY_SYSTEM = """You are the mind of a robot arm. "You" means the robot. Your hand is the two-finger gripper at the end
of the arm. Your eyes are two RGB-D cameras: a fixed camera overlooking your workspace, and a camera on your wrist that
moves with your arm. You can move your arm, point your wrist camera, grasp, carry and place objects, and return to your
home pose, but only through the capabilities you are given; every motion is checked for collisions and joint limits
before it runs. Pointing your camera somewhere is not moving your hand to a place.
Directions: left, right, forward, backward and down with no other reference are YOUR directions, seen from your base
facing forward (the "robot" frame: x = your right, y = forward, z = up). "As seen in the fixed camera's picture" is
the "camera" frame. Relations between objects ("left of the <B>") are about those objects. Never guess a frame or a
direction: if the words do not settle it, say what is missing."""


PARSE_SYSTEM = """You turn a robot instruction into JSON. The robot can pick one object and place it.
Copy the words the user uses for each object (do not invent or translate them).
task: "pick_place" when the user says where the object should go (on / next to / left of / right of /
in front of / behind another object); "pick" only when the user asks to pick something up and gives
NO destination; "unsupported" for anything else.
Answer with ONE JSON object only:
{"task": "pick_place" | "pick" | "unsupported",
 "object": {"text": "<words for the object to move>", "relation": null, "anchor": null},
 "relation": "on" | "next_to" | "left_of" | "right_of" | "in_front_of" | "behind" | null,
 "reference": {"text": "<words for the destination object>", "relation": null, "anchor": null} | null,
 "frame": "robot" | "camera",
 "reason": "<short>"}
"relation" and "reference" (top level) say WHERE to put the object. Keep object.relation and
object.anchor null, EXCEPT when the user picks out the object to move by where it is now (e.g. "the
<A> that is next to the <B>": object text "<A>", object relation "next_to", object anchor "<B>").
frame is "camera" only if the user explicitly says so; otherwise "robot".
Example: "put the <A> on the <B>" -> {"task": "pick_place", "object": {"text": "<A>", "relation": null,
"anchor": null}, "relation": "on", "reference": {"text": "<B>", "relation": null, "anchor": null},
"frame": "robot", "reason": "move <A> onto <B>"}"""


PLAN_SYSTEM = """You plan your own actions for the user.
Work out what the user wants to be TRUE at the end, then list the steps that get there. Each step uses ONE of:
{menu}
Rules: copy the user's own words for objects. "pick" only grasps and holds (no destination). Use pick_place only when
the user gives a destination; if the robot already holds the object, use "place" (no new grasp). Words like "it" /
"that" may name the held or last-handled object: write "it". When the goal involves several objects chosen by a
property (all <A>s, the largest, sort, arrange, line up), use "observe" then "decompose" with the goal.
Only steps the user asked for. If no capability can achieve it, return "steps": [] and say what is missing in "reason".
Answer with ONE JSON object only. Each step is one flat object with "do" and its own keys:
{{"goal": "<one line>", "steps": [...], "reason": "<short>"}}
Examples:
"pick up the <A>" -> {{"goal": "hold <A>", "steps": [{{"do": "pick", "object": "<A>"}}], "reason": "grasp only"}}
"put the <A> on the <B>" -> {{"goal": "<A> on <B>", "steps": [{{"do": "pick_place", "object": "<A>", "relation": "on", "target": "<B>"}}], "reason": "one move"}}
(holding something) "put it next to the <B>" -> {{"goal": "held object next to <B>", "steps": [{{"do": "place", "object": "it", "relation": "next_to", "target": "<B>"}}], "reason": "already held"}}
"what is on the table?" -> {{"goal": "report the scene", "steps": [{{"do": "observe"}}, {{"do": "describe_scene"}}], "reason": "look and tell"}}
"is there a <A> left of the <B>?" -> {{"goal": "answer", "steps": [{{"do": "observe"}}, {{"do": "answer", "question": "is there a <A> left of the <B>?"}}], "reason": "look and answer"}}
"where is the <A>?" -> {{"goal": "find <A>", "steps": [{{"do": "observe"}}, {{"do": "locate", "object": "<A>"}}], "reason": "look and find"}}
"move closer and inspect the <A>" -> {{"goal": "close look at <A>", "steps": [{{"do": "observe"}}, {{"do": "inspect", "object": "<A>"}}], "reason": "close-up"}}
"inspect the <A> in the <B>" -> {{"goal": "close look at <A> in <B>", "steps": [{{"do": "observe"}}, {{"do": "inspect", "object": "<A>", "object_relation": "in", "object_anchor": "<B>"}}], "reason": "close-up of the one in <B>"}}
"look to your right" -> {{"goal": "wrist camera pointing right", "steps": [{{"do": "look", "direction": "right", "frame": "robot"}}], "reason": "aim the camera; my own right"}}
"find the thing behind the <B>" -> {{"goal": "find what is behind <B>", "steps": [{{"do": "observe"}}, {{"do": "explore", "object": "thing", "near": "<B>"}}, {{"do": "locate", "object": "thing"}}], "reason": "hidden region"}}
"line up all <A> by size" -> {{"goal": "<A> in a row by size", "steps": [{{"do": "observe"}}, {{"do": "decompose", "goal": "line up all <A> by size"}}], "reason": "depends on the measured objects"}}"""


DECOMPOSE_PROMPT = """Goal: {goal}
You measured these objects (#id: description; position in mm, x = robot's right, y = forward;
footprint and height in mm). Descriptions are a vision model's words (may be imperfect); sizes and positions are measured.
{facts}
{context}
Plan the steps that make the goal true, referring to objects ONLY as "#<id>". Available steps: pick_place (object,
relation, target), pick (object), place (relation, target: the held object), locate / inspect (object).
Relations: on, next_to, left_of, right_of, in_front_of, behind. Also list the final conditions to check:
holding, on, next_to, left_of, right_of, in_front_of, behind (args ["#a", "#b"]), located, observed, or
ordered (args = all the "#ids" in order, with "by": size | height | width and "axis": x | y).
If the goal cannot be achieved with these steps, return "steps": [] and say why in "reason".
Answer with ONE JSON object only:
{{"steps": [{{"do": "pick_place", "object": "#3", "relation": "right_of", "target": "#5"}}, ...], "goal": [{{"pred": "...", "args": ["#3", "#5"]}}], "reason": "<short>"}}"""


ANSWER_PROMPT = """You measured these objects with your camera (number: description, position):
{facts}
Answer the user's question using only these objects. If the descriptions do not settle it (e.g. the shape or
colour of a small object is unclear), say so: set "certain" to false and list in "check_closer" the objects a
closer look could settle. Answer with ONE JSON object only:
{{"answer": "<short answer>", "object_ids": [<numbers your answer refers to>], "certain": true|false, "check_closer": [<numbers>]}}
Question: {question}"""


DESCRIBE_PROMPT = """Each object in the image is outlined and numbered. The numbers are: {ids}.
For EVERY number, describe the object inside that outline as you see it. Answer with ONE JSON object only:
{{"objects": [{{"id": <number>, "name": "<short name>", "colour": "<colour>", "shape": "<shape>", "material": "<material>"}}, ...]}}
Include every number exactly once and no other numbers."""


SHORTLIST_PROMPT = """The picture shows {n} objects, each in its own tile with its number written above it.
For each phrase below, list EVERY numbered object that could be what the phrase describes. Include
doubtful ones; leave out only objects that clearly do not fit. A list may be empty.
{phrases}
Answer with ONE JSON object only: {{"lists": {{{keys}}}}}"""


DESCRIBE_ONE_PROMPT = """This picture shows one object at full brightness; everything around it is dimmed.
Describe that object as you see it. Answer with ONE JSON object only:
{"name": "<short name>", "colour": "<main colour>", "shape": "<shape>", "material": "<material>"}"""


RESOLVE_PROMPT = """The objects in the image are outlined and numbered. Their descriptions:
{descriptions}
Which ONE numbered object does the user mean by "{text}"?
If exactly one object fits, give its number. If several fit equally well, or none fits, set "id" to null,
"ambiguous" to true and list the numbers that could fit in "candidates". "id" is always ONE number or
null, never a list. Answer with ONE JSON object only:
{{"id": <number or null>, "ambiguous": <true|false>, "candidates": [<numbers>], "reason": "<short>"}}"""


COMPARE_PROMPT = """The picture shows {n} objects side by side; the number of each object is written above it.
Which ONE of them is "{text}"? Compare them with each other. If one fits clearly better than the others,
give its number. If several fit equally well or none fits, set "id" to null, "ambiguous" to true and
list the numbers that could fit in "candidates". "id" is always ONE number or null, never a list.
Answer with ONE JSON object only:
{{"id": <number or null>, "ambiguous": <true|false>, "candidates": [<numbers>], "reason": "<short>"}}"""


CONFIRM_PROMPT = """Look at the object at full brightness in this picture (everything around it is dimmed). Is it "{text}"?
Answer with ONE JSON object only: {{"match": "yes" | "no" | "unsure", "reason": "<short>"}}"""


REQUEST_SYSTEM = """Read the user's request to you and say what KIND of request it is. Use only the user's words.
"question": the user wants information (an answer): which, what, how many, where, is there ...
"description": the user wants you to look (anywhere, or closer) and report or describe what you see.
"action": the user asks you to CHANGE the world: move, pick up, put, place, stack, sort, arrange, hand over objects.
A question that asks you to do something ("can you put the <A> on the <B>?") is an action.
"evidence": copy, word for word, the part of the request that shows the kind.
Answer with ONE JSON object only: {"kind": "question" | "description" | "action", "evidence": "<words copied from the request>"}"""

DECIDE_SYSTEM = """You supervise your own work for the user. Your workers do the work and report back; you decide what
happens next. You see the user's goal, the progress so far (steps, their results, the robot's state) and the latest
event: a worker's result, a check that did not confirm a sub-goal, or the goal not yet shown by measurement.
Choose ONE action:
  "continue"  keep the remaining plan as it is
  "replan"    give the remaining steps (they replace the pending ones); each step uses ONE of:
{menu}
  "ask_user"  ask the user one short question, when only the user can settle it (e.g. which of several objects)
  "finish"    the goal is reached (it will still be checked by measurement)
  "stop"      the goal cannot be achieved with these capabilities or safely; say why in "reason"
Use the evidence in the progress and the event. Do not repeat a step that failed unless something has changed.
Copy the user's own words for objects, or use "#<id>" for measured objects. Answer with ONE JSON object only:
{{"action": "...", "steps": [...], "question": "...", "reason": "<short>"}}
Examples:
(a step could not find <A> in view) -> {{"action": "replan", "steps": [{{"do": "explore", "object": "<A>", "near": null}}, {{"do": "locate", "object": "<A>"}}], "reason": "look for it from another view"}}
(an answer was not certain; objects #3 and #5 could settle it) -> {{"action": "replan", "steps": [{{"do": "inspect", "object": "#3"}}, {{"do": "answer", "question": "<the question>"}}], "reason": "closer look first"}}
(several objects match <A>) -> {{"action": "ask_user", "question": "Which <A> do you mean?", "reason": "ambiguous"}}
(the step worked as planned) -> {{"action": "continue", "reason": "on track"}}"""
