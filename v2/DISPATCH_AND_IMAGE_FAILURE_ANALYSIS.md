# Dispatch, images, and prompt failures in the 07:47 snack run

Reviewed 2026-11-11. This describes the captured run before the pipeline refactor. See [PIPELINE.md](PIPELINE.md)
for implemented controls, validation and the remaining perception/recovery limitations.

Source run: [`out/demo/2026-11-11/live_074531/01/events.jsonl`](../out/demo/2026-11-11/live_074531/01/events.jsonl). Event line numbers refer to the inspected snapshot. The simulator log identifies the scene as **workcell, hard, seed 42**, not `snack_kitchen`.

## Confirmed failure chain

1. Perception misidentifies the red mug on the tray as the requested fruit cup and food boxes as orange juice.
2. The master emits malformed multi-line output containing an echoed worker report and an additional manipulation plan. Its parser accepts only the first line, so the intended pick is omitted and perception repeats.
3. The master later requests placing the bottle despite no pick having executed.
4. Grounding emits a pick line followed by a place line. Its parser keeps only the pick, and the skill successfully picks the bottle without placing it.
5. The master claims placement in its memory despite the worker reporting that the bottle is held.
6. Later place requests target remembered tray ID `#16`; refreshed images mark that tray with other IDs. The skill correctly refuses an unmeasured reference.
7. All candidate view-recovery paths fail, and the master repeats placement without resolving the reference or recovery failure.

## 1. Master output is silently truncated

At **07:51:40**, event line 29, the stored master answer begins with:

```text
plan: perception: describe the small, white water bottle, and confirm if it is safe to pick it up. -> ok: ...
plan: manipulation: pick up the small, white water bottle.
memory: ...
```

The memory also contains a residual `<channel|>` marker and more plan text.

[`master.decide()`](src/agent/master.py), lines 123–130, takes only the first nonempty line and parses only the text after its first colon. It neither rejects the unexpected extra lines nor consumes them as additional steps. The loop therefore dispatches perception with a copied report in its instruction. This reproduces exactly in an offline replay: one step retained, worker `perception`, message containing `-> ok:`; the manipulation line is dropped.

At **07:54:06**, event line 40, the answer also echoes prompt sections such as “Plan still to do,” “Latest reports,” and “Budget left.” Those sections are silently ignored while `memory_of()` can still extract unsupported completion claims from later in the output.

The worker registry itself dispatches the parsed worker correctly. The incorrect action selection happens before dispatch, when malformed model output is partially accepted.

Required correction: validate the complete response. Either accept a precisely defined multi-line plan or reject it and request one repair. Reject echoed reports/templates and leaked control tokens; do not execute a valid-looking prefix of an invalid response. Preserve the natural-language messages if desired; JSON is not required for this correction.

## 2. Manipulation drops an explicitly generated placement

At **07:55:07**, event line 46, grounding answers:

```text
pick #14
place on #16
names: #14 = small white bottle; #16 = serving tray
```

The advertised grammar expects a compound action on one line: `pick #14 place on #16`. [`Manipulation.run()`](src/agent/workers.py), lines 162–187, reads only the first line and treats this malformed multi-line response as a valid pick. `_act()` therefore invokes mode `pick` with no destination.

An offline replay with the existing fake arm produced:

```text
skill calls: [('pick', 14, None, None, None)]
held afterward: 14
```

Thus this is not merely the master forgetting to issue a place: the grounding answer contains a place, and the software discards it. The worker's subsequent “picked up” report is consistent with the action that actually ran.

Required correction: parse the whole grounding response. Either normalize an unambiguous pick-then-place pair or reject it and repair; never silently downgrade it to pick-only. Validate the grounded action against the requested operation and current gripper state.

## 3. Image and textual referents disagree after tray identity changes

The saved numbered table views show the serving tray labelled:

| Saved image | Tray's displayed world ID |
|---|---|
| `scene_1_ids.jpg` | `#3` |
| `scene_2_ids.jpg` | `#3` |
| `scene_3_ids.jpg` | `#16` |
| `scene_4_ids.jpg` | `#17` |
| `scene_5_ids.jpg` | `#21` |

The tray is still visibly present. The model repeatedly answers `place on #16` after that ID no longer appears in the current image. [`Scene.text(numbers=True)`](src/agent/workers.py), lines 86–102, includes persistent on-surface/occluded world records, not just IDs measured in the current observation. `_act()` checks world membership rather than current visibility. This allows a remembered destination to reach the skill, which later rejects it in `_localizing()`.

Geometry also explains why identity association is unstable. The measured tray centre changes from approximately `[-0.462, 0.593]` m in `table_001` to `[-0.504, 0.515]` in `table_003`; its measured footprint also changes substantially. Later partial measurements have centres around `[-0.390, 0.619]` and smaller footprints. Those changes exceed the world model's 30 mm position / 20 mm footprint association gates. This is consistent with unstable segmentation/partial tray geometry generating new world IDs.

Required correction: distinguish current measured IDs from remembered records in the grounding context; require a chosen source/reference to be supported by the actual supplied observation, with a separate exception for the held object. Refresh and re-ground when the observation expires. Improve tray association without broadly loosening gates enough to merge unrelated nearby objects.

## 4. What the images do and do not establish

Current code passes:

- Plain table RGB to master planning/review and perception.
- Numbered table RGB to manipulation grounding.
- A red-box table RGB image to collision consultation.
- No wrist RGB to the master or perception worker in this path.

The images saved for the run show the correct table/workcell scene and the bottle physically held after the pick. There is no evidence here of a table/wrist camera swap. The important confirmed image-related defect is the mismatch between the current image's marks and the remembered object list supplied alongside it.

Exact per-call input tensors, complete prompts, image hashes, and source frame IDs are not logged at the model boundary. Consequently the event log cannot prove every individual call received exactly the intended frame or exclude all preprocessing defects. Add role, camera, frame ID, image variant, image dimensions, current ID-to-mark mapping, and selected references to each model-input event. Keep raw generation separate from the validated control response.

## 5. Prompts and memory amplify the defects

- The first perception response calls the red mug a fruit cup. The actual fruit cup is the small pale/yellow cup on the left; the orange drink is also on the left, distinct from the cracker and dessert boxes.
- The master repeatedly asks perception to certify “safe to pick up.” Perception is a visual language role; collision/reach checks belong to manipulation's measured tools. The response “safe” provides no executed motion or collision evidence.
- The master says the bottle was placed before any placement ran, then keeps that claim after the worker reports a pick. Current gripper state is supplied, but narrative memory contradicts it and is not validated.
- A planned perception step asks about juice “now in the gripper,” although that worker only sees table RGB. The master should know that limitation explicitly.
- Residual `<channel|>` markers reach memory. The model adapter/parser boundary needs validation; the log alone does not determine whether the marker originated in malformed generation, response extraction, or both.

Required correction: make executed/verified state authoritative, distinguish intention from completion, state each worker's sensor access explicitly, and route physical-feasibility questions to manipulation tools. Repeated uncertainty about object identity should lead to better evidence or clarification rather than a confident visual assertion.

## 6. Recovery failure is real, but is downstream

At event lines 113, 141, 175, 202, and 234, `park_for_view` records all nine candidates with `path_free: false`. Earlier pick planning also hits `j1_link` against unexplained depth columns and removes two after collision consultation. This resembles the self-geometry obstacle problem in the earlier run, but these parking events do not record their contacts, so the exact blocker is not proven by this log alone.

The outer master keeps reissuing placement; each new skill resets its local retries. The generic error “object #16 is not measured” is a **destination/reference** failure in mode `place`, not evidence that the held water bottle disappeared. Failure reports should identify source versus destination and require changed evidence before the same attempt repeats.

## Fix order

1. Eliminate silent truncation in master and grounding response parsing, including residual channel-token acceptance.
2. Keep reports typed for loop routing and separate verified progress from model-written memory.
3. Bind executable references to the supplied observation; repair current-image versus remembered-ID disagreements.
4. Clarify worker sensor/capability boundaries and handle visual ambiguity explicitly.
5. Log precise model inputs and parking contacts, then fix identity association and view recovery with that evidence.

No final `COMPLETED`, `STOPPED`, or `FAILED` result is present in the inspected event snapshot; it ends in DECIDE after another failed placement.
