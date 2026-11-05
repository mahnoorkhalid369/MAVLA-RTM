# Robustness fixes, 2026-10-25 / 26

Live runs on 2026-10-25 and 2026-10-26 (work cell scene, Gemma 4 E2B, then E4B) failed in repeatable ways: the agent
looped through the same steps until its budget ran out, refused correct answers from Gemma, could not put objects *into*
a tray, and ended tasks still holding a block without saying so. This document lists every issue seen, its cause, how it
was fixed, and the test that guards it.

**Constraints kept throughout**

- **Architecture unchanged.** Gemma (the supervisor's language model) remains the master that plans and decides; workers
  stay slaves that run one operation and report back; no worker calls another.
- **No matching or lookup logic added.** Which object the user means is decided by Gemma alone. The fixes only *remove*
  places where the pipeline overruled Gemma; no colour filters, keyword rules or object lists were added.
- **Gemma stays 4-bit** (`vlm/config.py quantization = "4bit"`); the model is now E4B (see the end).
- **Every instruction ends:** completed, asked, or stopped, within fixed limits.

Tests: `tests/integration/test_robustness.py` (16 tests replaying the live failures with a scripted, misbehaving Gemma),
`tests/unit/test_container_placement.py` (5), plus updated existing tests. Full suite: 601 passed.

---

## 1. Loops

### 1.1 `decompose` produced another `decompose`, endlessly
- **Seen:** "put all red blocks in the red tray": each decomposition returned only another `decompose` step
  (plan steps 1 → 4 → 5 → 7); the pick was never reached.
- **Cause:** Gemma was given the full operation menu while decomposing, **including `decompose` itself**, and nothing
  stopped a decomposition from containing it.
- **Fix:**
  - The menu Gemma decomposes with no longer offers `decompose` (`agent/capabilities/planning.py`, `run_decompose`).
    This removes the cause; the two checks below stay as backstops.
  - Gemma's answer is checked: a decomposition containing `decompose` is invalid, and Gemma is told why on its one retry
    (`agent/vlm_roles/roles.py`, `Roles.decompose`).
  - If one still gets through, the supervisor drops it before inserting the steps and logs `plan_rejected`
    (`agent/supervisor.py`, `_reviewing`).
- **Tests:** `test_a_decomposition_cannot_contain_decompose`,
  `test_a_nested_decompose_that_gets_through_is_dropped_and_free_text_conditions_are_not_added`.

### 1.2 The same step re-run into an unchanged world
- **Seen:** "place the object in red tray": Gemma replanned `observe` three times in a row; nothing moved, so nothing
  could change. Repeated `decompose` runs looped the same way.
- **Cause:** "new evidence" counted every new camera frame, so another photo of a static scene looked like progress.
  Steps that *completed* without effect (observe, decompose) were never recorded as attempts.
- **Fix** (`agent/supervisor.py`):
  - **New evidence** is now motion, a change of the held object, an answer from the user, a view from the wrist camera,
    or a completed sensing operation (explore / inspect / look). A repeated table-camera frame is not (`_signature`).
  - Every step without a measured effect is recorded, including ones that completed (`_remember`).
  - A planned step identical to one that already **completed** with no effect, with nothing changed since, is skipped,
    not run (`_dispatching`, `_repeats(completed_only=True)`). A step that **failed** may still be retried; the existing
    replan rule refuses an identical retry with nothing new.
- **Tests:** `test_a_gemma_that_always_observes_again_cannot_loop`, updated
  `test_replan_that_drops_the_objective_cannot_complete_the_goal`.

### 1.3 Changes of course with no effect
- **Seen:** several runs spent all three changes of course on re-planning that changed nothing, then failed.
- **Fix:** if Gemma changes course a second time and nothing has changed since its previous accepted change (same world
  state, same unmet criteria), the supervisor does not run it. Instead it asks the user a concrete question ("I am not
  making progress: … What should I do?").
  - The answer goes back to Gemma as new evidence.
  - "Stop" from the user cancels the task.
  - No answer ends the task as FAILED, listing the unmet goal.
  - (`agent/supervisor.py`, `_deciding`, `_ask_user_stuck`; `_progress_mark` defines "nothing changed".)
- The same question is asked when Gemma's replans are **refused** twice in a row as repeats. Live on 2026-10-26 (after
  the first fixes), four refused `decompose` replans used up the budget and the task ended without asking anyone.
- **Tests:** `test_a_gemma_that_keeps_replanning_without_progress_is_stopped_by_a_question`,
  `test_a_gemma_that_keeps_repeating_a_failed_step_is_stopped_by_a_question`, `test_a_decomposition_is_not_offered_decompose`.

### 1.4 Decomposition planned only "observe", or was used for one object
- **Seen (second live check, 2026-10-26):** for "pick the red block and place it in the red tray", Gemma chose
  `decompose`, and the decomposition returned just `observe` ("to confirm object locations"), although the measured
  objects were listed in its prompt.
- **Fix (prompt and menu wording only; no rule):**
  - The decomposition prompt says the listed objects are already measured, so the plan should be the actions themselves,
    without an observe step.
  - The menu says `decompose` is for goals that move **several** objects; one object to one place is a single
    `pick_place`, or `place` when it is already held.
- The progress guard (1.3) ended both live tasks within about 3 minutes with a question, instead of looping.

### 1.5 Hard ceilings
- At most 10 Gemma decisions per instruction (`MAX_CONSULTS`), in addition to the existing 14 worker steps and 3 changes
  of course. Whatever Gemma answers, the instruction ends.

### 1.6 Goal conditions piled up, some impossible to check
- **Seen:** each decomposition added its conditions again (four copies of `holding(#8)`), and free-text conditions such
  as "all red blocks are in the pink/red tray". Nothing can evaluate free text, so the goal stayed "unknown" forever
  and the loop continued.
- **Fix:**
  - A decomposition's conditions are added only if they are measurable predicates and not already in the goal; skipped
    ones are logged (`_new_conditions`).
  - Gemma's decomposition is also checked: its goal conditions must be measurable (`check_predicate`).
  - The **original** goal from the first plan is left as Gemma wrote it: an unmeasurable criterion is still reported
    honestly as "unverifiable", never as done.
- **Test:** the nested-decompose test above checks both.

---

## 2. Gemma's answers rejected for form, not content

### 2.1 JSON with extra text
- **Seen:** "Extra data: line 1 column 121" twice in a row (the task then failed). Gemma had sent a valid object
  followed by more text.
- **Fix:** the parser reads the **first complete JSON value** and ignores code fences, leading text and anything after
  it (`agent/vlm_roles/roles.py`, `_json`).
- **Test:** `test_the_first_json_object_is_read_whatever_follows_it`.

### 2.2 Arguments nested under `"argument"`
- **Seen (2026-10-25):** "decompose needs goal": Gemma wrote `{"argument": {"goal": …}}`; only `"arguments"` and
  `"args"` were unpacked.
- **Fix:** `"argument"` is unpacked too (`agent/vlm_roles/schemas.py`, `PlanStep.flat`).
- **Test:** `test_arguments_flat_or_nested_are_the_same_step`.

### 2.3 "continue" given with new steps when none remained
- **Seen:** "no steps remain: choose replan, finish, ask_user or stop", twice, ending the task.
- **Fix:** with nothing left to continue, `continue` plus new steps is taken as the replan it means (logged as such)
  (`check_decision`, `_deciding`).
- **Test:** `test_continue_with_new_steps_when_none_remain_is_a_replan`.

### 2.4 The user's own words written inside template brackets
- **Seen:** "put the yellow block in the blue tray" was planned as `on(<yellow block>, …)`: Gemma copied the prompts'
  bracket notation around the user's words. It was rejected twice as a template placeholder.
- **Fix:** if the text inside the brackets occurs word for word in the user's instruction, it **is** the user's words and
  the brackets are removed. Anything else in brackets (`<object_id>`, `#<held id>`) is still rejected as an unfilled slot
  (`agent/vlm_roles/roles.py`, `unwrap_user_words`, used by `plan_goal` and `decide`). The words are compared, not
  interpreted.
- **Test:** `test_the_users_own_words_in_template_brackets_are_their_words`.

### 2.5 A plan that contradicts its own goal
- **Seen (seventh live check):** the plan said `place relation=in red tray`, but its success criterion said
  `on(red block, red tray)`. The block was placed in the tray and the camera confirmed it (2.7 mm). The goal check then
  said, correctly, "not on top" (the block rests 36 mm below the rim), and the task ended FAILED.
- **Fix:** a plan whose steps put something `in` a destination while its success criterion uses another relation for
  that destination (or the other way round) goes back to Gemma with the contradiction stated
  (`agent/vlm_roles/roles.py`, `check_relations_agree`). This is structural: the plan is compared with itself.
- **Eighth live check:** told only to "make them agree", Gemma changed the **step** to `on` instead of the criterion to
  `in`. The block was released at rim height, dropped into the tray, and the placement check failed by 50 mm. The retry
  message first quoted the user's request; in the ninth live check Gemma still changed the step. It now also names the
  concrete correction: "Keep the steps and write the criterion as in(red block, red tray), unless the steps are what is
  wrong". The steps come from the user's words; the criterion was the boilerplate. Gemma still decides.
- **Test:** `test_a_plan_that_places_in_must_not_state_on_as_its_success`.

### 2.6 Bare numbers as object references
- **Seen:** Gemma wrote `pick_place '4'` while holding #4. The held-object check did not recognise `4`, so the step was
  refused as unsafe (another object), not as "already held".
- **Fix:** a bare number is a measured object's id, the same as `#4`, in step arguments and in goal conditions
  (`agent/capabilities/specs.py`, `_id`; `agent/vlm_roles/schemas.py`, `Condition`).
- **Test:** `test_a_bare_number_is_a_measured_object_id`.

---

## 3. Holding an object

### 3.1 `pick_place` used instead of `place` for a held object
- **Seen:** five runs with E2B: after a pick, Gemma chose `pick_place` (which starts with a grasp) again and again.
- **Cause:** the operation menu did not say clearly which operation puts down a held object. The refusal said why
  `pick_place` failed, but not what would work.
- **Fix:**
  - The menu now says `place` is the operation whenever the robot holds something, and that `pick_place` is never used
    after a pick or while holding (`agent/capabilities/specs.py`).
  - The refusal names the operation that works: "… The operation for a held object is place: place relation=in
    target=…" (`agent/capabilities/manipulation.py`, `run_pick_place`).
  - The supervisor still does not swap the operation itself: the decision stays Gemma's.
- **Tests:** `test_pick_place_of_the_held_object_names_the_operation_that_works`. With E4B, the first live plan for a
  held block was `place`.

### 3.2 `#<held>` copied from the menu
- **Seen:** "step 2: '#<held>' is a template placeholder", twice, so the instruction could not be planned.
- **Cause:** the menu's example for `place` contained `"object": "#<held id>"`, and Gemma copied it.
- **Fix:** `place` takes no object argument (it puts down whatever is held); the slot is gone from the menu.
- **Test:** `test_place_needs_no_object_and_its_menu_has_no_template_slot`.

### 3.3 Tasks ended with a block in the gripper, silently
- **Seen:** four tasks in a row stopped with a block lifted in the gripper; the final message said only "I did not
  report success".
- **Fix:** whenever a task ends without success while something is held, the final message says so and asks where to
  put it: "The robot is still holding #4 (yellow block, number 10 in the camera picture), lifted above the table: tell
  me where to put it down …" (`agent/supervisor.py`, `_finish`).
- **Test:** `test_a_task_that_stops_while_holding_says_so`.

### 3.4 A successful pick reported as failed ("not msgpack serializable")
- **Seen (third live check, 2026-10-26):** the red cube was grasped and lifted, then the step failed with
  `TypeError('Type is not msgpack serializable: numpy.float64')`. The robot was left holding it, and the next
  instruction was refused as unsafe.
- **Cause:** a regression from the tilted-grasp change of 2026-10-25. The grasp-height formula multiplies by
  `cos(tilt)`, which returns a NumPy number even at 0°. That value reached the skill's state, which LangGraph saves
  after every step, and its serializer rejects NumPy numbers.
- **Fix:** the grasp formula returns plain floats (`control/skills.py`, `_grasp_z`), and the skill converts its saved
  summary to plain numbers (`agent/pick_place.py`, `_planning_grasp`).
- **Test:** `test_grasp_numbers_are_plain_floats_for_the_checkpointed_skill_state`, which also packs the values with
  LangGraph's own serializer.

---

## 4. The pipeline overruled Gemma's correct answer

- **Seen (E4B, "place the object in red tray"):** Gemma answered "#13, explicitly labelled 'TRAY 1 (RED)'", which was
  correct. It was rejected eight times as "not one of the drawn ids".
- **Cause:** three steps of the existing object pipeline removed the tray before Gemma's final choice, and then refused
  its answer.
  1. The word split put "tray" under "other words" and kept only "red" as the appearance.
  2. The side-by-side comparison, judging "red" alone, kept the two red cubes.
  3. The final choice accepted only those two numbers.
- **Fix:** only vetoes were removed; no matching was added.
  - The fix that settled it (see the last bullet) was to stop narrowing by part of the words, so the tray stays a
    candidate. A first attempt that let Gemma's final answer be **any** object in the picture was withdrawn after the
    seventh live check: given the whole picture, Gemma picked neighbours of the right object (an orange cube for
    "yellow block"), as measured earlier (4 of 13 correct on the whole picture). Gemma now chooses **among the remaining
    candidates** (`ResolutionTools._resolve_described`).
  - The side-by-side comparison no longer narrows the candidates when the user's words include more than appearance
    (`ResolutionTools.resolve`).
  - The word-split prompt keeps the object's kind with its appearance (`REFERENCE_PROMPT`; prompt fingerprint updated in
    `tests/unit/test_structure.py`).
  - When several candidates remain after the look check, **Gemma makes the final choice** from the whole numbered
    picture, its descriptions and the measured facts. The user is asked only if Gemma itself says it cannot decide. Before
    this, two candidate trays went straight to the user (fifth live check, 2026-10-26)
    (`ResolutionTools.resolve`, "final choice" stage).
- **Not done (by decision):** no colour thresholds or other rules. Whether a pink block counts as "red" stays Gemma's
  judgement.

### 4.1 The held object offered as a destination
- **Seen (fourth live check, 2026-10-26):** the red cube was picked and lifted. Then, for "place it in the red tray", the
  candidates for "red tray" included the red cube **in the gripper** (seen lifted by the camera) as well as the real tray.
  They could not be told apart, so the step asked the user, and the scripted run cancelled.
- **Fix:** a destination is never the object being put down: the held one, or the one a `pick_place` moves
  (`agent/pick_place.py`, `_resolve_one`). The goal check applies the same rule: a relation's destination is never its own
  first object or the held object (`agent/supervisor.py`, `_ground_criteria`). This is a physical fact, not a judgement
  about the words; Gemma still chooses among everything else.
- **Test:** `test_a_destination_is_never_the_held_object`.

### 4.2 Goal objects re-matched on every review
- **Seen:** `on(#4, red tray)` named the tray in words, so every review re-ran the whole object pipeline (about 45 s
  each, the same result every time).
- **Fix:** once found, a goal object is checked by its id. A failed or ambiguous search is retried only when something
  changed (motion, an answer, a new view), not on every new frame (`agent/supervisor.py`, `_ground_criteria`).

### 4.3 Two numbering systems
- Objects have a world id (stable) and a number in each camera picture (renumbered per frame). User-facing messages that
  name a world id now also give its number in the latest camera picture: "object #13 (number 14 in the camera picture)"
  (`agent/pick_place.py`, `_unreachable_reason`; `agent/supervisor.py`, `_finish`).

---

## 5. Placing **into** a container (new)

- **Seen:** "put … in the red tray" was planned as `in_front_of` (beside the tray) or `on` (onto its 40 mm rim). No
  operation could place inside.
- **Fix:**
  - **`in` relation:** available for `place` and `pick_place` (`agent/capabilities/specs.py`, `PLACE_RELATIONS`). The menu
    explains it as "inside a container such as a tray or bin" versus "on: on top of a solid object".
  - **Placement** (`agent/tools/manipulation.py`, `_plan_in`):
    - the object goes to the container's centre, resting on its **measured floor**: the lowest point the camera sees
      inside it, or 5 mm above what the container stands on if the floor isn't visible (`_container_floor`);
    - for planning, the container is modelled as **four walls and a floor slab** inside its measured outline, not a
      solid block, so the gripper can go down inside it (`_container_boxes`);
    - the gripper is turned so the open fingers fit along the container's long side, and every motion is
      collision-checked against the walls;
    - an object that cannot fit is refused, with both sizes in the message.
  - **Prompts:** both the planning prompt (`PLAN_SYSTEM`) and the decomposition prompt (`DECOMPOSE_PROMPT`) list `in`
    among the measurable conditions. Before this, Gemma could only see `on` and wrote `on(red block, red tray)`. The
    prompt fingerprints were updated in `tests/unit/test_structure.py`, since a prompt change changes the experiment.
  - **Goal check:** `in(x, y)` holds when x's centre is within y's measured outline and x rests no higher than y's rim
    (`agent/goals.py`, `_in`, using the existing `relation_holds("in")`).
- **Verified offline:** a held 50 mm block plans into all three work-cell trays, with real kinematics and collision
  checks.
- **Tray spacing (scene fix):** in the sixth live check the red and blue trays were only about 4 cm apart and touched in
  the camera's view. The blue tray's outline took in the red tray's corner, so it was measured 347 mm long instead of
  200 mm, and its collision box blocked lowering into the red tray. The trays now stand about 10 cm apart
  (`scenarios/workcell.py TRAYS`).
- **Tests:** `tests/unit/test_container_placement.py`.
- **Placement check inside a container (tenth and eleventh live checks):** "z -14.5 mm, out of tolerance" although the
  block was in the tray. Cause: the block and the tray share a centre, and the check took the nearest measured object,
  which was the **tray** (its 40 mm rim), not the block. The tray floor estimate (4.4 mm) was right. A first fix (a
  one-sided height tolerance) was based on a wrong diagnosis and was withdrawn. Now, for `in`, the container is not a
  candidate for the placed object. It is recognised by its measured geometry (same rim height, container size), because
  the frame's labels are linked to world ids only after this check (`agent/tools/verification.py`, `verify_placement`). Test:
  `test_a_placement_inside_a_container_is_checked_on_the_object_not_the_container`.
- **"Does not fit" is final:** a held object larger than the container's inside is a measured fact. It now stops the task
  with that answer (`ManipulationTools.DETERMINISTIC`) instead of three identical re-plans. The message names the
  container and its size, so a wrongly identified destination is visible (eleventh live check: "blue tray" was taken for
  a 45 mm cube; that identification was Gemma's).
- **Completion message:** when the goal is shown by measurement after a step that reported a failure, the message says
  so, instead of quoting an earlier step ("holding object #12" after the block was in the tray) (`_verifying_goal`).
- **Pick message:** a completed pick reported "Done (verified by the table camera): ." (an empty placement). It now says
  "Done: holding object #N (finger contact confirmed after the lift)." (`agent/pick_place.py`, `honest_message`).
- **Known limit:** the tray floor is an estimate (the camera's lowest visible point inside the tray); if the walls hide
  the floor, the object is released a little above it and drops the last millimetres.

---

## 6. Knowing what is held, and understanding requests (2026-10-26, evening)

### 6.1 The robot held a block but believed it had lost it
- **Seen:** for "pick the white block", the fingers closed on the block at 30.5°. The finger check (it needs at least
  1.5° short of the close target) read 1.26°, 1.57° and 1.50°: fail, pass, fail. The table camera could not see the block
  in the gripper, so the robot reported "object lost while lifting". The next instruction started as "nothing held",
  planned a pick, and **opened the gripper on the block**.
- **Cause:** for borderline finger readings, only the table camera was asked. From 1.5 m it cannot see a small object in
  the gripper. The wrist camera, which looks straight between the fingers, was never asked.
- **Fix:**
  - **Wrist-camera check (`agent/tools/verification.py`, `wrist_sees_held`):** the wrist camera sits on J5, so the gripper
    axis always projects to the same pixels (centre column, rows ~424–514 at 1280 × 720). Depth there at the fingers means
    something is held; depth reaching the table means empty. Geometry only, no detector, no Gemma. The check runs:
    - for borderline finger readings, before the table camera (`confirm_held`);
    - at the start of every instruction whenever the fingers are not open (`revalidate_held`).
  - **Recovery:** a grasp the lift check could not confirm is recovered when the wrist camera sees the object.
  - **Unrecorded object:** if the fingers hold something no grasp recorded, the session marks it (`unknown_held`). Gemma is
    told, and `pick` / `pick_place` refuse to open the gripper on it.
  - **Borderline grasps:** at the grasp, a borderline finger reading no longer reopens the gripper. The robot lifts, and
    the retention check (fingers, then wrist camera) decides.
- **Tests:** `tests/unit/test_wrist_held.py`.

### 6.2 Answers in words were refused
- **Seen:** to "Several objects could be 'block'. Which one?", the user answered "you already have block in your grasp".
  It was refused as "not a number".
- **Fix:** an answer in words is never refused. The skill's question ends, and the user's words go to the supervisor as
  typed, where Gemma reads them in its next decision (`agent/ask.py`, `parse_answer`; `agent/pick_place.py`, `_cancelled`).
  Numbers still choose; "cancel", "stop" or an empty line cancel.
- **Tests:** `test_words_are_the_users_answer_passed_on_as_typed`,
  `test_an_answer_in_words_ends_the_skill_question_and_reaches_the_supervisor`.

### 6.3 Understanding a request before planning (new step)
- **Requested by the user:** Gemma should understand raw requests and hold a conversation when something is unclear,
  and plan only once it is clear.
- **How (`agent/supervisor.py`, `_understand`; `Roles.clarify`; `CLARIFY_SYSTEM`):**
  - Before any plan, Gemma reads the request with what the robot knows: what is held (including an unrecorded object)
    and the objects measured so far.
  - It answers **clear**, restating the request in the user's own words for the objects, or **ask**, with one question
    in its own words. The user answers in free text.
  - At most 3 questions (`MAX_CLARIFY`), then it plans. "Stop" cancels; no answer plans the request as given.
  - The plan is made from the request, the conversation and the understood request.
  - Permissions still come only from the user's own words (the request and the answers), never from Gemma's restatement.
- **First live check:** Gemma asked "Where is the white block?", which the camera can answer, and restated "the red
  tray" as "object #12 (red cube)". The prompt now forbids where/what-it-looks-like questions and requires the user's own
  words for objects.

### 6.4 A new request in an answer replaces the goal
- **Seen:** answering "place the block in red tray" to "I am not making progress" was bent into the old goal (pick the
  white block), and the robot re-observed until its budget ran out.
- **Fix:** a new decision, `new_goal`. Gemma uses it when the user's answer asks for something else. The current goal is
  dropped, and the new request goes through understanding and planning, within the same task budget.
- **Test:** `test_a_new_request_in_an_answer_becomes_the_goal`.

### 6.5 The tilted grasp is for reach only
- **Seen (smoke run):** top-down grasping collided with a neighbouring object by 0.7 mm, so the block was taken with a
  10° tilt. A tilted hold cannot be placed, so the following "put it in the red tray" was refused.
- **Fix:** the tilted fallback is used only when top-down fails on **reach**. Any other top-down failure (a collision, for
  example) is reported as it is (`ManipulationTools.plan_pick`).
- **Test:** `test_a_neighbour_in_the_way_is_not_dodged_by_tilting`.

## 7. Earlier fixes from the same runs (2026-10-25 / 26)

| Issue | Fix | Where |
|---|---|---|
| Out-of-reach object: Gemma replanned and looked for a "similar" object | A measured-infeasible pick or place stops the task with the reason; no replan, no substitute object | `agent/supervisor.py` `_reviewing` |
| Blocks behind the base could not be picked | Bounded tilted grasp (up to 40°) as the manipulation worker's internal fallback | `control/skills.py` `tilted_pick` |
| Planner could enter Epson's forbidden joint zones | J1/J2/J3 forbidden-combination check on every motion sample | `safety/joint_limits.py` |
| Fake objects from the shaded floor and frame posts | Masks that are mostly table surface are not objects; frame posts moved out of view | `perception/geometry.py`, `scenarios/workcell.py` |

---

## 8. Model

- **Gemma 4 E4B, 4-bit** (`agent/settings.py GEMMA_MODEL`). E2B is kept in `models/` for comparison.
- **GPU use:** about 4.15 GB for Gemma (E2B with YOLO used about 2.3 GB), about 8.1 GB for the full stack estimated. The
  models load before Isaac Sim starts (`out/gpu_profile/2026-10-26/summary.txt`).
- **Measured on 2026-10-26:**
  - E4B chooses the right operation for a held object, where E2B did not.
  - It identifies large distinct objects (the trays) correctly from the full picture.
  - In a cluttered group it often points at a neighbour: 4 of 13 correct on the whole numbered picture, with or without
    thinking. That's why the per-object descriptions remain the main route for choosing an object.

## 9. What would still fail, and why

- **Pink taken for red:** Gemma's colour judgement. Kept as Gemma's decision by design.
- **Picking an object back out of a tray:** may use the under-read height from section 5.
- **Answers that stay invalid:** if Gemma's answers remain invalid after its retry, the task now ends cleanly with a
  question or a stop, instead of looping.

---

## 10. The basic operations: bigger trays, contents, the moved object's record (2026-10-26, late)

Goal: pick alone, place alone and pick-and-place work from free-form instructions, with the trays holding several
blocks. Each fix answers a failure listed in `docs/history/OPEN_ISSUES_2026-10-26.md`.

### 10.1 Trays hold 3–4 blocks (`scenarios/workcell.py` v1.1)

- **Size:** 300 × 200 mm, 40 mm rim (`TRAY_SIZE`, `TRAY_RIM`); inside 288 × 188 mm.
- **Layout:** long side tangent to the reach ring at 0.70 m, 36° apart, so the trays are about 86 mm from each other
  and 51 mm from the mat. Every corner is inside the measured pick envelope, and all three trays are in the table
  camera's picture.
- **Frozen props:** the red tray now covers the floor spot of the frozen scene's microphone, which the work cell hides
  (it is visual only in the simulator). With a scenario camera (`TABLE_CAMERA` set, `control/robot.py SCENARIO_CELL`),
  the agent no longer excludes depth points inside the microphone's and the camera stand's boxes
  (`config/perception.py _fixtures`), and the scenario check skips their keep-clear zones (`scenarios/checks.py`).
  Otherwise the camera would have cut a corner off the red tray.
- **Checks:** `python -m scenarios.checks --scenario workcell` gives 0 problems at easy, medium and hard (seed 42);
  every tray and mat cell is placeable at yaw 0° and 90°.

### 10.2 "In" goes on a free spot (open issues A4, E3)

- **Seen:** the pink block sat at the red tray's centre. Placing the red block planned only for that centre, found
  "no collision-free way to lower" and retried three times, identically.
- **Fix** (`agent/tools/manipulation.py _free_spots`, `_plan_in`): candidate spots on the container's floor, on a 20 mm
  grid through its centre, nearest the centre first. Each spot is inside the walls and clear of every measured object
  already in the container (its footprint plus the 20 mm placement gap). Up to 12 spots are tried with the same
  collision checks as before. An empty tray still gets its centre.
- **A full tray** fails once, naming how many objects are in it: "no free spot inside the container (6 object(s)
  already in it)". That failure is deterministic, so it is not retried as if it were transient. The same holds when
  every free spot collides.
- **Tests:** `tests/unit/test_container_placement.py` (`test_a_block_goes_beside_what_is_already_in_the_tray`,
  `test_four_blocks_fit_in_a_workcell_tray`, `test_a_full_tray_is_refused_once_and_not_retried`).

### 10.3 The placement check and the world record (open issues A1, A2, C1)

- **Seen:** "place it on the red tray". The check measured the tray's rim as the block (z −120.5 mm). The failed
  measurement was still written into the block's record, so the world held the tray twice (#4 "pink block" with the
  tray's 202 × 142 mm, and #15) and lost the block.
- **Fix** (`agent/tools/verification.py verify_placement`):
  - **The destination is skipped for every relation,** not only `in`. It is recognised by its measured geometry: the
    destination's top height, most of its size, over its centre.
  - **A failed check writes a measurement into the moved object's record only if it has that object's footprint**
    (`_same_size`, the world model's 20 mm gate). Otherwise the record keeps the planned pose from `assume_placed`, and
    the check says so in `record`.
- **Tests:** `test_the_destination_is_never_taken_for_the_placed_object_and_never_written_into_its_record`,
  `test_a_failed_check_keeps_the_measurement_only_when_it_is_the_placed_object`.

### 10.4 The held object keeps its name across instructions (open issue B1)

- **Seen:** "pick the pink block" bound it to #4. The next instruction's goal check looked "pink block" up again in a
  frame without the held block (it was in the gripper) and took a red cube 934 mm from the tray.
- **Fix:** at the end of an instruction, the words bound to the object still in the gripper are kept on the tools
  (`Tools.held_words`). The next instruction starts with them as bindings, logged as "the object in the gripper,
  named so by an earlier instruction" (`agent/supervisor.py run`).
  - This is identity tracking, not matching: only the held object's own words carry over, and they are dropped once it
    is put down.
- **Test:** `tests/unit/test_basics.py test_the_held_object_keeps_the_words_that_named_it`.

### 10.5 Gemma's menu shows what the gripper allows (open issue D3)

- **Seen:** "place red block in red tray": the first plan was `place` with an empty gripper. Gemma's reason assumed it
  still held a block.
- **Fix:** operations carry the gripper state they need (`CapabilitySpec.gripper`: `place` needs an object held;
  `pick`, `pick_place`, `return_home`, `inspect` and `look` need an empty gripper). The menu for planning and for each
  decision notes this against the state now, worded as a condition so that "pick then place" still reads right:
  - `[needs an object in the gripper; it is EMPTY now: only after a pick]`
  - `[needs an empty gripper; it HOLDS #4 now: only after placing it]`

  Nothing is removed or blocked: Gemma still chooses, and the worker still refuses a wrong one
  (`agent/capabilities/specs.py gripper_note`). Decompositions get their state in their own context and no note.
- **Test:** `test_the_menu_notes_what_the_gripper_state_allows_as_a_condition`.

### 10.6 Understanding the request

- **Several objects fit:** the clarify step now asks when several measured objects fit the user's words and nothing
  tells them apart, offering the choices by place or size, not by number (`CLARIFY_SYSTEM`). When nothing has been
  measured yet, the resolution step still asks at execution time, as before.
- **`"request": null`** with a question (open issue F1) is accepted as an empty request (`Clarification`).
- **Test:** `test_a_clarify_question_may_leave_the_request_null`.

### 10.7 A successful pick-and-place reported as failed

- **Seen** (live 19:38, "pick the yellow block and place it in the green tray"): Gemma wrote the success criteria as
  `holding(yellow block); in(yellow block, green tray)`.
  - The block was placed and the camera confirmed it (1.5 mm). But criteria are checked on the final state, where
    `holding` is false after a place.
  - Gemma then observed twice, the 3 changes of course ran out, and the task was reported FAILED.
  - Earlier in the same run, a "replan" identical to the pending `place` had used one change of course for nothing.
- **Fix 1** (`agent/vlm_roles/roles.py check_end_state`, in `plan_goal` and `decompose`): a plan whose last gripper step
  is `place` or `pick_place` cannot state `holding(...)`. Gemma gets the plan back with the reason ("after the last step
  nothing is held … remove that criterion") and fixes it on its retry. A pick alone keeps `holding`.
- **Fix 2** (`agent/supervisor.py _deciding`): a replan whose steps equal the pending steps runs as "continue" and does
  not count as a change of course.
- **Tests** (`tests/integration/test_robustness.py`):
  - `test_a_plan_that_puts_the_object_down_must_not_state_holding_as_its_success`
  - `test_a_pick_alone_keeps_holding_as_its_success`
  - `test_a_replan_that_repeats_the_planned_steps_is_not_a_change_of_course`

### 10.8 Still open

From `OPEN_ISSUES_2026-10-26.md`:

- **A3:** following an object through a release by id; it is still found again by appearance and position.
- **C3 / D1:** "on" a container meaning "in".
- **D1:** decompositions that change the goal's objects.
- **E1, E2:** failure-class recovery, and the user's confirmation counting as completion.

These are the larger architecture items.
