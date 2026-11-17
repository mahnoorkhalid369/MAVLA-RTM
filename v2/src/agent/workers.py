"""The three workers and the scene they share. The master talks to each in plain sentences; each answers in sentences.

    perception    Gemma with the current table image: finds objects, describes the scene, answers questions about what is
                  visible, in its own words.
    manipulation  turns the master's sentence into an arm action and runs the validated pick / place skill
                  (agent/pick_place.py, collision-checked). This is the ONE place where words become measured objects:
                  grounding sees the numbered table and the catalogue (agent/catalogue.py); the wrist close-up is checked
                  against the catalogue names (SigLIP 2). Uncertain source identity gets one improved
                  side view and, when uniquely bound and usable, a bounded Gemma reference comparison.
                  Tools then inspect wrist RGB-D at the pre-grasp pose and replan before descent.
    verification  measures what the arm did in this instruction: the gripper still holds what was picked (fingers,
                  wrist camera), each placed object is in its relation to its target (positions, calibrated places);
                  and, before the request may count as done, a final check: Gemma looks at the table now (and at what
                  the gripper holds, through the wrist camera) and judges it against the user's own words, given
                  no names or reports from earlier steps. HOME joint feedback is supplied when relevant.

There are no pre-made object names: Gemma looks at the full camera picture each time and says what it sees in context.
Low-level values (internal numbers #12 and place ids R1, R2, ...; positions, sizes and heights in mm) are for the
grounding call and the code below it only: grounding sees each number in the text AND the picture, or is told why not
(the held object), and names the objects it chose in a few words, which the reports then use. Everywhere else (the
master, perception, the reports, the user) Gemma gets the picture and words; the calibrated places of the work cell
(flat markings the detector cannot measure: pickup area, mat, mat cells; config/layout.py) are named there.
Workers return a Report with separate status and prose; the loop routes on status, never on the prose.
"""

from __future__ import annotations

import re
import hashlib
import time
from copy import deepcopy

import cv2
import numpy as np

from agent.pick_place import PickPlace
from agent import catalogue, prompts
from agent.performance import options, phase
from agent.contracts import Outcome, Report, TaskContext, response_lines, sensor_epoch
from agent.plan_validation import requests_placement
from config import layout
from perception.somark import draw_marks, isolate, grid

RELATIONS = ("in", "on", "next_to", "left_of", "right_of", "in_front_of", "behind")
GROUND_THINKING = True  # the grounding call thinks first: slower (10-30 s), but it compares ("the larger yellow cube")
# correctly where the quick answer did not (offline check 2026-11-10: 6 of 6 right with thinking, 5 of 6 without)


class Scene:  # the current table observation, and the scene as text: in words, or numbered for grounding
    def __init__(self, tools, run_dir) -> None:
        self.tools, self.run_dir, self.obs, self.n = tools, run_dir, None, 0
        self.image_files = {}

    def current(self, force=False):
        t = self.tools
        if not force and self.obs is not None and t.is_fresh(self.obs):
            return self.obs
        obs = t.perceive("table")
        t.update_world(obs)
        catalogue.name_new(t, t.gemma, getattr(t, "matcher", None), obs, t.log)  # objects not seen before
        try:
            obs.id_image = draw_marks(obs.rgb, {obs.object_ids[k]: m for k, m in obs.marks.items()})
        except ValueError:
            obs.id_image = grid({obs.object_ids[k]: isolate(obs.rgb, m) for k, m in obs.marks.items()}, tile=224, cols=3)
        self.n += 1
        for name, img in (("view", obs.rgb), ("ids", obs.id_image)):
            path = self.run_dir / "images" / f"scene_{self.n}_{name}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.image_files[(obs.frame_id, name)] = str(path)
        self.obs = obs
        return obs

    def image_info(self, obs=None, numbered=False) -> dict:
        obs = obs if obs is not None else self.current()
        img = obs.id_image if numbered else obs.rgb
        return {"camera": "table", "frame_id": obs.frame_id, "variant": "numbered" if numbered else "plain",
                "path": self.image_files.get((obs.frame_id, "ids" if numbered else "view")),
                "shape": list(img.shape) if hasattr(img, "shape") else None,
                "sha256": hashlib.sha256(img.tobytes()).hexdigest() if hasattr(img, "tobytes") else None,
                "age_s": round(time.monotonic() - obs.taken_at, 3), "motion_count": obs.motion_count,
                "visible_ids": sorted(set(obs.object_ids.values())), "mark_to_object": dict(obs.object_ids)}

    def evidence_key(self):
        if self.obs is None:
            return ()
        shapes = []
        for oid in set(self.obs.object_ids.values()):
            obj = self.tools.world.objects.get(oid)
            if obj is not None:
                g = obj.geometry
                shapes.append(tuple(round(float(v), 2) for v in [*g["centre"], *g["footprint"], g["height"]]))
        return tuple(sorted(shapes)), self.tools.held_object

    def name(self, oid) -> str:  # "the tall pink block (on pickup area)": the words grounding used for it
        o = self.tools.world.objects.get(oid)
        if o is None:
            return "the object"
        what = self.tools.object_words.get(oid) or "object"
        where = "in the gripper" if oid == self.tools.held_object else self.where(o.geometry)
        return f"the {what}" + (f" ({where})" if where else "")

    @staticmethod
    def where(g) -> str:
        inside = [r for r in layout.REGIONS.values() if r.footprint_margin(g["centre"], [0, 0], None) >= 0]
        return f"on {min(inside, key=lambda r: r.size[0] * r.size[1]).name}" if inside else ""

    def text(self, numbers: bool = False, observation=None) -> str:
        t = self.tools
        held = "nothing" if t.held_object is None else self.name(t.held_object)[4:]
        if not numbers:
            ids = set(self.obs.object_ids.values()) if self.obs is not None else set(t.world.objects)
            known = [t.object_words[i] for i in sorted(ids) if i in t.object_words and i != t.held_object]
            return prompts.SCENE.format(catalogue=", ".join(known) or "none named yet",
                                        places=", ".join(r.name for r in layout.REGIONS.values()), held=held)
        lines = []
        obs = observation if observation is not None else self.current()
        visible = set(obs.object_ids.values())
        for oid, o in sorted(t.world.objects.items()):
            if oid not in visible or oid == t.held_object:
                continue
            g, words, where = o.geometry, t.object_words.get(oid), self.where(o.geometry)
            fx, fy = (round(v * 1000) for v in g["footprint"])
            x, y = (round(v * 1000) for v in g["centre"][:2])
            lines.append(prompts.GROUND_OBJECT.format(oid=oid, name=f"{words}, " if words else "",
                         width=fx, length=fy, height=round(g['height'] * 1000), x=x, y=y,
                         where=f", {where}" if where else ""))
        places = [prompts.GROUND_PLACE.format(rid=r.id, name=r.name, width=round(r.size[0] * 1000),
                  length=round(r.size[1] * 1000), x=round(r.centre[0] * 1000), y=round(r.centre[1] * 1000))
                  for r in layout.REGIONS.values()]
        if t.held_object is not None:
            held = prompts.GROUND_HELD.format(oid=t.held_object, held=held)
        return prompts.GROUND_SCENE.format(objects="\n".join(lines), places="\n".join(places), held=held)


class Perception:
    def __init__(self, gemma, scene: Scene, log) -> None:
        self.gemma, self.scene, self.log = gemma, scene, log

    def run(self, message: str) -> Report:
        obs = self.scene.current(force=True)
        prompt = prompts.PERCEIVE.format(scene=self.scene.text(), message=message)
        self.log("model_input", role="perception", prompt=prompt, images=[self.scene.image_info(obs)],
                 call_id=getattr(self.gemma, "call_count", 0) + 1)
        answer = self.gemma.ask(prompt, images=[obs.rgb],
                                thinking=False, max_new_tokens=160)
        self.log("worker", worker="perception", message=message, answer=answer)
        return _status(answer, {"OK": "ok", "UNSURE": "unsure", "NOT FOUND": "failed"})


def parse_action(answer: str):
    """Consume every action line; normalize an unambiguous split pick/place response."""
    lines, memory = response_lines(answer)
    if memory is not None:
        raise ValueError("grounding cannot write master memory")
    names = {}
    if lines[-1].lower().startswith("names:"):
        naming = lines.pop().partition(":")[2]
        for part in naming.split(";") if naming.strip() else []:
            match = re.fullmatch(r"\s*#(\d+)\s*=\s*(.+?)\s*", part)
            if not match:
                raise ValueError("names must use #number = description")
            names[int(match[1])] = match[2]
    if len(lines) == 2 and re.fullmatch(r"pick #\d+", lines[0], re.I) and re.fullmatch(
            r"place \w+ (?:#\d+|r\d+)", lines[1], re.I):
        lines = [" ".join(lines)]
    if len(lines) != 1:
        raise ValueError("return one action, or a pick followed by its placement; extra actions are not discarded")
    line = lines[0].strip("`. ")
    if line.lower() == "home":
        return "home", None, None, None, names
    if line.lower().startswith("cannot:") and line.partition(":")[2].strip():
        return "cannot", None, None, line.partition(":")[2].strip(), names
    match = re.fullmatch(r"(?:pick #(\d+))?\s*(?:place (\w+) (#\d+|r\d+))?", line.lower())
    if not match or not (match[1] or match[2]):
        raise ValueError("not one of the action lines: pick #n, place relation #n/Rn, pick #n place relation #n/Rn, home, cannot: reason")
    obj, rel, target = int(match[1]) if match[1] else None, match[2], match[3]
    if rel is not None and rel not in RELATIONS:
        raise ValueError(f"unsupported relation: {rel}")
    return "act", obj, rel, target, names


class Manipulation:
    def __init__(self, tools, gemma, scene: Scene, log, context=None) -> None:
        self.tools, self.gemma, self.scene, self.log = tools, gemma, scene, log
        self.context = context if context is not None else TaskContext()
        self.failed_actions = set()

    def _chosen(self, answer: str) -> dict:
        """Measured size of every #id in a grounding answer, for the log line (what the number refers to)."""
        out = {}
        for oid in dict.fromkeys(int(n) for n in re.findall(r"#(\d+)", answer)):
            o = self.tools.world.objects.get(oid)
            if o is None:
                out[oid] = "not a known id"
                continue
            g = o.geometry
            fp = g.get("footprint") or [0, 0]
            held = " (held)" if oid == self.tools.held_object else ""
            out[oid] = f"{fp[0] * 1000:.0f} x {fp[1] * 1000:.0f} mm, {g.get('height', 0) * 1000:.0f} mm tall{held}"
        return out

    def run(self, message: str) -> Report:
        report = self._ground_and_act(message)
        if report.code == "identity":  # the wrist close-up showed another object: choose again once, with the reason
            self.log("identity_retry", reason=report.text)
            report = self._ground_and_act(message, hint=report.text)
        return report

    def _check_fit(self, oid, rel, ref) -> None:
        """Physical plausibility only (Gemma still chooses): the object moved must fit in / on the destination."""
        world = self.tools.world.objects
        if oid is None or oid == ref or oid not in world or ref not in world:
            return
        g, d = world[oid].geometry, world[ref].geometry
        size = sorted(g["footprint"], reverse=True)
        dest = sorted(d.get("extents") or d["footprint"], reverse=True)
        room = [e - 2 * self.tools.CONTAINER_WALL_M for e in dest] if rel == "in" else [e * 1.1 for e in dest]
        if size[0] > room[0] or size[1] > room[1]:
            raise ValueError(
                f"#{ref} ({self.scene.name(ref)[4:]}: {dest[0] * 1000:.0f} x {dest[1] * 1000:.0f} mm, "
                f"{d['height'] * 1000:.0f} mm tall) is too small to put {self.scene.name(oid)} "
                f"({size[0] * 1000:.0f} x {size[1] * 1000:.0f} mm) {rel} it. Choose the destination the request names, "
                "by its catalogue name and size")

    def _ground_and_act(self, message: str, hint: str = "") -> Report:
        held = self.tools.revalidate_held()
        if held.get("held") == "unknown":
            return Report("failed", "the gripper holds an unidentified object; resolve it before another pick",
                          code="unknown_held")
        error = hint
        for attempt in range(2):
            obs = self.scene.current(force=True)
            text = prompts.GROUND.format(scene=self.scene.text(numbers=True, observation=obs), message=message)
            prompt = text + (prompts.RESPONSE_REPAIR.format(error=error) if error else "")
            images, info = [obs.id_image], [self.scene.image_info(obs, numbered=True)]
            prompt += prompts.GROUND_ALLOWED_IDS.format(ids=', '.join(f'#{k}' for k in sorted(obs.object_ids.values())))
            self.log("model_input", role="manipulation", prompt=prompt, images=info,
                     call_id=getattr(self.gemma, "call_count", 0) + 1)
            answer = self.gemma.ask(prompt,
                                    images=images, thinking=GROUND_THINKING, max_new_tokens=120)
            self.log("worker", worker="manipulation", message=message, answer=answer, thinking=self.gemma.last_thinking,
                     chosen=self._chosen(answer))
            try:
                kind, obj, rel, target, names = parse_action(answer)
                if kind == "act" and rel is None and (requests_placement(message) or getattr(self, "require_transfer", False)):
                    raise ValueError("delegated transfer requires pick AND place with a destination; pickup alone is incomplete")
                if not self.tools.is_fresh(obs):
                    raise ValueError("the grounding image expired or the robot moved; ground against the refreshed image")
                visible = set(obs.object_ids.values())
                if kind == "act" and obj is not None and obj not in visible and obj != self.tools.held_object:
                    raise ValueError(f"invalid source ID #{obj}; it is not an allowed object ID in frame {obs.frame_id}. "
                                     "This does not establish that the requested object is absent. Read the number joined to its outline.")
                if (kind == 'act' and obj is not None and obj != self.tools.held_object
                        and self.tools.world.objects[obj].geometry.get('support_confident') is False):
                    raise ValueError('selected source has no confirmed support/full-height geometry; select a whole-object '
                                     'candidate or refresh the RGB-D view. Do not infer that the object is absent.')
                if kind == "act" and target and target.startswith("#") and int(target[1:]) not in visible:
                    raise ValueError(f"destination {target} is not marked in the supplied image; choose a visible destination")
                if kind == "act" and target and target.startswith("r") and layout.region(target) is None:
                    raise ValueError(f"unknown fixed region: {target}")
                if kind == "act" and target and target.startswith("#") and rel in ("in", "on"):
                    normalize = getattr(self.tools, "placement_relation", None)
                    if normalize is not None:
                        rel = normalize(rel, int(target[1:]))
                    self._check_fit(obj if obj is not None else self.tools.held_object, rel, int(target[1:]))
                used = {obj, int(target[1:]) if target and target.startswith("#") else None}
                if kind == "act" and obj is None:
                    used.add(self.tools.held_object)
                if not set(names).issubset(used - {None}):
                    raise ValueError("name only the objects used by this action")
                if kind == "act":
                    if obj is None and self.tools.held_object is None:
                        return Report("failed", "the gripper is empty: there is nothing to put down",
                                      code="precondition", frame_id=obs.frame_id)
                    if obj is not None and self.tools.held_object not in (None, obj):
                        return Report("failed", "place the currently held object before attempting another pick",
                                      code="precondition", frame_id=obs.frame_id)
                    if not self.tools.is_fresh(obs):
                        raise ValueError("the grounding image expired; refresh before motion")
                    # Model inference can take a minute. Re-measure the scene immediately before planning,
                    # keeping the semantic binding only when the selected whole-object identities remain stable.
                    watched = {oid for oid in used if oid is not None and oid != self.tools.held_object}
                    before = {oid: (np.array(self.tools.world.objects[oid].geometry['centre'][:2]),
                                    float(self.tools.world.objects[oid].geometry['top_z'])) for oid in watched}
                    binding_frame = obs.frame_id
                    obs = self.scene.current(force=True)
                    visible = set(obs.object_ids.values())
                    for oid, (xy, z) in before.items():
                        g = self.tools.world.objects[oid].geometry
                        if (oid not in visible or np.linalg.norm(np.array(g['centre'][:2]) - xy) > .015
                                or abs(g['top_z'] - z) > .010):
                            raise ValueError('selected object/destination changed during visual grounding; '
                                             'bind the refreshed whole-object crop before motion')
                    self.log('grounding_refresh', binding_frame_id=binding_frame, execution_frame_id=obs.frame_id,
                             selected_ids=sorted(watched))
                break
            except ValueError as exc:
                error = str(exc)
        else:
            return Report("failed", 'visual grounding could not produce a valid confirmed object/action selection: ' + error,
                          code="grounding_invalid", frame_id=obs.frame_id,
                          details={'recovery': 'stop', 'required_change': 'inspect the saved candidate crops or refresh segmentation; '
                                   'object absence has not been established', 'valid_ids': sorted(obs.object_ids.values())})
        for oid, words in names.items():
            if oid in self.tools.world.objects and oid not in self.tools.object_words:
                self.tools.object_words[oid] = words.strip(" .")
        if kind == "cannot":
            return Report("failed", target, code="cannot_act", frame_id=obs.frame_id)
        if kind == "home":
            self.context.motion_attempted = True
            self.context.pending.add("home")
            r = self.tools.return_home()
            v = self.tools.verify_home() if r["ok"] else None
            if v and v["pass"]:
                self.context.record(Outcome("home"), "Returned HOME; joint feedback passed.")
            return Report("done", "back at the home pose") if v and v["pass"] else \
                Report("failed", f"the home pose was not reached ({r.get('reason') or 'joint check failed'})", code="home",
                       details={"stage": "home", "recovery": "stop", "held_object": self.tools.held_object,
                                "required_change": "a verified HOME path or joint-state correction is required"})
        if obj is not None and obj == self.tools.held_object and rel:
            obj = None  # "pick <the held object> place ...": it is already in the gripper, so this is the put-down
        self.log("grounded_action", frame_id=obs.frame_id, object_id=obj, relation=rel, destination=target,
                 object_geometry=self.tools.world.objects[obj].geometry if obj is not None else None,
                 destination_geometry=self.tools.world.objects[int(target[1:])].geometry
                 if target and target.startswith("#") else None)
        return self._act(obj, rel, target, obs)

    def _act(self, obj, rel, target, observation=None) -> Report:
        t, s = self.tools, self.scene
        region = layout.region(target) if target else None
        ref = int(target[1:]) if target and target.startswith("#") else None
        if target and region is None and ref not in t.world.objects:
            return Report("failed", "the place to put it down is not on the table")
        if obj is not None and obj not in t.world.objects:
            return Report("failed", "the object to pick up is not on the table")
        if obj is not None and t.held_object is not None:
            return Report("failed", f"the gripper already holds {s.name(t.held_object)}; it must be put down first")
        if obj is None and t.held_object is None:
            return Report("failed", "the gripper is empty: there is nothing to put down")
        if region is not None:
            rel = "in"  # a flat place: "on" it and "in" it are the same, inside its outline
        mode = "pick" if rel is None else ("place" if obj is None else "pick_place")
        held_before = t.held_object
        obs = observation
        attempt_key = (mode, obj if obj is not None else held_before, rel, target, s.evidence_key())
        grasp_key = ("grasp", obj, s.evidence_key()) if obj is not None else None
        if attempt_key in self.failed_actions or (grasp_key is not None and grasp_key in self.failed_actions):
            return Report("failed", "this action already failed with the same measured grasp and surroundings; "
                          "change the physical evidence or ask for help before retrying", code="unchanged_action",
                          frame_id=obs.frame_id if obs is not None else None)
        self.context.motion_attempted = True
        self.context.outcomes.pop("home", None)
        self.context.current.pop("home", None)
        self.context.pending.add(obj if obj is not None else held_before)
        skill = PickPlace(t, self.log)
        if obj is not None:
            skill.judge_source = self._source_judge(obj, obs)
        self._attempt = dict(skill=skill, mode=mode, obj=obj, rel=rel, ref=ref, region=region,
                             held_before=held_before, frame_id=obs.frame_id if obs is not None else None,
                             key=attempt_key, grasp_key=grasp_key, recoveries=0)
        skill.on_effect = self._physical_effect
        if self.gemma is not None and rel is not None:  # the placement is judged on the wrist picture, in words
            placed = s.name(obj if obj is not None else held_before)[4:].split(" (")[0]
            where = region.name if region is not None else t.object_words.get(ref, "destination")
            skill.judge_placement = lambda view: self._judge_placement(view, placed, rel, where)
        final_home = mode != "pick" and not getattr(self, "defer_home", False)
        r = skill.run(mode, obj, rel, ref, final_home=final_home, observation=obs,
                      region=region.id if region else None)
        t.chain_ready = bool(r["status"] == "COMPLETED" and mode != "pick" and not final_home)
        t.chain_ready_revision = getattr(t, "scene_revision", None)
        return self._skill_report(r)

    def _judge_placement(self, view: dict, placed: str, rel: str, where: str) -> tuple[str, str]:
        return judge_placement(self.gemma, self.log, view, placed, rel, where)

    def _source_judge(self, oid, obs):
        """Freeze one table instance, then permit ONE semantic fallback for its bound wrist body.

        The tool calls this only after unique depth binding and usable body visibility. No model
        answer can repair missing association, authorize motion or replace collision checks.
        """
        if (self.gemma is None or obs is None or not hasattr(obs, "marks")
                or not isinstance(getattr(obs, "rgb", None), np.ndarray)
                or not hasattr(self.tools, "scene_revision") or not hasattr(self.tools, "run_dir")):
            return None
        marks = [k for k, value in obs.object_ids.items() if value == oid]
        if len(marks) != 1 or obs.marks.get(marks[0]) is None:
            return None
        reference = isolate(obs.rgb, obs.marks[marks[0]], margin=.15).copy()
        t = self.tools
        expected = t.object_words.get(oid)
        geometry, revision, epoch = deepcopy(t.world.objects[oid].geometry), t.scene_revision, sensor_epoch(t)
        path = t.run_dir / "images" / f"{obs.frame_id}_source_{oid}_reference.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
        used = False
        sealed, scope_valid = False, True

        def begin_inspection():
            # The validated approach itself increments scene_revision. Scope the wrist sequence AFTER
            # that move, without changing executor invalidation or allowing later motion to refresh proof.
            nonlocal geometry, revision, sealed, scope_valid
            if sealed:
                return
            sealed = True
            current = t.world.objects[oid].geometry
            scope_valid = (epoch == sensor_epoch(t) and expected == t.object_words.get(oid)
                           and np.linalg.norm(np.array(current["centre"][:2])-geometry["centre"][:2]) <= .015
                           and abs(current["top_z"]-geometry["top_z"]) <= .010
                           and max(abs(a-b) for a,b in zip(current["footprint"],geometry["footprint"])) <= .010)
            geometry, revision = deepcopy(current), t.scene_revision

        def judge(view):
            nonlocal used
            def same_source():
                return (scope_valid and view.get("object_id") == oid and sensor_epoch(t) == epoch
                        and t.scene_revision == revision and t.object_words.get(oid) == expected
                        and t.world.objects[oid].geometry == geometry and t.held_object is None)
            if used or not expected or not same_source():
                return "unsure", "source reference expired or its confirmation budget is exhausted"
            used = True
            prompt = prompts.SOURCE_IDENTITY.format(expected=expected)
            self.log("model_input", role="manipulation", phase="source_identity", prompt=prompt,
                     images=[{"camera": "table", "variant": "source_reference", "frame_id": obs.frame_id,
                              "object_id": oid, "path": str(path)},
                             {"camera": "wrist", "variant": "bound_foreground", "frame_id": view["frame_id"],
                              "object_id": oid, "mark": view["mark"], "path": view["path"]}],
                     call_id=getattr(self.gemma, "call_count", 0) + 1)
            with phase("source_identity"):
                answer = self.gemma.ask(prompt, images=[reference, view["rgb"]], thinking=True, max_new_tokens=120)
            r = _status(answer, {"YES": "ok", "NO": "failed", "UNSURE": "unsure"})
            word = {"ok": "yes", "failed": "no"}.get(r.word, "unsure")
            reason = r.text
            if not same_source():
                word, reason = "unsure", "source or scene changed during semantic confirmation"
            self.log("source_identity_answer", object_id=oid, frame_id=view["frame_id"],
                     reference_frame=obs.frame_id, answer=answer, verdict=word, reason=reason)
            return word, reason
        judge.begin_inspection = begin_inspection
        return judge

    def _physical_effect(self, effect, oid):
        a = self._attempt
        if effect == "holding_verified":
            self.context.record(Outcome("holding", oid), f"Holding {self.scene.name(oid)}; retention verified.")
        elif effect in ("released", "placement_verified"):
            kind = "released" if effect == "released" else "placed"
            self.context.record(Outcome(kind, oid, a["rel"], a["ref"], a["region"].id if a["region"] else None),
                                f"Released {self.scene.name(oid)}; gripper empty; placement " +
                                ("not yet verified." if kind == "released" else "verified."))
            self.context.placements.pop(oid, None)
            receipt = getattr(a["skill"], "placement_evidence", None)
            if effect == "placement_verified" and receipt is not None:
                self.context.placements[oid] = {**receipt, "outcome": self.context.outcomes[oid]}

    def _skill_report(self, r):
        a, t, s = self._attempt, self.tools, self.scene
        oid = a["obj"] if a["obj"] is not None else a["held_before"]
        effects = r.get("effects")
        failure_kind = (r.get("failure") or {}).get("cls", "skill_failed")
        if r["status"] != "COMPLETED":
            if failure_kind in ("retention", "grasp", "blocked"):
                self.failed_actions.add(a["key"])
            if failure_kind in ("retention", "grasp") and a["grasp_key"] is not None:
                self.failed_actions.add(a["grasp_key"])
            if effects is not None:
                if effects.get("released"):
                    kind = "placed" if effects.get("placement_verified") else "released"
                    self.context.record(Outcome(kind, oid, a["rel"], a["ref"], a["region"].id if a["region"] else None),
                                        f"Released {s.name(oid)}; placement " + ("verified." if kind == "placed" else "NOT verified. Gripper is empty."))
                elif not effects.get("grasp_attempted") and failure_kind not in ("retention", "execution"):
                    # Inspection/planning failures are closed attempts, not unresolved payloads.
                    self.context.pending.discard(oid)
                else:
                    self.context.outcomes.pop(oid, None)
                    self.context.pending.add(oid)
                    self.context.current[oid] = "Grasp/transport interrupted; held state requires sensor reconciliation."
            recovery = "stop"
            required = {
                "sensor_epoch": "restart the session; saved object identities and images are invalid after a simulator reset",
                "inspection": "a uniquely associated whole-object wrist view or corrected object binding is required before descent",
                "identity": "the wrist close-up does not show the chosen object; choose again by catalogue name and size",
                "source_identity": "source inspection exhausted its side-view recovery; improve the view or object "
                                   "binding before another attempt; no grasp was attempted",
                "blocked": "a collision-free plan with obstacles preserved is required; repeating the command is disabled",
                "placement": "the wrist picture after release does not show the object where it was asked; look at the "
                             "table before deciding; the empty gripper cannot place it again",
                "execution": "reconcile actual joint, gripper and object state before any new motion",
                "home": "a verified HOME path and joint state are required",
            }.get(failure_kind, "physical correction or a new supported plan is required")
            if failure_kind == "inspection" and a["recoveries"] == 0:
                recovery, required = "inspect_wrist", "one fresh wrist measurement at the current pre-grasp pose"
            elif failure_kind == "retention" and a["recoveries"] == 0:
                recovery, required = "reconcile_held", "finger/wrist evidence must establish whether the same object is held"
            details = dict(stage=r.get("stage", "unknown"), object_id=oid, reference_id=a["ref"],
                           held_object=t.held_object, effects=effects or {}, recovery=recovery,
                           required_change=required, recovery_attempts=a["recoveries"], checks=r.get("verifications", [])[-3:])
            return Report("failed", _words(r["message"]), code=failure_kind, frame_id=a["frame_id"], details=details)
        if a["mode"] == "pick":
            text = f"picked up {s.name(r['object_id'])}; the fingers confirm it is held after the lift"
            self.context.record(Outcome("holding", r["object_id"]), text)
        else:
            placed = a["held_before"] if a["mode"] == "place" else r["object_id"]
            where = a["region"].name if a["region"] else s.name(a["ref"])
            home = " and returned home" if any(v.get("check") == "home" and v.get("pass") for v in r.get("verifications") or []) else ""
            text = f"put {s.name(placed)} {a['rel'].replace('_', ' ')} {where}; the wrist camera shows it there{home}"
            self.context.record(Outcome("placed", placed, a["rel"], a["ref"], a["region"].id if a["region"] else None), text)
        return Report("done", text, frame_id=a["frame_id"],
                      details={"stage": r.get("stage", "completed"), "object_id": oid,
                               "held_object": t.held_object, "recovery_attempts": a["recoveries"],
                               "effects": r.get("effects", {}), "checks": r.get("verifications", [])[-3:]})

    def recover(self, report):
        """One measured local recovery, with the frozen object/action, without another VLM plan or HOME trip."""
        a = self._attempt
        if a["recoveries"] >= 1:
            return Report("failed", "measured recovery budget exhausted", code="recovery_exhausted")
        a["recoveries"] += 1
        skill = a["skill"]
        oid = a["obj"] if a["obj"] is not None else a["held_before"]
        self.context.pending.add(oid)
        if report.details.get("recovery") == "inspect_wrist":
            return self._skill_report(skill.resume("INSPECTING_GRASP"))
        if report.details.get("recovery") == "reconcile_held":
            evidence = self.tools.revalidate_held()
            self.log("recovery_evidence", object_id=skill.object_id, evidence=evidence)
            if evidence.get("held") in ("verified", "recovered") and self.tools.held_object == skill.object_id:
                skill.effects["holding_verified"] = True
                skill.verifications.append({"check": "retention_recovered", "pass": True, "evidence": evidence})
                state = "COMPLETED" if a["mode"] == "pick" else ("PLACING" if skill.last_stage == "CARRYING" else "CARRYING")
                return self._skill_report(skill.resume(state))
            self.context.pending.discard(skill.object_id)
            self.context.record(Outcome("unknown", skill.object_id),
                                "The requested object's pose is unknown after failed retention; pickup is not complete.")
            return Report("failed", "retention recovery could not confirm the object is held; remeasure its pose before a new grasp",
                          code="retention_unresolved", details={"stage": "held_reconciliation", "recovery": "stop", "evidence": evidence})
        return Report("failed", "no measured recovery is available for this failure", code="recovery_unavailable")


class Verification:
    """Verification by the WRIST camera only: for each placed object the wrist camera looks down at where it was put
    and Gemma says in words whether it is there; a held object is checked by the fingers (and the wrist camera, in
    revalidate_held); HOME by joint feedback. No table camera, no measuring or matching of the placement."""

    def __init__(self, tools, context: TaskContext, scene: Scene, log, gemma=None) -> None:
        self.tools, self.context, self.scene, self.log, self.gemma = tools, context, scene, log, gemma
        self.views: dict = {}  # object id -> (context.revision, ok, what Gemma saw, wrist view): one look per state

    def _names(self, c) -> tuple[str, str, str]:
        t = self.tools
        region = layout.region(c.region_id)
        where = region.name if region is not None else t.object_words.get(c.reference_id, "the destination")
        return t.object_words.get(c.object_id, "object"), (c.relation or "on").replace("_", " "), where

    def _look(self, c) -> tuple[bool, str, dict | None]:
        """One placed object: the wrist camera looks at where it is now; Gemma judges (unsure -> once from farther)."""
        receipt = self._receipt(c)
        if receipt is not None:
            self.log("evidence_reused", object_id=c.object_id, frame_id=receipt["view"]["frame_id"], purpose="placement gate")
            return True, receipt["text"], receipt["view"]
        cached = self.views.get(c.object_id)
        if cached is not None and cached[0] == self._version():
            return cached[1:]
        t = self.tools
        epoch = sensor_epoch(t)
        placed, rel, where = self._names(c)
        o = t.world.objects.get(c.object_id)
        result = (False, f"no wrist view of the {placed}", None)
        if t.held_object is not None:
            # no viewpoint moves while holding: the wrist picture taken right after its release stands
            result = (False, f"placement evidence for the {placed} is stale; cannot move the view while holding", None)
        elif o is not None:
            for farther in (False, True):
                view = t.look_at(o.geometry["centre"], f"obj_{c.object_id}", farther, "verify placement")
                if not view["ok"]:
                    result = (False, f"no wrist view of the {placed} ({view['reason']})", None)
                    continue
                view.update(object_id=c.object_id, reference_id=c.reference_id, relation=c.relation)
                word, text = judge_placement(self.gemma, self.log, view, placed, rel, where)
                if view.get("epoch", epoch) != sensor_epoch(t):
                    return False, "simulator reset during placement verification; restart the session", None
                result = (word == "yes", text if word != "unsure" else f"unconfirmed: {text}", view)
                if word != "unsure":
                    break
        self.views[c.object_id] = (self._version(), *result)
        return result

    def _version(self):
        return self.context.revision, getattr(self.tools, "scene_revision", None), sensor_epoch(self.tools)

    def _receipt(self, c):
        if not options(self.tools).reuse_placement:
            return None
        r = self.context.placements.get(c.object_id)
        if (r is not None and r["outcome"] == c and r["word"] == "yes" and r["view"].get("ok")
                and r["scene_revision"] is not None
                and r["scene_revision"] == getattr(self.tools, "scene_revision", None)
                and r["epoch"] == sensor_epoch(self.tools)):
            return r
        return None

    def _home_after_looking(self, moved: bool) -> bool:
        if moved and self.tools.held_object is None:
            result = self.tools.return_home()
            if not result["ok"]:
                self.log("verification_home_failed", reason=result.get("reason", "HOME failed"))
                return False
        return True

    def final_check(self, request: str) -> Report:  # independent check against the complete original request
        """Gemma judges the user's own words against WRIST pictures only: one looking down at each placed object (the
        verification's pictures when nothing changed since) and, if something is held, the view of the gripper."""
        t, images, info, captions, moved = self.tools, [], [], [], False
        epoch = sensor_epoch(t)
        if t.held_object is not None:
            view = t.wrist_picture("final gripper")
            images.append(view["rgb"])
            captions.append(prompts.FINAL_GRIPPER_PICTURE.format(number=len(images)))
            info.append({"camera": "wrist", "variant": "plain", "frame_id": view["frame_id"], "path": view["path"]})
        for c in [c for c in self.context.outcomes.values() if c.kind == "placed"]:
            receipt = self._receipt(c)
            cached = self.views.get(c.object_id)
            view = receipt["view"] if receipt else (cached[3] if cached is not None and cached[0] == self._version() else None)
            if view is None and t.held_object is None:
                o = t.world.objects.get(c.object_id)
                view = t.look_at(o.geometry["centre"], f"obj_{c.object_id}", False, "final check") if o else None
                moved = True
                view = view if view and view["ok"] else None
            if view is not None:
                images.append(view["rgb"])
                captions.append(prompts.FINAL_PLACEMENT_PICTURE.format(number=len(images)))
                info.append({"camera": "wrist", "variant": "plain", "frame_id": view["frame_id"], "path": view["path"]})
        if not images:
            view = t.wrist_picture("final")
            images.append(view["rgb"])
            captions.append(prompts.FINAL_CURRENT_PICTURE)
            info.append({"camera": "wrist", "variant": "plain", "frame_id": view["frame_id"], "path": view["path"]})
        sensors = ""
        if "home" in self.context.outcomes:
            sensors = prompts.FINAL_HOME_REACHED if t.verify_home()["pass"] else prompts.FINAL_HOME_NOT_REACHED
        prompt = prompts.CHECK.format(request=request, pictures="\n".join(captions), sensors=sensors)
        self.log("model_input", role="verification", phase="final_check", prompt=prompt, images=info,
                 call_id=getattr(self.gemma, "call_count", 0) + 1)
        answer = self.gemma.ask(prompt, images=images, thinking=True, max_new_tokens=300)
        self.log("worker", worker="final_check", request=request, answer=answer, thinking=self.gemma.last_thinking)
        if epoch != sensor_epoch(t):
            return Report("no", "simulator reset during the final check; restart the session", code="sensor_epoch")
        if not self._home_after_looking(moved):
            return Report("no", "final picture was judged, but the return HOME failed", code="home")
        return _status(answer, {"YES": "yes", "NO": "no"})

    def run(self, message: str = "") -> Report:
        t, s, checks = self.tools, self.scene, list(self.context.outcomes.values())
        if self.context.pending:
            return Report("no", "an attempted arm action is unresolved; inspect or recover before finishing",
                          code="unresolved_action")
        if not checks:
            return Report("no", "the arm has not picked or placed anything in this instruction")
        if any(c.kind == "released" for c in checks):
            return Report("no", "the object was released and the gripper is empty, but placement remains unverified",
                          code="placement_unverified")
        if any(c.kind == "unknown" for c in checks):
            return Report("no", "an object's current pose is unknown after interrupted execution", code="pose_unknown")
        results, moved = [], False
        for c in checks:
            if c.kind == "home":
                ok = t.verify_home()["pass"]
                results.append((ok, "HOME joints verified" if ok else "HOME joint check failed"))
                continue
            if c.kind == "holding":
                ev = t.revalidate_held()
                ok = t.held_object == c.object_id and ev["held"] in ("verified", "recovered")
                results.append((ok, f"{'holding' if ok else 'NOT holding'} {s.name(c.object_id)}"))
                continue
            placed, rel, where = self._names(c)
            fresh = self._receipt(c) is None and self.views.get(c.object_id, (None,))[0] != self._version()
            ok, seen, view = self._look(c)
            moved |= fresh and view is not None
            results.append((ok, f"the {placed} is {'' if ok else 'NOT '}{rel} the {where} (wrist camera: {seen})"))
        self.log("worker", worker="verification", results=[{"ok": ok, "text": x} for ok, x in results])
        if not self._home_after_looking(moved):
            return Report("no", "return HOME failed after the placement check", code="home")
        text = "; ".join(x for _, x in results)
        return Report("yes", text) if all(ok for ok, _ in results) else Report("no", text)


def judge_placement(gemma, log, view: dict, placed: str, rel: str, where: str) -> tuple[str, str]:
    """Gemma looks at one wrist-camera picture: is the object where it was put? Words only (catalogue names); no
    numbers, measurements or matching. -> ("yes" | "no" | "unsure", what Gemma sees)."""
    prompt = prompts.PLACED.format(obj=placed, relation=rel.replace("_", " "), dest=where)
    log("model_input", role="verification", phase="placement", prompt=prompt,
        images=[{"camera": "wrist", "variant": "plain", "frame_id": view["frame_id"], "path": view["path"]}],
        call_id=getattr(gemma, "call_count", 0) + 1)
    with phase("placement"):
        answer = gemma.ask(prompt, images=[view["rgb"]], thinking=True, max_new_tokens=120)
    r = _status(answer, {"YES": "ok", "NO": "failed", "UNSURE": "unsure"})
    word = {"ok": "yes", "failed": "no"}.get(r.word, "unsure")
    log("placement_check", frame_id=view["frame_id"], word=word, text=r.text, thinking=gemma.last_thinking,
        sim_time=view.get("sim_time"), object_id=view.get("object_id"), reference_id=view.get("reference_id"),
        relation=view.get("relation"), prompt=prompt, path=view.get("path"))
    return word, r.text


def _status(answer: str, words: dict) -> Report:
    try:
        lines, memory = response_lines(answer)
        if memory is not None:
            raise ValueError("worker output cannot contain master memory")
        if len(lines) != 1:
            raise ValueError("return one status line; extra response lines cannot be ignored")
        head = lines[0]
    except ValueError as exc:
        return Report("unsure", str(exc), code="invalid_response")
    for key, word in words.items():
        if re.match(re.escape(key) + r"(?:\b|$)", head, re.I):
            return Report(word, head[len(key):].lstrip(" :.-\n"))
    return Report("unsure", head or "empty response", code="invalid_response")


def _words(text: str) -> str:  # the skill's messages carry numbers and measurements: those stay in the log
    text = re.sub(r"\s*\([^()]*\d[^()]*\)", "", text)  # "(0.92 m from the base)", "(10 mm, 5 mm, 5 deg)"
    text = re.sub(r"(?:object )?#\d+", "the object", text)
    text = re.sub(r"(?:obj|unseen)_\d+", "an obstacle", text)
    text = re.sub(r"\s*,?\s*(?:at )?[xyz] [+-]?\d+(?:\.\d+)?(?: ?mm)?", "", text)
    text = re.sub(r"[+-]?\d+(?:\.\d+)? ?(?:mm|deg|m)\b", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")
