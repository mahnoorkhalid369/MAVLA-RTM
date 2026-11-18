"""The master and its three workers. Each agent is a prompt plus what it may call; Gemma makes every decision.

run() handles one user instruction: look, plan, do the steps, report back to the master, check, answer.
Every log line goes through say(who, text, started): who it is from, and when the step started (for its duration).
"""

import re
import time

import prompts

MAX_STEPS, MAX_DECISIONS, MAX_QUESTIONS = 12, 8, 2
WORKERS = ("manipulation", "perception", "verification")


class Agents:
    def __init__(self, gemma, robot, ask_user, say=lambda who, text, started=None: print(f"[{who}] {text}")):
        self.gemma, self.robot, self.ask_user, self.say = gemma, robot, ask_user, say
        robot.choose = self.spot  # the robot asks back which of the free spots it measured to use
        self.names = {}  # object id -> the name Gemma gave it at the table (kept for the whole session)

    # ------------------------------------------------------------------ the scene, in words and pictures
    def look(self):
        started = time.monotonic()
        self.rgb, self.numbered, self.objects = self.robot.look()
        self.say("robot", f"new table picture: {len(self.objects)} objects outlined and measured", started)
        for oid, o in self.objects.items():
            if oid not in self.names:
                started = time.monotonic()
                size = f"{o['width']} x {o['length']} mm, {o['height']} mm tall"
                answer = self.ask(prompts.NAME.format(size=size), images=[o["crop"]], max_new_tokens=24)
                self.names[oid] = answer.strip().strip('."').lower()
                self.say("naming", f"#{oid} is a {self.names[oid]}", started)

    def scene(self):
        lines = [prompts.OBJECT.format(oid=oid, name=self.names[oid], **{k: o[k] for k in ("width", "length", "height", "x", "y")})
                 for oid, o in sorted(self.objects.items())]
        held = self.robot.holding()
        return prompts.SCENE.format(objects="\n".join(lines) or "(none)",
                                    held=f"#{held}: {self.names.get(held, 'an object')}" if held is not None else "nothing")

    # ------------------------------------------------------------------ master
    def ask(self, text, **options):
        """One Gemma call; the full text that was sent is saved in the event log."""
        log = getattr(self.gemma, "log", None)
        if log is not None:
            log("prompt", text=text, system=options.get("system") or "", thinking=options.get("thinking", False))
        return self.gemma.ask(text, **options)

    def master(self, prompt):
        """Ask the master. Returns (kind, value, memory): kind is plan, ask, done or stop.

        The master is the only agent that reasons step by step (thinking): it plans, and re-plans after a failure
        or a user answer. Its thinking has a token limit (launch.sh); if that gives no usable answer it is asked
        once more to answer directly. The workers and the final check always answer directly."""
        error = ""
        for thinking in (True, False):
            started = time.monotonic()
            answer = self.ask(prompt + error, images=[self.rgb], system=prompts.MASTER, thinking=thinking, max_new_tokens=400)
            self.say("master (thinking)" if thinking else "master (direct)", answer.strip(), started)
            steps, decision, memory = [], None, ""
            for line in answer.splitlines():
                key, _, text = line.strip().lstrip("-* ").partition(":")
                key, text = key.strip().lower(), text.strip()
                if key in WORKERS and text:
                    steps.append((key, text))
                elif key in ("ask", "done", "stop") and text and decision is None:
                    decision = (key, text)
                elif key == "memory":
                    memory = text
            if steps:
                return "plan", steps, memory
            if decision:
                return decision[0], decision[1], memory
            error = prompts.REPAIR.format(error="no worker step, ask, done or stop line was found")
        return "stop", "the master did not give a usable answer", ""

    # ------------------------------------------------------------------ workers
    def perception(self, message):
        answer = self.ask(prompts.PERCEIVE.format(scene=self.scene(), message=message), images=[self.rgb], max_new_tokens=120)
        return not answer.upper().startswith(("UNSURE", "NOT FOUND")), answer.strip()

    def context(self, message):
        """What a manipulation call is told besides the scene: the request, what was done, what follows."""
        self.message = message
        return dict(scene=self.scene(), request=self.request, message=message, done="\n".join(self.reports) or "(none)",
                    later="\n".join(text for worker, text in self.later) or "(none)")

    def manipulation(self, message):
        started = time.monotonic()
        answer = self.ask(prompts.GROUND.format(**self.context(message)), images=[self.numbered], max_new_tokens=120).strip()
        self.say("manipulation worker", f"chose: {answer}", started)
        ids = [int(n) for n in re.findall(r"#(\d+)", answer)]
        words = answer.lower().split()
        if any(i not in self.objects for i in ids):
            return False, f"the worker named an object number that is not on the table: {answer}"
        if words[:1] == ["home"]:
            return self.robot.home()
        if words[:1] == ["pick"] and len(ids) == 2 and "place" in words:
            relation = words[words.index("place") + 1]
            ok, text = self.robot.transfer(ids[0], relation, ids[1])
            return ok, f"pick {self.names[ids[0]]}, place {relation} {self.names[ids[1]]}: {text}"
        if words[:1] == ["pick"] and len(ids) == 1:
            ok, text = self.robot.pick(ids[0])
            return ok, f"pick {self.names[ids[0]]}: {text}"
        return False, f"nothing was moved; the worker answered: {answer}"

    def spot(self, picture, letters, picked, target):
        """The robot calls this while it plans a placement on a tray or in a basket: the picture shows the free
        spots it measured as lettered dots. Returns the number of the spot the worker chose."""
        started = time.monotonic()
        o = self.objects[picked]
        prompt = prompts.PLACE.format(**self.context(self.message), picked=picked, name=self.names[picked],
                                      size=f"{o['width']} x {o['length']} mm, {o['height']} mm tall", target=target,
                                      target_name=self.names[target], letters=", ".join(letters))
        answer = self.ask(prompt, images=[picture], max_new_tokens=60).strip()
        letter = re.match(r"\W*([A-Za-z])\b", answer)
        known = bool(letter) and letter[1].upper() in letters
        self.say("manipulation worker", f"spot {answer}" + ("" if known else " (not a marked spot; the middle one is used)"), started)
        return letters.index(letter[1].upper()) if known else None

    def verification(self, message):
        self.look()  # a new table picture, taken with the arm out of the way
        prompt = prompts.CHECK.format(request=self.request, scene=self.scene(), message=message)
        answer = self.ask(prompt, images=[self.rgb], max_new_tokens=200).strip()
        return answer.lower().startswith("yes"), answer

    # ------------------------------------------------------------------ one instruction
    def run(self, request):
        """Returns (status, answer): COMPLETED, STOPPED or FAILED."""
        self.request, self.robot.moved = request, False
        self.look()
        reports, failures, memory, steps_done, decisions, questions, checked = [], set(), "", 0, 0, 0, False
        self.reports, self.later = reports, []  # the manipulation worker is told what was done and what follows
        kind, value, memory = self.master(prompts.MASTER_PLAN.format(request=request, scene=self.scene()))
        while True:
            if kind == "plan":
                steps = self.later = list(value)
                while steps:
                    worker, message = steps.pop(0)
                    steps_done += 1
                    if steps_done > MAX_STEPS:
                        return "STOPPED", f"the budget of {MAX_STEPS} steps is used up"
                    self.say("loop", f"step {steps_done} of {MAX_STEPS}, to the {worker} worker: {message}")
                    started = time.monotonic()
                    ok, text = getattr(self, worker)(message)
                    checked = worker == "verification" and ok and not steps
                    reports.append(f"{steps_done}. {worker}: {message} -> {'OK' if ok else 'FAILED'}: {text}")
                    self.say(f"{worker} worker", f"report {'OK' if ok else 'FAILED'}: {text}", started)
                    if worker == "manipulation":
                        self.look()  # the table may have changed, also when the step failed partway
                    if not ok:
                        reason = text.split(": ", 1)[-1]
                        if (worker, reason) in failures:
                            return ("FAILED" if self.robot.moved else "STOPPED"), f"the same failure happened twice: {text}"
                        failures.add((worker, reason))
                        break  # the master decides what to do about a failure
                if ok and not steps and self.robot.moved:  # the whole plan worked: straight to the final check
                    kind, value = "done", "every planned step was carried out"
                    continue
                value = steps
            elif kind == "ask":
                questions += 1
                if questions > MAX_QUESTIONS:
                    return "STOPPED", f"still unclear after {MAX_QUESTIONS} questions: {value}"
                reply = self.ask_user(value)
                request = self.request = f"{request}\nRobot asked: {value}\nUser answered: {reply}"
                value = []
            elif kind == "stop":
                return ("FAILED" if self.robot.moved else "STOPPED"), value
            elif kind == "done":
                if not self.robot.moved or checked:
                    return "COMPLETED", value
                started = time.monotonic()
                ok, text = self.verification(f"Is everything the user asked for done? The requested items: {memory or request}")
                self.say("verification worker", f"final check: {text}", started)
                if ok:
                    return "COMPLETED", value
                reports.append(f"final check -> FAILED: {text}")
                value = []
            decisions += 1
            if decisions > MAX_DECISIONS:
                return "STOPPED", f"the budget of {MAX_DECISIONS} decisions is used up"
            plan = "\n".join(f"{w}: {m}" for w, m in value) or "(none)"
            kind, value, new_memory = self.master(prompts.MASTER_DECIDE.format(
                request=request, scene=self.scene(), memory=memory or "(none)", plan=plan,
                reports="\n".join(reports) or "(none)",
                budget=f"{MAX_STEPS - steps_done} steps, {MAX_DECISIONS - decisions} decisions"))
            memory = new_memory or memory
