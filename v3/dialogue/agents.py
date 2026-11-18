"""v3/dialogue: the master and its workers hold ONE conversation in plain language, and Gemma decides everything.

Every line of the conversation is "speaker -> addressee: text" and every agent reads all of it. The master plans and
says what happens next; the manipulation worker says which object and where, and points at both in the picture; the robot (arm.py)
answers with measured facts; the verification worker describes what it sees and the master judges.
Code keeps only: who a message is for, boxes drawn on the picture, measuring, safety, and a limit on turns.
"""

import re
import time

import prompts

MAX_TURNS, MAX_QUESTIONS, MAX_CORRECTIONS, MAX_SAME = 20, 3, 2, 2
OVERLAP = 0.15  # a box Gemma drew means a measured object when they share at least this much (0 apart, 1 identical)
MESSAGE = re.compile(r"^\W*(to manipulation|to perception|to verification|to user|done|stop)\s*:\s*(.+)", re.I | re.M | re.S)


class Agents:
    def __init__(self, gemma, robot, ask_user, say=lambda who, text, started=None: print(f"[{who}] {text}")):
        self.gemma, self.robot, self.ask_user, self.say = gemma, robot, ask_user, say
        self.names, self.talk = {}, []  # object id -> the name Gemma gave it (kept for the session); the conversation
        robot.names = self.names

    # ------------------------------------------------------------------ what everyone is told
    def look(self):
        started = time.monotonic()
        self.rgb, _, self.objects = self.robot.look()
        self.say("robot", f"new table picture: {len(self.objects)} objects outlined and measured", started)
        for oid, o in self.objects.items():
            if oid not in self.names:
                started = time.monotonic()
                answer = self.ask(prompts.NAME.format(size=self.size(oid)), images=[o["crop"]], max_new_tokens=24)
                self.names[oid] = answer.strip().strip('."').lower()
                self.say("naming", f"a {self.names[oid]}, {self.size(oid)}", started)

    def size(self, oid):
        o = self.objects[oid]
        return f"{o['width']} x {o['length']} mm, {o['height']} mm tall"

    def facts(self, positions=True):
        """The measured objects, one per line. The pointing worker gets them without positions: it has to look at
        the picture, and it writes any number it is given into its boxes."""
        lines = [f"- {self.names[oid]}: {self.size(oid)}" + (f", middle at x {o['x']:+d}, y {o['y']:+d} mm" if positions else "")
                 + (f", open on top, {o['hollow']} mm deep inside" if o.get("hollow", 0) > 10 else "")
                 for oid, o in sorted(self.objects.items())]
        held = self.robot.holding()
        return "\n".join(lines + [f"In the gripper: {self.names.get(held, 'an object') if held is not None else 'nothing'}"])

    def note(self, speaker, to, text, started=None):
        """One line of the conversation: kept for every later prompt, and logged."""
        self.talk.append(f"{speaker} -> {to}: {text}")
        self.say(f"{speaker} -> {to}", text, started)

    def ask(self, body, images, **options):
        blind = options.pop("blind", False)  # a checking worker reads what the user said, not what the team claims
        """One Gemma call with the facts and the whole conversation; the full text is saved in the event log."""
        text = body.format(facts=self.facts(), things=self.facts(False), talk="\n".join(t for t in self.talk if not blind or t.startswith("user -> master")), **options.pop("fill", {})) if "{talk}" in body else body
        text += options.pop("more", "")
        log = getattr(self.gemma, "log", None)
        if log is not None:
            log("prompt", text=text, system=options.get("system") or "", thinking=options.get("thinking", False))
        return self.gemma.ask(text, images=images, **options).strip()

    @staticmethod
    def line(answer, word):
        """What an answer says after "word:" on a line of its own, or None."""
        found = re.search(r"^\W*" + word + r"[ \t]*:[ \t]*(\S.*)$", answer, re.I | re.M)
        return found[1].strip() if found else None

    @staticmethod
    def box(answer, word):
        """The box written after a word in an answer, as fractions of the picture (x0, y0, x1, y1), or None."""
        found = re.search(word + r"\W*(\d+)\D+(\d+)\D+(\d+)\D+(\d+)", answer, re.I)
        if not found:
            return None
        y0, x0, y1, x1 = (min(int(v), 1000) / 1000 for v in found.groups())
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    # ------------------------------------------------------------------ master
    def master(self, think):
        """The master's next message: (addressee or done/stop, text, when it started). It reasons step by step
        (thinking) at the start, after a problem and after a user answer; otherwise it answers directly."""
        body, more = (prompts.MASTER_FIRST if len(self.talk) == 1 else prompts.MASTER_NEXT), ""
        for thinking in (True, False) if think else (False, False):
            started = time.monotonic()
            answer = self.ask(body, [self.rgb], more=more, system=prompts.MASTER, thinking=thinking, max_new_tokens=400)
            answer = re.sub(r"^(\W*)master\s*->\s*(?=(manipulation|perception|verification|user)\s*:)", r"\1to ", answer, flags=re.I | re.M)
            found = MESSAGE.search(answer)  # "master -> manipulation:" is how the conversation shows it; the model sometimes copies that
            if found:
                if answer[:found.start()].strip():
                    self.note("master", "all", answer[:found.start()].strip(), started)
                text = re.split(r"\n\W*(?:to \w+|done|stop)\s*:", found[2])[0].strip()
                return found[1].lower().replace("to ", ""), text, started
            plan = re.sub(r"\n\W*(to \w+|done|stop)\s*:\s*$", "", answer).strip()
            if body is prompts.MASTER_FIRST and plan:  # a plan without a message: keep the plan, ask for the message
                self.note("master", "all", plan, started)
                body = prompts.MASTER_NEXT
            else:
                self.say("master", f"(no message line) {answer}", started)
                more = prompts.REPAIR
        return "stop", "the master did not give a usable answer", time.monotonic()

    # ------------------------------------------------------------------ workers
    def point(self, prompt, picture):
        """Gemma draws a box in a picture. Returns (the box as fractions of the picture or None, its answer)."""
        started = time.monotonic()
        answer = self.ask(prompt, [picture], max_new_tokens=60)
        self.say("manipulation points", answer, started)
        return self.box(answer, "box"), answer

    def manipulation(self, message):
        """Moves one object. Returns False when the master has to think about what came back."""
        robot, names = self.robot, self.names
        h, w = self.rgb.shape[:2]
        started = time.monotonic()
        answer = self.ask(prompts.WHAT, [self.rgb], fill={"message": message}, max_new_tokens=120)
        pick, put, by = (self.line(answer, word) for word in ("pick up", "put down", "on the table by"))
        if pick is None:  # not the form: the worker is telling the master something
            self.note("manipulation", "master", answer, started)
            return False
        self.note("manipulation", "robot", answer, started)
        box, said = self.point(prompts.POINT.format(thing=pick), self.rgb)
        picked, overlap = robot.meant((box[0] * w, box[1] * h, box[2] * w, box[3] * h)) if box else (None, 0)
        if overlap < OVERLAP:
            self.note("robot", "master", f"Nothing was moved. The worker wanted to pick up '{pick}', but where it pointed "
                                         f"I have no single outlined and measured object ({said}).")
            return False
        fill = {"object": names[picked], "size": self.size(picked)}
        if put is None and by is None:
            self.note("robot", "manipulation", f"Understood: pick up the {names[picked]} ({self.size(picked)}) and hold it.")
            ok, text = robot.pick(picked)
            self.note("robot", "master", f"{names[picked]}: {text}")
            self.look()
            return ok
        box, said = self.point(prompts.POINT.format(thing=put or by), self.rgb)
        near, overlap = robot.meant((box[0] * w, box[1] * h, box[2] * w, box[3] * h), skip=picked) if box else (None, 0)
        if overlap < OVERLAP:
            self.note("robot", "master", f"Nothing was moved. The worker wanted to put the {names[picked]} down at '{put or by}', "
                                         f"but where it pointed I have no single outlined and measured object ({said}).")
            return False
        x, y, bw, bh = robot.boxes[near]
        if put:  # on or in a thing: v2 measures whether it is open on top
            support = near
            where = f"the {names[near]}, " + ("inside it" if self.objects[near].get("hollow", 0) > 10 else "on top of it")
        else:
            support, where = None, f"the free table by the {names[near]}"
        fill["where"] = where
        picture, view = robot.zoom((x, y, x + bw, y + bh), 0 if put else 110)
        self.note("robot", "manipulation", f"Understood: the {names[picked]} ({self.size(picked)}) goes to {where}. "
                                           "Your next picture is a close-up of that place.")
        plan, shown = None, picture
        for turn in range(MAX_CORRECTIONS + 1):
            started = time.monotonic()
            answer = " ".join(self.ask(prompts.CONFIRM if plan else prompts.WHERE, [shown], fill=fill, max_new_tokens=80).split())
            self.note("manipulation", "robot", answer, started)
            words = self.line(answer, "move it") if plan else answer
            if words is None:
                break  # the worker accepts the outlined place
            place, said = self.point(prompts.SPOT.format(words=words, **fill), picture)
            new, problem = (None, f"no box could be read in '{said}'") if place is None else robot.aim(
                picked, support, view[0] + (place[0] + place[2]) / 2 * view[2], view[1] + (place[1] + place[3]) / 2 * view[3])
            if new is None and plan is None and turn == 0:  # the worker hears why and chooses once more
                self.note("robot", "manipulation", f"That place is not possible: {problem}. Choose another place.")
                continue
            if new is None and plan is None:
                self.note("robot", "master", f"The {names[picked]} was not moved: {problem}")
                self.look()
                return False
            if new is None:
                self.note("robot", "manipulation", f"That is not possible ({problem}). I keep the outlined place.")
                break
            plan = new
            shown, off = robot.preview(plan, picture, view)
            self.note("robot", "manipulation",
                      f"The white outline in your next picture is where the {names[picked]} can stand: {off} mm from the spot you pointed at"
                      + (f". That is the nearest place where it fits and the arm can put it down safely: {robot.why_moved()}." if off > 25 else "."))
        ok, text = robot.carry_out(plan)
        self.note("robot", "master", f"{names[picked]} to {where}: {text}")
        self.look()  # the table has changed, also when the move failed partway
        if ok:  # act, look, report: an independent look at the place after every release
            self.places += [near] if near not in self.places else []
            self.describe(f"Where is the {names[picked]} now, and what stands in or on the {names[near]}?")
        return ok

    def perception(self, message):
        started = time.monotonic()
        self.note("perception", "master", self.ask(prompts.PERCEIVE, [self.rgb], fill={"message": message}, max_new_tokens=150, blind=True), started)
        return True

    def verification(self, message):
        started = time.monotonic()
        self.look()  # a new table picture, taken with the arm out of the way
        self.describe(message, started)
        return True

    def describe(self, message, started=None):
        """The verification worker looks at a plain close-up of the places the robot worked at in this request (the
        whole table when it has not put anything down yet) and is not told what the team believes happened."""
        started = started or time.monotonic()
        boxes = [self.robot.boxes[oid] for oid in self.places if oid in self.robot.boxes]
        picture = self.rgb
        if boxes:
            picture = self.robot.zoom((min(b[0] for b in boxes), min(b[1] for b in boxes),
                                       max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)), 60)[0]
        self.note("verification", "master", self.ask(prompts.VERIFY, [picture], fill={"message": message}, max_new_tokens=200, blind=True), started)

    # ------------------------------------------------------------------ one instruction
    def run(self, request):
        """Returns (status, answer): COMPLETED, STOPPED or FAILED."""
        self.robot.moved, self.talk, self.places = False, [], []
        self.note("user", "master", request)
        self.look()
        think, questions = True, 0
        for turn in range(MAX_TURNS):
            to, text, started = self.master(think)
            if to in ("done", "stop"):
                self.note("master", "user", text, started)
                return ("COMPLETED" if to == "done" else "FAILED" if self.robot.moved else "STOPPED"), text
            self.note("master", to, text, started)
            if to == "user":
                questions += 1
                if questions > MAX_QUESTIONS:
                    return "STOPPED", f"still unclear after {MAX_QUESTIONS} questions: {text}"
                started = time.monotonic()
                self.note("user", "master", self.ask_user(text), started)
                think = True
            else:
                think = not getattr(self, to)(text)
                if think and self.talk[-1] in self.talk[:-1] and self.talk.count(self.talk[-1]) > MAX_SAME:
                    return ("FAILED" if self.robot.moved else "STOPPED"), f"the same problem came back {MAX_SAME + 1} times: {self.talk[-1]}"
        return ("FAILED" if self.robot.moved else "STOPPED"), f"the limit of {MAX_TURNS} master turns is used up"
