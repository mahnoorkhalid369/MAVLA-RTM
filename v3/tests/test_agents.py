import numpy as np, agents
class FakeRobot:
    def __init__(self, fail_first=False): self.moved, self.log, self.fail_first = False, [], fail_first
    def look(self):
        img = np.zeros((8, 8, 3), np.uint8)
        return img, img, {i: dict(width=60, length=60, height=100, x=10 * i, y=20, crop=img) for i in (1, 2, 3)}
    def holding(self): return None
    def transfer(self, a, rel, b, spot=None):
        self.moved = True; self.log.append(("transfer", a, rel, b) + ((spot,) if spot else ()))
        if self.fail_first: self.fail_first = False; return False, "no safe motion was found"
        return True, "released"
    def pick(self, a): self.moved = True; self.log.append(("pick", a)); return True, "held"
    def home(self): return True, "home"
class FakeGemma:
    def __init__(self, script): self.script, self.prompts = list(script), []
    def ask(self, text, images=(), system=None, thinking=False, max_new_tokens=0):
        self.prompts.append(text)
        if text.startswith("This is a close-up"): return ["milk carton", "water bottle", "blue tray"][sum(p.startswith("This is a close-up") for p in self.prompts) - 1]
        return self.script.pop(0)
def case(name, script, request, expect, answers=(), **kw):
    g, r, asked = FakeGemma(script), FakeRobot(**kw), []
    a = agents.Agents(g, r, lambda q: (asked.append(q), list(answers)[len(asked) - 1])[1], say=lambda *line: None)
    status, answer = a.run(request)
    ok = (status, r.log, len(asked)) == expect and not g.script
    print(("PASS" if ok else "FAIL"), name, "|", status, r.log, "questions", len(asked), "unused answers", len(g.script))
case("single transfer", ["manipulation: pick up the milk and place it in the tray\nmemory: milk -> tray: pending", "pick #1 place in #3",
     "yes: the milk carton is in the tray"],
     "put the milk in the tray", ("COMPLETED", [("transfer", 1, "in", 3)], 0))
case("two objects, one at a time", ["manipulation: pick up the milk and place it in the tray\nmanipulation: pick up the water bottle and place it in the tray\nmemory: x",
     "pick #1 place in #3", "pick #2 place in #3", "yes: both"],
     "put the milk and the water in the tray", ("COMPLETED", [("transfer", 1, "in", 3), ("transfer", 2, "in", 3)], 0))
case("ask at plan time, user cancels", ["ask: there is no apple on the table; which object do you mean?\nmemory: apple -> tray: pending", "stop: the user cancelled\nmemory: apple -> tray: failed"],
     "place the apple on the tray", ("STOPPED", [], 1), answers=["never mind"])
case("worker refuses to substitute", ["manipulation: pick up the apple and place it on the tray\nmemory: x", "cannot: there is no apple", "stop: no apple on the table\nmemory: x"],
     "place the apple on the tray", ("STOPPED", [], 0))
case("failure goes to the master, which re-plans", ["manipulation: pick up the milk and place it in the tray\nmemory: x", "pick #1 place in #3",
     "manipulation: pick up the milk and place it next to the tray\nmemory: x", "pick #1 place next_to #3", "yes: ok"],
     "put the milk in the tray", ("COMPLETED", [("transfer", 1, "in", 3), ("transfer", 1, "next_to", 3)], 0), fail_first=True)
case("plan misses the second object, final check sends it back", ["manipulation: pick up the milk and place it in the tray\nmemory: milk -> tray: pending; water -> tray: pending",
     "pick #1 place in #3", "no: the water bottle is not in the tray", "manipulation: pick up the water bottle and place it in the tray\nmemory: milk -> tray: done; water -> tray: pending",
     "pick #2 place in #3", "yes: both are in the tray"],
     "put the milk and the water in the tray", ("COMPLETED", [("transfer", 1, "in", 3), ("transfer", 2, "in", 3)], 0))
case("question only, no motion", ["perception: what objects are on the table?\nmemory: none", "OK: a milk carton, a water bottle and a tray", "done: a milk carton, a water bottle and a tray\nmemory: none"],
     "what is on the table?", ("COMPLETED", [], 0))
case("final check says no, master stops", ["manipulation: pick up the milk and place it in the tray\nmemory: x", "pick #1 place in #3",
     "no: the milk carton is beside the tray", "stop: the milk ended beside the tray\nmemory: x"], "put the milk in the tray", ("FAILED", [("transfer", 1, "in", 3)], 0))
case("invented object number", ["manipulation: pick up the milk and place it in the tray\nmemory: x", "pick #9 place in #3", "stop: unclear\nmemory: x"],
     "put the milk in the tray", ("STOPPED", [], 0))
class StuckRobot(FakeRobot):
    def transfer(self, a, rel, b, spot=None): self.log.append(("transfer", a, rel, b)); return False, "no safe motion was found: collision"
g, r = FakeGemma(["manipulation: pick up the milk and place it in the tray\nmemory: x", "pick #1 place in #3",
                  "manipulation: pick up the milk and place it on the tray\nmemory: x", "pick #1 place on #3"]), StuckRobot()
status, answer = agents.Agents(g, r, lambda q: "", say=lambda *line: None).run("put the milk in the tray")
print(("PASS" if status == "STOPPED" and len(r.log) == 2 and not g.script else "FAIL"), "same failure twice ends the request |", status, r.log)
g = FakeGemma(["C: beside the milk, leaves room for the bottle", "the middle"])
a = agents.Agents(g, FakeRobot(), lambda q: "", say=lambda *line: None)
a.request, a.reports, a.later, a.message = "put the milk and the water in the tray", [], [("manipulation", "water")], "milk"
a.look()
picks = [a.spot(a.rgb, ["A", "B", "C", "D"], 1, 3), a.spot(a.rgb, ["A", "B", "C", "D"], 1, 3)]
print(("PASS" if picks == [2, None] and "Free spots: A, B, C, D" in g.prompts[-1] else "FAIL"), "worker picks a lettered spot |", picks)
