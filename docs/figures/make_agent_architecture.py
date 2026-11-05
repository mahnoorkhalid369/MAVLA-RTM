"""Build docs/figures/agent_architecture.svg (paper-style system figure) from the repo's own screenshots."""
import base64, io, sys
from html import escape
from PIL import Image

ROOT = sys.argv[1]
OUT = f"{ROOT}/docs/figures/agent_architecture.svg"


def b64(path, box, size):
    im = Image.open(path).convert("RGB").crop(box).resize(size, Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


scene = b64(f"{ROOT}/out/tasks/table_20261007_181229.png", (250, 70, 1090, 590), (680, 420))
wrist = b64(f"{ROOT}/out/observe_view.png", (0, 0, 640, 360), (320, 180))

BLUE, PALE, GREEN, PGREEN, INK, MUTE, CODE = "#2a6db0", "#e3f1fb", "#3ea33a", "#dff3d8", "#1b1f24", "#5b6470", "#f7fafc"
parts = []
A = parts.append


def box(x, y, w, h, fill=PALE, stroke="none", sw=0, r=16, extra=""):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def text(x, y, s, size=15, weight=400, fill=INK, anchor="start", family="Inter, Helvetica, Arial, sans-serif", style=""):
    A(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
      f'font-family="{family}" {style}>{s}</text>')


def mono(x, y, lines, size=13, lh=19, fill=INK):
    for i, ln in enumerate(lines):
        text(x, y + i * lh, ln, size=size, fill=fill, family="'JetBrains Mono', Menlo, Consolas, monospace",
             style='xml:space="preserve"')


def arrow(d, color=BLUE, w=3, head="ah"):
    A(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" marker-end="url(#{head})"/>')


def label(x, y, s, size=13, fill=MUTE, anchor="middle", rot=None, weight=500):
    t = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    A(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
      f'font-family="Inter, Helvetica, Arial, sans-serif"{t}>{s}</text>')


W, H = 1660, 1060
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
A(f'''<defs>
<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>
<marker id="ag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>
<filter id="sh" x="-5%" y="-5%" width="110%" height="115%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12"/></filter>
<clipPath id="clipScene"><rect x="48" y="98" width="370" height="229" rx="10"/></clipPath>
</defs>''')
box(0, 0, W, H, fill="#ffffff", r=0)

# ---------- Environment ----------
box(30, 60, 406, 290, fill="#ececec")
text(233, 88, "Environment · Isaac Sim 6.1", 19, 600, anchor="middle")
A(f'<image href="{scene}" x="48" y="98" width="370" height="229" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipScene)"/>')
A(f'<rect x="276" y="236" width="132" height="76" fill="#000" rx="4"/>')
A(f'<image href="{wrist}" x="278" y="238" width="128" height="72" preserveAspectRatio="xMidYMid slice"/>')
A('<rect x="276" y="216" width="132" height="20" fill="#000" rx="4"/>')
label(342, 230, "wrist D435 (RGB-D)", 11, "#ffffff", weight=600)
A(f'<rect x="58" y="300" width="208" height="20" rx="4" fill="#ffffffd0"/>')
label(162, 314, "VT6L arm · Robotiq 2F-85 · table camera", 11, INK, weight=500)

# ---------- Actions + executor (top) ----------
box(480, 20, 160, 48, fill=PALE)
text(560, 50, "Motion", 18, 600, anchor="middle")
box(700, 12, 370, 66, fill="#eef0f2")
text(885, 38, "Collision-checked planner → RosExecutor", 16, 600, anchor="middle")
text(885, 60, "ROS 2 Jazzy · simulator safety layer (independent)", 13, 400, MUTE, anchor="middle")
arrow("M700,44 L644,44")
arrow("M480,44 L232,44 L232,56")

# ---------- Measured scene state ----------
box(480, 104, 360, 230, extra='filter="url(#sh)"')
text(500, 132, "Measured scene state", 17, 600)
box(496, 144, 328, 176, fill="#ffffff", r=8)
mono(510, 168, ["Object #10   (YOLO mask + depth)",
                "  center (m):  ...    size:  ...",
                "  top z:  ...         score: ...",
                "  label:  red cube   (Gemma, hypothesis)",
                "Object #12",
                "  center (m):  ...    size:  ...",
                "  · · · · · ·",
                "Robot: joints · fingers · held: none"], 12.5, 19)
arrow("M436,205 L476,205")
label(456, 196, "RGB-D", 11)

# ---------- Task description ----------
box(40, 430, 360, 118, fill=PALE)
text(220, 472, "Task (text instruction):", 20, 600, anchor="middle")
text(220, 512, "“put the red cube in the tray”", 21, 500, "#d6332b", anchor="middle")
arrow("M400,490 L536,490")

# ---------- Supervisor ----------
A(f'<rect x="540" y="388" width="560" height="282" rx="20" fill="{PGREEN}" stroke="{GREEN}" stroke-width="5" filter="url(#sh)"/>')
text(820, 422, "Supervisor  (LangGraph)", 22, 700, anchor="middle")
box(566, 436, 508, 160, fill="#ffffff", stroke="#222", sw=1.2, r=8)
text(820, 460, "TASK MEMORY", 15, 700, anchor="middle")
mono(582, 484, ["Original goal:  fixed for the whole task",
                "  success:  in(red cube, tray)",
                "Subtasks:  worker · result · evidence",
                "Facts (measured) | hypotheses (Gemma)",
                "Remaining:  unmet criteria  ·  failures"], 12.5, 21)
mono(560, 618, ["INTERPRETING → DISPATCHING → DELEGATING → REVIEWING", "        → DECIDING → … → VERIFYING_GOAL → COMPLETED"],
     12, 17, "#245d22")
text(820, 660, "deterministic: interfaces · authorization · safety stops · budget", 12.5, 500, MUTE, anchor="middle")
arrow("M660,334 L660,384")

# ---------- Gemma ----------
A(f'<rect x="30" y="720" width="480" height="318" rx="20" fill="{PGREEN}" stroke="{GREEN}" stroke-width="5" filter="url(#sh)"/>')
text(270, 754, "Gemma 4  (VLM roles)", 22, 700, anchor="middle")
box(52, 768, 436, 252, fill="#ffffff", stroke="#222", sw=1.2, r=8)
text(270, 792, "SYSTEM PROMPT", 15, 700, anchor="middle")
rows = [("Plan:", 600), ("  Plan your own actions for the user's request.", 400),
        ("  First state the goal: what must be TRUE at the end,", 400),
        ("  as measurable success criteria.", 400),
        ("Workers and operations:", 600), ("  {menu}", 400),
        ("Decide (after every worker report):", 600),
        ("  continue · replan · ask_user · finish · stop", 400),
        ("Other roles:", 600), ("  classify · describe · resolve · answer · confirm", 400)]
for i, (s, wgt) in enumerate(rows):
    text(68, 818 + i * 20, escape(s), 13, wgt, style='xml:space="preserve" font-style="italic"' if wgt == 600 else 'xml:space="preserve"')

# ---------- Goal plan ----------
box(560, 740, 470, 290)
text(580, 770, "Goal plan  (strict JSON schema)", 17, 600)
mono(580, 800, ['GOAL     the red cube is in the tray',
                'SUCCESS  in(red cube, tray)', '',
                '1. {"do": "observe"}            perception',
                '2. {"do": "locate",',
                '    "object": "red cube"}        perception',
                '3. {"do": "pick_place",         manipulation',
                '    "object": "red cube",',
                '    "relation": "in",',
                '    "target": "tray"}',
                '4. {"do": "return_home"}        manipulation'], 12.5, 20)

# ---------- WorkerResult ----------
box(1170, 400, 450, 210)
text(1190, 430, "WorkerResult", 17, 600)
box(1186, 442, 418, 152, fill=CODE, r=8)
mono(1200, 466, ["status:       completed",
                 "outputs:      grasped #10 · placed in #12",
                 "evidence:     in(#10, #12) from depth",
                 "uncertainty:  …",
                 "needs:        (advisory only)",
                 "→ subtask verified, then goal re-checked"], 12.5, 21)
arrow("M1170,505 L1104,505")

# ---------- Workers ----------
A(f'<rect x="1090" y="720" width="540" height="318" rx="20" fill="{PGREEN}" stroke="{GREEN}" stroke-width="5" filter="url(#sh)"/>')
text(1360, 754, "Workers", 22, 700, anchor="middle")
box(1112, 768, 496, 252, fill="#ffffff", stroke="#222", sw=1.2, r=8)
wk = [("perception", "observe · describe_scene · answer · locate"),
      ("inspection", "inspect · explore · look  (wrist camera)"),
      ("manipulation", "pick · place · pick_place · return_home"),
      ("planning", "decompose"),
      ("verification", "verify  (measured predicates)")]
for i, (n, ops) in enumerate(wk):
    y = 800 + i * 36
    text(1128, y, n, 14, 700)
    text(1250, y, ops, 13.5, 400, "#30363d")
text(1360, 994, "trusted tools only · never start other workers", 12.5, 500, MUTE, anchor="middle")
text(1360, 1012, "· never declare the task complete", 12.5, 500, MUTE, anchor="middle")
arrow("M1360,720 L1360,614")

# ---------- thick green control arrows ----------
arrow("M600,672 L420,716", GREEN, 9, "ag")
label(470, 682, "plan / decide", 14, "#245d22", rot=-14, weight=600)
arrow("M1040,672 L1200,716", GREEN, 9, "ag")
label(1180, 690, "delegate WorkerRequest", 14, "#245d22", rot=15, weight=600)
arrow("M510,880 L556,880")
arrow("M795,740 L795,674")
label(803, 708, "validated · 1 retry with the error", 12, MUTE, anchor="start")
# manipulation / inspection motions go up the right side to the executor
arrow("M1630,840 L1645,840 L1645,44 L1074,44", GREEN, 7, "ag")
label(1636, 300, "motion steps (manipulation · inspection)", 14, "#245d22", rot=-90, weight=600)
arrow("M1070,72 L1170,72 L1170,140 L1384,140 L1384,396", BLUE, 3)
label(1278, 132, "execution result · joint / finger feedback", 12)

A("</svg>")
open(OUT, "w").write("\n".join(parts))
print(OUT)
