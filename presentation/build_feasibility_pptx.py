#!/usr/bin/env python3
"""Generate the StatementGuard "Feasibility & Implementation Plan" PowerPoint.

Pure python-pptx, no network. 16:9, dark theme matching the project decks.
Content is grounded in PROJECT_PLAN.md (build/buy/hybrid, the 6 phases, risks,
success metrics) and the codebase-grounded improvements review.

Usage:
    PYTHONPATH=/projects/sandbox/.pylibs python3 build_feasibility_pptx.py
Output:
    statementguard-feasibility-and-plan.pptx  (next to this script)
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- palette ----
BG     = RGBColor(0x0D, 0x11, 0x17)
PANEL  = RGBColor(0x16, 0x1B, 0x22)
ACCENT = RGBColor(0x4F, 0x9C, 0xFF)
GREEN  = RGBColor(0x3F, 0xB9, 0x50)
AMBER  = RGBColor(0xD2, 0x99, 0x22)
RED    = RGBColor(0xF8, 0x51, 0x49)
MUTED  = RGBColor(0x8B, 0x94, 0x9E)
LINE   = RGBColor(0x30, 0x36, 0x3D)
WHITE  = RGBColor(0xF0, 0xF3, 0xF7)
TEXT   = RGBColor(0xC9, 0xD1, 0xD9)

FONT = "Calibri"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    r.fill.solid(); r.fill.fore_color.rgb = BG; r.line.fill.background()
    r.shadow.inherit = False
    sp = r._element; sp.getparent().remove(sp); s.shapes._spTree.insert(2, sp)
    return s


def box(s, x, y, w, h, fill=None, line=None, line_w=0.75, radius=False):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None: shp.line.fill.background()
    else: shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def R(txt, size=14, color=TEXT, bold=False, italic=False, font=FONT):
    return (txt, size, color, bold, italic, font)


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.0, wrap=True):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.line_spacing = line_spacing
        for (txt, size, color, bold, italic, font) in para:
            rr = p.add_run(); rr.text = txt
            rr.font.size = Pt(size); rr.font.color.rgb = color
            rr.font.bold = bold; rr.font.italic = italic; rr.font.name = font
    return tb


def kicker(s, txt):
    text(s, Inches(0.6), Inches(0.35), Inches(11.5), Inches(0.4), [[R(txt.upper(), 11, MUTED, True)]])


def heading(s, txt, size=28, y=0.55, color=WHITE):
    text(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.9), [[R(txt, size, color, True)]])
    box(s, Inches(0.62), Inches(y + size/72.0 + 0.12), Inches(0.9), Pt(3), fill=ACCENT)


def pill(s, x, y, txt, color=ACCENT, w=None):
    ww = w or Inches(0.15 + 0.095 * len(txt))
    box(s, x, y, ww, Inches(0.32), fill=PANEL, line=color, line_w=1, radius=True)
    text(s, x, y - Inches(0.01), ww, Inches(0.34), [[R(txt, 10, color, True)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return ww


def card(s, x, y, w, h, title=None, title_color=WHITE, body=None, body_size=12,
         pill_txt=None, pill_color=ACCENT, accent=None):
    box(s, x, y, w, h, fill=PANEL, line=accent or LINE, line_w=1.25, radius=True)
    pad = Inches(0.18); cy = y + pad
    if pill_txt:
        pill(s, x + pad, cy, pill_txt, pill_color); cy += Inches(0.42)
    if title:
        text(s, x + pad, cy, w - 2*pad, Inches(0.5), [[R(title, 14, title_color, True)]])
        cy += Inches(0.44)
    if body:
        paras = body if isinstance(body, list) else [body]
        runs = [[R(p, body_size, TEXT, False)] for p in paras]
        text(s, x + pad, cy, w - 2*pad, y + h - cy - pad, runs, space_after=3, line_spacing=1.05)


def footer(s, txt):
    text(s, Inches(0.6), Inches(7.08), Inches(12), Inches(0.33), [[R(txt, 8, MUTED, False)]])


def stat(s, x, y, w, h, num, label, col=ACCENT):
    box(s, x, y, w, h, fill=PANEL, line=LINE, line_w=1, radius=True)
    text(s, x, y + Inches(0.18), w, Inches(0.8), [[R(num, 30, col, True)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, y + h - Inches(0.95), w, Inches(0.9), [[R(label, 10.5, MUTED, False)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)


# ============================ SLIDES ============================

# 1 — Title
s = slide()
kicker(s, "StatementGuard · Business & Delivery")
text(s, Inches(0.6), Inches(2.4), Inches(12.1), Inches(1.6),
     [[R("Feasibility &", 40, WHITE, True)], [R("Implementation Plan", 40, WHITE, True)]],
     line_spacing=1.0)
box(s, Inches(0.62), Inches(4.05), Inches(1.1), Pt(4), fill=ACCENT)
text(s, Inches(0.6), Inches(4.3), Inches(12), Inches(0.8),
     [[R("Can we build it, should we, and how do we get there safely — "
         "a staged path from working prototype to production.", 16, TEXT, False)]])
footer(s, "Grounded in PROJECT_PLAN.md + a direct codebase review · github.com/elline-c/AI-in-finance-")

# 2 — Is it feasible? verdict
s = slide()
heading(s, "Is it feasible? — Yes, with staged investment")
card(s, Inches(0.6), Inches(1.75), Inches(3.9), Inches(2.25), title="Technical", title_color=GREEN,
     pill_txt="PROVEN", pill_color=GREEN, accent=GREEN,
     body=["The hard part — the decision engine (4 agents + scoring + routing + evidence + signed "
           "audit) — already runs end-to-end. Feasibility is demonstrated, not theoretical."],
     body_size=11.5)
card(s, Inches(4.72), Inches(1.75), Inches(3.9), Inches(2.25), title="Operational", title_color=AMBER,
     pill_txt="ACHIEVABLE", pill_color=AMBER, accent=AMBER,
     body=["Remaining work is known and bounded: real OCR, real image forensics, a reviewer UI, and "
           "production hardening (auth, DB, ops). All are standard engineering, not research bets."],
     body_size=11.5)
card(s, Inches(8.84), Inches(1.75), Inches(3.9), Inches(2.25), title="Regulatory", title_color=ACCENT,
     pill_txt="ALIGNED", pill_color=ACCENT, accent=ACCENT,
     body=["Built to MAS FEAT from day one: explainable evidence, human owns every decision, signed "
           "audit trail. No autonomous reject. De-risks compliance sign-off."],
     body_size=11.5)
text(s, Inches(0.6), Inches(4.35), Inches(12.1), Inches(1.0),
     [[R("Bottom line:  ", 15, WHITE, True),
       R("the riskiest question — 'does the detection logic work and can it be explained?' — is already "
         "answered. What remains is integration and hardening on a de-risked path.", 15, TEXT, False)]],
     line_spacing=1.1)

# 3 — Feasibility dimensions table
s = slide()
heading(s, "Feasibility across every dimension")
rows = [
    ("Technical", "Decision engine complete; real OCR/forensics are integrations behind interfaces", "HIGH", GREEN),
    ("Economic", "Hybrid (buy OCR+forensics, build the policy layer) avoids rebuilding commodity parts", "HIGH", GREEN),
    ("Schedule", "~4-6 months of staged phases, each shipping something usable and de-risking the next", "MEDIUM", AMBER),
    ("Operational", "Needs auth, a real DB, CI/Docker and monitoring before live — known work, not unknowns", "MEDIUM", AMBER),
    ("Legal / compliance", "FEAT-aligned by design; PDPA handling + retention still to formalise for real PII", "MEDIUM", AMBER),
    ("Data", "Biggest lead-time item: a labelled eval set (genuine + known-tampered) must be assembled", "WATCH", RED),
]
# header
text(s, Inches(0.9), Inches(1.6), Inches(2.6), Inches(0.4), [[R("Dimension", 12, ACCENT, True)]])
text(s, Inches(3.6), Inches(1.6), Inches(7.3), Inches(0.4), [[R("Assessment", 12, ACCENT, True)]])
text(s, Inches(11.3), Inches(1.6), Inches(1.6), Inches(0.4), [[R("Rating", 12, ACCENT, True)]])
box(s, Inches(0.6), Inches(1.98), Inches(12.1), Pt(1.5), fill=LINE)
y = 2.1
for dim, note, rating, col in rows:
    box(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.72), fill=PANEL, line=LINE, line_w=0.75, radius=True)
    box(s, Inches(0.6), Inches(y), Inches(0.1), Inches(0.72), fill=col)
    text(s, Inches(0.9), Inches(y), Inches(2.6), Inches(0.72), [[R(dim, 12.5, WHITE, True)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(3.6), Inches(y), Inches(7.5), Inches(0.72), [[R(note, 11, TEXT, False)]], anchor=MSO_ANCHOR.MIDDLE)
    pill(s, Inches(11.3), Inches(y + 0.2), rating, col)
    y += 0.82

# 4 — Build vs buy vs hybrid
s = slide()
heading(s, "Build vs. buy vs. hybrid")
opts = [
    ("BUY a forensics API", AMBER, "Fast, battle-tested, maintained.",
     "Recurring cost; less control; data may leave premises.", "Wins on speed, small team."),
    ("BUILD everything", RED, "Full control, own the IP, data stays in-house.",
     "Slow; needs ML + forensics expertise.", "Wins on differentiation / data-residency."),
    ("HYBRID  (recommended)", GREEN, "Buy OCR + pixel forensics; build the orchestration, "
     "financial logic, scoring, routing & audit.", "Some integration work.",
     "Wins for most teams — you own the parts that encode your credit policy."),
]
x = 0.6
w = 3.97
for i, (name, col, pro, con, when) in enumerate(opts):
    h = 3.3 if i != 2 else 3.3
    box(s, Inches(x), Inches(1.75), Inches(w), Inches(h), fill=PANEL,
        line=col, line_w=(2 if i == 2 else 1), radius=True)
    text(s, Inches(x+0.2), Inches(1.95), Inches(w-0.4), Inches(0.5), [[R(name, 14, col, True)]])
    text(s, Inches(x+0.2), Inches(2.55), Inches(w-0.4), Inches(0.9),
         [[R("+ ", 12, GREEN, True), R(pro, 11.5, TEXT, False)]], line_spacing=1.05)
    text(s, Inches(x+0.2), Inches(3.55), Inches(w-0.4), Inches(0.9),
         [[R("- ", 12, RED, True), R(con, 11.5, TEXT, False)]], line_spacing=1.05)
    text(s, Inches(x+0.2), Inches(4.5), Inches(w-0.4), Inches(0.8),
         [[R(when, 11, MUTED, True)]], line_spacing=1.05)
    x += 4.12
text(s, Inches(0.6), Inches(5.35), Inches(12.1), Inches(1.4),
     [[R("Recommendation: start HYBRID.  ", 14, GREEN, True),
       R("The prototype already IS the orchestration + financial-logic + scoring + audit layer we'd own. "
         "Treat OCR and pixel forensics as pluggable providers behind an interface so a vendor can later be "
         "swapped for an in-house model without touching the pipeline.", 13, TEXT, False)]],
     line_spacing=1.1)
footer(s, "Source: PROJECT_PLAN.md §2 (build/buy/hybrid)")

# 5 — Where we are now (baseline)
s = slide()
heading(s, "Starting point: a real head start")
card(s, Inches(0.6), Inches(1.7), Inches(5.95), Inches(3.5), pill_txt="ALREADY BUILT", pill_color=GREEN,
     accent=GREEN,
     body=["• Full 4-agent pipeline, end-to-end",
           "• Risk + confidence scoring, routing, corroboration",
           "• Readable, reason-coded evidence on every flag",
           "• Three role-separated apps (customer / analyst / banker)",
           "• Cross-case memory + two ML nodes (fail-safe)",
           "• Signed, hash-chained audit trail + verifier",
           "• 71 test modules; 6-layout statement parser"], body_size=12.5)
card(s, Inches(6.78), Inches(1.7), Inches(5.95), Inches(3.5), pill_txt="STILL TO BUILD", pill_color=AMBER,
     accent=AMBER,
     body=["• Real PDF + OCR ingestion (input is typed data today)",
           "• Real pixel/layer image forensics",
           "• Authentication, RBAC & sessions on the apps",
           "• Production database (replace the JSON store)",
           "• Historical cash-flow baselines from real data",
           "• CI, containers, health checks, monitoring",
           "• Shadow-mode pilot + controlled rollout"], body_size=12.5)
text(s, Inches(0.6), Inches(5.45), Inches(12.1), Inches(0.9),
     [[R("The brain is complete; real-world inputs, access control, and production hardening are what remain.",
         14, WHITE, True)]], align=PP_ALIGN.CENTER)

# 6 — The 6 phases (implementation plan)
s = slide()
heading(s, "Implementation plan — 6 staged phases")
phases = [
    ("0", "Foundations", "1-2 wks", "Decide build/buy; assemble the labelled eval set; set success targets.", GREEN),
    ("1", "Harden the core", "2-3 wks", "Tests, versioned input schema, config-driven thresholds, CI, auth.", GREEN),
    ("2", "Real ingestion", "3-4 wks", "OCR + PDF/metadata extraction mapped to the NormalizedStatement contract.", ACCENT),
    ("3", "Real forensics", "3-5 wks", "Pixel/layer/compression analysis; keep cheap explainable heuristics.", ACCENT),
    ("4", "Analyst UI + HITL", "3-4 wks", "Queue, evidence side-by-side, approve/reject + reason, feedback loop.", AMBER),
    ("5", "Pilot & hardening", "3-4 wks", "LOS integration, shadow mode, security review, load test, runbooks.", AMBER),
    ("6", "Controlled rollout", "ongoing", "Auto-clear above a proven bar; monitor drift; periodic FEAT review.", AMBER),
]
y = 1.72
rh = 0.68
for num, name, dur, desc, col in phases:
    box(s, Inches(0.6), Inches(y), Inches(12.1), Inches(rh), fill=PANEL, line=LINE, line_w=0.75, radius=True)
    box(s, Inches(0.6), Inches(y), Inches(0.62), Inches(rh), fill=col, radius=False)
    text(s, Inches(0.6), Inches(y), Inches(0.62), Inches(rh), [[R(num, 18, BG, True)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(1.4), Inches(y), Inches(2.5), Inches(rh), [[R(name, 13, WHITE, True)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(3.95), Inches(y), Inches(1.15), Inches(rh), [[R(dur, 11, col, True)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.2), Inches(y), Inches(7.4), Inches(rh), [[R(desc, 11, TEXT, False)]], anchor=MSO_ANCHOR.MIDDLE)
    y += rh + 0.1
footer(s, "Each phase ships something usable and de-risks the next · estimates from PROJECT_PLAN.md §4a")

# 7 — Timeline / roadmap bar
s = slide()
heading(s, "Indicative timeline (~4-6 months to pilot)")
# month axis
left = 0.6; total_w = 12.1; months = 6
text(s, Inches(left), Inches(1.65), Inches(total_w), Inches(0.35),
     [[R("Month   1            2            3            4            5            6         ->  ongoing",
         11, MUTED, True, False, MONO)]])
bars = [
    ("Phase 0-1  Foundations + harden core", 0.0, 1.3, GREEN),
    ("Phase 2    Real ingestion (OCR)", 1.1, 1.4, ACCENT),
    ("Phase 3    Real forensics", 2.3, 1.7, ACCENT),
    ("Phase 4    Analyst UI + human-in-the-loop", 3.4, 1.3, AMBER),
    ("Phase 5    Integration, shadow pilot, hardening", 4.5, 1.5, AMBER),
    ("Phase 6    Controlled rollout", 5.8, 0.4, RED),
]
unit = total_w / months
y = 2.15
for label, start, dur, col in bars:
    text(s, Inches(left), Inches(y - 0.02), Inches(4.0), Inches(0.4), [[R(label, 10.5, TEXT, False)]],
         anchor=MSO_ANCHOR.MIDDLE)
    bx = left + 0.0 + start * unit
    # place bars under the axis region, offset to the right of labels row by drawing full-width track
    box(s, Inches(left + start*unit*0 + 0.0 + 0.0), Inches(y+0.38), Inches(0.001), Inches(0.001), fill=BG)
    y += 0.0
    # draw bar on its own track line
    track_y = y + 0.42
    box(s, Inches(left + start*unit), Inches(track_y), Inches(max(dur*unit,0.4)), Inches(0.34),
        fill=col, radius=True)
    y = track_y + 0.52
text(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(0.9),
     [[R("Phases overlap where safe (ingestion and forensics data-science can run parallel to UI work). "
         "Rollout is deliberately gradual — auto-clear only above a confidence bar proven in the pilot.",
         12.5, MUTED, False)]], line_spacing=1.1)
footer(s, "Illustrative — actuals depend on vendor choice and eval-set availability")

# 8 — Resources & dependencies
s = slide()
heading(s, "What it takes — resources & dependencies")
card(s, Inches(0.6), Inches(1.7), Inches(3.9), Inches(3.5), title="Team (parallel workstreams)",
     body=["• Engineering — pipeline, integrations, UI, infra",
           "• Data science / forensics — OCR + image models, threshold calibration, drift",
           "• Risk & compliance — FEAT, audit, retention, sign-off gates",
           "• Ops / credit — analyst workflow, training, feedback discipline"], body_size=11.5)
card(s, Inches(4.72), Inches(1.7), Inches(3.9), Inches(3.5), title="Key decisions (up front)",
     body=["• Build / buy / hybrid + vendor shortlist",
           "• Data-residency constraint (must data stay on our infra?)",
           "• Budget: forensics vendor vs. in-house ML headcount",
           "• Success-metric targets (so 'done' is measurable)"], body_size=11.5)
card(s, Inches(8.84), Inches(1.7), Inches(3.9), Inches(3.5), title="Critical dependency",
     pill_txt="LONGEST LEAD TIME", pill_color=RED, accent=RED,
     body=["A labelled evaluation set — genuine statements + known-tampered examples across banks "
           "and formats.",
           "It is the single most valuable asset; everything is measured against it. Start assembling "
           "it in week one."], body_size=11.5)
text(s, Inches(0.6), Inches(5.45), Inches(12.1), Inches(0.9),
     [[R("You don't need real customer data to start — ", 13, WHITE, True),
       R("synthetic clean + deliberately tampered statements are the right way to begin and can be created today.",
         13, TEXT, False)]], align=PP_ALIGN.CENTER)

# 9 — Risks & mitigations
s = slide()
heading(s, "Risks & how we manage them")
risks = [
    ("Too few tampered examples to evaluate", "Curate + synthesise controlled tampering; partner with the fraud team."),
    ("Overfitting to today's forgery tricks", "Keep rules explainable; scheduled drift review; refresh the eval set."),
    ("False positives frustrate analysts", "Start conservative; tune thresholds from real overrides; always show evidence."),
    ("Vendor lock-in / data residency", "Providers sit behind interfaces; keep orchestration + policy in-house."),
    ("Bias / fairness (FEAT)", "Test flag rates across segments; no protected attributes in scoring."),
    ("Sensitive-data exposure", "Encrypt at rest/in transit; least-privilege access; retention limits; audit log."),
]
y = 1.7
for risk, mit in risks:
    box(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.78), fill=PANEL, line=LINE, line_w=0.75, radius=True)
    text(s, Inches(0.85), Inches(y + 0.06), Inches(4.5), Inches(0.66),
         [[R("! ", 12, AMBER, True), R(risk, 12, WHITE, True)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.5), Inches(y + 0.06), Inches(7.0), Inches(0.66),
         [[R(mit, 11.5, TEXT, False)]], anchor=MSO_ANCHOR.MIDDLE)
    y += 0.88
footer(s, "Source: PROJECT_PLAN.md §7 (risks & mitigations)")

# 10 — Success metrics
s = slide()
heading(s, "How we'll know it's working")
metrics = [
    ("Recall on tampered docs", "The catch rate — the metric that protects the bank.", GREEN),
    ("Auto-clear precision", "Protects analyst time & customers from false alarms.", ACCENT),
    ("% auto-cleared", "The efficiency win — only counts if precision holds.", ACCENT),
    ("Time-to-decision", "Versus the current manual baseline.", AMBER),
    ("Analyst override rate", "How often humans disagree with the band (calibration).", AMBER),
    ("Explainability", "% of flags an analyst rates 'clear evidence' (FEAT).", GREEN),
]
x0, y0, w, h, gx, gy = 0.6, 1.75, 3.9, 1.55, 0.2, 0.25
for i, (t, d, col) in enumerate(metrics):
    r, c = divmod(i, 3)
    x = x0 + c * (w + gx); y = y0 + r * (h + gy)
    box(s, Inches(x), Inches(y), Inches(w), Inches(h), fill=PANEL, line=LINE, line_w=1, radius=True)
    box(s, Inches(x), Inches(y), Inches(0.1), Inches(h), fill=col)
    text(s, Inches(x+0.22), Inches(y+0.14), Inches(w-0.35), Inches(0.5), [[R(t, 13, WHITE, True)]])
    text(s, Inches(x+0.22), Inches(y+0.62), Inches(w-0.35), Inches(0.85), [[R(d, 11, MUTED, False)]], line_spacing=1.05)
text(s, Inches(0.6), Inches(5.35), Inches(12.1), Inches(1.1),
     [[R("Targets are set in Phase 0 on a held-out, labelled eval set.  ", 13, ACCENT, True),
       R("A green build is not success — success is these numbers on data the system never trained on.",
         13, TEXT, False)]], align=PP_ALIGN.CENTER, line_spacing=1.1)

# 11 — Go/no-go & immediate next steps
s = slide()
kicker(s, "Decision")
heading(s, "Recommendation & immediate next steps", y=0.9)
card(s, Inches(0.6), Inches(2.0), Inches(5.95), Inches(3.1), title="Recommendation", title_color=GREEN,
     pill_txt="PROCEED — STAGED", pill_color=GREEN, accent=GREEN,
     body=["Feasible and de-risked. Proceed hybrid, phase by phase, with a go/no-go gate after the "
           "shadow pilot (Phase 5): only widen auto-clear once precision and recall on the eval set "
           "clear agreed targets.",
           "The prototype already proves the hardest part works and is explainable."], body_size=12.5)
card(s, Inches(6.78), Inches(2.0), Inches(5.95), Inches(3.1), title="This week", title_color=ACCENT,
     body=["1.  Decide build / buy / hybrid + any data-residency constraint.",
           "2.  Kick off assembling the labelled eval set (longest lead time).",
           "3.  Approve Phase 1 — harden the prototype (tests, config, schema, auth). Low risk, can "
           "start immediately.",
           "4.  Agree the success-metric targets so 'done' is measurable."], body_size=12)
text(s, Inches(0.6), Inches(5.45), Inches(12.1), Inches(1.0),
     [[R("Principle that never changes:  ", 14, WHITE, True),
       R("computer flags, human decides — explainable evidence and a signed audit trail at every step.",
         14, TEXT, False)]], align=PP_ALIGN.CENTER, line_spacing=1.1)
footer(s, "StatementGuard · feasibility & implementation plan · github.com/elline-c/AI-in-finance-")

out = __file__.rsplit("/", 1)[0] + "/statementguard-feasibility-and-plan.pptx"
prs.save(out)
print("wrote", out, "slides:", len(prs.slides._sldIdLst))
