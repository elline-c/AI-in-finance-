#!/usr/bin/env python3
"""Generate the StatementGuard "Future Improvements" PowerPoint deck.

Pure python-pptx, no network. Produces a 16:9 .pptx with the same dark theme
as the project deck. Findings are grounded in a direct review of the codebase
(see citations on each slide).

Usage:
    PYTHONPATH=/projects/sandbox/.pylibs python3 build_improvements_pptx.py
Output:
    statementguard-future-improvements.pptx  (next to this script)
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---- palette (matches the HTML/PDF deck) ----
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

EMU = 914400
SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    r.fill.solid(); r.fill.fore_color.rgb = BG
    r.line.fill.background()
    r.shadow.inherit = False
    # send bg to back
    sp = r._element
    sp.getparent().remove(sp)
    s.shapes._spTree.insert(2, sp)
    return s


def box(s, x, y, w, h, fill=None, line=None, line_w=0.75, radius=False):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.0, wrap=True):
    """runs: list of paragraphs; each paragraph is a list of (txt, size, color, bold, italic, font)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        for (txt, size, color, bold, italic, font) in para:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(size); r.font.color.rgb = color
            r.font.bold = bold; r.font.italic = italic
            r.font.name = font
    return tb


def R(txt, size=14, color=TEXT, bold=False, italic=False, font=FONT):
    return (txt, size, color, bold, italic, font)


def kicker(s, txt):
    text(s, Inches(0.6), Inches(0.35), Inches(11), Inches(0.4),
         [[R(txt.upper(), 11, MUTED, True)]])


def heading(s, txt, size=28, y=0.55, color=WHITE):
    text(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.9),
         [[R(txt, size, color, True)]])
    # accent divider
    box(s, Inches(0.62), Inches(y + size/72.0 + 0.12), Inches(0.9), Pt(3), fill=ACCENT)


def pill(s, x, y, txt, color=ACCENT, w=None):
    ww = w or Inches(0.15 + 0.095 * len(txt))
    b = box(s, x, y, ww, Inches(0.32), fill=PANEL, line=color, line_w=1, radius=True)
    text(s, x, y - Inches(0.01), ww, Inches(0.34), [[R(txt, 10, color, True)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return ww


def card(s, x, y, w, h, title=None, title_color=WHITE, body=None, body_size=12,
         pill_txt=None, pill_color=ACCENT, accent=None):
    box(s, x, y, w, h, fill=PANEL, line=accent or LINE, line_w=1.25, radius=True)
    pad = Inches(0.18)
    cy = y + pad
    if pill_txt:
        pill(s, x + pad, cy, pill_txt, pill_color)
        cy += Inches(0.42)
    if title:
        tb = text(s, x + pad, cy, w - 2*pad, Inches(0.5),
                  [[R(title, 14, title_color, True)]])
        cy += Inches(0.42)
    if body:
        paras = body if isinstance(body, list) else [body]
        runs = [[R(p, body_size, TEXT, False)] for p in paras]
        text(s, x + pad, cy, w - 2*pad, y + h - cy - pad, runs,
             space_after=3, line_spacing=1.05)


def mono(s, x, y, w, h, lines):
    """lines: list of (txt, color)."""
    box(s, x, y, w, h, fill=RGBColor(0x01, 0x04, 0x09), line=LINE, line_w=1, radius=True)
    tb = s.shapes.add_textbox(x + Inches(0.15), y + Inches(0.1), w - Inches(0.3), h - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (txt, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2); p.line_spacing = 1.0
        r = p.add_run(); r.text = txt
        r.font.size = Pt(11); r.font.name = MONO; r.font.color.rgb = color


def footer(s, txt):
    text(s, Inches(0.6), Inches(7.05), Inches(12), Inches(0.35),
         [[R(txt, 8, MUTED, False)]])


# ============================ SLIDES ============================

# 1 — Title
s = slide()
kicker(s, "StatementGuard · Engineering Review")
text(s, Inches(0.6), Inches(2.4), Inches(12.1), Inches(1.2),
     [[R("Future Improvements", 40, WHITE, True)]])
box(s, Inches(0.62), Inches(3.5), Inches(1.1), Pt(4), fill=ACCENT)
text(s, Inches(0.6), Inches(3.75), Inches(12), Inches(0.8),
     [[R("A prioritized roadmap to take the prototype to production — "
         "grounded in a direct review of the codebase.", 16, TEXT, False)]])
text(s, Inches(0.6), Inches(4.6), Inches(12), Inches(1.2),
     [[R("Themes: ", 13, MUTED, True),
       R("security · persistence · configurability · ML maturity · observability · code hygiene",
         13, ACCENT, False)]])
footer(s, "github.com/elline-c/AI-in-finance-  ·  findings verified against fraud_detect/ source")

# 2 — How we got here / method
s = slide()
heading(s, "How these were identified")
card(s, Inches(0.6), Inches(1.7), Inches(3.9), Inches(2.2), title="Codebase review",
     body=["Read the orchestration, data layer, nodes, ML, web apps, audit chain and tests — "
           "not just the docs. Each finding below cites a real file."], body_size=12.5)
card(s, Inches(4.7), Inches(1.7), Inches(3.9), Inches(2.2), title="Verified in source",
     body=["No-auth confirmed: zero Depends/HTTPBasic/OAuth2/session in *app.py. "
           "JSON store with threading.Lock + whole-file rewrite confirmed in store.py."],
     body_size=12.5)
card(s, Inches(8.8), Inches(1.7), Inches(3.9), Inches(2.2), title="Severity-ranked",
     body=["Grouped RED (blocks production), ORANGE (gaps), and hygiene. "
           "Ordered by risk-reduction per unit of effort."], body_size=12.5)
text(s, Inches(0.6), Inches(4.3), Inches(12.1), Inches(2.4),
     [[R("What's already strong (keep it):", 15, WHITE, True)],
      [R("•  Audit trail is genuinely production-grade — ", 13, TEXT, False),
       R("SHA-256 hash chaining + Ed25519 signing + pluggable Signer (KMS/HSM-ready) + a 4-state verifier",
         13, GREEN, False),
       R("  (fraud_detect/audit/).", 13, TEXT, False)],
      [R("•  Strong typing & Pydantic contract; missing fields become explicit nulls, never hallucinated (models.py).", 13, TEXT, False)],
      [R("•  Governance by construction — machine never auto-rejects; separation of duties enforced in banker_app.py.", 13, TEXT, False)],
      [R("•  Solid upload hardening + prompt-injection sanitising on the web intake path.", 13, TEXT, False)]],
     space_after=6, line_spacing=1.05)

# 3 — Priority matrix overview
s = slide()
heading(s, "The roadmap at a glance")
rows = [
    ("P1", "Authentication, RBAC & sessions on all 3 web apps", "RED", RED),
    ("P2", "Replace JSON store with a real database; complete the audit wiring", "RED", RED),
    ("P3", "Externalize & version all thresholds/weights (config-driven)", "ORANGE", AMBER),
    ("P4", "ML maturity: seeds, reproducible data, retrain, real OCR + pixel forensics", "ORANGE", AMBER),
    ("P5", "Observability & ops: CI, Docker, health, structured logs, metrics, pinning", "ORANGE", AMBER),
    ("P6", "Code hygiene: de-dup helpers, timezone-aware datetimes, web integration tests", "HYGIENE", MUTED),
]
y = 1.75
for pid, label, sev, col in rows:
    box(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.72), fill=PANEL, line=LINE, line_w=1, radius=True)
    box(s, Inches(0.6), Inches(y), Inches(0.12), Inches(0.72), fill=col)
    text(s, Inches(0.85), Inches(y), Inches(0.9), Inches(0.72), [[R(pid, 16, col, True)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(1.8), Inches(y), Inches(9.3), Inches(0.72), [[R(label, 13.5, TEXT, False)]],
         anchor=MSO_ANCHOR.MIDDLE)
    pill(s, Inches(11.2), Inches(y + 0.2), sev, col)
    y += 0.85

# 4 — P1 Authentication
s = slide()
kicker(s, "Priority 1 · RED — blocks any real deployment")
heading(s, "Add authentication, RBAC & sessions", y=0.9)
card(s, Inches(0.6), Inches(2.0), Inches(5.9), Inches(2.5), pill_txt="THE PROBLEM", pill_color=RED,
     accent=RED,
     body=["All three FastAPI apps are completely open. A grep for Depends / HTTPBasic / OAuth2 / "
           "login / session / Authorization across webapp.py, customer_app.py and banker_app.py "
           "returns nothing.",
           "Anyone who can reach the port can approve loans, reject applicants, or read every "
           "statement and its evidence."], body_size=12)
card(s, Inches(6.7), Inches(2.0), Inches(6.0), Inches(2.5), pill_txt="THE FIX", pill_color=GREEN,
     body=["• Authentication on every route (SSO / OIDC in prod).",
           "• Role-based access: customer vs analyst vs loan officer are already distinct apps — "
           "enforce the roles, don't just separate the ports.",
           "• Server-side sessions, CSRF protection on state-changing POSTs, and audit of who viewed "
           "each case.",
           "• Rate-limiting on the public customer upload endpoint."], body_size=11.5)
text(s, Inches(0.6), Inches(4.8), Inches(12), Inches(0.6),
     [[R("Why first: ", 13, AMBER, True),
       R("this is the single change that separates 'demo' from 'deployable'. Everything else assumes "
         "a trusted operator.", 13, TEXT, False)]])
footer(s, "Evidence: no auth primitives found in fraud_detect/*app*.py")

# 5 — P2 Persistence
s = slide()
kicker(s, "Priority 2 · RED — data integrity & scale")
heading(s, "Real database + complete the audit wiring", y=0.9)
card(s, Inches(0.6), Inches(2.0), Inches(5.9), Inches(2.6), pill_txt="THE PROBLEM", pill_color=RED,
     accent=RED,
     body=["CaseStore (store.py) persists every case to one JSON file, rewritten whole on each update, "
           "guarded only by an in-process threading.Lock — unsafe across processes/workers and won't scale.",
           "Some events (INFO_REQUESTED, CREDIT_*) are written to the case store but are not all carried "
           "into the signed audit chain, so the tamper-evident record is incomplete."], body_size=11.5)
card(s, Inches(6.7), Inches(2.0), Inches(6.0), Inches(2.6), pill_txt="THE FIX", pill_color=GREEN,
     body=["• Move to Postgres (or similar) with proper transactions, row-level locking and migrations.",
           "• Route ALL decision/lifecycle events through the Ed25519-signed audit chain — the chain code "
           "already supports it; just wire the remaining event types.",
           "• Keep the Signer behind its interface and back it with a KMS/HSM in production.",
           "• Define data retention & encryption-at-rest for statement PII (PDPA)."], body_size=11)
text(s, Inches(0.6), Inches(4.9), Inches(12), Inches(0.6),
     [[R("Good news: ", 13, GREEN, True),
       R("the audit chain is already production-shaped (hash chain + signatures + verifier). This is wiring "
         "and storage, not a rebuild.", 13, TEXT, False)]])
footer(s, "Evidence: store.py uses threading.Lock + json.dump whole-file replace; audit/ implements the signed chain")

# 6 — P3 Config
s = slide()
kicker(s, "Priority 3 · ORANGE — operability")
heading(s, "Externalize & version all thresholds", y=0.9)
card(s, Inches(0.6), Inches(2.0), Inches(5.9), Inches(2.6), pill_txt="THE PROBLEM", pill_color=AMBER,
     accent=AMBER,
     body=["Detection thresholds and scoring weights are hard-coded in the nodes — structuring window "
           "(14 days), near-threshold band, σ multipliers, gap days, the auto-clear ceiling, and the "
           "fabrication-rule weights in fabrication.py.",
           "Analysts will need to tune these against real false-positive/false-negative rates, but today "
           "that means a code change and redeploy."], body_size=11.5)
card(s, Inches(6.7), Inches(2.0), Inches(6.0), Inches(2.6), pill_txt="THE FIX", pill_color=GREEN,
     body=["• Lift every threshold into a versioned config object (FabricationConfig already shows the "
           "pattern — extend it to all nodes).",
           "• Record which config version scored each case, in the audit trail.",
           "• Let risk/compliance review and change thresholds without a deploy.",
           "• Add a calibration harness that reports precision/recall as thresholds move."], body_size=11)
text(s, Inches(0.6), Inches(4.9), Inches(12), Inches(0.6),
     [[R("Payoff: ", 13, ACCENT, True),
       R("turns tuning from an engineering task into a governed, auditable operations task.", 13, TEXT, False)]])
footer(s, "Evidence: hard-coded constants across fraud_detect/nodes/*.py; FabricationConfig in nodes/fabrication.py")

# 7 — P4 ML maturity
s = slide()
kicker(s, "Priority 4 · ORANGE — the 'AI' needs to be real")
heading(s, "ML maturity & real inputs", y=0.9)
col_w = Inches(3.92)
card(s, Inches(0.6), Inches(1.95), col_w, Inches(2.7), title="Reproducibility",
     body=["GNN trainer sets no random seed — results aren't repeatable on a 336-edge graph. "
           "Training reads ~/Downloads/... hard-coded paths.",
           "Fix: seed everything, parameterize data paths, snapshot the dataset version."],
     body_size=11)
card(s, Inches(4.72), Inches(1.95), col_w, Inches(2.7), title="Coverage & retraining",
     body=["Live GNN lookup is account-level and scores only 62 / 1,201 accounts; covers 1 of 5 AML "
           "typologies.",
           "The row-level GBM is disabled (false alarms on clean business files).",
           "Fix: retrain on representative statements; expand graph coverage; re-enable GBM once precise."],
     body_size=11)
card(s, Inches(8.84), Inches(1.95), col_w, Inches(2.7), title="Real inputs",
     body=["Node 1 OCR and Node 2 forensics are heuristic stand-ins — no real PDF OCR, no pixel/layer "
           "image analysis.",
           "Fix: integrate real OCR + pixel forensics behind the existing provider interfaces "
           "(hybrid buy/build)."],
     body_size=11)
text(s, Inches(0.6), Inches(4.95), Inches(12), Inches(0.9),
     [[R("Discipline to keep: ", 13, GREEN, True),
       R("the EVALUATION.md honesty ('a green build is not success'). Report offline vs live numbers "
         "separately and never overclaim.", 13, TEXT, False)]])
footer(s, "Evidence: no manual_seed in gnn/train_gnn.py; ~/Downloads paths in ml/train_*.py; random_state=42 in sklearn only")

# 8 — P5 Observability
s = slide()
kicker(s, "Priority 5 · ORANGE — you can't run what you can't see")
heading(s, "Observability & operations", y=0.9)
left = [
    "No CI pipeline, Dockerfile, health checks, or metrics endpoints.",
    "Logging is minimal and unstructured.",
    "Dependencies not fully pinned — builds aren't reproducible.",
    "No deployment story (shadow-mode pilot is in the plan, not the code).",
]
right = [
    "Add CI (lint + type-check + the 71 test modules) on every PR.",
    "Containerize; add /health and /ready; pin dependencies with a lockfile.",
    "Structured JSON logging with a correlation id per case through all nodes.",
    "Metrics: throughput, per-node latency, auto-clear rate, escalation rate, model-abstain rate; alert on drift.",
]
card(s, Inches(0.6), Inches(2.0), Inches(5.9), Inches(3.4), pill_txt="GAPS TODAY", pill_color=AMBER,
     accent=AMBER, body=["• " + x for x in left], body_size=12.5)
card(s, Inches(6.7), Inches(2.0), Inches(6.0), Inches(3.4), pill_txt="TARGET", pill_color=GREEN,
     body=["• " + x for x in right], body_size=12)
footer(s, "Evidence: no CI/Docker/health/metrics found; sparse logging across fraud_detect/")

# 9 — P6 Hygiene
s = slide()
kicker(s, "Priority 6 · Hygiene — pay down quietly")
heading(s, "Code hygiene & smaller wins", y=0.9)
items = [
    ("Timezone-aware time", "datetime.utcnow() is used pervasively (e.g. store.py) and is deprecated/naive. "
                            "Switch to timezone-aware datetime.now(timezone.utc)."),
    ("De-duplicate helpers", "Upload-hardening and text-scrub (_scrub) helpers are duplicated across the web "
                             "apps. Extract one shared module."),
    ("Web-app integration tests", "Tests cover detection well, but the FastAPI routes/guardrails have thinner "
                                  "coverage. Add request-level tests (incl. the banker clearance guard)."),
    ("Dataset-dependent skips", "Several tests skip when external datasets are absent. Ship small committed "
                                "fixtures so core paths always run in CI."),
    ("Dead imports / tidy", "Prune unused imports and tighten module boundaries as the config refactor lands."),
]
y = 1.95
for t, b in items:
    box(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.92), fill=PANEL, line=LINE, line_w=1, radius=True)
    text(s, Inches(0.8), Inches(y + 0.08), Inches(3.2), Inches(0.8), [[R(t, 13, ACCENT, True)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.1), Inches(y + 0.08), Inches(8.4), Inches(0.8), [[R(b, 11.5, TEXT, False)]],
         anchor=MSO_ANCHOR.MIDDLE)
    y += 1.02

# 10 — Sequenced plan / close
s = slide()
kicker(s, "Summary")
heading(s, "A sensible sequence", y=0.9)
phases = [
    ("Now", "P1 auth + P6 timezone/hygiene", "Make it safe to put in front of anyone.", GREEN),
    ("Next", "P2 database + full audit wiring, P3 config", "Make the data trustworthy and tunable.", ACCENT),
    ("Then", "P4 ML maturity + real OCR/forensics, P5 ops", "Make the 'AI' real and operable.", AMBER),
]
x = 0.6
for tag, what, why, col in phases:
    box(s, Inches(x), Inches(2.0), Inches(3.97), Inches(3.0), fill=PANEL, line=col, line_w=1.5, radius=True)
    text(s, Inches(x+0.2), Inches(2.2), Inches(3.6), Inches(0.5), [[R(tag, 20, col, True)]])
    text(s, Inches(x+0.2), Inches(2.85), Inches(3.6), Inches(1.0), [[R(what, 13.5, WHITE, True)]])
    text(s, Inches(x+0.2), Inches(3.95), Inches(3.6), Inches(0.9), [[R(why, 12, MUTED, False)]])
    x += 4.12
text(s, Inches(0.6), Inches(5.4), Inches(12.1), Inches(1.2),
     [[R("Guiding principle unchanged: ", 14, WHITE, True),
       R("computer flags, human decides. Every improvement must preserve explainable evidence, the "
         "no-auto-reject ceiling, and the signed audit trail.", 14, TEXT, False)]],
     line_spacing=1.1)
footer(s, "StatementGuard · future-improvements review · github.com/elline-c/AI-in-finance-")

out = __file__.rsplit("/", 1)[0] + "/statementguard-future-improvements.pptx"
prs.save(out)
print("wrote", out, "slides:", len(prs.slides._sldIdLst))
