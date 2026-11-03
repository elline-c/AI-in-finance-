#!/usr/bin/env python3
"""Render the StatementGuard deck to a landscape 16:9 PDF.

Pure fpdf2 - no browser, no network. Reproduces the deck's dark theme and the
same content as statementguard-deck-offline.html, one slide per page.

Usage:
    PYTHONPATH=/projects/sandbox/.pylibs python3 build_pdf.py
Output:
    statementguard-deck.pdf  (next to this script)
"""
from fpdf import FPDF

# ---- palette (matches the HTML deck) ----
BG      = (13, 17, 23)      # #0d1117
PANEL   = (22, 27, 34)      # #161b22
ACCENT  = (79, 156, 255)    # #4f9cff
GREEN   = (63, 185, 80)
AMBER   = (210, 153, 34)
RED     = (248, 81, 73)
MUTED   = (139, 148, 158)
LINE    = (48, 54, 61)
WHITE   = (240, 243, 247)
TEXT    = (201, 209, 217)
CODE_BG = (1, 4, 9)

# 16:9 landscape page in mm
PAGE_W, PAGE_H = 254.0, 142.875   # 10in x 5.625in
MARGIN = 14.0

FONT = "Helvetica"
MONO = "Courier"


class Deck(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format=(PAGE_H, PAGE_W))
        self.set_auto_page_break(False)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.total = 0
        self.idx = 0

    # ---- primitives ----
    def bg(self):
        self.set_fill_color(*BG)
        self.rect(0, 0, PAGE_W, PAGE_H, "F")

    def progress(self):
        # top progress bar
        frac = (self.idx - 1) / max(1, self.total - 1)
        self.set_fill_color(*LINE)
        self.rect(0, 0, PAGE_W, 1.1, "F")
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, PAGE_W * frac, 1.1, "F")

    def pageno(self):
        self.set_xy(PAGE_W - 34, PAGE_H - 8)
        self.set_font(FONT, "", 7)
        self.set_text_color(*MUTED)
        self.cell(20, 4, f"{self.idx} / {self.total}", align="R")

    def kicker(self, text):
        self.set_xy(MARGIN, MARGIN)
        self.set_font(FONT, "B", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, text.upper())

    def heading(self, text, y=None, size=22, color=WHITE):
        self.set_xy(MARGIN, y if y is not None else MARGIN + 4)
        self.set_font(FONT, "B", size)
        self.set_text_color(*color)
        self.cell(0, size * 0.52, text)
        self.ln(size * 0.52)

    def divider(self, w=24):
        y = self.get_y() + 1.5
        self.set_fill_color(*ACCENT)
        self.rect(MARGIN, y, w, 1.1, "F")
        self.set_y(y + 4)

    def body(self, text, size=10, color=TEXT, lh=5.0, w=None):
        self.set_font(FONT, "", size)
        self.set_text_color(*color)
        self.set_x(MARGIN)
        self.multi_cell(w or (PAGE_W - 2 * MARGIN), lh, text)

    # ---- helpers ----
    def new_slide(self, kicker=None):
        self.add_page()
        self.idx += 1
        self.bg()
        self.progress()
        self.pageno()
        if kicker:
            self.kicker(kicker)

    def card(self, x, y, w, h, title=None, lines=None, pill=None, pill_color=ACCENT,
             title_color=WHITE, body_size=8.2):
        self.set_fill_color(*PANEL)
        self.set_draw_color(*LINE)
        self.set_line_width(0.25)
        self.rect(x, y, w, h, "DF")
        cx, cy = x + 4, y + 3.5
        if pill:
            self.set_xy(cx, cy)
            self.set_font(FONT, "B", 7)
            tw = self.get_string_width(pill) + 4
            r, g, b = pill_color
            self.set_fill_color(int(r*0.22+PANEL[0]*0.78),
                                int(g*0.22+PANEL[1]*0.78),
                                int(b*0.22+PANEL[2]*0.78))
            self.rect(cx, cy, tw, 4.2, "F")
            self.set_text_color(*pill_color)
            self.set_xy(cx, cy + 0.3)
            self.cell(tw, 3.6, pill, align="C")
            cy += 6
        if title:
            self.set_xy(cx, cy)
            self.set_font(FONT, "B", 10)
            self.set_text_color(*title_color)
            self.multi_cell(w - 8, 4.6, title)
            cy = self.get_y() + 1.2
        if lines:
            self.set_xy(cx, cy)
            self.set_font(FONT, "", body_size)
            self.set_text_color(*TEXT)
            self.multi_cell(w - 8, body_size * 0.46 + 1.0, lines)

    def stat(self, x, y, w, h, num, label):
        self.set_fill_color(*PANEL)
        self.set_draw_color(*LINE)
        self.set_line_width(0.25)
        self.rect(x, y, w, h, "DF")
        self.set_xy(x, y + h * 0.28)
        self.set_font(FONT, "B", 20)
        self.set_text_color(*ACCENT)
        self.cell(w, 8, num, align="C")
        self.set_xy(x + 2, y + h * 0.62)
        self.set_font(FONT, "", 7.2)
        self.set_text_color(*MUTED)
        self.multi_cell(w - 4, 3.4, label, align="C")

    def mono_block(self, x, y, w, h, lines):
        """lines: list of (text, color)."""
        self.set_fill_color(*CODE_BG)
        self.set_draw_color(*LINE)
        self.set_line_width(0.25)
        self.rect(x, y, w, h, "DF")
        self.set_font(MONO, "", 7.4)
        ty = y + 3
        for text, color in lines:
            self.set_xy(x + 3, ty)
            self.set_text_color(*color)
            self.cell(w - 6, 3.6, text)
            ty += 3.9

    def node(self, x, y, w, h, name, sub, kind="core"):
        border = {"core": (95, 99, 104), "ml": ACCENT, "ok": GREEN, "rev": AMBER}[kind]
        fill = {"core": PANEL, "ml": (18, 26, 40), "ok": (16, 32, 20), "rev": (34, 28, 14)}[kind]
        self.set_fill_color(*fill)
        self.set_draw_color(*border)
        self.set_line_width(0.3)
        self.rect(x, y, w, h, "DF")
        self.set_xy(x + 1.5, y + 2.2)
        self.set_font(FONT, "B", 7.2)
        self.set_text_color(*ACCENT)
        self.multi_cell(w - 3, 3.2, name, align="C")
        self.set_xy(x + 1.5, y + 6.4)
        self.set_font(FONT, "", 6.4)
        self.set_text_color(*TEXT)
        self.multi_cell(w - 3, 2.9, sub, align="C")

    def arrow(self, x, y):
        self.set_font(FONT, "B", 11)
        self.set_text_color(*MUTED)
        self.set_xy(x, y)
        self.cell(6, 6, chr(62), align="C")  # '>'

    def lead_center(self, text, y, size=9):
        self.set_xy(MARGIN, y)
        self.set_font(FONT, "I", size)
        self.set_text_color(*MUTED)
        self.multi_cell(PAGE_W - 2 * MARGIN, size * 0.5, text, align="C")

    def table(self, x, y, col_w, rows, header, row_h=5.6, fs=7.6):
        self.set_font(FONT, "B", fs)
        self.set_text_color(*ACCENT)
        cx = x
        for i, head in enumerate(header):
            self.set_xy(cx, y)
            self.cell(col_w[i], row_h, head)
            cx += col_w[i]
        self.set_draw_color(*LINE)
        self.set_line_width(0.4)
        self.line(x, y + row_h, x + sum(col_w), y + row_h)
        ry = y + row_h + 0.6
        self.set_line_width(0.2)
        for row in rows:
            cx = x
            for i, cell in enumerate(row):
                self.set_xy(cx, ry)
                self.set_font(FONT, "", fs)
                self.set_text_color(*TEXT)
                self.multi_cell(col_w[i], row_h * 0.75, cell)
                cx += col_w[i]
            self.line(x, ry + row_h, x + sum(col_w), ry + row_h)
            ry += row_h + 0.4


def build():
    d = Deck()
    slides = SLIDES
    d.total = len(slides)
    for fn in slides:
        fn(d)
    out = __file__.rsplit("/", 1)[0] + "/statementguard-deck.pdf"
    d.output(out)
    print("wrote", out, "pages:", d.total)


# ============================ slides ============================

def s_title(d):
    d.new_slide("AI in Finance | Prototype Demonstration")
    d.heading("StatementGuard", y=48, size=30)
    # 'prototype' tag
    d.set_xy(MARGIN + 92, 52)
    d.set_font(FONT, "B", 8)
    d.set_fill_color(28, 40, 58)
    d.rect(MARGIN + 92, 52, 26, 6, "F")
    d.set_text_color(*ACCENT)
    d.set_xy(MARGIN + 92, 52.8)
    d.cell(26, 4.4, "PROTOTYPE", align="C")
    d.set_y(62)
    d.divider()
    d.body("AI screening of corporate-loan bank statements for tampering - before a credit analyst ever sees them.",
           size=12, color=WHITE, lh=6)
    d.ln(1)
    d.body("A multi-agent pipeline that auto-clears clean files, routes suspicious ones to a human, and attaches "
           "readable evidence to every flag. Built to MAS FEAT principles - the machine never approves or rejects "
           "a customer on its own.", size=9.5, color=MUTED, lh=5)
    d.set_xy(MARGIN, PAGE_H - 10)
    d.set_font(FONT, "", 6.5); d.set_text_color(*MUTED)
    d.cell(0, 4, "github.com/elline-c/AI-in-finance-  |  ~25.7k lines of Python  |  71 test modules")


def s_problem(d):
    d.new_slide()
    d.heading("The problem"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    d.card(MARGIN, y, w, 46, title="Today",
           lines="Tampered statements reach credit assessment because catching them relies on slow, "
                 "inconsistent manual review. Analysts read every file by hand; fakes that look plausible get through.")
    d.card(MARGIN + w + 8, y, w, 46, title="What we need",
           lines="Screen EVERY submitted statement automatically, auto-clear the clean ones so analysts focus only "
                 "on flagged files, and attach explainable evidence to each flag.")
    d.lead_center("The guiding rule throughout: the system only gathers evidence and, at worst, routes a case to a "
                  "human - it never rejects a customer automatically.", y + 52)


def s_feat(d):
    d.new_slide()
    d.heading("Designed for governance (MAS FEAT)"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 3 * 6) / 4
    feats = [
        ("Fairness", "No protected attributes in scoring; flag rates tested across segments."),
        ("Ethics", "Fail-safe: poor data or low confidence escalates - never auto-clears."),
        ("Accountability", "Every AI output and every human decision is recorded to an audit log."),
        ("Transparency", "Every flag carries a stable reason code + plain-English evidence."),
    ]
    for i, (t, b) in enumerate(feats):
        d.card(MARGIN + i * (w + 6), y, w, 40, title=t, lines=b, body_size=7.8)
    y2 = y + 46
    w3 = (PAGE_W - 2 * MARGIN - 2 * 6) / 3
    gov = [
        ("PDPA", "Prototype uses only synthetic data; PII access-controlled & encrypted in production.", ACCENT),
        ("MAS TRM", "Immutable, signed audit chain over every screening and reviewer decision.", ACCENT),
        ("HUMAN OWNS CALL", "Most severe automated outcome is a hand-off to a person.", GREEN),
    ]
    for i, (p, b, c) in enumerate(gov):
        d.card(MARGIN + i * (w3 + 6), y2, w3, 32, lines=b, pill=p, pill_color=c, body_size=7.8)


def s_pipeline(d):
    d.new_slide()
    d.heading("The pipeline at a glance"); d.divider()
    y = d.get_y() + 2
    nw, nh, gap = 36, 17, 6
    chain = [("Input", "PDF statement", "core"), ("Node 1", "Ingestion & parse", "core"),
             ("Node 2", "Visual forensics", "core"), ("Node 3", "Financial logic", "core"),
             ("Node 4", "Risk synthesis", "core")]
    total_w = len(chain) * nw + (len(chain) - 1) * gap
    x = (PAGE_W - total_w) / 2
    for i, (n, s, k) in enumerate(chain):
        d.node(x, y, nw, nh, n, s, k)
        if i < len(chain) - 1:
            d.arrow(x + nw + (gap - 6) / 2, y + nh / 2 - 3)
        x += nw + gap
    # ml row
    y2 = y + nh + 6
    ml = [("Security layer", "prompt-injection guard"), ("Node 5", "cross-case memory"),
          ("Node 6a", "ML - GNN transaction"), ("Node 6b", "ML - statement forgery")]
    mw = 46
    tot2 = len(ml) * mw + (len(ml) - 1) * gap
    x = (PAGE_W - tot2) / 2
    for n, s in ml:
        d.node(x, y2, mw, nh, n, s, "ml"); x += mw + gap
    # outcomes
    y3 = y2 + nh + 6
    d.node((PAGE_W / 2) - 78, y3, 72, nh, "AUTO_CLEAR", "low risk | confident | no findings -> approval rec.", "ok")
    d.arrow(PAGE_W / 2 - 3, y3 + nh / 2 - 3)
    d.node((PAGE_W / 2) + 6, y3, 72, nh, "HUMAN_REVIEW", "any material finding | low confidence | poor data", "rev")
    d.lead_center("All agents feed Node 4. Findings from independent agents that agree raise a corroboration bonus.",
                  y3 + nh + 4)


def s_node1(d):
    d.new_slide()
    d.heading("Node 1 - Ingestion  (the mailroom)"); d.divider()
    bullets = [
        "Opens the PDF and extracts text, hidden metadata (author, software, dates), font fingerprints, the header",
        "   (holder, UEN, period, balances) and the transaction table.",
        "Missing or unreadable fields are reported HONESTLY as data-quality findings - it never guesses.",
        "Tries six layout profiles; best page coverage wins. Ties that disagree are flagged rather than decided.",
        "A self-check compares printed section totals with parsed rows, so our own reading mistake isn't",
        "   mistaken for document fraud.",
    ]
    d.set_x(MARGIN); d.set_font(FONT, "", 9); d.set_text_color(*TEXT)
    for b in bullets:
        prefix = "-  " if not b.startswith("   ") else "   "
        d.set_x(MARGIN)
        d.multi_cell(PAGE_W - 2 * MARGIN, 5.2, (prefix + b.strip()) if not b.startswith("   ") else b)
    y = d.get_y() + 3
    w = (PAGE_W - 2 * MARGIN - 2 * 6) / 3
    d.stat(MARGIN, y, w, 30, "6", "layout profiles")
    d.stat(MARGIN + w + 6, y, w, 30, "100%", "rows read / balances replay (synthetic eval, all layouts)")
    d.stat(MARGIN + 2 * (w + 6), y, w, 30, "0.36s", "longest file parsed")


def s_node2(d):
    d.new_slide()
    d.heading("Node 2 - Visual Forensics  (the document examiner)"); d.divider()
    d.lead_center("Looks for signs the file was edited, independent of the numbers.", d.get_y())
    y = d.get_y() + 4
    rows = [
        ["Localised font mismatch (overwritten number)", "VF_FONT_MISMATCH", "MED"],
        ["Mixed date formats / currency notation in rows", "VF_DATE_FORMAT_MIX | VF_CURRENCY_MIX", "MED"],
        ["Image/PDF editor in metadata (Photoshop, Foxit)", "VF_EDIT_SOFTWARE", "HIGH"],
        ["File modified >60s after creation by non-export tool", "VF_MODIFIED_AFTER_CREATION", "MED"],
        ["Creation date before the period it claims to cover", "VF_CREATION_BEFORE_PERIOD", "MED"],
    ]
    d.table(MARGIN, y, [108, 90, 20], rows, ["Signal", "Reason code", "Severity"])
    d.lead_center("Adobe's PDF libraries are deliberately NOT flagged - real banks use them. Tuned so VF_CURRENCY_MIX "
                  "went from 200/200 false fires on one bank layout to 0 / 2,000.", y + 44)


def s_node3(d):
    d.new_slide()
    d.heading("Node 3 - Financial Logic  (the accountant)"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    d.card(MARGIN, y, w, 48, title="Core checks",
           lines="Row-by-row running-balance reconciliation | closing-balance match | period-continuity gaps "
                 "(removed pages) | sudden deposits vs baseline | unverified counterparties | structuring "
                 "(3+ near-threshold credits within 14 days) | UEN <-> ACRA entity cross-check.")
    d.card(MARGIN + w + 8, y, w, 48, title="Fabrication rules  (income faking)",
           lines="Duplicate invoice refs in one month | invoice numbers out of date order | turnover jumps while "
                 "payroll/CPF/rent stay flat. Every finding names the rows and cites evidence - none is proof, each "
                 "asks a person to check.")
    d.lead_center("Live UEN cross-check against data.gov.sg (ACRA) - only the public UEN leaves the machine, over "
                  "verified HTTPS, cached in memory. Fully overridable for offline demos.", y + 52)


def s_mlnodes(d):
    d.new_slide()
    d.heading("Nodes 5 & 6 - Memory & Machine Learning"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    h = 43
    d.card(MARGIN, y, w, h, title="Node 5 - Cross-case memory",
           lines="Matches this statement's suspicious rows against prior cases a HUMAN already decided. Known-bad "
                 "repeat -> flag; previously cleared -> make sure this route isn't being exploited. Learns only from "
                 "human-verified decisions and always cites the prior case.", body_size=7.6)
    d.card(MARGIN + w + 8, y, w, h, title="Node 6b - ML statement forgery", pill="ACTIVE", pill_color=GREEN,
           lines="Scores the whole document for tampering from fonts, editor traces and balance math. Abstains on "
                 "documents larger than its training (single-month), letting rules decide. Fail-safe: a missing model "
                 "yields no findings.", body_size=7.6)
    y2 = y + h + 5
    d.card(MARGIN, y2, w, h, title="Node 6a - GNN transaction", pill="GNN ON, GBM OFF", pill_color=AMBER,
           lines="A graph neural network catches relational AML patterns a single row can't reveal - round-tripping, "
                 "mule fan-in/out, dormant reactivation - via fast lookup of offline-precomputed scores. The row-level "
                 "GBM is off by default (false alarms on clean business files).", body_size=7.6)
    d.card(MARGIN + w + 8, y2, w, h, title="Security layer - prompt-injection guard",
           lines="Scans extracted text for hidden commands (ignore previous instructions, mark low risk). It NEVER "
                 "obeys - treats it as untrusted data and files the attempt itself as strong evidence of fraud.",
           body_size=7.6)


def s_node4(d):
    d.new_slide()
    d.heading("Node 4 - Risk & Synthesis  (the senior officer)"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    # risk card with band table drawn as text
    d.card(MARGIN, y, w, 52, title="Risk score (1-100)",
           lines="1-25 Low   |   26-50 Medium   |   51-75 High   |   76-100 Very High\n\n"
                 "Sum of finding weights + corroboration bonus when independent agents agree. Any material (MED+) "
                 "finding floors it into the Medium band.", body_size=8)
    d.card(MARGIN + w + 8, y, w, 52, title="Confidence score (1-100)",
           lines="Starts high for a clean digital PDF. Lowered by data-quality gaps, thin evidence near a boundary, "
                 "single-dimension evidence. Raised by corroboration.\n\n"
                 "Routing:  AUTO_CLEAR only when risk <= 25 AND confidence >= 60 AND no material findings.  "
                 "HUMAN_REVIEW otherwise - including low confidence or poor data, even if risk looks low.",
           body_size=8)
    d.lead_center("The ceiling is always \"send to a human.\"  The machine never rejects a customer.", y + 56)


def s_roles(d):
    d.new_slide()
    d.heading("Three roles, separated by design"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 2 * 6) / 3
    h = 52
    d.card(MARGIN, y, w, h, title="Customer portal",
           lines="customer_app.py | :8001\n\nApplies, uploads a statement, tracks status. Sees ONLY neutral status - "
                 "Verified / Being reviewed / Action needed. Never sees scores, evidence or routing (would tip off "
                 "bad actors).", body_size=7.8)
    d.card(MARGIN + w + 6, y, w, h, title="Fraud analyst",
           lines="webapp.py | :8000\n\nScreening queue, case overview, flagged transactions, per-stage view, PDF. "
                 "Can Request info / Approve / Reject - each with a mandatory reason + optional training feedback.",
           body_size=7.8)
    d.card(MARGIN + 2 * (w + 6), y, w, h, title="Loan officer",
           lines="banker_app.py | :8002\n\nOnly sees fraud-cleared cases. Makes the lending call (Approve / Decline / "
                 "Refer). CANNOT override a fraud outcome - screening decides trust, the banker decides lending.",
           body_size=7.8)
    d.lead_center("All three apps share one case store & audit log. The application reference number IS the case id - "
                  "a submission appears in the analyst queue automatically.", y + h + 4)


def s_demo_setup(d):
    d.new_slide()
    d.heading("Live demonstration", size=22); d.divider()
    d.lead_center("No external dependencies for the core pipeline - pure Python 3.", d.get_y())
    lines = [
        ("# 1) generate synthetic samples (clean + deliberately tampered)", MUTED),
        ("$ python -m fraud_detect.synthetic", TEXT),
        ("", TEXT),
        ("# 2) run the full pipeline end-to-end", MUTED),
        ("$ python demo.py", TEXT),
        ("", TEXT),
        ("# 3) or run any statement through the CLI", MUTED),
        ("$ python3 -m statement_guard.cli samples/clean_statement.json", TEXT),
        ("$ python3 -m statement_guard.cli samples/tampered_statement.json", TEXT),
        ("", TEXT),
        ("# 4) the three web apps (shared case store)", MUTED),
        ("$ uvicorn fraud_detect.customer_app:app --port 8001   # applicant", TEXT),
        ("$ uvicorn fraud_detect.webapp:app        --port 8000   # analyst", TEXT),
        ("$ uvicorn fraud_detect.banker_app:app    --port 8002   # loan officer", TEXT),
    ]
    d.mono_block(MARGIN, d.get_y() + 4, PAGE_W - 2 * MARGIN, 72, lines)


def s_demo_clean(d):
    d.new_slide()
    d.heading("Demo A - a clean statement"); d.divider()
    lines = [
        ("StatementGuard - clean_statement.json", ACCENT),
        ("Node 1 Ingestion ........ OK   balances reconcile, all fields present", GREEN),
        ("Node 2 Forensics ........ OK   no editor traces, consistent fonts/formats", GREEN),
        ("Node 3 Financial ........ OK   running balance replays row-by-row", GREEN),
        ("Node 6b ML forgery ...... OK   low forgery likelihood", GREEN),
        ("----------------------------------------------------------", MUTED),
        ("Risk score ......... 1 / 100   (Low)", GREEN),
        ("Confidence ......... High", GREEN),
        ("Decision ........... AUTO_CLEAR -> approval recommendation", GREEN),
    ]
    d.mono_block(MARGIN, d.get_y() + 2, PAGE_W - 2 * MARGIN, 52, lines)
    d.lead_center("Clean + confident + no material findings -> the analyst never has to touch it.", d.get_y() + 56)


def s_demo_tampered(d):
    d.new_slide()
    d.heading("Demo B - a tampered statement"); d.divider()
    lines = [
        ("StatementGuard - tampered_statement.json", ACCENT),
        ("Node 1 Ingestion ........ 1 data-quality note", AMBER),
        ("Node 2 Forensics ........ VF_EDIT_SOFTWARE  (OpenPage Writer 3.8)", RED),
        ("                         VF_MODIFIED_AFTER_CREATION", RED),
        ("Node 3 Financial ........ FL_BALANCE_BREAK  row 14 doesn't reconcile", RED),
        ("                         FL_STRUCTURING    3 credits @ ~$9.8k / 14 days", RED),
        ("                         FL_UEN_NAME_MISMATCH", RED),
        ("Node 6b ML forgery ...... elevated likelihood", AMBER),
        ("----------------------------------------------------------", MUTED),
        ("Risk score ......... High  (+corroboration: visual AND financial agree)", RED),
        ("Confidence ......... lowered by data-quality gap", AMBER),
        ("Decision ........... HUMAN_REVIEW  - 10 pieces of evidence attached", AMBER),
    ]
    d.mono_block(MARGIN, d.get_y() + 2, PAGE_W - 2 * MARGIN, 60, lines)
    d.lead_center("Every flag carries a reason code + the raw data it came from - fully auditable.", d.get_y() + 64)


def s_evidence(d):
    d.new_slide()
    d.heading("Every flag is auditable evidence"); d.divider()
    d.lead_center("Findings are typed, validated objects - not opaque scores. Missing fields become explicit nulls, "
                  "never hallucinated values.", d.get_y())
    lines = [
        ("{", TEXT),
        ('  "reason_code": "FL_BALANCE_BREAK",', GREEN),
        ('  "severity":    "HIGH",', GREEN),
        ('  "stage":       "financial_logic",', GREEN),
        ('  "explanation": "Running balance at row 14 is 48,210.00 but the', GREEN),
        ('                  printed balance is 58,210.00 - a 10,000.00 break.",', GREEN),
        ('  "evidence": {', TEXT),
        ('      "row": 14, "printed_balance": 58210.00,', ACCENT),
        ('      "computed_balance": 48210.00, "delta": 10000.00', ACCENT),
        ("  }", TEXT),
        ("}", TEXT),
    ]
    d.mono_block(MARGIN, d.get_y() + 4, PAGE_W - 2 * MARGIN, 52, lines)
    d.lead_center("The analyst sees which stage fired, on which row, and why.", d.get_y() + 56)


def s_validation(d):
    d.new_slide()
    d.heading("Does it actually catch fakes?"); d.divider()
    d.lead_center("Synthetic generator plants ground-truth issues; tests assert each is detected.", d.get_y())
    y = d.get_y() + 4
    rows = [
        ["Altered balance / number", "FL_BALANCE_BREAK | VF_FONT_MISMATCH"],
        ["Format inconsistency", "VF_DATE_FORMAT_MIX | VF_CURRENCY_MIX"],
        ["Metadata mismatch", "VF_EDIT_SOFTWARE | VF_CREATION_BEFORE_PERIOD"],
        ["Period gap | suspicious txn | structuring", "FL_PERIOD_GAP | FL_SUDDEN_DEPOSIT | FL_STRUCTURING"],
        ["UEN mismatch", "FL_UEN_NAME_MISMATCH"],
    ]
    d.table(MARGIN, y, [95, 128], rows, ["Planted tamper", "Detected by"])
    y2 = y + 42
    w = (PAGE_W - 2 * MARGIN - 2 * 6) / 3
    d.stat(MARGIN, y2, w, 28, "71", "test modules")
    d.stat(MARGIN + w + 6, y2, w, 28, "13", "detection tests - every planted issue caught, clean auto-clears, nothing auto-rejects")
    d.stat(MARGIN + 2 * (w + 6), y2, w, 28, "0 / 2000", "false fires after currency-check tuning")


def s_gnn(d):
    d.new_slide()
    d.heading("GNN evaluation - honest, measured"); d.divider()
    d.lead_center("Offline edge-level eval on the synthetic transaction dataset. Reported as what it is, with caveats.",
                  d.get_y())
    y = d.get_y() + 4
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    d.card(MARGIN, y, w, 42, title="Where the GNN adds value",
           lines="On round-tripping accounts it catches 5 that Node 3 misses, at the live threshold with no false "
                 "alarms. Together, GNN + Node 3 catch 19 / 27 fraud accounts (70%) vs 52% rules-alone.", body_size=8)
    d.card(MARGIN + w + 8, y, w, 42, title="Stated limits",
           lines="Narrow: covers 1 of 5 AML typologies; scores 62 / 1,201 accounts. Live path is account-level only "
                 "(offline numbers are an upper bound). Synthetic data -> graph lift may shrink on real data. Only 27 "
                 "fraud test accounts - wide intervals.", body_size=8)
    d.lead_center("\"A green build is not success.\" We separate measured results from aspiration, and never overclaim.",
                  y + 46)


def s_realsim(d):
    d.new_slide()
    d.heading("What's real vs. simulated"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 8) / 2
    d.card(MARGIN, y, w, 50, pill="DONE & WORKING", pill_color=GREEN,
           lines="- The entire 4-agent decision engine, end-to-end\n- Scoring, confidence, routing, corroboration\n"
                 "- Readable evidence on every flag\n- 3 web apps | cross-case memory | ML nodes\n"
                 "- Immutable signed audit chain | test suite", body_size=8.4)
    d.card(MARGIN + w + 8, y, w, 50, pill="SIMULATED / REMAINING", pill_color=AMBER,
           lines="- Real PDF + OCR reading (input is typed data today)\n- Real pixel/layer image forensics "
                 "(metadata + rules now)\n- Historical baselines from real cash-flow data\n"
                 "- Production DB, integration, security hardening", body_size=8.4)
    d.lead_center("In short: the brain is complete; real-world inputs and production hardening are what remain.", y + 54)


def s_roadmap(d):
    d.new_slide()
    d.heading("Roadmap - 6 staged phases"); d.divider()
    y = d.get_y() + 2
    rows = [
        ["0 | Foundations", "Gather examples, agree the goal"],
        ["1 | Harden core", "Make the prototype sturdy"],
        ["2 | Real ingestion", "Read real PDF statements (OCR)"],
        ["3 | Real forensics", "Detect real image tampering"],
        ["4 | Analyst UI", "Build the staff review screen"],
        ["5 | Shadow pilot", "Safely test alongside humans"],
        ["6 | Rollout", "Turn it on gradually"],
    ]
    d.table(MARGIN, y, [70, 150], rows, ["Phase", "In five words"], row_h=5.2)
    d.lead_center("Each phase ships something usable and de-risks the next. Recommended build strategy: hybrid - buy "
                  "OCR + pixel forensics, build the orchestration, financial-logic, scoring & audit layer we own.",
                  y + 48)


def s_metrics(d):
    d.new_slide()
    d.heading("How we'll measure success"); d.divider()
    y = d.get_y() + 1
    w = (PAGE_W - 2 * MARGIN - 2 * 6) / 3
    items = [
        ("Recall on tampered docs", "The catch rate - the metric that protects the bank."),
        ("Auto-clear precision", "Protects analyst time & customers from false alarms."),
        ("% auto-cleared", "The efficiency win - only counts if precision holds."),
        ("Time-to-decision", "vs. the current manual baseline."),
        ("Analyst override rate", "How often humans disagree with the band (calibration)."),
        ("Explainability", "% of flags an analyst rates clear evidence (FEAT)."),
    ]
    for i, (t, b) in enumerate(items):
        r, c = divmod(i, 3)
        d.card(MARGIN + c * (w + 6), y + r * 36, w, 32, title=t, lines=b, body_size=7.8)
    d.lead_center("Targets are set on a held-out, labelled eval set - the single most valuable asset.", y + 76)


def s_close(d):
    d.new_slide("Summary")
    d.heading("Computer flags. Human decides.", y=42, size=26)
    d.set_y(54); d.divider()
    d.set_x(MARGIN); d.set_font(FONT, "", 9.5); d.set_text_color(*TEXT)
    for b in [
        "A working, end-to-end multi-agent screening pipeline for tampered bank statements.",
        "Explainable evidence on every flag; never auto-rejects a customer - MAS FEAT by design.",
        "Three separated roles, an immutable audit trail, and an honest view of real vs. simulated.",
        "A clear, staged path from prototype to production.",
    ]:
        d.set_x(MARGIN)
        d.multi_cell(PAGE_W - 2 * MARGIN, 6, "-  " + b)
    d.ln(2)
    d.set_x(MARGIN); d.set_font(FONT, "I", 10); d.set_text_color(*MUTED)
    d.cell(0, 6, "Thank you - questions & live demo welcome.")
    d.set_xy(MARGIN, PAGE_H - 10)
    d.set_font(FONT, "", 6.5); d.set_text_color(*MUTED)
    d.cell(0, 4, "StatementGuard | AI in Finance prototype | github.com/elline-c/AI-in-finance-")


SLIDES = [
    s_title, s_problem, s_feat, s_pipeline, s_node1, s_node2, s_node3, s_mlnodes,
    s_node4, s_roles, s_demo_setup, s_demo_clean, s_demo_tampered, s_evidence,
    s_validation, s_gnn, s_realsim, s_roadmap, s_metrics, s_close,
]

if __name__ == "__main__":
    build()
