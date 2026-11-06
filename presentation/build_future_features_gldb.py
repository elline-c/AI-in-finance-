#!/usr/bin/env python3
"""StatementGuard — Future Features & Where It Can Grow (GLDB template).

Less technical: frames improvements as future product capabilities and features
the prototype can expand into, in the Green Link Digital Bank house style.

Usage:
    PYTHONPATH=/projects/sandbox/.pylibs python3 build_future_features_gldb.py
"""
from gldb_theme import *
from pptx.util import Inches, Pt, Emu

prs = new_prs()

# 1 — Title
title_slide(prs, "StatementGuard", "Future Features",
            subtitle="Where the prototype can grow  |  Product roadmap  |  November 2026")

# 2 — Agenda
s = content_slide(prs, "What we'll cover")
items = [
    ("Overview", "What StatementGuard does today, in one line.", ORANGE),
    ("New capabilities", "Features we can add to make it more useful.", NAVY),
    ("Better experience", "Making life easier for customers and staff.", ORANGE),
    ("Smarter over time", "How it keeps improving as it learns.", NAVY),
]
y = 1.7
for i, (t, d, col) in enumerate(items, 1):
    circle(s, Inches(0.7), Inches(y), Inches(0.75), col, number=i, size=18)
    text(s, Inches(1.9), Inches(y), Inches(3.6), Inches(0.75),
         [[R(t, 18, NAVY, True, font=HEAD_FONT)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.8), Inches(y), Inches(6.6), Inches(0.75),
         [[R(d, 14, GREY, False)]], anchor=MSO_ANCHOR.MIDDLE)
    y += 1.25
footer(s, prs, 2)

# 3 — Section divider
section_divider(prs, "01", "Where we are today")

# 4 — Today in one line (recap, non-technical)
s = content_slide(prs, "What StatementGuard does today")
card(s, Inches(0.7), Inches(1.7), Inches(5.95), Inches(3.4), heading="The job it does",
     body=["It reviews a company's bank statement for signs it has been faked or altered, "
           "before a loan officer looks at it.",
           "",
           "Clean statements are cleared automatically. Anything suspicious is passed to a "
           "person — with a plain-English explanation of what looked wrong.",
           "",
           "It never approves or rejects a customer on its own. A human always decides."],
     body_size=13)
card(s, Inches(6.85), Inches(1.7), Inches(5.8), Inches(3.4), heading="Why it matters",
     body=["• Faster decisions — staff skip the obviously-clean files.",
           "• Fewer fakes slip through to credit assessment.",
           "• Every flag comes with evidence, so decisions can be explained and trusted.",
           "• Built to the regulator's fairness and transparency expectations."],
     body_size=13, heading_color=ORANGE)
text(s, Inches(0.7), Inches(5.4), Inches(12), Inches(0.9),
     [[R("Today it is a working prototype. The pages that follow show the features it can grow into.",
         15, NAVY, True, font=HEAD_FONT)]], align=PP_ALIGN.CENTER)
footer(s, prs, 4)

# 5 — Section divider
section_divider(prs, "02", "New capabilities")

# 6 — Three families of new capabilities (template 3-pillar style)
s = content_slide(prs, "Three ways to make it more capable", two_line=False)
numbered_card(s, Inches(0.7), Inches(1.7), Inches(3.9), Inches(4.6), 1, ORANGE,
              "Read any document",
              ["Accept a photo or scan of a statement, not just a clean file.",
               "Handle more banks, more formats, and multi-page statements automatically.",
               "Pull in supporting documents (invoices, tax filings) and cross-check them."],
              body_size=13)
numbered_card(s, Inches(4.72), Inches(1.7), Inches(3.9), Inches(4.6), 2, NAVY,
              "Spot more tricks",
              ["Detect edited images and copy-pasted logos, not just altered numbers.",
               "Catch fabricated income and shell-company patterns.",
               "Recognise repeat offenders and known-bad counterparties across cases."],
              body_size=13)
numbered_card(s, Inches(8.74), Inches(1.7), Inches(3.9), Inches(4.6), 3, ORANGE,
              "Check against the real world",
              ["Verify the company against official business registries.",
               "Compare declared income to expected cash-flow patterns.",
               "Flag statements that don't match a customer's known history."],
              body_size=13)
footer(s, prs, 6)

# 7 — Feature backlog table (capabilities, value, when)
s = content_slide(prs, "Candidate features and the value they add")
widths = [Inches(4.4), Inches(5.5), Inches(2.6)]
x = Inches(0.7); y = Inches(1.7)
table_row(s, x, y, widths, ["Feature", "What it gives the business", "Effort"], header=True); y += Inches(0.72)
rows = [
    ("Photo / scan upload", "Customers apply from their phone; fewer rejected submissions", "Medium"),
    ("Image tamper detection", "Catches edited screenshots and pasted figures", "Medium"),
    ("Supporting-doc cross-check", "Confirms income claims against invoices and filings", "Medium"),
    ("Company registry verification", "Confirms the business is real and active", "Low"),
    ("Repeat-fraud memory", "Learns from past cases; stops known tricks returning", "Low"),
    ("Multi-language statements", "Serves regional customers and SMEs", "Medium"),
]
for i, (f, v, e) in enumerate(rows):
    table_row(s, x, y, widths, [f, v, e], zebra=(i % 2 == 1)); y += Inches(0.72)
footer(s, prs, 7)

# 8 — Section divider
section_divider(prs, "03", "Better experience")

# 9 — Customer vs staff experience (Today/Tomorrow two-column like template p8)
s = content_slide(prs, "A smoother experience for everyone")
pill_header(s, Inches(0.7), Inches(1.7), Inches(5.95), "For customers", fill=NAVY)
text(s, Inches(0.9), Inches(2.55), Inches(5.7), Inches(3.2),
     [[R("• Apply and upload from a phone in minutes", 14, INK)],
      [R("• Clear, friendly status updates — no jargon", 14, INK)],
      [R("• Simple prompts if something needs re-uploading", 14, INK)],
      [R("• Faster answers on their loan application", 14, INK)],
      [R("• Confidence their data is handled securely", 14, INK)]],
     space_after=10, line_spacing=1.1)
pill_header(s, Inches(6.85), Inches(1.7), Inches(5.8), "For bank staff", fill=ORANGE)
text(s, Inches(7.05), Inches(2.55), Inches(5.55), Inches(3.2),
     [[R("• A clean review screen with evidence side-by-side", 14, INK)],
      [R("• Flagged items highlighted, clutter hidden", 14, INK)],
      [R("• One click to approve, decline, or ask for more", 14, INK)],
      [R("• A dashboard of workload, risk and trends", 14, INK)],
      [R("• A full, searchable history of every decision", 14, INK)]],
     space_after=10, line_spacing=1.1)
footer(s, prs, 9)

# 10 — Trust, safety & access (features framed simply)
s = content_slide(prs, "Features that build trust")
cards = [
    ("Secure sign-in", "Staff log in with their bank credentials; each role sees only what it should.", ORANGE),
    ("Privacy by design", "Customer data is protected, access is controlled, and records are kept only as long as needed.", NAVY),
    ("Tamper-proof history", "Every decision is recorded in a sealed log that can't be quietly changed — ideal for audits.", ORANGE),
]
x = 0.7; w = 3.9
for (h, b, col) in cards:
    card(s, Inches(x), Inches(1.8), Inches(w), Inches(3.4), heading=h, body=[b],
         body_size=13.5, heading_color=col)
    x += 4.11
text(s, Inches(0.7), Inches(5.5), Inches(12), Inches(0.8),
     [[R("These features turn a prototype into something the bank and the regulator can rely on.",
         15, NAVY, True, font=HEAD_FONT)]], align=PP_ALIGN.CENTER)
footer(s, prs, 10)

# 11 — Section divider
section_divider(prs, "04", "Smarter over time")

# 12 — Learning loop (how it keeps improving)
s = content_slide(prs, "It gets smarter the more it's used")
steps = [
    ("1", "Screen", "Every statement is checked and scored.", ORANGE),
    ("2", "Decide", "Staff approve, decline, or ask for more — and say why.", NAVY),
    ("3", "Learn", "Those decisions teach the system what to watch for.", ORANGE),
    ("4", "Improve", "Detection sharpens; fewer false alarms over time.", NAVY),
]
x = 0.7; w = 2.9
for (n, h, b, col) in steps:
    box(s, Inches(x), Inches(1.9), Inches(w), Inches(2.9), fill=CARD, radius=True)
    circle(s, Inches(x + 0.25), Inches(2.15), Inches(0.75), col, number=n, size=18)
    text(s, Inches(x + 0.25), Inches(3.15), Inches(w - 0.5), Inches(0.5),
         [[R(h, 18, NAVY, True, font=HEAD_FONT)]])
    text(s, Inches(x + 0.25), Inches(3.75), Inches(w - 0.5), Inches(0.9),
         [[R(b, 12.5, INK)]], line_spacing=1.1)
    x += 3.03
# arrows between
for ax in [3.5, 6.53, 9.56]:
    text(s, Inches(ax), Inches(2.9), Inches(0.5), Inches(0.6),
         [[R(">", 24, ORANGE, True, font=HEAD_FONT)]], align=PP_ALIGN.CENTER)
text(s, Inches(0.7), Inches(5.3), Inches(12), Inches(0.9),
     [[R("A guardrail stays fixed: the system only ever learns from decisions a person made — "
         "never from its own guesses.", 14, GREY, False, italic=True)]], align=PP_ALIGN.CENTER)
footer(s, prs, 12)

# 13 — Longer-horizon ideas (ambition)
s = content_slide(prs, "Bigger ideas for the future")
ideas = [
    ("Expand beyond loans", "Use the same screening for account opening, trade finance, and onboarding."),
    ("Serve more segments", "Tailored checks for SMEs, ESG / green-loan applicants, and regional markets."),
    ("Early-warning signals", "Spot emerging fraud patterns across many applications, not just one."),
    ("Self-service for partners", "Let trusted partners pre-check documents before they reach the bank."),
]
y = 1.8
for i, (h, b) in enumerate(ideas):
    col = ORANGE if i % 2 == 0 else NAVY
    circle(s, Inches(0.7), Inches(y), Inches(0.6), col)
    text(s, Inches(1.65), Inches(y - 0.05), Inches(3.9), Inches(0.7),
         [[R(h, 16, NAVY, True, font=HEAD_FONT)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.7), Inches(y - 0.05), Inches(6.9), Inches(0.7),
         [[R(b, 13.5, INK)]], anchor=MSO_ANCHOR.MIDDLE)
    y += 1.15
footer(s, prs, 13)

# 14 — Takeaway
s = content_slide(prs, "The opportunity in one line")
box(s, Inches(0.7), Inches(2.0), Inches(11.95), Inches(2.6), fill=CARD, radius=True)
text(s, Inches(1.2), Inches(2.0), Inches(11.0), Inches(2.6),
     [[R("A proven prototype that can grow into a full document-trust platform — ", 20, NAVY, True, font=HEAD_FONT)],
      [R("faster for customers, easier for staff, and smarter every month — ", 20, NAVY, True, font=HEAD_FONT)],
      [R("while a person always makes the final call.", 20, ORANGE, True, font=HEAD_FONT)]],
     anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, line_spacing=1.15)
footer(s, prs, 14)

# 15 — Closing
closing_slide(prs, "Thank you", contact="StatementGuard  |  future features  |  November 2026")

out = __file__.rsplit("/", 1)[0] + "/statementguard-future-features.pptx"
prs.save(out)
print("wrote", out, "slides:", len(prs.slides._sldIdLst))
