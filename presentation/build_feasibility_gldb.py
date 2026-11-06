#!/usr/bin/env python3
"""StatementGuard — Feasibility & Implementation Plan (GLDB template).

Green Link Digital Bank house style. Content grounded in PROJECT_PLAN.md
(build/buy/hybrid, 6 phases, risks, success metrics) and the codebase review.

Usage:
    PYTHONPATH=/projects/sandbox/.pylibs python3 build_feasibility_gldb.py
"""
from gldb_theme import *
from pptx.util import Inches, Pt, Emu

prs = new_prs()

# 1 — Title
title_slide(prs, "StatementGuard", "Feasibility & Plan",
            subtitle="Can we build it, should we, and how  |  Implementation roadmap  |  November 2026")

# 2 — Agenda
s = content_slide(prs, "What we'll cover")
items = [
    ("Feasibility", "Is it realistic to build — technically and commercially?", ORANGE),
    ("Approach", "Build, buy, or a bit of both?", NAVY),
    ("The plan", "Six phases from prototype to live.", ORANGE),
    ("Risks & success", "What could go wrong, and how we'll measure it.", NAVY),
]
y = 1.7
for i, (t, d, col) in enumerate(items, 1):
    circle(s, Inches(0.7), Inches(y), Inches(0.75), col, number=i, size=18)
    text(s, Inches(1.9), Inches(y), Inches(3.4), Inches(0.75),
         [[R(t, 18, NAVY, True, font=HEAD_FONT)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.6), Inches(y), Inches(6.8), Inches(0.75),
         [[R(d, 14, GREY, False)]], anchor=MSO_ANCHOR.MIDDLE)
    y += 1.25
footer(s, prs, 2)

# 3 — Section divider
section_divider(prs, "01", "Feasibility")

# 4 — Verdict (3-pillar style)
s = content_slide(prs, "Is it feasible? Yes — with staged investment")
numbered_card(s, Inches(0.7), Inches(1.7), Inches(3.9), Inches(4.5), 1, ORANGE,
              "Technically proven",
              ["The hardest part — the decision engine that scores statements and explains every "
               "flag — already works end-to-end. This isn't a research bet."], body_size=13)
numbered_card(s, Inches(4.72), Inches(1.7), Inches(3.9), Inches(4.5), 2, NAVY,
              "Commercially sensible",
              ["Buy the commodity parts (reading documents, image checks); build the parts that "
               "encode our own credit policy. Avoids reinventing the wheel."], body_size=13)
numbered_card(s, Inches(8.74), Inches(1.7), Inches(3.9), Inches(4.5), 3, ORANGE,
              "Compliance-ready",
              ["Designed around the regulator's fairness and transparency rules from day one. "
               "A person always owns the decision."], body_size=13)
footer(s, prs, 4)

# 5 — Feasibility across dimensions (table + rating)
s = content_slide(prs, "Feasibility across every dimension")
widths = [Inches(2.9), Inches(7.7), Inches(1.35)]
x = Inches(0.7); y = Inches(1.7)
table_row(s, x, y, widths, ["Dimension", "Assessment", "Rating"], header=True); y += Inches(0.72)
rows = [
    ("Technical", "Core engine complete; reading documents and image checks are add-ons", "High"),
    ("Commercial", "Hybrid approach avoids rebuilding commodity parts", "High"),
    ("Schedule", "About four to six months of staged work to a pilot", "Medium"),
    ("Operational", "Needs secure sign-in, a proper database and monitoring before going live", "Medium"),
    ("Compliance", "Fairness and transparency built in; data-handling rules to formalise", "Medium"),
    ("Data", "A set of real and known-fake examples is the longest lead-time item", "Watch"),
]
for i, (dim, note, rate) in enumerate(rows):
    table_row(s, x, y, widths, [dim, note, rate], zebra=(i % 2 == 1)); y += Inches(0.72)
footer(s, prs, 5)

# 6 — Section divider
section_divider(prs, "02", "Approach")

# 7 — Build vs buy vs hybrid
s = content_slide(prs, "Build, buy, or hybrid?")
# three option cards
data = [
    ("Buy", NAVY, ["Fast and battle-tested", "Maintained for you"],
     ["Ongoing cost", "Less control; data may leave our systems"]),
    ("Build", ORANGE, ["Full control and ownership", "Data stays in-house"],
     ["Slower", "Needs specialist expertise"]),
    ("Hybrid", GREEN, ["Buy the commodity parts", "Build the parts that are truly ours"],
     ["Some integration work", "— but the best overall fit"]),
]
x = 0.7; w = 3.9
for (name, col, pros, cons) in data:
    box(s, Inches(x), Inches(1.7), Inches(w), Inches(3.5), fill=CARD, radius=True)
    pill_header(s, Inches(x + 0.25), Inches(1.95), Inches(w - 0.5), name, fill=col)
    yy = 2.75
    for p in pros:
        text(s, Inches(x + 0.3), Inches(yy), Inches(w - 0.6), Inches(0.5),
             [[R("+  ", 13, GREEN, True), R(p, 12.5, INK)]]); yy += 0.52
    for c in cons:
        text(s, Inches(x + 0.3), Inches(yy), Inches(w - 0.6), Inches(0.5),
             [[R("-  ", 13, RED, True), R(c, 12.5, INK)]]); yy += 0.52
    x += 4.11
text(s, Inches(0.7), Inches(5.45), Inches(12), Inches(1.1),
     [[R("Recommendation: start hybrid.  ", 15, NAVY, True, font=HEAD_FONT),
       R("The prototype already is the part we'd want to own. Treat document-reading and image checks "
         "as swappable suppliers, so we're never locked in.", 14, INK)]], line_spacing=1.1)
footer(s, prs, 7)

# 8 — Where we are now (head start) — Today/Tomorrow two-column
s = content_slide(prs, "A real head start")
pill_header(s, Inches(0.7), Inches(1.7), Inches(5.95), "Already built", fill=NAVY)
text(s, Inches(0.9), Inches(2.55), Inches(5.7), Inches(3.3),
     [[R("• The full screening engine, end-to-end", 14, INK)],
      [R("• Risk scoring, routing and plain-English evidence", 14, INK)],
      [R("• Separate views for customer, analyst and loan officer", 14, INK)],
      [R("• A tamper-proof record of every decision", 14, INK)],
      [R("• A broad test suite", 14, INK)]],
     space_after=9, line_spacing=1.1)
pill_header(s, Inches(6.85), Inches(1.7), Inches(5.8), "Still to build", fill=ORANGE)
text(s, Inches(7.05), Inches(2.55), Inches(5.55), Inches(3.3),
     [[R("• Reading real scanned / photographed statements", 14, INK)],
      [R("• Deeper image-tampering checks", 14, INK)],
      [R("• Secure sign-in and a production database", 14, INK)],
      [R("• Monitoring and deployment set-up", 14, INK)],
      [R("• A supervised pilot before going live", 14, INK)]],
     space_after=9, line_spacing=1.1)
footer(s, prs, 8)

# 9 — Section divider
section_divider(prs, "03", "The plan")

# 10 — Six phases
s = content_slide(prs, "Six phases from prototype to live")
phases = [
    ("0", "Foundations", "1-2 wks", "Agree the approach; gather test examples; set targets.", ORANGE),
    ("1", "Strengthen the core", "2-3 wks", "Make the prototype sturdy and add secure sign-in.", NAVY),
    ("2", "Read real documents", "3-4 wks", "Accept real scanned and photographed statements.", ORANGE),
    ("3", "Spot real fakes", "3-5 wks", "Add deeper image-tampering detection.", NAVY),
    ("4", "Build the staff screen", "3-4 wks", "The review workspace where people decide.", ORANGE),
    ("5", "Pilot quietly", "3-4 wks", "Run alongside the manual process; measure accuracy.", NAVY),
    ("6", "Switch on gradually", "ongoing", "Auto-clear the easy cases, carefully, over time.", ORANGE),
]
y = 1.65; rh = 0.68
for (n, name, dur, desc, col) in phases:
    box(s, Inches(0.7), Inches(y), Inches(11.95), Inches(rh), fill=CARD, radius=True)
    circle(s, Inches(0.78), Inches(y + 0.04), Inches(rh - 0.08), col, number=n, size=16)
    text(s, Inches(1.6), Inches(y), Inches(2.9), Inches(rh), [[R(name, 14, NAVY, True, font=HEAD_FONT)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(4.5), Inches(y), Inches(1.2), Inches(rh), [[R(dur, 12, col, True, font=HEAD_FONT)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(5.8), Inches(y), Inches(6.7), Inches(rh), [[R(desc, 12.5, INK)]],
         anchor=MSO_ANCHOR.MIDDLE)
    y += rh + 0.08
footer(s, prs, 10)

# 11 — Timeline
s = content_slide(prs, "Indicative timeline — about 4 to 6 months to pilot")
left = 0.9; total_w = 11.6; months = 6
# month axis labels
for m in range(months + 1):
    mx = left + m * (total_w / months)
    text(s, Inches(mx - 0.3), Inches(1.65), Inches(0.9), Inches(0.35),
         [[R(("M" + str(m)) if m > 0 else "Start", 11, GREY, True)]], align=PP_ALIGN.CENTER)
box(s, Inches(left), Inches(2.05), Inches(total_w), Pt(1.2), fill=LINE)
bars = [
    ("Foundations + strengthen core", 0.0, 1.3, ORANGE),
    ("Read real documents", 1.1, 1.4, NAVY),
    ("Spot real fakes", 2.3, 1.7, NAVY),
    ("Build the staff review screen", 3.4, 1.3, ORANGE),
    ("Pilot & hardening", 4.5, 1.5, NAVY),
    ("Switch on gradually", 5.8, 0.4, GREEN),
]
unit = total_w / months
y = 2.3
for (label, start, dur, col) in bars:
    box(s, Inches(left + start * unit), Inches(y), Inches(max(dur * unit, 0.5)), Inches(0.42),
        fill=col, radius=True)
    text(s, Inches(left + start * unit + 0.1), Inches(y), Inches(dur * unit + 3.0), Inches(0.42),
         [[R(label, 11.5, WHITE if dur*unit > 2.4 else INK, True)]], anchor=MSO_ANCHOR.MIDDLE)
    y += 0.58
text(s, Inches(0.7), Inches(6.0), Inches(12), Inches(0.9),
     [[R("Some phases overlap safely. Go-live is deliberately gradual — the easy cases are cleared "
         "automatically only once the pilot proves it's accurate.", 13, GREY)]], line_spacing=1.1)
footer(s, prs, 11)

# 12 — Section divider
section_divider(prs, "04", "Risks & success")

# 13 — What it takes (resources & dependency)
s = content_slide(prs, "What it takes")
card(s, Inches(0.7), Inches(1.7), Inches(3.9), Inches(3.6), heading="The team",
     body=["• Engineering", "• Data & document specialists", "• Risk & compliance", "• Operations & credit"],
     body_size=13.5)
card(s, Inches(4.72), Inches(1.7), Inches(3.9), Inches(3.6), heading="Decisions up front", heading_color=ORANGE,
     body=["• Build, buy or hybrid", "• Where data is allowed to live", "• Budget & staffing",
           "• What 'success' looks like"], body_size=13.5)
card(s, Inches(8.74), Inches(1.7), Inches(3.9), Inches(3.6), heading="The key dependency", heading_color=NAVY,
     body=["A set of real and known-fake statement examples.",
           "It's the most valuable asset — everything is measured against it. Start collecting it in week one."],
     body_size=13.5)
text(s, Inches(0.7), Inches(5.55), Inches(12), Inches(0.8),
     [[R("Good news: we can start today with made-up clean and tampered statements — no real customer data needed yet.",
         14, NAVY, True, font=HEAD_FONT)]], align=PP_ALIGN.CENTER)
footer(s, prs, 13)

# 14 — Risks & mitigations
s = content_slide(prs, "Risks and how we manage them")
widths = [Inches(5.0), Inches(6.95)]
x = Inches(0.7); y = Inches(1.7)
table_row(s, x, y, widths, ["Risk", "How we manage it"], header=True); y += Inches(0.72)
rows = [
    ("Too few fake examples to test on", "Create controlled fakes; work with the fraud team"),
    ("Fraud tricks keep changing", "Keep checks explainable; review and refresh regularly"),
    ("False alarms annoy staff", "Start cautious; tune from real feedback; always show evidence"),
    ("Locked into one supplier", "Keep suppliers swappable; own the core ourselves"),
    ("Fairness concerns", "Test across customer groups; no sensitive attributes used"),
    ("Sensitive data exposure", "Encrypt, limit access, keep only as long as needed"),
]
for i, (rk, mt) in enumerate(rows):
    table_row(s, x, y, widths, [rk, mt], zebra=(i % 2 == 1)); y += Inches(0.72)
footer(s, prs, 14)

# 15 — Success metrics (stat-style)
s = content_slide(prs, "How we'll know it's working")
metrics = [
    ("Catch rate", "How many fakes we stop", False),
    ("False alarms", "Kept low to protect staff time", True),
    ("Auto-cleared", "The efficiency win", False),
]
x = 0.7; w = 3.9
for (v, l, hi) in metrics:
    box(s, Inches(x), Inches(1.8), Inches(w), Inches(2.3), fill=(NAVY if hi else CARD), radius=True)
    text(s, Inches(x), Inches(2.1), Inches(w), Inches(0.9),
         [[R(v, 24, (ORANGE if hi else NAVY), True, font=HEAD_FONT)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(x), Inches(3.1), Inches(w), Inches(0.8),
         [[R(l, 13, (WHITE if hi else GREY))]], align=PP_ALIGN.CENTER)
    x += 4.11
text(s, Inches(0.7), Inches(4.4), Inches(12), Inches(1.6),
     [[R("Also tracked: ", 14, NAVY, True, font=HEAD_FONT),
       R("time-to-decision versus manual, how often staff disagree with the system, and how clearly each "
         "flag is evidenced.", 14, INK)],
      [R("", 6, INK)],
      [R("A green build is not success — success is these numbers on examples the system has never seen before.",
         14, GREY, False, italic=True)]], line_spacing=1.15, space_after=6)
footer(s, prs, 15)

# 16 — Recommendation
s = content_slide(prs, "Recommendation & next steps")
box(s, Inches(0.7), Inches(1.7), Inches(5.95), Inches(1.0), fill=NAVY, radius=True)
text(s, Inches(0.95), Inches(1.7), Inches(5.6), Inches(1.0),
     [[R("Proceed — staged, with a go / no-go gate after the pilot", 15, WHITE, True, font=HEAD_FONT)]],
     anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)
text(s, Inches(0.72), Inches(2.95), Inches(5.95), Inches(2.3),
     [[R("Feasible and de-risked. Move ahead hybrid, phase by phase. Only widen automatic clearing once "
         "the pilot proves it's accurate on unseen examples.", 14, INK)]], line_spacing=1.15)
card(s, Inches(6.85), Inches(1.7), Inches(5.8), Inches(3.55), heading="This week", heading_color=ORANGE,
     body=["1.  Decide build / buy / hybrid and where data can live.",
           "2.  Start gathering real and known-fake examples.",
           "3.  Approve Phase 1 — strengthen the prototype and add secure sign-in.",
           "4.  Agree the success targets so 'done' is measurable."], body_size=13.5)
text(s, Inches(0.7), Inches(5.55), Inches(12), Inches(0.8),
     [[R("The principle never changes: the computer flags, a person decides.", 15, NAVY, True, font=HEAD_FONT)]],
     align=PP_ALIGN.CENTER)
footer(s, prs, 16)

# 17 — Closing
closing_slide(prs, "Thank you", contact="StatementGuard  |  feasibility & plan  |  November 2026")

out = __file__.rsplit("/", 1)[0] + "/statementguard-feasibility-and-plan.pptx"
prs.save(out)
print("wrote", out, "slides:", len(prs.slides._sldIdLst))
