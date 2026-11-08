"""Template narrative built from the breakdown: the fallback and the yardstick.

It has the same shape as the LLM's output (``ReviewerNarrative``), cites the same ids and
must pass the same validator, so the LLM version is judged against it, not against how
fluent it sounds. It needs no model, no network and gives the same text every time.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from ..models import ConfidenceLine, Claim, ReviewerNarrative, Routing
from . import validator
from .breakdown import ALSO_LABELS, CONFIDENCE_BASE, MEDIUM_FLOOR
from .context import ExplainContext, Source

MAX_BASELINE_DRIVERS = 5
MAX_BASELINE_CHECKS = 3
_HOW_TO_RE = re.compile(r"How to confirm:\s*(.+?)(?:\s+Common false alarms:|\s*$)", re.DOTALL)



def band_word(value: str) -> str:
    return value.lower().replace("_", " ")


def _sentence_fit(parts: list[str], limit: int) -> str:
    """Join sentences while they fit in ``limit`` characters."""
    out = ""
    for p in parts:
        nxt = (out + " " + p).strip()
        if len(nxt) > limit:
            break
        out = nxt
    return out


def _groups(ctx: ExplainContext) -> list[tuple[str, list[Source]]]:
    """Finding sources grouped by code, heaviest group first."""
    groups: "OrderedDict[str, list[Source]]" = OrderedDict()
    for s in ctx.finding_sources():
        groups.setdefault(next(iter(s.codes)), []).append(s)
    return sorted(groups.items(), key=lambda kv: (-sum(s.weight for s in kv[1]),
                                                  kv[1][0].finding_index or 0))


def _driver_groups(ctx: ExplainContext) -> list[tuple[str, list[Source]]]:
    """The groups to explain: heaviest total first, but never without the group that holds the
    heaviest single finding (the validator requires that one to be covered)."""
    groups = _groups(ctx)
    chosen = groups[:MAX_BASELINE_DRIVERS]
    srcs = ctx.finding_sources()
    if srcs:
        top = max(srcs, key=lambda s: (s.weight, -(s.finding_index or 0)))
        if top.weight > 0 and all(code not in top.codes for code, _ in chosen):
            chosen = chosen[:MAX_BASELINE_DRIVERS - 1] + [next(g for g in groups if g[0] in top.codes)]
    return chosen


def _pages(group: list[Source]) -> str:
    pages = sorted({p for s in group for p in s.pages})
    return (" on " + ", ".join(f"page {p}" for p in pages[:4])) if pages else ""


def _knowledge_for(ctx: ExplainContext, code: str):
    """(source id, 'how to confirm' text) of the catalogue entry for ``code``, if retrieved."""
    for n, doc in enumerate(ctx.knowledge, 1):
        if doc.doc_id == f"CODE:{code}":
            m = _HOW_TO_RE.search(doc.text)
            if m:
                return f"K{n}", " ".join(m.group(1).split())
    return None


def build_baseline(ctx: ExplainContext) -> ReviewerNarrative:
    """The deterministic narrative for ``ctx.report`` (which must carry its explanation)."""
    r, ex = ctx.report, ctx.report.explanation
    if ex is None or "ROUTE" not in ctx.sources:
        raise ValueError("the report has no score explanation to narrate")

    route = ("auto-cleared with an approval recommendation" if r.routing == Routing.AUTO_CLEAR
             else "sent to a human reviewer")
    headline = Claim(
        text=(f"Risk {r.risk_score}/100 ({band_word(r.risk_band.value)}), confidence "
              f"{r.confidence_score}/100 ({band_word(r.confidence_band.value)}); {route}."),
        cites=["ROUTE"])

    drivers: list[Claim] = []
    for code, group in _driver_groups(ctx):
        f = r.findings[group[0].finding_index]
        total = sum(1 for x in r.findings if x.code == code)     # may exceed the sources listed
        cited = group[:validator.MAX_CITES]                       # pages may only come from these
        lo, hi = min(s.weight for s in group), max(s.weight for s in group)
        if total == 1:
            text = f"{f.title} [{code}]{_pages(cited)} adds {lo:g} points."
        else:
            each = f"{lo:g} points each" if lo == hi else f"between {lo:g} and {hi:g} points each"
            text = f"{f.title} [{code}] was raised {total} times{_pages(cited)}, adding {each}."
        drivers.append(Claim(text=text, cites=[s.id for s in cited]))
    if "BONUS" in ctx.sources:
        drivers.append(Claim(
            text=(f"Two or more independent agents each raised a HIGH-severity finding, which "
                  f"adds {_points(ex, 'BONUS')} points."), cites=["BONUS"]))
    if "FLOOR" in ctx.sources:
        drivers.append(Claim(
            text=(f"A MEDIUM or HIGH finding is present, so the score is lifted to "
                  f"{MEDIUM_FLOOR}, the bottom of the MEDIUM band."), cites=["FLOOR"]))

    adjustments: list[ConfidenceLine] = ex.confidence_lines
    base, rest = adjustments[0], adjustments[1:]
    # Only the "why": which adjustments moved the score. What the score means for the analyst
    # is added by the page from ``explanation.confidence_meaning``.
    parts = [f"It starts at {CONFIDENCE_BASE}."]
    parts += [f"{l.reading.rstrip('.')} ({l.points:+g})." for l in rest]
    conf_text = _sentence_fit(parts, validator.MAX_LONG_CHARS)
    used = [base.id] + [l.id for l in rest if l.reading.rstrip('.') in conf_text]
    confidence = Claim(text=conf_text, cites=used[:validator.MAX_CITES])

    statement = ex.routing.statement
    if ex.routing.also_applies:
        statement += " Also true: " + ", ".join(ALSO_LABELS[a] for a in ex.routing.also_applies) + "."
    routing = Claim(
        text=_sentence_fit([statement], validator.MAX_LONG_CHARS) or ex.routing.statement[:validator.MAX_LONG_CHARS],
        cites=["ROUTE"] + [i for i in ex.routing.finding_ids if i in ctx.sources][:3])

    checks: list[Claim] = []
    for code, group in _groups(ctx):
        if len(checks) >= MAX_BASELINE_CHECKS:
            break
        hit = _knowledge_for(ctx, code)
        if hit:
            kid, how = hit
            text = f"{code}: {how}"
            if len(text) <= validator.MAX_CLAIM_CHARS:
                checks.append(Claim(text=text, cites=[s.id for s in group[:3]] + [kid]))

    similar: list[Claim] = []
    for n, c in enumerate(ctx.cases, 1):
        shared = (f" with {len(c.shared_codes)} shared finding code(s)" if c.shared_codes else "")
        same = " It is the same company." if c.same_company else ""
        note = f" Reviewer note: \u201c{c.reason}\u201d" if c.reason else ""
        text = (f"{c.case_id} was {c.decision.lower()} by a human reviewer{shared}.{same}{note}")
        if len(text) > validator.MAX_CLAIM_CHARS:
            text = f"{c.case_id} was {c.decision.lower()} by a human reviewer{shared}.{same}"
        similar.append(Claim(text=text, cites=[f"P{n}"]))

    return ReviewerNarrative(
        headline=headline, risk_drivers=drivers, confidence_reading=confidence,
        routing_explanation=routing, suggested_checks=checks, similar_cases=similar)


def _points(ex, line_id: str) -> str:
    for l in ex.risk_lines:
        if l.id == line_id:
            return f"{l.points:g}"
    return "0"


def citation_labels(ctx: ExplainContext) -> dict[str, str]:
    """Readable label for every id, stored with the notes so the UI can show chips."""
    return {s.id: s.label for s in ctx.sources.values()}
