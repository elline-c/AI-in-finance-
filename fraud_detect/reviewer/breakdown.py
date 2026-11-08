"""Deterministic explanation of the risk score, the confidence score and the routing.

Why this lives here and not inside Node 4
-----------------------------------------
``fraud_detect/nodes/`` is fingerprinted and hash-pinned by the ML-retraining work
(``tests/ml_retraining/protected_hashes.json``), so Node 4 is left byte-identical. This
module reads Node 4's own output instead and **checks itself against it**:

* the risk lines add up to ``report.risk_score`` exactly,
* the confidence lines add up to ``report.confidence_score`` exactly,
* the routing derived here equals ``report.routing``.

When any of these fails, ``ScoreExplanation.reconciled`` is False and ``notes`` says
why, so a drift in Node 4 shows up as a visible warning, never as a wrong explanation.
The confidence adjustments are not re-implemented: they are read from Node 4's own
``_compute_confidence`` notes. The few numbers Node 4 keeps inside private functions
(``CORROBORATION_BONUS``, ``MEDIUM_FLOOR``, ``CONFIDENCE_BASE``) are compared with the
pinned values in ``tests/test_reviewer_breakdown.py``.

It costs nothing to run, needs no model and is the baseline the optional LLM reviewer
has to beat.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import (
    ConfidenceLine,
    Finding,
    IngestionResult,
    RiskLine,
    RiskReport,
    Routing,
    RoutingCause,
    ScoreExplanation,
    Severity,
)
from ..nodes import risk_synthesis as node4

# Numbers Node 4 holds inside private functions (pinned in protected_hashes.json).
CORROBORATION_BONUS = 10      # added when 2+ independent agents raise a HIGH finding
MEDIUM_FLOOR = 26             # lowest score of the MEDIUM band; applied when a MEDIUM+ finding exists
CONFIDENCE_BASE = 85          # confidence of a cleanly read digital PDF before adjustments
SCORE_MIN, SCORE_MAX = 1, 100

_MATERIAL = (Severity.MEDIUM, Severity.HIGH)
_EPS = 1e-6
_MAX_CODES_SHOWN = 5          # codes named in the routing statement (heaviest first)

# Routing causes
MATERIAL_FINDING = "MATERIAL_FINDING"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
RISK_THRESHOLD = "RISK_THRESHOLD"
AUTO_CLEAR = "AUTO_CLEAR"

# How a routing trigger is worded in a sentence (used by the template narrative and the page).
ALSO_LABELS = {
    MATERIAL_FINDING: "a MEDIUM or HIGH finding",
    LOW_CONFIDENCE: "a confidence score below the auto-clear minimum",
    RISK_THRESHOLD: "a risk score above the auto-clear limit",
}


def finding_id(index: int) -> str:
    """Citation id of the finding at position ``index`` of ``report.findings``."""
    return f"F{index + 1}"


def _g(x: float) -> str:
    return f"{x:g}"


def _where(f: Finding) -> str:
    parts = []
    if f.page:
        parts.append(f"page {f.page}")
    if f.field:
        parts.append(str(f.field))
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def risk_lines(findings: list[Finding]) -> list[RiskLine]:
    """Separate lines whose points add up to Node 4's risk score."""
    lines: list[RiskLine] = []
    for i, f in enumerate(findings):
        where = _where(f)
        lines.append(RiskLine(
            id=finding_id(i), kind="FINDING", points=float(f.weight), label=f.title,
            reading=(f"{f.title}. {f.severity.value} severity, adds {_g(f.weight)} point(s)"
                     + (f" ({where})" if where else "") + "."),
            finding_code=f.code, severity=f.severity, page=f.page, field=f.field))

    total = float(sum(f.weight for f in findings))

    high_agents = sorted({f.agent for f in findings if f.severity == Severity.HIGH})
    if len(high_agents) >= 2:
        total += CORROBORATION_BONUS
        lines.append(RiskLine(
            id="BONUS", kind="CORROBORATION_BONUS", points=float(CORROBORATION_BONUS),
            label="Corroboration bonus",
            reading=(f"{len(high_agents)} independent agents each raised a HIGH-severity "
                     f"finding ({', '.join(high_agents)}), so {CORROBORATION_BONUS} points "
                     "are added because agreement between independent checks is stronger "
                     "evidence than one check alone.")))

    rounded = round(total)
    if abs(rounded - total) > _EPS:
        lines.append(RiskLine(
            id="ROUND", kind="ROUNDING", points=rounded - total, label="Rounding",
            reading="The finding weights add up to a fractional number; the score is "
                    "rounded to a whole number."))

    limited = max(SCORE_MIN, min(SCORE_MAX, rounded))
    if limited != rounded:
        if rounded > SCORE_MAX:
            why = (f"The lines above add up to {rounded} points, but a score cannot exceed "
                   f"{SCORE_MAX}, so it is capped. The case is far past the VERY_HIGH band.")
        elif not findings:
            why = f"Nothing was raised, so the score is the minimum of {SCORE_MIN}."
        else:
            why = f"A score cannot be below {SCORE_MIN}, so it is raised to the minimum."
        lines.append(RiskLine(
            id="LIMIT", kind="RANGE_LIMIT", points=float(limited - rounded),
            label=f"Kept within {SCORE_MIN}-{SCORE_MAX}", reading=why))

    if any(f.severity in _MATERIAL for f in findings) and limited < MEDIUM_FLOOR:
        lines.append(RiskLine(
            id="FLOOR", kind="MEDIUM_FLOOR", points=float(MEDIUM_FLOOR - limited),
            label="MEDIUM-band floor",
            reading=(f"At least one MEDIUM or HIGH finding exists, so the score is lifted to "
                     f"{MEDIUM_FLOOR}, the bottom of the MEDIUM band. This keeps the band "
                     "in line with the routing: such a case always goes to a human.")))
    return lines


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
_NOTE_RE = re.compile(r"^([+-]\d+):\s*(.*)$")


def _classify_note(text: str) -> tuple[str, str]:
    """(code, plain-language reading) for one of Node 4's confidence notes."""
    low = text.lower()
    if "data-quality" in low:
        m = re.search(r"(\d+)\s+data-quality", low)
        n = m.group(1) if m else "Some"
        return ("CONF_DATA_QUALITY",
                f"{n} data-quality problem(s) were found while reading the PDF (for example "
                "unreadable rows or missing header fields), so the system trusts its own "
                "reading of the statement less.")
    if "no transactions parsed" in low:
        return ("CONF_NO_TRANSACTIONS",
                "No transaction rows could be read, so there is little to base a judgement on.")
    if "corroborate" in low:
        return ("CONF_CORROBORATION",
                "More than one independent agent raised findings. The signals support each "
                "other, so the system is more certain of the call.")
    if "single dimension" in low:
        return ("CONF_SINGLE_DIMENSION",
                "Only the visual checks or only the financial checks raised concerns. A "
                "one-sided signal is more likely to be a false alarm, so check the other "
                "side by hand.")
    if "thin evidence" in low:
        return ("CONF_THIN_EVIDENCE",
                "There is only one finding and the risk score is between 20 and 40, where "
                "a small change could move the case to another band, so the call is thinly "
                "supported.")
    return ("CONF_OTHER", f"Node 4 applied this adjustment: {text}.")


def confidence_lines(ing: IngestionResult, findings: list[Finding], risk: int
                     ) -> tuple[list[ConfidenceLine], list[str]]:
    """Lines whose points add up to Node 4's confidence score, plus any problems found.

    The adjustments come from Node 4's own ``_compute_confidence`` notes, so the rules
    are not duplicated here.
    """
    _, notes = node4._compute_confidence(ing, findings, risk)
    lines = [ConfidenceLine(
        id="A1", code="CONF_BASE", points=CONFIDENCE_BASE,
        label="Starting point",
        reading=(f"Every case starts at {CONFIDENCE_BASE}, the confidence of a cleanly "
                 "readable digital PDF. Adjustments then move it up or down."))]
    problems: list[str] = []
    total = CONFIDENCE_BASE
    for note in notes:
        m = _NOTE_RE.match(note)
        if not m:
            problems.append(f"unrecognised Node 4 confidence note: {note!r}")
            continue
        points, text = int(m.group(1)), m.group(2)
        code, reading = _classify_note(text)
        if code == "CONF_OTHER":
            problems.append(f"unknown confidence adjustment: {note!r}")
        total += points
        lines.append(ConfidenceLine(id=f"A{len(lines) + 1}", code=code, points=points,
                                    label=text, reading=reading))
    limited = max(SCORE_MIN, min(SCORE_MAX, round(total)))
    if limited != total:
        lines.append(ConfidenceLine(
            id=f"A{len(lines) + 1}", code="CONF_LIMIT", points=limited - total,
            label=f"Kept within {SCORE_MIN}-{SCORE_MAX}",
            reading=f"Scores are kept between {SCORE_MIN} and {SCORE_MAX}."))
    return lines, problems


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def routing_cause(findings: list[Finding], risk: int, confidence: int
                  ) -> tuple[RoutingCause, Routing]:
    """What actually caused the routing, using the same order of checks as Node 4."""
    material = [i for i, f in enumerate(findings) if f.severity in _MATERIAL]
    triggers: list[str] = []
    if material:
        triggers.append(MATERIAL_FINDING)
    if confidence < node4.MIN_CONFIDENCE_FOR_AUTO:
        triggers.append(LOW_CONFIDENCE)
    if risk > node4.AUTO_CLEAR_MAX_RISK:
        triggers.append(RISK_THRESHOLD)

    if not triggers:
        return (RoutingCause(
            primary=AUTO_CLEAR,
            statement=(f"Risk {risk} is within the auto-clear limit of "
                       f"{node4.AUTO_CLEAR_MAX_RISK}, confidence {confidence} meets the "
                       f"minimum of {node4.MIN_CONFIDENCE_FOR_AUTO}, and there is no MEDIUM "
                       "or HIGH finding, so an approval recommendation is generated. The "
                       "system never rejects a customer.")),
                Routing.AUTO_CLEAR)

    primary, also = triggers[0], triggers[1:]
    if primary == MATERIAL_FINDING:
        weight_by_code: dict[str, float] = {}
        for i in material:
            weight_by_code[findings[i].code] = weight_by_code.get(findings[i].code, 0.0) + findings[i].weight
        ranked = sorted(weight_by_code, key=lambda c: (-weight_by_code[c], c))
        shown = ", ".join(ranked[:_MAX_CODES_SHOWN])
        extra = len(ranked) - _MAX_CODES_SHOWN
        shown += f" and {extra} other code(s)" if extra > 0 else ""
        statement = (f"{len(material)} finding(s) of MEDIUM or HIGH severity ({shown}) "
                     f"are present. Any such finding sends the case to a human reviewer, "
                     f"whatever the numeric score is.")
    elif primary == LOW_CONFIDENCE:
        statement = (f"Confidence is {confidence}, below the {node4.MIN_CONFIDENCE_FOR_AUTO} "
                     "needed for auto-clear, so a human reviewer must look at the case even "
                     "though no MEDIUM or HIGH finding was raised.")
    else:
        statement = (f"The risk score {risk} is above the auto-clear limit of "
                     f"{node4.AUTO_CLEAR_MAX_RISK}, so a human reviewer must look at the case.")
    return (RoutingCause(primary=primary, statement=statement, also_applies=also,
                         finding_ids=[finding_id(i) for i in material]),
            Routing.HUMAN_REVIEW)


def confidence_meaning(confidence: int, cause: RoutingCause) -> str:
    """What the confidence score means for the analyst, in words.

    This is a statement of policy and of the numbers, so it is built here and not left to a
    model: it is right every time and the model can spend its words on *why* the score is
    what it is.
    """
    minimum = node4.MIN_CONFIDENCE_FOR_AUTO
    scope = ("Confidence says how sure the system is of its risk call. It does not measure how "
             "likely it is that the statement was tampered with.")
    if confidence < minimum:
        lead = (f"At {confidence}, confidence is below the auto-clear minimum of {minimum}, so a "
                "person must review this case whatever the risk score is.")
    elif cause.primary == AUTO_CLEAR:
        lead = (f"At {confidence}, confidence meets the auto-clear minimum of {minimum}, which "
                "lets a clean, low-risk case clear without a person.")
    else:
        lead = (f"At {confidence}, confidence meets the auto-clear minimum of {minimum}, so it is "
                "not what sends this case to a person. The findings and the risk score do that.")
    return f"{lead} {scope}"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def explain_scores(ing: IngestionResult, report: RiskReport) -> ScoreExplanation:
    """Breakdown of ``report``'s scores and routing, reconciled against the report."""
    findings = report.findings
    r_lines = risk_lines(findings)
    c_lines, problems = confidence_lines(ing, findings, report.risk_score)
    cause, derived_routing = routing_cause(findings, report.risk_score, report.confidence_score)

    risk_total = round(sum(l.points for l in r_lines), 6)
    conf_total = round(sum(l.points for l in c_lines), 6)

    notes = list(problems)
    if abs(risk_total - report.risk_score) > _EPS:
        notes.append(f"risk lines add up to {_g(risk_total)}, but the report says {report.risk_score}")
    if abs(conf_total - report.confidence_score) > _EPS:
        notes.append(f"confidence lines add up to {_g(conf_total)}, "
                     f"but the report says {report.confidence_score}")
    if derived_routing != report.routing:
        notes.append(f"derived routing {derived_routing.value} differs from the report's "
                     f"{report.routing.value}")
    return ScoreExplanation(
        risk_lines=r_lines, risk_total=risk_total,
        confidence_lines=c_lines, confidence_total=conf_total,
        routing=cause, confidence_meaning=confidence_meaning(report.confidence_score, cause),
        reconciled=not notes, notes=notes)


def safe_explain_scores(ing: IngestionResult, report: RiskReport) -> Optional[ScoreExplanation]:
    """``explain_scores`` that never raises: screening must not depend on the explanation."""
    import logging
    try:
        return explain_scores(ing, report)
    except Exception:  # noqa: BLE001 - explanation is an optional extra
        logging.getLogger(__name__).warning("score explanation unavailable", exc_info=True)
        return None


def format_breakdown(expl: ScoreExplanation, report: RiskReport) -> str:
    """Plain-text rendering for terminals and logs."""
    out = [f"RISK {report.risk_score}/100 ({report.risk_band.value})"]
    out += [f"  {l.points:+g}  {l.label}" + (f"  [{l.finding_code}]" if l.finding_code else "")
            for l in expl.risk_lines]
    out.append(f"  = {_g(expl.risk_total)}")
    out.append(f"CONFIDENCE {report.confidence_score}/100 ({report.confidence_band.value})")
    out += [f"  {l.points:+g}  {l.label}" for l in expl.confidence_lines]
    out.append(f"  = {_g(expl.confidence_total)}")
    if expl.confidence_meaning:
        out.append(f"  {expl.confidence_meaning}")
    out.append(f"ROUTING {report.routing.value}: {expl.routing.statement}")
    if not expl.reconciled:
        out.append("WARNING: breakdown does not reconcile: " + "; ".join(expl.notes))
    return "\n".join(out)
