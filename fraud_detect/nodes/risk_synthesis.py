"""Node 4 — Risk Scoring & Synthesis Agent.

Aggregates findings from ingestion (data quality), visual forensics, and
financial logic into:

    * a 1-100 **risk score** and band (LOW/MEDIUM/HIGH/VERY_HIGH),
    * a 1-100 **confidence score** and band (how sure the system is of that call),
    * a **routing** decision (AUTO_CLEAR vs HUMAN_REVIEW) with a reason,
    * a plain-language summary and the full list of reason-coded evidence.

Scoring model (transparent, auditable):
    risk = clamp(sum of finding weights, 1, 100), with diversity bonus for
    corroboration across independent agents.

Confidence model:
    starts high and is reduced by data-quality problems, thin evidence, and
    conflicting signals (tampering flags but perfect arithmetic, or vice-versa).
    Low confidence forces HUMAN_REVIEW even when risk looks low.

The machine never rejects a customer: the most severe automated outcome is a
route to a human reviewer.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from ..models import (
    AgentReport,
    ConfidenceBand,
    Finding,
    IngestionResult,
    RiskBand,
    RiskReport,
    Routing,
    Severity,
)

AGENT = "Node4:RiskSynthesis"

# Routing thresholds
AUTO_CLEAR_MAX_RISK = 25          # <= this AND high enough confidence -> auto-clear
MIN_CONFIDENCE_FOR_AUTO = 60      # below this, always send to a human


def _band_risk(score: int) -> RiskBand:
    if score <= 25:
        return RiskBand.LOW
    if score <= 50:
        return RiskBand.MEDIUM
    if score <= 75:
        return RiskBand.HIGH
    return RiskBand.VERY_HIGH


def _band_conf(score: int) -> ConfidenceBand:
    if score <= 25:
        return ConfidenceBand.LOW
    if score <= 50:
        return ConfidenceBand.MEDIUM
    if score <= 75:
        return ConfidenceBand.HIGH
    return ConfidenceBand.VERY_HIGH


def _case_id(source_path: str) -> str:
    return "CASE-" + hashlib.sha1(source_path.encode()).hexdigest()[:10].upper()


def _compute_risk(findings: list[Finding]) -> int:
    if not findings:
        return 1
    base = sum(f.weight for f in findings)
    # Corroboration bonus: distinct agents raising HIGH-severity flags.
    high_agents = {f.agent for f in findings if f.severity == Severity.HIGH}
    if len(high_agents) >= 2:
        base += 10
    score = max(1, min(100, round(base)))
    # Any material (MEDIUM+) finding means the statement is not "clean": floor the
    # score into at least the MEDIUM band so the risk band matches the routing.
    material = any(f.severity in (Severity.MEDIUM, Severity.HIGH) for f in findings)
    if material:
        score = max(score, 26)
    return score


def _compute_confidence(ing: IngestionResult, findings: list[Finding],
                        risk: int) -> tuple[int, list[str]]:
    """Return (confidence 1-100, notes explaining the confidence call)."""
    conf = 85  # start reasonably confident on a clean digital PDF
    notes: list[str] = []

    # Data-quality erosion
    dq = ing.data_quality_findings
    if dq:
        penalty = min(40, 8 * len(dq))
        conf -= penalty
        notes.append(f"-{penalty}: {len(dq)} data-quality issue(s) reduce certainty")

    if not ing.transactions:
        conf -= 25
        notes.append("-25: no transactions parsed")

    # Corroboration raises confidence
    agents_flagging = {f.agent for f in findings}
    if len(agents_flagging) >= 2:
        conf += 10
        notes.append("+10: independent agents corroborate the finding")

    # Conflicting evidence lowers confidence:
    # visual tampering flagged but arithmetic perfectly reconciles (or vice-versa)
    visual = any(f.agent.startswith("Node2") for f in findings)
    financial = any(f.agent.startswith("Node3") for f in findings)
    if visual != financial and (visual or financial):
        # only one dimension flagged -> weaker, could be a false positive
        conf -= 12
        notes.append("-12: evidence limited to a single dimension (visual OR financial)")

    # Very few, low-weight findings near a boundary -> less certain
    if 0 < len(findings) <= 1 and 20 <= risk <= 40:
        conf -= 10
        notes.append("-10: thin evidence near a risk-band boundary")

    conf = max(1, min(100, round(conf)))
    return conf, notes


def _route(risk: int, confidence: int,
           findings: list[Finding]) -> tuple[Routing, str]:
    # A statement is only auto-cleared when it is genuinely clean: low numeric
    # risk, sufficient confidence, AND no medium-or-higher severity finding.
    material = [f for f in findings
                if f.severity in (Severity.MEDIUM, Severity.HIGH)]
    if material:
        return (Routing.HUMAN_REVIEW,
                f"{len(material)} inconsistency/tampering finding(s) of MEDIUM+ severity "
                f"present; escalating to a human reviewer even though the numeric risk "
                f"is {risk}.")
    if confidence < MIN_CONFIDENCE_FOR_AUTO:
        return (Routing.HUMAN_REVIEW,
                f"Confidence ({confidence}) below the auto-clear threshold "
                f"({MIN_CONFIDENCE_FOR_AUTO}); escalating to a human reviewer.")
    if risk <= AUTO_CLEAR_MAX_RISK:
        return (Routing.AUTO_CLEAR,
                f"Low risk ({risk}) with sufficient confidence ({confidence}) and no "
                f"material findings; auto-generating an approval recommendation.")
    return (Routing.HUMAN_REVIEW,
            f"Risk score ({risk}) exceeds the auto-clear threshold "
            f"({AUTO_CLEAR_MAX_RISK}); escalating to a human reviewer.")


def synthesize(ing: IngestionResult, reports: Iterable[AgentReport]) -> RiskReport:
    findings: list[Finding] = list(ing.data_quality_findings)
    for r in reports:
        findings.extend(r.findings)

    risk = _compute_risk(findings)
    risk_band = _band_risk(risk)
    confidence, conf_notes = _compute_confidence(ing, findings, risk)
    conf_band = _band_conf(confidence)
    routing, routing_reason = _route(risk, confidence, findings)

    period = None
    if ing.statement_period_start and ing.statement_period_end:
        period = f"{ing.statement_period_start} to {ing.statement_period_end}"

    n_high = sum(1 for f in findings if f.severity == Severity.HIGH)
    summary = (
        f"{len(findings)} finding(s) across the visual-forensics and financial-logic "
        f"agents ({n_high} high-severity). Risk {risk}/100 ({risk_band.value}); "
        f"confidence {confidence}/100 ({conf_band.value}). "
        f"Confidence rationale: {'; '.join(conf_notes) if conf_notes else 'no adjustments'}."
    )

    return RiskReport(
        case_id=_case_id(ing.source_path),
        source_path=ing.source_path,
        account_holder=ing.account_holder,
        uen=ing.uen,
        statement_period=period,
        risk_score=risk,
        risk_band=risk_band,
        confidence_score=confidence,
        confidence_band=conf_band,
        routing=routing,
        routing_reason=routing_reason,
        findings=findings,
        summary=summary,
    )
