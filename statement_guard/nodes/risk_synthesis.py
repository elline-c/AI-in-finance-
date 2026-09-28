"""Node 4: Risk Scoring & Synthesis Agent.

Aggregates every finding into:
  * a risk score 1-100 (banded Low / Medium / High / Very High),
  * an inverted confidence score 1-100 (High / Medium / Low / Very Low), and
  * a routing decision.

Routing rules (from the ideation):
  * Low risk AND high confidence AND good data quality  -> AUTO-CLEAR.
  * Any CRITICAL finding, OR Medium+ risk, OR low confidence, OR poor data
    quality, OR conflicting evidence  -> HUMAN-IN-THE-LOOP.
The machine never rejects on its own — the strongest action it can take alone is
"auto-clear"; everything else escalates.
"""

from __future__ import annotations

from ..contracts import (
    DataQuality,
    Finding,
    Node,
    Severity,
    confidence_band,
    risk_band,
)

AUTO_CLEAR = "AUTO_CLEAR"
HUMAN_REVIEW = "HUMAN_IN_THE_LOOP"


class RiskSynthesisAgent:
    node = Node.RISK_SYNTHESIS

    def run(
        self, findings: list[Finding], data_quality: DataQuality
    ) -> dict:
        scored = [f for f in findings if f.severity is not Severity.INFO]

        # --- risk score 1-100 --------------------------------------------- #
        raw_points = sum(f.severity.points for f in scored)
        risk = max(1, min(100, raw_points)) if scored else 1
        band = risk_band(risk)

        # --- confidence score --------------------------------------------- #
        # Start fully confident, then subtract for the things that make the
        # model *unsure of its own score* (not for tampering itself).
        confidence = 95
        reasons: list[str] = []
        if data_quality.is_poor:
            confidence -= 35
            reasons.append("poor data quality / OCR reliability")
        if data_quality.ocr_confidence < 0.6:
            confidence -= 15
            reasons.append("very low OCR confidence")

        # Conflicting evidence: some nodes clean, others screaming.
        has_critical = any(f.severity is Severity.CRITICAL for f in scored)
        nodes_with_high = {
            f.node for f in scored if f.severity in (Severity.HIGH, Severity.CRITICAL)
        }
        conflicting = has_critical and len(nodes_with_high) == 1 and len(scored) <= 1
        if conflicting:
            confidence -= 20
            reasons.append("single isolated critical signal (needs corroboration)")

        # A borderline score near a band edge lowers confidence.
        if any(abs(risk - edge) <= 3 for edge in (25, 50, 75)):
            confidence -= 10
            reasons.append("risk score sits on a band boundary")

        confidence = max(1, min(100, confidence))
        conf_band = confidence_band(confidence)

        # --- routing decision --------------------------------------------- #
        escalate = (
            has_critical
            or risk > 25
            or confidence < 76
            or data_quality.is_poor
        )
        decision = HUMAN_REVIEW if escalate else AUTO_CLEAR

        route_reasons: list[str] = []
        if has_critical:
            route_reasons.append("at least one CRITICAL tampering indicator")
        if risk > 25:
            route_reasons.append(f"risk score {risk} exceeds auto-clear ceiling (25)")
        if confidence < 76:
            route_reasons.append(f"confidence {confidence} below high-confidence (76)")
        if data_quality.is_poor:
            route_reasons.append("data quality insufficient for an automated call")
        if not route_reasons:
            route_reasons.append(
                "no significant issues, high confidence, good data quality"
            )

        recommendation = (
            "Auto-cleared: generate approval-recommendation summary for the "
            "analyst's file. No tampering indicators found."
            if decision == AUTO_CLEAR
            else "Escalated to a credit analyst. The system does not approve or "
            "reject autonomously; a human makes the final call."
        )

        return {
            "risk_score": risk,
            "risk_band": band,
            "confidence_score": confidence,
            "confidence_band": conf_band,
            "confidence_reasons": reasons or ["clean, corroborated signals"],
            "decision": decision,
            "routing_reasons": route_reasons,
            "recommendation": recommendation,
            "findings_count": len(scored),
        }
