"""Policy Engine — the final, authoritative screening-outcome gate.

Where the Risk Synthesis agent (Node 4) *scores and recommends*, the Policy
Engine *decides and enforces*. It is deliberately a separate, dumb, rule-based
layer so that no amount of clever input can talk its way past it:

  * It consumes only VALIDATED findings and the synthesis summary — never the
    raw document. The document's text cannot reach it as instructions.
  * It applies HARD OVERRIDES that only ever move the outcome in the *safe*
    direction (towards human review). A finding can force escalation; nothing
    in the input can force an auto-clear or lower the risk. This is the
    structural guarantee behind "the system cannot simply obey the document's
    request to lower the risk score."
  * It LOGS every anti-hack / tampering signal to an immutable audit trail
    (the "suspicious instruction is logged and may trigger human review" step).

This mirrors the real-world separation between a scoring model and the policy
that governs how its score is used.
"""

from __future__ import annotations

from typing import Any

from ..contracts import Finding, Severity

# Finding codes that, if present, must ALWAYS force human review regardless of
# the computed score. These are attacks on the detector or hard tamper signals.
_FORCE_REVIEW_CODES = frozenset(
    {
        "AH_EMBEDDED_INSTRUCTION",
        "AH_CONTENT_INJECTION",
        "AH_RESERVED_KEY_INJECTION",
        "AH_TRUST_SIGNAL_SPOOF",
        "AH_MALFORMED_PAYLOAD",
        "AH_ARRAY_OVERFLOW",
        "AH_KEY_EXPLOSION",
        "AH_EXCESSIVE_NESTING",
        "AH_TYPE_CONFUSION",
    }
)

AUTO_CLEAR = "AUTO_CLEAR"
HUMAN_REVIEW = "HUMAN_IN_THE_LOOP"


class PolicyEngine:
    """Final decision authority. Can only ever escalate, never relax."""

    def decide(
        self, summary: dict[str, Any], findings: list[Finding]
    ) -> dict[str, Any]:
        audit_log: list[dict[str, Any]] = []
        decision = summary.get("decision", HUMAN_REVIEW)
        overrides: list[str] = []

        # --- hard overrides (safe-direction only) ------------------------ #
        triggered = [f for f in findings if f.code in _FORCE_REVIEW_CODES]
        for f in triggered:
            audit_log.append(
                {
                    "event": "ANTI_HACK_SIGNAL",
                    "code": f.code,
                    "severity": f.severity.value,
                    "title": f.title,
                    "evidence": f.evidence,
                }
            )

        if triggered and decision == AUTO_CLEAR:
            decision = HUMAN_REVIEW
            overrides.append(
                "Policy override: an input-integrity / tampering signal was "
                "present, so the case cannot be auto-cleared."
            )

        # Any CRITICAL finding is non-negotiable escalation.
        if any(f.severity is Severity.CRITICAL for f in findings) and decision == AUTO_CLEAR:
            decision = HUMAN_REVIEW
            overrides.append(
                "Policy override: a CRITICAL finding is present; auto-clear is "
                "not permitted."
            )

        # --- invariant: the engine never auto-rejects -------------------- #
        # The strongest autonomous action remains escalation to a human.
        if decision not in (AUTO_CLEAR, HUMAN_REVIEW):
            decision = HUMAN_REVIEW
            overrides.append(
                "Policy override: unrecognised decision coerced to human review."
            )

        # The policy engine NEVER lowers the computed risk score. We assert the
        # synthesis score is passed through untouched — a document asking to be
        # marked low-risk has no path to reduce it.
        enforced_risk = summary.get("risk_score")

        instruction_attempt = any(
            f.code == "AH_EMBEDDED_INSTRUCTION" for f in findings
        )
        if instruction_attempt:
            audit_log.append(
                {
                    "event": "EMBEDDED_INSTRUCTION_IGNORED",
                    "note": "Document attempted to instruct the system; request "
                    "was logged and ignored. Risk score was NOT lowered.",
                }
            )

        return {
            "final_decision": decision,
            "enforced_risk_score": enforced_risk,
            "policy_overrides": overrides,
            "instruction_injection_detected": instruction_attempt,
            "audit_log": audit_log,
        }
