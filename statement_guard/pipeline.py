"""Orchestrator: wires the worker nodes into the sequenced pipeline and
produces the final auditable report.

    Node0 (anti-hack gate) -> Node1 -> Node2 -> Node3 -> Node4 -> routing
"""

from __future__ import annotations

from typing import Any

from .contracts import Finding
from .nodes.financial_logic import FinancialLogicAgent
from .nodes.ingestion import IngestionAgent
from .nodes.input_integrity import InputIntegrityAgent
from .nodes.policy_engine import PolicyEngine
from .nodes.risk_synthesis import RiskSynthesisAgent
from .nodes.visual_forensics import VisualForensicsAgent


def screen_statement(raw: dict[str, Any]) -> dict[str, Any]:
    """Run one statement through the full pipeline and return an audit report."""

    # Node 0 — input integrity / anti-hack gate (runs before anything trusts
    # the payload). Returns a sanitized copy and any anti-tamper findings; a
    # CRITICAL finding here guarantees escalation via the routing rules.
    raw, findings = InputIntegrityAgent().run(raw)

    # Node 1 — ingestion & preprocessing (now operating on sanitized input)
    stmt, ingest_findings = IngestionAgent().run(raw)
    findings += ingest_findings

    # Node 2 — visual forensics
    findings += VisualForensicsAgent().run(stmt)

    # Node 3 — financial logic & validation
    findings += FinancialLogicAgent().run(stmt)

    # Node 4 — risk scoring & synthesis (recommends a score + routing)
    summary = RiskSynthesisAgent().run(findings, stmt.data_quality)

    # Node 5 — policy engine: the authoritative decision gate. It consumes only
    # validated findings + the summary (never the raw document), applies hard
    # safe-direction overrides, and logs any instruction-injection attempt. It
    # can escalate but never relax, and never lowers the risk score — so the
    # document cannot talk its way into an auto-clear.
    policy = PolicyEngine().decide(summary, findings)
    summary["decision"] = policy["final_decision"]
    summary["policy_overrides"] = policy["policy_overrides"]
    if policy["policy_overrides"]:
        summary["routing_reasons"] = (
            summary.get("routing_reasons", []) + policy["policy_overrides"]
        )

    return {
        "case": {
            "account_holder": stmt.account_holder,
            "account_number": stmt.account_number,
            "period": f"{stmt.period_start} .. {stmt.period_end}",
            "currency": stmt.currency,
            "transactions": len(stmt.transactions),
        },
        "data_quality": stmt.data_quality.to_dict(),
        "summary": summary,
        "policy": policy,
        "findings": [f.to_dict() for f in _sort_findings(findings)],
    }


def _sort_findings(findings: list[Finding]) -> list[Finding]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return sorted(findings, key=lambda f: order[f.severity.value])
