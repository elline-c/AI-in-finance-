"""Orchestrator — wires the worker nodes into a sequential pipeline.

    PDF ──► Node 1 Ingestion ──► Security Layer (scan extracted text) ─┐
                                 Node 2 Visual Forensics ──────────────┤
                                 Node 3 Financial Logic ───────────────┴─► Node 4 Risk Synthesis
                                                                                 │
                                                          risk & confidence ─────┤
                                                                                 ▼
                                   ┌───────────── routing ─────────────┐
                                   │                                   │
                          AUTO_CLEAR (low risk,               HUMAN_REVIEW (medium/high risk
                          high confidence):                   OR low confidence):
                          auto approval-recommendation        credit analyst reviews
                                                              (machine never rejects)

Conditional decisions (per the design):
    * No significant issues            -> LOW risk    -> auto-clear
    * Some inconsistencies             -> MEDIUM risk -> human review
    * Strong tampering indicators      -> HIGH/VERY_HIGH -> human review
    * Embedded instruction / low confidence / poor data -> human review
"""
from __future__ import annotations

from typing import Optional

from .models import AgentReport, RiskReport
from .nodes import (
    financial_logic,
    ingestion,
    memory,
    risk_synthesis,
    security,
    visual_forensics,
)
from .store import CaseStore


def run_pipeline(pdf_path: str, store: Optional[CaseStore] = None) -> RiskReport:
    """Run one statement through the full pipeline and return the risk report.

    When a ``store`` is supplied, the cross-case memory agent (Node 5) is run:
    the current statement's suspicious transactions are compared against prior,
    human-decided cases, and the resulting report carries fingerprints so future
    screenings can recognise a recurrence.
    """
    # Node 1 — ingestion: extract the (untrusted) document text, metadata, etc.
    ing = ingestion.ingest(pdf_path)

    # Security layer — scan the extracted text for embedded instructions and
    # record their location + extracted text as evidence. The text is never
    # obeyed; it is only inspected and reported.
    security_report = AgentReport(agent=security.AGENT, findings=security.scan(ing))

    # Nodes 2 & 3 (independent; both consume the ingestion result)
    visual_report = visual_forensics.analyze(ing)
    financial_report = financial_logic.analyze(ing)

    reports = [security_report, visual_report, financial_report]

    # Node 5 — cross-case memory (only when a store is provided). It matches the
    # suspicious rows flagged by nodes 2 & 3 against prior verified cases.
    detection_findings = visual_report.findings + financial_report.findings
    if store is not None:
        prior_memory = store.recall_prior_fingerprints()
        memory_report = memory.analyze(
            ing, prior_memory, current_findings=detection_findings
        )
        reports.append(memory_report)

    # Node 4 — risk synthesis
    report = risk_synthesis.synthesize(ing, reports)

    # Always attach fingerprints of this statement's suspicious transactions so
    # the report can be persisted and recalled by future screenings, whether or
    # not a store was passed to this run.
    report.transaction_fingerprints = memory.build_fingerprints(
        ing, detection_findings
    )

    return report


if __name__ == "__main__":  # pragma: no cover
    import sys

    from .synthetic import generate_clean, generate_tampered

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        generate_clean("samples")
        generate_tampered("samples")
        paths = ["samples/clean_201812345A.pdf", "samples/tampered_200923456B.pdf"]

    for p in paths:
        r = run_pipeline(p)
        print("=" * 78)
        print(f"{p}")
        print(f"  case_id       : {r.case_id}")
        print(f"  holder / uen  : {r.account_holder}  /  {r.uen}")
        print(f"  period        : {r.statement_period}")
        print(f"  RISK          : {r.risk_score}/100  ({r.risk_band.value})")
        print(f"  CONFIDENCE    : {r.confidence_score}/100  ({r.confidence_band.value})")
        print(f"  ROUTING       : {r.routing.value}")
        print(f"  reason        : {r.routing_reason}")
        print(f"  findings      : {len(r.findings)}")
        for f in r.findings:
            print(f"     - [{f.severity.value:6}] {f.code:28} (w={f.weight}) {f.title}")
