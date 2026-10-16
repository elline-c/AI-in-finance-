"""Orchestrator — wires the 4 worker nodes into a sequential pipeline.

    PDF ──► Node 1 Ingestion ──► Node 2 Visual Forensics ─┐
                                 Node 3 Financial Logic ──┴─► Node 4 Risk Synthesis
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
    * Low AI confidence / poor data    -> human review regardless of risk band
"""
from __future__ import annotations

from typing import Optional

from .models import RiskReport
from .nodes import financial_logic, ingestion, memory, risk_synthesis, visual_forensics
from .store import CaseStore


def run_pipeline(pdf_path: str, store: Optional[CaseStore] = None) -> RiskReport:
    """Run one statement through the full pipeline.

    If a ``store`` is provided, the cross-case Memory agent (Node 5) is run: it
    reads the fingerprints of all *prior* cases from the store and flags any
    transaction that recurs from an earlier screening. Without a store the
    pipeline behaves exactly as before (no memory).
    """
    # Node 1
    ing = ingestion.ingest(pdf_path)
    # Nodes 2 & 3 (independent; both consume the ingestion result)
    visual_report = visual_forensics.analyze(ing)
    financial_report = financial_logic.analyze(ing)
    agent_reports = [visual_report, financial_report]

    # Findings from the forensic agents — used both to focus memory matching and
    # to decide which (suspicious) transactions are worth remembering.
    forensic_findings = visual_report.findings + financial_report.findings

    # Node 5 — cross-case memory (only when a store is available to recall from)
    if store is not None:
        current_id = risk_synthesis._case_id(ing.source_path)
        prior_memory = store.recall_prior_fingerprints(exclude_case_id=current_id)
        memory_report = memory.analyze(
            ing, prior_memory,
            current_case_id=current_id,
            current_findings=forensic_findings,
        )
        agent_reports.append(memory_report)

    # Node 4 — synthesis
    report = risk_synthesis.synthesize(ing, agent_reports)

    # Attach fingerprints of this case's *suspicious* transactions so future
    # screenings can recall it (routine transactions are not remembered).
    report.transaction_fingerprints = memory.build_fingerprints(ing, forensic_findings)
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
