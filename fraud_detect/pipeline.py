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

from .models import RiskReport
from .nodes import financial_logic, ingestion, risk_synthesis, visual_forensics


def run_pipeline(pdf_path: str) -> RiskReport:
    # Node 1
    ing = ingestion.ingest(pdf_path)
    # Nodes 2 & 3 (independent; both consume the ingestion result)
    visual_report = visual_forensics.analyze(ing)
    financial_report = financial_logic.analyze(ing)
    # Node 4
    report = risk_synthesis.synthesize(ing, [visual_report, financial_report])
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
