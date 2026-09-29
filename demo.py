"""Phase 6 — end-to-end demonstration.

Runs the full flow for three statements (clean, borderline, tampered):

    statement  ->  4 agents  ->  risk + confidence  ->  evidence  ->  human review

and prints a readable report per statement, then simulates a credit analyst's
decision so the audit trail is populated. Finally prints the stored cases so you
can open them in the web UI.

Run:
    python demo.py
Then start the reviewer UI:
    uvicorn fraud_detect.webapp:app --port 8000
"""
from __future__ import annotations

from fraud_detect.models import ReviewDecision
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import (
    generate_clean,
    generate_medium,
    generate_tampered,
)

BAR = "=" * 78


def _print_report(r) -> None:
    print(BAR)
    print(f"STATEMENT : {r.source_path}")
    print(f"CASE      : {r.case_id}")
    print(f"HOLDER    : {r.account_holder}  (UEN {r.uen})")
    print(f"PERIOD    : {r.statement_period}")
    print("-" * 78)
    print(f"RISK       : {r.risk_score:>3}/100  [{r.risk_band.value}]")
    print(f"CONFIDENCE : {r.confidence_score:>3}/100  [{r.confidence_band.value}]")
    print(f"ROUTING    : {r.routing.value}")
    print(f"             {r.routing_reason}")
    if r.findings:
        print("-" * 78)
        print(f"EVIDENCE   : {len(r.findings)} finding(s)")
        for f in r.findings:
            loc = f" (p.{f.page}{', ' + f.field if f.field else ''})" if f.page or f.field else ""
            print(f"  • [{f.severity.value}] {f.code}{loc}")
            print(f"      {f.evidence}")
    else:
        print("EVIDENCE   : none — clean statement, approval recommendation auto-generated")


def main() -> None:
    store = CaseStore("data/cases.json")

    samples = [
        ("Clean statement", generate_clean("samples")),
        ("Borderline (formatting only)", generate_medium("samples")),
        ("Tampered statement", generate_tampered("samples")),
    ]

    reports = []
    for label, gen in samples:
        print()
        print("#" * 78)
        print(f"#  {label}")
        print("#" * 78)
        r = run_pipeline(gen.path)
        store.upsert_report(r)
        _print_report(r)
        reports.append((label, r, gen))

    # --- simulate the human-in-the-loop step -------------------------------
    print()
    print(BAR)
    print("HUMAN-IN-THE-LOOP (simulated credit-analyst decisions)")
    print(BAR)
    for label, r, gen in reports:
        if r.routing.value == "AUTO_CLEAR":
            print(f"  {r.case_id} [{label}]: auto-cleared, no human action required.")
            continue
        # Analyst confirms tampering on the fully-tampered file, clears the borderline one.
        if not gen.is_clean and r.risk_band.value == "VERY_HIGH":
            store.record_decision(r.case_id, ReviewDecision.REJECTED, reviewer="Jane Tan (Credit Risk)",
                                  reason="Confirmed altered running balances and image-editing metadata.",
                                  feedback_for_training="True positive — keep weights.")
            print(f"  {r.case_id} [{label}]: REJECTED by analyst (tampering confirmed).")
        else:
            store.record_decision(r.case_id, ReviewDecision.APPROVED, reviewer="Jane Tan (Credit Risk)",
                                  reason="Formatting differences explained by a bank system migration; no tampering.",
                                  feedback_for_training="False positive on currency/date mix — consider softening.")
            print(f"  {r.case_id} [{label}]: APPROVED by analyst (concern cleared).")

    print()
    print(BAR)
    print("STORED CASES (open these in the web UI):")
    for c in store.list_cases():
        print(f"  {c.case_id}  risk={c.report.risk_score:>3} [{c.report.risk_band.value:9}] "
              f"routing={c.report.routing.value:12} decision={c.review.decision.value}")
    print(BAR)
    print("Start the reviewer UI with:  uvicorn fraud_detect.webapp:app --port 8000")
    print("Then browse http://127.0.0.1:8000/")


if __name__ == "__main__":
    main()
