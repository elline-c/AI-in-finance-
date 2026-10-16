"""Cross-case memory demo (Node 5).

Shows the pipeline *remembering* across cases:

  1. Screen a tampered statement  -> escalates to a human.
  2. A human REJECTS it (recorded in the case store with an audit trail).
  3. Screen a *second* statement that contains the same suspicious
     counterparty/amount  -> Node 5 recalls the earlier case and reports that a
     near-identical transaction was seen (and rejected) before.

This demonstrates the key capability: the system can state in the report that a
similar transaction has occurred before and whether it passed or was stopped —
so an analyst is not blind to repeats across applications.

Run:
    python demo_memory.py
"""
from __future__ import annotations

from pathlib import Path

from fraud_detect.models import ReviewDecision
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_tampered

BAR = "=" * 78


def _print_memory_findings(report, label: str) -> None:
    mem = [f for f in report.findings if f.agent.startswith("Node5")]
    print(f"\n>>> {label}: case {report.case_id}, routing={report.routing.value}")
    if not mem:
        print("    (Node 5 memory: no prior-case matches)")
        return
    print(f"    Node 5 raised {len(mem)} cross-case memory finding(s):")
    for f in mem:
        print(f"      • [{f.severity.value}] {f.code}")
        print(f"          {f.evidence}")


def main() -> None:
    # Fresh store so the demo is reproducible.
    store_path = Path("data/cases_memory_demo.json")
    if store_path.exists():
        store_path.unlink()
    store = CaseStore(str(store_path))

    print(BAR)
    print("STEP 1 — Screen a tampered statement (first time the pattern is seen)")
    print(BAR)
    gen1 = generate_tampered("samples")
    r1 = run_pipeline(gen1.path, store=store)
    store.upsert_report(r1)
    _print_memory_findings(r1, "First screening")
    print(f"    -> expected: NO memory matches (nothing seen before).")

    print()
    print(BAR)
    print("STEP 2 — A human analyst REJECTS the tampered case")
    print(BAR)
    store.record_decision(
        r1.case_id, ReviewDecision.REJECTED,
        reviewer="Jane Tan (Credit Risk)",
        reason="Confirmed altered balances and image-editing metadata.",
        feedback_for_training="True positive.",
    )
    print(f"    Recorded REJECTED on {r1.case_id}.")

    print()
    print(BAR)
    print("STEP 3 — A NEW application arrives containing the same suspicious route")
    print(BAR)
    # Re-submission: the same statement content arrives under a new file (a
    # fraudster re-using the same doctored statement on a fresh application).
    # Copying gen1 guarantees the suspicious transactions — and therefore the
    # fingerprints — are identical, so Node 5 recognises the recurrence.
    src = Path(gen1.path)
    distinct_path = src.with_name("reapplication_" + src.name)
    distinct_path.write_bytes(src.read_bytes())

    r2 = run_pipeline(str(distinct_path), store=store)
    store.upsert_report(r2)
    _print_memory_findings(r2, "Second screening (re-application)")
    print(f"    -> expected: Node 5 flags that a similar transaction was")
    print(f"       seen/rejected before, citing {r1.case_id}.")

    print()
    print(BAR)
    print("AUDIT TRAIL for the first case (shows the human decision is recorded):")
    case = store.get(r1.case_id)
    for entry in case.audit_log:
        print(f"    {entry['ts']}  {entry['event']}: {entry['detail']}")
    print(BAR)


if __name__ == "__main__":
    main()
