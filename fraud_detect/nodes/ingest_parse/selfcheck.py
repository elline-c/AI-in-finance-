"""Node 1 self-check (Requirement 9).

Separates our own reading mistakes from document fraud: when every row's
running balance reconciles row by row but a printed section total disagrees
with the parsed sum, we emit ``DQ_PARSE_INCONSISTENT`` (LOW, weight 3).
Row-by-row arithmetic problems are left to Node 3 (FL_BALANCE_BREAK /
FL_CLOSING_MISMATCH) and are never reported here.
"""
from __future__ import annotations

from typing import Any

from fraud_detect.models import Finding, Severity, Transaction

TOLERANCE = 0.01
_EPS = 1e-9  # float slack so a difference of exactly 0.01 is not flagged


def _rows_reconcile(opening: float | None, rows: list[Transaction]) -> bool:
    """True when each row's balance equals the previous balance plus its amount."""
    if not rows:
        return True
    prev = opening if opening is not None else rows[0].balance - rows[0].amount
    for r in rows:
        if abs(round(prev + r.amount, 2) - r.balance) > TOLERANCE + _EPS:
            return False
        prev = r.balance
    return True


def self_check(section: Any, rows: list[Transaction]) -> list[Finding]:
    """Compare printed section totals with parsed sums (see module docstring)."""
    from fraud_detect.nodes.ingestion import AGENT  # lazy: avoids an import cycle

    # Replaying from opening to the printed closing is informational only; any
    # closing mismatch is Node 3's FL_CLOSING_MISMATCH, so it is not reported.
    if not _rows_reconcile(section.opening, rows):
        return []

    debits = round(sum(-r.amount for r in rows if r.amount < 0), 2)
    credits = round(sum(r.amount for r in rows if r.amount > 0), 2)

    findings: list[Finding] = []
    for label, printed, parsed in (
        ("Total Withdrawals", section.printed_total_withdrawals, debits),
        ("Total Deposits", section.printed_total_deposits, credits),
    ):
        if printed is None:
            continue
        if abs(printed - parsed) > TOLERANCE + _EPS:
            findings.append(Finding(
                agent=AGENT,
                code="DQ_PARSE_INCONSISTENT",
                title="Printed total differs from parsed rows",
                severity=Severity.LOW,
                weight=3,
                page=getattr(section, "page", None),
                evidence=(
                    f"Printed {label} is {printed:.2f} but the parsed rows sum to "
                    f"{parsed:.2f} even though every row balance reconciles. "
                    "We may have misread this page; please check it."
                ),
            ))
    return findings
