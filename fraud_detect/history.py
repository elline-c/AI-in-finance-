"""What Node 3 knows about a company from its earlier, trusted statements.

Kept in its own module so the case store (which collects the history) and Node 3
(which uses it) can share the type without importing each other.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

from .models import Transaction


@dataclass(frozen=True)
class TrailingHistory:
    """Behaviour of one company, pooled from earlier human-approved statements.

    ``gap_days``: days of silence between active days, measured inside each statement.
    ``credits``: size of every credit.
    ``monthly_credits``: total credits of each COMPLETE calendar month.
    ``case_ids``: the earlier cases this came from (quoted in evidence for audit).
    """
    gap_days: tuple[int, ...] = ()
    credits: tuple[float, ...] = ()
    monthly_credits: tuple[float, ...] = ()
    case_ids: tuple[str, ...] = ()

    def describe(self) -> str:
        n = len(self.case_ids)
        ids = ", ".join(self.case_ids)
        return f"{n} approved earlier statement{'s' if n != 1 else ''}: {ids}"


def credit_sizes(transactions: Iterable[Transaction]) -> list[float]:
    """Amounts of all credits (positive amounts)."""
    return [t.amount for t in transactions if t.amount is not None and t.amount > 0]


def _month_is_complete(year: int, month: int, start: date, end: date) -> bool:
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    return start <= first and last <= end


def monthly_credit_totals(
    transactions: Sequence[Transaction],
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
) -> list[float]:
    """Total credits for each COMPLETE calendar month, oldest first.

    Months are judged against the declared statement period. When the period is
    unknown, the first and last month with data are dropped because they may be
    partial; with fewer than three months of data nothing is returned.
    """
    totals: dict[tuple[int, int], float] = {}
    for t in transactions:
        if t.date is None or t.amount is None or t.amount <= 0:
            continue
        key = (t.date.year, t.date.month)
        totals[key] = totals.get(key, 0.0) + t.amount
    seen = sorted({(t.date.year, t.date.month) for t in transactions if t.date is not None})
    if period_start is not None and period_end is not None:
        months = [m for m in seen if _month_is_complete(m[0], m[1], period_start, period_end)]
    else:
        months = seen[1:-1] if len(seen) >= 3 else []
    return [totals.get(m, 0.0) for m in months]
