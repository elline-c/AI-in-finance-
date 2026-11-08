"""Pure month-coverage / segmentation math.

This module is deliberately dependency-light: every function here operates on
plain ``datetime.date`` values and simple tuples, so it can be unit-tested with
the standard library alone (no pydantic / pdfplumber import needed). The
financial-logic agent wraps these helpers to emit explainable Findings.

A "period" is a ``(period_start, period_end)`` pair of ``date`` objects. The
"year-month" of a date is the ``(year, month)`` tuple; a period covers every
year-month from its start's year-month through its end's year-month inclusive.

IMPORTANT (order-independence): every function derives month identity solely
from the parsed period dates. None of them look at list position, filenames, or
upload order for identity. The ``source`` label a caller attaches is used only
for human-readable evidence, never to decide which month a period belongs to.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

# A covered span is represented by its inclusive (year, month) endpoints.
YearMonth = tuple[int, int]

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def month_name(ym: YearMonth) -> str:
    """Human-readable month label, e.g. (2024, 3) -> 'March 2024'."""
    year, month = ym
    return f"{_MONTH_NAMES[month - 1]} {year}"


def _ym(d: date) -> YearMonth:
    return (d.year, d.month)


def _ym_index(ym: YearMonth) -> int:
    """Absolute month index so year-months can be compared / differenced."""
    year, month = ym
    return year * 12 + (month - 1)


def _ym_from_index(idx: int) -> YearMonth:
    return (idx // 12, idx % 12 + 1)


def months_in_period(start: date, end: date) -> list[YearMonth]:
    """Every year-month a single period touches, inclusive, in order.

    If ``end`` precedes ``start`` the period is treated as empty (the caller is
    responsible for reporting such an inversion separately if it matters).
    """
    lo, hi = _ym_index(_ym(start)), _ym_index(_ym(end))
    if hi < lo:
        return []
    return [_ym_from_index(i) for i in range(lo, hi + 1)]


def covered_months(periods: list[tuple[Optional[date], Optional[date]]]) -> list[YearMonth]:
    """Sorted, de-duplicated set of year-months covered by all periods.

    Periods with a missing bound are skipped (they carry no month identity).
    """
    covered: set[YearMonth] = set()
    for start, end in periods:
        if start is None or end is None:
            continue
        for ym in months_in_period(start, end):
            covered.add(ym)
    return sorted(covered)


def missing_months(periods: list[tuple[Optional[date], Optional[date]]]) -> list[YearMonth]:
    """Year-months absent from the *interior* of the covered range.

    Only gaps strictly between the earliest and latest covered month count as
    missing; we never invent months before the first or after the last covered
    month. Pure function of the parsed dates (order-independent).
    """
    covered = covered_months(periods)
    if len(covered) < 2:
        return []
    lo, hi = _ym_index(covered[0]), _ym_index(covered[-1])
    present = {_ym_index(ym) for ym in covered}
    return [_ym_from_index(i) for i in range(lo, hi + 1) if i not in present]


def duplicate_months(
    periods: list[tuple[Optional[date], Optional[date], object]],
) -> list[tuple[YearMonth, list[object]]]:
    """Year-months covered by more than one period.

    Each input item is ``(start, end, source)`` where ``source`` is any label
    (page range, file name, index) used only for evidence wording. Returns a
    sorted list of ``(year_month, [sources that cover it])`` for every month
    touched by two or more periods. Detects both exact calendar-month
    duplication and overlapping date ranges (an overlap always shares at least
    one year-month).
    """
    hits: dict[YearMonth, list[object]] = {}
    for start, end, source in periods:
        if start is None or end is None:
            continue
        for ym in months_in_period(start, end):
            hits.setdefault(ym, []).append(source)
    dupes = [(ym, sources) for ym, sources in hits.items() if len(sources) > 1]
    return sorted(dupes, key=lambda item: _ym_index(item[0]))


def out_of_order_segments(
    ordered: list[tuple[Optional[date], Optional[date], object]],
) -> list[tuple[object, YearMonth, object, YearMonth]]:
    """Detect segments that appear in non-chronological document order.

    ``ordered`` is the list of ``(start, end, source)`` in the order the
    segments physically appear in the document (e.g. ascending page order). We
    compare each segment's start year-month against the previous segment's
    start year-month; a later-positioned segment covering an earlier month is an
    inversion. Returns tuples of
    ``(prev_source, prev_year_month, curr_source, curr_year_month)``.

    Note: document-position ordering is the only place physical order is used,
    and only to *detect* an ordering anomaly — month identity itself still comes
    purely from the dates.
    """
    inversions: list[tuple[object, YearMonth, object, YearMonth]] = []
    prev: Optional[tuple[YearMonth, object]] = None
    for start, end, source in ordered:
        if start is None:
            continue
        ym = _ym(start)
        if prev is not None and _ym_index(ym) < _ym_index(prev[0]):
            inversions.append((prev[1], prev[0], source, ym))
        # Advance the high-water mark so each earlier-than-latest segment is
        # reported against the latest month seen so far.
        if prev is None or _ym_index(ym) >= _ym_index(prev[0]):
            prev = (ym, source)
    return inversions
