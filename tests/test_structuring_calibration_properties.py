"""Property test for the calibrated FL_STRUCTURING rule (task 10.4).

Feature: node3-fabrication-rules, Property 11: Structuring needs a cluster in time

**Validates: Requirements 11.1, 11.2, 11.3**
"""
from __future__ import annotations

import random
import re
from datetime import date, timedelta
from typing import Optional

from hypothesis import given, settings, strategies as st

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes import financial_logic

from tests.fabrication.fixtures import make_ingestion

D0 = date(2026, 1, 1)
WINDOW_DAYS = 120
SPAN = timedelta(days=14)

_date = st.one_of(st.none(), st.integers(0, WINDOW_DAYS).map(lambda o: D0 + timedelta(days=o)))
# near-threshold credit: 9,500.00 <= amount < 10,000.00, two decimals
_near = st.tuples(_date, st.integers(950_000, 999_999).map(lambda c: c / 100))
# noise: debits near the threshold, credits below 9,500 or at/above 10,000, any date
_debit = st.tuples(_date, st.integers(950_000, 1_000_000).map(lambda c: -c / 100))
_out_credit = st.tuples(
    _date,
    st.one_of(st.integers(1, 949_999), st.integers(1_000_000, 5_000_000)).map(lambda c: c / 100),
)
_noise = st.one_of(_debit, _out_credit)


def _build(rows: list[tuple[Optional[date], float]]) -> list[Transaction]:
    return [
        Transaction(
            index=i, page=1, date=d,
            date_raw=d.strftime("%d %b %Y") if d else "",
            description="CASH DEPOSIT" if a > 0 else "CASH WITHDRAWAL", amount=a,
        )
        for i, (d, a) in enumerate(rows, start=1)
    ]


def _finding(txns: list[Transaction]):
    ing = make_ingestion(txns, period=None)
    found = [f for f in financial_logic.analyze(ing).findings if f.code == "FL_STRUCTURING"]
    assert len(found) <= 1
    return found[0] if found else None


def _is_near(t: Transaction) -> bool:
    return 9_500.00 <= t.amount < 10_000.00


def _oracle(txns: list[Transaction]) -> Optional[set[int]]:
    """Independent brute force: indices of the earliest qualifying span, else None."""
    near = [t for t in txns if _is_near(t)]
    undated = [t for t in near if t.date is None]
    dated = [t for t in near if t.date is not None]
    for start in sorted({t.date for t in dated}):
        inside = [t for t in dated if start <= t.date <= start + SPAN]
        if len(inside) + len(undated) >= 3:
            return {t.index for t in inside + undated}
    if len(undated) >= 3:
        return {t.index for t in undated}
    return None


def _rows_in(evidence: str) -> set[int]:
    return {int(n) for n in re.findall(r"#(\d+)\(", evidence)}


@settings(max_examples=200, deadline=None)
@given(
    near=st.lists(_near, max_size=8),
    noise=st.lists(_noise, max_size=8),
    extra=_noise,
    seed=st.integers(0, 10_000),
)
def test_structuring_needs_a_cluster_in_time(near, noise, extra, seed):
    """Feature: node3-fabrication-rules, Property 11: Structuring needs a cluster in time"""
    txns = _build(near + noise)
    expected = _oracle(txns)
    finding = _finding(txns)

    # reported iff the oracle finds a qualifying span
    assert (finding is not None) == (expected is not None)
    if finding is not None:
        assert finding.severity == Severity.HIGH
        assert finding.weight == 20
        assert _rows_in(finding.evidence) == expected

    # invariant to the order of rows in the input
    shuffled = list(txns)
    random.Random(seed).shuffle(shuffled)
    again = _finding(shuffled)
    assert (again is not None) == (finding is not None)
    if finding is not None:
        assert _rows_in(again.evidence) == _rows_in(finding.evidence)

    # adding a debit or an out-of-range credit never changes the result
    bigger = _build(near + noise + [extra])
    after = _finding(bigger)
    assert (after is not None) == (finding is not None)
    if finding is not None:
        assert _rows_in(after.evidence) == _rows_in(finding.evidence)
