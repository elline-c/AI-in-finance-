"""Self-consistency checks for the fabrication fixtures."""
from __future__ import annotations

from datetime import date

from tests.fabrication import fixtures as fx

KEYWORDS = ("RENT", "CPF", "PAYROLL", "LOAN", "TRANSFER", "INTEREST")


def _replays(txns, opening):
    bal = opening
    for t in txns:
        bal = round(bal + t.amount, 2)
        assert t.balance == bal


def _in_order(txns):
    pairs = [(t.date, int(fx.extract_ref(t.description)[3:])) for t in fx.credit_rows(txns)]
    return all(a[1] <= b[1] for a, b in zip(pairs, pairs[1:]))


def test_business_file_shape():
    txns, ing = fx.business_file()
    assert [t.index for t in txns] == list(range(1, len(txns) + 1))
    assert all(a.date <= b.date for a, b in zip(txns, txns[1:]))
    assert fx.PERIOD_START <= txns[0].date and txns[-1].date <= fx.PERIOD_END
    _replays(txns, fx.OPENING_BALANCE)
    assert ing.closing_balance == txns[-1].balance
    assert ing.statement_period_start == date(2026, 4, 1)
    for ym in fx.MONTHS:
        n = sum(1 for t in fx.credit_rows(txns) if (t.date.year, t.date.month) == ym)
        assert 10 <= n <= 14
    assert _in_order(txns)
    assert len({fx.extract_ref(t.description) for t in fx.credit_rows(txns)}) == len(fx.credit_rows(txns))


def test_keywords_only_where_intended():
    txns, _ = fx.business_file()
    for t in txns:
        hits = [k for k in KEYWORDS if k in t.description.split() or k in t.description]
        if t.amount > 0 or not t.description.startswith(("RENT", "PAYROLL", "CPF")):
            assert hits == [], t.description


def test_helpers():
    txns, _ = fx.business_file()
    opening = fx.opening_of(txns)
    row = fx.credit_rows(txns)[10]
    n = len(txns)
    copied = fx.insert_copy(txns, row.index, date(2026, 6, 20), 1234.56)
    assert len(copied) == n + 1
    _replays(copied, opening)
    new = [t for t in copied if t.amount == 1234.56][0]
    assert fx.extract_ref(new.description) == fx.extract_ref(row.description)
    assert [t.index for t in copied] == list(range(1, n + 2))

    rows = [fx.credit_rows(txns)[5].index, fx.credit_rows(txns)[40].index]
    moved = fx.move_references_out_of_order(txns, rows)
    assert not _in_order(moved)
    assert fx.customer_of(moved[rows[0] - 1].description) == fx.customer_of(txns[rows[0] - 1].description)

    scaled = fx.scale_month_credits(txns, (2026, 5), 2.0)
    _replays(scaled, opening)
    assert sum(t.amount for t in fx.credit_rows(scaled) if t.date.month == 5) > \
        1.9 * sum(t.amount for t in fx.credit_rows(txns) if t.date.month == 5)
    assert txns[0].balance == fx.business_file()[0][0].balance  # input untouched
