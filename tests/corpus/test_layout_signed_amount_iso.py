"""Task 8.1: tests for the ``signed_amount_iso`` renderer (Requirements 2.2-2.4, 2.8, 3.1-3.6).

Written before the layout spec exists (task 8.2). Until the layout is registered the
tests skip with a clear reason.

Fixture geometry (tests/ingest_parse/fixtures.py): one signed Amount column with right
edge 450, Balance 555. Credits print ``+SGD`` and debits ``-SGD`` before the number;
dates are ``yyyy-mm-dd``.
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from tests.corpus.helpers import (
    compare_rows,
    load_renderer,
    make_test_ledger,
    render_and_ingest,
    replayed_balances,
    text_at,
    words_of,
)

LAYOUT = "signed_amount_iso"
SEED = 8101
AMOUNT_X1, BAL_X1 = 450.0, 555.0
EDGE_TOL = 2.0
_cache: dict = {}


def _renderer_or_skip():
    renderer = load_renderer(LAYOUT)
    if renderer is None:
        pytest.skip(f"layout {LAYOUT!r} is not implemented yet (renderer registry has no entry)")
    return renderer


def _rendered(n_months: int, start: date = date(2026, 1, 1)):
    key = (n_months, start)
    if key not in _cache:
        renderer = _renderer_or_skip()
        ledger = make_test_ledger(n_months, SEED, getattr(renderer, "max_rows_per_month", None),
                                  start=start)
        _cache[key] = render_and_ingest(renderer, n_months, SEED, ledger=ledger)
    return _cache[key]


@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_reads_every_row(n_months):
    r = _rendered(n_months)
    ing = r.ingestion
    assert ing.layout_profile == LAYOUT
    problems = compare_rows(r.ledger.rows, ing.transactions, replayed_balances(r.ledger))
    assert problems == [], problems[:5]


def test_rows_print_iso_dates_and_signed_sgd_amounts():
    r = _rendered(6)
    assert r.result.printed
    for p in r.result.printed:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.date_text.strip()), p.date_text
        assert re.fullmatch(r"[+-]SGD\s?[\d,]+\.\d{2}", p.amount_text.strip()), p.amount_text
        assert "SGD" in p.balance_text, p.balance_text
    amounts = {row.row_id: row.amount for row in r.ledger.rows}
    for p in r.result.printed:
        if p.row_id in amounts:
            expected = "+SGD" if amounts[p.row_id] > 0 else "-SGD"
            assert p.amount_text.strip().startswith(expected), (p.row_id, p.amount_text)


def test_single_amount_column_credits_and_debits_share_one_right_edge():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    amounts = {row.row_id: row.amount for row in r.ledger.rows}
    seen = set()
    for row_id, cells in r.result.row_map.cells.items():
        assert set(cells) >= {"date", "amount", "balance"}
        box = cells["amount"]
        hit = [w for w in words
               if w.page == box.page and abs(w.top - box.top) <= 3.0
               and w.x0 >= box.x0 - EDGE_TOL and w.x1 <= box.x1 + EDGE_TOL]
        assert hit, box
        assert abs(max(w.x1 for w in hit) - AMOUNT_X1) <= EDGE_TOL, (row_id, box)
        seen.add(amounts[row_id] > 0)
    assert seen == {True, False}, "expected both credits and debits in a 6-month ledger"


def test_header_has_amount_column_and_no_separate_money_columns():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    texts = {w.text for w in words if w.page == 1}
    assert {"Date", "Amount", "Balance"} <= texts
    assert not ({"Withdrawals", "Deposits", "Debit", "Credit"} & texts)


def test_balance_right_edge_and_date_present():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    for row_id, cells in r.result.row_map.cells.items():
        box = cells["balance"]
        hit = [w for w in words
               if w.page == box.page and abs(w.top - box.top) <= 3.0
               and w.x0 >= box.x0 - EDGE_TOL and w.x1 <= box.x1 + EDGE_TOL]
        assert hit, box
        assert abs(max(w.x1 for w in hit) - BAL_X1) <= EDGE_TOL, (row_id, box)
        d = cells["date"]
        assert text_at(words, d.page, d.top, d.x0, d.x1)
