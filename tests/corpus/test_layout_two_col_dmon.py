"""Task 6.1: tests for the ``two_col_dmon`` renderer (Requirements 2.2-2.4, 2.6, 2.10, 3.1-3.6).

Written before the renderer exists (task 6.2). Until the layout is registered the tests
skip with a clear reason.

Fixture geometry (tests/ingest_parse/fixtures.py): Money out right edge 392,
Money in 472, Balance 557. Each month stays on one page and prints "Page 1 of 1".
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from tests.corpus.helpers import (
    compare_rows,
    load_renderer,
    make_test_ledger,
    page_texts_of,
    render_and_ingest,
    replayed_balances,
    text_at,
    words_of,
)

LAYOUT = "two_col_dmon"
SEED = 6101
OUT_X1, IN_X1, BAL_X1 = 392.0, 472.0, 557.0
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


def test_node1_reads_statement_crossing_year_end():
    r = _rendered(6, start=date(2025, 10, 1))
    years = {m.year for m in r.ledger.months}
    assert years == {2025, 2026}
    assert r.ingestion.layout_profile == LAYOUT
    problems = compare_rows(r.ledger.rows, r.ingestion.transactions, replayed_balances(r.ledger))
    assert problems == [], problems[:5]


def test_every_monthly_page_prints_page_1_of_1():
    r = _rendered(6)
    texts = page_texts_of(r.result.pdf)
    assert len(texts) == 6 == r.result.n_pages
    for i, t in enumerate(texts, start=1):
        assert "Page 1 of 1" in t, f"page {i}"
    assert r.result.spilled_months == ()
    assert all(n == 1 for n in r.result.row_map.pages_per_month)


def test_rows_print_day_month_without_year_and_s_dollar_amounts():
    r = _rendered(6)
    assert r.result.printed
    for p in r.result.printed:
        assert re.fullmatch(r"\d{1,2} [A-Z][a-z]{2}", p.date_text.strip()), p.date_text
        assert p.amount_text.strip().startswith("S$"), p.amount_text
        assert p.balance_text.strip().startswith("S$"), p.balance_text


def test_header_lines():
    r = _rendered(6)
    first = page_texts_of(r.result.pdf)[0]
    assert "Account No." in first
    assert "Company UEN" in first
    assert re.search(
        r"Statement Period \d{2} [A-Z][a-z]{2} \d{4} to \d{2} [A-Z][a-z]{2} \d{4}", first
    ), first


def test_no_month_spills_with_row_cap():
    renderer = _renderer_or_skip()
    cap = getattr(renderer, "max_rows_per_month", None)
    assert cap is not None and cap > 0
    r = _rendered(12)
    assert r.result.n_pages == 12
    assert r.result.spilled_months == ()


def _right_edge(words, box):
    hit = [w for w in words
           if w.page == box.page and abs(w.top - box.top) <= 3.0
           and w.x0 >= box.x0 - EDGE_TOL and w.x1 <= box.x1 + EDGE_TOL]
    assert hit, box
    return max(w.x1 for w in hit)


def test_column_right_edges_match_fixture():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    amounts = {row.row_id: row.amount for row in r.ledger.rows}
    for row_id, cells in r.result.row_map.cells.items():
        amt = _right_edge(words, cells["amount"])
        expected = IN_X1 if amounts[row_id] > 0 else OUT_X1
        assert abs(amt - expected) <= EDGE_TOL, (row_id, amt, expected)
        bal = _right_edge(words, cells["balance"])
        assert abs(bal - BAL_X1) <= EDGE_TOL, (row_id, bal)
        assert text_at(words, cells["date"].page, cells["date"].top,
                       cells["date"].x0, cells["date"].x1)
