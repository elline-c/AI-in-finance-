"""Task 9.1: tests for the ``two_col_ddmmyy_cr`` renderer (Requirements 2.2-2.4, 2.9, 3.1-3.6).

Written before the renderer exists (task 9.2). Until the layout is registered the tests
skip with a clear reason.

Fixture geometry (tests/ingest_parse/fixtures.py): Debit right edge 395, Credit 470,
balance number right edge 546, followed by a separate ``CR`` word (x about 550-560).
Dates print as ``dd-mm-yy``. ``CR`` follows every balance cell (brought-forward and
closing rows included) but is not printed in the summary box.
"""
from __future__ import annotations

import re

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

LAYOUT = "two_col_ddmmyy_cr"
SEED = 9101
OUT_X1, IN_X1, BAL_X1 = 395.0, 470.0, 546.0
EDGE_TOL = 2.0
CR_MIN_X0 = 548.0           # fixture: CR word spans x 550.6 - 560.0
SUMMARY_BOX_BOTTOM = 170.0  # fixture: summary box lines sit at top 142 and 154; table header at 183
_cache: dict = {}


def _renderer_or_skip():
    renderer = load_renderer(LAYOUT)
    if renderer is None:
        pytest.skip(f"layout {LAYOUT!r} is not implemented yet (renderer registry has no entry)")
    return renderer


def _rendered(n_months: int):
    if n_months not in _cache:
        renderer = _renderer_or_skip()
        ledger = make_test_ledger(n_months, SEED, getattr(renderer, "max_rows_per_month", None))
        _cache[n_months] = render_and_ingest(renderer, n_months, SEED, ledger=ledger)
    return _cache[n_months]


def _cr_words(words):
    return [w for w in words if w.text == "CR" and w.x0 >= CR_MIN_X0]


@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_reads_every_row(n_months):
    r = _rendered(n_months)
    ing = r.ingestion
    assert ing.layout_profile == LAYOUT
    problems = compare_rows(r.ledger.rows, ing.transactions, replayed_balances(r.ledger))
    assert problems == [], problems[:5]


def test_dates_print_dd_mm_yy():
    r = _rendered(6)
    assert r.result.printed
    for p in r.result.printed:
        assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", p.date_text.strip()), p.date_text


def test_cr_follows_every_balance_cell():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    for row_id, cells in r.result.row_map.cells.items():
        b = cells["balance"]
        hit = [w for w in words if w.text == "CR" and w.page == b.page
               and abs(w.top - b.top) <= 3.0 and w.x0 >= CR_MIN_X0]
        assert hit, f"no CR after balance of {row_id}"


def test_cr_on_brought_forward_and_closing_rows_and_not_in_summary_box():
    r = _rendered(6)
    words = words_of(r.result.pdf)
    n_months = len(r.ledger.months)
    # one CR per transaction row plus one each for the brought-forward and closing rows
    assert len(_cr_words(words)) == len(r.ledger.rows) + 2 * n_months

    for key in ("BROUGHT", "CLOSING"):
        marks = [w for w in words if w.text == key]
        assert len(marks) == n_months, key
        for m in marks:
            assert any(c.page == m.page and abs(c.top - m.top) <= 3.0 for c in _cr_words(words)), \
                f"no CR on the {key} row (page {m.page})"

    # the summary box (above the table header on each month's first page) has no CR
    first_pages = {w.page for w in words if w.text == "BROUGHT"}
    for c in _cr_words(words):
        if c.page in first_pages:
            assert c.top > SUMMARY_BOX_BOTTOM, c


def _right_edge(words, box, exclude_cr=False):
    hit = [w for w in words
           if w.page == box.page and abs(w.top - box.top) <= 3.0
           and w.x0 >= box.x0 - EDGE_TOL and w.x1 <= box.x1 + EDGE_TOL
           and not (exclude_cr and w.text == "CR")]
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
        bal = _right_edge(words, cells["balance"], exclude_cr=True)
        assert abs(bal - BAL_X1) <= EDGE_TOL, (row_id, bal)
        assert text_at(words, cells["date"].page, cells["date"].top,
                       cells["date"].x0, cells["date"].x1)
