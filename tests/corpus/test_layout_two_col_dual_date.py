"""Task 7.1: tests for the ``two_col_dual_date`` renderer (Requirements 2.2-2.5, 2.7, 3.1-3.6).

Written before the layout exists (task 7.2). Until the renderer is registered every test
skips with a clear reason; once it is, they run for real.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

import pytest

from fraud_detect.corpus.schema import SimulatorKnobs
from tests.corpus.helpers import (
    DEFAULT_START,
    compare_rows,
    load_renderer,
    make_test_ledger,
    page_texts_of,
    render_and_ingest,
    replayed_balances,
    words_of,
)

LAYOUT = "two_col_dual_date"
SEED = 7101
EDGE_TOL = 2.0
DEBIT_X1, CREDIT_X1, BALANCE_X1 = 420.0, 490.0, 559.0
TXN_DATE_X0, VALUE_DATE_X0, DESC_X0 = 36.0, 104.0, 172.0
DATE_RE = re.compile(r"^\d{2} [A-Z][a-z]{2} \d{4}$")

_cache: dict = {}


@pytest.fixture(scope="module")
def renderer():
    r = load_renderer(LAYOUT)
    if r is None:
        pytest.skip(f"layout {LAYOUT!r} is not implemented yet (renderer registry has no entry)")
    return r


def _rendered(renderer, n_months: int, seed: int = SEED):
    key = (n_months, seed)
    if key not in _cache:
        _cache[key] = render_and_ingest(renderer, n_months, seed)
    return _cache[key]


def _spilling(renderer):
    """Two months of 70-80 rows each: more than fits on a first page."""
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger

    if "spill" not in _cache:
        ledger = simulate_ledger(make_company(SEED, "C0002"), DEFAULT_START, 2, SEED,
                                 SimulatorKnobs(rows_per_month_range=(70, 80)), None)
        _cache["spill"] = render_and_ingest(renderer, 2, SEED, ledger=ledger)
    return _cache["spill"]


def _parse(text: str) -> Optional[date]:
    try:
        return datetime.strptime(text, "%d %b %Y").date()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Node 1 round trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_reads_every_row(renderer, n_months):
    r = _rendered(renderer, n_months)
    ing = r.ingestion
    assert ing.layout_profile == LAYOUT
    assert compare_rows(r.ledger.rows, ing.transactions, replayed_balances(r.ledger)) == []
    assert ing.statement_period_start == r.ledger.months[0]
    assert not [f for f in ing.data_quality_findings if f.severity.value in ("MEDIUM", "HIGH")]


@pytest.mark.parametrize("n_months", [6, 12])
def test_replayed_closing_equals_printed_closing(renderer, n_months):
    r = _rendered(renderer, n_months)
    assert len(r.result.section_summaries) == n_months
    bal = r.ledger.opening_balance
    by_month: dict[int, float] = {}
    for row in r.ledger.rows:
        bal = round(bal + row.amount, 2)
        by_month[row.month_index] = bal
    for i, summary in enumerate(r.result.section_summaries):
        assert summary.closing == pytest.approx(by_month[i], abs=0.005)


# --------------------------------------------------------------------------
# two dates on every row
# --------------------------------------------------------------------------
def test_every_row_prints_transaction_and_value_date(renderer):
    r = _rendered(renderer, 6)
    words = words_of(r.result.pdf)
    for row in r.ledger.rows:
        cell = r.result.row_map.cells[row.row_id]["date"]
        assert DATE_RE.match(cell.text), (row.row_id, cell.text)
        assert _parse(cell.text) == row.date
        assert cell.x0 == pytest.approx(TXN_DATE_X0, abs=EDGE_TOL)

        on_line = [w for w in words if w.page == cell.page and abs(w.top - cell.top) <= 3.0]
        txn_toks = [w for w in on_line if TXN_DATE_X0 - EDGE_TOL <= w.x0 < VALUE_DATE_X0 - EDGE_TOL]
        val_toks = [w for w in on_line if VALUE_DATE_X0 - EDGE_TOL <= w.x0 < DESC_X0 - EDGE_TOL]
        txn_text = " ".join(w.text for w in sorted(txn_toks, key=lambda w: w.x0))
        val_text = " ".join(w.text for w in sorted(val_toks, key=lambda w: w.x0))
        assert DATE_RE.match(txn_text), (row.row_id, txn_text)
        assert DATE_RE.match(val_text), (row.row_id, val_text)
        assert _parse(txn_text) == row.date
        value = _parse(val_text)
        assert value is not None and 0 <= (value - row.date).days <= 2


def test_value_dates_are_deterministic_for_a_seed(renderer):
    ledger = make_test_ledger(2, SEED)
    a = renderer.render(ledger, _style(), SEED)
    b = renderer.render(ledger, _style(), SEED)
    assert [w.text for w in words_of(a.pdf)] == [w.text for w in words_of(b.pdf)]


def _style():
    from tests.corpus.helpers import DEFAULT_STYLE
    return DEFAULT_STYLE


def test_table_header_has_both_date_columns(renderer):
    r = _rendered(renderer, 6)
    first = page_texts_of(r.result.pdf)[0]
    assert re.search(r"Txn\s+Date", first)
    assert re.search(r"Value\s+Date", first)
    for word in ("Particulars", "Debit", "Credit", "Balance"):
        assert word in first


# --------------------------------------------------------------------------
# continuation pages
# --------------------------------------------------------------------------
def test_long_month_continues_on_a_second_page(renderer):
    r = _spilling(renderer)
    assert max(r.result.row_map.pages_per_month) >= 2
    assert r.result.spilled_months
    assert compare_rows(r.ledger.rows, r.ingestion.transactions, replayed_balances(r.ledger)) == []


def test_continuation_page_repeats_header_without_period_or_brought_forward(renderer):
    r = _spilling(renderer)
    texts = page_texts_of(r.result.pdf)
    ppm = r.result.row_map.pages_per_month
    assert sum(ppm) == len(texts)
    page_no = 0
    seen_continuation = False
    for pages in ppm:
        for k in range(pages):
            text = texts[page_no + k]
            if k == 0:
                assert "Statement Period" in text
                assert re.search(r"BROUGHT\s+FORWARD", text)
            else:
                seen_continuation = True
                assert "Statement Period" not in text
                assert not re.search(r"BROUGHT\s+FORWARD", text)
                assert re.search(r"Txn\s+Date", text) and "Particulars" in text
        page_no += pages
    assert seen_continuation


def test_summary_box_equals_replay_on_spilled_month(renderer):
    r = _spilling(renderer)
    bal = r.ledger.opening_balance
    for i, summary in enumerate(r.result.section_summaries):
        rows = [x for x in r.ledger.rows if x.month_index == i]
        assert summary.opening == pytest.approx(bal, abs=0.005)
        deb = round(-sum(x.amount for x in rows if x.amount < 0), 2)
        cred = round(sum(x.amount for x in rows if x.amount > 0), 2)
        assert summary.total_withdrawals == pytest.approx(deb, abs=0.005)
        assert summary.total_deposits == pytest.approx(cred, abs=0.005)
        bal = round(bal + sum(x.amount for x in rows), 2)
        assert summary.closing == pytest.approx(bal, abs=0.005)


# --------------------------------------------------------------------------
# geometry: right edges match the fixture columns
# --------------------------------------------------------------------------
def test_right_edges_match_fixture_columns(renderer):
    r = _rendered(renderer, 6)
    words = words_of(r.result.pdf)
    for row in r.ledger.rows:
        cells = r.result.row_map.cells[row.row_id]
        want = CREDIT_X1 if row.amount > 0 else DEBIT_X1
        assert cells["amount"].x1 == pytest.approx(want, abs=EDGE_TOL), row.row_id
        assert cells["balance"].x1 == pytest.approx(BALANCE_X1, abs=EDGE_TOL), row.row_id
        # and the words really drawn in the PDF agree with the map
        for name, edge in (("amount", want), ("balance", BALANCE_X1)):
            box = cells[name]
            hit = [w for w in words if w.page == box.page and abs(w.top - box.top) <= 3.0
                   and abs(w.x1 - edge) <= EDGE_TOL]
            assert hit, (row.row_id, name, edge)
