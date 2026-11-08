"""Task 5.1: failing tests for the ``two_col_slash`` renderer (Requirements 2.2-2.5, 2.10, 3.1-3.6).

Written before the renderer exists (tasks 5.2). Until ``get_renderer("two_col_slash")``
returns a renderer, every test here is skipped with a clear reason.

Column right edges come from the fixture ``two_col_slash_fixture`` in
``tests/ingest_parse/fixtures.py``: Withdrawals 400, Deposits 475, Balance 555.
"""
from __future__ import annotations

import calendar
import re
from datetime import date

import pytest

from fraud_detect.corpus.schema import (
    CompanyProfile,
    Ledger,
    LedgerRow,
    Party,
)
from fraud_detect.models import Severity
from tests.corpus.helpers import (
    compare_rows,
    page_texts_of,
    render_and_ingest,
    render_ledger,
    replayed_balances,
    load_renderer,
)

LAYOUT = "two_col_slash"
SEED = 5151
EDGE_TOL = 2.0
X_WITHDRAWALS, X_DEPOSITS, X_BALANCE = 400.0, 475.0, 555.0
_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

_cache: dict = {}


@pytest.fixture(scope="module")
def renderer():
    r = load_renderer(LAYOUT)
    if r is None:
        pytest.skip(f"layout {LAYOUT!r} is not implemented yet (task 5.2)")
    return r


def _round_trip(renderer, n_months: int):
    key = n_months
    if key not in _cache:
        _cache[key] = render_and_ingest(renderer, n_months, SEED)
    return _cache[key]


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def _hand_ledger(n_rows: int) -> Ledger:
    """One month with ``n_rows`` simple rows (more than fit on one page)."""
    company = CompanyProfile(
        company_id="C9999", name="SLASH TEST TRADING PTE. LTD.", uen="201234567K",
        address_lines=("3 Fictional Drive", "Singapore 990001"), account_number="FX-1000-2000-3",
        archetype="services", customers=(Party("ORCHARDLEIGH TRADING"),),
        suppliers=(Party("BRINKWATER SUPPLIES"),), landlord="NORTHBROOK HOLDINGS",
        payroll_base=5000.0, seed=1,
    )
    rows = []
    for i in range(n_rows):
        credit = i % 2 == 0
        rows.append(LedgerRow(
            row_id=f"C9999-{i:04d}", month_index=0, date=date(2026, 3, 1 + (i * 27) // max(n_rows - 1, 1)),
            description=(f"FAST FROM CUSTOMER{i:03d} INV{1000 + i}" if credit else f"GIRO SUPPLIER{i:03d} REF{i:05d}"),
            amount=(1000.0 + i) if credit else -(300.0 + i), kind="receipt" if credit else "supplier",
        ))
    return Ledger(company=company, opening_balance=50000.0, months=(date(2026, 3, 1),), rows=tuple(rows))


# --------------------------------------------------------------------------
# Node 1 round trip (6 and 12 months)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_reads_every_row(renderer, n_months):
    r = _round_trip(renderer, n_months)
    ing, ledger = r.ingestion, r.ledger
    problems = compare_rows(ledger.rows, ing.transactions, replayed_balances(ledger))
    assert problems == [], problems[:5]
    assert ing.layout_profile == LAYOUT


@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_header_period_and_balances(renderer, n_months):
    r = _round_trip(renderer, n_months)
    ing, ledger = r.ingestion, r.ledger
    assert ing.account_holder == ledger.company.name
    assert ing.uen == ledger.company.uen
    assert ing.statement_period_start == ledger.months[0]
    assert ing.statement_period_end == _month_end(ledger.months[-1])
    assert ing.opening_balance == pytest.approx(ledger.opening_balance, abs=0.005)
    assert ing.closing_balance == pytest.approx(replayed_balances(ledger)[-1], abs=0.005)
    assert len(ing.segments) == len(ledger.months)
    bad = [f.code for f in ing.data_quality_findings if f.severity in (Severity.MEDIUM, Severity.HIGH)]
    assert bad == []


# --------------------------------------------------------------------------
# dates and column edges
# --------------------------------------------------------------------------
def test_dates_print_as_dd_mm_yyyy(renderer):
    r = _round_trip(renderer, 6)
    for p, row in zip(r.result.printed, r.ledger.rows):
        assert _DATE_RE.match(p.date_text), p
        assert p.date_text == row.date.strftime("%d/%m/%Y")


def test_amount_and_balance_right_edges_match_fixture_columns(renderer):
    r = _round_trip(renderer, 6)
    cells = r.result.row_map.cells
    for row in r.ledger.rows:
        entry = cells[row.row_id]
        want_amount = X_DEPOSITS if row.amount > 0 else X_WITHDRAWALS
        assert abs(entry["amount"].x1 - want_amount) <= EDGE_TOL, (row.row_id, entry["amount"])
        assert abs(entry["balance"].x1 - X_BALANCE) <= EDGE_TOL, (row.row_id, entry["balance"])


# --------------------------------------------------------------------------
# summary box
# --------------------------------------------------------------------------
def test_summary_box_equals_replay(renderer):
    r = _round_trip(renderer, 6)
    ledger, res = r.ledger, r.result
    assert len(res.section_summaries) == len(ledger.months)
    bal = ledger.opening_balance
    texts = page_texts_of(res.pdf)
    for m, summ in zip(ledger.months, res.section_summaries):
        rows = [x for x in ledger.rows if x.month_index == ledger.months.index(m)]
        deposits = round(sum(x.amount for x in rows if x.amount > 0), 2)
        withdrawals = round(-sum(x.amount for x in rows if x.amount < 0), 2)
        closing = round(bal + deposits - withdrawals, 2)
        assert summ.month == f"{m.year:04d}-{m.month:02d}"
        assert summ.opening == pytest.approx(bal, abs=0.005)
        assert summ.total_deposits == pytest.approx(deposits, abs=0.005)
        assert summ.total_withdrawals == pytest.approx(withdrawals, abs=0.005)
        assert summ.closing == pytest.approx(closing, abs=0.005)
        bal = closing
    # the values are printed in the box on the section's first page
    first = texts[0]
    s = res.section_summaries[0]
    for v in (s.opening, s.total_withdrawals, s.total_deposits, s.closing):
        assert f"{abs(v):,.2f}" in first


def test_node1_section_totals_equal_summary_box(renderer):
    r = _round_trip(renderer, 6)
    secs = r.ingestion.sections
    assert len(secs) == len(r.result.section_summaries)
    for got, want in zip(secs, r.result.section_summaries):
        assert got.opening == pytest.approx(want.opening, abs=0.005)
        assert got.closing == pytest.approx(want.closing, abs=0.005)
        assert got.total_deposits == pytest.approx(want.total_deposits, abs=0.005)
        assert got.total_withdrawals == pytest.approx(want.total_withdrawals, abs=0.005)


# --------------------------------------------------------------------------
# continuation page
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spilled(renderer):
    ledger = _hand_ledger(70)
    return render_and_ingest(renderer, 1, SEED, ledger=ledger)


def test_long_month_continues_on_second_page(spilled):
    res = spilled.result
    assert res.n_pages == 2
    assert res.row_map.pages_per_month == (2,)
    assert res.spilled_months == ("2026-03",)
    pages = {p.page for p in res.printed}
    assert pages == {1, 2}
    on_first = sum(1 for p in res.printed if p.page == 1)
    assert 30 <= on_first < len(spilled.ledger.rows)


def test_continuation_page_repeats_table_header_only(spilled):
    p1, p2 = page_texts_of(spilled.result.pdf)
    header = "Date Description Withdrawals Deposits Balance"
    flat = lambda t: re.sub(r"\s+", " ", t)
    assert header in flat(p1)
    assert header in flat(p2)
    assert "Statement Period" in p1 and "Statement Period" not in p2
    assert "BROUGHT FORWARD" in flat(p1).upper() and "BROUGHT FORWARD" not in flat(p2).upper()
    assert "CLOSING BALANCE" in flat(p2).upper()


def test_node1_reads_spilled_month_completely(spilled):
    ing, ledger = spilled.ingestion, spilled.ledger
    problems = compare_rows(ledger.rows, ing.transactions, replayed_balances(ledger))
    assert problems == [], problems[:5]
    assert len(ing.segments) == 1
    assert ing.closing_balance == pytest.approx(replayed_balances(ledger)[-1], abs=0.005)


def test_render_is_deterministic(renderer):
    ledger = _hand_ledger(10)
    a = render_ledger(renderer, ledger, SEED)
    b = render_ledger(renderer, ledger, SEED)
    assert a.row_map == b.row_map and a.printed == b.printed
