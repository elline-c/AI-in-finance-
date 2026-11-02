"""Task 4.1: tests for the ``generator_v1`` renderer (Requirements 2.2, 2.3, 2.5, 2.10, 2.11, 3.1-3.6).

Written before the renderer exists (task 4.2). Until the registry has a ``generator_v1``
entry every test here skips with a clear reason.
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from fraud_detect.corpus.schema import SimulatorKnobs
from tests.corpus.helpers import (
    DEFAULT_START,
    compare_rows,
    load_renderer,
    page_texts_of,
    render_and_ingest,
    replayed_balances,
    words_of,
)

LAYOUT = "generator_v1"
SEED = 8101
_cache: dict = {}


def _renderer_or_skip():
    renderer = load_renderer(LAYOUT)
    if renderer is None:
        pytest.skip(f"layout {LAYOUT!r} is not implemented yet (task 4.2)")
    return renderer


def _rendered(n_months: int):
    if n_months not in _cache:
        _cache[n_months] = render_and_ingest(_renderer_or_skip(), n_months, SEED)
    return _cache[n_months]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _money(v: float) -> str:
    return f"{v:,.2f}"


@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_reads_every_row(n_months):
    r = _rendered(n_months)
    ing = r.ingestion
    assert compare_rows(r.ledger.rows, ing.transactions, replayed_balances(r.ledger)) == []
    assert replayed_balances(r.ledger)[-1] == pytest.approx(ing.transactions[-1].balance)


@pytest.mark.parametrize("n_months", [6, 12])
def test_node1_picks_profile_and_header(n_months):
    r = _rendered(n_months)
    ing = r.ingestion
    assert ing.layout_profile == LAYOUT
    assert ing.account_holder is not None
    assert ing.account_holder.upper() == r.ledger.company.name.upper()
    assert ing.uen == r.ledger.company.uen
    first = r.ledger.months[0]
    last = r.ledger.months[-1]
    assert ing.statement_period_start == date(first.year, first.month, 1)
    assert ing.statement_period_end is not None
    assert (ing.statement_period_end.year, ing.statement_period_end.month) == (last.year, last.month)
    assert len(ing.sections) == n_months


@pytest.mark.parametrize("n_months", [6, 12])
def test_no_medium_or_high_data_quality_finding(n_months):
    ing = _rendered(n_months).ingestion
    bad = [f.code for f in ing.data_quality_findings if str(getattr(f.severity, "value", f.severity)) in ("MEDIUM", "HIGH")]
    assert bad == []


def test_replayed_closing_balance_equals_printed_closing():
    r = _rendered(6)
    assert r.ingestion.sections[-1].closing == pytest.approx(replayed_balances(r.ledger)[-1])


def test_period_line_printed_once_per_monthly_section():
    r = _rendered(6)
    text = "\n".join(page_texts_of(r.result.pdf))
    assert len(re.findall(r"Statement Period:", text)) == 6


def test_opening_and_closing_lines_equal_replay():
    r = _rendered(6)
    text = _norm("\n".join(page_texts_of(r.result.pdf)))
    summaries = r.result.section_summaries
    assert len(summaries) == 6
    assert summaries[0].opening == pytest.approx(r.ledger.opening_balance)
    bal = replayed_balances(r.ledger)
    assert summaries[-1].closing == pytest.approx(bal[-1])
    for prev, nxt in zip(summaries, summaries[1:]):
        assert nxt.opening == pytest.approx(prev.closing)
    for s in summaries:
        assert re.search(r"Opening Balance:\s*(SGD)?\s*" + re.escape(_money(s.opening)), text)
        assert re.search(r"Closing Balance:\s*(SGD)?\s*" + re.escape(_money(s.closing)), text)


def test_footer_does_not_disturb_row_reading():
    r = _rendered(6)
    for text in page_texts_of(r.result.pdf):
        assert "specimen" in text.lower()
    assert len(r.ingestion.transactions) == len(r.ledger.rows)
    assert not any("specimen" in (t.description or "").lower() for t in r.ingestion.transactions)
    # footer sits below every table row on its page
    words = words_of(r.result.pdf)
    rows_bottom = {}
    for box in (c for cells in r.result.row_map.cells.values() for c in cells.values()):
        rows_bottom[box.page] = max(rows_bottom.get(box.page, 0.0), box.top)
    for w in words:
        if w.text.lower().startswith("specimen"):
            assert w.top > rows_bottom.get(w.page, 0.0)


def test_long_month_continues_on_second_page():
    renderer = _renderer_or_skip()
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger

    company = make_company(SEED, "C0002")
    knobs = SimulatorKnobs(rows_per_month_range=(110, 120))
    ledger = simulate_ledger(company, DEFAULT_START, 2, SEED, knobs, None)
    r = render_and_ingest(renderer, 2, SEED, ledger=ledger)
    assert r.result.n_pages > 2
    assert max(r.result.row_map.pages_per_month) >= 2
    assert compare_rows(ledger.rows, r.ingestion.transactions, replayed_balances(ledger)) == []
    assert r.ingestion.layout_profile == LAYOUT
