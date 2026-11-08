"""Task 10.3: property tests for the all-layout round trip and the geometry.

Feature: synthetic-corpus-generator, Property 1: Node 1 reads every genuine generated statement completely
Feature: synthetic-corpus-generator, Property 7: Printed cells sit where the layout says, and the footer is there

Each example renders a PDF, so the example budget is small.
"""
from __future__ import annotations

import calendar
import re
from datetime import date

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.models import Severity
from tests.corpus.helpers import (
    compare_rows,
    load_renderer,
    page_texts_of,
    render_and_ingest,
    replayed_balances,
    text_at,
    words_of,
)

EDGE_TOL = 2.0
MM = 72 / 25.4

# Right edges (points) of the amount / balance columns, from tests/ingest_parse/fixtures.py
# (as used by the per-layout tests). Credits and debits use their own column where the layout has two.
# (debit_x1, credit_x1, balance_x1)
RIGHT_EDGES = {
    "two_col_dmon": (392.0, 472.0, 557.0),
    "two_col_slash": (400.0, 475.0, 555.0),
    "two_col_dual_date": (420.0, 490.0, 559.0),
    "signed_amount_iso": (450.0, 450.0, 555.0),
    "two_col_ddmmyy_cr": (395.0, 470.0, 546.0),
}
# generator_v1 values are right-aligned at (amount_x1, balance_x1)
GEN_V1_RIGHT = (150 * MM, 195 * MM)

SETTINGS = settings(
    max_examples=6, deadline=None, derandomize=True,
    suppress_health_check=list(HealthCheck),
)
seeds = st.integers(min_value=0, max_value=10_000)
months = st.integers(min_value=6, max_value=12)

pytestmark = pytest.mark.parametrize("layout", LAYOUTS)


def _renderer_or_skip(layout: str):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout!r} is not implemented yet")
    return renderer


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def _cell_words(words, box):
    return [w for w in words
            if w.page == box.page and abs(w.top - box.top) <= 3.0
            and w.x0 >= box.x0 - EDGE_TOL and w.x1 <= box.x1 + EDGE_TOL and w.text != "CR"]


@SETTINGS
@given(seed=seeds, n_months=months)
def test_property_1_node1_reads_every_genuine_statement(layout, seed, n_months):
    """**Validates: Requirements 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**"""
    renderer = _renderer_or_skip(layout)
    r = render_and_ingest(renderer, n_months, seed)
    ing, ledger = r.ingestion, r.ledger
    balances = replayed_balances(ledger)

    problems = compare_rows(ledger.rows, ing.transactions, balances)
    assert problems == [], problems[:5]
    assert ing.layout_profile == layout
    assert ing.closing_balance == pytest.approx(balances[-1], abs=0.005)
    assert ing.transactions[-1].balance == pytest.approx(balances[-1], abs=0.005)
    assert (ing.account_holder or "").upper() == ledger.company.name.upper()
    assert ing.uen == ledger.company.uen
    assert ing.statement_period_start == ledger.months[0]
    assert ing.statement_period_end == _month_end(ledger.months[-1])
    bad = [f.code for f in ing.data_quality_findings if f.severity in (Severity.MEDIUM, Severity.HIGH)]
    assert bad == []


@SETTINGS
@given(seed=seeds, n_months=months)
def test_property_7_cells_sit_on_layout_columns_and_footer_present(layout, seed, n_months):
    """**Validates: Requirements 2.10, 2.11, 2.12**"""
    renderer = _renderer_or_skip(layout)
    r = render_and_ingest(renderer, n_months, seed, ingest=False)
    ledger, res = r.ledger, r.result
    words = words_of(res.pdf)
    cells = res.row_map.cells

    assert set(cells) == {row.row_id for row in ledger.rows}
    for row in ledger.rows:
        entry = cells[row.row_id]
        for name in ("date", "description", "amount", "balance"):
            box = entry[name]
            found = re.sub(r"\s+", " ", text_at(words, box.page, box.top, box.x0, box.x1)).strip()
            assert found == re.sub(r"\s+", " ", box.text).strip(), (row.row_id, name, found, box)

        amount_w = _cell_words(words, entry["amount"])
        balance_w = _cell_words(words, entry["balance"])
        assert amount_w and balance_w, row.row_id
        if layout == "generator_v1":
            # The renderer right-aligns values under the header words (150 mm / 195 mm);
            # the fixture left-anchors them at 120 mm / 160 mm. Node 1 reads both, so we pin the
            # renderer's own columns: constant right edge for every row.
            assert max(w.x1 for w in amount_w) == pytest.approx(GEN_V1_RIGHT[0], abs=EDGE_TOL), row.row_id
            assert max(w.x1 for w in balance_w) == pytest.approx(GEN_V1_RIGHT[1], abs=EDGE_TOL), row.row_id
        else:
            debit_x1, credit_x1, bal_x1 = RIGHT_EDGES[layout]
            want = credit_x1 if row.amount > 0 else debit_x1
            assert max(w.x1 for w in amount_w) == pytest.approx(want, abs=EDGE_TOL), (row.row_id, entry["amount"])
            assert max(w.x1 for w in balance_w) == pytest.approx(bal_x1, abs=EDGE_TOL), (row.row_id, entry["balance"])

    texts = page_texts_of(res.pdf)
    assert len(texts) == res.n_pages >= 1
    for i, text in enumerate(texts, start=1):
        low = text.lower()
        assert "specimen" in low and "synthetic" in low, f"page {i} lacks the specimen footer"
