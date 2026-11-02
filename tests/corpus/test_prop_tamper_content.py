"""Task 15.3: property test for content fraud.

Feature: synthetic-corpus-generator, Property 14: Fabrication tampers copy real receipts the way they should
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import content  # noqa: F401  (registers the tampers)
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

PBT = settings(max_examples=25, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
NAMES = ("fabricated_receipts", "cross_month_copy", "turnover_inflation")
FIXED_KINDS = {"payroll", "cpf", "rent"}

cases = st.tuples(st.sampled_from(NAMES), st.sampled_from(sorted(LAYOUTS)),
                  st.integers(0, 10_000), st.integers(3, 8))


def _ym(d):
    return (d.year, d.month)


@PBT
@given(cases)
def test_fabrication_tampers_copy_real_receipts_the_way_they_should(case):
    """**Validates: Requirements 7.1, 7.2, 7.3, 7.5, 7.6**"""
    name, layout, seed, months = case
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout} not implemented")
    genuine = make_test_ledger(months, seed)
    result = CATALOG[name].plan(genuine, renderer, rng(seed, "prop-tamper", name))
    ledger = result.ledger

    gen_ids = {r.row_id for r in genuine.rows}
    inserted = [r for r in ledger.rows if r.row_id not in gen_ids]
    assert inserted
    # the inserted rows, and only they, are Fraud_Rows
    assert {r.row_id for r in ledger.rows if r.fraud} == {r.row_id for r in inserted}
    assert not any(r.fraud for r in genuine.rows)
    assert all(r.amount > 0 for r in inserted)

    gen = {r.row_id: r for r in genuine.rows}
    new = {r.row_id: r for r in ledger.rows}
    entries = result.label.affected_rows
    assert {e["row_id"] for e in entries} == {r.row_id for r in inserted}

    if name == "turnover_inflation":
        old_refs = {r.invoice_ref for r in genuine.rows if r.invoice_ref}
        assert len({r.month_index for r in inserted}) >= 2
        assert all(r.invoice_ref and r.invoice_ref not in old_refs for r in inserted)
        assert [r for r in ledger.rows if r.kind in FIXED_KINDS] == \
               [r for r in genuine.rows if r.kind in FIXED_KINDS]
        return

    if name == "fabricated_receipts":
        assert 6 <= len(inserted) <= 12
    for e in entries:
        src, cp = gen[e["copies_row_id"]], new[e["row_id"]]  # each copy names its genuine row
        assert src.kind == "receipt" and not src.fraud
        assert cp.customer == src.customer and cp.invoice_ref == src.invoice_ref
        assert cp.date != src.date and cp.amount != src.amount
        if name == "fabricated_receipts":
            assert _ym(cp.date) == _ym(src.date)
        else:
            assert _ym(cp.date) != _ym(src.date)
