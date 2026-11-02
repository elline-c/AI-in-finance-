"""Content-fraud tampers (task 15.1).

These tests are written first and fail with ImportError until task 15.2 adds
``fraud_detect/corpus/tamper/content.py`` (which registers the three tampers in
``CATALOG``).

Conventions (same as ``test_tamper_document.py``):
  * ``CATALOG[name].plan(ledger, renderer, rng) -> TamperResult``
  * ``ledger_ops.apply_edits(ledger, edits)`` replays the recorded edit list.

Recording choices these tests pin down (the design leaves the key name open):
  * ``label.method == "content"``;
  * inserted rows are ``insert_row`` edits and are the only rows with ``fraud=True``;
  * ``label.affected_rows`` has one entry per inserted row (``row_id`` = the inserted
    row); an entry for a copy also carries ``"copies_row_id"`` = the genuine receipt
    it copies (Requirement 7.6). Inserted rows that copy nothing carry no such key.
"""
from __future__ import annotations

import pytest

from fraud_detect.corpus.ledger import replay
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import content  # noqa: F401  (registers the tampers)
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import DEFAULT_STYLE, load_renderer, make_test_ledger

SEEDS = range(8)
NAMES = ("fabricated_receipts", "cross_month_copy", "turnover_inflation")
FIXED_KINDS = {"payroll", "cpf", "rent"}


def _renderer():
    r = load_renderer("two_col_slash")
    if r is None:
        pytest.skip("renderer not implemented yet")
    return r


def _plan(name: str, seed: int):
    ledger = make_test_ledger(6, 2000 + seed)
    result = CATALOG[name].plan(ledger, _renderer(), rng(seed, "tamper-test", name))
    return ledger, result


def _inserted(genuine, result):
    ids = {r.row_id for r in genuine.rows}
    return [r for r in result.ledger.rows if r.row_id not in ids]


def _by_id(ledger):
    return {r.row_id: r for r in ledger.rows}


def _ym(d):
    return (d.year, d.month)


def _consistent(ledger) -> bool:
    if ledger.printed_balances is None:
        return True
    return all(abs(a - b) < 0.005 for a, b in zip(ledger.printed_balances, replay(ledger)))


def _copy_pairs(genuine, result):
    """[(inserted row, genuine source row)] from the label."""
    gen = _by_id(genuine)
    new = _by_id(result.ledger)
    pairs = []
    for entry in result.label.affected_rows:
        assert "copies_row_id" in entry, entry
        pairs.append((new[entry["row_id"]], gen[entry["copies_row_id"]]))
    return pairs


# -- registration and common label ----------------------------------------
def test_catalog_registers_the_three_content_tampers():
    for name in NAMES:
        assert name in CATALOG and CATALOG[name].name == name


@pytest.mark.parametrize("name", NAMES)
def test_label_basics_and_fraud_rows_are_exactly_the_inserted_rows(name):
    for s in SEEDS:
        genuine, result = _plan(name, s)
        ins = _inserted(genuine, result)
        assert ins, "no rows inserted"
        label = result.label
        assert label.is_tampered and label.tamper_types == (name,)
        assert label.method == "content"
        assert label.arithmetic_consistent is True
        # only inserted rows are Fraud_Rows
        assert {r.row_id for r in result.ledger.rows if r.fraud} == {r.row_id for r in ins}
        assert not any(r.fraud for r in genuine.rows)
        assert {e["row_id"] for e in label.affected_rows} == {r.row_id for r in ins}
        # every inserted row is a credit receipt
        assert all(r.amount > 0 and r.kind == "receipt" for r in ins)
        # genuine rows are untouched and keep their relative order
        kept = [r for r in result.ledger.rows if not r.fraud]
        assert kept == list(genuine.rows)
        assert [e.op for e in result.edits].count("insert_row") == len(ins)


@pytest.mark.parametrize("name", NAMES)
def test_balances_are_recomputed_and_edit_log_replays(name):
    for s in SEEDS:
        genuine, result = _plan(name, s)
        assert _consistent(result.ledger)
        assert ledger_ops.apply_edits(genuine, result.edits) == result.ledger
        # the genuine opening balance is unchanged; later rows moved by the inserted credits
        assert result.ledger.opening_balance == genuine.opening_balance
        assert replay(result.ledger)[-1] == pytest.approx(
            replay(genuine)[-1] + sum(r.amount for r in _inserted(genuine, result)), abs=0.01)


@pytest.mark.parametrize("name", NAMES)
def test_summary_boxes_and_following_openings_follow_the_new_rows(name):
    renderer = _renderer()
    genuine, result = _plan(name, 1)
    rendered = renderer.render(result.ledger, DEFAULT_STYLE, 7)
    bal = result.ledger.opening_balance
    summaries = rendered.section_summaries
    assert len(summaries) == len(result.ledger.months)
    for m, s in enumerate(summaries):
        rows = [r for r in result.ledger.rows if r.month_index == m]
        assert s.opening == pytest.approx(bal, abs=0.005)
        assert s.total_deposits == pytest.approx(sum(r.amount for r in rows if r.amount > 0), abs=0.005)
        assert s.total_withdrawals == pytest.approx(-sum(r.amount for r in rows if r.amount < 0), abs=0.005)
        bal = round(bal + sum(r.amount for r in rows), 2)
        assert s.closing == pytest.approx(bal, abs=0.005)


# -- fabricated_receipts ---------------------------------------------------
def test_fabricated_receipts_copy_customer_and_reference_in_the_same_month():
    for s in SEEDS:
        genuine, result = _plan("fabricated_receipts", s)
        ins = _inserted(genuine, result)
        assert 6 <= len(ins) <= 12
        pairs = _copy_pairs(genuine, result)
        assert len(pairs) == len(ins)
        for new, src in pairs:
            assert src.kind == "receipt" and not src.fraud
            assert new.customer == src.customer and new.invoice_ref == src.invoice_ref
            assert src.invoice_ref and src.invoice_ref in new.description
            assert new.date != src.date
            assert new.amount != src.amount
            assert _ym(new.date) == _ym(src.date)
            assert new.month_index == src.month_index
            assert _ym(new.date) == _ym(result.ledger.months[new.month_index])


def test_fabricated_receipts_dates_keep_the_ledger_in_date_order():
    for s in SEEDS:
        _, result = _plan("fabricated_receipts", s)
        dates = [r.date for r in result.ledger.rows]
        assert dates == sorted(dates)


# -- cross_month_copy ------------------------------------------------------
def test_cross_month_copy_lands_in_a_different_calendar_month():
    for s in SEEDS:
        genuine, result = _plan("cross_month_copy", s)
        ins = _inserted(genuine, result)
        assert len(ins) >= 1
        for new, src in _copy_pairs(genuine, result):
            assert src.kind == "receipt" and not src.fraud
            assert new.customer == src.customer and new.invoice_ref == src.invoice_ref
            assert src.invoice_ref in new.description
            assert new.amount != src.amount
            assert _ym(new.date) != _ym(src.date)
            assert new.month_index != src.month_index
            assert _ym(new.date) == _ym(result.ledger.months[new.month_index])


# -- turnover_inflation ----------------------------------------------------
def test_turnover_inflation_adds_new_references_in_at_least_two_months():
    for s in SEEDS:
        genuine, result = _plan("turnover_inflation", s)
        ins = _inserted(genuine, result)
        assert len({r.month_index for r in ins}) >= 2
        old_refs = {r.invoice_ref for r in genuine.rows if r.invoice_ref}
        new_refs = [r.invoice_ref for r in ins]
        assert all(new_refs) and not (set(new_refs) & old_refs)
        assert len(set(new_refs)) == len(new_refs)
        assert all(e["row_id"] in {r.row_id for r in ins} for e in result.label.affected_rows)
        for e in result.label.affected_rows:
            assert "copies_row_id" not in e
        assert sorted({e["month"] for e in result.label.affected_rows}) == list(result.label.affected_months)


def test_turnover_inflation_leaves_payroll_cpf_and_rent_rows_equal():
    for s in SEEDS:
        genuine, result = _plan("turnover_inflation", s)
        before = [r for r in genuine.rows if r.kind in FIXED_KINDS]
        after = [r for r in result.ledger.rows if r.kind in FIXED_KINDS]
        assert before and after == before


# -- Node 3 sanity (not a target) -----------------------------------------
def test_node3_finds_same_month_copies_of_fabricated_receipts():
    from fraud_detect.models import Transaction
    from fraud_detect.nodes.fabrication import FabricationConfig, check_duplicate_invoice_refs

    genuine, result = _plan("fabricated_receipts", 2)
    txns = [Transaction(index=i, date=r.date, amount=r.amount, description=r.description)
            for i, r in enumerate(result.ledger.rows, start=1)]
    codes = {f.code for f in check_duplicate_invoice_refs(txns, FabricationConfig())}
    assert "FL_DUP_INVOICE_REF" in codes
