"""Ledger-level document tampers (task 13.1).

These tests are written first and fail with ImportError until task 13.2 adds
``fraud_detect/corpus/tamper/ledger_ops.py`` and ``catalog.py``.

Interfaces used (from the design):
  * ``CATALOG[name].plan(ledger, renderer, rng) -> TamperResult``
  * ``ledger_ops.apply_edits(ledger, edits) -> Ledger`` replays a recorded edit list
    (the "edit-log replay" of task 13.2) on the genuine ledger.
Rerender results are picked by scanning seeds for ``label.method == "rerender"``;
the overlay variant keeps the genuine ledger and is covered by task 14.
"""
from __future__ import annotations

from datetime import date

import pytest

from fraud_detect.corpus.ledger import replay
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

SEEDS = range(12)
EDIT_OPS = {"set_amount": "amount", "set_date": "date", "set_balance": "balance"}


def _renderer():
    r = load_renderer("two_col_slash")
    if r is None:
        pytest.skip("renderer not implemented yet")
    return r


def _plan(name: str, seed: int):
    ledger = make_test_ledger(6, 1000 + seed)
    result = CATALOG[name].plan(ledger, _renderer(), rng(seed, "tamper-test", name))
    return ledger, result


def _rerenders(name: str):
    """Yield (genuine ledger, result) for every seed whose method is rerender."""
    found = []
    for s in SEEDS:
        ledger, result = _plan(name, s)
        if result.label.method == "rerender":
            found.append((ledger, result))
    assert found, f"{name}: no rerender variant produced in {len(list(SEEDS))} seeds"
    return found


def _primary_edits(result):
    return [e for e in result.edits if e.op in EDIT_OPS]


def _consistent(ledger) -> bool:
    if ledger.printed_balances is None:
        return True
    return all(abs(a - b) < 0.005 for a, b in zip(ledger.printed_balances, replay(ledger)))


def _changed_rows(old, new, attr):
    assert [r.row_id for r in old.rows] == [r.row_id for r in new.rows]
    return [o.row_id for o, n in zip(old.rows, new.rows) if getattr(o, attr) != getattr(n, attr)]


def _is_edited(row_ids, result):
    return {r["row_id"] for r in result.label.affected_rows} == set(row_ids)


def test_catalog_registers_the_three_document_tampers():
    for name in ("edited_amount", "altered_date", "broken_balance"):
        assert name in CATALOG
        assert CATALOG[name].name == name
        assert CATALOG[name].plan(make_test_ledger(6, 5), _renderer(), rng(5, "x")).label.is_tampered


# -- edited_amount ---------------------------------------------------------
def test_edited_amount_rerender_changes_exactly_one_amount_and_records_it():
    for ledger, result in _rerenders("edited_amount"):
        changed = _changed_rows(ledger, result.ledger, "amount")
        assert len(changed) == 1
        (edit,) = _primary_edits(result)
        assert edit.op == "set_amount" and edit.row_id == changed[0]
        old = next(r for r in ledger.rows if r.row_id == changed[0])
        new = next(r for r in result.ledger.rows if r.row_id == changed[0])
        assert float(edit.original) == pytest.approx(old.amount)
        assert float(edit.new) == pytest.approx(new.amount)
        # nothing else about the rows changed
        for attr in ("date", "description", "kind"):
            assert _changed_rows(ledger, result.ledger, attr) == []
        assert result.label.tamper_types == ("edited_amount",)


def test_edited_amount_records_the_recompute_choice_as_arithmetic_consistent():
    flags = set()
    for ledger, result in _rerenders("edited_amount"):
        assert result.label.arithmetic_consistent == _consistent(result.ledger)
        flags.add(result.label.arithmetic_consistent)
    # the choice is random; both outcomes must be possible across seeds
    assert flags == {True, False}


def test_edited_amount_fraud_rows_are_exactly_the_edited_rows():
    for ledger, result in _rerenders("edited_amount"):
        edited = [e.row_id for e in _primary_edits(result)]
        assert _is_edited(edited, result)
        assert {r.row_id for r in result.ledger.rows if r.fraud} == set(edited)
        assert not any(r.fraud for r in ledger.rows)


# -- altered_date ----------------------------------------------------------
def _breaks(ledger_before, ledger_after, row_id) -> bool:
    rows = ledger_after.rows
    i = next(k for k, r in enumerate(rows) if r.row_id == row_id)
    d = rows[i].date
    first = ledger_after.months[rows[i].month_index]
    outside = (d.year, d.month) != (first.year, first.month)
    out_of_order = any(rows[k].date > rows[k + 1].date for k in range(len(rows) - 1))
    return outside or out_of_order


def test_altered_date_rerender_changes_exactly_one_date_and_records_it():
    for ledger, result in _rerenders("altered_date"):
        changed = _changed_rows(ledger, result.ledger, "date")
        assert len(changed) == 1
        (edit,) = _primary_edits(result)
        assert edit.op == "set_date" and edit.row_id == changed[0]
        old = next(r for r in ledger.rows if r.row_id == changed[0]).date
        new = next(r for r in result.ledger.rows if r.row_id == changed[0]).date
        assert isinstance(new, date) and new != old
        assert old.isoformat() in edit.original and new.isoformat() in edit.new
        assert _changed_rows(ledger, result.ledger, "amount") == []
        assert result.label.tamper_types == ("altered_date",)
        assert _is_edited([changed[0]], result)


def test_altered_date_flag_is_set_only_when_order_or_month_is_broken():
    seen = set()
    for ledger, result in _rerenders("altered_date"):
        row_id = _primary_edits(result)[0].row_id
        assert result.label.date_inconsistent == _breaks(ledger, result.ledger, row_id)
        seen.add(result.label.date_inconsistent)
        assert result.label.arithmetic_consistent == _consistent(result.ledger)
    assert seen == {True, False}


# -- broken_balance --------------------------------------------------------
def test_broken_balance_changes_exactly_one_printed_balance_and_nothing_else():
    ledger, result = _plan("broken_balance", 3)
    new = result.ledger
    assert [r.amount for r in new.rows] == [r.amount for r in ledger.rows]
    assert [r.date for r in new.rows] == [r.date for r in ledger.rows]
    assert new.printed_balances is not None
    genuine = replay(ledger)
    diff = [i for i, (p, g) in enumerate(zip(new.printed_balances, genuine)) if abs(p - g) > 0.005]
    assert len(diff) == 1
    (edit,) = _primary_edits(result)
    assert edit.op == "set_balance" and edit.row_id == new.rows[diff[0]].row_id
    assert float(edit.original) == pytest.approx(genuine[diff[0]])
    assert float(edit.new) == pytest.approx(new.printed_balances[diff[0]])
    assert result.label.arithmetic_consistent is False
    assert result.label.tamper_types == ("broken_balance",)
    assert _is_edited([edit.row_id], result)


# -- edit log and the arithmetic flag --------------------------------------
@pytest.mark.parametrize("name", ["edited_amount", "altered_date", "broken_balance"])
def test_applying_recorded_edits_to_genuine_ledger_reproduces_edited_ledger(name):
    for s in range(6):
        ledger, result = _plan(name, s)
        if result.label.method == "overlay":
            continue
        assert ledger_ops.apply_edits(ledger, result.edits) == result.ledger


@pytest.mark.parametrize("name", ["edited_amount", "altered_date", "broken_balance"])
def test_arithmetic_consistent_iff_printed_balances_equal_replay(name):
    for s in SEEDS:
        _, result = _plan(name, s)
        if result.label.method == "overlay":
            continue
        assert result.label.arithmetic_consistent == _consistent(result.ledger)


def test_genuine_ledger_is_arithmetically_consistent():
    assert _consistent(make_test_ledger(6, 1))
