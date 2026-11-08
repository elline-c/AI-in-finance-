"""Task 13.3: property tests for single-edit tampers and the arithmetic flag.

Feature: synthetic-corpus-generator, Property 10: The arithmetic flag is honest
Feature: synthetic-corpus-generator, Property 11: Single-edit tampers change exactly one thing

Only properties that hold for either method are tested: ledger-level checks apply
to results whose ``label.method`` is ``"rerender"`` (an overlay keeps the genuine
ledger); label-level checks apply to every result.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.ledger import replay
from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

PBT = settings(max_examples=60, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
NAMES = ("edited_amount", "altered_date", "broken_balance")
EDIT_OPS = ("set_amount", "set_date", "set_balance")

cases = st.tuples(st.sampled_from(NAMES), st.sampled_from(sorted(LAYOUTS)),
                  st.integers(0, 10_000), st.integers(2, 8))


def _run(name, layout, seed, months):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout} not implemented")
    ledger = make_test_ledger(months, seed)
    result = CATALOG[name].plan(ledger, renderer, rng(seed, "prop-tamper", name))
    return ledger, result


def _consistent(ledger) -> bool:
    if ledger.printed_balances is None:
        return True
    return all(abs(a - b) < 0.005 for a, b in zip(ledger.printed_balances, replay(ledger)))


def _diff(old, new, attr):
    assert [r.row_id for r in old.rows] == [r.row_id for r in new.rows]
    return [o.row_id for o, n in zip(old.rows, new.rows) if getattr(o, attr) != getattr(n, attr)]


def _breaks(ledger, row_id) -> bool:
    rows = ledger.rows
    i = next(k for k, r in enumerate(rows) if r.row_id == row_id)
    first = ledger.months[rows[i].month_index]
    outside = (rows[i].date.year, rows[i].date.month) != (first.year, first.month)
    unordered = any(rows[k].date > rows[k + 1].date for k in range(len(rows) - 1))
    return outside or unordered


# -- Property 10 -----------------------------------------------------------
@PBT
@given(cases)
def test_arithmetic_flag_is_honest(case):
    """**Validates: Requirements 5.3, 5.8, 7.4, 8.8**"""
    name, layout, seed, months = case
    ledger, result = _run(name, layout, seed, months)
    assert _consistent(ledger)
    if result.label.method != "rerender":
        return
    assert result.label.arithmetic_consistent == _consistent(result.ledger)
    if name == "broken_balance":
        assert result.label.arithmetic_consistent is False
        bad = [i for i, (p, g) in enumerate(zip(result.ledger.printed_balances, replay(result.ledger)))
               if abs(p - g) > 0.005]
        # exactly one printed balance differs from the genuine replay
        genuine = replay(ledger)
        diff = [i for i, (p, g) in enumerate(zip(result.ledger.printed_balances, genuine))
                if abs(p - g) > 0.005]
        assert len(diff) == 1
        assert bad  # at least that balance is inconsistent with the replay
    if name == "altered_date":
        assert result.label.arithmetic_consistent is True


# -- Property 11 -----------------------------------------------------------
@PBT
@given(cases)
def test_single_edit_tampers_change_exactly_one_thing(case):
    """**Validates: Requirements 4.2, 5.1, 5.2, 5.3**"""
    name, layout, seed, months = case
    ledger, result = _run(name, layout, seed, months)
    label = result.label
    assert label.is_tampered and label.tamper_types == (name,)
    assert len(label.affected_rows) == 1
    if label.method != "rerender":
        return
    primary = [e for e in result.edits if e.op in EDIT_OPS]
    assert len(primary) == 1
    edit = primary[0]
    assert label.affected_rows[0]["row_id"] == edit.row_id
    assert edit.original != edit.new
    assert label.affected_rows[0]["original"] == edit.original
    assert label.affected_rows[0]["new"] == edit.new
    # replaying the recorded edits reproduces the edited ledger
    assert ledger_ops.apply_edits(ledger, result.edits) == result.ledger
    assert {r.row_id for r in result.ledger.rows if r.fraud} == {edit.row_id}

    if name == "edited_amount":
        assert edit.op == "set_amount"
        assert _diff(ledger, result.ledger, "amount") == [edit.row_id]
        assert _diff(ledger, result.ledger, "date") == []
    elif name == "altered_date":
        assert edit.op == "set_date"
        assert _diff(ledger, result.ledger, "date") == [edit.row_id]
        assert _diff(ledger, result.ledger, "amount") == []
        assert label.date_inconsistent == _breaks(result.ledger, edit.row_id)
    else:
        assert edit.op == "set_balance"
        assert _diff(ledger, result.ledger, "amount") == []
        assert _diff(ledger, result.ledger, "date") == []
