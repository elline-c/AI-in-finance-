"""Property tests for the subtle amount edit (task 6.3).

Feature: generator-styling-variety, Properties 13 to 15. Ledgers come from
``simulate_ledger`` (through ``make_test_ledger``) with a drawn company seed. Each property
runs at least 100 examples against the real ``tamper/subtle.py`` and a real renderer.
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper.subtle import NAME, plan_subtle_amount_edit
from fraud_detect.ml.retraining.corpus_gates import classify_subtle
from tests.corpus.helpers import load_renderer, make_test_ledger

LAYOUTS_UNDER_TEST = ("generator_v1", "two_col_slash", "two_col_dmon", "signed_amount_iso")

_SETTINGS = settings(max_examples=100, deadline=None, database=None,
                     suppress_health_check=list(HealthCheck))

ledgers = st.builds(
    lambda seed, months: make_test_ledger(months, seed),
    st.integers(min_value=0, max_value=10**6), st.integers(min_value=1, max_value=3))
layouts = st.sampled_from(LAYOUTS_UNDER_TEST)
streams = st.integers(min_value=0, max_value=10**6)


def _plan(ledger, layout, seed):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout} not available")
    return plan_subtle_amount_edit(ledger, renderer, rng(seed, "tamper-test", NAME))


def _entry(result):
    assert len(result.label.affected_rows) == 1
    return result.label.affected_rows[0]


# Feature: generator-styling-variety, Property 13: A subtle amount edit changes exactly one eligible amount
# **Validates: Requirements 7.2, 7.5**
@_SETTINGS
@given(ledger=ledgers, layout=layouts, seed=streams)
def test_prop13_changes_exactly_one_eligible_amount(ledger, layout, seed):
    result = _plan(ledger, layout, seed)
    entry = _entry(result)
    assert entry["field"] == "amount"
    assert result.label.tamper_types == (NAME,)

    eligible = [r for r in ledger.rows if abs(r.amount) >= 100.0]
    if eligible:
        allowed = {r.row_id for r in eligible}
    else:
        top = max(abs(r.amount) for r in ledger.rows)
        allowed = {r.row_id for r in ledger.rows if abs(r.amount) == top}
    assert entry["row_id"] in allowed

    if result.label.method == "rerender":
        assert len(result.ledger.rows) == len(ledger.rows)
        changed = [a.row_id for a, b in zip(ledger.rows, result.ledger.rows)
                   if a.amount != b.amount]
        assert changed == [entry["row_id"]]
        assert not result.file_ops
    else:
        assert result.ledger == ledger          # the overlay leaves the ledger alone
        assert len(result.file_ops) == 1
        assert result.file_ops[0].row_id == entry["row_id"]
        assert result.file_ops[0].field == "amount"


# Feature: generator-styling-variety, Property 14: A subtle amount edit stays between 1% and 5%
# **Validates: Requirements 7.3, 7.4**
@_SETTINGS
@given(ledger=ledgers, layout=layouts, seed=streams)
def test_prop14_change_stays_between_one_and_five_percent(ledger, layout, seed):
    entry = _entry(_plan(ledger, layout, seed))
    old, new = Decimal(entry["original"]), Decimal(entry["new"])
    assert old != new
    assert 0.010 <= float(abs(new - old) / abs(old)) <= 0.050
    assert new == new.quantize(Decimal("0.01"))
    assert "amount_le_5pct" in classify_subtle([entry], 0.05)


# Feature: generator-styling-variety, Property 15: A subtle amount edit is deterministic and replayable
# **Validates: Requirements 7.7, 7.8**
@_SETTINGS
@given(ledger=ledgers, layout=layouts, seed=streams)
def test_prop15_deterministic_and_replayable(ledger, layout, seed):
    first = _plan(ledger, layout, seed)
    second = _plan(ledger, layout, seed)
    assert first == second
    assert first.label.method in {"overlay", "rerender"}

    if first.label.method == "rerender":
        replayed = ledger_ops.apply_edits(ledger, first.edits)
        assert replayed == first.ledger
        ops = [e.op for e in first.edits]
        assert first.label.arithmetic_consistent == ("recompute_balances" in ops)
        assert first.label.arithmetic_consistent == ledger_ops.is_arithmetic_consistent(replayed)
    else:
        assert first.edits == ()
        assert first.label.arithmetic_consistent is False
