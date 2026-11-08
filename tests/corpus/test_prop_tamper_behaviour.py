"""Task 16.3: property test for behavioural tampers.

Feature: synthetic-corpus-generator, Property 15: Behavioural tampers have the right shape

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7**
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

PBT = settings(max_examples=25, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
NAMES = ("structuring", "sudden_large_deposit", "shell_counterparty", "rapid_pass_through",
         "dormant_reactivation", "period_gap", "uen_mismatch")
QUIET = {"service_charge", "interest"}
SHELL_WORDS = ("UNKNOWN", "LLC", "HOLDINGS")

cases = st.tuples(st.sampled_from(NAMES), st.integers(0, 10_000), st.integers(6, 10))


def _inserted(ledger, result):
    old = {r.row_id for r in ledger.rows}
    return [r for r in result.ledger.rows if r.row_id not in old]


@PBT
@given(cases)
def test_behavioural_tampers_have_the_right_shape(case):
    name, seed, months = case
    renderer = load_renderer("two_col_slash")
    if renderer is None:
        pytest.skip("renderer not implemented")
    ledger = make_test_ledger(months, seed)
    result = CATALOG[name].plan(ledger, renderer, rng(seed, "prop-behaviour", name))
    new = result.ledger
    assert result.label.is_tampered and result.label.tamper_types == (name,)

    if name == "structuring":
        rows = _inserted(ledger, result)
        assert 3 <= len(rows) <= 5
        assert all(9500.00 <= r.amount <= 9999.99 for r in rows)
        days = (max(r.date for r in rows) - min(r.date for r in rows)).days
        assert days <= 14 or days > 30
    elif name == "sudden_large_deposit":
        (row,) = _inserted(ledger, result)
        ratio = row.amount / (sum(r.amount for r in ledger.rows if r.amount > 0) / len(ledger.months))
        assert ratio > 3 or 1.5 * 0.98 <= ratio <= 2.5 * 1.02
    elif name == "shell_counterparty":
        (row,) = _inserted(ledger, result)
        assert row.amount >= 50000.00
        assert any(w in row.description.upper() for w in SHELL_WORDS)
    elif name == "rapid_pass_through":
        rows = _inserted(ledger, result)
        credits = [r for r in rows if r.amount > 0]
        debits = [r for r in rows if r.amount < 0]
        assert 2 <= len(credits) <= 4 and len(debits) == len(credits)
        unused = list(debits)
        for c in credits:
            match = next((d for d in unused if 1 <= (d.date - c.date).days <= 3
                          and 0.95 - 1e-9 <= -d.amount / c.amount <= 1.0 + 1e-9), None)
            assert match is not None
            unused.remove(match)
    elif name == "dormant_reactivation":
        k = 0
        while k < months and all(r.kind in QUIET for r in new.rows if r.month_index == k):
            k += 1
        assert 3 <= k <= 5
    elif name == "period_gap":
        (missing,) = [m for m in ledger.months if m not in new.months]
        assert missing not in (ledger.months[0], ledger.months[-1])
        assert len(new.months) == len(ledger.months) - 1
        assert result.label.affected_months == (f"{missing.year:04d}-{missing.month:02d}",)
    else:
        assert new.printed_holder and new.printed_holder != ledger.company.name
        assert new.company.name == ledger.company.name
