"""The clean synthetic statement must satisfy Node 3's 6-12 month period rule.

``generate_clean`` used to span only about two months, so the period-length rule
(``FL_PERIOD_LENGTH``) flagged a statement that was meant to be clean. These tests
pin the generator to the rule from both sides: the default and the boundary values
6 and 12 must pass, while values just outside the window must still be flagged, so
the generator cannot drift back out of policy (or the rule go silent) unnoticed.
"""
from __future__ import annotations

import pytest

from fraud_detect.nodes import financial_logic, ingestion
from fraud_detect.nodes.financial_logic import period_months
from fraud_detect.synthetic import generate_clean


def _ingest(tmp_path, **kwargs):
    return ingestion.ingest(generate_clean(str(tmp_path), **kwargs).path)


def _codes(ing) -> set[str]:
    return {f.code for f in financial_logic.analyze(ing).findings}


def _rule_months(ing) -> int:
    """Whole months between the declared period dates, counted as the rule does."""
    return period_months(ing.statement_period_start, ing.statement_period_end)


def test_default_clean_statement_is_inside_the_policy_window(tmp_path):
    ing = _ingest(tmp_path)
    assert 6 <= _rule_months(ing) <= 12
    assert "FL_PERIOD_LENGTH" not in _codes(ing)


def test_default_clean_statement_trips_no_node3_rule_at_all(tmp_path):
    """Clean means clean: lengthening it must not introduce gaps, breaks, etc."""
    assert _codes(_ingest(tmp_path)) == set()


@pytest.mark.parametrize("months", [6, 9, 12])
def test_months_inside_the_window_pass(tmp_path, months):
    ing = _ingest(tmp_path, months=months)
    assert _rule_months(ing) == months
    assert "FL_PERIOD_LENGTH" not in _codes(ing)


@pytest.mark.parametrize("months", [2, 5, 13])
def test_months_outside_the_window_are_still_flagged(tmp_path, months):
    ing = _ingest(tmp_path, months=months)
    assert _rule_months(ing) == months
    assert "FL_PERIOD_LENGTH" in _codes(ing)


def test_clean_statement_has_no_unexplained_gaps(tmp_path):
    """Node 3 flags gaps over 25 days; a long statement must stay evenly spaced."""
    dates = [t.date for t in _ingest(tmp_path).transactions if t.date]
    assert dates, "no transactions were parsed"
    assert max((b - a).days for a, b in zip(dates, dates[1:])) <= 25


def test_generation_is_deterministic(tmp_path):
    def rows(sub):
        ing = _ingest(tmp_path / sub)
        return [(t.date, t.description, t.amount, t.balance) for t in ing.transactions]

    assert rows("a") == rows("b")
