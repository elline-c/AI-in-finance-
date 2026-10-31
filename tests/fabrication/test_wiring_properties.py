"""Task 7.3: property tests for ``analyze_fabrication``.

Feature: node3-fabrication-rules
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from fraud_detect.nodes import fabrication, financial_logic
from fraud_detect.models import Transaction

from .fixtures import make_ingestion

AGENT = "Node3:FinancialLogic"
DUP = "FL_DUP_INVOICE_REF"
SEQ = "FL_INVOICE_SEQUENCE_BREAK"
TURN = "FL_TURNOVER_COST_MISMATCH"
SKIP = "DQ_RULE_SKIPPED"
ALLOWED = {DUP, SEQ, TURN, SKIP}

PBT = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])

_descriptions = st.one_of(
    st.none(),
    st.just(""),
    st.text(max_size=40),
    st.builds(
        lambda c, n: f"FAST FROM {c} INV{n}",
        st.sampled_from(["ALPHA", "BETA CO", "GAMMA LTD"]),
        st.integers(min_value=0, max_value=99999).map(lambda n: f"{n:04d}"),
    ),
    st.sampled_from(["PAYROLL", "RENT", "LOAN", "CPF CONTRIBUTION", "PAYROLL JAN 2026"]),
)
_amounts = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(min_value=-1e7, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=1e7, allow_nan=False, allow_infinity=False),
)
_dates = st.one_of(
    st.none(),
    st.integers(min_value=0, max_value=(date(2027, 12, 31) - date(2025, 1, 1)).days)
    .map(lambda n: date(2025, 1, 1) + timedelta(days=n)),
)
_transactions = st.builds(
    Transaction,
    index=st.integers(min_value=1, max_value=60),  # duplicates allowed
    date=_dates,
    description=_descriptions,
    amount=_amounts,
    page=st.one_of(st.none(), st.integers(min_value=1, max_value=5)),
)
_period = st.one_of(
    st.none(),
    st.just((date(2025, 1, 1), date(2027, 12, 31))),
    st.just((date(2026, 1, 1), date(2026, 6, 30))),
)


@st.composite
def _ingestions(draw):
    txns = draw(st.lists(_transactions, max_size=40))
    if draw(st.booleans()):  # unique indexes
        txns = [t.model_copy(update={"index": i}) for i, t in enumerate(txns, start=1)]
    return make_ingestion(txns, period=draw(_period))


def _dumps(findings):
    return [f.model_dump() for f in findings]


def _boom(*_a, **_k):
    raise RuntimeError("secret detail")


# Feature: node3-fabrication-rules, Property 9: Rules never crash and skips are visible
# **Validates: Requirements 2.5, 9.1, 9.2, 9.3**
@PBT
@given(ing=_ingestions())
def test_property_9_rules_never_crash_and_skips_are_visible(ing):
    baseline = fabrication.analyze_fabrication(ing)
    for f in baseline:
        assert f.agent == AGENT
        assert f.code in ALLOWED
    assert SKIP not in {f.code for f in baseline}

    for attr, code in (
        ("check_invoice_sequence", SEQ),
        ("check_duplicate_invoice_refs", DUP),
        ("check_turnover_vs_costs", TURN),
    ):
        with patch.object(fabrication, attr, _boom):
            patched = fabrication.analyze_fabrication(ing)
        skips = [f for f in patched if f.code == SKIP]
        assert len(skips) == 1
        assert skips[0].field == code
        assert skips[0].agent == AGENT
        rest = [f for f in patched if f.code != SKIP]
        expected = [f for f in baseline if f.code != code]
        if attr == "check_invoice_sequence":
            # Rule A loses the likelier-copy hint; compare codes/fields and Rule C exactly.
            assert [(f.code, f.field) for f in rest] == [(f.code, f.field) for f in expected]
            assert _dumps(f for f in rest if f.code == TURN) == _dumps(f for f in expected if f.code == TURN)
        else:
            assert _dumps(rest) == _dumps(expected)


# Feature: node3-fabrication-rules, Property 10: Determinism and no mutation
# **Validates: Requirements 6.6, 9.5**
@PBT
@given(ing=_ingestions())
def test_property_10_determinism_and_no_mutation(ing):
    before = ing.model_dump()
    first = fabrication.analyze_fabrication(ing)
    second = fabrication.analyze_fabrication(ing)
    assert _dumps(first) == _dumps(second)
    r1 = financial_logic.analyze(ing)
    r2 = financial_logic.analyze(ing)
    assert r1.model_dump() == r2.model_dump()
    assert ing.model_dump() == before
