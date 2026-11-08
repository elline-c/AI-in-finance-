"""Property tests for FabricationConfig validation and invoice-reference reading.

Property 1: Invoice reference normalisation        (Requirements 2.1, 2.2, 2.3)
Property 12: Invalid configuration is rejected     (Requirement 1.3)

Decision: a reference needs at least three digits after the prefix
(INV12 gives none; INV0042 == inv-042 == INV #042).
"""
from __future__ import annotations

import re

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    FabricationConfig,
    extract_invoice_ref,
    normalise_counterparty,
)

SETTINGS = settings(max_examples=150, deadline=None)

SEPARATORS = ["", " ", "-", "#", " # ", " - "]

# Letters and spaces only, never ending in a digit (no digits at all), trimmed.
customers = st.text(
    alphabet=st.sampled_from(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")),
    max_size=30,
).map(str.strip)


def _case(prefix: str, flags: list[bool]) -> str:
    return "".join(c.upper() if f else c.lower() for c, f in zip(prefix, flags))


@st.composite
def reference_descriptions(draw):
    n = draw(st.integers(min_value=100, max_value=10**7))
    zeros = draw(st.integers(min_value=0, max_value=6))
    sep = draw(st.sampled_from(SEPARATORS))
    prefix = _case("INV", draw(st.lists(st.booleans(), min_size=3, max_size=3)))
    customer = draw(customers)
    lead = (customer + " ") if customer else ""
    return n, f"{lead}{prefix}{sep}{'0' * zeros}{n}", customer


# Feature: node3-fabrication-rules, Property 1: Invoice reference normalisation
@SETTINGS
@given(reference_descriptions())
def test_property_1_reference_is_normalised(case):
    """**Validates: Requirements 2.1, 2.2, 2.3**"""
    n, desc, _customer = case
    assert extract_invoice_ref(desc, DEFAULT_CONFIG) == ("INV", n)
    cp = normalise_counterparty(desc, DEFAULT_CONFIG)
    assert re.search(r"INV[\s\-#]*\d", cp) is None
    assert str(n) not in cp.split()


# Feature: node3-fabrication-rules, Property 1: Invoice reference normalisation
@SETTINGS
@given(
    st.integers(min_value=100, max_value=10**6),
    st.integers(min_value=0, max_value=5),
    st.lists(st.sampled_from(SEPARATORS), min_size=2, max_size=2),
    customers,
)
def test_property_1_variants_agree(n, zeros, seps, customer):
    """**Validates: Requirements 2.1, 2.2, 2.3**"""
    lead = (customer + " ") if customer else ""
    a = f"{lead}INV{seps[0]}{n}"
    b = f"{lead}inv{seps[1]}{'0' * zeros}{n}"
    assert extract_invoice_ref(a, DEFAULT_CONFIG) == extract_invoice_ref(b, DEFAULT_CONFIG) == ("INV", n)
    assert normalise_counterparty(a, DEFAULT_CONFIG) == normalise_counterparty(b, DEFAULT_CONFIG)


# Feature: node3-fabrication-rules, Property 1: Invoice reference normalisation
@SETTINGS
@given(
    customers,
    st.sampled_from(SEPARATORS),
    st.one_of(
        st.just(""),
        st.integers(min_value=0, max_value=99).map(lambda i: str(i)),  # 1-2 digits
    ),
)
def test_property_1_no_or_too_few_digits_gives_none(customer, sep, digits):
    """**Validates: Requirements 2.2, 2.3**"""
    lead = (customer + " ") if customer else ""
    assert extract_invoice_ref(f"{lead}INV{sep}{digits}", DEFAULT_CONFIG) is None


# --- Property 12 -----------------------------------------------------------

INT_FIELDS = [
    "dup_escalate_min_refs", "instalment_max_count", "seq_min_refs", "seq_min_breaks",
    "turnover_min_months", "turnover_min_jump_months", "evidence_max_groups",
]
WEIGHT_FIELDS = ["dup_isolated_weight", "dup_escalated_weight", "seq_weight", "turnover_weight"]

pos_int = st.integers(min_value=1, max_value=1000)
weight = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)


@st.composite
def valid_kwargs(draw):
    kw = {f: draw(pos_int) for f in INT_FIELDS}
    kw.update({f: draw(weight) for f in WEIGHT_FIELDS})
    kw["instalment_max_days"] = draw(st.integers(min_value=0, max_value=365))
    kw["credit_jump_ratio"] = draw(st.floats(min_value=1.0001, max_value=100, allow_nan=False))
    kw["fixed_cost_stability_pct"] = draw(st.floats(min_value=0, max_value=0.9999, allow_nan=False))
    kw["turnover_min_stable_share"] = draw(st.floats(min_value=0.0001, max_value=1, allow_nan=False))
    return kw


@st.composite
def invalid_field(draw):
    kind = draw(st.sampled_from(["ratio", "stability", "count", "weight", "share"]))
    if kind == "ratio":
        return "credit_jump_ratio", draw(st.floats(max_value=1.0, allow_nan=False, allow_infinity=False))
    if kind == "stability":
        v = draw(st.one_of(
            st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
        ))
        return "fixed_cost_stability_pct", v
    if kind == "count":
        return draw(st.sampled_from(INT_FIELDS)), draw(st.integers(max_value=0, min_value=-1000))
    if kind == "weight":
        return draw(st.sampled_from(WEIGHT_FIELDS)), draw(
            st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False)
        )
    v = draw(st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
    ))
    return "turnover_min_stable_share", v


# Feature: node3-fabrication-rules, Property 12: Invalid configuration is rejected
@SETTINGS
@given(valid_kwargs())
def test_property_12_valid_config_constructs(kw):
    """**Validates: Requirements 1.3**"""
    cfg = FabricationConfig(**kw)
    for k, v in kw.items():
        assert getattr(cfg, k) == v


# Feature: node3-fabrication-rules, Property 12: Invalid configuration is rejected
@SETTINGS
@given(valid_kwargs(), invalid_field())
def test_property_12_single_invalid_field_rejected(kw, bad):
    """**Validates: Requirements 1.3**"""
    name, value = bad
    kw = dict(kw, **{name: value})
    with pytest.raises(ValueError):
        FabricationConfig(**kw)
