"""Property tests for the money reader."""
from __future__ import annotations

import time

from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.ingest_parse.money import parse_money

PREFIXES = ["S$", "SGD", "SGD ", "S$ ", "$", ""]
SIGNS = ["", "-", "+", "paren"]
SUFFIXES = [None, "CR", "DR"]


def _fmt_number(cents: int, thousands: bool) -> str:
    whole, frac = divmod(cents, 100)
    body = f"{whole:,}" if thousands else str(whole)
    return f"{body}.{frac:02d}"


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 2: Money round trip
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
@settings(max_examples=200, deadline=None)
@given(
    cents=st.integers(min_value=0, max_value=500_000_000),
    prefix=st.sampled_from(PREFIXES),
    thousands=st.booleans(),
    sign=st.sampled_from(SIGNS),
    sign_before_prefix=st.booleans(),
    suffix=st.sampled_from(SUFFIXES),
    lower=st.booleans(),
)
def test_money_round_trip(cents, prefix, thousands, sign, sign_before_prefix, suffix, lower):
    num = _fmt_number(cents, thousands)
    if sign == "paren":
        text = f"({prefix}{num})"
    elif sign_before_prefix:
        text = f"{sign}{prefix}{num}"
    else:
        text = f"{prefix}{sign}{num}"
    if suffix:
        text += f" {suffix}"
    if lower:
        text = text.lower()

    magnitude = cents / 100
    explicit = {"": None, "-": -1, "+": 1, "paren": -1}[sign]
    expected = -magnitude if explicit == -1 else magnitude
    if suffix == "DR":
        expected = -abs(expected)

    got = parse_money(text)
    assert got is not None, text
    assert got.value == round(expected, 2)
    assert got.explicit_sign == explicit
    assert got.suffix == suffix


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 2: Money round trip
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
@settings(max_examples=200, deadline=None)
@given(st.text(max_size=200))
def test_arbitrary_text_never_raises(text):
    result = parse_money(text)
    assert result is None or isinstance(result.value, float)


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 2: Money round trip
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
@settings(max_examples=200, deadline=None)
@given(
    chunk=st.sampled_from(["1,", "((", "  ", "9", "S$", "-", "1234567890"]),
    reps=st.integers(min_value=100, max_value=20_000),
)
def test_very_long_input_returns_none_quickly(chunk, reps):
    start = time.perf_counter()
    assert parse_money(chunk * reps) is None
    assert time.perf_counter() - start < 0.5
