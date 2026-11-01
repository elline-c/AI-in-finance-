"""Property tests for the date reader."""
from __future__ import annotations

import time
from datetime import date, timedelta

from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.ingest_parse.dates import (
    parse_day_month,
    parse_full_date,
    parse_period,
)

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

dates_any = st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31))


def _fmt(d: date, style: str) -> str:
    if style == "slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    if style == "iso":
        return f"{d.year}-{d.month:02d}-{d.day:02d}"
    if style == "dmony2":
        return f"{d.day:02d} {MON[d.month - 1]} {d.year}"
    if style == "dmony1":
        return f"{d.day} {MON[d.month - 1]} {d.year}"
    if style == "dash2":
        return f"{d.day:02d}-{d.month:02d}-{d.year % 100:02d}"
    if style == "dash4":
        return f"{d.day:02d}-{d.month:02d}-{d.year}"
    raise AssertionError(style)


STYLES = ["slash", "iso", "dmony2", "dmony1", "dash2", "dash4"]


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 1: Date round trip
# **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
@settings(max_examples=200, deadline=None)
@given(d=dates_any, style=st.sampled_from(STYLES))
def test_full_date_round_trip(d, style):
    parsed = parse_full_date(_fmt(d, style))
    assert parsed.value == d
    assert parsed.year_inferred is False


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 1: Date round trip
# **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
@settings(max_examples=200, deadline=None)
@given(
    start=dates_any,
    length=st.integers(min_value=0, max_value=400),
    style=st.sampled_from(STYLES),
    colon=st.booleans(),
    sep=st.sampled_from(["-", "to", "\u2013"]),
)
def test_period_round_trip(start, length, style, colon, sep):
    end = start + timedelta(days=length)
    if end.year > 2099:
        return
    line = f"Statement Period{':' if colon else ''} {_fmt(start, style)} {sep} {_fmt(end, style)}"
    result = parse_period(line)
    assert result is not None
    assert result[0].value == start
    assert result[1].value == end


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 1: Date round trip
# **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
@settings(max_examples=200, deadline=None)
@given(
    start=dates_any,
    # Under a year, so each day/month matches only one date in the period.
    length=st.integers(min_value=0, max_value=364),
    offset=st.integers(min_value=0, max_value=364),
    padded=st.booleans(),
)
def test_day_month_takes_year_from_period(start, length, offset, padded):
    end = start + timedelta(days=length)
    if end.year > 2099:
        return
    target = start + timedelta(days=min(offset, length))
    day = f"{target.day:02d}" if padded else str(target.day)
    parsed = parse_day_month(f"{day} {MON[target.month - 1]}", start, end)
    assert parsed.value == target
    assert parsed.year_inferred is True


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 1: Date round trip
# **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=300), start=dates_any)
def test_arbitrary_text_never_raises(text, start):
    parse_full_date(text)
    parse_period(text)
    parse_day_month(text, start, start + timedelta(days=30))


# Feature: multi-layout-ingestion-and-fabrication-rules, Property 1: Date round trip
# **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
@settings(max_examples=100, deadline=None)
@given(
    chunk=st.sampled_from(["1", "a", " ", "-", "/", "Statement Period ", "1 Jan ", "to "]),
    n=st.integers(min_value=1000, max_value=5000),
)
def test_long_strings_finish_fast(chunk, n):
    text = chunk * n
    t0 = time.perf_counter()
    parse_full_date(text)
    parse_period(text)
    parse_day_month(text, date(2026, 1, 1), date(2026, 12, 31))
    assert time.perf_counter() - t0 < 2.0
