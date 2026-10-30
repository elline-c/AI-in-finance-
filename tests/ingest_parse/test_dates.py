"""Task 2.1: tests for the Node 1 date reader (Requirements 2.1 to 2.8).

Test-first: ``fraud_detect.nodes.ingest_parse.dates`` is implemented in task 2.2,
so these tests fail with ImportError until then. All values are fictional.
"""
from __future__ import annotations

from datetime import date

import pytest

from fraud_detect.nodes.ingest_parse.dates import (
    ParsedDate,
    parse_day_month,
    parse_full_date,
    parse_period,
)

# --- parse_full_date: every style (Req 2.1, 2.2, 2.7) ----------------------

FULL_DATE_CASES = [
    ("dd/mm/yyyy", "04/05/2026", date(2026, 5, 4)),
    ("dd/mm/yyyy day>12", "27/11/2025", date(2025, 11, 27)),
    ("yyyy-mm-dd", "2026-04-30", date(2026, 4, 30)),
    ("dd Mon yyyy", "04 Apr 2026", date(2026, 4, 4)),
    ("d Mon yyyy", "4 Apr 2026", date(2026, 4, 4)),
    ("dd-mm-yy", "04-05-26", date(2026, 5, 4)),
    ("dd-mm-yyyy", "04-05-2026", date(2026, 5, 4)),
    ("two-digit year -> 20yy", "31-12-99", date(2099, 12, 31)),
    ("two-digit year 00", "01-01-00", date(2000, 1, 1)),
    ("day-first, both <=12", "01/02/2026", date(2026, 2, 1)),
    ("month name case-insensitive", "4 APR 2026", date(2026, 4, 4)),
]


@pytest.mark.parametrize("label,raw,expected", FULL_DATE_CASES, ids=[c[0] for c in FULL_DATE_CASES])
def test_parse_full_date_styles(label, raw, expected):
    result = parse_full_date(raw)
    assert isinstance(result, ParsedDate)
    assert result.value == expected
    assert result.raw == raw
    assert result.year_inferred is False


@pytest.mark.parametrize(
    "raw",
    ["", "not a date", "31/02/2026", "2026-13-01", "32 Apr 2026", "4 Foo 2026", "13/13/2026"],
)
def test_parse_full_date_unreadable_keeps_raw(raw):
    result = parse_full_date(raw)
    assert result.value is None
    assert result.raw == raw


def test_parse_full_date_long_line_is_fast_and_unread():
    raw = "1" * 10_000
    result = parse_full_date(raw)
    assert result.value is None
    assert result.raw == raw


# --- parse_period (Req 2.6) -------------------------------------------------

PERIOD_CASES = [
    ("no colon, 'to'", "Statement Period 01/04/2026 to 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("colon, hyphen", "Statement Period: 01/04/2026 - 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("colon, 'to'", "Statement Period: 01/04/2026 to 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("no colon, hyphen", "Statement Period 01/04/2026 - 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("en dash", "Statement Period 01/04/2026 \u2013 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("en dash, no spaces", "Statement Period: 01/04/2026\u201330/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("d Mon yyyy with 'to'", "Statement Period 1 Apr 2026 to 30 Sep 2026", date(2026, 4, 1), date(2026, 9, 30)),
    ("iso dates", "Statement Period 2026-04-01 to 2026-04-30", date(2026, 4, 1), date(2026, 4, 30)),
    ("dd-mm-yy dates", "Statement Period 01-04-26 to 30-04-26", date(2026, 4, 1), date(2026, 4, 30)),
    ("case-insensitive label", "STATEMENT PERIOD 01/04/2026 TO 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
    ("embedded in longer line", "Page 1 of 1  Statement Period 01/04/2026 to 30/04/2026", date(2026, 4, 1), date(2026, 4, 30)),
]


@pytest.mark.parametrize("label,text,start,end", PERIOD_CASES, ids=[c[0] for c in PERIOD_CASES])
def test_parse_period_variants(label, text, start, end):
    result = parse_period(text)
    assert result is not None
    first, second = result
    assert isinstance(first, ParsedDate) and isinstance(second, ParsedDate)
    assert first.value == start
    assert second.value == end
    assert first.year_inferred is False and second.year_inferred is False


def test_parse_period_keeps_raw_text_of_each_end():
    first, second = parse_period("Statement Period 1 Apr 2026 to 30 Sep 2026")
    assert first.raw == "1 Apr 2026"
    assert second.raw == "30 Sep 2026"


@pytest.mark.parametrize("text", ["", "Opening Balance 100.00", "Statement Period", "no period here 01/04/2026"])
def test_parse_period_returns_none_when_absent(text):
    assert parse_period(text) is None


# --- parse_day_month: "d Mon" with the year from the period (Req 2.1, 2.3-2.5, 2.8)

APR = (date(2026, 4, 1), date(2026, 4, 30))
YEAR_END = (date(2025, 12, 1), date(2026, 1, 31))

DAY_MONTH_CASES = [
    ("d Mon uses period year", "4 Apr", APR, date(2026, 4, 4)),
    ("dd Mon uses period year", "04 Apr", APR, date(2026, 4, 4)),
    ("first day of period", "1 Apr", APR, date(2026, 4, 1)),
    ("last day of period", "30 Apr", APR, date(2026, 4, 30)),
    ("period spans months, start year", "15 Dec", YEAR_END, date(2025, 12, 15)),
    ("year-end crossing: Jan gets end year", "5 Jan", YEAR_END, date(2026, 1, 5)),
    ("year-end crossing: Dec gets start year", "31 Dec", YEAR_END, date(2025, 12, 31)),
    ("six-month period within one year", "15 Aug", (date(2026, 4, 1), date(2026, 9, 30)), date(2026, 8, 15)),
]


@pytest.mark.parametrize("label,raw,period,expected", DAY_MONTH_CASES, ids=[c[0] for c in DAY_MONTH_CASES])
def test_parse_day_month_assigns_year(label, raw, period, expected):
    result = parse_day_month(raw, *period)
    assert result.value == expected
    assert result.raw == raw
    assert result.year_inferred is True


OUT_OF_PERIOD_CASES = [
    ("month after period", "5 Jun", APR),
    ("month before period", "5 Feb", APR),
    ("day past period end", "31 Dec", (date(2025, 12, 1), date(2025, 12, 30))),
    ("wrapped year still outside", "5 Jan", (date(2025, 12, 1), date(2025, 12, 31))),
    ("year-end: month between but after end", "15 Feb", YEAR_END),
]


@pytest.mark.parametrize("label,raw,period", OUT_OF_PERIOD_CASES, ids=[c[0] for c in OUT_OF_PERIOD_CASES])
def test_parse_day_month_out_of_period_gives_no_date_and_keeps_raw(label, raw, period):
    result = parse_day_month(raw, *period)
    assert result.value is None
    assert result.raw == raw


@pytest.mark.parametrize("raw", ["", "Apr", "31 Feb", "32 Apr", "4 Foo", "garbage"])
def test_parse_day_month_unreadable_keeps_raw(raw):
    result = parse_day_month(raw, *APR)
    assert result.value is None
    assert result.raw == raw
