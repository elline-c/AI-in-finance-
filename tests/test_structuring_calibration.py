"""FL_STRUCTURING calibration (Requirement 11, task 10.1).

The rule fires only when three or more near-threshold credits fit inside a span of
``_STRUCTURING_CLUSTER_DAYS`` (14) days. Near-threshold means 9,500.00 <= amount < 10,000.00.

Pinned semantics
----------------
* Span = last - first <= 14 days (inclusive): dated credits on day 0, 7 and 14 fire;
  day 0, 7 and 15 are silent.
* A span contains the dated near-threshold credits whose dates lie inside it PLUS every
  undated near-threshold credit (an undated row cannot be placed in time, so it counts as
  inside every span). Two dated credits within 14 days plus one undated fires; two dated
  credits 60 days apart plus one undated is silent (no span holds both dated rows, so the
  best span has 1 dated + 1 undated = 2).
* Three or more undated near-threshold credits fire on their own.
* Sliding window: spans start at each dated near-threshold credit in date order; the
  earliest qualifying span is the one reported (deterministic), and its evidence names the
  span's dates and its rows ("#N(amount)" row list, as in the current evidence format).
* Only credits (positive amounts) count; debits are ignored.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from typing import Optional

import pytest

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes import financial_logic, ingestion
from fraud_detect.synthetic import generate_tampered

from tests.fabrication.fixtures import make_ingestion

D0 = date(2026, 5, 1)


def _txns(rows: list[tuple[Optional[date], float]]) -> list[Transaction]:
    """rows -> Transactions indexed 1..n (balances left unset; irrelevant here)."""
    return [
        Transaction(
            index=i, page=1, date=d,
            date_raw=d.strftime("%d %b %Y") if d else "",
            description="CASH DEPOSIT" if a > 0 else "CASH WITHDRAWAL", amount=a,
        )
        for i, (d, a) in enumerate(rows, start=1)
    ]


def _structuring(rows):
    ing = make_ingestion(_txns(rows), period=None)
    return [f for f in financial_logic.analyze(ing).findings if f.code == "FL_STRUCTURING"]


def _days(*offsets, amount=9_800.00):
    return [(D0 + timedelta(days=o), amount) for o in offsets]


def _date_forms(d: date) -> list[str]:
    return [d.isoformat(), d.strftime("%d %b %Y"), d.strftime("%-d %b %Y")]


def _mentions(text: str, d: date) -> bool:
    return any(f in text for f in _date_forms(d))


# ---- fires ---------------------------------------------------------------- #
def test_four_consecutive_day_deposits_fire_high_weight_20():
    rows = [(D0 + timedelta(days=i), 9_797.00 + i) for i in range(4)]  # 9,797..9,800
    found = _structuring(rows)
    assert len(found) == 1
    assert found[0].severity == Severity.HIGH
    assert found[0].weight == 20


def test_three_on_one_day_fire():
    assert len(_structuring(_days(0, 0, 0))) == 1


def test_three_within_exactly_14_days_fire():
    assert len(_structuring(_days(0, 7, 14))) == 1


# ---- silent (new behaviour) ------------------------------------------------ #
def test_three_spread_over_40_days_silent():
    assert _structuring(_days(0, 20, 40)) == []


def test_statement_f_pattern_silent():
    rows = [
        (date(2026, 7, 28), 9_721.98),
        (date(2026, 8, 1), 9_926.42),
        (date(2026, 8, 28), 9_817.90),
    ]
    assert _structuring(rows) == []


def test_day_0_7_15_silent():
    assert _structuring(_days(0, 7, 15)) == []


# ---- undated rows ----------------------------------------------------------- #
def test_two_dated_within_span_plus_undated_fires():
    rows = _days(0, 5) + [(None, 9_700.00)]
    assert len(_structuring(rows)) == 1


def test_two_dated_60_days_apart_plus_undated_silent():
    rows = _days(0, 60) + [(None, 9_700.00)]
    assert _structuring(rows) == []


def test_three_undated_fire():
    assert len(_structuring([(None, 9_600.00)] * 3)) == 1


# ---- amount boundaries and direction ---------------------------------------- #
def test_amount_exactly_9500_counts():
    rows = [(D0, 9_500.00), (D0 + timedelta(days=1), 9_500.00), (D0 + timedelta(days=2), 9_500.00)]
    assert len(_structuring(rows)) == 1


def test_amount_exactly_10000_does_not_count():
    rows = [(D0, 10_000.00), (D0 + timedelta(days=1), 9_800.00), (D0 + timedelta(days=2), 9_800.00)]
    assert _structuring(rows) == []


def test_amount_just_below_9500_does_not_count():
    rows = [(D0, 9_499.99), (D0 + timedelta(days=1), 9_800.00), (D0 + timedelta(days=2), 9_800.00)]
    assert _structuring(rows) == []


def test_debits_are_ignored():
    rows = [(D0 + timedelta(days=i), -9_800.00) for i in range(4)]
    assert _structuring(rows) == []
    # two credits + a debit of near-threshold size is still only two credits
    assert _structuring(_days(0, 1) + [(D0 + timedelta(days=2), -9_800.00)]) == []


# ---- evidence and determinism ------------------------------------------------ #
def test_evidence_names_span_dates_and_rows():
    rows = _days(0, 3, 6)
    found = _structuring(rows)
    assert len(found) == 1
    ev = found[0].evidence
    assert _mentions(ev, D0) and _mentions(ev, D0 + timedelta(days=6)), ev
    for n in (1, 2, 3):
        assert f"#{n}(" in ev, ev
    assert "9,800" in ev


def test_earliest_qualifying_span_is_reported():
    first = _days(0, 2, 4)
    second = _days(100, 101, 102)
    rows = first + second                       # rows 1-3 early, rows 4-6 late
    found = _structuring(rows)
    assert len(found) == 1
    ev = found[0].evidence
    assert all(f"#{n}(" in ev for n in (1, 2, 3)), ev
    assert not any(f"#{n}(" in ev for n in (4, 5, 6)), ev
    assert found[0].evidence == _structuring(rows)[0].evidence      # deterministic


def test_row_order_in_file_does_not_change_the_reported_span():
    rows = _days(0, 2, 4) + _days(100, 101, 102)
    shuffled = list(reversed(rows[:3])) + rows[3:]
    ev = _structuring(shuffled)[0].evidence
    assert _mentions(ev, D0) and _mentions(ev, D0 + timedelta(days=4)), ev


# ---- generator plant still fires --------------------------------------------- #
def test_generator_planted_structuring_still_fires():
    d = tempfile.mkdtemp(prefix="structuring_cal_")
    gen = generate_tampered(d, seed=42, issues=["STRUCTURING"])
    ing = ingestion.ingest(gen.path)
    codes = {f.code for f in financial_logic.analyze(ing).findings}
    assert "FL_STRUCTURING" in codes
