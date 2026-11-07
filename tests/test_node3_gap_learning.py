"""Node 3 FL_PERIOD_GAP: the silence limit is learned from the company's own rhythm."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction
from fraud_detect.nodes import financial_logic as fl
from fraud_detect.nodes.financial_logic import _period_continuity, analyze, gap_limit


def _ing(days_from_start: list[int], start: date = date(2025, 1, 1)) -> IngestionResult:
    txns = [
        Transaction(index=i + 1, date=start + timedelta(days=d), description="X",
                    amount=1.0, balance=None, page=1)
        for i, d in enumerate(days_from_start)
    ]
    return IngestionResult(source_path="t.pdf", page_count=1, raw_text="",
                           metadata=DocumentMetadata(), transactions=txns)


def _gap_findings(ing, trailing=None):
    return [f for f in _period_continuity(ing, trailing) if f.code == "FL_PERIOD_GAP"]


# --------------------------------------------------------------------------- #
# gap_limit: the learning rule itself
# --------------------------------------------------------------------------- #
def test_too_few_gaps_uses_default():
    limit, basis = gap_limit([5, 6])
    assert limit == fl._GAP_DAYS
    assert "default" in basis


def test_busy_account_gets_the_floor():
    limit, _ = gap_limit([1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2])
    assert limit == fl._GAP_MIN_DAYS


def test_sparse_account_gets_a_longer_limit_than_busy():
    busy, _ = gap_limit([1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2])
    sparse, _ = gap_limit([9, 12, 8, 14, 10, 11, 9, 13, 10, 12])
    assert sparse > busy


def test_limit_never_exceeds_the_cap():
    limit, _ = gap_limit([20, 40, 10, 55, 30, 25, 45, 15, 50, 35])
    assert limit == fl._GAP_MAX_DAYS


def test_one_long_silence_cannot_stretch_its_own_limit():
    normal = [1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1]
    with_silence, _ = gap_limit(normal + [30])
    without, _ = gap_limit(normal)
    assert with_silence == without


# --------------------------------------------------------------------------- #
# _period_continuity: end to end on a statement
# --------------------------------------------------------------------------- #
def test_busy_account_flags_a_ten_day_silence():
    days = list(range(0, 30)) + list(range(40, 70))      # daily, then a 11-day gap
    found = _gap_findings(_ing(days))
    assert len(found) == 1
    assert "11-day gap" in found[0].evidence


def test_busy_account_does_not_flag_a_weekend():
    days = [d for d in range(0, 60) if d % 7 not in (5, 6)]   # weekdays only
    assert _gap_findings(_ing(days)) == []


def test_sparse_account_is_not_flagged_for_its_normal_rhythm():
    days = [0, 10, 21, 30, 41, 51, 61, 72, 81, 92, 102, 112]  # about every 10 days
    assert _gap_findings(_ing(days)) == []


def test_removed_month_is_still_flagged_on_a_sparse_account():
    days = [0, 10, 21, 30, 41, 51, 61, 72, 81, 92, 140, 150, 160]  # 48-day silence
    found = _gap_findings(_ing(days))
    assert len(found) == 1
    assert "48-day gap" in found[0].evidence


def test_few_transactions_fall_back_to_the_old_25_day_rule():
    assert len(_gap_findings(_ing([0, 5, 40]))) == 1          # 35-day gap
    assert _gap_findings(_ing([0, 5, 24])) == []               # 19-day gap


def test_several_transactions_on_one_day_are_one_active_day():
    days = [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12]
    assert _gap_findings(_ing(days)) == []


def test_evidence_states_the_limit_and_where_it_came_from():
    days = list(range(0, 30)) + list(range(40, 70))
    ev = _gap_findings(_ing(days))[0].evidence
    assert "normal limit is about 7 days" in ev
    assert "learned" in ev


def test_undated_rows_are_ignored():
    ing = _ing(list(range(0, 15)))
    ing.transactions.append(Transaction(index=99, date=None, description="X", amount=1.0))
    assert _gap_findings(ing) == []


# --------------------------------------------------------------------------- #
# trailing history
# --------------------------------------------------------------------------- #
def test_trailing_history_is_used_when_enough_gaps_are_given():
    # Statement alone looks daily-ish (limit 7); trailing history says this firm
    # normally posts every 10 days, so a 12-day silence is not unusual.
    days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23]
    assert len(_gap_findings(_ing(days))) == 1
    trailing = [9, 12, 8, 14, 10, 11, 9, 13, 10, 12]
    assert _gap_findings(_ing(days), trailing) == []


def test_trailing_history_names_its_source():
    days = list(range(0, 30)) + list(range(70, 80))
    trailing = [1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2]
    ev = _gap_findings(_ing(days), trailing)[0].evidence
    assert "trailing statements" in ev


def test_too_little_trailing_history_falls_back_to_the_statement():
    days = list(range(0, 30)) + list(range(40, 70))
    with_thin_history = _gap_findings(_ing(days), [20, 25])
    assert with_thin_history == _gap_findings(_ing(days))


def test_analyze_passes_trailing_gaps_through():
    days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23]
    trailing = [9, 12, 8, 14, 10, 11, 9, 13, 10, 12]
    with_h = [f.code for f in analyze(_ing(days), trailing_gap_days=trailing).findings]
    without = [f.code for f in analyze(_ing(days)).findings]
    assert "FL_PERIOD_GAP" in without
    assert "FL_PERIOD_GAP" not in with_h


@pytest.mark.parametrize("bad", [[], None])
def test_empty_or_missing_trailing_history_is_harmless(bad):
    days = list(range(0, 30)) + list(range(40, 70))
    assert len(_gap_findings(_ing(days), bad)) == 1
