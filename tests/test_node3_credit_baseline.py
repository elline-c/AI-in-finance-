"""Node 3: sudden deposits judged against the company's own history, and the
rapid pass-through (velocity) check."""
from __future__ import annotations

from datetime import date

import pytest

from fraud_detect.history import TrailingHistory, monthly_credit_totals
from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction
from fraud_detect.nodes import financial_logic as fl
from fraud_detect.nodes.financial_logic import analyze

UEN = "209999999Z"
# One month of ordinary credits: median 1,275, a normal month totals 16,750.
AMTS = [800, 1200, 1500, 900, 2000, 1100, 1300, 1700, 1000, 1400, 1600, 1250]


def _base(scale: float = 1.0, months=range(1, 10)):
    return [(date(2025, m, 1 + 2 * i), f"FAST FROM CUSTOMER {i}", a * scale)
            for m in months for i, a in enumerate(AMTS)]


def _ing(rows, *, start=date(2025, 1, 1), end=date(2025, 9, 30), uen=UEN) -> IngestionResult:
    rows = sorted(rows, key=lambda r: r[0])           # stable: same-day rows keep their order
    txns = [Transaction(index=i, date=d, description=desc, amount=a, page=1)
            for i, (d, desc, a) in enumerate(rows, start=1)]
    return IngestionResult(source_path="t.pdf", page_count=1, raw_text="",
                           metadata=DocumentMetadata(), uen=uen, account_holder="Test Co Pte Ltd",
                           statement_period_start=start, statement_period_end=end,
                           transactions=txns)


def _find(ing, code, **kw):
    return [f for f in analyze(ing, **kw).findings if f.code == code]


# --------------------------------------------------------------------------- #
# Sudden deposit, judged against the company itself
# --------------------------------------------------------------------------- #
def test_spike_is_flagged_from_the_statement_alone_with_no_declared_baseline():
    ing = _ing(_base() + [(date(2025, 5, 14), "INWARD REMITTANCE", 40_000.0)])
    found = _find(ing, "FL_SUDDEN_DEPOSIT")
    assert len(found) == 1
    f = found[0]
    assert f.severity.value == "MEDIUM" and f.weight == 12      # a prompt, not an alarm
    assert "this statement" in f.evidence
    assert "invoice or contract" in f.evidence                  # says what to ask for


def test_ordinary_credits_are_not_flagged():
    assert _find(_ing(_base()), "FL_SUDDEN_DEPOSIT") == []


def test_the_same_amount_is_normal_for_a_bigger_company_but_a_spike_for_a_small_one():
    spike = (date(2025, 5, 14), "INWARD REMITTANCE", 20_000.0)
    small = _ing(_base(1.0) + [spike])
    big = _ing(_base(20.0) + [spike])            # this company's normal credits are ~20x larger
    assert len(_find(small, "FL_SUDDEN_DEPOSIT")) == 1
    assert _find(big, "FL_SUDDEN_DEPOSIT") == []


def test_a_moderate_spike_of_one_and_a_half_normal_months_is_flagged():
    ing = _ing(_base() + [(date(2025, 6, 9), "INWARD REMITTANCE", 1.5 * 16_750)])
    assert len(_find(ing, "FL_SUDDEN_DEPOSIT")) == 1


def test_a_large_but_not_unusual_share_of_a_month_is_not_flagged():
    # 6,000 is above the normal credit fence but under half a normal month (8,375).
    ing = _ing(_base() + [(date(2025, 6, 9), "INWARD REMITTANCE", 6_000.0)])
    assert _find(ing, "FL_SUDDEN_DEPOSIT") == []


def test_spikes_do_not_distort_their_own_baseline():
    ing = _ing(_base() + [(date(2025, 3, 5), "INWARD REMITTANCE", 90_000.0),
                          (date(2025, 7, 5), "INWARD REMITTANCE", 120_000.0)])
    assert len(_find(ing, "FL_SUDDEN_DEPOSIT")) == 2


def test_too_little_data_and_no_declared_baseline_gives_no_finding():
    short = _ing(_base(months=range(1, 3)) + [(date(2025, 2, 20), "INWARD REMITTANCE", 90_000.0)],
                 end=date(2025, 2, 28))                      # only 2 complete months
    assert _find(short, "FL_SUDDEN_DEPOSIT") == []


def test_declared_baseline_is_the_fallback_for_a_short_statement(monkeypatch):
    monkeypatch.setitem(fl.CASHFLOW_BASELINE, UEN, {"avg_monthly_credit": 10_000.0})
    rows = _base(months=range(1, 3))
    short = lambda amt: _ing(rows + [(date(2025, 2, 20), "INWARD REMITTANCE", amt)],
                             end=date(2025, 2, 28))
    hit = _find(short(35_000.0), "FL_SUDDEN_DEPOSIT")
    assert len(hit) == 1 and "declared" in hit[0].evidence
    assert _find(short(25_000.0), "FL_SUDDEN_DEPOSIT") == []   # not above 3x the declared month


def test_the_statements_own_months_beat_the_declared_baseline(monkeypatch):
    # A wildly wrong declared figure must not decide anything when the statement itself
    # has enough data.
    monkeypatch.setitem(fl.CASHFLOW_BASELINE, UEN, {"avg_monthly_credit": 100_000_000.0})
    ing = _ing(_base() + [(date(2025, 5, 14), "INWARD REMITTANCE", 40_000.0)])
    assert len(_find(ing, "FL_SUDDEN_DEPOSIT")) == 1


def _big_company_history():
    return TrailingHistory(credits=tuple(a * 20.0 for a in AMTS * 6),
                           monthly_credits=(16_750.0 * 20,) * 6, gap_days=(),
                           case_ids=("CASE-OLD",))


def test_approved_history_is_preferred_over_the_statement():
    ing = _ing(_base() + [(date(2025, 5, 14), "INWARD REMITTANCE", 20_000.0)])
    assert len(_find(ing, "FL_SUDDEN_DEPOSIT")) == 1                     # alone: a spike
    assert _find(ing, "FL_SUDDEN_DEPOSIT", history=_big_company_history()) == []   # normal for them


def test_evidence_names_the_cases_the_history_came_from():
    ing = _ing(_base() + [(date(2025, 5, 14), "INWARD REMITTANCE", 900_000.0)])
    f = _find(ing, "FL_SUDDEN_DEPOSIT", history=_big_company_history())[0]
    assert "CASE-OLD" in f.evidence


def test_thin_history_falls_back_to_the_statement():
    thin = TrailingHistory(credits=(1.0, 2.0), monthly_credits=(3.0,), case_ids=("CASE-X",))
    ing = _ing(_base() + [(date(2025, 5, 14), "INWARD REMITTANCE", 40_000.0)])
    f = _find(ing, "FL_SUDDEN_DEPOSIT", history=thin)[0]
    assert "this statement" in f.evidence


def test_monthly_totals_only_count_complete_months():
    rows = _ing(_base(months=range(1, 5)) + [(date(2025, 5, 3), "X", 1000.0)]).transactions
    full = monthly_credit_totals(rows, date(2025, 1, 1), date(2025, 4, 30))
    assert len(full) == 4
    partial_start = monthly_credit_totals(rows, date(2025, 1, 15), date(2025, 4, 30))
    assert len(partial_start) == 3                       # January starts mid-month
    unknown_period = monthly_credit_totals(rows, None, None)
    assert len(unknown_period) == 3                      # first and last month dropped


# --------------------------------------------------------------------------- #
# Rapid pass-through
# --------------------------------------------------------------------------- #
CREDIT = (date(2025, 3, 10), "TRANSFER FROM ALPHA CAPITAL", 20_000.0)


def _pt(*extra, base=None):
    return _ing((base if base is not None else _base()) + list(extra))


def test_money_leaving_again_within_days_is_flagged():
    ing = _pt(CREDIT, (date(2025, 3, 12), "TRANSFER TO DELTA SUPPLY", -19_400.0))
    found = _find(ing, "FL_RAPID_PASS_THROUGH")
    assert len(found) == 1
    f = found[0]
    assert f.severity.value == "MEDIUM" and f.weight == 12
    assert "19,400.00" in f.evidence and "contract or invoice" in f.evidence


def test_a_large_deposit_that_also_leaves_is_escalated():
    ing = _pt(CREDIT, (date(2025, 3, 12), "TRANSFER TO DELTA SUPPLY", -19_400.0))
    dep = _find(ing, "FL_SUDDEN_DEPOSIT")
    assert len(dep) == 1 and dep[0].severity.value == "HIGH" and dep[0].weight == 20
    assert "FL_RAPID_PASS_THROUGH" in dep[0].evidence


def test_money_that_stays_is_not_flagged():
    assert _find(_pt(CREDIT), "FL_RAPID_PASS_THROUGH") == []


@pytest.mark.parametrize("day,out", [
    (16, -19_400.0),        # leaves 6 days later: outside the 3-day window
    (12, -9_000.0),         # only about half leaves
    (12, -25_000.0),        # far more leaves than arrived: not "the same money"
])
def test_outside_the_window_or_the_amount_band_is_not_flagged(day, out):
    ing = _pt(CREDIT, (date(2025, 3, day), "TRANSFER TO DELTA SUPPLY", out))
    assert _find(ing, "FL_RAPID_PASS_THROUGH") == []


def test_a_payment_printed_before_the_credit_cannot_have_been_funded_by_it():
    same_day_before = (date(2025, 3, 10), "TRANSFER TO DELTA SUPPLY", -19_400.0)
    rows = _base() + [same_day_before, CREDIT]        # debit row comes first on the day
    assert _find(_ing(rows), "FL_RAPID_PASS_THROUGH") == []


def test_scheduled_costs_after_a_receipt_are_not_pass_through():
    ing = _pt(CREDIT, (date(2025, 3, 12), "PAYROLL BATCH", -19_400.0))
    assert _find(ing, "FL_RAPID_PASS_THROUGH") == []


def test_small_credits_are_not_considered():
    small = (date(2025, 3, 10), "FAST FROM CUSTOMER 1", 3_000.0)     # under 3x the normal credit
    ing = _pt(small, (date(2025, 3, 11), "FAST TO SUPPLIER", -2_900.0))
    assert _find(ing, "FL_RAPID_PASS_THROUGH") == []


def test_one_payment_cannot_be_matched_to_two_credits():
    c2 = (date(2025, 3, 10), "TRANSFER FROM BETA CAPITAL", 20_000.0)
    ing = _pt(CREDIT, c2, (date(2025, 3, 12), "TRANSFER TO DELTA SUPPLY", -19_400.0))
    found = _find(ing, "FL_RAPID_PASS_THROUGH")
    assert len(found) == 1 and found[0].evidence.startswith("1 large credit")


def test_repeated_pass_through_is_escalated_to_high():
    rows = []
    for m in (2, 4, 6):
        rows += [(date(2025, m, 10), "TRANSFER FROM ALPHA CAPITAL", 20_000.0),
                 (date(2025, m, 11), "TRANSFER TO DELTA SUPPLY", -19_500.0)]
    f = _find(_pt(*rows), "FL_RAPID_PASS_THROUGH")[0]
    assert f.severity.value == "HIGH" and f.weight == 20
    assert f.evidence.startswith("3 large credit")


def test_money_split_over_two_payments_needs_the_setting(monkeypatch):
    ing = _pt(CREDIT, (date(2025, 3, 11), "TRANSFER TO DELTA SUPPLY", -10_000.0),
              (date(2025, 3, 11), "TRANSFER TO ECHO SUPPLY", -9_500.0))
    assert _find(ing, "FL_RAPID_PASS_THROUGH") == []
    monkeypatch.setattr(fl, "_PASS_MAX_PAYMENTS", 2)
    assert len(_find(ing, "FL_RAPID_PASS_THROUGH")) == 1


def test_too_few_credits_to_know_what_is_normal_gives_no_finding():
    rows = [(date(2025, 3, 1), "FAST FROM A", 5_000.0), (date(2025, 3, 2), "FAST TO B", -4_900.0)]
    assert _find(_ing(rows), "FL_RAPID_PASS_THROUGH") == []


def test_undated_rows_are_ignored():
    ing = _pt(CREDIT)
    ing.transactions.append(Transaction(index=999, date=None, description="X", amount=-19_400.0))
    assert _find(ing, "FL_RAPID_PASS_THROUGH") == []
