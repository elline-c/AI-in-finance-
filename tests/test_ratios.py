"""Tests for fraud_detect.nodes.ratios.lending_ratios (FEAT-001, Part A).

Green-by-construction: every asserted number is hand-computed from the exact
formula the helper uses, so the suite passes once pydantic/pytest are available
on the user's laptop. Covers full-input known values, missing-input handling,
divide-by-zero guards, and the DSCR < 1.2 concern boundary.
"""
from __future__ import annotations

from fraud_detect.models import CreditMemo
from fraud_detect.nodes.ratios import DSCR_CONCERN_THRESHOLD, lending_ratios


def _full_memo(**overrides):
    base = dict(
        accounts_receivable=50000.0,
        accounts_payable=40000.0,
        inventory=20000.0,
        total_debt=120000.0,
        monthly_debt_repayments=5000.0,
        reported_annual_revenue=600000.0,
        equity=60000.0,
    )
    base.update(overrides)
    return CreditMemo(**base)


# --------------------------------------------------------------------------- #
# Full inputs -> known results
# --------------------------------------------------------------------------- #
def test_full_inputs_known_values():
    memo = _full_memo()
    # closing cash balance passed through metrics; net monthly cash flow 10000.
    metrics = {"closing_balance": 30000.0, "net_monthly_cashflow": 10000.0}
    r = lending_ratios(memo, metrics)

    # current_assets = 50000 + 20000 + 30000 = 100000
    # current_liabilities = 40000 + 5000*12 = 100000
    assert r["current_ratio"]["value"] == 1.0
    # quick_assets = 100000 - 20000 = 80000; 80000 / 100000 = 0.8
    assert r["quick_ratio"]["value"] == 0.8
    # debt_to_equity = 120000 / 60000 = 2.0
    assert r["debt_to_equity"]["value"] == 2.0
    # dscr = (10000*12) / (5000*12) = 120000 / 60000 = 2.0
    assert r["dscr"]["value"] == 2.0
    assert r["dscr"]["concern"] is False
    # debt_to_revenue = 120000 / 600000 = 0.2
    assert r["debt_to_revenue"]["value"] == 0.2

    # Every entry carries formula + interpretation + caveat.
    for name, entry in r.items():
        assert entry["formula"]
        assert entry["interpretation"]
        assert "not audited" in entry["caveat"]


# --------------------------------------------------------------------------- #
# Missing-input handling: None value + explanatory note, no crash
# --------------------------------------------------------------------------- #
def test_missing_inputs_return_none_with_note():
    memo = CreditMemo()  # entirely empty
    r = lending_ratios(memo, {})
    for name in ("current_ratio", "quick_ratio", "debt_to_equity",
                 "dscr", "debt_to_revenue"):
        assert r[name]["value"] is None
        assert "Need" in r[name]["interpretation"]
    # DSCR still exposes a (False) concern flag when it cannot be computed.
    assert r["dscr"]["concern"] is False


def test_none_memo_does_not_crash():
    r = lending_ratios(None, {"closing_balance": 10000.0})
    assert r["current_ratio"]["value"] is None
    assert r["dscr"]["value"] is None


# --------------------------------------------------------------------------- #
# Divide-by-zero guards -> None value + note
# --------------------------------------------------------------------------- #
def test_zero_equity_guards_debt_to_equity():
    memo = _full_memo(equity=0.0)
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 10000.0})
    assert r["debt_to_equity"]["value"] is None
    assert "equity" in r["debt_to_equity"]["interpretation"]


def test_zero_revenue_guards_debt_to_revenue():
    memo = _full_memo(reported_annual_revenue=0.0)
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 10000.0})
    assert r["debt_to_revenue"]["value"] is None
    assert "reported_annual_revenue" in r["debt_to_revenue"]["interpretation"]


def test_zero_liabilities_guards_current_and_quick():
    # accounts_payable 0 AND no debt repayments -> current_liabilities = 0.
    memo = _full_memo(accounts_payable=0.0, monthly_debt_repayments=0.0)
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 10000.0})
    # current_liabilities = 0 + 0*12 = 0 -> guarded to None
    assert r["current_ratio"]["value"] is None
    assert r["quick_ratio"]["value"] is None


def test_zero_debt_service_guards_dscr():
    memo = _full_memo(monthly_debt_repayments=0.0)
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 10000.0})
    # annual_debt_service = 0 -> dscr guarded to None
    assert r["dscr"]["value"] is None
    assert r["dscr"]["concern"] is False


# --------------------------------------------------------------------------- #
# DSCR concern threshold boundary (< 1.2 is a concern; >= 1.2 is not)
# --------------------------------------------------------------------------- #
def test_dscr_below_threshold_flags_concern():
    # net monthly 5000 -> annual OCF 60000; debt service 5000*12 = 60000.
    # dscr = 60000 / 60000 = 1.0 < 1.2 -> concern.
    memo = _full_memo()
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 5000.0})
    assert r["dscr"]["value"] == 1.0
    assert r["dscr"]["value"] < DSCR_CONCERN_THRESHOLD
    assert r["dscr"]["concern"] is True
    assert r["dscr"]["concern_note"]


def test_dscr_at_threshold_is_not_a_concern():
    # net monthly 6000 -> annual OCF 72000; debt service 60000.
    # dscr = 72000 / 60000 = 1.2 == threshold -> NOT a concern (>=).
    memo = _full_memo()
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 6000.0})
    assert r["dscr"]["value"] == DSCR_CONCERN_THRESHOLD
    assert r["dscr"]["concern"] is False


def test_dscr_just_above_threshold_is_not_a_concern():
    # net monthly 7000 -> annual OCF 84000; debt service 60000.
    # dscr = 84000 / 60000 = 1.4 > 1.2 -> not a concern.
    memo = _full_memo()
    r = lending_ratios(memo, {"closing_balance": 30000.0,
                              "net_monthly_cashflow": 7000.0})
    assert r["dscr"]["value"] == 1.4
    assert r["dscr"]["concern"] is False
