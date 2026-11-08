"""Tests for fraud_detect.nodes.loan_sizing.indicative_loan_capacity.

Green-by-construction: asserted numbers are hand-computed from the exact formula
the helper uses (and the annuity factor is recomputed inline so the amortisation
is pinned), so the suite passes once pydantic/pytest are available on the user's
laptop. Covers a normal positive-surplus case, the zero-rate branch, the
no-surplus -> 0 case, missing inputs -> insufficient, the binding-constraint
selection, and the fit-vs-requested verdict.
"""
from __future__ import annotations

from fraud_detect.models import CreditMemo
from fraud_detect.nodes.loan_sizing import (
    DEFAULT_ANNUAL_RATE,
    DEFAULT_SIZING_DSCR,
    DEFAULT_TENOR_YEARS,
    REVENUE_MULTIPLE_CAP,
    TARGET_MAX_DEBT_TO_EQUITY,
    indicative_loan_capacity,
)


def _annuity(monthly_payment: float, annual_rate: float, years: int) -> float:
    n = years * 12
    i = annual_rate / 12
    if i == 0:
        return monthly_payment * n
    return monthly_payment * (1 - (1 + i) ** (-n)) / i


# --------------------------------------------------------------------------- #
# Normal case: positive surplus -> sensible principal, affordability binds.
# --------------------------------------------------------------------------- #
def test_normal_positive_surplus_affordability_binds():
    # net monthly 10000 -> annual OCF 120000; existing debt service 2000*12 =
    # 24000; surplus 96000; affordable annual 96000/1.25 = 76800; monthly 6400.
    memo = CreditMemo(
        monthly_debt_repayments=2000.0,
        reported_annual_revenue=5_000_000.0,   # huge -> revenue cap won't bind
        equity=10_000_000.0, total_debt=0.0,   # huge headroom -> leverage won't bind
    )
    metrics = {"net_monthly_cashflow": 10000.0}
    res = indicative_loan_capacity(memo, metrics)

    assert res["available"] is True
    assert res["insufficient_inputs"] is False
    assert res["surplus_annual"] == 96000.0
    assert res["affordable_annual_payment"] == 76800.0
    assert res["affordable_monthly_payment"] == 6400.0

    expected_principal = _annuity(6400.0, DEFAULT_ANNUAL_RATE, DEFAULT_TENOR_YEARS)
    assert abs(res["indicative_max"] - expected_principal) < 1e-6
    assert res["binding_key"] == "affordability"
    assert "affordability" in res["binding_constraint"]
    # A 5-year 8% loan on 6,400/mo is roughly 315k-316k.
    assert 300_000 < res["indicative_max"] < 330_000


# --------------------------------------------------------------------------- #
# Zero-rate branch: principal == affordable_monthly * n.
# --------------------------------------------------------------------------- #
def test_zero_rate_branch_is_linear():
    memo = CreditMemo(
        monthly_debt_repayments=0.0,
        reported_annual_revenue=5_000_000.0,
        equity=10_000_000.0, total_debt=0.0,
    )
    metrics = {"net_monthly_cashflow": 12500.0}  # annual 150000; cushion 1.25
    res = indicative_loan_capacity(memo, metrics, annual_rate=0.0, tenor_years=5)
    # affordable annual = 150000 / 1.25 = 120000; monthly = 10000; n = 60.
    assert res["affordable_monthly_payment"] == 10000.0
    assert res["indicative_max"] == 10000.0 * 60  # == 600000


# --------------------------------------------------------------------------- #
# No surplus -> capacity 0 with a clear message.
# --------------------------------------------------------------------------- #
def test_no_surplus_capacity_zero():
    memo = CreditMemo(monthly_debt_repayments=20000.0)  # 240k/yr existing
    metrics = {"net_monthly_cashflow": 5000.0}          # 60k/yr OCF -> negative surplus
    res = indicative_loan_capacity(memo, metrics)
    assert res["available"] is True
    assert res["indicative_max"] == 0.0
    aff = next(c for c in res["caps"] if c["key"] == "affordability")
    assert aff["value"] == 0.0
    assert "no surplus" in aff["note"].lower()


# --------------------------------------------------------------------------- #
# Missing inputs -> insufficient, no fabricated number, no exception.
# --------------------------------------------------------------------------- #
def test_missing_memo_is_insufficient():
    res = indicative_loan_capacity(None, {"net_monthly_cashflow": 10000.0})
    assert res["available"] is False
    assert res["insufficient_inputs"] is True
    assert res["indicative_max"] is None
    assert res["verdict"].startswith("cannot assess")


def test_missing_cashflow_is_insufficient():
    res = indicative_loan_capacity(CreditMemo(monthly_debt_repayments=1000.0), {})
    assert res["available"] is False
    assert res["indicative_max"] is None


# --------------------------------------------------------------------------- #
# Binding-constraint selection: leverage binds; revenue binds.
# --------------------------------------------------------------------------- #
def test_leverage_cap_binds():
    # Big surplus (affordability large) and big revenue, but tiny equity headroom.
    memo = CreditMemo(
        monthly_debt_repayments=0.0,
        reported_annual_revenue=5_000_000.0,   # revenue cap = 5,000,000
        equity=100_000.0, total_debt=150_000.0,  # max debt 200k, headroom 50k
    )
    metrics = {"net_monthly_cashflow": 50000.0}
    res = indicative_loan_capacity(memo, metrics)
    assert res["binding_key"] == "leverage"
    # headroom = 100000 * 2.0 - 150000 = 50000
    assert res["indicative_max"] == 100_000.0 * TARGET_MAX_DEBT_TO_EQUITY - 150_000.0
    assert "leverage" in res["binding_constraint"]


def test_revenue_cap_binds():
    # Big affordability + big leverage headroom, but small revenue multiple cap.
    memo = CreditMemo(
        monthly_debt_repayments=0.0,
        reported_annual_revenue=120_000.0,     # revenue cap = 120,000 * 1.0
        equity=10_000_000.0, total_debt=0.0,   # leverage headroom huge
    )
    metrics = {"net_monthly_cashflow": 50000.0}  # affordability large
    res = indicative_loan_capacity(memo, metrics)
    assert res["binding_key"] == "revenue_multiple"
    assert res["indicative_max"] == 120_000.0 * REVENUE_MULTIPLE_CAP


# --------------------------------------------------------------------------- #
# Fit-vs-requested verdict: within / above.
# --------------------------------------------------------------------------- #
def _memo_for_verdict():
    return CreditMemo(
        monthly_debt_repayments=0.0,
        reported_annual_revenue=5_000_000.0,
        equity=10_000_000.0, total_debt=0.0,
    )


def test_verdict_within_capacity():
    res = indicative_loan_capacity(
        _memo_for_verdict(), {"net_monthly_cashflow": 50000.0},
        requested_amount=10_000.0, currency="SGD",
    )
    assert res["verdict"] == "within indicative capacity"
    assert res["verdict_tone"] == "ok"


def test_verdict_above_capacity():
    res = indicative_loan_capacity(
        _memo_for_verdict(), {"net_monthly_cashflow": 50000.0},
        requested_amount=999_000_000.0, currency="SGD",
    )
    assert res["verdict"] == "above indicative capacity"
    assert res["verdict_tone"] == "warn"


def test_verdict_no_requested_amount():
    res = indicative_loan_capacity(
        _memo_for_verdict(), {"net_monthly_cashflow": 50000.0},
    )
    assert res["verdict"] == "no requested amount captured"


def test_assumptions_default_values():
    res = indicative_loan_capacity(
        _memo_for_verdict(), {"net_monthly_cashflow": 50000.0})
    a = res["assumptions"]
    assert a["annual_rate"] == DEFAULT_ANNUAL_RATE
    assert a["tenor_years"] == DEFAULT_TENOR_YEARS
    assert a["dscr_cushion"] == DEFAULT_SIZING_DSCR
