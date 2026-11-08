"""Task 6.3: property test for Rule C, ``check_turnover_vs_costs``.

Feature: node3-fabrication-rules, Property 8: Turnover rule needs both conditions

**Validates: Requirements 8.3, 8.4, 8.5, 8.6**

The oracle is computed in the test from the generated monthly series
(``statistics.median``), independently of the rule's code. Generated values
keep clear of the floating-point boundaries (>= 3% from the jump ratio,
>= 2 percentage points from the stability band edge).
"""
from __future__ import annotations

import calendar
import statistics
from dataclasses import replace
from datetime import date

from hypothesis import assume, given, settings, strategies as st

from fraud_detect.models import Transaction
from fraud_detect.nodes.fabrication import DEFAULT_CONFIG, check_turnover_vs_costs

from .fixtures import make_ingestion, recompute_balances

CODE = "FL_TURNOVER_COST_MISMATCH"
YEAR = 2026
BASE_FIXED = 20_000.0

TURNOVER_MULTS = st.sampled_from([1.0, 1.0, 1.0, 1.05, 0.9, 1.5, 1.8, 2.5])
FIXED_MULTS = st.sampled_from([1.0, 1.0, 1.0, 1.04, 0.96, 1.3, 0.7, 0.0])


@st.composite
def series(draw):
    n = draw(st.integers(4, 6))
    first = draw(st.integers(1, 12 - n + 1))
    base_t = draw(st.integers(50_000, 200_000))
    months = list(range(first, first + n))
    turnover = [round(base_t * draw(TURNOVER_MULTS), 2) for _ in months]
    fixed = [round(BASE_FIXED * draw(FIXED_MULTS), 2) for _ in months]
    return months, turnover, fixed


def build(months, turnover, fixed):
    """Per month: 3 ordinary credits summing to turnover, PAYROLL + CPF summing to
    fixed costs, plus noise that must not count (LOAN credit, supplier debit)."""
    rows = []
    for m, t, f in zip(months, turnover, fixed):
        a = round(t * 0.5, 2)
        b = round(t * 0.3, 2)
        c = round(t - a - b, 2)
        for day, desc, amt in (
            (5, "FAST FROM ALPHA TRADING", a),
            (10, "FAST FROM BETA SUPPLIES", b),
            (15, "GIRO FROM GAMMA CO", c),
            (12, "LOAN DRAWDOWN", 777_000.0),
            (18, "SUPPLIER PAYMENT", -1_234.56),
        ):
            rows.append((date(YEAR, m, day), desc, amt))
        if f > 0:
            p = round(f * 0.7, 2)
            rows.append((date(YEAR, m, 25), "PAYROLL", -p))
            rows.append((date(YEAR, m, 26), "CPF CONTRIBUTION", -round(f - p, 2)))
    rows.sort(key=lambda r: r[0])
    txns = [Transaction(index=0, date=d, date_raw=d.isoformat(), description=desc, amount=amt)
            for d, desc, amt in rows]
    txns = recompute_balances(txns, 5_000_000.0)
    last = months[-1]
    period = (date(YEAR, months[0], 1), date(YEAR, last, calendar.monthrange(YEAR, last)[1]))
    return make_ingestion(txns, period=period)


def oracle(months, turnover, fixed, cfg):
    """Return (reported, jump_month_numbers, margins_ok)."""
    n = len(months)
    costed = [x for x in fixed if x > 0]
    med_fixed = statistics.median(costed) if costed else 0.0
    ok = True
    stable = []
    for x in fixed:
        if med_fixed > 0:
            dev = abs(x - med_fixed) / med_fixed
            ok = ok and abs(dev - cfg.fixed_cost_stability_pct) >= 0.02
            stable.append(dev <= cfg.fixed_cost_stability_pct)
        else:
            stable.append(False)
    judged = (len(costed) >= cfg.turnover_min_months
              and sum(stable) / n >= cfg.turnover_min_stable_share)
    jumps = []
    for i, m in enumerate(months):
        others = statistics.median(turnover[:i] + turnover[i + 1:])
        if others > 0:
            ok = ok and abs(turnover[i] / others / cfg.credit_jump_ratio - 1) >= 0.03
            if stable[i] and turnover[i] > cfg.credit_jump_ratio * others:
                jumps.append(m)
    reported = judged and len(jumps) >= cfg.turnover_min_jump_months
    return reported, jumps, ok


def check(months, turnover, fixed, cfg):
    reported, jumps, ok = oracle(months, turnover, fixed, cfg)
    assume(ok)
    findings = check_turnover_vs_costs(build(months, turnover, fixed), cfg)
    assert bool(findings) == reported
    if reported:
        assert len(findings) == 1 and findings[0].code == CODE
        ev = findings[0].evidence
        for m in months:
            assert (f"{YEAR}-{m:02d}" in ev) == (m in jumps)
    return reported, jumps


# Feature: node3-fabrication-rules, Property 8: Turnover rule needs both conditions
@settings(max_examples=150, deadline=None)
@given(series())
def test_reported_iff_judged_and_enough_jump_months(s):
    check(*s, DEFAULT_CONFIG)


# Feature: node3-fabrication-rules, Property 8: Turnover rule needs both conditions
@settings(max_examples=150, deadline=None)
@given(
    series(),
    st.sampled_from([1.1, 1.2, 1.35, 1.6]),
    st.integers(1, 3),
    st.sampled_from([0.5, 2 / 3, 0.75]),
)
def test_config_moves_the_boundary_with_the_oracle(s, ratio, min_jumps, share):
    cfg = replace(DEFAULT_CONFIG, credit_jump_ratio=ratio, turnover_min_jump_months=min_jumps,
                  turnover_min_stable_share=share)
    check(*s, cfg)


# Feature: node3-fabrication-rules, Property 8: Turnover rule needs both conditions
@settings(max_examples=150, deadline=None)
@given(series())
def test_raising_jump_month_fixed_costs_beyond_band_removes_them(s):
    months, turnover, fixed = s
    _, jumps = check(months, turnover, fixed, DEFAULT_CONFIG)
    assume(jumps)
    raised = [x * 4 if m in jumps else x for m, x in zip(months, fixed)]
    reported, new_jumps = check(months, turnover, raised, DEFAULT_CONFIG)
    # a jump month whose fixed costs are 4x its own are far outside the band
    # unless the median moved with them; the oracle decides, and it must agree
    # that a month with fixed costs above 1.1x the new median is not a jump month.
    new_med = statistics.median([x for x in raised if x > 0])
    for m, x in zip(months, raised):
        if x > new_med * 1.1:
            assert m not in new_jumps


# Feature: node3-fabrication-rules, Property 8: Turnover rule needs both conditions
@settings(max_examples=150, deadline=None)
@given(series(), st.data())
def test_never_raises_on_rows_with_missing_fields(s, data):
    ing = build(*s)
    junk = []
    for _ in range(data.draw(st.integers(0, 8))):
        junk.append(Transaction(
            index=0,
            date=data.draw(st.one_of(st.none(), st.dates(date(2025, 1, 1), date(2027, 12, 31)))),
            description=data.draw(st.one_of(st.none(), st.just(""),
                                            st.sampled_from(["PAYROLL", "RENT", "LOAN", "X INV12"]))),
            amount=data.draw(st.one_of(st.none(), st.floats(-1e6, 1e6, allow_nan=False))),
        ))
    ing = ing.model_copy(update={"transactions": list(ing.transactions) + junk})
    result = check_turnover_vs_costs(ing, DEFAULT_CONFIG)
    assert isinstance(result, list) and len(result) <= 1
