"""Task 6.1: tests for Rule C, ``check_turnover_vs_costs`` (FL_TURNOVER_COST_MISMATCH).
Written before the function exists (task 6.2); expected to fail with
ImportError until then.

Pinned behaviour
----------------
* One finding per file: MEDIUM, weight ``turnover_weight`` (8), agent
  ``Node3:FinancialLogic``, code ``FL_TURNOVER_COST_MISMATCH``.
* Months are calendar months, labelled ``YYYY-MM`` (e.g. ``2026-05``) in the
  evidence. Evidence names every Jump_Month and, for each, contains:
    - the month label,
    - the word ``median`` (median turnover of the OTHER full months, and the
      median fixed costs),
    - turnover, median turnover, fixed costs and median fixed costs, each
      formatted ``f"{value:,.2f}"`` (e.g. ``81,810.09``),
    - the ratio formatted ``f"{ratio:.2f}x"`` (e.g. ``1.25x``).
  Months that are not Jump_Months (and partial months) are not named.
* Full months: with a declared period a month counts only when the period
  covers it from its first to its last day; with no period every month that
  holds a dated row counts.
* Turnover excludes credits holding LOAN/CAPITAL/TRANSFER/INTEREST as whole
  words; Fixed_Costs are debits holding CPF/PAYROLL/SALARY/RENT as whole words
  (case-insensitive). GST is not a default fixed-cost keyword.
* Rows with a missing date, amount or description never raise and are left out.
"""
from __future__ import annotations

import statistics
from dataclasses import replace
from datetime import date

import pytest

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    FabricationConfig,
    check_turnover_vs_costs,
)

from .fixtures import (
    GST_MONTH,
    MONTHS,
    business_file,
    make_ingestion,
    opening_of,
    recompute_balances,
    scale_month_credits,
)

CFG = DEFAULT_CONFIG
MAY = (2026, 5)
AUG = (2026, 8)
JUL = GST_MONTH
FIXED_WORDS = ("CPF", "PAYROLL", "SALARY", "RENT")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _label(key):
    return f"{key[0]}-{key[1]:02d}"


def _in_month(t, key):
    return t.date is not None and (t.date.year, t.date.month) == key


def _scale_fixed(txns, month, factor, word="PAYROLL"):
    """Multiply debits holding ``word`` in ``month`` by ``factor``."""
    out = [
        t.model_copy(update={"amount": round(t.amount * factor, 2)})
        if (t.amount or 0) < 0 and _in_month(t, month) and word in (t.description or "").upper()
        else t.model_copy()
        for t in txns
    ]
    return recompute_balances(out, opening_of(txns))


def _add_row(txns, d, description, amount):
    """Insert a row dated ``d`` (after rows on or before ``d``); balances recomputed."""
    txns = list(txns)
    row = Transaction(index=0, date=d, date_raw=d.strftime("%d %b %Y"),
                      description=description, amount=amount)
    pos = sum(1 for t in txns if t.date is not None and t.date <= d)
    txns.insert(pos, row)
    return recompute_balances(txns, opening_of(txns[:pos] + txns[pos + 1:]))


def _turnover(txns, key):
    return round(sum(t.amount for t in txns if (t.amount or 0) > 0 and _in_month(t, key)), 2)


def _fixed(txns, key):
    return round(sum(-t.amount for t in txns
                     if (t.amount or 0) < 0 and _in_month(t, key)
                     and any(w in (t.description or "").upper().split() for w in FIXED_WORDS)), 2)


def _two_jump_file(period=(date(2026, 4, 1), date(2026, 9, 30))):
    """Business file with credits x1.5 in May and Aug; fixed costs untouched."""
    txns, _ = business_file()
    txns = scale_month_credits(txns, MAY, 1.5)
    txns = scale_month_credits(txns, AUG, 1.5)
    return txns, make_ingestion(txns, period=period)


def _run(txns, period=(date(2026, 4, 1), date(2026, 9, 30)), cfg=CFG):
    return check_turnover_vs_costs(make_ingestion(txns, period=period), cfg)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def test_baseline_business_file_is_silent():
    _, ing = business_file()
    assert check_turnover_vs_costs(ing, CFG) == []


def test_two_jump_months_with_flat_fixed_costs_gives_one_medium_finding():
    _, ing = _two_jump_file()
    findings = check_turnover_vs_costs(ing, CFG)
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "FL_TURNOVER_COST_MISMATCH"
    assert f.severity == Severity.MEDIUM
    assert f.weight == 8.0
    assert f.agent == "Node3:FinancialLogic"


def test_evidence_lists_each_jump_month_with_full_numbers():
    txns, ing = _two_jump_file()
    (f,) = check_turnover_vs_costs(ing, CFG)
    ev = f.evidence
    months = [(2026, m) for m in range(4, 10)]
    turnover = {k: _turnover(txns, k) for k in months}
    fixed = {k: _fixed(txns, k) for k in months}
    median_fixed = statistics.median(fixed.values())
    assert "median" in ev
    for key in (MAY, AUG):
        others = statistics.median(v for k, v in turnover.items() if k != key)
        ratio = turnover[key] / others
        assert _label(key) in ev
        assert f"{turnover[key]:,.2f}" in ev
        assert f"{others:,.2f}" in ev
        assert f"{ratio:.2f}x" in ev
        assert f"{fixed[key]:,.2f}" in ev
    assert f"{median_fixed:,.2f}" in ev
    for key in months:                      # non-jump months are not named
        if key not in (MAY, AUG):
            assert _label(key) not in ev


def test_single_jump_month_is_silent():
    txns, _ = business_file()
    txns = scale_month_credits(txns, AUG, 1.5)
    assert _run(txns) == []


# --------------------------------------------------------------------------
# fixed costs rising with credits
# --------------------------------------------------------------------------
def test_fixed_costs_rising_with_credits_in_jump_months_is_silent():
    txns, _ = _two_jump_file()
    for month in (MAY, AUG):
        txns = _scale_fixed(txns, month, 1.5)       # 28,000 -> 37,000 (> +10%)
    assert _run(txns) == []


def test_only_jump_months_with_flat_costs_count():
    # May's costs rise with credits, Aug's stay flat: one Jump_Month only -> silent.
    txns, _ = _two_jump_file()
    txns = _scale_fixed(txns, MAY, 1.5)
    assert _run(txns) == []


def test_gst_month_does_not_block_by_default():
    txns, _ = business_file()
    txns = scale_month_credits(txns, JUL, 2.2)
    txns = scale_month_credits(txns, AUG, 1.5)
    findings = _run(txns)
    assert len(findings) == 1
    assert _label(JUL) in findings[0].evidence
    assert _label(AUG) in findings[0].evidence


def test_gst_keyword_makes_the_gst_month_a_non_jump_month():
    txns, _ = business_file()
    txns = scale_month_credits(txns, JUL, 2.2)
    txns = scale_month_credits(txns, AUG, 1.5)
    cfg = replace(CFG, fixed_cost_keywords=FIXED_WORDS + ("GST",))
    # July fixed costs 37,500 vs median 28,000 (+34%): not a Jump_Month; Aug alone -> silent
    assert _run(txns, cfg=cfg) == []
    # with a third jump month the file reports, but never names July
    txns = scale_month_credits(txns, MAY, 1.5)
    findings = _run(txns, cfg=cfg)
    assert len(findings) == 1
    assert _label(JUL) not in findings[0].evidence
    assert _label(MAY) in findings[0].evidence and _label(AUG) in findings[0].evidence


# --------------------------------------------------------------------------
# full months
# --------------------------------------------------------------------------
def test_partial_first_and_last_months_are_skipped():
    txns, _ = _two_jump_file()
    txns = scale_month_credits(txns, (2026, 4), 3.0)
    txns = scale_month_credits(txns, (2026, 9), 3.0)
    findings = _run(txns, period=(date(2026, 4, 15), date(2026, 9, 15)))
    # April and September are partial: May and Aug remain the jump months
    assert len(findings) == 1
    ev = findings[0].evidence
    assert _label(MAY) in ev and _label(AUG) in ev
    assert "2026-04" not in ev and "2026-09" not in ev


def test_jump_in_partial_month_does_not_count():
    txns, _ = business_file()
    txns = scale_month_credits(txns, (2026, 4), 3.0)
    txns = scale_month_credits(txns, AUG, 1.5)
    assert _run(txns, period=(date(2026, 4, 15), date(2026, 9, 15))) == []


def test_unknown_period_counts_every_month_with_a_dated_row():
    txns, _ = business_file()
    txns = scale_month_credits(txns, (2026, 4), 3.0)
    txns = scale_month_credits(txns, AUG, 1.5)
    findings = _run(txns, period=None)
    assert len(findings) == 1
    assert _label((2026, 4)) in findings[0].evidence


def test_fewer_than_four_full_months_is_silent():
    txns, _ = business_file()
    txns = scale_month_credits(txns, JUL, 2.0)
    txns = scale_month_credits(txns, AUG, 2.0)
    assert len(_run(txns)) == 1                                   # control: full period reports
    assert _run(txns, period=(date(2026, 6, 1), date(2026, 8, 31))) == []   # 3 full months


def test_fewer_than_four_months_with_fixed_costs_is_silent():
    txns, _ = _two_jump_file()
    gone = [t for t in txns
            if not ((t.amount or 0) < 0 and t.date and t.date.month in (4, 6, 7)
                    and any(w in (t.description or "").upper().split() for w in FIXED_WORDS))]
    gone = recompute_balances(gone, opening_of(txns))
    assert _run(gone) == []


# --------------------------------------------------------------------------
# stability share
# --------------------------------------------------------------------------
def _unstable_costs_file():
    """Payroll factors 1.0,1.3,1.6,1.0,1.3,1.6 for Apr..Sep; credits x1.5 in May and Aug.

    Fixed costs are 28,000 / 33,400 / 38,800 (median 33,400): only May and Aug
    (the two jump months) sit inside the band, i.e. 2 of 6 months are stable.
    """
    txns, _ = _two_jump_file()
    for (y, m), f in zip(MONTHS, (1.0, 1.3, 1.6, 1.0, 1.3, 1.6)):
        txns = _scale_fixed(txns, (y, m), f)
    return txns


def test_fewer_than_two_thirds_stable_months_is_silent():
    assert _run(_unstable_costs_file()) == []


def test_stable_share_override_lets_the_same_file_report():
    cfg = replace(CFG, turnover_min_stable_share=0.3)
    findings = _run(_unstable_costs_file(), cfg=cfg)
    assert len(findings) == 1


# --------------------------------------------------------------------------
# turnover exclusions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("description", [
    "LOAN DRAWDOWN", "loan drawdown", "CAPITAL INJECTION", "TRANSFER FROM DIRECTOR",
    "INTEREST CREDIT",
])
def test_excluded_credits_are_not_turnover(description):
    txns, _ = business_file()
    txns = scale_month_credits(txns, MAY, 1.5)
    txns = _add_row(txns, date(2026, 8, 20), description, 60_000.00)
    assert _run(txns) == []          # only May is a jump month


def test_same_size_ordinary_credit_does_make_a_jump_month():
    txns, _ = business_file()
    txns = scale_month_credits(txns, MAY, 1.5)
    txns = _add_row(txns, date(2026, 8, 20), "FAST FROM QUILLON DESIGN INV9001", 60_000.00)
    findings = _run(txns)
    assert len(findings) == 1
    assert _label(AUG) in findings[0].evidence


# --------------------------------------------------------------------------
# whole-word, case-insensitive keywords
# --------------------------------------------------------------------------
def _with_debits_in_jump_months(description, amount=20_000.00):
    txns, _ = _two_jump_file()
    for month in (MAY, AUG):
        txns = _add_row(txns, date(month[0], month[1], 10), description, -amount)
    return txns


def test_current_account_fee_is_not_rent():
    assert len(_run(_with_debits_in_jump_months("CURRENT ACCOUNT FEE"))) == 1


def test_rent_as_a_whole_word_counts():
    assert _run(_with_debits_in_jump_months("RENT OSTERLY RISE HOLDINGS")) == []


def test_keyword_matching_is_case_insensitive():
    assert _run(_with_debits_in_jump_months("rent osterly rise holdings")) == []
    assert _run(_with_debits_in_jump_months("Payroll Bonus Run")) == []


# --------------------------------------------------------------------------
# no whole-file ratio trigger (Requirement 8.8)
# --------------------------------------------------------------------------
def test_high_credits_everywhere_is_silent():
    txns, _ = business_file()
    for key in MONTHS:
        txns = scale_month_credits(txns, key, 3.0)
    credits = sum(t.amount for t in txns if t.amount > 0)
    debits = -sum(t.amount for t in txns if t.amount < 0)
    assert credits / debits > 2.5                      # whole-file ratio is very high
    assert _run(txns) == []


def test_fixed_costs_scaling_with_credits_everywhere_is_silent():
    txns, _ = _two_jump_file()
    for key in MONTHS:
        txns = _scale_fixed(txns, key, 1.5 if key in (MAY, AUG) else 1.0)
    assert _run(txns) == []


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------
def test_ratio_override_reports_smaller_jumps():
    txns, ing = business_file()
    assert check_turnover_vs_costs(ing, CFG) == []                    # default: largest ratio ~1.11
    low = replace(CFG, credit_jump_ratio=1.1)
    findings = check_turnover_vs_costs(ing, low)
    assert len(findings) == 1
    assert findings[0].code == "FL_TURNOVER_COST_MISMATCH"


def test_min_jump_months_override_reports_a_single_jump():
    txns, _ = business_file()
    txns = scale_month_credits(txns, AUG, 1.5)
    assert _run(txns) == []
    findings = _run(txns, cfg=replace(CFG, turnover_min_jump_months=1))
    assert len(findings) == 1
    assert _label(AUG) in findings[0].evidence


def test_weight_override():
    _, ing = _two_jump_file()
    (f,) = check_turnover_vs_costs(ing, replace(CFG, turnover_weight=3.0))
    assert f.weight == 3.0


def test_config_type_is_the_fabrication_config():
    assert isinstance(CFG, FabricationConfig)


# --------------------------------------------------------------------------
# determinism, purity, robustness
# --------------------------------------------------------------------------
def test_deterministic_and_input_not_mutated():
    _, ing = _two_jump_file()
    before = ing.model_copy(deep=True)
    first = check_turnover_vs_costs(ing, CFG)
    second = check_turnover_vs_costs(ing, CFG)
    assert first == second
    assert ing == before


def test_rows_with_missing_fields_never_raise_and_are_left_out():
    txns, _ = _two_jump_file()
    junk = [
        Transaction(index=0, date=None, description="PAYROLL", amount=-5_000.0),
        Transaction(index=0, date=None, description="FAST FROM X INV1234", amount=90_000.0),
        Transaction(index=0, date=date(2026, 6, 10), description="SOMETHING", amount=None),
        Transaction(index=0, date=date(2026, 6, 11), description=None, amount=5_000.0),
        Transaction(index=0, date=date(2026, 6, 12), description=None, amount=-5_000.0),
        Transaction(index=0, date=None, description=None, amount=None),
    ]
    withjunk = recompute_balances(list(txns) + junk, opening_of(txns))
    result = _run(withjunk)
    assert isinstance(result, list)
    # the two real jump months are still found (the description-less credit adds
    # 5,000 to June only, which does not make June a jump month)
    assert len(result) == 1


def test_only_junk_rows_never_raise():
    junk = [
        Transaction(index=1, date=None, description=None, amount=None),
        Transaction(index=2, date=date(2026, 5, 1), description=None, amount=None),
        Transaction(index=3, date=None, description="RENT", amount=-1.0),
    ]
    assert _run(junk) == []


def test_no_transactions_gives_empty_list():
    assert check_turnover_vs_costs(make_ingestion([]), CFG) == []
