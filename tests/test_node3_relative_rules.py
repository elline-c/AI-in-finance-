"""Node 3: structuring judged relative to the company, the whole-word
'payer never seen before' rule, and the FinancialLogicConfig object."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from fraud_detect.history import TrailingHistory, counterparty_key
from fraud_detect.models import Transaction
from fraud_detect.nodes import financial_logic as fl
from fraud_detect.nodes.financial_logic import analyze
from fraud_detect.nodes.logic_config import DEFAULT_LOGIC_CONFIG, FinancialLogicConfig

from tests.test_node3_credit_baseline import UEN, _base, _find, _ing


def _cluster(*, amount=9_700.0, start=date(2025, 5, 6), n=3, step=3):
    return [(start + timedelta(days=step * i), "CASH DEPOSIT", amount) for i in range(n)]


# --------------------------------------------------------------------------- #
# FinancialLogicConfig
# --------------------------------------------------------------------------- #
def test_default_config_matches_the_shipped_values():
    c = DEFAULT_LOGIC_CONFIG
    assert (c.gap_default_days, c.gap_min_days, c.gap_max_days) == (25, 10, 35)
    assert c.structuring_levels == (10_000.0,) and c.structuring_band == 0.05
    assert c.structuring_cluster_days == 14 and c.structuring_min_credits == 3
    assert (c.min_period_months, c.max_period_months) == (6, 12)


@pytest.mark.parametrize("change", [
    {"gap_min_days": 40},                       # floor above the cap
    {"pass_match_ratio": 1.2},                  # lower bound above upper bound
    {"structuring_levels": ()},                 # nothing to watch
    {"structuring_levels": (0.0,)},
    {"structuring_band": 1.5},
    {"min_period_months": 13},
    {"pass_max_payments": 0},
    {"credit_min_obs": True},                   # bool is not a count
    {"spike_month_share": -0.1},
])
def test_invalid_settings_are_rejected_early(change):
    with pytest.raises(ValueError):
        replace(DEFAULT_LOGIC_CONFIG, **change)


def test_the_config_is_immutable():
    with pytest.raises(Exception):
        DEFAULT_LOGIC_CONFIG.gap_min_days = 1            # type: ignore[misc]


def test_a_setting_changes_what_the_rule_does():
    days = [date(2025, 1, 1) + timedelta(days=d) for d in (0, 5, 40)]          # a 35-day gap
    rows = [(d, "X", 1.0) for d in days]
    assert len([f for f in analyze(_ing(rows)).findings if f.code == "FL_PERIOD_GAP"]) == 1
    lax = replace(DEFAULT_LOGIC_CONFIG, gap_default_days=40)
    assert [f for f in analyze(_ing(rows), config=lax).findings if f.code == "FL_PERIOD_GAP"] == []


def test_balance_tolerance_is_a_setting():
    t = [Transaction(index=1, date=date(2025, 1, 1), description="X", amount=100.0, balance=1100.05)]
    ing = _ing([])
    ing.transactions, ing.opening_balance = t, 1000.0
    assert [f.code for f in analyze(ing).findings if f.code == "FL_BALANCE_BREAK"] == ["FL_BALANCE_BREAK"]
    loose = replace(DEFAULT_LOGIC_CONFIG, balance_tolerance=0.10)
    assert [f for f in analyze(ing, config=loose).findings if f.code == "FL_BALANCE_BREAK"] == []


# --------------------------------------------------------------------------- #
# Structuring, relative to the company
# --------------------------------------------------------------------------- #
def test_a_cluster_big_for_the_company_fires_and_says_how_big():
    ing = _ing(_base() + _cluster())                      # 29,100 against a 16,750 month
    f = _find(ing, "FL_STRUCTURING")
    assert len(f) == 1 and f[0].severity.value == "HIGH" and f[0].weight == 20
    assert "normal month of credits" in f[0].evidence and "this statement" in f[0].evidence


def test_the_same_cluster_is_noise_for_a_much_bigger_company():
    ing = _ing(_base(20.0) + _cluster())                  # 29,100 against a ~335,000 month
    assert _find(ing, "FL_STRUCTURING") == []
    off = replace(DEFAULT_LOGIC_CONFIG, structuring_month_share=0.0)
    assert len(_find(ing, "FL_STRUCTURING", config=off)) == 1


def _fixed_fee_company():
    # Every credit is a fixed ~9,800 fee: near-level amounts are this company's norm.
    rows = []
    for m in range(1, 10):
        for i in range(12):
            rows.append((date(2025, m, 1 + 2 * i), f"FAST FROM CLIENT {i}", 9_600.0 + 15 * i))
    return rows


def test_a_company_whose_normal_credit_is_near_the_level_is_not_flagged():
    ing = _ing(_fixed_fee_company())
    assert _find(ing, "FL_STRUCTURING") == []
    off = replace(DEFAULT_LOGIC_CONFIG, structuring_routine_share=2.0)
    assert len(_find(ing, "FL_STRUCTURING", config=off)) == 1


def test_the_cluster_does_not_vouch_for_itself():
    # 10 ordinary credits + 3 near-level ones: the 3 would be 23% of 13 credits, but
    # judged against the OTHER credits they are the only near-level ones.
    ordinary = [(date(2025, m, 2 + i), f"FAST FROM CLIENT {i}", 1_000.0 + 10 * i)
                for m in (4, 5, 6) for i in range(4)]
    ing = _ing(ordinary + _cluster(start=date(2025, 5, 20)),
               start=date(2025, 4, 1), end=date(2025, 6, 30))
    assert len(_find(ing, "FL_STRUCTURING")) == 1


def test_without_any_baseline_the_absolute_rule_is_unchanged():
    ing = _ing(_cluster(), start=date(2025, 5, 1), end=date(2025, 5, 31))
    f = _find(ing, "FL_STRUCTURING")
    assert len(f) == 1 and "normal month" not in f[0].evidence


def test_a_declared_monthly_figure_also_sets_the_scale(monkeypatch):
    short = dict(start=date(2025, 5, 1), end=date(2025, 5, 31))
    monkeypatch.setitem(fl.CASHFLOW_BASELINE, UEN, {"avg_monthly_credit": 1_000_000.0})
    assert _find(_ing(_cluster(), **short), "FL_STRUCTURING") == []        # 2.9% of a month
    monkeypatch.setitem(fl.CASHFLOW_BASELINE, UEN, {"avg_monthly_credit": 100_000.0})
    assert len(_find(_ing(_cluster(), **short), "FL_STRUCTURING")) == 1     # 29%


def test_other_reporting_levels_can_be_watched():
    near_20k = _cluster(amount=19_500.0)
    ing = _ing(near_20k, start=date(2025, 5, 1), end=date(2025, 5, 31))
    assert _find(ing, "FL_STRUCTURING") == []                               # default: 10,000 only
    both = replace(DEFAULT_LOGIC_CONFIG, structuring_levels=(10_000.0, 20_000.0))
    f = _find(ing, "FL_STRUCTURING", config=both)
    assert len(f) == 1 and "20,000" in f[0].evidence


def test_each_level_reports_its_own_cluster():
    rows = _cluster(amount=9_700.0) + _cluster(amount=19_400.0, start=date(2025, 8, 4))
    ing = _ing(rows, start=date(2025, 5, 1), end=date(2025, 8, 31))
    both = replace(DEFAULT_LOGIC_CONFIG, structuring_levels=(10_000.0, 20_000.0))
    assert len(_find(ing, "FL_STRUCTURING", config=both)) == 2


def test_the_minimum_cluster_size_is_a_setting():
    ing = _ing(_cluster(n=2), start=date(2025, 5, 1), end=date(2025, 5, 31))
    assert _find(ing, "FL_STRUCTURING") == []
    two = replace(DEFAULT_LOGIC_CONFIG, structuring_min_credits=2)
    assert len(_find(ing, "FL_STRUCTURING", config=two)) == 1


# --------------------------------------------------------------------------- #
# counterparty_key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", [
    ("FAST FROM ACME PTE. LTD. INV12", "GIRO ACME PTE LTD"),
    ("PAYNOW FROM Acme Pte Ltd", "TRANSFER FROM ACME"),
    ("fast from acme pte. ltd. inv-0045", "ACME LIMITED"),
])
def test_the_same_payer_gives_the_same_key(a, b):
    assert counterparty_key(a) == counterparty_key(b) == "ACME"


@pytest.mark.parametrize("desc", ["INWARD REMITTANCE", "CASH DEPOSIT", "INTEREST CREDIT", "", None, 5])
def test_descriptions_without_a_party_give_no_key(desc):
    assert counterparty_key(desc) == ""


def test_inward_tt_is_a_rail_not_a_name():
    assert counterparty_key("INWARD TT UNKNOWN LLC") == "UNKNOWN"
    assert counterparty_key("TRANSFER FROM UNKNOWN TRADING LLC") == "UNKNOWN TRADING"


# --------------------------------------------------------------------------- #
# Large credit from a payer never seen before
# --------------------------------------------------------------------------- #
def _cp(ing, **kw):
    return _find(ing, "FL_UNVERIFIED_COUNTERPARTY", **kw)


def test_letters_inside_a_word_no_longer_trigger_it():
    # "BATTERY" and "REMITTANCE" contain "TT"; the old substring match fired on both.
    short = dict(start=date(2025, 5, 1), end=date(2025, 5, 31))
    for desc in ("FAST FROM BATTERY WORKS PTE LTD", "INWARD REMITTANCE", "MATTRESS HOLDING CO"):
        assert _cp(_ing([(date(2025, 5, 9), desc, 80_000.0)], **short)) == []


def test_a_shell_like_name_still_fires_when_there_is_no_baseline():
    short = dict(start=date(2025, 5, 1), end=date(2025, 5, 31))
    f = _cp(_ing([(date(2025, 5, 9), "TRANSFER FROM UNKNOWN TRADING LLC", 80_000.0)], **short))
    assert len(f) == 1 and "contains LLC, UNKNOWN" in f[0].evidence
    small = _ing([(date(2025, 5, 9), "TRANSFER FROM UNKNOWN TRADING LLC", 40_000.0)], **short)
    assert _cp(small) == []                              # under the fallback amount


def test_a_new_name_is_flagged_when_large_for_the_company_without_any_shell_word():
    ing = _ing(_base() + [(date(2025, 5, 14), "FAST FROM NEWCO PTE LTD", 40_000.0)])
    f = _cp(ing)
    assert len(f) == 1
    assert "'NEWCO'" in f[0].evidence and "other row of this statement" in f[0].evidence


def test_a_payer_seen_elsewhere_on_the_statement_is_not_new():
    ing = _ing(_base() + [(date(2025, 5, 14), "FAST FROM CUSTOMER 3", 40_000.0)])
    assert _cp(ing) == []
    assert len(_find(ing, "FL_SUDDEN_DEPOSIT")) == 1     # still large, and reported as such


def test_a_payer_in_the_approved_history_is_not_new():
    ing = _ing(_base() + [(date(2025, 5, 14), "FAST FROM NEWCO PTE LTD", 40_000.0)])
    hist = TrailingHistory(credits=tuple(a for a in (800.0, 1200.0, 1500.0) * 5),
                           monthly_credits=(16_750.0,) * 4, case_ids=("CASE-OLD",),
                           counterparties=("NEWCO",))
    assert _cp(ing, history=hist) == []


def test_a_large_credit_from_a_new_name_is_not_flagged_if_not_large_for_the_company():
    ing = _ing(_base() + [(date(2025, 5, 14), "FAST FROM NEWCO PTE LTD", 5_000.0)])
    assert _cp(ing) == []


def test_a_shell_like_word_lowers_the_bar_to_the_normal_credit_range():
    # 5,000 is above this company's normal credits but under half a normal month.
    shell = _ing(_base() + [(date(2025, 5, 14), "FAST FROM UNKNOWN TRADING LLC", 5_000.0)])
    plain = _ing(_base() + [(date(2025, 5, 14), "FAST FROM NEWCO PTE LTD", 5_000.0)])
    assert len(_cp(shell)) == 1 and _cp(plain) == []


def test_known_vendors_are_never_flagged():
    ing = _ing(_base() + [(date(2025, 5, 14), "DBS Merchant Payout", 90_000.0)])
    assert _cp(ing) == []


def test_the_fallback_amount_is_a_setting():
    short = dict(start=date(2025, 5, 1), end=date(2025, 5, 31))
    ing = _ing([(date(2025, 5, 9), "TRANSFER FROM UNKNOWN TRADING LLC", 40_000.0)], **short)
    low = replace(DEFAULT_LOGIC_CONFIG, counterparty_fallback_amount=30_000.0)
    assert len(_cp(ing, config=low)) == 1


def test_the_shell_words_are_a_setting():
    short = dict(start=date(2025, 5, 1), end=date(2025, 5, 31))
    ing = _ing([(date(2025, 5, 9), "TRANSFER FROM OFFSHORE TRADING", 80_000.0)], **short)
    assert _cp(ing) == []
    mine = replace(DEFAULT_LOGIC_CONFIG, counterparty_shell_words=("OFFSHORE",))
    assert len(_cp(ing, config=mine)) == 1
