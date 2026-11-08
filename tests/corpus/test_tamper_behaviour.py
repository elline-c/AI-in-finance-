"""Behavioural tampers (task 16.1).

Written first: these fail with a registry ``KeyError`` until task 16.2 adds
``fraud_detect/corpus/tamper/behaviour.py`` and registers the seven tampers.

Interface (from the design): ``CATALOG[name].plan(ledger, renderer, rng) -> TamperResult``.
Variants (fast/slow, large/moderate) are chosen by the seed, so the tests scan seeds
and classify each result from the data itself; both variants must occur.
Inserted rows are the rows whose ``row_id`` is not in the genuine ledger.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pytest

from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper.catalog import CATALOG
from tests.corpus.helpers import load_renderer, make_test_ledger

SEEDS = range(16)
N_MONTHS = 9
BEHAVIOURAL = ("structuring", "sudden_large_deposit", "shell_counterparty",
               "rapid_pass_through", "dormant_reactivation", "period_gap", "uen_mismatch")
REPLAYABLE = tuple(n for n in BEHAVIOURAL if n != "uen_mismatch")
SHELL_WORDS = ("UNKNOWN", "LLC", "HOLDINGS")
QUIET_KINDS = {"service_charge", "interest"}


def _renderer():
    r = load_renderer("two_col_slash")
    if r is None:
        pytest.skip("renderer not implemented yet")
    return r


@lru_cache(maxsize=None)
def _genuine(seed: int):
    return make_test_ledger(N_MONTHS, 2000 + seed)


@lru_cache(maxsize=None)
def _plan(name: str, seed: int):
    ledger = _genuine(seed)
    return ledger, CATALOG[name].plan(ledger, _renderer(), rng(seed, "behaviour-test", name))


def _all(name: str):
    return [_plan(name, s) for s in SEEDS]


def _inserted(ledger, result):
    old = {r.row_id for r in ledger.rows}
    return [r for r in result.ledger.rows if r.row_id not in old]


def _avg_monthly_credit(ledger) -> float:
    return sum(r.amount for r in ledger.rows if r.amount > 0) / len(ledger.months)


def _balances(ledger):
    if ledger.printed_balances is not None:
        return list(ledger.printed_balances)
    out, bal = [], ledger.opening_balance
    for r in ledger.rows:
        bal = round(bal + r.amount, 2)
        out.append(bal)
    return out


def _sections_consistent(ledger) -> bool:
    bals = _balances(ledger)
    if len(bals) != len(ledger.rows):
        return False
    by_month = defaultdict(list)
    for i, r in enumerate(ledger.rows):
        by_month[r.month_index].append(i)
    for idx in by_month.values():
        for a, b in zip(idx, idx[1:]):
            if abs(bals[b] - bals[a] - ledger.rows[b].amount) > 0.005:
                return False
    return True


def _fraud_ids(result) -> set[str]:
    return {r.row_id for r in result.ledger.rows if r.fraud}


# -- every behavioural tamper -------------------------------------------------
def test_catalog_registers_the_seven_behavioural_tampers():
    for name in BEHAVIOURAL:
        assert name in CATALOG
        assert CATALOG[name].name == name
        assert CATALOG[name].group == "Behaviour"


@pytest.mark.parametrize("name", BEHAVIOURAL)
def test_label_and_sections_are_consistent(name):
    for ledger, result in _all(name):
        label = result.label
        assert label.is_tampered and label.tamper_types == (name,)
        assert label.method in CATALOG[name].methods
        assert label.arithmetic_consistent is True
        assert _sections_consistent(result.ledger)
        assert label.method != "overlay" and not result.file_ops


@pytest.mark.parametrize("name", BEHAVIOURAL)
def test_plan_is_deterministic_for_the_same_seed(name):
    ledger = _genuine(1)
    a = CATALOG[name].plan(ledger, _renderer(), rng(1, "det", name))
    b = CATALOG[name].plan(ledger, _renderer(), rng(1, "det", name))
    assert a == b


@pytest.mark.parametrize("name", REPLAYABLE)
def test_recorded_edits_replay_to_the_edited_ledger(name):
    for ledger, result in _all(name):
        assert result.edits
        assert ledger_ops.apply_edits(ledger, result.edits) == result.ledger


@pytest.mark.parametrize("name", ["structuring", "sudden_large_deposit", "shell_counterparty",
                                  "rapid_pass_through"])
def test_insert_tampers_flag_exactly_the_inserted_rows(name):
    for ledger, result in _all(name):
        inserted = _inserted(ledger, result)
        assert inserted
        assert {r.row_id for r in inserted} == _fraud_ids(result)
        assert {a["row_id"] for a in result.label.affected_rows} == _fraud_ids(result)
        assert not any(r.fraud for r in ledger.rows)
        kept = [r for r in result.ledger.rows if not r.fraud]
        assert kept == list(ledger.rows)


# -- structuring ----------------------------------------------------------------
def test_structuring_inserts_3_to_5_just_under_threshold_credits():
    spans = set()
    for ledger, result in _all("structuring"):
        rows = _inserted(ledger, result)
        assert 3 <= len(rows) <= 5
        assert all(9500.00 <= r.amount <= 9999.99 for r in rows)
        days = (max(r.date for r in rows) - min(r.date for r in rows)).days
        spans.add("fast" if days <= 14 else "slow")
        assert days <= 14 or days > 30
    assert spans == {"fast", "slow"}


# -- sudden_large_deposit -------------------------------------------------------
def test_sudden_large_deposit_is_one_credit_above_the_average_multiple():
    variants = set()
    for ledger, result in _all("sudden_large_deposit"):
        (row,) = _inserted(ledger, result)
        assert row.amount > 0
        ratio = row.amount / _avg_monthly_credit(ledger)
        if ratio > 3:
            variants.add("large")
        else:
            assert 1.5 * 0.98 <= ratio <= 2.5 * 1.02, ratio
            variants.add("moderate")
    assert variants == {"large", "moderate"}


# -- shell_counterparty ---------------------------------------------------------
def test_shell_counterparty_is_a_big_credit_from_a_shell_like_name():
    for ledger, result in _all("shell_counterparty"):
        (row,) = _inserted(ledger, result)
        assert row.amount >= 50000.00
        assert any(w in row.description.upper() for w in SHELL_WORDS)


# -- rapid_pass_through ---------------------------------------------------------
def test_rapid_pass_through_adds_credit_debit_pairs_within_three_days():
    for ledger, result in _all("rapid_pass_through"):
        rows = _inserted(ledger, result)
        credits = [r for r in rows if r.amount > 0]
        debits = [r for r in rows if r.amount < 0]
        assert 2 <= len(credits) <= 4 and len(debits) == len(credits)
        unused = list(debits)
        for c in credits:
            match = next((d for d in unused
                          if 1 <= (d.date - c.date).days <= 3
                          and 0.95 - 1e-9 <= -d.amount / c.amount <= 1.0 + 1e-9
                          and d.description != c.description), None)
            assert match is not None, f"no pass-through debit for {c}"
            unused.remove(match)


# -- dormant_reactivation -------------------------------------------------------
def test_dormant_reactivation_leaves_only_charges_in_first_3_to_5_months():
    lengths = set()
    for ledger, result in _all("dormant_reactivation"):
        by_month = defaultdict(list)
        for r in result.ledger.rows:
            by_month[r.month_index].append(r)
        k = 0
        while k in by_month and all(r.kind in QUIET_KINDS for r in by_month[k]):
            k += 1
        assert 3 <= k <= 5
        lengths.add(k)
        assert all(any(r.kind not in QUIET_KINDS for r in by_month[m]) for m in range(k, N_MONTHS))
        # later months keep their genuine activity
        genuine_later = [r for r in ledger.rows if r.month_index >= k]
        kept_later = [r for r in result.ledger.rows if r.month_index >= k]
        assert [(r.row_id, r.date, r.description, r.amount) for r in kept_later] == \
               [(r.row_id, r.date, r.description, r.amount) for r in genuine_later]
        assert len(result.ledger.months) == N_MONTHS
    assert len(lengths) >= 2


# -- period_gap -----------------------------------------------------------------
def test_period_gap_removes_one_middle_month_and_lists_it():
    for ledger, result in _all("period_gap"):
        new = result.ledger
        assert len(new.months) == len(ledger.months) - 1
        (missing,) = [m for m in ledger.months if m not in new.months]
        assert missing not in (ledger.months[0], ledger.months[-1])
        tag = f"{missing.year:04d}-{missing.month:02d}"
        assert result.label.affected_months == (tag,)
        assert not any((r.date.year, r.date.month) == (missing.year, missing.month) for r in new.rows)
        gone = {r.row_id for r in ledger.rows if (r.date.year, r.date.month) == (missing.year, missing.month)}
        assert {r.row_id for r in new.rows} == {r.row_id for r in ledger.rows} - gone
        # remaining rows are untouched and the sections stay consistent
        genuine = {r.row_id: r for r in ledger.rows}
        for r in new.rows:
            assert (r.date, r.description, r.amount) == (genuine[r.row_id].date,
                                                        genuine[r.row_id].description,
                                                        genuine[r.row_id].amount)
            assert new.months[r.month_index].month == r.date.month
        assert not any(r.fraud for r in new.rows)


# -- uen_mismatch ---------------------------------------------------------------
def test_uen_mismatch_prints_a_different_holder_and_keeps_the_registered_name():
    for ledger, result in _all("uen_mismatch"):
        new = result.ledger
        assert new.printed_holder and new.printed_holder.strip()
        assert new.printed_holder != ledger.company.name
        assert new.company.name == ledger.company.name
        assert new.company.uen == ledger.company.uen
        assert new.rows == ledger.rows and new.months == ledger.months
        assert result.label.method == "content"
