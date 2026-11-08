"""Tests for the Node 1 self-check (Requirement 9).

Written ahead of the implementation (task 8.2); they fail with
ModuleNotFoundError until ``selfcheck.py`` exists.
"""
from __future__ import annotations

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes.ingest_parse.header import SectionHeader
from fraud_detect.nodes.ingest_parse.selfcheck import self_check

CODE = "DQ_PARSE_INCONSISTENT"


def _header(opening=1000.0, closing=None, withdrawals=None, deposits=None) -> SectionHeader:
    return SectionHeader(
        holder=None,
        uen=None,
        account_number=None,
        currency=None,
        period_start=None,
        period_end=None,
        statement_date=None,
        opening=opening,
        closing=closing,
        printed_total_withdrawals=withdrawals,
        printed_total_deposits=deposits,
        page=1,
    )


def _rows(opening: float, amounts: list[float]) -> list[Transaction]:
    rows, bal = [], opening
    for i, a in enumerate(amounts, start=1):
        bal = round(bal + a, 2)
        rows.append(Transaction(index=i, amount=a, balance=bal, page=1))
    return rows


# opening 1000: +500, -200, -50.50 -> 1249.50; deposits 500, withdrawals 250.50
AMOUNTS = [500.0, -200.0, -50.50]


def _codes(findings):
    return [f.code for f in findings]


def test_consistent_section_has_no_finding():
    h = _header(closing=1249.50, withdrawals=250.50, deposits=500.0)
    assert self_check(h, _rows(1000.0, AMOUNTS)) == []


def test_withdrawal_total_mismatch_emits_finding():
    h = _header(closing=1249.50, withdrawals=300.00, deposits=500.0)
    findings = self_check(h, _rows(1000.0, AMOUNTS))
    assert _codes(findings) == [CODE]
    f = findings[0]
    assert f.severity == Severity.LOW
    assert f.weight == 3
    assert f.agent == "Node1:Ingestion"


def test_deposit_total_mismatch_emits_finding():
    h = _header(closing=1249.50, withdrawals=250.50, deposits=450.0)
    assert _codes(self_check(h, _rows(1000.0, AMOUNTS))) == [CODE]


def test_difference_within_tolerance_is_ignored():
    h = _header(closing=1249.50, withdrawals=250.51, deposits=499.99)
    assert self_check(h, _rows(1000.0, AMOUNTS)) == []


def test_difference_just_above_tolerance_is_flagged():
    h = _header(closing=1249.50, withdrawals=250.52, deposits=500.0)
    assert _codes(self_check(h, _rows(1000.0, AMOUNTS))) == [CODE]


def test_unread_printed_totals_are_skipped():
    h = _header(closing=1249.50, withdrawals=None, deposits=None)
    assert self_check(h, _rows(1000.0, AMOUNTS)) == []


def test_only_unread_total_is_skipped_other_still_compared():
    h = _header(closing=1249.50, withdrawals=None, deposits=450.0)
    assert _codes(self_check(h, _rows(1000.0, AMOUNTS))) == [CODE]


def test_row_balance_break_is_left_to_node3():
    rows = _rows(1000.0, AMOUNTS)
    rows[1] = rows[1].model_copy(update={"balance": 9999.99})  # row does not follow
    h = _header(closing=1249.50, withdrawals=300.00, deposits=500.0)
    assert CODE not in _codes(self_check(h, rows))


def test_closing_mismatch_alone_is_left_to_node3():
    h = _header(closing=5000.0, withdrawals=250.50, deposits=500.0)
    assert self_check(h, _rows(1000.0, AMOUNTS)) == []


def test_empty_rows_with_no_totals_has_no_finding():
    h = _header(opening=1000.0, closing=1000.0)
    assert self_check(h, []) == []


def test_empty_rows_with_nonzero_printed_total_is_not_a_row_arithmetic_issue():
    # Nothing to reconcile row by row, so the rows are trivially consistent;
    # a printed total of 100 against a parsed sum of 0 is a reading mismatch.
    h = _header(opening=1000.0, closing=1000.0, withdrawals=100.0, deposits=0.0)
    assert _codes(self_check(h, [])) == [CODE]


def test_negative_balances_replay_correctly():
    # opening 100: -300 -> -200; +50 -> -150
    rows = _rows(100.0, [-300.0, 50.0])
    h = _header(opening=100.0, closing=-150.0, withdrawals=300.0, deposits=50.0)
    assert self_check(h, rows) == []


def test_negative_balances_with_total_mismatch_flagged():
    rows = _rows(100.0, [-300.0, 50.0])
    h = _header(opening=100.0, closing=-150.0, withdrawals=250.0, deposits=50.0)
    assert _codes(self_check(h, rows)) == [CODE]
