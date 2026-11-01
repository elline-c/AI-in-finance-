"""Task 6.1: tests for the direction resolver (Requirement 6, design Component 5).

API under test (``fraud_detect.nodes.ingest_parse.direction``):

    resolve_direction(row, layout, prev_balance) -> Resolved | None

    Resolved(amount: float,                       # signed: credits +, debits -
             method: 'column' | 'printed_sign' | 'balance_change',
             agrees_with_balance: bool | None,    # None when no previous balance
             finding_code: str | None = None)     # 'DQ_DIRECTION_UNCERTAIN' or None

Reporting convention:
  * ``None`` returned  -> no direction could be determined; the caller emits
    ``DQ_UNREADABLE_ROW`` for the row and leaves it out of the transactions.
  * ``Resolved.finding_code == 'DQ_DIRECTION_UNCERTAIN'`` -> the row is kept
    (with the direction stated by the rules) and the caller emits that finding.

Table-reader conventions: ``AmountCell.money.value`` is the printed MAGNITUDE;
the printed sign lives in ``money.explicit_sign`` / ``money.suffix``;
``RawRow.balance`` is a signed Money.

These tests are expected to fail with ModuleNotFoundError until task 6.2.
"""
from __future__ import annotations

import pytest

from fraud_detect.nodes.ingest_parse.money import Money
from fraud_detect.nodes.ingest_parse.table import AmountCell, RawRow, TableLayout
from fraud_detect.nodes.ingest_parse.direction import resolve_direction

UNCERTAIN = "DQ_DIRECTION_UNCERTAIN"


# ---------------------------------------------------------------- helpers
def _money(mag, sign=None, suffix=None):
    return Money(value=mag, explicit_sign=sign, suffix=suffix)


def _cell(mag, column=None, sign=None):
    return AmountCell(money=_money(mag, sign), x0=0.0, x1=1.0, column=column)


def _row(cells, balance, suffix=None):
    return RawRow(
        date_text="01/01/2025",
        value_date_text=None,
        description="TEST",
        amount_cells=cells,
        balance=_money(balance, suffix=suffix),
        line_text="",
    )


def _layout(kind):
    return TableLayout(header_y=0.0, columns={}, kind=kind)


TWO = _layout("two_column")
SINGLE = _layout("single_signed")
UNKNOWN = _layout("unknown")


# ---------------------------------------------------------------- column
def test_column_money_out_is_negative():
    r = resolve_direction(_row([_cell(50.0, "money_out")], 950.0), TWO, 1000.0)
    assert r.method == "column"
    assert r.amount == -50.0
    assert r.agrees_with_balance is True
    assert r.finding_code is None


def test_column_money_in_is_positive():
    r = resolve_direction(_row([_cell(75.5, "money_in")], 1075.5), TWO, 1000.0)
    assert r.method == "column"
    assert r.amount == 75.5
    assert r.agrees_with_balance is True
    assert r.finding_code is None


def test_column_conflicting_balance_keeps_column_direction_and_flags():
    # column says debit, balance rose
    r = resolve_direction(_row([_cell(50.0, "money_out")], 1050.0), TWO, 1000.0)
    assert r.method == "column"
    assert r.amount == -50.0
    assert r.agrees_with_balance is False
    assert r.finding_code == UNCERTAIN


def test_column_credit_but_balance_fell_keeps_credit_and_flags():
    r = resolve_direction(_row([_cell(50.0, "money_in")], 950.0), TWO, 1000.0)
    assert r.amount == 50.0
    assert r.finding_code == UNCERTAIN


def test_column_magnitude_is_printed_not_recomputed_from_balance():
    # edited amount: printed 50 but balance only moved by 20
    r = resolve_direction(_row([_cell(50.0, "money_out")], 980.0), TWO, 1000.0)
    assert r.amount == -50.0
    assert r.method == "column"


def test_column_first_row_without_prev_balance():
    r = resolve_direction(_row([_cell(50.0, "money_out")], 950.0), TWO, None)
    assert r.method == "column"
    assert r.amount == -50.0
    assert r.agrees_with_balance is None
    assert r.finding_code is None


# ---------------------------------------------------------------- printed sign
def test_printed_minus_is_debit():
    r = resolve_direction(_row([_cell(13.47, "amount", sign=-1)], 986.53), SINGLE, 1000.0)
    assert r.method == "printed_sign"
    assert r.amount == -13.47
    assert r.agrees_with_balance is True
    assert r.finding_code is None


def test_printed_plus_is_credit():
    r = resolve_direction(_row([_cell(11.08, "amount", sign=1)], 1011.08), SINGLE, 1000.0)
    assert r.method == "printed_sign"
    assert r.amount == 11.08
    assert r.finding_code is None


def test_printed_sign_conflicting_balance_keeps_sign_and_flags():
    r = resolve_direction(_row([_cell(13.47, "amount", sign=-1)], 1013.47), SINGLE, 1000.0)
    assert r.method == "printed_sign"
    assert r.amount == -13.47
    assert r.agrees_with_balance is False
    assert r.finding_code == UNCERTAIN


def test_printed_sign_first_row_without_prev_balance():
    r = resolve_direction(_row([_cell(13.47, "amount", sign=-1)], 986.53), SINGLE, None)
    assert r.method == "printed_sign"
    assert r.amount == -13.47
    assert r.agrees_with_balance is None
    assert r.finding_code is None


def test_unsigned_amount_in_single_signed_table_is_read_as_credit():
    r = resolve_direction(_row([_cell(20.0, "amount")], 1020.0), SINGLE, 1000.0)
    assert r is not None
    assert r.amount == 20.0


# ---------------------------------------------------------------- balance change
def test_balance_change_debit():
    r = resolve_direction(_row([_cell(40.0)], 960.0), UNKNOWN, 1000.0)
    assert r.method == "balance_change"
    assert r.amount == -40.0
    assert r.agrees_with_balance is True
    assert r.finding_code is None


def test_balance_change_credit():
    r = resolve_direction(_row([_cell(40.0)], 1040.0), UNKNOWN, 1000.0)
    assert r.method == "balance_change"
    assert r.amount == 40.0
    assert r.finding_code is None


def test_balance_change_within_tolerance_is_not_flagged():
    r = resolve_direction(_row([_cell(40.0)], 1039.99), UNKNOWN, 1000.0)
    assert r.amount == 40.0
    assert r.finding_code is None


def test_balance_change_mismatch_keeps_direction_prints_amount_and_flags():
    # balance fell by 40 but the printed amount is 45
    r = resolve_direction(_row([_cell(45.0)], 960.0), UNKNOWN, 1000.0)
    assert r.method == "balance_change"
    assert r.amount == -45.0  # magnitude is the printed amount
    assert r.finding_code == UNCERTAIN


def test_balance_change_mismatch_on_credit_side():
    r = resolve_direction(_row([_cell(45.0)], 1040.0), UNKNOWN, 1000.0)
    assert r.amount == 45.0
    assert r.finding_code == UNCERTAIN


# ---------------------------------------------------------------- no evidence
def test_no_column_no_sign_no_prev_balance_is_unreadable():
    assert resolve_direction(_row([_cell(40.0)], 960.0), UNKNOWN, None) is None


def test_no_amount_cells_is_unreadable():
    assert resolve_direction(_row([], 960.0), TWO, 1000.0) is None


def test_no_column_no_sign_and_unchanged_balance_is_unreadable():
    assert resolve_direction(_row([_cell(40.0)], 1000.0), UNKNOWN, 1000.0) is None


# ---------------------------------------------------------------- CR/DR balances
@pytest.mark.parametrize("suffix", [None, "CR"])
def test_cr_suffix_on_balance_does_not_break(suffix):
    r = resolve_direction(_row([_cell(50.0, "money_out")], 950.0, suffix=suffix), TWO, 1000.0)
    assert r.amount == -50.0
    assert r.agrees_with_balance is True


def test_cr_suffix_balance_change_method():
    r = resolve_direction(_row([_cell(40.0)], 1040.0, suffix="CR"), UNKNOWN, 1000.0)
    assert r.method == "balance_change"
    assert r.amount == 40.0
