"""Task 2.1: tests for FabricationConfig, DEFAULT_CONFIG and the reference reader.

These tests are written before ``fraud_detect/nodes/fabrication.py`` exists
(task 2.2) and are expected to fail with ModuleNotFoundError until then.

Pinned interface shapes
-----------------------
* ``extract_invoice_ref(description, cfg)`` returns ``None`` when nothing
  matches, otherwise a tuple ``(prefix_key, number)`` with the upper-cased
  prefix and the integer value of the number, e.g. ``("INV", 3277)``.
* ``CreditRef.ref`` is the normalised string ``prefix_key + str(number)``,
  e.g. ``"INV3277"``; ``CreditRef.ref_number`` is the integer ``3277``.
* ``normalise_counterparty(description, cfg)`` returns the description with the
  invoice reference and any leading rail prefix removed, upper-cased, with
  runs of spaces collapsed, e.g. ``"CINDERPATH HOLDINGS"``.
* ``extract_credit_refs(txns, cfg)`` returns ``list[CreditRef]`` in statement
  (input) order, skipping rows that are not credits with a reference.
"""
from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from fraud_detect.models import Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    CreditRef,
    FabricationConfig,
    extract_credit_refs,
    extract_invoice_ref,
    normalise_counterparty,
)

CFG = DEFAULT_CONFIG


def _txn(index, description, amount, d=None, page=1):
    return Transaction(index=index, date=d, description=description, amount=amount, page=page)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_default_config_values():
    c = DEFAULT_CONFIG
    assert isinstance(c, FabricationConfig)
    assert c.dup_escalate_min_refs == 3
    assert c.dup_isolated_weight == 10.0
    assert c.dup_escalated_weight == 22.0
    assert c.dup_min_payments_same_month == 2
    assert c.dup_cross_month_weight == 3.0
    assert c.dup_far_apart_weight == 10.0
    assert c.instalment_max_days == 45
    assert c.instalment_max_count == 3
    assert c.evidence_max_groups == 20
    assert c.seq_min_refs == 6
    assert c.seq_min_breaks == 2
    assert c.seq_weight == 5.0
    assert c.fixed_cost_keywords == ("CPF", "PAYROLL", "SALARY", "RENT")
    assert c.turnover_exclude_keywords == ("LOAN", "CAPITAL", "TRANSFER", "INTEREST")
    assert c.credit_jump_ratio == 1.2
    assert c.fixed_cost_stability_pct == 0.10
    assert c.turnover_min_months == 4
    assert c.turnover_min_stable_share == pytest.approx(2 / 3)
    assert c.turnover_min_jump_months == 2
    assert c.turnover_weight == 8.0
    assert c.invoice_patterns
    assert c.rail_prefixes


def test_default_config_equals_fresh_instance():
    assert DEFAULT_CONFIG == FabricationConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"credit_jump_ratio": 1.0},
        {"credit_jump_ratio": 0.5},
        {"fixed_cost_stability_pct": -0.01},
        {"fixed_cost_stability_pct": 1.0},
        {"fixed_cost_stability_pct": 1.5},
        {"dup_escalate_min_refs": 0},
        {"dup_min_payments_same_month": 1},
        {"dup_min_payments_same_month": 0},
        {"dup_cross_month_weight": -1.0},
        {"dup_far_apart_weight": -0.5},
        {"instalment_max_count": 0},
        {"seq_min_refs": 0},
        {"seq_min_breaks": 0},
        {"turnover_min_months": 0},
        {"turnover_min_jump_months": 0},
        {"evidence_max_groups": 0},
        {"dup_isolated_weight": -1.0},
        {"dup_escalated_weight": -0.1},
        {"seq_weight": -5.0},
        {"turnover_weight": -8.0},
        {"turnover_min_stable_share": 0.0},
        {"turnover_min_stable_share": 1.01},
        {"turnover_min_stable_share": -0.5},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        FabricationConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"credit_jump_ratio": 1.01},
        {"fixed_cost_stability_pct": 0.0},
        {"fixed_cost_stability_pct": 0.99},
        {"dup_escalate_min_refs": 1},
        {"dup_isolated_weight": 0.0},
        {"turnover_min_stable_share": 1.0},
        {"turnover_min_stable_share": 0.01},
    ],
)
def test_boundary_valid_config_constructs(kwargs):
    assert isinstance(FabricationConfig(**kwargs), FabricationConfig)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.credit_jump_ratio = 2.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# extract_invoice_ref
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "description",
    ["INV0042", "inv-042", "INV #042", "INV # 0042", "Inv 0042", "FAST FROM ACME PTE LTD INV0042"],
)
def test_same_reference_in_all_spellings(description):
    # Leading zeros removed, separators ignored, case ignored -> ("INV", 42).
    # NOTE: the default pattern in design.md is r"\b(INV)[\s\-#]*0*(\d{3,})\b",
    # which needs 3+ digits, so a bare "INV42" / "INV #42" (2 digits) is NOT a
    # reference under the design, while "INV12" must give none. Requirement 2.2
    # lists INV42; that conflict is reported to the orchestrator, so only the
    # zero-padded spellings are asserted here.
    assert extract_invoice_ref(description, CFG) == ("INV", 42)


def test_reference_returns_prefix_and_integer():
    assert extract_invoice_ref("FAST FROM CINDERPATH HOLDINGS INV3277", CFG) == ("INV", 3277)


@pytest.mark.parametrize("description", ["Customer Payment INV", "INV", "Payment for goods", ""])
def test_no_reference_when_no_digits(description):
    assert extract_invoice_ref(description, CFG) is None


def test_two_digit_reference_is_not_a_reference():
    assert extract_invoice_ref("INV12", CFG) is None


def test_reference_is_case_insensitive():
    assert extract_invoice_ref("acme inv3277", CFG) == extract_invoice_ref("ACME INV3277", CFG)
    assert extract_invoice_ref("acme inv3277", CFG) == ("INV", 3277)


def test_reference_after_500_chars_is_not_found():
    filler = "A" * 505
    assert extract_invoice_ref(f"{filler} INV3277", CFG) is None
    # Control: the same reference inside the first 500 characters is found.
    assert extract_invoice_ref(f"{'A' * 100} INV3277", CFG) == ("INV", 3277)


# --------------------------------------------------------------------------- #
# normalise_counterparty
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "description, expected",
    [
        ("FAST FROM CINDERPATH HOLDINGS INV3277", "CINDERPATH HOLDINGS"),
        ("PAYNOW FROM CINDERPATH HOLDINGS INV3277", "CINDERPATH HOLDINGS"),
        ("GIRO CINDERPATH HOLDINGS INV3277", "CINDERPATH HOLDINGS"),
        ("fast from cinderpath holdings inv3277", "CINDERPATH HOLDINGS"),
        ("CINDERPATH   HOLDINGS   INV3277", "CINDERPATH HOLDINGS"),
        ("CINDERPATH HOLDINGS INV0042", "CINDERPATH HOLDINGS"),
    ],
)
def test_counterparty_normalised(description, expected):
    assert normalise_counterparty(description, CFG) == expected


def test_counterparty_same_with_and_without_rail():
    a = normalise_counterparty("FAST FROM ACME TRADING INV0123", CFG)
    b = normalise_counterparty("acme  trading inv-123", CFG)
    assert a == b == "ACME TRADING"


# --------------------------------------------------------------------------- #
# extract_credit_refs
# --------------------------------------------------------------------------- #
def test_credit_ref_carries_row_page_date_amount():
    t = _txn(16, "FAST FROM CINDERPATH HOLDINGS INV3277", 6606.41, date(2026, 4, 13), page=2)
    refs = extract_credit_refs([t], CFG)
    assert refs == [
        CreditRef(
            row=16,
            page=2,
            date=date(2026, 4, 13),
            amount=6606.41,
            ref="INV3277",
            ref_number=3277,
            counterparty="CINDERPATH HOLDINGS",
        )
    ]


def test_extract_credit_refs_skips_unusable_rows_without_raising():
    txns = [
        _txn(1, "FAST FROM ACME INV0100", -500.0, date(2026, 1, 5)),   # debit
        _txn(2, None, 100.0, date(2026, 1, 6)),                        # no description
        _txn(3, "", 100.0, date(2026, 1, 7)),                          # empty description
        _txn(4, "FAST FROM ACME INV0101", 0.0, date(2026, 1, 8)),      # zero amount
        _txn(5, "FAST FROM ACME INV0102", None, date(2026, 1, 9)),     # no amount
        _txn(6, "Customer Payment INV", 100.0, date(2026, 1, 10)),     # no reference
        _txn(7, "FAST FROM ACME INV0103", 250.0, date(2026, 1, 11)),   # the only usable row
    ]
    refs = extract_credit_refs(txns, CFG)
    assert [r.row for r in refs] == [7]
    assert refs[0].ref == "INV103"


def test_extract_credit_refs_keeps_statement_order():
    txns = [
        _txn(5, "ACME INV0300", 10.0, date(2026, 3, 1)),
        _txn(2, "BETA INV0100", 10.0, date(2026, 1, 1)),
        _txn(9, "GAMMA INV0200", 10.0, date(2026, 2, 1)),
    ]
    assert [r.row for r in extract_credit_refs(txns, CFG)] == [5, 2, 9]


def test_extract_credit_refs_undated_row_is_kept_with_none_date():
    refs = extract_credit_refs([_txn(1, "ACME INV0300", 10.0, None)], CFG)
    assert len(refs) == 1 and refs[0].date is None


def test_extract_credit_refs_empty_input():
    assert extract_credit_refs([], CFG) == []
