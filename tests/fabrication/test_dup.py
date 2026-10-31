"""Task 4.1: tests for Rule A, ``check_duplicate_invoice_refs``.

Written before the function exists (task 4.2); expected to fail with
ImportError until then.

Pinned behaviour
----------------
* A group is two or more credits with the same counterparty and reference
  (leading zeros ignored). Debits are ignored; a different counterparty or a
  different reference makes a different key.
* Fewer than ``dup_escalate_min_refs`` groups -> one MEDIUM finding per group
  (weight ``dup_isolated_weight``). At or above it -> exactly one HIGH
  finding (weight ``dup_escalated_weight``) listing every group.
* Evidence per occurrence: date as ``13 Apr 2026`` (``no date`` when missing),
  amount as ``6,606.41``, ``page P`` and ``row N``. Evidence says it cannot tell
  which row is the copy.
* ``field`` is ``rows a, b`` ascending, unique (rows of all groups when
  escalated); ``page`` is the page of the latest named row (highest row).
* ``likely_copies`` is a hint only: exactly one group row in the set adds
  "row N breaks the invoice number order, so it is the likelier copy".
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    FabricationConfig,
    check_duplicate_invoice_refs,
    extract_credit_refs,
    find_order_breaks,
)

from .fixtures import business_file, credit_rows, insert_copy

CFG = DEFAULT_CONFIG
D0 = date(2026, 4, 1)
INSTALMENT = "may be an instalment payment; check the invoice"
IDENTICAL = "identical amount, possible duplicate posting"


def _fmt(d):
    return d.strftime("%d %b %Y")


def _tx(row, day, customer, ref, amount, page=None, *, credit=True, desc=None):
    """day: offset from D0, or None for an undated row."""
    return Transaction(
        index=row,
        date=None if day is None else D0 + timedelta(days=day),
        description=desc or f"FAST FROM {customer} {ref}",
        amount=amount if credit else -amount,
        page=page if page is not None else (row - 1) // 10 + 1,
    )


def _pairs(n, *, start_row=1, gap=5, span=10):
    """n groups of two credits each: distinct customers/refs, different amounts."""
    out = []
    row = start_row
    for g in range(n):
        base = g * gap
        out.append(_tx(row, base, f"CUSTOMER{g:02d}", f"INV{1000 + g}", 1000.0 + g))
        out.append(_tx(row + 1, base + span, f"CUSTOMER{g:02d}", f"INV{1000 + g}", 2000.0 + g))
        row += 2
    return out


def _check(txns, cfg=CFG, likely=frozenset()):
    return check_duplicate_invoice_refs(txns, cfg, likely)


# ---- silence ----------------------------------------------------------------

def test_unique_references_are_silent():
    txns = [_tx(i, i, f"CUSTOMER{i:02d}", f"INV{2000 + i}", 500.0 + i) for i in range(1, 9)]
    assert _check(txns) == []


def test_unchanged_business_file_is_silent():
    txns, _ = business_file()
    assert _check(txns) == []


def test_no_references_returns_empty():
    txns = [
        Transaction(index=1, date=D0, description="FAST FROM ACME", amount=100.0, page=1),
        Transaction(index=2, date=D0, description="Customer Payment INV", amount=100.0, page=1),
        Transaction(index=3, date=D0, description="RENT", amount=-50.0, page=1),
    ]
    assert _check(txns) == []
    assert _check([]) == []


def test_same_reference_under_two_customers_is_silent():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 900.0), _tx(2, 3, "BETA TRADING", "INV3000", 900.0)]
    assert _check(txns) == []


def test_same_customer_different_references_is_silent():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 900.0), _tx(2, 3, "ALPHA TRADING", "INV3001", 900.0)]
    assert _check(txns) == []


def test_debits_are_ignored():
    txns = [
        _tx(1, 0, "ALPHA TRADING", "INV3000", 900.0),
        _tx(2, 3, "ALPHA TRADING", "INV3000", 900.0, credit=False),
    ]
    assert _check(txns) == []


# ---- one repeated reference ---------------------------------------------------

def test_one_group_gives_one_medium_finding_with_pinned_fields():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 6606.41, page=1),
            _tx(2, 4, "ALPHA TRADING", "INV3000", 8318.10, page=1)]
    out = _check(txns)
    assert len(out) == 1
    f = out[0]
    assert f.code == "FL_DUP_INVOICE_REF"
    assert f.agent == "Node3:FinancialLogic"
    assert f.severity == Severity.MEDIUM
    assert f.weight == 10.0 == CFG.dup_isolated_weight
    assert INSTALMENT in f.evidence
    assert IDENTICAL not in f.evidence


def test_evidence_lists_date_amount_page_row_for_every_occurrence():
    txns = [
        _tx(3, 0, "ALPHA TRADING", "INV3000", 6606.41, page=1),
        _tx(14, 10, "ALPHA TRADING", "INV3000", 8318.10, page=2),
        _tx(27, 20, "ALPHA TRADING", "INV3000", 1234.56, page=3),
    ]
    (f,) = _check(txns)
    for row, day, amount, page in [(3, 0, "6,606.41", 1), (14, 10, "8,318.10", 2), (27, 20, "1,234.56", 3)]:
        assert _fmt(D0 + timedelta(days=day)) in f.evidence
        assert amount in f.evidence
        assert f"page {page}" in f.evidence
        assert f"row {row}" in f.evidence
    assert f.field == "rows 3, 14, 27"
    assert f.page == 3


def test_field_is_ascending_even_when_input_order_differs():
    txns = [_tx(9, 5, "ALPHA TRADING", "INV3000", 700.0, page=1),
            _tx(4, 1, "ALPHA TRADING", "INV3000", 800.0, page=2)]
    (f,) = _check(txns)
    assert f.field == "rows 4, 9"
    assert f.page == 1  # page of the latest named row (row 9)


def test_evidence_says_it_cannot_tell_which_row_is_the_copy():
    (f,) = _check(_pairs(1))
    assert "cannot tell which row" in f.evidence.lower()


# ---- instalment sentence ------------------------------------------------------

def test_group_of_three_within_45_days_keeps_instalment_sentence():
    txns = [_tx(i, i * 22, "ALPHA TRADING", "INV3000", 100.0 * i) for i in (1, 2, 3)]  # span 44
    (f,) = _check(txns)
    assert INSTALMENT in f.evidence


def test_group_of_four_leaves_instalment_sentence_out():
    txns = [_tx(i, i, "ALPHA TRADING", "INV3000", 100.0 * i) for i in (1, 2, 3, 4)]
    (f,) = _check(txns)
    assert INSTALMENT not in f.evidence


def test_span_of_exactly_45_days_keeps_sentence_and_46_removes_it():
    near = [_tx(1, 0, "ALPHA TRADING", "INV3000", 100.0), _tx(2, 45, "ALPHA TRADING", "INV3000", 200.0)]
    far = [_tx(1, 0, "ALPHA TRADING", "INV3000", 100.0), _tx(2, 46, "ALPHA TRADING", "INV3000", 200.0)]
    assert INSTALMENT in _check(near)[0].evidence
    assert INSTALMENT not in _check(far)[0].evidence


def test_span_over_45_days_leaves_instalment_sentence_out():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 100.0), _tx(2, 90, "ALPHA TRADING", "INV3000", 200.0)]
    (f,) = _check(txns)
    assert INSTALMENT not in f.evidence


def test_identical_amounts_say_possible_duplicate_posting():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 750.0), _tx(2, 3, "ALPHA TRADING", "INV3000", 750.0)]
    (f,) = _check(txns)
    assert IDENTICAL in f.evidence


# ---- escalation ------------------------------------------------------------------

def test_fewer_than_three_groups_give_one_medium_each():
    out = _check(_pairs(2))
    assert len(out) == 2
    assert all(f.severity == Severity.MEDIUM and f.weight == 10.0 for f in out)
    assert {f.field for f in out} == {"rows 1, 2", "rows 3, 4"}


def test_three_or_more_groups_give_exactly_one_high_listing_every_group():
    txns = _pairs(4)
    out = _check(txns)
    assert len(out) == 1
    f = out[0]
    assert f.code == "FL_DUP_INVOICE_REF"
    assert f.agent == "Node3:FinancialLogic"
    assert f.severity == Severity.HIGH
    assert f.weight == 22.0 == CFG.dup_escalated_weight
    for g in range(4):
        assert f"CUSTOMER{g:02d}" in f.evidence
        assert f"INV{1000 + g}" in f.evidence
    for row in range(1, 9):
        assert f"row {row}" in f.evidence
    assert f.field == "rows 1, 2, 3, 4, 5, 6, 7, 8"
    assert f.page == txns[-1].page


def test_escalated_field_is_unique_and_ascending_across_groups():
    # interleaved rows: group A rows 1,5; group B rows 2,6; group C rows 3,7
    txns = []
    for g, (r1, r2) in enumerate([(1, 5), (2, 6), (3, 7)]):
        txns.append(_tx(r1, g, f"CUSTOMER{g}", f"INV{1000 + g}", 100.0 + g, page=1))
        txns.append(_tx(r2, 20 + g, f"CUSTOMER{g}", f"INV{1000 + g}", 200.0 + g, page=2))
    (f,) = _check(sorted(txns, key=lambda t: -t.index))
    assert f.field == "rows 1, 2, 3, 5, 6, 7"
    assert f.page == 2


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_boundary_follows_dup_escalate_min_refs(k):
    for threshold in (1, 5):
        cfg = FabricationConfig(dup_escalate_min_refs=threshold)
        out = _check(_pairs(k), cfg)
        if k < threshold:
            assert len(out) == k
            assert all(f.severity == Severity.MEDIUM and f.weight == 10.0 for f in out)
        else:
            assert len(out) == 1
            assert out[0].severity == Severity.HIGH and out[0].weight == 22.0


def test_changing_threshold_moves_only_the_boundary():
    txns = _pairs(3)
    default = _check(txns)
    assert len(default) == 1 and default[0].severity == Severity.HIGH
    raised = _check(txns, FabricationConfig(dup_escalate_min_refs=4))
    assert len(raised) == 3
    assert [f.field for f in raised] == ["rows 1, 2", "rows 3, 4", "rows 5, 6"]
    assert all(f.code == "FL_DUP_INVOICE_REF" for f in raised)


# ---- evidence cap -----------------------------------------------------------------

def test_evidence_max_groups_cuts_list_and_ends_with_and_n_more():
    cfg = FabricationConfig(evidence_max_groups=2)
    (f,) = _check(_pairs(5), cfg)
    assert f.severity == Severity.HIGH
    assert "CUSTOMER00" in f.evidence and "CUSTOMER01" in f.evidence
    for g in (2, 3, 4):
        assert f"CUSTOMER{g:02d}" not in f.evidence
    assert f.evidence.rstrip().rstrip(".").endswith("and 3 more")


def test_no_and_more_suffix_when_all_groups_fit():
    (f,) = _check(_pairs(3))
    assert " more" not in f.evidence


# ---- likelier copy -----------------------------------------------------------------

def test_exactly_one_row_in_likely_copies_is_named_as_hint():
    txns = [_tx(5, 0, "ALPHA TRADING", "INV3000", 900.0), _tx(8, 9, "ALPHA TRADING", "INV3000", 950.0)]
    (f,) = _check(txns, likely={8})
    assert "row 8 breaks the invoice number order, so it is the likelier copy" in f.evidence
    low = f.evidence.lower()
    assert "likelier" in low
    assert "is the copy" not in low
    assert "forged" not in low


def test_both_or_neither_row_in_likely_copies_names_no_copy():
    txns = [_tx(5, 0, "ALPHA TRADING", "INV3000", 900.0), _tx(8, 9, "ALPHA TRADING", "INV3000", 950.0)]
    for likely in (set(), {5, 8}, {99}):
        (f,) = _check(txns, likely=likely)
        assert "likelier" not in f.evidence.lower()


def test_likely_copy_is_named_per_group_only():
    txns = [
        _tx(1, 0, "ALPHA", "INV3000", 900.0), _tx(2, 4, "ALPHA", "INV3000", 950.0),
        _tx(3, 8, "BETA", "INV3001", 900.0), _tx(4, 12, "BETA", "INV3001", 950.0),
    ]
    out = _check(txns, likely={2, 3, 4})
    by_field = {f.field: f for f in out}
    assert "row 2 breaks the invoice number order" in by_field["rows 1, 2"].evidence
    assert "likelier" not in by_field["rows 3, 4"].evidence.lower()


def test_hint_works_with_real_order_breaks_from_the_business_file():
    txns, _ = business_file()
    src = credit_rows(txns)[10]
    changed = insert_copy(txns, src.index, src.date + timedelta(days=3), src.amount + 111.11)
    refs = extract_credit_refs(changed, CFG)
    breaks = find_order_breaks(refs, CFG)
    (f,) = _check(changed, likely=breaks)
    assert f.severity == Severity.MEDIUM
    assert "cannot tell which row" in f.evidence.lower()
    named = [r for r in breaks if f"row {r} breaks the invoice number order" in f.evidence]
    in_group = [int(x) for x in f.field.removeprefix("rows ").split(", ")]
    assert len(set(in_group) & breaks) == len(named)


# ---- normalisation, undated rows -------------------------------------------------------

def test_leading_zero_variants_are_the_same_reference():
    txns = [
        _tx(1, 0, "ALPHA TRADING", "INV0042", 900.0, desc="FAST FROM ALPHA TRADING INV0042"),
        _tx(2, 4, "ALPHA TRADING", "INV042", 950.0, desc="FAST FROM ALPHA TRADING INV042"),
    ]
    (f,) = _check(txns)
    assert f.field == "rows 1, 2"


def test_undated_row_is_still_grouped_and_shown_as_no_date():
    txns = [_tx(1, 0, "ALPHA TRADING", "INV3000", 900.0, page=1),
            _tx(2, None, "ALPHA TRADING", "INV3000", 950.0, page=1)]
    (f,) = _check(txns)
    assert f.field == "rows 1, 2"
    assert "no date" in f.evidence
    assert _fmt(D0) in f.evidence


# ---- determinism and purity -------------------------------------------------------------

def test_deterministic_and_input_not_mutated():
    txns = _pairs(4) + [_tx(20, 0, "OTHER", "INV9000", 10.0)]
    before = copy.deepcopy(txns)
    likely = {2, 4}
    first = _check(txns, likely=likely)
    second = _check(txns, likely=likely)
    assert [f.model_dump() for f in first] == [f.model_dump() for f in second]
    assert txns == before
    assert likely == {2, 4}
