"""Task 4.1: tests for Rule A, ``check_duplicate_invoice_refs``.

Written before the function exists (task 4.2); expected to fail with
ImportError until then.

Pinned behaviour
----------------
* A group is ``dup_min_payments_same_month`` (default 2) or more credits with the same
  counterparty and reference IN ONE CALENDAR MONTH (leading zeros ignored; an undated
  credit counts as inside every month). Repeats spread over months are only a weak LOW
  note (``dup_cross_month_weight``, default 3) and never count toward the HIGH escalation.
  Fewer than the threshold below means MEDIUM per group. Debits are ignored; a different counterparty or a
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


def _pairs(n, *, start_row=1, gap=3, span=10):
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


# ---- same month vs across months (option 2) -----------------------------------------

XM_NOTE_TOKEN = "Weak signal"


def _pair(row, customer, ref, d1, d2, a1=900.0, a2=950.0):
    return [_tx(row, d1, customer, ref, a1), _tx(row + 1, d2, customer, ref, a2)]


def test_same_month_pair_is_medium_with_weight_10():
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 2, 20))          # 3 Apr and 21 Apr
    assert f.severity == Severity.MEDIUM and f.weight == 10.0


def test_cross_month_pair_is_one_weak_low_note():
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 27, 32))          # 28 Apr and 3 May
    assert f.code == "FL_DUP_INVOICE_REF"
    assert f.severity == Severity.LOW
    assert f.weight == 3.0 == CFG.dup_cross_month_weight
    assert f.field == "rows 1, 2"
    assert XM_NOTE_TOKEN in f.evidence
    assert INSTALMENT in f.evidence                              # close in time: likely instalment
    assert "likelier" not in f.evidence.lower()


def test_cross_month_repeats_are_one_finding_for_the_whole_file():
    txns = _pair(1, "A", "INV3000", 27, 32) + _pair(3, "B", "INV3001", 28, 33) \
        + _pair(5, "C", "INV3002", 29, 34) + _pair(7, "D", "INV3003", 25, 40)
    out = _check(txns)
    assert len(out) == 1 and out[0].severity == Severity.LOW and out[0].weight == 3.0
    assert out[0].field == "rows 1, 2, 3, 4, 5, 6, 7, 8"


def test_cross_month_repeats_never_count_toward_the_high_escalation():
    same = _pairs(2)                                              # two same-month groups
    spread = _pair(10, "X", "INV5000", 27, 32) + _pair(12, "Y", "INV5001", 28, 33) \
        + _pair(14, "Z", "INV5002", 29, 34)
    out = _check(same + spread)
    assert sorted(f.severity.value for f in out) == ["LOW", "MEDIUM", "MEDIUM"]
    assert not any(f.severity == Severity.HIGH for f in out)


def test_three_same_month_groups_escalate_and_a_spread_group_is_added_separately():
    out = _check(_pairs(3) + _pair(20, "X", "INV5000", 27, 32))
    assert [f.severity for f in out] == [Severity.HIGH, Severity.LOW]


def test_one_invoice_paid_in_two_months_is_not_a_same_month_group():
    # Apr 5, Apr 20 (same-month group) and May 3 (left over alone): one MEDIUM only.
    txns = [_tx(1, 4, "ALPHA", "INV3000", 900.0), _tx(2, 19, "ALPHA", "INV3000", 950.0),
            _tx(3, 32, "ALPHA", "INV3000", 700.0)]
    (f,) = _check(txns)
    assert f.severity == Severity.MEDIUM and f.field == "rows 1, 2"


def test_min_payments_threshold_of_three_demotes_a_pair_to_the_weak_note():
    cfg = FabricationConfig(dup_min_payments_same_month=3)
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 2, 20), cfg)
    assert f.severity == Severity.LOW
    three = [_tx(i, i * 3, "ALPHA", "INV3000", 100.0 * i) for i in (1, 2, 3)]
    (g,) = _check(three, cfg)
    assert g.severity == Severity.MEDIUM


def test_undated_credit_counts_as_inside_the_month():
    txns = [_tx(1, 2, "ALPHA", "INV3000", 900.0), _tx(2, None, "ALPHA", "INV3000", 950.0)]
    (f,) = _check(txns)
    assert f.severity == Severity.MEDIUM


def test_two_undated_credits_form_a_same_month_group():
    txns = [_tx(1, None, "ALPHA", "INV3000", 900.0), _tx(2, None, "ALPHA", "INV3000", 950.0)]
    (f,) = _check(txns)
    assert f.severity == Severity.MEDIUM


def test_cross_month_note_can_be_silenced_with_a_zero_weight_but_still_reports():
    cfg = FabricationConfig(dup_cross_month_weight=0.0)
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 27, 32), cfg)
    assert f.weight == 0.0 and f.severity == Severity.LOW


# ---- cross-month repeats are judged on distance -------------------------------------

def test_pair_months_apart_is_medium_with_the_far_apart_weight():
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 2, 150))          # 3 Apr and 1 Sep
    assert f.code == "FL_DUP_INVOICE_REF"
    assert f.severity == Severity.MEDIUM
    assert f.weight == 10.0 == CFG.dup_far_apart_weight
    assert f.field == "rows 1, 2"
    assert "too far apart" in f.evidence
    assert INSTALMENT not in f.evidence
    assert "likelier" not in f.evidence.lower()


def test_forty_five_days_is_still_weak_and_forty_six_is_medium():
    near = _check(_pair(1, "ALPHA", "INV3000", 0, 45))
    far = _check(_pair(1, "ALPHA", "INV3000", 0, 46))
    assert [f.severity for f in near] == [Severity.LOW]
    assert [f.severity for f in far] == [Severity.MEDIUM]


def test_four_payments_across_months_cannot_be_an_instalment():
    # Needs the same-month threshold of 3 so that 2 + 2 payments are not same-month groups.
    cfg = FabricationConfig(dup_min_payments_same_month=3)
    txns = [_tx(1, 19, "ALPHA", "INV3000", 100.0), _tx(2, 27, "ALPHA", "INV3000", 200.0),
            _tx(3, 31, "ALPHA", "INV3000", 300.0), _tx(4, 38, "ALPHA", "INV3000", 400.0)]
    assert len({t.date.month for t in txns}) == 2                   # Apr and May, 2 + 2
    out = _check(txns, cfg)
    assert [f.severity for f in out] == [Severity.MEDIUM]            # 19 days, but 4 payments
    assert "too many times" in out[0].evidence


def test_far_apart_groups_are_one_finding_and_do_not_escalate():
    far = _pair(1, "A", "INV3000", 2, 150) + _pair(3, "B", "INV3001", 3, 160) \
        + _pair(5, "C", "INV3002", 4, 170) + _pair(7, "D", "INV3003", 5, 180)
    out = _check(far)
    assert len(out) == 1
    assert out[0].severity == Severity.MEDIUM and out[0].weight == 10.0
    assert out[0].field == "rows 1, 2, 3, 4, 5, 6, 7, 8"


def test_far_apart_groups_do_not_count_toward_the_high_summary():
    same = _pairs(2)
    far = _pair(10, "X", "INV5000", 2, 150) + _pair(12, "Y", "INV5001", 3, 160) \
        + _pair(14, "Z", "INV5002", 4, 170)
    out = _check(same + far)
    assert not any(f.severity == Severity.HIGH for f in out)
    assert sorted((f.severity.value, f.weight) for f in out) == [
        ("MEDIUM", 10.0), ("MEDIUM", 10.0), ("MEDIUM", 10.0)]


def test_close_and_far_cross_month_groups_are_reported_separately():
    txns = _pair(1, "NEAR", "INV3000", 27, 32) + _pair(3, "FAR", "INV3001", 2, 150)
    out = _check(txns)
    by_sev = {f.severity: f for f in out}
    assert set(by_sev) == {Severity.LOW, Severity.MEDIUM}
    assert by_sev[Severity.LOW].field == "rows 1, 2" and by_sev[Severity.MEDIUM].field == "rows 3, 4"


def test_far_apart_weight_override():
    cfg = FabricationConfig(dup_far_apart_weight=14.0)
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 2, 150), cfg)
    assert f.weight == 14.0 and f.severity == Severity.MEDIUM


def test_distance_limit_follows_the_instalment_setting():
    cfg = FabricationConfig(instalment_max_days=20)
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 27, 52), cfg)        # 25 days apart
    assert f.severity == Severity.MEDIUM
    (g,) = _check(_pair(1, "ALPHA", "INV3000", 27, 32), cfg)        # 5 days apart
    assert g.severity == Severity.LOW


def test_same_month_repeat_is_unaffected_by_distance_rule():
    (f,) = _check(_pair(1, "ALPHA", "INV3000", 2, 28))               # same month, 26 days
    assert f.severity == Severity.MEDIUM and f.weight == 10.0
