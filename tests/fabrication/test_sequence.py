"""Task 3.1: tests for Rule B, ``find_order_breaks`` and ``check_invoice_sequence``.

Written before the functions exist (task 3.2); expected to fail with
ImportError until then.

Pinned behaviour
----------------
* Credits with a reference AND a date are sorted by (date, reference number),
  so same-day rows are always "in order".
* The longest non-decreasing run of reference numbers is kept; every other
  credit is an order-break row. Ties between equal-length longest runs are
  broken towards the EARLIEST rows: the kept run is the one whose sorted
  positions are lexicographically smallest, so later rows are the ones named.
* Rows without a date are left out (never placed, never raise, not counted
  towards ``seq_min_refs``).
* Fewer than ``seq_min_refs`` dated references, or fewer than
  ``seq_min_breaks`` break rows -> no finding. Otherwise exactly one LOW
  ``FL_INVOICE_SEQUENCE_BREAK`` finding, weight ``seq_weight``, ``field``
  ``rows a, b`` ascending (``row N`` for a single row), ``page`` of the latest
  named row, evidence naming every row with date, reference and amount.
* Rotation in ``move_references_out_of_order(txns, [r1, r2])`` swaps the two
  rows' invoice numbers. With at least three credits between r1 and r2 (by
  date) the unique longest run drops exactly r1 and r2, so those two rows are
  the expected breaks. With fewer in between, equal-length runs exist and the
  named rows are not the moved ones, so the tests avoid that case.
"""
from __future__ import annotations

from datetime import date, timedelta

from fraud_detect.models import Severity, Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    FabricationConfig,
    check_invoice_sequence,
    extract_credit_refs,
    find_order_breaks,
)

from .fixtures import business_file, credit_rows, move_references_out_of_order

CFG = DEFAULT_CONFIG
D0 = date(2026, 4, 1)


def _txns(spec, *, per_page=3):
    """spec: list of (day_offset or None, ref_number); row index = position + 1."""
    out = []
    for i, (off, ref) in enumerate(spec, start=1):
        out.append(Transaction(
            index=i,
            date=None if off is None else D0 + timedelta(days=off),
            description=f"FAST FROM CUSTOMER{i:02d} INV{ref}",
            amount=1000.0 + i,
            page=(i - 1) // per_page + 1,
        ))
    return out


def _breaks(txns, cfg=CFG):
    return find_order_breaks(extract_credit_refs(txns, cfg), cfg)


IN_ORDER = [(i, 101 + i) for i in range(8)]


# ---- find_order_breaks -----------------------------------------------------

def test_in_order_list_has_no_breaks():
    assert _breaks(_txns(IN_ORDER)) == set()


def test_two_moved_rows_are_named_exactly_on_business_file():
    txns, _ = business_file()
    credits = credit_rows(txns)
    r1, r2 = credits[2].index, credits[8].index   # >= 3 credits between them
    moved = move_references_out_of_order(txns, [r1, r2])
    assert _breaks(moved) == {r1, r2}


def test_business_file_unchanged_is_silent():
    txns, _ = business_file()
    assert _breaks(txns) == set()
    assert check_invoice_sequence(txns, CFG) == []


def test_rows_outside_longest_run_are_breaks():
    # by date: 101 102 108 103 104 105 106 -> the 108 (row 3) is the only break
    spec = [(0, 101), (1, 102), (2, 108), (3, 103), (4, 104), (5, 105), (6, 106)]
    assert _breaks(_txns(spec)) == {3}
    # two displaced: 101 109 103 104 110 105 106 -> rows 2 and 5 break
    spec = [(0, 101), (1, 109), (2, 103), (3, 104), (4, 110), (5, 105), (6, 106)]
    assert _breaks(_txns(spec)) == {2, 5}


def test_same_day_rows_in_any_order_are_not_breaks():
    spec = [(0, 105), (0, 103), (0, 104), (1, 108), (1, 106), (1, 107), (2, 109)]
    assert _breaks(_txns(spec)) == set()


def test_tie_between_equal_longest_runs_keeps_earliest_rows():
    # 104 105 106 | 101 102 103 : two runs of length 3; the earliest rows
    # (1-3) are kept, so rows 4-6 are the breaks.
    spec = [(0, 104), (1, 105), (2, 106), (3, 101), (4, 102), (5, 103)]
    assert _breaks(_txns(spec)) == {4, 5, 6}


def test_row_without_date_is_left_out_and_never_raises():
    spec = [(0, 101), (1, 102), (None, 999), (2, 103), (3, 104), (4, 105), (5, 106)]
    assert _breaks(_txns(spec)) == set()
    # an undated row is not named even when its number is far out of line
    spec = [(0, 101), (1, 102), (None, 100), (2, 103), (3, 110), (4, 104), (5, 105), (6, 106)]
    assert _breaks(_txns(spec)) == {5}


def test_debits_and_rows_without_reference_are_ignored():
    txns = _txns(IN_ORDER)
    txns[3] = txns[3].model_copy(update={"amount": -txns[3].amount})   # debit
    txns[4] = txns[4].model_copy(update={"description": "FAST FROM PLAIN CUSTOMER"})
    assert _breaks(txns) == set()


def test_empty_input_gives_no_breaks():
    assert find_order_breaks([], CFG) == set()


# ---- check_invoice_sequence: silence --------------------------------------

def test_in_order_list_is_silent():
    assert check_invoice_sequence(_txns(IN_ORDER), CFG) == []


def test_single_moved_row_is_silent_by_default():
    spec = [(0, 101), (1, 102), (2, 108), (3, 103), (4, 104), (5, 105), (6, 106)]
    assert check_invoice_sequence(_txns(spec), CFG) == []


def test_same_day_rows_in_any_order_are_silent():
    spec = [(0, 106), (0, 101), (0, 105), (0, 102), (0, 104), (0, 103), (0, 107)]
    assert check_invoice_sequence(_txns(spec), CFG) == []


def test_fewer_than_min_refs_is_silent_even_with_breaks():
    # 5 dated references with two clear breaks: below seq_min_refs (6)
    spec = [(0, 105), (1, 101), (2, 106), (3, 102), (4, 107)]
    assert CFG.seq_min_refs == 6
    assert _breaks(_txns(spec)) != set()          # the order really is broken
    assert check_invoice_sequence(_txns(spec), CFG) == []


def test_undated_rows_do_not_count_towards_min_refs():
    spec = [(0, 105), (1, 101), (2, 106), (3, 102), (4, 107), (None, 103)]
    assert check_invoice_sequence(_txns(spec), CFG) == []


def test_no_transactions_is_silent():
    assert check_invoice_sequence([], CFG) == []


# ---- check_invoice_sequence: finding shape --------------------------------

TWO_BREAKS = [(0, 101), (1, 109), (2, 103), (3, 104), (4, 110), (5, 105), (6, 106)]


def test_finding_shape_for_two_breaks():
    txns = _txns(TWO_BREAKS)        # break rows 2 and 5; pages 1 and 2 (3 per page)
    findings = check_invoice_sequence(txns, CFG)
    assert len(findings) == 1
    f = findings[0]
    assert f.agent == "Node3:FinancialLogic"
    assert f.code == "FL_INVOICE_SEQUENCE_BREAK"
    assert f.severity == Severity.LOW
    assert f.weight == 5
    assert f.field == "rows 2, 5"
    assert f.page == txns[4].page == 2


def test_field_is_ascending_and_page_is_latest_row_page():
    # the latest named row is NOT on the highest page overall
    spec = [(0, 101), (1, 109), (2, 103), (3, 104), (4, 110), (5, 105), (6, 106),
            (7, 107), (8, 108)]
    txns = _txns(spec, per_page=2)
    f = check_invoice_sequence(txns, CFG)[0]
    assert f.field == "rows 2, 5"
    assert f.page == 3            # row 5 -> page 3 (rows 5-6)
    assert max(t.page for t in txns) == 5


def test_evidence_names_rows_with_date_reference_and_amount_and_is_a_hint():
    txns = _txns(TWO_BREAKS)
    ev = check_invoice_sequence(txns, CFG)[0].evidence
    for t, ref in ((txns[1], "INV109"), (txns[4], "INV110")):
        assert t.date.strftime("%d %b %Y") in ev
        assert ref in ev
        assert f"{t.amount:,.2f}" in ev
        assert f"row {t.index}" in ev
    low = ev.lower()
    assert "supporting" in low
    assert "late" in low or "out of order" in low


def test_rotated_business_file_gives_one_finding_naming_moved_rows():
    txns, _ = business_file()
    credits = credit_rows(txns)
    r1, r2 = credits[2].index, credits[8].index
    f = check_invoice_sequence(move_references_out_of_order(txns, [r1, r2]), CFG)
    assert len(f) == 1
    assert f[0].field == f"rows {min(r1, r2)}, {max(r1, r2)}"
    assert f[0].code == "FL_INVOICE_SEQUENCE_BREAK"


def test_running_twice_gives_identical_findings():
    txns = _txns([(0, 104), (1, 105), (2, 106), (3, 101), (4, 102), (5, 103)])
    first = check_invoice_sequence(txns, CFG)
    second = check_invoice_sequence(txns, CFG)
    assert first == second
    assert len(first) == 1 and first[0].field == "rows 4, 5, 6"


def test_input_is_not_mutated():
    txns = _txns(TWO_BREAKS)
    before = [t.model_copy() for t in txns]
    check_invoice_sequence(txns, CFG)
    assert txns == before


def test_debits_with_out_of_order_references_are_ignored():
    txns = _txns(IN_ORDER)
    extra = Transaction(index=99, date=D0 + timedelta(days=3),
                        description="FAST TO SUPPLIER INV900", amount=-50.0, page=1)
    assert check_invoice_sequence(txns + [extra], CFG) == []


# ---- custom configuration ---------------------------------------------------

def test_min_breaks_one_reports_a_single_break_as_one_row():
    spec = [(0, 101), (1, 102), (2, 108), (3, 103), (4, 104), (5, 105), (6, 106)]
    cfg = FabricationConfig(seq_min_breaks=1)
    findings = check_invoice_sequence(_txns(spec), cfg)
    assert len(findings) == 1
    assert findings[0].field == "row 3"
    assert findings[0].page == 1


def test_seq_weight_override_changes_weight():
    cfg = FabricationConfig(seq_weight=9.0)
    f = check_invoice_sequence(_txns(TWO_BREAKS), cfg)
    assert len(f) == 1 and f[0].weight == 9.0
    assert f[0].severity == Severity.LOW


def test_seq_min_refs_override_lets_short_lists_report():
    spec = [(0, 105), (1, 101), (2, 106), (3, 102), (4, 107)]
    cfg = FabricationConfig(seq_min_refs=5)
    assert len(check_invoice_sequence(_txns(spec), cfg)) == 1
