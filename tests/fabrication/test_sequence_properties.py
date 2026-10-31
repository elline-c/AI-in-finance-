"""Task 3.3: property tests for Rule B (invoice sequence break).

Feature: node3-fabrication-rules, Property 6: Sequence break detection
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""
from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given, settings, strategies as st

from fraud_detect.models import Transaction
from fraud_detect.nodes.fabrication import (
    DEFAULT_CONFIG,
    check_invoice_sequence,
    extract_credit_refs,
    find_order_breaks,
)

CFG = DEFAULT_CONFIG
D0 = date(2026, 1, 1)
PBT = settings(max_examples=150, deadline=None)


def _txn(index, offset, ref, amount=None):
    return Transaction(
        index=index,
        date=D0 + timedelta(days=offset),
        description=f"FAST FROM CUSTOMER{index:03d} INV{ref}",
        amount=amount if amount is not None else 1000.0 + index,
        page=(index - 1) // 4 + 1,
    )


@st.composite
def in_order_credits(draw, *, allow_same_day):
    """(offsets, refs): refs rise with date; equal refs only on the same day."""
    n = draw(st.integers(min_value=CFG.seq_min_refs, max_value=16))
    min_gap = 0 if allow_same_day else 1
    gaps = draw(st.lists(st.integers(min_gap, 5), min_size=n - 1, max_size=n - 1))
    offsets, refs = [0], [draw(st.integers(100, 500))]
    for g in gaps:
        offsets.append(offsets[-1] + g)
        step_min = 0 if g == 0 else 1
        refs.append(refs[-1] + draw(st.integers(step_min, 6)))
    return offsets, refs


def _build(offsets, refs):
    return [_txn(i, o, r) for i, (o, r) in enumerate(zip(offsets, refs), start=1)]


@st.composite
def rotated_case(draw):
    """Strictly rising refs on distinct days with two rows' refs swapped."""
    offsets, refs = draw(in_order_credits(allow_same_day=False))
    n = len(refs)
    i = draw(st.integers(0, n - 5))
    j = draw(st.integers(i + 4, n - 1))          # >= 3 credits between
    swapped = list(refs)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    if len(set(refs)) != len(refs):              # refs strictly rise here
        raise AssertionError("generator must give distinct refs")
    return offsets, swapped, i + 1, j + 1


# Feature: node3-fabrication-rules, Property 6: Sequence break detection
@PBT
@given(case=in_order_credits(allow_same_day=True), data=st.data())
def test_in_order_credits_give_no_finding_even_when_same_day_rows_shuffled(case, data):
    txns = _build(*case)
    txns = data.draw(st.permutations(txns))
    assert find_order_breaks(extract_credit_refs(txns, CFG), CFG) == set()
    assert check_invoice_sequence(txns, CFG) == []


# Feature: node3-fabrication-rules, Property 6: Sequence break detection
@PBT
@given(case=rotated_case(), data=st.data())
def test_swapped_rows_are_named_exactly(case, data):
    offsets, refs, r1, r2 = case
    txns = data.draw(st.permutations(_build(offsets, refs)))
    assert find_order_breaks(extract_credit_refs(txns, CFG), CFG) == {r1, r2}
    findings = check_invoice_sequence(txns, CFG)
    assert len(findings) == 1
    f = findings[0]
    assert f.code == "FL_INVOICE_SEQUENCE_BREAK"
    assert f.field == f"rows {min(r1, r2)}, {max(r1, r2)}"


# Feature: node3-fabrication-rules, Property 6: Sequence break detection
@PBT
@given(
    spec=st.lists(st.tuples(st.integers(0, 12), st.integers(100, 120)),
                  min_size=0, max_size=20),
    data=st.data(),
)
def test_break_set_is_independent_of_input_order_and_deterministic(spec, data):
    txns = [_txn(i, o, r) for i, (o, r) in enumerate(spec, start=1)]
    permuted = data.draw(st.permutations(txns))
    base = find_order_breaks(extract_credit_refs(txns, CFG), CFG)
    assert find_order_breaks(extract_credit_refs(permuted, CFG), CFG) == base
    first = check_invoice_sequence(permuted, CFG)
    assert check_invoice_sequence(permuted, CFG) == first
    assert [f.field for f in check_invoice_sequence(txns, CFG)] == [f.field for f in first]


# Feature: node3-fabrication-rules, Property 6: Sequence break detection
@PBT
@given(
    case=rotated_case(),
    debits=st.lists(st.tuples(st.integers(0, 40), st.integers(100, 999)), max_size=8),
)
def test_input_not_mutated_and_debits_ignored(case, debits):
    offsets, refs, r1, r2 = case
    txns = _build(offsets, refs)
    extra = [
        _txn(100 + k, o, r, amount=-500.0 - k) for k, (o, r) in enumerate(debits)
    ]
    combined = txns + extra
    before = [t.model_copy() for t in combined]
    got = check_invoice_sequence(combined, CFG)
    assert combined == before
    assert got == check_invoice_sequence(txns, CFG)
    assert len(got) == 1 and got[0].field == f"rows {min(r1, r2)}, {max(r1, r2)}"
