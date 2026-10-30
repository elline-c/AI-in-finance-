"""Task 6.3: property tests for the direction resolver.

Property 3: Printed amount is preserved   -- Validates: Requirements 5.9, 6.4
Property 4: Direction follows the evidence -- Validates: Requirements 6.1, 6.2, 6.3, 6.4
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.ingest_parse.direction import DIRECTION_UNCERTAIN, resolve_direction
from fraud_detect.nodes.ingest_parse.money import Money
from fraud_detect.nodes.ingest_parse.table import AmountCell, RawRow, TableLayout

SETTINGS = settings(max_examples=200, deadline=None)

cents = st.integers(min_value=1, max_value=10_000_000)  # printed magnitude, in cents
balances = st.integers(min_value=-10_000_000, max_value=10_000_000)  # cents
layouts = st.sampled_from(["two_column", "single_signed", "unknown"])
# how the balance moved relative to the printed amount
steps = st.sampled_from(["consistent", "opposite", "wrong_size_same_dir", "unchanged"])


def _row(mag_c, column=None, sign=None, suffix=None, bal_c=0):
    cell = AmountCell(
        money=Money(value=mag_c / 100, explicit_sign=sign, suffix=suffix),
        x0=0.0, x1=1.0, column=column,
    )
    return RawRow(
        date_text="01/01/2025", value_date_text=None, description="T",
        amount_cells=[cell], balance=Money(value=bal_c / 100, explicit_sign=None, suffix=None), line_text="",
    )


def _layout(kind):
    return TableLayout(header_y=0.0, columns={}, kind=kind)


def _new_balance(prev_c, mag_c, credit, step):
    """Balance (cents) after a step relative to the claimed direction."""
    s = 1 if credit else -1
    if step == "consistent":
        return prev_c + s * mag_c
    if step == "opposite":
        return prev_c - s * mag_c
    if step == "wrong_size_same_dir":
        return prev_c + s * (mag_c + 5)
    return prev_c


def _balance_agrees(credit, mag_c, prev_c, bal_c):
    delta = bal_c - prev_c
    if delta == 0:
        return False  # magnitudes are >= 1 cent
    return (delta > 0) == credit


# ------------------------------------------------------------ Property 3
@SETTINGS
@given(
    mag=cents, prev=balances, step=steps, kind=layouts,
    evidence=st.sampled_from(["money_out", "money_in", "minus", "plus", "CR", "DR", "none"]),
    credit_guess=st.booleans(), has_prev=st.booleans(),
)
def test_printed_amount_is_preserved(mag, prev, step, kind, evidence, credit_guess, has_prev):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 3: Printed amount is preserved

    **Validates: Requirements 5.9, 6.4**
    """
    bal = _new_balance(prev, mag, credit_guess, step)
    column = evidence if evidence in ("money_out", "money_in") else ("amount" if evidence != "none" else None)
    sign = {"minus": -1, "plus": 1}.get(evidence)
    suffix = evidence if evidence in ("CR", "DR") else None
    row = _row(mag, column=column, sign=sign, suffix=suffix, bal_c=bal)
    r = resolve_direction(row, _layout(kind), prev / 100 if has_prev else None)
    if r is None:
        return
    # never recomputed from the balance, even when the balance step disagrees
    assert round(abs(r.amount) * 100) == mag


# ------------------------------------------------------------ Property 4
@SETTINGS
@given(mag=cents, prev=balances, step=steps, kind=layouts, column=st.sampled_from(["money_out", "money_in"]))
def test_column_evidence_decides_direction(mag, prev, step, kind, column):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 4: Direction follows the evidence

    **Validates: Requirements 6.1, 6.4**
    """
    credit = column == "money_in"
    bal = _new_balance(prev, mag, not credit, step)  # balance step may point either way
    r = resolve_direction(_row(mag, column=column, bal_c=bal), _layout(kind), prev / 100)
    assert r.method == "column"
    assert (r.amount > 0) == credit
    agrees = _balance_agrees(credit, mag, prev, bal)
    assert r.agrees_with_balance is agrees
    assert r.finding_code == (None if agrees else DIRECTION_UNCERTAIN)


@SETTINGS
@given(mag=cents, bal=balances, kind=layouts, column=st.sampled_from(["money_out", "money_in"]))
def test_column_without_previous_balance_has_no_finding(mag, bal, kind, column):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 4: Direction follows the evidence

    **Validates: Requirements 6.1**
    """
    r = resolve_direction(_row(mag, column=column, bal_c=bal), _layout(kind), None)
    assert r.method == "column"
    assert (r.amount > 0) == (column == "money_in")
    assert r.agrees_with_balance is None and r.finding_code is None


@SETTINGS
@given(
    mag=cents, prev=balances, step=steps, kind=layouts, has_prev=st.booleans(),
    marker=st.sampled_from([("sign", -1), ("sign", 1), ("suffix", "CR"), ("suffix", "DR")]),
)
def test_printed_sign_used_when_no_column(mag, prev, step, kind, has_prev, marker):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 4: Direction follows the evidence

    **Validates: Requirements 6.2, 6.4**
    """
    which, val = marker
    credit = val in (1, "CR")
    bal = _new_balance(prev, mag, not credit, step)
    row = _row(
        mag, column="amount",
        sign=val if which == "sign" else None, suffix=val if which == "suffix" else None,
        bal_c=bal,
    )
    r = resolve_direction(row, _layout(kind), prev / 100 if has_prev else None)
    assert r.method == "printed_sign"
    assert (r.amount > 0) == credit
    if not has_prev:
        assert r.agrees_with_balance is None and r.finding_code is None
    else:
        agrees = _balance_agrees(credit, mag, prev, bal)
        assert r.finding_code == (None if agrees else DIRECTION_UNCERTAIN)


@SETTINGS
@given(mag=cents, prev=balances, credit=st.booleans(), step=steps)
def test_balance_change_used_only_when_no_column_or_sign(mag, prev, credit, step):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 4: Direction follows the evidence

    **Validates: Requirements 6.3, 6.4**
    """
    bal = _new_balance(prev, mag, credit, step)
    r = resolve_direction(_row(mag, bal_c=bal), _layout("unknown"), prev / 100)
    delta = bal - prev
    if delta == 0:
        assert r is None  # no evidence at all
        return
    assert r.method == "balance_change"
    assert (r.amount > 0) == (delta > 0)
    # consistent row: no finding; magnitude disagreement flags uncertainty
    if abs(delta) == mag:
        assert r.finding_code is None and r.agrees_with_balance is True
    else:
        assert r.finding_code == DIRECTION_UNCERTAIN


@SETTINGS
@given(mag=cents, bal=balances, kind=layouts)
def test_no_evidence_returns_none(mag, bal, kind):
    """Feature: multi-layout-ingestion-and-fabrication-rules, Property 4: Direction follows the evidence

    **Validates: Requirements 6.3**
    """
    assert resolve_direction(_row(mag, bal_c=bal), _layout(kind), None) is None
