"""Property tests for the simulator (task 2.4): Properties 2 to 6 of the design.

Each example simulates a ledger (cheap, no PDF), so the example counts are modest.
"""
from __future__ import annotations

import re
import statistics
from datetime import date
from decimal import Decimal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.ledger import replay, simulate_ledger
from fraud_detect.corpus.schema import SimulatorKnobs
from fraud_detect.models import Transaction
from fraud_detect.nodes.fabrication import DEFAULT_CONFIG, extract_credit_refs

SETTINGS = settings(
    max_examples=25, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

seeds = st.integers(min_value=0, max_value=10_000)
starts = st.builds(lambda y, m: date(y, m, 1), st.integers(2023, 2026), st.integers(1, 12))
n_months = st.integers(6, 12)

RAIL_RE = re.compile(r"^(FAST FROM|PAYNOW FROM|GIRO)\s+(.+?)\s+INV(\d{3,})\s*$")


def _ledger(seed, start, n, knobs=None, cap=None):
    knobs = knobs or SimulatorKnobs()
    return simulate_ledger(make_company(seed, "C0001"), start, n, seed, knobs, cap)


def _ym(d):
    return d.year, d.month


def _receipts(ledger):
    return [r for r in ledger.rows if r.kind == "receipt"]


# --------------------------------------------------------------------------- Property 2
@SETTINGS
@given(seed=seeds, start=starts, n=n_months)
def test_property_2_receipts_have_readable_rising_references(seed, start, n):
    """**Validates: Requirements 1.3, 1.4**"""
    ledger = _ledger(seed, start, n, SimulatorKnobs(invoice_disorder=0.0))
    receipts = _receipts(ledger)
    assert receipts
    for r in receipts:
        m = RAIL_RE.match(r.description)
        assert m and m.group(2) == r.customer and f"INV{m.group(3)}" == r.invoice_ref
    txns = [Transaction(index=i, date=r.date, description=r.description, amount=r.amount)
            for i, r in enumerate(receipts, start=1)]
    refs = extract_credit_refs(txns, DEFAULT_CONFIG)
    assert [c.ref for c in refs] == [r.invoice_ref for r in receipts]
    nums = [int(r.invoice_ref[3:]) for r in receipts]
    assert nums == sorted(nums)


# --------------------------------------------------------------------------- Property 3
@SETTINGS
@given(seed=seeds, start=starts, n=n_months)
def test_property_3_fixed_costs_flat_and_balance_behaves(seed, start, n):
    """**Validates: Requirements 1.5, 1.6, 1.7, 1.10**"""
    ledger = _ledger(seed, start, n)
    for kind in ("payroll", "cpf", "rent"):
        amounts = [abs(r.amount) for r in ledger.rows if r.kind == kind]
        assert len(amounts) == n
        med = statistics.median(amounts)
        assert all(abs(a - med) <= 0.10 * med for a in amounts), (kind, amounts)
    gst_quarters = {(_ym(r.date)[0], (r.date.month - 1) // 3) for r in ledger.rows if r.kind == "gst"}
    for m in ledger.months:
        if m.month in (3, 6, 9, 12):
            assert (m.year, (m.month - 1) // 3) in gst_quarters
    for r in ledger.rows:
        assert Decimal(repr(r.amount)).as_tuple().exponent >= -2
    assert Decimal(repr(ledger.opening_balance)).as_tuple().exponent >= -2
    balances = replay(ledger)
    assert min(balances) >= 0
    # each month opens at the previous close: replay is one running total, and every
    # month's first balance equals the previous month's last plus that row's amount.
    prev = ledger.opening_balance
    for r, b in zip(ledger.rows, balances):
        assert abs(b - round(prev + r.amount, 2)) < 1e-9
        prev = b


# --------------------------------------------------------------------------- Property 4
@SETTINGS
@given(seed=seeds, start=starts, n=n_months)
def test_property_4_ledger_has_requested_shape(seed, start, n):
    """**Validates: Requirements 1.1, 1.2, 1.11, 12.2**"""
    ledger = _ledger(seed, start, n)
    assert len(ledger.months) == n
    idx = [m.year * 12 + m.month for m in ledger.months]
    assert idx == list(range(idx[0], idx[0] + n))
    assert ledger.months[0] == start
    dates = [r.date for r in ledger.rows]
    assert dates == sorted(dates)
    for r in ledger.rows:
        assert _ym(r.date) == _ym(ledger.months[r.month_index])
    assert ledger.company.name.endswith("PTE. LTD.")
    assert 5 <= len(ledger.company.customers) <= 25
    for r in ledger.rows:
        if r.kind in ("receipt", "card", "supplier"):
            assert r.date.weekday() < 5, r
        if "GIRO" in r.description.upper() and r.kind != "receipt":
            assert r.date.weekday() < 5, r


# --------------------------------------------------------------------------- Property 5
@SETTINGS
@given(seed=seeds, start=starts, n=n_months, rate=st.floats(0.0, 0.5))
def test_property_5_repeated_references_are_hard_cases(seed, start, n, rate):
    """**Validates: Requirements 1.8**"""
    ledger = _ledger(seed, start, n, SimulatorKnobs(instalment_rate=rate))
    by_ref: dict[str, list] = {}
    for r in _receipts(ledger):
        by_ref.setdefault(r.invoice_ref, []).append(r)
    for ref, rows in by_ref.items():
        if len(rows) >= 2:
            assert all(r.hard_case and r.hard_case.startswith("instalment") for r in rows), ref
            assert len({r.customer for r in rows}) == 1
    hard = [r for r in ledger.rows if r.hard_case]
    assert all(len(by_ref[r.invoice_ref]) >= 2 for r in hard)


@SETTINGS
@given(seed=seeds, start=starts, n=n_months)
def test_property_5_zero_rate_gives_no_repeats(seed, start, n):
    """**Validates: Requirements 1.8**"""
    ledger = _ledger(seed, start, n, SimulatorKnobs(instalment_rate=0.0))
    refs = [r.invoice_ref for r in _receipts(ledger)]
    assert len(refs) == len(set(refs))
    assert not any(r.hard_case for r in ledger.rows)


# --------------------------------------------------------------------------- Property 6
@SETTINGS
@given(seed=seeds, start=starts, n=n_months, cap=st.integers(18, 50))
def test_property_6_row_cap_respected_and_trim_recorded(seed, start, n, cap):
    """**Validates: Requirements 1.12**"""
    full = _ledger(seed, start, n)
    capped = _ledger(seed, start, n, cap=cap)
    kept = {r.row_id for r in capped.rows}
    assert kept <= {r.row_id for r in full.rows}
    trims = dict(capped.trims)
    for m in range(n):
        full_m = [r for r in full.rows if r.month_index == m]
        cap_m = [r for r in capped.rows if r.month_index == m]
        assert len(cap_m) <= cap
        dropped = [r for r in full_m if r.row_id not in kept]
        assert trims.get(m, 0) == len(dropped)
        if len(full_m) <= cap:
            assert not dropped
        # no non-card row dropped while a card payment remains in that month
        if any(r.kind != "card" for r in dropped):
            assert not any(r.kind == "card" for r in cap_m)
        # card payments are dropped smallest first
        if dropped and any(r.kind == "card" for r in dropped):
            kept_cards = [abs(r.amount) for r in cap_m if r.kind == "card"]
            lost_cards = [abs(r.amount) for r in dropped if r.kind == "card"]
            if kept_cards:
                assert max(lost_cards) <= min(kept_cards)
