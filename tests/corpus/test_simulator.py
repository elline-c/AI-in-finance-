"""Tests for the company and ledger simulator (Requirements 1.1 to 1.12).

Written before the code (task 2.1); they fail with ``ModuleNotFoundError`` until
``fraud_detect/corpus/company.py`` (task 2.2) and ``ledger.py`` (task 2.3) exist.

Conventions these tests pin down (the design left them open):

* ``LedgerRow.kind`` values: "receipt" (FAST FROM / PAYNOW FROM / GIRO customer
  receipts), "payroll", "cpf", "rent", "gst", "supplier", "utility", "card",
  "service_charge", "interest".
* GST: for every calendar quarter whose last month (Mar, Jun, Sep, Dec) is inside
  the statement, at least one "gst" row is dated in one of that quarter's months.
* Invoice instalments: a second part repeats the same (customer, invoice_ref);
  every row of an invoice that is paid in more than one part has ``hard_case``
  set to a string starting with "instalment" (e.g. "instalment",
  "instalment_identical").
* Row cap trim record: ``Ledger`` gets a new field
  ``trims: tuple[tuple[int, int], ...] = ()`` holding ``(month_index, n_dropped)``
  for every month where rows were dropped to meet ``max_rows_per_month``.
  (Task 2.2/2.3 adds it to ``schema.Ledger``; the manifest writer reads it later.)
  The cap must not change the random draws, so a capped ledger is the uncapped
  ledger with some rows removed: card payments go first, and a non-card row is
  dropped only in a month where no card payment is left.
* ``invoice_disorder`` is the share of receipts whose invoice number is swapped
  with another; any value >= 0.3 must break the rising order for some seed.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import replace
from datetime import date
from functools import lru_cache

import pytest

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.ledger import replay, simulate_ledger
from fraud_detect.corpus.schema import SimulatorKnobs
from fraud_detect.models import Transaction
from fraud_detect.nodes.fabrication import DEFAULT_CONFIG, extract_credit_refs

KNOBS = SimulatorKnobs()
START = date(2025, 1, 1)
RAIL_RE = re.compile(r"^(FAST FROM|PAYNOW FROM|GIRO)\b.*\bINV\d{3,}\s*$")
FIXED_WORDS = ("RENT", "CPF", "PAYROLL", "SALARY", "LOAN", "CAPITAL", "TRANSFER", "INTEREST")
SHELL_SUBSTRINGS = ("UNKNOWN", "LLC", "HOLDINGS", "TT")


# --------------------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def _company(seed: int = 1):
    return make_company(seed, "C0001")


@lru_cache(maxsize=None)
def _ledger(seed: int = 1, n: int = 12, start: date = START, knobs: SimulatorKnobs = KNOBS,
            cap: int | None = None):
    return simulate_ledger(_company(seed), start, n, seed, knobs, cap)


def _rows(ledger, kind):
    return [r for r in ledger.rows if r.kind == kind]


def _receipts(ledger):
    return _rows(ledger, "receipt")


def _ym(d: date) -> tuple[int, int]:
    return d.year, d.month


def _month_totals(ledger, kind):
    totals: dict[int, float] = {}
    for r in _rows(ledger, kind):
        totals[r.month_index] = totals.get(r.month_index, 0.0) + abs(r.amount)
    return totals


def _whole_word(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text.upper()) is not None


def _ref_numbers(ledger):
    return [int(re.search(r"\d+", r.invoice_ref).group()) for r in _receipts(ledger)]


# --------------------------------------------------------------------------- shape (1.1)
@pytest.mark.parametrize("n", [6, 12])
def test_month_count_and_consecutive_months(n):
    ledger = _ledger(1, n)
    assert len(ledger.months) == n
    expected = [(2025 + (m // 12), m % 12 + 1) for m in range(n)]
    assert [_ym(m) for m in ledger.months] == expected
    assert all(m.day == 1 for m in ledger.months)


def test_rows_in_date_order_and_inside_their_month():
    ledger = _ledger(1, 12)
    dates = [r.date for r in ledger.rows]
    assert dates == sorted(dates)
    for r in ledger.rows:
        assert _ym(r.date) == _ym(ledger.months[r.month_index])
    assert len({r.row_id for r in ledger.rows}) == len(ledger.rows)


def test_start_in_late_year_crosses_year_end():
    ledger = _ledger(2, 8, date(2025, 9, 1))
    assert [_ym(m) for m in ledger.months][-1] == (2026, 4)


# --------------------------------------------------------------------------- company (1.2)
@pytest.mark.parametrize("seed", range(10))
def test_company_profile_shape(seed):
    c = make_company(seed, "C0007")
    assert c.company_id == "C0007"
    assert c.name.endswith("PTE. LTD.")
    assert 5 <= len(c.customers) <= 25
    assert len(c.suppliers) >= 1
    assert re.fullmatch(r"\d{8,9}[A-Z]", c.uen)
    assert c.address_lines and c.account_number


def test_company_deterministic_and_seed_sensitive():
    assert make_company(5, "C0001") == make_company(5, "C0001")
    assert make_company(5, "C0001") != make_company(6, "C0001")


def test_customer_and_supplier_names_do_not_trip_node3_keywords():
    for seed in range(40):
        c = make_company(seed, "C0001")
        for p in (*c.customers, *c.suppliers):
            name = p.name.upper()
            for w in FIXED_WORDS:
                assert not _whole_word(w, name), (seed, p.name, w)
            for s in SHELL_SUBSTRINGS:
                assert s not in name, (seed, p.name, s)


def test_receipt_and_credit_descriptions_avoid_shell_text():
    for seed in (1, 2, 3):
        for r in _ledger(seed, 12).rows:
            if r.amount > 0:
                for s in SHELL_SUBSTRINGS:
                    assert s not in r.description.upper(), (seed, r.description)


def test_receipt_descriptions_avoid_fixed_cost_words():
    for r in _receipts(_ledger(1, 12)):
        for w in FIXED_WORDS:
            assert not _whole_word(w, r.description), r.description


# --------------------------------------------------------------------------- receipts (1.3, 1.4)
def test_receipt_descriptions_have_rail_customer_and_invoice():
    ledger = _ledger(1, 12)
    names = {p.name for p in ledger.company.customers}
    receipts = _receipts(ledger)
    assert receipts
    for r in receipts:
        assert RAIL_RE.match(r.description), r.description
        assert r.amount > 0
        assert r.customer in names and r.customer in r.description
        assert re.fullmatch(r"INV\d{3,}", r.invoice_ref)
        assert r.invoice_ref in r.description


def test_extract_credit_refs_reads_every_receipt():
    ledger = _ledger(1, 12)
    receipts = _receipts(ledger)
    txns = [Transaction(index=i, date=r.date, description=r.description, amount=r.amount)
            for i, r in enumerate(receipts, start=1)]
    refs = extract_credit_refs(txns, DEFAULT_CONFIG)
    assert len(refs) == len(receipts)
    assert [c.ref for c in refs] == [r.invoice_ref for r in receipts]


def test_invoice_numbers_never_decrease_with_date_at_disorder_zero():
    for seed in range(1, 6):
        nums = _ref_numbers(_ledger(seed, 12))
        assert nums == sorted(nums), seed


def test_disorder_knob_can_break_invoice_order():
    knobs = replace(KNOBS, invoice_disorder=0.5)
    broken = [n for n in (_ref_numbers(_ledger(s, 12, START, knobs)) for s in range(1, 6))
              if n != sorted(n)]
    assert broken


# --------------------------------------------------------------------------- fixed costs and GST (1.5, 1.6)
@pytest.mark.parametrize("kind", ["payroll", "cpf", "rent"])
def test_fixed_costs_every_month_within_ten_percent_of_median(kind):
    for seed in (1, 2, 3):
        ledger = _ledger(seed, 12)
        totals = _month_totals(ledger, kind)
        assert set(totals) == set(range(12)), (seed, kind)
        med = statistics.median(totals.values())
        for m, v in totals.items():
            assert abs(v - med) <= 0.10 * med, (seed, kind, m, v, med)
        assert all(r.amount < 0 for r in _rows(ledger, kind))


def test_description_keywords_for_fixed_costs():
    ledger = _ledger(1, 6)
    assert all("PAYROLL" in r.description.upper() for r in _rows(ledger, "payroll"))
    assert all("CPF" in r.description.upper() for r in _rows(ledger, "cpf"))
    assert all("RENT" in r.description.upper() for r in _rows(ledger, "rent"))
    assert all("GST" in r.description.upper() for r in _rows(ledger, "gst"))


@pytest.mark.parametrize("n,start", [(12, date(2025, 1, 1)), (6, date(2025, 1, 1)),
                                      (9, date(2025, 4, 1)), (8, date(2025, 9, 1))])
def test_gst_row_in_each_quarter_whose_last_month_is_inside(n, start):
    ledger = _ledger(1, n, start)
    months = {_ym(m) for m in ledger.months}
    gst_months = {_ym(r.date) for r in _rows(ledger, "gst")}
    checked = 0
    for y in (start.year, start.year + 1):
        for q_end in (3, 6, 9, 12):
            if (y, q_end) not in months:
                continue
            quarter = {(y, q_end - 2), (y, q_end - 1), (y, q_end)}
            assert quarter & gst_months, (y, q_end)
            checked += 1
    assert checked >= 1
    assert all(r.amount < 0 for r in _rows(ledger, "gst"))


# --------------------------------------------------------------------------- other rows, decimals (1.7)
def test_supplier_utility_card_charge_interest_rows_present():
    ledger = _ledger(1, 12)
    for kind in ("supplier", "utility", "card", "service_charge", "interest"):
        assert _rows(ledger, kind), kind
    assert all(r.description.startswith("FAST TO ") for r in _rows(ledger, "supplier"))
    assert all("NETS" in r.description.upper() or "CARD" in r.description.upper()
               for r in _rows(ledger, "card"))
    assert all("SERVICE CHARGE" in r.description.upper() for r in _rows(ledger, "service_charge"))
    assert all(r.amount > 0 for r in _rows(ledger, "interest"))
    assert len(_rows(ledger, "service_charge")) >= 12


def test_every_amount_and_balance_has_two_decimals():
    ledger = _ledger(1, 12)
    for r in ledger.rows:
        assert round(r.amount, 2) == r.amount and r.amount != 0
    assert round(ledger.opening_balance, 2) == ledger.opening_balance
    for b in replay(ledger):
        assert round(b, 2) == b


# --------------------------------------------------------------------------- instalments (1.8)
def _repeats(ledger):
    seen: dict[tuple, list] = {}
    for r in _receipts(ledger):
        seen.setdefault((r.customer, r.invoice_ref), []).append(r)
    return {k: v for k, v in seen.items() if len(v) > 1}


def test_instalment_rate_zero_has_no_repeated_reference():
    knobs = replace(KNOBS, instalment_rate=0.0)
    for seed in (1, 2, 3):
        ledger = _ledger(seed, 12, START, knobs)
        assert not _repeats(ledger)
        assert all(r.hard_case is None for r in _receipts(ledger))


def test_positive_instalment_rate_gives_recorded_hard_cases():
    knobs = replace(KNOBS, instalment_rate=0.5)
    ledger = _ledger(1, 12, START, knobs)
    repeats = _repeats(ledger)
    assert repeats
    for parts in repeats.values():
        for r in parts:
            assert r.hard_case and r.hard_case.startswith("instalment")
    flagged = {r.row_id for r in ledger.rows if r.hard_case}
    assert flagged == {r.row_id for parts in repeats.values() for r in parts}


# --------------------------------------------------------------------------- balances (1.9, 1.10)
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_replay_never_below_zero_and_chains_from_opening(seed):
    ledger = _ledger(seed, 12)
    balances = replay(ledger)
    assert len(balances) == len(ledger.rows)
    assert ledger.opening_balance >= 0
    assert min(balances) >= 0
    prev = ledger.opening_balance
    for r, b in zip(ledger.rows, balances):
        assert b == pytest.approx(round(prev + r.amount, 2), abs=0.005)
        prev = b


def test_each_month_opens_at_previous_close():
    ledger = _ledger(1, 12)
    balances = replay(ledger)
    opening = ledger.opening_balance
    for m in range(len(ledger.months)):
        idx = [i for i, r in enumerate(ledger.rows) if r.month_index == m]
        assert idx, m
        before = balances[idx[0] - 1] if idx[0] > 0 else opening
        assert before == (balances[idx[0] - 1] if m else opening)
        assert idx == list(range(idx[0], idx[-1] + 1))   # month rows are contiguous
        if m:
            prev_idx = [i for i, r in enumerate(ledger.rows) if r.month_index == m - 1]
            assert balances[prev_idx[-1]] == before


def test_turnover_scales_with_growth_and_seasonality_limits():
    flat = replace(KNOBS, growth_range=(0.0, 0.0), seasonality_strength=0.0)
    wild = replace(KNOBS, growth_range=(0.03, 0.03), seasonality_strength=0.3)
    def credits(ledger):
        return sum(r.amount for r in _receipts(ledger))
    a, b = _ledger(1, 12, START, flat), _ledger(1, 12, START, wild)
    assert credits(a) > 0 and credits(b) > 0
    assert credits(a) != credits(b)


# --------------------------------------------------------------------------- weekdays (1.11)
def test_receipts_card_supplier_and_giro_fall_on_weekdays():
    for seed in (1, 2):
        ledger = _ledger(seed, 12)
        for r in ledger.rows:
            if r.kind in ("receipt", "card", "supplier") or r.description.startswith("GIRO"):
                assert r.date.weekday() < 5, (r.kind, r.description, r.date)
    kinds_seen = {r.kind for r in _ledger(1, 12).rows}
    assert {"receipt", "card", "supplier"} <= kinds_seen


# --------------------------------------------------------------------------- row cap (1.12)
CAP = 35
HEAVY = replace(KNOBS, rows_per_month_range=(45, 60))


def test_row_cap_respected_card_dropped_first_and_trim_recorded():
    full = _ledger(1, 12, START, HEAVY, None)
    capped = _ledger(1, 12, START, HEAVY, CAP)
    per_month = lambda l, m, k=None: [r for r in l.rows if r.month_index == m and (k is None or r.kind == k)]
    trimmed_months = set()
    for m in range(12):
        assert len(per_month(capped, m)) <= CAP, m
        dropped = len(per_month(full, m)) - len(per_month(capped, m))
        if dropped > 0:
            trimmed_months.add(m)
            if per_month(capped, m, "card"):
                # card payments remain, so no non-card row may have been dropped
                for kind in {r.kind for r in full.rows} - {"card"}:
                    assert len(per_month(capped, m, kind)) == len(per_month(full, m, kind)), (m, kind)
    assert trimmed_months, "knobs/cap should force a trim"
    assert dict(capped.trims) == {
        m: len(per_month(full, m)) - len(per_month(capped, m)) for m in trimmed_months
    }
    assert full.trims == ()


def test_row_cap_dropped_card_payments_are_the_smallest():
    full = _ledger(1, 12, START, HEAVY, None)
    capped = _ledger(1, 12, START, HEAVY, CAP)
    for m, _ in capped.trims:
        kept = sorted(abs(r.amount) for r in capped.rows if r.month_index == m and r.kind == "card")
        all_ = sorted(abs(r.amount) for r in full.rows if r.month_index == m and r.kind == "card")
        gone = list(all_)
        for v in kept:
            gone.remove(v)
        if kept and gone:
            assert max(gone) <= min(kept)


def test_no_trim_without_cap_or_under_cap():
    assert _ledger(1, 12).trims == ()
    assert _ledger(1, 12, START, KNOBS, 500).trims == ()


# --------------------------------------------------------------------------- determinism (10.1)
def test_same_seed_same_ledger():
    a = simulate_ledger(_company(1), START, 7, 11, KNOBS, None)
    b = simulate_ledger(make_company(1, "C0001"), START, 7, 11, KNOBS, None)
    assert a == b


def test_different_seed_different_ledger():
    a = simulate_ledger(_company(1), START, 7, 11, KNOBS, None)
    b = simulate_ledger(_company(1), START, 7, 12, KNOBS, None)
    assert a.rows != b.rows
