"""The seven behavioural tampers (Requirement 8), all planned as ledger edits.

Rows are inserted/removed with the recorded edit ops of ``ledger_ops`` (so the edit
log replays to the edited ledger) and printed balances follow the amounts again
(``recompute_balances``), keeping every monthly section internally consistent.
Variants (fast/slow, large/moderate) come from the document seed via ``rng``.
``uen_mismatch`` changes only the printed holder (method ``content``).
"""
from __future__ import annotations

import calendar
import random
from dataclasses import replace
from datetime import date, timedelta

from fraud_detect.corpus.schema import EditOp, Ledger, LedgerRow, TamperLabel, TamperResult
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper._registry import info_for, register

_QUIET_KINDS = ("service_charge", "interest")

_SHELL_NAMES = ("UNKNOWN TRADING LLC", "UNKNOWN VENTURES LLC", "GLOBAL ASSETS HOLDINGS",
                "PACIFIC UNKNOWN LLC", "STERLING HOLDINGS LLC", "UNKNOWN CAPITAL HOLDINGS")
_PARTIES = ("ALPHA CAPITAL", "BRIGHT SOURCE", "CORAL BRIDGE", "DELTA SUPPLY",
            "EASTERN LINK", "FALCON PARTNERS", "GOLDEN HARBOUR", "HORIZON TRADE")
_OTHER_HOLDERS = ("ORCHID LOGISTICS PTE LTD", "MERLION TRADING PTE LTD", "BAYFRONT SERVICES PTE LTD",
                  "TANJONG ENTERPRISE PTE LTD", "KALLANG FOODS PTE LTD", "SENTOSA WORKS PTE LTD")


def _money(x: float) -> str:
    return f"{x:.2f}"


def _tag(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _span(ledger: Ledger) -> tuple[date, date]:
    first = ledger.months[0].replace(day=1)
    last = ledger.months[-1]
    return first, last.replace(day=calendar.monthrange(last.year, last.month)[1])


def _month_index(ledger: Ledger, d: date) -> int:
    for i, m in enumerate(ledger.months):
        if (m.year, m.month) == (d.year, d.month):
            return i
    raise ValueError(f"{d} is outside the ledger months")


def _pick_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def _inserts(ledger: Ledger, specs) -> tuple[list[EditOp], Ledger]:
    """Insert ``specs`` (date, description, amount, kind) in date order, recording edits."""
    existing = {r.row_id for r in ledger.rows}
    edits: list[EditOp] = []
    cur = ledger
    for k, (d, desc, amount, kind) in enumerate(specs, start=1):
        row_id = f"X{k:03d}"
        assert row_id not in existing
        row = LedgerRow(row_id=row_id, month_index=_month_index(cur, d), date=d,
                        description=desc, amount=round(amount, 2), kind=kind, fraud=True)
        pos = sum(1 for r in cur.rows if r.date <= d)
        after = cur.rows[pos - 1].row_id if pos > 0 else None
        edit = EditOp("insert_row", row_id, "row", None, ledger_ops._row_json(row, after))
        edits.append(edit)
        cur = ledger_ops.apply_edit(cur, edit)
    return edits, cur


def _finish(name: str, genuine: Ledger, edits: list[EditOp], *, inserted: bool,
            months: tuple[str, ...] = (), notes: str = "") -> TamperResult:
    edits = list(edits) + [EditOp("recompute_balances")]
    edited = ledger_ops.apply_edits(genuine, edits)
    affected: tuple[dict, ...] = ()
    if inserted:
        old = {r.row_id for r in genuine.rows}
        affected = tuple(
            {"row_id": r.row_id, "row_index": i + 1, "month": _tag(r.date), "field": "row",
             "original": "", "new": _money(r.amount)}
            for i, r in enumerate(edited.rows) if r.row_id not in old)
        months = tuple(sorted({a["month"] for a in affected}))
    label = TamperLabel(
        is_tampered=True, tamper_types=(name,), method="rerender",
        arithmetic_consistent=ledger_ops.is_arithmetic_consistent(edited),
        affected_rows=affected, affected_months=months,
        should_be_seen_by=info_for(name).should_be_seen_by, notes=notes)
    return TamperResult(ledger=edited, edits=tuple(edits), file_ops=(), label=label)


@register("structuring")
def plan_structuring(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    first, last = _span(ledger)
    n = rng.randint(3, 5)
    slow = rng.random() < 0.5
    room = (last - first).days
    if slow and room < 62:
        slow = False
    if slow:
        total = rng.randint(32, 60)
        offsets = sorted([0, total] + [rng.randint(1, total - 1) for _ in range(n - 2)])
    else:
        total = rng.randint(3, 14)
        offsets = sorted([0, total] + [rng.randint(0, total) for _ in range(n - 2)])
    start = _pick_date(rng, first, last - timedelta(days=total))
    specs = [(start + timedelta(days=o), "CASH DEPOSIT",
              min(9999.99, round(rng.uniform(9500.00, 9999.99), 2)), "cash_deposit")
             for o in offsets]
    edits, _ = _inserts(ledger, specs)
    return _finish("structuring", ledger, edits, inserted=True,
                   notes="slow variant (over 30 days)" if slow else "fast variant (within 14 days)")


@register("sudden_large_deposit")
def plan_sudden_large_deposit(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    first, last = _span(ledger)
    avg = sum(r.amount for r in ledger.rows if r.amount > 0) / len(ledger.months)
    large = rng.random() < 0.5
    ratio = rng.uniform(3.2, 6.0) if large else rng.uniform(1.55, 2.45)
    d = _pick_date(rng, first, last)
    edits, _ = _inserts(ledger, [(d, "INWARD REMITTANCE", round(avg * ratio, 2), "transfer_in")])
    return _finish("sudden_large_deposit", ledger, edits, inserted=True,
                   notes="large variant (above 3x)" if large else "moderate variant (1.5x to 2.5x)")


@register("shell_counterparty")
def plan_shell_counterparty(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    first, last = _span(ledger)
    d = _pick_date(rng, first, last)
    name = rng.choice(_SHELL_NAMES)
    amount = round(rng.uniform(50000.00, 250000.00), 2)
    edits, _ = _inserts(ledger, [(d, f"TRANSFER FROM {name}", amount, "transfer_in")])
    return _finish("shell_counterparty", ledger, edits, inserted=True)


@register("rapid_pass_through")
def plan_rapid_pass_through(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    first, last = _span(ledger)
    specs = []
    for _ in range(rng.randint(2, 4)):
        d_in = _pick_date(rng, first, last - timedelta(days=3))
        d_out = d_in + timedelta(days=rng.randint(1, 3))
        src, dst = rng.sample(_PARTIES, 2)
        amount = round(rng.uniform(20000.00, 90000.00), 2)
        out = round(amount * rng.uniform(0.951, 0.999), 2)
        specs.append((d_in, f"TRANSFER FROM {src}", amount, "transfer_in"))
        specs.append((d_out, f"TRANSFER TO {dst}", -out, "transfer_out"))
    specs.sort(key=lambda s: s[0])
    edits, _ = _inserts(ledger, specs)
    return _finish("rapid_pass_through", ledger, edits, inserted=True)


@register("dormant_reactivation")
def plan_dormant_reactivation(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    k = min(rng.randint(3, 5), len(ledger.months) - 1)
    edits = [EditOp("delete_row", r.row_id, "row", ledger_ops._row_json(r), None)
             for r in ledger.rows if r.month_index < k and r.kind not in _QUIET_KINDS]
    months = tuple(_tag(m) for m in ledger.months[:k])
    return _finish("dormant_reactivation", ledger, edits, inserted=False, months=months,
                   notes=f"first {k} months keep only charges and interest")


@register("period_gap")
def plan_period_gap(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    m = rng.randint(1, len(ledger.months) - 2)
    edits = [EditOp("delete_month", None, str(m))]
    return _finish("period_gap", ledger, edits, inserted=False,
                   months=(_tag(ledger.months[m]),),
                   notes="later sections open at their own opening balance")


@register("uen_mismatch")
def plan_uen_mismatch(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    pool = [n for n in _OTHER_HOLDERS if n != ledger.company.name]
    holder = rng.choice(pool)
    edited = replace(ledger, printed_holder=holder)
    label = TamperLabel(
        is_tampered=True, tamper_types=("uen_mismatch",), method="content",
        arithmetic_consistent=ledger_ops.is_arithmetic_consistent(edited),
        should_be_seen_by=info_for("uen_mismatch").should_be_seen_by,
        notes=f"printed holder {holder!r}; registered name kept")
    return TamperResult(ledger=edited, edits=(), file_ops=(), label=label)
