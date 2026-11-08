"""Content-fraud tampers: ``fabricated_receipts``, ``cross_month_copy`` and
``turnover_inflation``. All three insert credit receipts (``insert_row`` edits) and
re-render; the printed balances follow the new rows (``recompute_balances``).

Label conventions: ``method == "content"``; one ``affected_rows`` entry per inserted
row; an entry for a copy also has ``"copies_row_id"`` (the genuine receipt copied).
"""
from __future__ import annotations

import calendar
import random
import re
from datetime import date

from fraud_detect.corpus.schema import EditOp, Ledger, LedgerRow, TamperLabel, TamperResult
from fraud_detect.corpus.tamper._registry import info_for, register
from fraud_detect.corpus.tamper.ledger_ops import (
    _month_key, _row_json, apply_edit, apply_edits, is_arithmetic_consistent)

_INV = re.compile(r"INV(\d+)")


def _weekdays(first: date) -> list[date]:
    n = calendar.monthrange(first.year, first.month)[1]
    days = [first.replace(day=d) for d in range(1, n + 1)]
    return [d for d in days if d.weekday() < 5] or days


def _receipts(ledger: Ledger) -> list[LedgerRow]:
    return [r for r in ledger.rows
            if r.kind == "receipt" and r.amount > 0 and r.invoice_ref and not r.fraud]


def _new_id(ledger: Ledger, counter: list[int]) -> str:
    used = {r.row_id for r in ledger.rows}
    while True:
        counter[0] += 1
        rid = f"F{counter[0]:04d}"
        if rid not in used:
            return rid


def _insert(ledger: Ledger, row: LedgerRow) -> tuple[Ledger, EditOp]:
    """Insert ``row`` after the last row dated on or before it (keeps date order)."""
    after = None
    for r in ledger.rows:
        if r.date <= row.date:
            after = r.row_id
    e = EditOp("insert_row", row.row_id, None, None, _row_json(row, after))
    return apply_edit(ledger, e), e


def _finish(genuine: Ledger, name: str, planned: list[tuple[LedgerRow, str | None]],
            notes: str) -> TamperResult:
    cur = genuine
    edits: list[EditOp] = []
    for row, _ in planned:
        cur, e = _insert(cur, row)
        edits.append(e)
    edits.append(EditOp("recompute_balances"))
    edited = apply_edits(genuine, edits)
    pos = {r.row_id: i + 1 for i, r in enumerate(edited.rows)}
    entries = []
    for row, src_id in planned:
        entry = {"row_id": row.row_id, "row_index": pos[row.row_id],
                 "month": _month_key(edited.months[row.month_index]),
                 "field": "row", "original": None, "new": f"{row.amount:.2f}"}
        if src_id is not None:
            entry["copies_row_id"] = src_id
        entries.append(entry)
    label = TamperLabel(
        is_tampered=True, tamper_types=(name,), method="content",
        arithmetic_consistent=is_arithmetic_consistent(edited),
        affected_rows=tuple(entries),
        affected_months=tuple(sorted({e["month"] for e in entries})),
        should_be_seen_by=info_for(name).should_be_seen_by, notes=notes)
    return TamperResult(ledger=edited, edits=tuple(edits), file_ops=(), label=label)


def _copy_row(ledger: Ledger, src: LedgerRow, month_index: int, new_date: date,
              counter: list[int], rng: random.Random) -> LedgerRow:
    amount = src.amount
    while abs(amount - src.amount) < 0.01:
        amount = round(src.amount * rng.uniform(0.5, 1.6) + rng.choice((0, 0.37)), 2)
    return LedgerRow(
        row_id=_new_id(ledger, counter), month_index=month_index, date=new_date,
        description=src.description, amount=amount, kind="receipt",
        customer=src.customer, invoice_ref=src.invoice_ref, fraud=True)


@register("fabricated_receipts")
def plan_fabricated_receipts(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    pool = _receipts(ledger)
    n = rng.randint(6, 12)
    sources = rng.sample(pool, n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]
    counter = [0]
    planned = []
    for src in sources:
        days = [d for d in _weekdays(ledger.months[src.month_index]) if d != src.date]
        planned.append((_copy_row(ledger, src, src.month_index, rng.choice(days), counter, rng),
                        src.row_id))
    return _finish(ledger, "fabricated_receipts", planned,
                   "Extra receipts copy a genuine receipt's customer and invoice number "
                   "in the same month.")


@register("cross_month_copy")
def plan_cross_month_copy(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    pool = _receipts(ledger)
    n = min(rng.randint(2, 5), len(pool))
    counter = [0]
    planned = []
    for src in rng.sample(pool, n):
        others = [m for m in range(len(ledger.months)) if m != src.month_index]
        m = rng.choice(others)
        planned.append((_copy_row(ledger, src, m, rng.choice(_weekdays(ledger.months[m])),
                                  counter, rng), src.row_id))
    return _finish(ledger, "cross_month_copy", planned,
                   "Receipt copies land in a different calendar month than the original.")


@register("turnover_inflation")
def plan_turnover_inflation(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    pool = _receipts(ledger)
    amounts = sorted(r.amount for r in pool)
    median = amounts[len(amounts) // 2]
    nums = [int(m.group(1)) for r in ledger.rows
            for m in [_INV.search(r.invoice_ref or "")] if m]
    next_num = max(nums, default=ledger.company.first_invoice_number) + 1
    n_months = len(ledger.months)
    months = sorted(rng.sample(range(n_months), rng.randint(2, min(4, n_months))))
    customers = [p.name for p in ledger.company.customers]
    counter = [0]
    planned = []
    for m in months:
        for _ in range(rng.randint(2, 4)):
            cust = rng.choice(customers)
            ref = f"INV{next_num}"
            next_num += 1
            planned.append((LedgerRow(
                row_id=_new_id(ledger, counter), month_index=m,
                date=rng.choice(_weekdays(ledger.months[m])),
                description=f"{rng.choice(('FAST FROM', 'PAYNOW FROM'))} {cust} {ref}",
                amount=round(median * rng.uniform(1.0, 3.0), 2), kind="receipt",
                customer=cust, invoice_ref=ref, fraud=True), None))
    return _finish(ledger, "turnover_inflation", planned,
                   "Extra receipts with new invoice numbers in two or more months.")
