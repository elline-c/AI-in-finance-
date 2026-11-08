"""Edits that change the ledger, their replay, and the three document tampers
(``edited_amount``, ``altered_date``, ``broken_balance``) done by re-rendering.

Edit log (``EditOp``; values are the printed strings)::

    set_amount          row_id, original/new "1234.50"
    set_date            row_id, original/new "2026-01-05"
    set_balance         row_id, original/new "1234.50" (printed balance after that row)
    freeze_balances     keep the printed balances as they are now (balance not redrawn)
    recompute_balances  printed balances follow the amounts again (replay)
    insert_row          new = JSON row fields (+ "after_row_id", null = at the start)
    delete_row          row_id, original = JSON row fields
    delete_month        field = month_index; the next sections keep their own balances

Edited rows get ``fraud=True`` (insert/set ops); ``apply_edits`` is deterministic, so
replaying a recorded list on the genuine ledger reproduces the edited ledger exactly.
"""
from __future__ import annotations

import calendar
import json
import random
from dataclasses import replace
from datetime import date, timedelta

from fraud_detect.corpus.ledger import replay
from fraud_detect.corpus.schema import EditOp, Ledger, LedgerRow, TamperLabel, TamperResult
from fraud_detect.corpus.tamper._registry import info_for, register

_ROW_FIELDS = ("month_index", "date", "description", "amount", "kind",
               "customer", "invoice_ref", "hard_case")


def _money(x: float) -> str:
    return f"{x:.2f}"


def _row_json(row: LedgerRow, after_row_id: str | None = None) -> str:
    d = {k: getattr(row, k) for k in _ROW_FIELDS}
    d["date"] = row.date.isoformat()
    d["row_id"] = row.row_id
    d["after_row_id"] = after_row_id
    return json.dumps(d, sort_keys=True)


def _row_from_json(text: str) -> tuple[LedgerRow, str | None]:
    d = json.loads(text)
    row = LedgerRow(row_id=d["row_id"], month_index=d["month_index"],
                    date=date.fromisoformat(d["date"]), description=d["description"],
                    amount=float(d["amount"]), kind=d["kind"], customer=d.get("customer"),
                    invoice_ref=d.get("invoice_ref"), hard_case=d.get("hard_case"), fraud=True)
    return row, d.get("after_row_id")


def _index(ledger: Ledger, row_id: str) -> int:
    for i, r in enumerate(ledger.rows):
        if r.row_id == row_id:
            return i
    raise KeyError(f"row {row_id!r} not in ledger")


def current_balances(ledger: Ledger) -> tuple[float, ...]:
    """Printed balances: the explicit ones, or the replay when none are stored."""
    if ledger.printed_balances is not None:
        return tuple(ledger.printed_balances)
    return tuple(replay(ledger))


def apply_edit(ledger: Ledger, e: EditOp) -> Ledger:
    """Apply one recorded edit and return a new ledger."""
    rows = list(ledger.rows)
    if e.op == "set_amount":
        i = _index(ledger, e.row_id)
        rows[i] = replace(rows[i], amount=round(float(e.new), 2), fraud=True)
        return replace(ledger, rows=tuple(rows))
    if e.op == "set_date":
        i = _index(ledger, e.row_id)
        rows[i] = replace(rows[i], date=date.fromisoformat(e.new), fraud=True)
        return replace(ledger, rows=tuple(rows))
    if e.op == "set_balance":
        i = _index(ledger, e.row_id)
        bals = list(current_balances(ledger))
        bals[i] = round(float(e.new), 2)
        rows[i] = replace(rows[i], fraud=True)
        return replace(ledger, rows=tuple(rows), printed_balances=tuple(bals))
    if e.op == "freeze_balances":
        return replace(ledger, printed_balances=current_balances(ledger))
    if e.op == "recompute_balances":
        return replace(ledger, printed_balances=None)
    if e.op == "insert_row":
        row, after = _row_from_json(e.new)
        pos = 0 if after is None else _index(ledger, after) + 1
        bals = None
        if ledger.printed_balances is not None:
            # keep earlier printed balances; the new row's balance follows the replay
            old = list(ledger.printed_balances)
            prev = old[pos - 1] if pos > 0 else ledger.opening_balance
            old.insert(pos, round(prev + row.amount, 2))
            bals = tuple(old)
        rows.insert(pos, row)
        return replace(ledger, rows=tuple(rows), printed_balances=bals)
    if e.op == "delete_row":
        i = _index(ledger, e.row_id)
        bals = None
        if ledger.printed_balances is not None:
            bals = tuple(b for k, b in enumerate(ledger.printed_balances) if k != i)
        del rows[i]
        return replace(ledger, rows=tuple(rows), printed_balances=bals)
    if e.op == "delete_month":
        m = int(e.field)
        # later sections keep the balances they had (they open at their own opening)
        bals = current_balances(ledger)
        keep = [k for k, r in enumerate(rows) if r.month_index != m]
        new_rows = tuple(replace(rows[k], month_index=rows[k].month_index - (rows[k].month_index > m))
                         for k in keep)
        months = tuple(x for k, x in enumerate(ledger.months) if k != m)
        return replace(ledger, rows=new_rows, months=months,
                       printed_balances=tuple(bals[k] for k in keep))
    raise ValueError(f"unknown edit op {e.op!r}")


def apply_edits(ledger: Ledger, edits) -> Ledger:
    """Replay a recorded edit list on ``ledger`` (the edit-log replay)."""
    for e in edits:
        ledger = apply_edit(ledger, e)
    return ledger


# ---------------------------------------------------------------------------
# helpers shared by the tampers
# ---------------------------------------------------------------------------
def is_arithmetic_consistent(ledger: Ledger) -> bool:
    if ledger.printed_balances is None:
        return True
    return all(abs(a - b) < 0.005 for a, b in zip(ledger.printed_balances, replay(ledger)))


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _affected(ledger: Ledger, row_id: str, field: str, original: str, new: str) -> dict:
    i = _index(ledger, row_id)
    row = ledger.rows[i]
    return {"row_id": row_id, "row_index": i + 1,
            "month": _month_key(ledger.months[row.month_index]),
            "field": field, "original": original, "new": new}


def _date_breaks(ledger: Ledger, row_id: str) -> bool:
    rows = ledger.rows
    row = rows[_index(ledger, row_id)]
    first = ledger.months[row.month_index]
    outside = (row.date.year, row.date.month) != (first.year, first.month)
    out_of_order = any(rows[k].date > rows[k + 1].date for k in range(len(rows) - 1))
    return outside or out_of_order


def _result(genuine: Ledger, edits: list[EditOp], name: str, row_id: str, field: str,
            original: str, new: str, *, date_inconsistent: bool = False,
            notes: str = "") -> TamperResult:
    edited = apply_edits(genuine, edits)
    info = info_for(name)
    label = TamperLabel(
        is_tampered=True, tamper_types=(name,), method="rerender",
        arithmetic_consistent=is_arithmetic_consistent(edited),
        date_inconsistent=date_inconsistent,
        affected_rows=(_affected(genuine, row_id, field, original, new),),
        affected_months=(_affected(genuine, row_id, field, original, new)["month"],),
        should_be_seen_by=info.should_be_seen_by, notes=notes)
    return TamperResult(ledger=edited, edits=tuple(edits), file_ops=(), label=label)


# ---------------------------------------------------------------------------
# the three document tampers (re-render variant)
# ---------------------------------------------------------------------------
@register("edited_amount")
def plan_edited_amount(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    row = ledger.rows[rng.randrange(len(ledger.rows))]
    recompute = rng.random() < 0.5
    factor = rng.choice((rng.uniform(1.2, 3.0), rng.uniform(0.2, 0.8)))
    new = round(row.amount * factor, 2)
    if abs(new - row.amount) < 0.01:
        new = round(row.amount + (1.0 if row.amount >= 0 else -1.0), 2)
    if rng.random() < 0.5:     # drawn last, so the re-render variant is unchanged
        from fraud_detect.corpus.tamper import pdf_ops
        text = pdf_ops.replace_number(
            pdf_ops.cell_text(ledger, renderer, row.row_id, "amount"), new)
        return pdf_ops.overlay_result(
            ledger, "edited_amount", row.row_id, "amount", _money(row.amount), _money(new),
            text, rng, notes="Amount overlaid; balances left as printed.")
    edits: list[EditOp] = []
    if not recompute:
        edits.append(EditOp("freeze_balances"))
    edits.append(EditOp("set_amount", row.row_id, "amount", _money(row.amount), _money(new)))
    if recompute:
        edits.append(EditOp("recompute_balances"))
    return _result(ledger, edits, "edited_amount", row.row_id, "amount",
                   _money(row.amount), _money(new),
                   notes="balances recomputed" if recompute else "balances left as printed")


def _month_bounds(d: date) -> tuple[date, date]:
    return d.replace(day=1), d.replace(day=calendar.monthrange(d.year, d.month)[1])


@register("altered_date")
def plan_altered_date(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    rows = ledger.rows
    want_break = rng.random() < 0.5
    order = list(range(len(rows)))
    rng.shuffle(order)

    def candidates(i: int) -> tuple[list[date], list[date]]:
        r = rows[i]
        lo, hi = _month_bounds(ledger.months[r.month_index])
        lo_b = max(lo, rows[i - 1].date) if i > 0 else lo
        hi_b = min(hi, rows[i + 1].date) if i + 1 < len(rows) else hi
        keep = [lo_b + timedelta(days=k) for k in range((hi_b - lo_b).days + 1)
                if lo_b + timedelta(days=k) != r.date] if lo_b <= hi_b else []
        breaking = [lo + timedelta(days=k) for k in range((hi - lo).days + 1)
                    if not (lo_b <= lo + timedelta(days=k) <= hi_b)]
        breaking += [lo - timedelta(days=rng.randint(1, 5)), hi + timedelta(days=rng.randint(1, 5))]
        return keep, breaking

    pick = None
    for i in order:
        keep, breaking = candidates(i)
        pool = breaking if want_break else keep
        if pool:
            pick = (i, rng.choice(pool))
            break
    if pick is None:   # no row has room to stay in order; fall back to a breaking move
        i = order[0]
        pick = (i, candidates(i)[1][-1])
    i, new_date = pick
    row = rows[i]
    edits = [EditOp("set_date", row.row_id, "date", row.date.isoformat(), new_date.isoformat())]
    if rng.random() < 0.5:
        from fraud_detect.corpus.tamper import pdf_ops
        return pdf_ops.overlay_result(
            ledger, "altered_date", row.row_id, "date", row.date.isoformat(),
            new_date.isoformat(), pdf_ops.format_date(renderer, new_date), rng,
            date_inconsistent=_date_breaks(apply_edits(ledger, edits), row.row_id),
            arithmetic_consistent=True, notes="Date overlaid.")
    return _result(ledger, edits, "altered_date", row.row_id, "date",
                   row.date.isoformat(), new_date.isoformat(),
                   date_inconsistent=_date_breaks(apply_edits(ledger, edits), row.row_id))


@register("broken_balance")
def plan_broken_balance(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    bals = current_balances(ledger)
    i = rng.randrange(len(ledger.rows))
    row = ledger.rows[i]
    delta = round(rng.choice((-1, 1)) * rng.uniform(10.0, 5000.0), 2)
    if abs(delta) < 0.01:
        delta = 10.0
    new = round(bals[i] + delta, 2)
    edits = [EditOp("set_balance", row.row_id, "balance", _money(bals[i]), _money(new))]
    # Method comes from the parity of the delta in cents (no extra rng draw), so the
    # re-render variant of every seed stays as it was before the overlay variant existed.
    if int(round(abs(delta) * 100)) % 2 == 1:
        from fraud_detect.corpus.tamper import pdf_ops
        shown = new if new >= 0 else round(bals[i] + abs(delta), 2)
        text = pdf_ops.replace_number(
            pdf_ops.cell_text(ledger, renderer, row.row_id, "balance"), shown)
        return pdf_ops.overlay_result(
            ledger, "broken_balance", row.row_id, "balance", _money(bals[i]), _money(shown),
            text, rng, notes="Printed balance overlaid.")
    return _result(ledger, edits, "broken_balance", row.row_id, "balance",
                   _money(bals[i]), _money(new))
