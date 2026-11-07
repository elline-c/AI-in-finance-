"""Subtle amount change: ``subtle_amount_edit``.

One amount is changed by 1.0% to 4.9% (random sign). Eligible rows have an absolute
amount of at least 100.00; if none does, the row with the largest absolute amount is
used. The relative change is drawn up to 8 times (a small amount can round back to the
same value or leave the band); after that a ``ValueError`` names the row.

Method (overlay or re-render) and balance handling (recomputed or left as printed) come
from the tamper random stream, so the result is deterministic and replayable.
"""
from __future__ import annotations

import random

from fraud_detect.corpus.schema import EditOp, Ledger
from fraud_detect.corpus.tamper import ledger_ops, pdf_ops
from fraud_detect.corpus.tamper._registry import register

NAME = "subtle_amount_edit"
MIN_ELIGIBLE = 100.00
LOW_PCT, HIGH_PCT = 1.0, 4.9
MAX_DRAWS = 8


def _in_band(old: float, new: float) -> bool:
    rel = abs(new - old) / abs(old)
    return 0.01 <= rel <= 0.049 + 1e-9 and round(new, 2) != round(old, 2)


@register(NAME)
def plan_subtle_amount_edit(ledger: Ledger, renderer, rng: random.Random):
    rows = ledger.rows
    eligible = [r for r in rows if abs(r.amount) >= MIN_ELIGIBLE]
    if not eligible:
        eligible = [max(rows, key=lambda r: abs(r.amount))]
    row = eligible[rng.randrange(len(eligible))]

    new = None
    for _ in range(MAX_DRAWS):
        pct = rng.uniform(LOW_PCT, HIGH_PCT)
        sign = rng.choice((-1.0, 1.0))
        cand = round(row.amount * (1.0 + sign * pct / 100.0), 2)
        if row.amount != 0 and _in_band(row.amount, cand):
            new = cand
            break
    if new is None:
        raise ValueError(
            f"subtle_amount_edit: row {row.row_id} (amount {row.amount:.2f}) cannot be "
            f"changed by {LOW_PCT}% to {HIGH_PCT}%")

    recompute = rng.random() < 0.5
    use_overlay = rng.random() < 0.5
    old_s, new_s = ledger_ops._money(row.amount), ledger_ops._money(new)
    if use_overlay:
        text = pdf_ops.replace_number(
            pdf_ops.cell_text(ledger, renderer, row.row_id, "amount"), new)
        return pdf_ops.overlay_result(
            ledger, NAME, row.row_id, "amount", old_s, new_s, text, rng,
            notes="Amount overlaid by 1% to 5%; balances left as printed.")
    edits: list[EditOp] = []
    if not recompute:
        edits.append(EditOp("freeze_balances"))
    edits.append(EditOp("set_amount", row.row_id, "amount", old_s, new_s))
    if recompute:
        edits.append(EditOp("recompute_balances"))
    return ledger_ops._result(
        ledger, edits, NAME, row.row_id, "amount", old_s, new_s,
        notes="balances recomputed" if recompute else "balances left as printed")
