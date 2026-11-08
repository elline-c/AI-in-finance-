"""Direction resolver (Requirement 6, design Component 5).

Order of evidence: column position, printed sign, balance change. The magnitude
is always the printed amount, never recomputed from balances, so an edited
amount still surfaces as FL_BALANCE_BREAK in Node 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .table import AmountCell, RawRow, TableLayout

DIRECTION_UNCERTAIN = "DQ_DIRECTION_UNCERTAIN"
_TOL = 0.01


@dataclass(frozen=True)
class Resolved:
    amount: float  # signed: credits +, debits -
    method: Literal["column", "printed_sign", "balance_change"]
    agrees_with_balance: Optional[bool]  # None when there was no previous balance
    finding_code: Optional[str] = None


def _agrees(credit: bool, magnitude: float, delta: float) -> bool:
    if round(abs(delta), 2) < _TOL:
        # balance did not move: only consistent with a zero amount
        # (strict: a genuine one-cent movement is a movement)
        return round(magnitude, 2) < _TOL
    return (delta > 0) == credit


def _pick_cell(row: RawRow) -> Optional[AmountCell]:
    cells = row.amount_cells
    for c in cells:
        if c.column in ("money_out", "money_in"):
            return c
    for c in cells:
        if c.money.explicit_sign in (1, -1) or c.money.suffix:
            return c
    return cells[0] if cells else None


def resolve_direction(
    row: RawRow, layout: TableLayout, prev_balance: Optional[float]
) -> Optional[Resolved]:
    cell = _pick_cell(row)
    if cell is None:
        return None
    magnitude = abs(cell.money.value)
    delta: Optional[float] = None
    if prev_balance is not None and row.balance is not None:
        delta = row.balance.value - prev_balance

    credit: Optional[bool] = None
    method: Optional[str] = None

    if cell.column == "money_out":
        credit, method = False, "column"
    elif cell.column == "money_in":
        credit, method = True, "column"
    elif cell.money.explicit_sign in (1, -1):
        credit, method = cell.money.explicit_sign == 1, "printed_sign"
    elif cell.money.suffix in ("CR", "DR"):
        credit, method = cell.money.suffix == "CR", "printed_sign"
    elif cell.column == "amount" and layout.kind == "single_signed":
        # an unsigned amount in a single-signed table is read as a credit
        credit, method = True, "printed_sign"

    if method is not None:
        agrees = None if delta is None else _agrees(credit, magnitude, delta)
        return Resolved(
            amount=magnitude if credit else -magnitude,
            method=method,  # type: ignore[arg-type]
            agrees_with_balance=agrees,
            finding_code=DIRECTION_UNCERTAIN if agrees is False else None,
        )

    # balance change only
    if delta is None or round(abs(delta), 2) < _TOL:
        return None
    credit = delta > 0
    mismatch = round(abs(magnitude - abs(delta)), 2) > _TOL
    return Resolved(
        amount=magnitude if credit else -magnitude,
        method="balance_change",
        agrees_with_balance=not mismatch,
        finding_code=DIRECTION_UNCERTAIN if mismatch else None,
    )
