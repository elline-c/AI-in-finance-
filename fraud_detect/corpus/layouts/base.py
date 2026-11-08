"""Renderer interface and small helpers shared by all layout renderers.

Conventions
-----------
* ``CellBox.page`` and ``PrintedRow.page`` are 1-based.
* ``CellBox.top`` is in pdfplumber's "top" style (distance from the page top to the
  top of the glyph box); ``x0`` / ``x1`` are the left and right edges of the text.
* Renderers set the PDF Title and Subject themselves (both say "synthetic").
* Only reportlab built-in fonts are used.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from reportlab.pdfbase.pdfmetrics import stringWidth

from fraud_detect.corpus.schema import (
    CellBox,
    Ledger,
    PrintedRow,
    RenderResult,
    RowMap,
    StyleKnobs,
)

CELL_NAMES: tuple[str, ...] = ("date", "description", "amount", "balance")

# Helvetica descent is -207/1000 em: the glyph box spans baseline-0.207*size
# up to baseline+0.793*size, which is what pdfminer/pdfplumber report.
_HELV_DESCENT = 0.207


@runtime_checkable
class LayoutRenderer(Protocol):
    """One renderer per layout name (see ``schema.LAYOUTS``)."""

    name: str
    max_rows_per_month: Optional[int]   # None = unlimited (month may spill)

    def render(self, ledger: Ledger, style: StyleKnobs, seed: int) -> RenderResult: ...


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def top_from_baseline(page_height: float, baseline_y: float, size: float) -> float:
    """pdfplumber-style ``top`` of text drawn with its baseline at ``baseline_y``."""
    return round(page_height - baseline_y - size * (1.0 - _HELV_DESCENT), 2)


def baseline_from_top(page_height: float, top: float, size: float) -> float:
    """Inverse of :func:`top_from_baseline` (reportlab y of the baseline)."""
    return page_height - top - size * (1.0 - _HELV_DESCENT)


def text_width(text: str, font: str, size: float) -> float:
    return stringWidth(text, font, size)


def cell_box(page: int, page_height: float, baseline_y: float, size: float,
             text: str, font: str, *, x_left: Optional[float] = None,
             x_right: Optional[float] = None) -> CellBox:
    """Build a :class:`CellBox` for text drawn left-aligned at ``x_left`` or
    right-aligned at ``x_right`` (exactly one must be given)."""
    if (x_left is None) == (x_right is None):
        raise ValueError("give exactly one of x_left / x_right")
    w = text_width(text, font, size)
    if x_left is None:
        x1 = float(x_right)
        x0 = x1 - w
    else:
        x0 = float(x_left)
        x1 = x0 + w
    return CellBox(page=page, top=top_from_baseline(page_height, baseline_y, size),
                   x0=round(x0, 2), x1=round(x1, 2), text=text)


# --------------------------------------------------------------------------
# row map accumulation
# --------------------------------------------------------------------------
class RowMapBuilder:
    """Collects cell boxes and printed rows while a renderer draws."""

    def __init__(self) -> None:
        self._cells: dict[str, dict[str, CellBox]] = {}
        self._printed: list[PrintedRow] = []
        self._pages_per_month: list[int] = []

    def add_row(self, row_id: str, cells: dict[str, CellBox], balance: float) -> None:
        missing = [n for n in CELL_NAMES if n not in cells]
        if missing:
            raise ValueError(f"row {row_id}: missing cells {missing}")
        if row_id in self._cells:
            raise ValueError(f"row {row_id} mapped twice")
        self._cells[row_id] = dict(cells)
        self._printed.append(PrintedRow(
            row_id=row_id,
            page=cells["date"].page,
            date_text=cells["date"].text,
            description_text=cells["description"].text,
            amount_text=cells["amount"].text,
            balance_text=cells["balance"].text,
            balance=balance,
        ))

    def end_month(self, pages_used: int) -> None:
        self._pages_per_month.append(int(pages_used))

    @property
    def printed(self) -> tuple[PrintedRow, ...]:
        return tuple(self._printed)

    def row_map(self) -> RowMap:
        return RowMap(cells=dict(self._cells), pages_per_month=tuple(self._pages_per_month))


# --------------------------------------------------------------------------
# paging helpers
# --------------------------------------------------------------------------
def split_into_pages(n_rows: int, first_capacity: int, continuation_capacity: int) -> list[int]:
    """Number of rows on each page for a month with ``n_rows`` rows.

    Always at least one page (an empty month still prints its section).
    """
    if first_capacity < 1 or continuation_capacity < 1:
        raise ValueError("page capacities must be positive")
    counts = [min(n_rows, first_capacity)]
    left = n_rows - counts[0]
    while left > 0:
        take = min(left, continuation_capacity)
        counts.append(take)
        left -= take
    return counts


def rows_by_month(ledger: Ledger) -> list[list]:
    """Ledger rows grouped by ``month_index`` in print order (one list per month)."""
    groups: list[list] = [[] for _ in ledger.months]
    for row in ledger.rows:
        groups[row.month_index].append(row)
    return groups


def printed_balances(ledger: Ledger) -> list[float]:
    """Balance printed after each row: ``printed_balances`` or the replay."""
    if ledger.printed_balances is not None:
        return list(ledger.printed_balances)
    bal, out = ledger.opening_balance, []
    for r in ledger.rows:
        bal = round(bal + r.amount, 2)
        out.append(bal)
    return out


def holder_name(ledger: Ledger) -> str:
    """Name printed as account holder: ``printed_holder`` when set, else the company."""
    return ledger.printed_holder or ledger.company.name


def fmt_money(value: float) -> str:
    """``1,234.50`` (two decimals, thousands separators, no sign)."""
    return f"{abs(value):,.2f}"
