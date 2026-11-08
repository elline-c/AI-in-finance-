"""Data table for the five table layouts (column positions and wording, not code).

Geometry comes from the in-repo fixtures (``tests/ingest_parse/fixtures.py``).
Adding a layout means adding one :class:`TableSpec` entry to ``SPECS``; the drawing
engine in ``dataset_family.py`` reads every option from here.

Date patterns understand ``%d`` (2 digits), ``%e`` (day without zero), ``%m``,
``%Y``, ``%y`` and ``%b`` (English month abbreviation, locale independent).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HeaderWord:
    """One word of the table header: left-anchored at ``x`` or right-anchored at ``x``."""

    text: str
    anchor: str   # "l" (x is the left edge) or "r" (x is the right edge)
    x: float


@dataclass(frozen=True)
class TableSpec:
    name: str
    # dates
    row_date_fmt: str                 # date cell on each row
    period_date_fmt: str              # period / statement date lines in the header
    date_x0: float
    value_date_x0: Optional[float] = None   # second date column (transaction + value date)
    value_date_max_days: int = 2            # value date = date + 0..N days (from the seed)
    # description
    desc_x0: float = 100.0
    desc_max_x1: float = 340.0        # description text is shrunk to end before this
    # table header words
    header: tuple[HeaderWord, ...] = ()
    # money columns (right edges). Two columns: out_x1 / in_x1. One signed column: amount_x1.
    out_x1: Optional[float] = None
    in_x1: Optional[float] = None
    amount_x1: Optional[float] = None
    bal_x1: float = 555.0
    money_prefix: str = ""            # printed before every number, e.g. "S$" or "SGD "
    bal_suffix: str = ""              # e.g. "CR" after every balance cell (not in the summary box)
    bal_suffix_x0: float = 550.6
    summary_labels: tuple[str, str, str, str] = (
        "Opening Balance", "Total Withdrawals", "Total Deposits", "Closing Balance")
    # page furniture
    page_marker: str = "n_of_m"       # "n_of_m" -> "Page k of N"; "1_of_1" -> always "Page 1 of 1"
    first_page_rows: int = 43         # rows that fit on a section's first page
    cont_page_rows: int = 54          # rows that fit on a continuation page
    # volume
    max_rows_per_month: Optional[int] = None   # None = unlimited (month may spill)
    real_spill_seen: bool = True      # False = spill is an extension not seen in real samples


SPECS: dict[str, TableSpec] = {
    "two_col_slash": TableSpec(
        name="two_col_slash",
        row_date_fmt="%d/%m/%Y", period_date_fmt="%d/%m/%Y",
        date_x0=40.0, desc_x0=100.0, desc_max_x1=335.0,
        header=(
            HeaderWord("Date", "l", 40.0), HeaderWord("Description", "l", 100.0),
            HeaderWord("Withdrawals", "r", 400.0), HeaderWord("Deposits", "r", 475.0),
            HeaderWord("Balance", "r", 555.0),
        ),
        out_x1=400.0, in_x1=475.0, bal_x1=555.0,
        max_rows_per_month=None, real_spill_seen=True,
    ),
    # Every month stays on one page and prints "Page 1 of 1" (design D3). The cap keeps
    # the rows plus the brought-forward and closing lines within the first page.
    "two_col_dmon": TableSpec(
        name="two_col_dmon",
        row_date_fmt="%e %b", period_date_fmt="%d %b %Y",
        date_x0=38.0, desc_x0=90.0, desc_max_x1=335.0,
        header=(
            HeaderWord("Date", "l", 38.0), HeaderWord("Description", "l", 90.0),
            HeaderWord("Money out", "r", 392.0), HeaderWord("Money in", "r", 472.0),
            HeaderWord("Balance", "r", 557.0),
        ),
        out_x1=392.0, in_x1=472.0, bal_x1=557.0, money_prefix="S$",
        page_marker="1_of_1", first_page_rows=43,
        max_rows_per_month=40, real_spill_seen=False,
    ),
    # Transaction date and value date columns (value date = date + 0..2 days, from the seed).
    "two_col_dual_date": TableSpec(
        name="two_col_dual_date",
        row_date_fmt="%d %b %Y", period_date_fmt="%d %b %Y",
        date_x0=36.0, value_date_x0=104.0, value_date_max_days=2,
        desc_x0=172.0, desc_max_x1=392.0,
        header=(
            # Helvetica-Bold is wider than the fixture's font: the second word of each
            # date heading sits a little further right so pdfplumber keeps the words apart.
            HeaderWord("Txn", "l", 36.0), HeaderWord("Date", "l", 54.0),
            HeaderWord("Value", "l", 104.0), HeaderWord("Date", "l", 129.0),
            HeaderWord("Particulars", "l", 172.0),
            HeaderWord("Debit", "r", 420.0), HeaderWord("Credit", "r", 490.0),
            HeaderWord("Balance", "r", 559.0),
        ),
        out_x1=420.0, in_x1=490.0, bal_x1=559.0,
        max_rows_per_month=None, real_spill_seen=True,
    ),
    # dd-mm-yy dates; a separate "CR" word follows every balance cell (not in the summary box).
    "two_col_ddmmyy_cr": TableSpec(
        name="two_col_ddmmyy_cr",
        row_date_fmt="%d-%m-%y", period_date_fmt="%d/%m/%Y",
        date_x0=36.0, desc_x0=92.0, desc_max_x1=365.0,
        header=(
            HeaderWord("DATE", "l", 36.0), HeaderWord("DESCRIPTION", "l", 92.0),
            HeaderWord("DEBIT", "r", 395.0), HeaderWord("CREDIT", "r", 470.0),
            HeaderWord("BALANCE", "r", 560.0),
        ),
        out_x1=395.0, in_x1=470.0, bal_x1=546.0,
        bal_suffix="CR", bal_suffix_x0=550.6,
        max_rows_per_month=None, real_spill_seen=True,
    ),
    # One signed Amount column (+SGD credits, -SGD debits), ISO dates.
    "signed_amount_iso": TableSpec(
        name="signed_amount_iso",
        row_date_fmt="%Y-%m-%d", period_date_fmt="%Y-%m-%d",
        date_x0=40.0, desc_x0=110.0, desc_max_x1=340.0,
        header=(
            HeaderWord("Date", "l", 40.0), HeaderWord("Details", "l", 110.0),
            HeaderWord("Amount", "r", 450.0), HeaderWord("Balance", "r", 555.0),
        ),
        amount_x1=450.0, bal_x1=555.0, money_prefix="SGD ",
        summary_labels=("Opening Balance", "Total Debits", "Total Credits", "Closing Balance"),
        max_rows_per_month=None, real_spill_seen=True,
    ),
}
