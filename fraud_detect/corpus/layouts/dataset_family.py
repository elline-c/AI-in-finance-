"""One drawing engine for the table layouts, driven by a :class:`TableSpec`.

Skeleton per monthly section: title, bank, page marker, holder block, summary box,
table header, brought-forward row, transaction rows, closing row, specimen footer.
A month with more rows than fit continues on further pages that repeat the table
header and print no period line and no brought-forward row.

Only reportlab built-in fonts are used. All positions are written in pdfplumber
"top" terms, so the row map is exact by construction.
"""
from __future__ import annotations

import io
import re
from datetime import date, timedelta
from typing import Optional

from reportlab.lib.pagesizes import A4

from fraud_detect.corpus.finish import creation_datetime, draw_footer, finish_pdf, new_canvas
from fraud_detect.corpus.layouts.base import (
    RowMapBuilder,
    baseline_from_top,
    cell_box,
    holder_name,
    printed_balances,
    rows_by_month,
    split_into_pages,
    text_width,
)
from fraud_detect.corpus.layouts.specs import SPECS, TableSpec
from fraud_detect.corpus.schema import CellBox, Ledger, RenderResult, SectionSummary, StyleKnobs
from fraud_detect.corpus.seeding import rng as make_rng

PAGE_W, PAGE_H = A4
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
SIZE = 8.0
MIN_DESC_SIZE = 6.0

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_TOKEN = re.compile(r"%([dembYy])")

# Fixture geometry (tests/ingest_parse/fixtures.py)
_SUMMARY_X0 = (40.0, 170.8, 301.6, 432.5)
_SUMMARY_X1 = (160.8, 291.6, 422.5, 553.3)
_TOP_BF, _TOP_ROWS_1 = 199.0, 212.4          # first page: brought-forward row, first txn row
_TOP_HEADER_1, _TOP_HEADER_N, _TOP_ROWS_N = 183.0, 73.0, 89.0
_RIGHT_COL_X = 300.0


def fmt_date(d: date, pattern: str) -> str:
    def sub(m: re.Match) -> str:
        t = m.group(1)
        return {"d": f"{d.day:02d}", "e": str(d.day), "m": f"{d.month:02d}",
                "b": _MONTHS[d.month - 1], "Y": f"{d.year:04d}", "y": f"{d.year % 100:02d}"}[t]
    return _TOKEN.sub(sub, pattern)


def _num(v: float) -> str:
    return f"{abs(v):,.2f}"


class TableRenderer:
    """A :class:`LayoutRenderer` for one :class:`TableSpec`."""

    def __init__(self, spec: TableSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.max_rows_per_month: Optional[int] = spec.max_rows_per_month

    # ---- text helpers ------------------------------------------------------
    def _money(self, v: float, *, sign: bool = False) -> str:
        s = self.spec.money_prefix + _num(v)
        if sign:
            return ("+" if v > 0 else "-") + s
        return "-" + s if v < 0 else s

    # ---- drawing primitives ------------------------------------------------
    @staticmethod
    def _put(c, page: int, top: float, text: str, *, x_left: Optional[float] = None,
             x_right: Optional[float] = None, font: str = FONT, size: float = SIZE) -> CellBox:
        base = baseline_from_top(PAGE_H, top, size)
        c.setFont(font, size)
        if x_left is not None:
            c.drawString(x_left, base, text)
        else:
            c.drawRightString(x_right, base, text)
        return cell_box(page, PAGE_H, base, size, text, font, x_left=x_left, x_right=x_right)

    def _table_header(self, c, page: int, top: float) -> None:
        for w in self.spec.header:
            if w.anchor == "l":
                self._put(c, page, top, w.text, x_left=w.x, font=FONT_BOLD)
            else:
                self._put(c, page, top, w.text, x_right=w.x, font=FONT_BOLD)

    def _balance_cell(self, c, page: int, top: float, value: float) -> CellBox:
        spec = self.spec
        box = self._put(c, page, top, self._money(value), x_right=spec.bal_x1)
        if spec.bal_suffix:
            self._put(c, page, top, spec.bal_suffix, x_left=spec.bal_suffix_x0)
        return box

    def _page_head(self, c, page: int, text: str, bank: str) -> None:
        self._put(c, page, 30, "STATEMENT OF ACCOUNT", x_left=40.0, font=FONT_BOLD, size=10.0)
        self._put(c, page, 44, bank, x_left=40.0)
        self._put(c, page, 56, text, x_left=40.0)
        self._put(c, page, 66, "SPECIMEN", x_left=40.0)

    def _first_page_block(self, c, page: int, ledger: Ledger, style: StyleKnobs, month: date,
                          marker: str, summary: SectionSummary) -> None:
        spec = self.spec
        self._page_head(c, page, marker, style.bank_name)
        last = date(month.year, month.month, 1)
        nxt = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        end = nxt - timedelta(days=1)
        pf = spec.period_date_fmt
        left = [holder_name(ledger), *ledger.company.address_lines, "BUSINESS CURRENT ACCOUNT"]
        right = [
            f"Account No. {ledger.company.account_number}",
            f"Company UEN {ledger.company.uen}",
            f"Statement Period {fmt_date(last, pf)} to {fmt_date(end, pf)}",
            f"Statement Date {fmt_date(nxt, pf)}",
            "Currency SGD",
        ]
        for i in range(max(len(left), len(right))):
            top = 78.0 + 12.0 * i
            if i < len(left):
                self._put(c, page, top, left[i], x_left=40.0)
            if i < len(right):
                self._put(c, page, top, right[i], x_left=_RIGHT_COL_X)
        vals = (summary.opening, summary.total_withdrawals, summary.total_deposits, summary.closing)
        for lab, x0 in zip(spec.summary_labels, _SUMMARY_X0):
            self._put(c, page, 142.0, lab, x_left=x0)
        for v, x1 in zip(vals, _SUMMARY_X1):
            self._put(c, page, 154.0, self._money(v), x_right=x1)

    def _fit_description(self, text: str) -> tuple[float, str]:
        """Font size (shrunk if needed) and text that end before ``desc_max_x1``."""
        spec = self.spec
        room = spec.desc_max_x1 - spec.desc_x0
        size = SIZE
        while size > MIN_DESC_SIZE and text_width(text, FONT, size) > room:
            size = round(size - 0.25, 2)
        while len(text) > 1 and text_width(text, FONT, size) > room:
            text = text[:-1].rstrip()
        return size, text

    def _txn_row(self, c, page: int, top: float, row, balance: float, seed: int) -> dict[str, CellBox]:
        spec = self.spec
        cells: dict[str, CellBox] = {}
        cells["date"] = self._put(c, page, top, fmt_date(row.date, spec.row_date_fmt), x_left=spec.date_x0)
        if spec.value_date_x0 is not None:
            days = make_rng(seed, "value_date", row.row_id).randint(0, spec.value_date_max_days)
            vd = row.date + timedelta(days=days)
            self._put(c, page, top, fmt_date(vd, spec.row_date_fmt), x_left=spec.value_date_x0)
        size, desc = self._fit_description(row.description)
        cells["description"] = self._put(c, page, top, desc, x_left=spec.desc_x0, size=size)
        if spec.amount_x1 is not None:
            cells["amount"] = self._put(c, page, top, self._money(row.amount, sign=True),
                                        x_right=spec.amount_x1)
        else:
            x1 = spec.in_x1 if row.amount > 0 else spec.out_x1
            cells["amount"] = self._put(c, page, top, self._money(abs(row.amount)), x_right=x1)
        cells["balance"] = self._balance_cell(c, page, top, balance)
        return cells

    # ---- main entry point ---------------------------------------------------
    def render(self, ledger: Ledger, style: StyleKnobs, seed: int) -> RenderResult:
        spec = self.spec
        step = style.row_step if style.row_step and style.row_step > 0 else 12.4
        buf = io.BytesIO()
        c = new_canvas(buf, A4, style=style)
        rb = RowMapBuilder()
        balances = dict(zip((r.row_id for r in ledger.rows), printed_balances(ledger)))
        summaries: list[SectionSummary] = []
        spilled: list[str] = []
        page = 0
        prev_close = ledger.opening_balance

        for mi, (month, rows) in enumerate(zip(ledger.months, rows_by_month(ledger))):
            opening = prev_close
            closing = balances[rows[-1].row_id] if rows else opening
            deposits = round(sum(r.amount for r in rows if r.amount > 0), 2)
            withdrawals = round(-sum(r.amount for r in rows if r.amount < 0), 2)
            label = f"{month.year:04d}-{month.month:02d}"
            summary = SectionSummary(month=label, opening=opening, total_withdrawals=withdrawals,
                                     total_deposits=deposits, closing=closing)
            summaries.append(summary)

            counts = split_into_pages(len(rows), spec.first_page_rows, spec.cont_page_rows)
            n_pages = len(counts)
            if n_pages > 1:
                spilled.append(label)
            pos = 0
            for k, n in enumerate(counts):
                page += 1
                shown_k = 1 if spec.page_marker == "1_of_1" else k + 1
                shown_n = 1 if spec.page_marker == "1_of_1" else n_pages
                marker = f"Page {shown_k} of {shown_n}"
                if k == 0:
                    self._first_page_block(c, page, ledger, style, month, marker, summary)
                    self._table_header(c, page, _TOP_HEADER_1)
                    self._put(c, page, _TOP_BF, "BALANCE BROUGHT FORWARD", x_left=spec.desc_x0)
                    self._balance_cell(c, page, _TOP_BF, opening)
                    top = _TOP_ROWS_1
                else:
                    self._page_head(c, page, marker, style.bank_name)
                    self._table_header(c, page, _TOP_HEADER_N)
                    top = _TOP_ROWS_N
                for row in rows[pos:pos + n]:
                    cells = self._txn_row(c, page, round(top, 2), row, balances[row.row_id], seed)
                    rb.add_row(row.row_id, cells, balances[row.row_id])
                    top += step
                pos += n
                if k == n_pages - 1:
                    self._put(c, page, round(top, 2), "CLOSING BALANCE", x_left=spec.desc_x0)
                    self._balance_cell(c, page, round(top, 2), closing)
                draw_footer(c, PAGE_W)
                c.showPage()
            rb.end_month(n_pages)
            prev_close = closing

        c.save()
        pdf = finish_pdf(buf.getvalue(), style=style, created=creation_datetime(ledger, style, seed))
        return RenderResult(pdf=pdf, row_map=rb.row_map(), printed=rb.printed,
                            section_summaries=tuple(summaries), n_pages=page,
                            spilled_months=tuple(spilled))


TABLE_RENDERERS: dict[str, TableRenderer] = {name: TableRenderer(s) for name, s in SPECS.items()}
