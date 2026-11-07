"""Renderer for the ``generator_v1`` layout: colon-label skeleton, one signed amount
column (``SGD-1,234.00``), ``dd/mm/yyyy`` dates, left-aligned text.

Each monthly section starts on a fresh page with its own ``Statement Period:`` line.
A month with more rows than fit continues on a page that carries only an account
line and a page marker (no period line), then the repeated table header.
Geometry follows the Node 1 test fixture ``generator_v1_fixture``. Clean-room:
reportlab built-in Helvetica only.
"""
from __future__ import annotations

import calendar
import io
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

from fraud_detect.corpus.finish import (
    creation_datetime,
    draw_address,
    draw_footer,
    finish_pdf,
    new_canvas,
)
from fraud_detect.corpus.layouts.base import (
    RowMapBuilder,
    cell_box,
    holder_name,
    printed_balances,
    rows_by_month,
    split_into_pages,
    text_width,
)
from fraud_detect.corpus.schema import (
    Ledger,
    RenderResult,
    SectionSummary,
    StyleKnobs,
)

_PAGE_W, _PAGE_H = A4
_FONT, _BOLD, _SIZE = "Helvetica", "Helvetica-Bold", 9
_X_DATE = 20 * mm
_X_DESC = 55 * mm
_X_AMOUNT_HDR = 120 * mm
_X_BAL_HDR = 160 * mm
_X_AMOUNT_RIGHT = 150 * mm
_X_BAL_RIGHT = 195 * mm
_MIN_ROW_Y = 62.0           # lowest baseline a table row may use
_DESC_MAX_W = 118.0         # keeps descriptions clear of the amount column


def _fit(text: str) -> str:
    while text and text_width(text, _FONT, _SIZE) > _DESC_MAX_W:
        text = text[:-1]
    return text.rstrip()


def _signed(amount: float) -> str:
    return f"SGD{'-' if amount < 0 else ''}{abs(amount):,.2f}"


class GeneratorV1Renderer:
    name = "generator_v1"
    max_rows_per_month = None

    def render(self, ledger: Ledger, style: StyleKnobs, seed: int) -> RenderResult:
        step = style.row_step if style.row_step > 0 else 14.0
        buf = io.BytesIO()
        c = new_canvas(buf, A4, style=style)
        builder = RowMapBuilder()
        balances = printed_balances(ledger)
        groups = rows_by_month(ledger)
        company = ledger.company
        title = f"{style.bank_name} | Corporate Statement"
        summaries: list[SectionSummary] = []
        spilled: list[str] = []
        n_pages = 0
        row_pos = 0
        opening = ledger.opening_balance

        for m_idx, first_day in enumerate(ledger.months):
            rows = groups[m_idx]
            last_day = date(first_day.year, first_day.month,
                            calendar.monthrange(first_day.year, first_day.month)[1])
            month_key = f"{first_day.year:04d}-{first_day.month:02d}"
            month_bal = balances[row_pos:row_pos + len(rows)]
            closing = month_bal[-1] if month_bal else opening

            first_start = self._header_end(True)
            cont_start = self._header_end(False)
            first_cap = max(1, int((first_start - _MIN_ROW_Y) // step) + 1)
            cont_cap = max(1, int((cont_start - _MIN_ROW_Y) // step) + 1)
            counts = split_into_pages(len(rows), first_cap, cont_cap)

            idx = 0
            y = _MIN_ROW_Y
            for p_in_month, count in enumerate(counts):
                page_no = p_in_month + 1
                n_pages += 1
                y = self._header(c, page_no, title, holder_name(ledger), company.uen,
                                 company.account_number, first_day, last_day, opening, style)
                for _ in range(count):
                    row = rows[idx]
                    bal = month_bal[idx]
                    idx += 1
                    desc = _fit(row.description)
                    date_t = row.date.strftime("%d/%m/%Y")
                    amt_t = _signed(row.amount)
                    bal_t = f"SGD{bal:,.2f}"
                    c.setFont(_FONT, _SIZE)
                    c.drawString(_X_DATE, y, date_t)
                    c.drawString(_X_DESC, y, desc)
                    c.drawRightString(_X_AMOUNT_RIGHT, y, amt_t)
                    c.drawRightString(_X_BAL_RIGHT, y, bal_t)
                    builder.add_row(row.row_id, {
                        "date": cell_box(n_pages, _PAGE_H, y, _SIZE, date_t, _FONT, x_left=_X_DATE),
                        "description": cell_box(n_pages, _PAGE_H, y, _SIZE, desc, _FONT, x_left=_X_DESC),
                        "amount": cell_box(n_pages, _PAGE_H, y, _SIZE, amt_t, _FONT, x_right=_X_AMOUNT_RIGHT),
                        "balance": cell_box(n_pages, _PAGE_H, y, _SIZE, bal_t, _FONT, x_right=_X_BAL_RIGHT),
                    }, bal)
                    y -= step
                if p_in_month == len(counts) - 1:
                    c.setFont(_BOLD, 10)
                    c.drawString(_X_DATE, y - 4 * mm, f"Closing Balance: SGD{closing:,.2f}")
                draw_footer(c, _PAGE_W, margin=_X_DATE, font=style.font_regular)
                c.showPage()

            builder.end_month(len(counts))
            if len(counts) > 1:
                spilled.append(month_key)
            summaries.append(SectionSummary(
                month=month_key,
                opening=opening,
                total_withdrawals=round(sum(-r.amount for r in rows if r.amount < 0), 2),
                total_deposits=round(sum(r.amount for r in rows if r.amount > 0), 2),
                closing=closing,
            ))
            opening = closing
            row_pos += len(rows)

        c.save()
        pdf = finish_pdf(buf.getvalue(), style=style,
                         created=creation_datetime(ledger, style, seed))
        return RenderResult(pdf=pdf, row_map=builder.row_map(), printed=builder.printed,
                            section_summaries=tuple(summaries), n_pages=n_pages,
                            spilled_months=tuple(spilled))

    # -- geometry -----------------------------------------------------------
    @staticmethod
    def _header_end(first_page: bool) -> float:
        """Baseline of the first table row on a first / continuation page."""
        y = _PAGE_H - 33 * mm
        y -= (7 if first_page else 2) * 5 * mm
        y -= 3 * mm
        return y - 6 * mm

    @staticmethod
    def _header(c, page_no: int, title: str, holder: str, uen: str, account: str,
                first_day: date, last_day: date, opening: float,
                style: StyleKnobs) -> float:
        c.setFont(_BOLD, 16)
        c.drawString(_X_DATE, _PAGE_H - 25 * mm, title)
        # Balanced_Styling only (no-op for style_version 1): bank address line between
        # the title (baseline 25 mm) and the first header line (33 mm).
        draw_address(c, _X_DATE, _PAGE_H - 29 * mm, style)
        c.setFont(_FONT, _SIZE)
        y = _PAGE_H - 33 * mm
        if page_no == 1:
            lines = [
                f"Account Holder: {holder}",
                f"UEN: {uen}",
                f"Account No: {account}",
                f"Statement Period: {first_day:%d/%m/%Y} - {last_day:%d/%m/%Y}",
                "Currency: SGD",
                f"Opening Balance: SGD{opening:,.2f}",
                f"Page {page_no}",
            ]
        else:
            lines = [f"Account No: {account}  (continued)", f"Page {page_no}"]
        for line in lines:
            c.drawString(_X_DATE, y, line)
            y -= 5 * mm
        y -= 3 * mm
        c.setFont(_BOLD, _SIZE)
        c.drawString(_X_DATE, y, "Date")
        c.drawString(_X_DESC, y, "Description")
        c.drawString(_X_AMOUNT_HDR, y, "Amount")
        c.drawString(_X_BAL_HDR, y, "Balance")
        return y - 6 * mm


RENDERER = GeneratorV1Renderer()
