"""Finishing step: fixed-output canvas, footer, style knobs and PDF metadata.

Everything here is deterministic for a given seed: reportlab runs in invariant
mode (fixed dates and document id) and the final metadata is written with pypdf
using dates derived from the statement period, never the wall clock.
"""
from __future__ import annotations

import calendar
import io
from datetime import date, datetime, timedelta
from typing import Optional

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

from fraud_detect.corpus.schema import GENERATOR_VERSION, Ledger, StyleKnobs
from fraud_detect.corpus.seeding import rng as make_rng

# Fictional institutions only (no real brand).
BANK_NAMES: tuple[str, ...] = (
    "NORTHERN TEST BANK",
    "HARBOUR LION BANK",
    "MERLION SAMPLE BANK",
    "ORCHARD SPECIMEN BANK",
    "KAMPONG DEMO BANK",
    "BAYFRONT MODEL BANK",
    "CASTLE ROCK SAMPLE BANK",
    "LANTERN COVE BANK",
)

PRODUCERS: tuple[str, ...] = (
    "ReportLab PDF Library",
    "Statement Engine 4.2",
    "BankDoc Composer",
    "LedgerPrint Service",
)

STREETS: tuple[str, ...] = (
    "1 Specimen Road", "22 Sample Avenue", "7 Demo Street", "15 Fictional Way",
    "3 Mock Crescent", "40 Example Link",
)

FOOTER_TEXT = "SPECIMEN - Fictional institution - Synthetic data, not a real bank document"
DOC_TITLE = "Synthetic specimen bank statement"
DOC_SUBJECT = "Synthetic data for testing only; not a real bank document"
CREATOR = f"synthetic-corpus-generator {GENERATOR_VERSION}"

FOOTER_FONT = "Helvetica"
FOOTER_SIZE = 6.5
FOOTER_BASELINE = 24.0


def make_style(master_seed: int, *parts: object) -> StyleKnobs:
    """Small random look-and-feel variations (see design: StyleKnobs)."""
    r = make_rng(master_seed, "style", *parts)
    street = r.choice(STREETS)
    return StyleKnobs(
        bank_name=r.choice(BANK_NAMES),
        address_lines=(street, "Singapore " + f"{r.randint(0, 999999):06d}"),
        producer=r.choice(PRODUCERS),
        creation_time=f"{r.randint(6, 20):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}",
        row_step=round(r.uniform(12.0, 12.8), 2),
        second_address_line=r.random() < 0.5,
    )


def creation_datetime(ledger: Ledger, style: StyleKnobs, seed: int) -> datetime:
    """Fixed creation moment: 1 to 3 days after the last statement month ends."""
    last = ledger.months[-1] if ledger.months else date(2026, 1, 1)
    end = date(last.year, last.month, calendar.monthrange(last.year, last.month)[1])
    day = end + timedelta(days=make_rng(seed, "created").randint(1, 3))
    try:
        hh, mm, ss = (int(x) for x in style.creation_time.split(":"))
    except ValueError:
        hh, mm, ss = 9, 0, 0
    return datetime(day.year, day.month, day.day, hh, mm, ss)


def pdf_date(moment: datetime) -> str:
    """PDF date string (UTC) such as ``D:20260203093000Z``."""
    return "D:" + moment.strftime("%Y%m%d%H%M%S") + "Z"


def new_canvas(buffer: io.BytesIO, pagesize: tuple[float, float], *,
               title: str = DOC_TITLE, subject: str = DOC_SUBJECT,
               style: Optional[StyleKnobs] = None) -> Canvas:
    """A reportlab canvas in fixed-output mode with synthetic Title and Subject."""
    c = Canvas(buffer, pagesize=pagesize, invariant=1, pageCompression=0)
    c.setTitle(title)
    c.setSubject(subject)
    c.setAuthor(style.bank_name if style else "")
    c.setCreator((style.creator if style else "") or CREATOR)
    return c


def draw_footer(c: Canvas, page_width: float, *, margin: float = 40.0,
                font: str = FOOTER_FONT) -> None:
    """Specimen footer on the current page (built-in Helvetica by default)."""
    c.setFont(font, FOOTER_SIZE)
    c.drawString(margin, FOOTER_BASELINE, FOOTER_TEXT)


def draw_address(c: Canvas, x: float, y_baseline: float, style: StyleKnobs) -> None:
    """Print the bank address line (Balanced_Styling only); no-op otherwise."""
    if not style.print_address:
        return
    from fraud_detect.corpus import styling  # lazy: avoids an import cycle

    c.setFont(style.font_regular, 6.5)
    c.drawString(x, y_baseline, styling.address_text(style))


def finish_pdf(pdf: bytes, *, style: StyleKnobs, created: datetime,
               title: str = DOC_TITLE, subject: str = DOC_SUBJECT,
               modified: Optional[datetime] = None) -> bytes:
    """Rewrite the metadata with fixed values (Producer from the style, fixed dates).

    ``modified`` defaults to the creation moment, so a finished genuine document has
    ModDate equal to CreationDate.
    """
    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({
        "/Title": title,
        "/Subject": subject,
        "/Author": style.bank_name,
        "/Creator": style.creator or CREATOR,
        "/Producer": style.producer or "ReportLab PDF Library",
        "/CreationDate": pdf_date(created),
        "/ModDate": pdf_date(modified or created),
    })
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
