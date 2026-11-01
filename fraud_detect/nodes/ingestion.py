"""Node 1 — Ingestion & Preprocessing Agent.

Responsibilities:
    * Open the PDF, extract text page-by-page (OCR stand-in for digital PDFs).
    * Extract file metadata (author / creator / producer / dates).
    * Capture a per-page font fingerprint (for the visual-forensics agent).
    * Parse the statement header (holder, UEN, account no, period, balances).
    * Parse the transaction table into a structured, typed list.
    * Emit explicit data-quality findings for missing / unreadable fields
      instead of guessing.

The output is a strict, validated ``IngestionResult`` (the JSON contract).
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Optional

import pdfplumber
from pypdf import PdfReader

from ..models import (
    DocumentMetadata,
    Finding,
    FontUsage,
    IngestionResult,
    PageText,
    Severity,
    StatementSegment,
    Transaction,
)

from .ingest_parse.parser import parse_pages
from .ingest_parse.table import _group_lines
from .ingest_parse.types import Line, PageContent, Word

AGENT = "Node1:Ingestion"

# Matches every "Statement Period:" header; used with finditer so a combined
# PDF with several monthly sections is split into one segment per header.
_PERIOD_RE = re.compile(
    r"Statement Period:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})"
)

# A row looks like: "04/01/2025 M1 Limited SGD-2,945.32 SGD52,054.68"
# amount/balance may use SGD or S$ prefix and optional leading + / -.
_MONEY = r"(?:SGD|S\$|\$)?\s*[+-]?[\d,]+\.\d{2}"
_ROW_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<amount>[+-]?(?:SGD|S\$|\$)?[+-]?[\d,]+\.\d{2})\s+"
    r"(?P<balance>(?:SGD|S\$|\$)?[\d,]+\.\d{2})\s*$"
)


def _parse_money(raw: str) -> Optional[float]:
    if raw is None:
        return None
    cleaned = raw.replace("SGD", "").replace("S$", "").replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(raw: str) -> tuple[Optional[date], str]:
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date(), raw
        except ValueError:
            continue
    return None, raw


def _extract_metadata(path: str) -> DocumentMetadata:
    try:
        reader = PdfReader(path)
        m = reader.metadata or {}
        return DocumentMetadata(
            author=m.get("/Author"),
            creator=m.get("/Creator"),
            producer=m.get("/Producer"),
            creation_date=m.get("/CreationDate"),
            mod_date=m.get("/ModDate"),
            title=m.get("/Title"),
        )
    except Exception:
        return DocumentMetadata()


def _extract_fonts(pdf: "pdfplumber.PDF") -> list[FontUsage]:
    usage: list[FontUsage] = []
    for i, page in enumerate(pdf.pages, start=1):
        counts: dict[str, int] = {}
        for ch in page.chars:
            name = ch.get("fontname", "unknown")
            counts[name] = counts.get(name, 0) + 1
        usage.append(FontUsage(page=i, fonts=counts))
    return usage


def _page_content(page_no: int, page: "pdfplumber.page.Page") -> PageContent:
    """Word boxes and grouped text lines for one page (for the layout-tolerant parser)."""
    words = [
        Word(text=w["text"], x0=float(w["x0"]), x1=float(w["x1"]), top=float(w["top"]))
        for w in page.extract_words(keep_blank_chars=False, use_text_flow=False)
    ]
    lines = [Line(text=" ".join(w.text for w in grp), top=grp[0].top) for grp in _group_lines(words)]
    return PageContent(page=page_no, lines=lines, words=words)


def ingest(path: str) -> IngestionResult:
    dq: list[Finding] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        page_texts = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(pdf.pages)]
        page_contents = [_page_content(i + 1, p) for i, p in enumerate(pdf.pages)]
        fonts = _extract_fonts(pdf)

    raw_text = "\n".join(t for _, t in page_texts)
    metadata = _extract_metadata(path)

    if not raw_text.strip():
        dq.append(Finding(
            agent=AGENT, code="DQ_NO_TEXT", title="No extractable text",
            severity=Severity.HIGH, weight=15,
            evidence="The document produced no extractable text; it may be a poor-quality "
                     "scan requiring OCR fallback / manual review.",
        ))

    # ---- layout-tolerant parse (header, sections, rows, self-check) --------
    parsed = parse_pages(page_contents)
    source_name = os.path.basename(path)
    segments = [seg.model_copy(update={"source": source_name}) for seg in parsed.segments]
    # DQ_NO_TEXT is decided above from the extracted text; skip the parser's copy.
    dq.extend(f for f in parsed.findings if f.code != "DQ_NO_TEXT")

    holder, uen = parsed.holder, parsed.uen
    account_number, currency = parsed.account_number, parsed.currency
    period_start, period_end = parsed.period_start, parsed.period_end
    opening_balance, closing_balance = parsed.opening_balance, parsed.closing_balance
    transactions = parsed.transactions

    return IngestionResult(
        source_path=path,
        page_count=page_count,
        raw_text=raw_text,
        page_texts=[PageText(page=pno, text=txt) for pno, txt in page_texts],
        metadata=metadata,
        fonts=fonts,
        segments=segments,
        account_holder=holder,
        uen=uen,
        account_number=account_number,
        statement_period_start=period_start,
        statement_period_end=period_end,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        declared_currency=currency,
        transactions=transactions,
        layout_profile=parsed.layout_profile,
        parse_coverage=parsed.parse_coverage,
        sections=parsed.sections,
        data_quality_findings=dq,
    )


def ingest_many(paths: list[str]) -> list[IngestionResult]:
    """Ingest a SET of statement files submitted as one application.

    Additive convenience over ``ingest(path)``: it simply ingests each path and
    returns the results. The returned list follows the input order purely as a
    convenience — month identity is carried inside each result's parsed period
    dates (and segments), NOT by list position, so screening the same files in
    any order yields the same month identification and coverage findings.
    """
    return [ingest(p) for p in paths]
