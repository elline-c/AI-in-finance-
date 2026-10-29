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


def _grab(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def ingest(path: str) -> IngestionResult:
    dq: list[Finding] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        page_texts = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(pdf.pages)]
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

    # ---- header fields -----------------------------------------------------
    holder = _grab(r"Account Holder:\s*(.+)", raw_text)
    uen = _grab(r"UEN:\s*([A-Z0-9]+)", raw_text)
    account_number = _grab(r"Account No:\s*([0-9]+)", raw_text)
    currency = _grab(r"Currency:\s*([A-Za-z$]+)", raw_text)

    # ---- statement period segmentation ------------------------------------
    # Find EVERY "Statement Period:" header along with the page it sits on, so a
    # combined PDF (several monthly statements stapled together) is split into
    # one StatementSegment per header. We iterate page_texts (which preserves
    # page attribution) rather than raw_text (which joins pages and loses it).
    #
    # A segment spans from the page its header is on up to the page just before
    # the next header's page; the final segment runs to the last page. Month
    # identity for each segment comes solely from its own printed dates.
    raw_headers: list[tuple[int, Optional[date], Optional[date]]] = []
    for page_no, text in page_texts:
        for hm in _PERIOD_RE.finditer(text):
            h_start, _ = _parse_date(hm.group(1))
            h_end, _ = _parse_date(hm.group(2))
            raw_headers.append((page_no, h_start, h_end))

    source_name = os.path.basename(path)
    segments: list[StatementSegment] = []
    for i, (page_no, h_start, h_end) in enumerate(raw_headers):
        if i + 1 < len(raw_headers):
            next_page = raw_headers[i + 1][0]
            page_end = max(page_no, next_page - 1)
        else:
            page_end = page_count
        segments.append(StatementSegment(
            period_start=h_start,
            period_end=h_end,
            page_start=page_no,
            page_end=page_end,
            source=source_name,
        ))

    # Top-level period stays backward compatible: earliest start / latest end
    # across all segments (for a single-header PDF this is just that header).
    period_start = period_end = None
    starts = [s.period_start for s in segments if s.period_start is not None]
    ends = [s.period_end for s in segments if s.period_end is not None]
    if starts:
        period_start = min(starts)
    if ends:
        period_end = max(ends)

    opening_balance = _parse_money(_grab(r"Opening Balance:\s*([^\n]+)", raw_text) or "")
    closing_balance = _parse_money(_grab(r"Closing Balance:\s*([^\n]+)", raw_text) or "")

    # mandatory-field data-quality checks (explicit nulls, no guessing)
    for label, value, code in [
        ("account holder", holder, "DQ_MISSING_HOLDER"),
        ("UEN", uen, "DQ_MISSING_UEN"),
        ("statement period", period_start, "DQ_MISSING_PERIOD"),
        ("opening balance", opening_balance, "DQ_MISSING_OPENING"),
    ]:
        if value in (None, ""):
            dq.append(Finding(
                agent=AGENT, code=code, title=f"Missing mandatory field: {label}",
                severity=Severity.MEDIUM, weight=8, field=label,
                evidence=f"Could not read the {label} from the statement; flagged rather than inferred.",
            ))

    # ---- transactions ------------------------------------------------------
    transactions: list[Transaction] = []
    idx = 0
    for page_no, text in page_texts:
        for line in text.splitlines():
            m = _ROW_RE.match(line.strip())
            if not m:
                continue
            d, d_raw = _parse_date(m.group("date"))
            amt = _parse_money(m.group("amount"))
            bal = _parse_money(m.group("balance"))
            if amt is None or bal is None:
                dq.append(Finding(
                    agent=AGENT, code="DQ_UNREADABLE_ROW",
                    title="Unreadable transaction row", severity=Severity.LOW,
                    weight=3, page=page_no,
                    evidence=f"Row could not be fully parsed: {line!r}",
                ))
                continue
            transactions.append(Transaction(
                index=idx, date=d, date_raw=d_raw, description=m.group("desc").strip(),
                amount=amt, balance=bal, page=page_no,
            ))
            idx += 1

    if not transactions:
        dq.append(Finding(
            agent=AGENT, code="DQ_NO_TXNS", title="No transactions parsed",
            severity=Severity.HIGH, weight=15,
            evidence="No transaction rows could be parsed from the statement.",
        ))

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
