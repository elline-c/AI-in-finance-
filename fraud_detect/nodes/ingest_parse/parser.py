"""Parser entry point: ``parse_pages(pages) -> ParseResult`` (task 9.2).

Glues the components together: profile selection (``layouts``), section
splitting and header reading (``sections`` / ``header``), direction resolution
(``direction``) and the self-check (``selfcheck``). It never raises: an
unexpected error while reading a page becomes ``DQ_LAYOUT_UNRECOGNISED`` for
that page and the other pages still parse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ...models import (
    Finding,
    SectionSummary,
    Severity,
    StatementSegment,
    Transaction,
)
from .dates import parse_day_month, parse_full_date
from .direction import DIRECTION_UNCERTAIN, resolve_direction
from .header import AGENT, SectionHeader, missing_field_findings, read_header
from .layouts import choose_profile
# ``read_rows`` is imported into this namespace on purpose: rows are read inside
# ``layouts.parse_page``, and tests patch the name on ``table``, ``layouts`` and
# here to simulate a page-level failure.
from .table import RawRow, TableLayout, find_table, read_rows  # noqa: F401
from .sections import Section, build_file_result, split_into_sections
from .selfcheck import self_check
from .types import PageContent


@dataclass
class ParseResult:
    transactions: list[Transaction] = field(default_factory=list)
    holder: Optional[str] = None
    uen: Optional[str] = None
    account_number: Optional[str] = None
    currency: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    segments: list[StatementSegment] = field(default_factory=list)
    sections: list[SectionSummary] = field(default_factory=list)
    layout_profile: Optional[str] = None
    parse_coverage: Optional[float] = None
    findings: list[Finding] = field(default_factory=list)


def _no_text_finding() -> Finding:
    return Finding(
        agent=AGENT, code="DQ_NO_TEXT", title="No extractable text",
        severity=Severity.HIGH, weight=15,
        evidence="The document produced no extractable text; it may be a poor-quality "
                 "scan requiring OCR fallback / manual review.",
    )


def _page_problem(page: int, err: Exception) -> Finding:
    return Finding(
        agent=AGENT, code="DQ_LAYOUT_UNRECOGNISED", title="Statement layout not recognised",
        severity=Severity.MEDIUM, weight=8, field="layout", page=page,
        evidence=f"Layout not recognised on page {page}: an unexpected error occurred "
                 f"while reading it ({type(err).__name__}). Other pages were still read.",
    )


def _row_date(row: RawRow, header: SectionHeader) -> Optional[date]:
    text = row.date_text
    value = parse_full_date(text).value
    if value is not None:
        return value
    s = header.period_start.value if header.period_start else None
    e = header.period_end.value if header.period_end else None
    if s is not None and e is not None and s <= e:
        return parse_day_month(text, s, e).value
    return None


def _no_txns_finding() -> Finding:
    return Finding(
        agent=AGENT, code="DQ_NO_TXNS", title="No transactions parsed",
        severity=Severity.HIGH, weight=15,
        evidence="No transaction rows could be parsed from the statement.",
    )


def _read_section(
    sec: Section,
    rows: list[RawRow],
    layouts_by_page: dict[int, TableLayout],
    prev_balance: Optional[float],
    start_index: int,
    findings: list[Finding],
) -> tuple[list[Transaction], Optional[float]]:
    header = sec.header
    prev = header.opening if header.opening is not None else prev_balance
    txns: list[Transaction] = []
    for row in rows:
        try:
            layout = layouts_by_page.get(row.page) or TableLayout(0.0, {}, "unknown")
            resolved = resolve_direction(row, layout, prev)
            if resolved is None:
                findings.append(Finding(
                    agent=AGENT, code="DQ_UNREADABLE_ROW",
                    title="Unreadable transaction row", severity=Severity.LOW,
                    weight=3, page=row.page,
                    evidence=f"Row could not be fully parsed: {row.line_text!r}",
                ))
            else:
                if resolved.finding_code == DIRECTION_UNCERTAIN:
                    findings.append(Finding(
                        agent=AGENT, code=DIRECTION_UNCERTAIN,
                        title="Credit/debit direction uncertain", severity=Severity.LOW,
                        weight=3, page=row.page,
                        evidence=f"The direction read for this row disagrees with the balance "
                                 f"change ({row.line_text!r}); the printed amount is kept.",
                    ))
                txns.append(Transaction(
                    index=start_index + len(txns),
                    date=_row_date(row, header),
                    date_raw=row.date_text,
                    description=row.description.strip(),
                    amount=resolved.amount,
                    balance=row.balance.value,
                    page=row.page,
                    value_date_raw=row.value_date_text,
                    direction_method=resolved.method,
                ))
            prev = row.balance.value
        except Exception as err:  # isolate: one bad row/page never stops the rest
            findings.append(_page_problem(row.page, err))
    return txns, prev


def _fill_unprinted_summary(summary: Optional[SectionSummary], txns: list[Transaction]) -> None:
    """Fill SectionSummary values the Section does not print, from its parsed rows.

    Only the summary is filled: the printed totals used by the self-check stay
    ``None``, so a derived value is never compared with itself.
    """
    if summary is None or not txns:
        return
    if summary.closing is None and txns[-1].balance is not None:
        summary.closing = txns[-1].balance
    amounts = [t.amount for t in txns if t.amount is not None]
    if summary.total_withdrawals is None:
        summary.total_withdrawals = round(sum(-a for a in amounts if a < 0), 2)
    if summary.total_deposits is None:
        summary.total_deposits = round(sum(a for a in amounts if a > 0), 2)


def _fill_from_continuation(sec: Section) -> None:
    """Some layouts print the closing balance (or totals) on a later page of the
    same section; fill only the fields the first page did not give."""
    h = sec.header
    for page in sec.pages[1:]:
        if h.opening is not None and h.closing is not None \
                and h.printed_total_withdrawals is not None and h.printed_total_deposits is not None:
            return
        try:
            later = read_header(page.lines, page.words, None, page.page)
        except Exception:
            continue
        for attr in ("opening", "closing", "printed_total_withdrawals", "printed_total_deposits"):
            if getattr(h, attr) is None:
                setattr(h, attr, getattr(later, attr))


def _parse(pages: list[PageContent]) -> ParseResult:
    result = ParseResult()
    findings = result.findings

    if not any((p.lines or p.words) and p.text.strip() for p in pages):
        findings.append(_no_text_finding())

    sections = split_into_sections(pages)
    for sec in sections:
        _fill_from_continuation(sec)
    file_result = build_file_result(sections)
    headers = [s.header for s in sections]

    def first(attr: str):
        return next((getattr(h, attr) for h in headers if getattr(h, attr) is not None), None)

    result.holder = first("holder")
    result.uen = first("uen")
    result.account_number = first("account_number")
    result.currency = first("currency")
    result.opening_balance = file_result.opening_balance
    result.closing_balance = file_result.closing_balance
    starts = [h.period_start.value for h in headers if h.period_start and h.period_start.value]
    ends = [h.period_end.value for h in headers if h.period_end and h.period_end.value]
    result.period_start = min(starts) if starts else None
    result.period_end = max(ends) if ends else None
    # no printed period anywhere: keep the old "no segments" behaviour
    if starts or ends:
        result.segments = file_result.segments
    result.sections = file_result.summaries

    # mandatory-field findings are file level (same as before)
    file_header = SectionHeader()
    file_header.holder, file_header.uen = result.holder, result.uen
    file_header.opening = result.opening_balance
    if starts and headers:
        file_header.period_start = next(h.period_start for h in headers if h.period_start and h.period_start.value)
    findings.extend(missing_field_findings(file_header))

    # ---- profile selection and rows
    choice = choose_profile(pages)
    result.layout_profile = choice.profile.name if choice.profile else None
    result.parse_coverage = round(choice.coverage, 4)
    # a page with no text at all is DQ_NO_TEXT territory, not a layout problem
    blank = {p.page for p in pages if not p.text.strip() and not p.words}
    findings.extend(f for f in choice.findings if f.page not in blank)

    layouts_by_page: dict[int, TableLayout] = {}
    for p in pages:
        try:
            lay = find_table(p.words, choice.profile)
            if lay is not None:
                layouts_by_page[p.page] = lay
        except Exception as err:
            findings.append(_page_problem(p.page, err))

    prev: Optional[float] = None
    for i, sec in enumerate(sections):
        sec_rows = [r for r in choice.rows if sec.page_start <= r.page <= sec.page_end]
        txns, prev = _read_section(sec, sec_rows, layouts_by_page, prev,
                                   len(result.transactions), findings)
        result.transactions.extend(txns)
        try:
            findings.extend(self_check(sec.header, txns))
        except Exception as err:
            findings.append(_page_problem(sec.page_start, err))
        _fill_unprinted_summary(result.sections[i] if i < len(result.sections) else None, txns)

    # A layout that prints no closing balance (e.g. a generator page without the
    # closing line) takes the last printed running balance of the last Section.
    if result.closing_balance is None and result.sections and result.sections[-1].closing is not None:
        result.closing_balance = result.sections[-1].closing

    if not result.transactions:
        findings.append(_no_txns_finding())
    return result


def parse_pages(pages: list[PageContent]) -> ParseResult:
    """Parse already-extracted pages. Never raises."""
    try:
        return _parse(list(pages))
    except Exception as err:
        page = pages[0].page if pages else 1
        res = ParseResult()
        res.findings.append(_page_problem(page, err))
        res.findings.append(_no_txns_finding())
        return res
