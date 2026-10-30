"""Section splitting and file-level roll-up (Requirement 8).

A page with a ``Statement Period`` line starts a new Section; a page without
one is a continuation of the previous Section. The printed page counter
("Page 1 of 1") is never consulted. The header is read from a Section's first
page only, so continuation pages cannot raise ``DQ_MISSING_PERIOD``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ...models import Finding, SectionSummary, StatementSegment
from .header import SectionHeader, missing_field_findings, read_header
from .types import PageContent

_PERIOD_LINE_RE = re.compile(r"statement[ \t]{1,5}period", re.IGNORECASE)


@dataclass
class Section:
    page_start: int
    page_end: int
    pages: list[PageContent]
    header: SectionHeader


@dataclass
class FileResult:
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    segments: list[StatementSegment] = field(default_factory=list)
    summaries: list[SectionSummary] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


def _has_period_line(page: PageContent) -> bool:
    return any(_PERIOD_LINE_RE.search(ln.text) for ln in page.lines)


def _make_section(pages: list[PageContent]) -> Section:
    first = pages[0]
    header = read_header(first.lines, first.words, None, first.page)
    return Section(page_start=first.page, page_end=pages[-1].page, pages=pages, header=header)


def split_into_sections(pages: list[PageContent]) -> list[Section]:
    groups: list[list[PageContent]] = []
    for page in pages:
        if not groups or _has_period_line(page):
            groups.append([page])
        else:
            groups[-1].append(page)
    return [_make_section(g) for g in groups]


def build_file_result(sections: list[Section]) -> FileResult:
    result = FileResult()
    if not sections:
        return result
    result.opening_balance = sections[0].header.opening
    result.closing_balance = sections[-1].header.closing
    for sec in sections:
        h = sec.header
        start = h.period_start.value if h.period_start is not None else None
        end = h.period_end.value if h.period_end is not None else None
        result.segments.append(StatementSegment(
            period_start=start, period_end=end,
            page_start=sec.page_start, page_end=sec.page_end,
        ))
        result.summaries.append(SectionSummary(
            period_start=start, period_end=end,
            page_start=sec.page_start, page_end=sec.page_end,
            opening=h.opening, closing=h.closing,
            total_deposits=h.printed_total_deposits,
            total_withdrawals=h.printed_total_withdrawals,
        ))
        result.findings.extend(missing_field_findings(h))
    return result
