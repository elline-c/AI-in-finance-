"""Tests for section splitting and file-level balances (Requirement 8).

Written ahead of the implementation (task 8.4); they fail with
ModuleNotFoundError until ``fraud_detect/nodes/ingest_parse/sections.py``
exists.

API these tests pin down (module ``sections``):

``split_into_sections(pages: list[PageContent]) -> list[Section]``
    Walks the pages in order. A page containing a ``Statement Period`` line
    starts a new Section; a page without one is a Continuation_Page of the
    previous Section. The printed page counter ("Page 1 of 1") is ignored.
    A first page with no period still opens a Section (with no period).
    Each ``Section`` has:
      - ``page_start`` / ``page_end``: 1-based page numbers (inclusive)
      - ``pages``: the ``PageContent`` objects it spans
      - ``header``: a ``SectionHeader`` read from the section's first page
        (the header page); period, opening, closing, printed totals live here.

``build_file_result(sections: list[Section]) -> FileResult``
    File-level roll-up. ``FileResult`` has:
      - ``opening_balance``: first Section's opening (None if no sections)
      - ``closing_balance``: LAST Section's closing (None if no sections)
      - ``segments``: one ``StatementSegment`` per Section
      - ``summaries``: one ``SectionSummary`` per Section
      - ``findings``: header findings (e.g. ``DQ_MISSING_PERIOD``) for all
        Sections, in section order
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from fraud_detect.models import SectionSummary, StatementSegment
from fraud_detect.nodes.ingest_parse.sections import build_file_result, split_into_sections
from fraud_detect.nodes.ingest_parse.types import Line, PageContent

from .fixtures import continuation_fixture, six_page_page1of1_fixture, two_col_slash_fixture

TOL = 0.01


def _strip_period(page: PageContent) -> PageContent:
    """Copy of a page with its 'Statement Period' line (and words) removed."""
    lines = [ln for ln in page.lines if "statement period" not in ln.text.lower()]
    tops = {ln.top for ln in page.lines if "statement period" in ln.text.lower()}
    words = [w for w in page.words if w.top not in tops]
    return PageContent(page=page.page, lines=lines, words=words)


def _set_counter(page: PageContent, text: str) -> PageContent:
    lines = [
        replace(ln, text=text) if ln.text.lower().startswith("page ") and " of " in ln.text.lower() else ln
        for ln in page.lines
    ]
    return PageContent(page=page.page, lines=lines, words=page.words)


# --------------------------------------------------------------------------
# 8.1 / 8.2 section starts and continuation pages
# --------------------------------------------------------------------------
def test_single_page_file_is_one_section():
    fx = two_col_slash_fixture()
    sections = split_into_sections(fx.pages)
    assert len(sections) == 1
    assert (sections[0].page_start, sections[0].page_end) == (1, 1)


def test_page_without_statement_period_continues_previous_section():
    fx = continuation_fixture()
    assert len(fx.pages) == 2
    sections = split_into_sections(fx.pages)
    assert len(sections) == 1
    sec = sections[0]
    assert (sec.page_start, sec.page_end) == (1, 2)
    assert [p.page for p in sec.pages] == [1, 2]
    # header is read from page 1, which carries the period
    exp = fx.sections[0]
    assert sec.header.period_start.value == exp.period_start
    assert sec.header.period_end.value == exp.period_end
    assert sec.header.opening == pytest.approx(exp.opening, abs=TOL)
    assert sec.header.closing == pytest.approx(exp.closing, abs=TOL)


def test_new_section_starts_at_each_statement_period_line():
    pages = six_page_page1of1_fixture().pages
    sections = split_into_sections(pages)
    assert [(s.page_start, s.page_end) for s in sections] == [(i, i) for i in range(1, 7)]


def test_mixed_file_continuation_then_new_period():
    """Month 1 runs over two pages, month 2 starts on page 3."""
    from .fixtures import _make  # in-module builder: layout 1 over two months

    fx = _make("two_col_slash", rows_first_page=4, months=((2026, 7), (2026, 8)))
    sections = split_into_sections(fx.pages)
    assert len(fx.pages) == 4
    assert [(s.page_start, s.page_end) for s in sections] == [(1, 2), (3, 4)]
    assert sections[0].header.period_start.value == date(2026, 7, 1)
    assert sections[1].header.period_start.value == date(2026, 8, 1)


# --------------------------------------------------------------------------
# 8.3 printed page counter is ignored
# --------------------------------------------------------------------------
def test_page_1_of_1_on_every_page_is_ignored_for_section_starts():
    pages = six_page_page1of1_fixture().pages
    assert all(any("Page 1 of 1" in ln.text for ln in p.lines) for p in pages)
    assert len(split_into_sections(pages)) == 6


def test_page_counter_wording_does_not_change_sections():
    """Same pages with counters rewritten (honest or not) give the same split."""
    pages = six_page_page1of1_fixture().pages
    for text in ("Page 1 of 6", "Page 3 of 3", "Page 1"):
        changed = [_set_counter(p, text) for p in pages]
        assert len(split_into_sections(changed)) == 6


def test_continuation_page_labelled_page_1_of_1_is_still_continuation():
    fx = continuation_fixture()
    pages = [fx.pages[0], _set_counter(fx.pages[1], "Page 1 of 1")]
    sections = split_into_sections(pages)
    assert len(sections) == 1
    assert (sections[0].page_start, sections[0].page_end) == (1, 2)


# --------------------------------------------------------------------------
# 8.7 six monthly pages -> six segments with distinct, contiguous periods
# --------------------------------------------------------------------------
def test_six_monthly_pages_give_six_segments_with_distinct_periods():
    fx = six_page_page1of1_fixture()
    result = build_file_result(split_into_sections(fx.pages))
    assert len(result.segments) == 6
    assert all(isinstance(s, StatementSegment) for s in result.segments)
    periods = [(s.period_start, s.period_end) for s in result.segments]
    assert len(set(periods)) == 6
    assert periods == [(e.period_start, e.period_end) for e in fx.sections]
    assert [(s.page_start, s.page_end) for s in result.segments] == [(i, i) for i in range(1, 7)]


def test_six_sections_have_contiguous_balances():
    sections = split_into_sections(six_page_page1of1_fixture().pages)
    for prev, cur in zip(sections, sections[1:]):
        assert cur.header.opening == pytest.approx(prev.header.closing, abs=TOL)


# --------------------------------------------------------------------------
# 8.5 / 8.6 file-level balances
# --------------------------------------------------------------------------
def test_file_opening_is_first_section_opening_and_closing_is_last_section_closing():
    fx = six_page_page1of1_fixture()
    sections = split_into_sections(fx.pages)
    result = build_file_result(sections)
    assert result.opening_balance == pytest.approx(fx.sections[0].opening, abs=TOL)
    assert result.closing_balance == pytest.approx(fx.sections[-1].closing, abs=TOL)
    assert result.opening_balance == sections[0].header.opening
    assert result.closing_balance == sections[-1].header.closing
    # the middle sections' balances are not what the file reports
    assert result.closing_balance != pytest.approx(fx.sections[0].closing, abs=TOL)


def test_continuation_file_uses_header_page_balances():
    fx = continuation_fixture()
    result = build_file_result(split_into_sections(fx.pages))
    assert result.opening_balance == pytest.approx(fx.sections[0].opening, abs=TOL)
    assert result.closing_balance == pytest.approx(fx.sections[0].closing, abs=TOL)


# --------------------------------------------------------------------------
# 8.4 one SectionSummary per section
# --------------------------------------------------------------------------
def test_one_section_summary_per_section_with_all_fields():
    fx = six_page_page1of1_fixture()
    result = build_file_result(split_into_sections(fx.pages))
    assert len(result.summaries) == 6
    for i, (summary, exp) in enumerate(zip(result.summaries, fx.sections), start=1):
        assert isinstance(summary, SectionSummary)
        assert summary.period_start == exp.period_start
        assert summary.period_end == exp.period_end
        assert (summary.page_start, summary.page_end) == (i, i)
        assert summary.opening == pytest.approx(exp.opening, abs=TOL)
        assert summary.closing == pytest.approx(exp.closing, abs=TOL)
        assert summary.total_deposits == pytest.approx(exp.total_deposits, abs=TOL)
        assert summary.total_withdrawals == pytest.approx(exp.total_withdrawals, abs=TOL)


def test_summary_page_range_covers_continuation_pages():
    result = build_file_result(split_into_sections(continuation_fixture().pages))
    assert len(result.summaries) == 1
    assert (result.summaries[0].page_start, result.summaries[0].page_end) == (1, 2)
    assert (result.segments[0].page_start, result.segments[0].page_end) == (1, 2)


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------
def test_first_page_without_statement_period_is_one_section_with_no_period():
    page = _strip_period(two_col_slash_fixture().pages[0])
    sections = split_into_sections([page])
    assert len(sections) == 1
    assert (sections[0].page_start, sections[0].page_end) == (1, 1)
    assert sections[0].header.period_start is None
    result = build_file_result(sections)
    assert len(result.segments) == 1
    assert result.segments[0].period_start is None
    assert result.summaries[0].period_start is None
    assert "DQ_MISSING_PERIOD" in {f.code for f in result.findings}


def test_period_less_first_page_then_period_page_gives_two_sections():
    fx = six_page_page1of1_fixture()
    pages = [_strip_period(fx.pages[0])] + fx.pages[1:3]
    sections = split_into_sections(pages)
    assert [(s.page_start, s.page_end) for s in sections] == [(1, 1), (2, 2), (3, 3)]
    assert sections[0].header.period_start is None
    assert sections[1].header.period_start.value == fx.sections[1].period_start


def test_continuation_does_not_flag_missing_period():
    result = build_file_result(split_into_sections(continuation_fixture().pages))
    assert "DQ_MISSING_PERIOD" not in {f.code for f in result.findings}


def test_empty_page_list():
    assert split_into_sections([]) == []
    result = build_file_result([])
    assert result.opening_balance is None
    assert result.closing_balance is None
    assert result.segments == []
    assert result.summaries == []


def test_page_with_no_text_does_not_crash():
    blank = PageContent(page=1, lines=[Line("")], words=[])
    sections = split_into_sections([blank])
    assert len(sections) == 1
    result = build_file_result(sections)
    assert result.opening_balance is None
    assert result.closing_balance is None
