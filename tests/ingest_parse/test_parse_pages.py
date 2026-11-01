"""End-to-end tests for the ``parse_pages`` entry point (task 9.1).

Written ahead of the implementation (task 9.2); they fail with ImportError
until ``parse_pages`` exists.

API these tests pin down
------------------------
``parse_pages(pages: list[PageContent]) -> ParseResult``, importable from
``fraud_detect.nodes.ingest_parse.parser`` (module created by task 9.2).
It never raises on odd input. ``ParseResult`` is a dataclass with:

  transactions : list[fraud_detect.models.Transaction]
      signed ``amount`` (credits +, debits -, magnitude exactly as printed),
      printed ``balance``, ``date`` / ``date_raw``, ``description``, ``page``,
      ``value_date_raw`` and ``direction_method``; ``index`` runs 0..n-1.
  holder, uen, account_number, currency : Optional[str]
  period_start, period_end              : Optional[date]   (file level)
  opening_balance                       : first Section's opening
  closing_balance                       : LAST Section's closing
  segments : list[StatementSegment]     (one per Section)
  sections : list[SectionSummary]       (one per Section)
  layout_profile : Optional[str]        (chosen profile name)
  parse_coverage : Optional[float]      (0..1)
  findings : list[Finding]              (DQ_MISSING_*, DQ_LAYOUT_UNRECOGNISED,
      DQ_DIRECTION_UNCERTAIN, DQ_PARSE_INCONSISTENT, DQ_UNREADABLE_ROW,
      DQ_NO_TXNS when no transactions, DQ_NO_TEXT when no text at all)

Error isolation: an unexpected exception while reading one page is caught and
becomes ``DQ_LAYOUT_UNRECOGNISED`` with ``page`` set to that page; other pages
still parse. The isolation test makes ``table.read_rows`` raise for one page
by patching it on the ``table`` module and on every module that imported the
name (``layouts``, ``parser``), so the parser may import it either way.

Run with AUDIT_SIGNING_KEY_PATH and FRAUD_ENABLE_TXN_MODEL unset.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import date

import pytest

from fraud_detect.models import DocumentMetadata, IngestionResult
from fraud_detect.nodes import financial_logic
from fraud_detect.nodes.ingest_parse import layouts, table
from fraud_detect.nodes.ingest_parse.parser import parse_pages
from fraud_detect.nodes.ingest_parse.types import Line, PageContent, Word

from .fixtures import (
    ALL_FIXTURES,
    continuation_fixture,
    six_page_page1of1_fixture,
    two_col_slash_fixture,
)

TOL = 0.01
MONTH_CODES = {"FL_MONTH_DUPLICATE", "FL_MONTH_MISSING", "FL_MONTH_OUT_OF_ORDER"}


def _codes(result) -> list[str]:
    return [f.code for f in result.findings]


def _norm(s: str | None) -> str:
    return " ".join((s or "").split())


# --------------------------------------------------------------------------
# each of the six layouts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_FIXTURES))
class TestEachLayout:
    def test_profile_name_and_coverage(self, name):
        res = parse_pages(ALL_FIXTURES[name]().pages)
        assert res.layout_profile == name
        assert res.parse_coverage is not None and res.parse_coverage >= 0.9
        assert "DQ_LAYOUT_UNRECOGNISED" not in _codes(res)

    def test_header_fields(self, name):
        fx = ALL_FIXTURES[name]()
        exp = fx.sections[0]
        res = parse_pages(fx.pages)
        assert res.holder == fx.holder
        assert res.account_number == fx.account
        assert res.uen == fx.uen
        assert res.currency == "SGD"
        assert res.period_start == exp.period_start
        assert res.period_end == exp.period_end
        assert res.opening_balance == pytest.approx(exp.opening, abs=TOL)
        assert res.closing_balance == pytest.approx(exp.closing, abs=TOL)
        assert len(res.segments) == 1 and len(res.sections) == 1
        assert res.sections[0].total_withdrawals == pytest.approx(exp.total_withdrawals, abs=TOL)
        assert res.sections[0].total_deposits == pytest.approx(exp.total_deposits, abs=TOL)

    def test_transactions_signed_and_dated(self, name):
        fx = ALL_FIXTURES[name]()
        res = parse_pages(fx.pages)
        expected = fx.sections[0].txns
        assert len(res.transactions) == len(expected) == fx.n_txns
        assert [t.index for t in res.transactions] == list(range(len(expected)))
        for got, want in zip(res.transactions, expected):
            assert got.amount == pytest.approx(want.amount, abs=TOL)
            assert got.balance == pytest.approx(want.balance, abs=TOL)
            assert got.date == want.date
            assert got.date_raw
            assert got.page is not None
            assert got.direction_method in {"column", "printed_sign", "balance_change"}
            assert _norm(got.description) == _norm(want.description)

    def test_replayed_balance_equals_printed_closing(self, name):
        res = parse_pages(ALL_FIXTURES[name]().pages)
        bal = res.opening_balance
        for t in res.transactions:
            bal = round(bal + t.amount, 2)
            assert bal == pytest.approx(t.balance, abs=TOL)
        assert bal == pytest.approx(res.closing_balance, abs=TOL)

    def test_clean_fixture_has_no_parse_findings(self, name):
        fx = ALL_FIXTURES[name]()
        bad = {"DQ_LAYOUT_UNRECOGNISED", "DQ_DIRECTION_UNCERTAIN", "DQ_PARSE_INCONSISTENT",
               "DQ_UNREADABLE_ROW", "DQ_NO_TXNS", "DQ_NO_TEXT"}
        assert not bad & set(_codes(parse_pages(fx.pages)))


def test_dual_date_layout_keeps_value_date():
    res = parse_pages(ALL_FIXTURES["two_col_dual_date"]().pages)
    assert all(t.value_date_raw for t in res.transactions)


def test_continuation_page_joins_same_section():
    fx = continuation_fixture()
    assert len(fx.pages) == 2
    res = parse_pages(fx.pages)
    assert len(res.segments) == 1
    assert (res.segments[0].page_start, res.segments[0].page_end) == (1, 2)
    assert len(res.transactions) == fx.n_txns
    assert {t.page for t in res.transactions} == {1, 2}
    assert "DQ_MISSING_PERIOD" not in _codes(res)
    assert res.closing_balance == pytest.approx(fx.sections[0].closing, abs=TOL)


# --------------------------------------------------------------------------
# six monthly pages, each printing "Page 1 of 1"
# --------------------------------------------------------------------------
def _as_ingestion(res, n_pages: int) -> IngestionResult:
    return IngestionResult(
        source_path="six_page.pdf",
        page_count=n_pages,
        raw_text="",
        metadata=DocumentMetadata(),
        segments=res.segments,
        account_holder=res.holder,
        uen=res.uen,
        account_number=res.account_number,
        statement_period_start=res.period_start,
        statement_period_end=res.period_end,
        opening_balance=res.opening_balance,
        closing_balance=res.closing_balance,
        declared_currency=res.currency,
        transactions=res.transactions,
        layout_profile=res.layout_profile,
        parse_coverage=res.parse_coverage,
        sections=res.sections,
        data_quality_findings=res.findings,
    )


class TestSixPagePage1Of1:
    def setup_method(self):
        self.fx = six_page_page1of1_fixture()
        self.res = parse_pages(self.fx.pages)

    def test_six_segments_with_distinct_periods(self):
        segs = self.res.segments
        assert len(segs) == 6 and len(self.res.sections) == 6
        periods = [(s.period_start, s.period_end) for s in segs]
        assert len(set(periods)) == 6
        assert periods == [(s.period_start, s.period_end) for s in self.fx.sections]
        assert [s.page_start for s in segs] == [1, 2, 3, 4, 5, 6]

    def test_file_balances_come_from_first_and_last_section(self):
        secs = self.res.sections
        assert self.res.opening_balance == secs[0].opening == pytest.approx(self.fx.sections[0].opening, abs=TOL)
        assert self.res.closing_balance == secs[-1].closing == pytest.approx(self.fx.sections[-1].closing, abs=TOL)
        assert self.res.period_start == date(2026, 4, 1)
        assert self.res.period_end == date(2026, 9, 30)

    def test_all_transactions_read_and_chain_reconciles(self):
        assert len(self.res.transactions) == self.fx.n_txns
        bal = self.res.opening_balance
        for t in self.res.transactions:
            bal = round(bal + t.amount, 2)
            assert bal == pytest.approx(t.balance, abs=TOL)
        assert bal == pytest.approx(self.res.closing_balance, abs=TOL)

    def test_node3_emits_no_month_findings(self):
        ing = _as_ingestion(self.res, len(self.fx.pages))
        report = financial_logic.analyze(ing)
        assert MONTH_CODES.isdisjoint(f.code for f in report.findings)


# --------------------------------------------------------------------------
# robustness (Requirement 17)
# --------------------------------------------------------------------------
def test_pathological_10000_char_line_is_fast():
    fx = two_col_slash_fixture()
    page = fx.pages[0]
    junk = "01/07/2026 " + "1,234.5" * 1500  # > 10,000 characters, digit/comma heavy
    assert len(junk) >= 10_000
    top = 700.0
    evil = PageContent(
        page=page.page,
        lines=[*page.lines, Line(junk, top)],
        words=[*page.words, Word(junk.split()[0], 40.0, 90.0, top), Word(junk.split()[1], 100.0, 9000.0, top)],
    )
    start = time.perf_counter()
    res = parse_pages([evil])
    assert time.perf_counter() - start < 1.0
    assert res.layout_profile is not None  # still returns a result, no exception


def test_pathological_line_without_table_is_fast():
    junk = "9" * 10_000
    page = PageContent(page=1, lines=[Line(junk, 10.0)], words=[Word(junk, 0.0, 47000.0, 10.0)])
    start = time.perf_counter()
    res = parse_pages([page])
    assert time.perf_counter() - start < 1.0
    assert res.transactions == []


def test_page_without_text_gives_dq_no_text():
    res = parse_pages([PageContent(page=1)])
    codes = _codes(res)
    assert "DQ_NO_TEXT" in codes
    assert res.transactions == []


def _patch_read_rows_to_fail_on(monkeypatch, bad_page: PageContent) -> None:
    original = table.read_rows
    bad_words = list(bad_page.words)

    def flaky(page_words, *args, **kwargs):
        if list(page_words) == bad_words:
            raise RuntimeError("boom")
        return original(page_words, *args, **kwargs)

    monkeypatch.setattr(table, "read_rows", flaky)
    monkeypatch.setattr(layouts, "read_rows", flaky, raising=False)
    from fraud_detect.nodes.ingest_parse import parser

    monkeypatch.setattr(parser, "read_rows", flaky, raising=False)


def test_unexpected_error_on_one_page_is_isolated(monkeypatch):
    fx = six_page_page1of1_fixture()
    bad = fx.pages[2]  # page 3
    _patch_read_rows_to_fail_on(monkeypatch, bad)
    res = parse_pages(fx.pages)  # must not raise

    flagged = [f for f in res.findings if f.code == "DQ_LAYOUT_UNRECOGNISED"]
    assert [f.page for f in flagged] == [3]
    per_page = {p: 0 for p in range(1, 7)}
    for t in res.transactions:
        per_page[t.page] += 1
    for p in (1, 2, 4, 5, 6):
        assert per_page[p] == len(fx.sections[0].txns), f"page {p} should still parse"
    assert per_page[3] == 0 or per_page[3] < len(fx.sections[0].txns)
    assert len(res.segments) == 6


# --------------------------------------------------------------------------
# direction and edited amounts
# --------------------------------------------------------------------------
def _map_words(page: PageContent, fn) -> PageContent:
    return replace(page, words=[fn(w) for w in page.words])


def test_amount_in_wrong_column_emits_direction_uncertain():
    fx = two_col_slash_fixture()
    page = fx.pages[0]
    # the 1,200.00 rent debit is printed under Deposits instead of Withdrawals
    shifted = _map_words(
        page,
        lambda w: replace(w, x0=w.x0 + 75.0, x1=w.x1 + 75.0) if w.text == "1,200.00" else w,
    )
    res = parse_pages([shifted])
    uncertain = [f for f in res.findings if f.code == "DQ_DIRECTION_UNCERTAIN"]
    assert len(uncertain) == 1
    assert uncertain[0].page == 1
    # column direction is kept (credit) with the printed magnitude; not guessed
    rent = next(t for t in res.transactions if "RENT" in (t.description or ""))
    assert rent.amount == pytest.approx(1200.00, abs=TOL)
    assert rent.direction_method == "column"


def test_edited_printed_amount_is_kept_as_printed():
    fx = two_col_slash_fixture()
    page = fx.pages[0]
    # tamper: the 18.40 card spend is edited to 81.40; balance column untouched
    edited = PageContent(
        page=page.page,
        lines=[replace(ln, text=ln.text.replace(" 18.40 ", " 81.40 ")) for ln in page.lines],
        words=[replace(w, text="81.40") if w.text == "18.40" else w for w in page.words],
    )
    res = parse_pages([edited])
    txns = res.transactions
    idx = next(i for i, t in enumerate(txns) if "QUILLFEATHER" in (t.description or ""))
    assert txns[idx].amount == pytest.approx(-81.40, abs=TOL)  # printed amount, not balance delta
    prev = txns[idx - 1].balance
    # the balance step does not reconcile, so Node 3 would raise FL_BALANCE_BREAK
    assert abs(prev + txns[idx].amount - txns[idx].balance) > TOL
    assert len(txns) == fx.n_txns
