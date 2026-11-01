"""Unit tests for the Header_Reader (Requirements 4.1-4.9).

Test-first: ``fraud_detect.nodes.ingest_parse.header`` is implemented in task
4.2, so this module fails with ModuleNotFoundError until then. All values are
fictional.

API under test (design Component 3):
  * ``read_header(lines, words, profile, page) -> SectionHeader``; empty fields
    are ``None`` (never guessed). The tests pass ``profile=None``: header
    reading must not depend on a profile for these inputs.
  * ``missing_field_findings(header) -> list[Finding]`` returns the existing
    Node 1 mandatory-field findings (agent ``Node1:Ingestion``, MEDIUM, weight
    8) for holder, UEN, period and opening balance, in that order.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from fraud_detect.models import Severity
from fraud_detect.nodes.ingest_parse.header import (
    SectionHeader,
    missing_field_findings,
    read_header,
)
from fraud_detect.nodes.ingest_parse.types import Line, PageContent, Word

from .fixtures import (
    ALL_FIXTURES,
    continuation_fixture,
    six_page_page1of1_fixture,
    two_col_dmon_fixture,
)

PROFILES = list(ALL_FIXTURES)


def read(page: PageContent) -> SectionHeader:
    return read_header(page.lines, page.words, None, page.page)


def page_from_lines(*texts: str) -> PageContent:
    """Build a simple page: one line per text, words spaced left to right."""
    lines, words = [], []
    for i, text in enumerate(texts):
        top = 30.0 + 12 * i
        lines.append(Line(text, top))
        x = 40.0
        for tok in text.split():
            w = 4.7 * len(tok)
            words.append(Word(tok, round(x, 1), round(x + w, 1), top))
            x += w + 2.3
    return PageContent(page=1, lines=lines, words=words)


def replace_line_at(page: PageContent, top: float, text: str, word_text: str | None = None) -> PageContent:
    """Replace the line (and optionally all words) at a given vertical position."""
    lines = [Line(text, ln.top) if ln.top == top else ln for ln in page.lines]
    words = page.words
    if word_text is not None:
        words = [Word(word_text, w.x0, w.x1, w.top) if w.top == top else w for w in words]
    return PageContent(page.page, lines, words)


# ---- all six layouts: 4.1, 4.5, 4.9 ------------------------------------
@pytest.mark.parametrize("name", PROFILES)
def test_header_block_of_each_layout(name):
    fx = ALL_FIXTURES[name]()
    sec = fx.sections[0]
    h = read(fx.pages[0])
    assert h.holder == fx.holder
    assert h.account_number == fx.account
    assert h.uen == fx.uen
    assert h.currency == "SGD"
    assert h.period_start.value == sec.period_start
    assert h.period_end.value == sec.period_end
    assert h.page == fx.pages[0].page
    assert h.opening == pytest.approx(sec.opening)
    if name == "generator_v1":
        # our generator prints only the opening balance in its header
        assert h.closing is None
        assert h.printed_total_withdrawals is None
        assert h.printed_total_deposits is None
    else:
        assert h.closing == pytest.approx(sec.closing)
        assert h.printed_total_withdrawals == pytest.approx(sec.total_withdrawals)
        assert h.printed_total_deposits == pytest.approx(sec.total_deposits)
        assert h.statement_date.value == sec.period_end + timedelta(days=1)  # 4.9


# ---- holder: 4.1, 4.2, 4.7 ---------------------------------------------
def test_holder_is_left_half_of_account_no_line():
    h = read(page_from_lines(
        "STATEMENT OF ACCOUNT", "Corvane Trust", "Page 1 of 1", "SPECIMEN",
        "NORTHGATE FICTIONAL TRADING PTE. LTD. Account No. FX 1111 2222 3333",
    ))
    assert h.holder == "NORTHGATE FICTIONAL TRADING PTE. LTD."
    assert h.account_number == "FX 1111 2222 3333"


def test_business_current_account_line_is_not_the_holder():
    fx = two_col_dmon_fixture(with_uen=True)
    assert any("BUSINESS CURRENT ACCOUNT" in ln.text for ln in fx.pages[0].lines)
    h = read(fx.pages[0])
    assert h.holder == "MERIDIAN QUAY TRADING PTE. LTD."
    assert "BUSINESS" not in (h.holder or "")


@pytest.mark.parametrize("label", ["Account Holder:", "Account Holder", "Account Name:", "account name"])
def test_holder_label_variants(label):
    h = read(page_from_lines("STATEMENT OF ACCOUNT", f"{label} Lowfield Sample Pte Ltd", "Currency: SGD"))
    assert h.holder == "Lowfield Sample Pte Ltd"


def test_ambiguous_holder_is_left_empty():
    h = read(page_from_lines(
        "STATEMENT OF ACCOUNT", "Corvane Trust", "Page 1 of 1", "SPECIMEN",
        "ALPHA SAMPLE TRADING PTE. LTD.", "BETA SAMPLE HOLDINGS PTE. LTD.",
    ))
    assert h.holder is None


# ---- UEN: 4.3, 4.4 -----------------------------------------------------
@pytest.mark.parametrize("label", ["UEN", "Company UEN", "UEN:", "Company UEN:"])
def test_uen_label_variants(label):
    h = read(page_from_lines(f"{label} 201234567K"))
    assert h.uen == "201234567K"


def test_ten_character_uen_is_accepted():
    assert read(page_from_lines("UEN T08LL0001A")).uen == "T08LL0001A"
    assert read(page_from_lines("UEN 209947062H")).uen == "209947062H"


@pytest.mark.parametrize("bad", ["12345", "2012345671", "20123456", "not-a-uen!"])
def test_uen_that_does_not_fit_pattern_is_ignored(bad):
    assert read(page_from_lines(f"Company UEN {bad}")).uen is None


def test_no_uen_label_gives_none():
    fx = ALL_FIXTURES["two_col_slash"]()
    assert fx.uen is None
    assert read(fx.pages[0]).uen is None


# ---- summary row: 4.5 --------------------------------------------------
@pytest.mark.parametrize("name, prefix", [("two_col_dmon", "S$"), ("signed_amount_iso", "SGD"), ("two_col_slash", "")])
def test_summary_row_prefix_styles_are_mapped_by_label(name, prefix):
    fx = ALL_FIXTURES[name]()
    sec = fx.sections[0]
    values_line = next(ln.text for ln in fx.pages[0].lines if ln.top == 154)
    assert (prefix in values_line) if prefix else ("S$" not in values_line and "SGD" not in values_line)
    h = read(fx.pages[0])
    assert (h.opening, h.printed_total_withdrawals, h.printed_total_deposits, h.closing) == pytest.approx(
        (sec.opening, sec.total_withdrawals, sec.total_deposits, sec.closing)
    )


def test_summary_row_mapped_by_label_not_by_order():
    # values printed in a different order than the labels expect: still keyed by label
    h = read(page_from_lines(
        "Closing Balance Opening Balance Total Deposits Total Withdrawals",
        "9,000.00 1,000.00 500.00 2,500.00",
    ))
    assert h.closing == pytest.approx(9000.00)
    assert h.opening == pytest.approx(1000.00)
    assert h.printed_total_deposits == pytest.approx(500.00)
    assert h.printed_total_withdrawals == pytest.approx(2500.00)


# ---- summary row: 4.6 --------------------------------------------------
def test_merged_digit_summary_is_read_from_word_positions():
    fx = ALL_FIXTURES["two_col_slash"]()
    sec = fx.sections[0]
    page = replace_line_at(fx.pages[0], 154, "1417,,307010..6040 99..0,1")  # text path unusable
    h = read(page)
    assert (h.opening, h.printed_total_withdrawals, h.printed_total_deposits, h.closing) == pytest.approx(
        (sec.opening, sec.total_withdrawals, sec.total_deposits, sec.closing)
    )


def test_unreadable_summary_is_left_empty():
    fx = ALL_FIXTURES["two_col_slash"]()
    page = replace_line_at(fx.pages[0], 154, "n/a n/a n/a n/a", word_text="n/a")
    h = read(page)
    assert h.opening is None
    assert h.closing is None
    assert h.printed_total_withdrawals is None
    assert h.printed_total_deposits is None
    # the rest of the header is still read
    assert h.holder == fx.holder
    assert h.account_number == fx.account


# ---- multi-page files --------------------------------------------------
def test_each_page1of1_page_has_its_own_period_and_summary():
    fx = six_page_page1of1_fixture()
    assert len(fx.pages) == len(fx.sections) == 6
    for page, sec in zip(fx.pages, fx.sections):
        h = read(page)
        assert h.page == page.page
        assert h.period_start.value == sec.period_start
        assert h.period_end.value == sec.period_end
        assert h.opening == pytest.approx(sec.opening)
        assert h.closing == pytest.approx(sec.closing)
        assert h.holder == fx.holder
        assert h.uen == fx.uen


def test_continuation_page_has_no_period_and_no_opening():
    fx = continuation_fixture()
    assert len(fx.pages) == 2
    h = read(fx.pages[1])
    assert h.period_start is None and h.period_end is None
    assert h.opening is None


# ---- missing fields: 4.8 -----------------------------------------------
EXPECTED = {
    "holder": "DQ_MISSING_HOLDER",
    "uen": "DQ_MISSING_UEN",
    "period": "DQ_MISSING_PERIOD",
    "opening": "DQ_MISSING_OPENING",
}


def test_complete_header_has_no_missing_findings():
    fx = ALL_FIXTURES["generator_v1"]()
    assert missing_field_findings(read(fx.pages[0])) == []


def test_personal_layout_without_uen_gives_dq_missing_uen_only():
    fx = ALL_FIXTURES["two_col_slash"]()
    findings = missing_field_findings(read(fx.pages[0]))
    assert [f.code for f in findings] == ["DQ_MISSING_UEN"]


@pytest.mark.parametrize("missing", list(EXPECTED))
def test_each_missing_field_gives_matching_dq_code(missing):
    fx = ALL_FIXTURES["generator_v1"]()
    h = read(fx.pages[0])
    if missing == "period":
        h.period_start = None
        h.period_end = None
    else:
        setattr(h, missing, None)
    findings = missing_field_findings(h)
    assert [f.code for f in findings] == [EXPECTED[missing]]
    f = findings[0]
    assert f.agent == "Node1:Ingestion"
    assert f.severity == Severity.MEDIUM
    assert f.weight == 8


def test_all_missing_returns_four_findings_in_fixed_order():
    h = read(page_from_lines("STATEMENT OF ACCOUNT", "Currency: SGD"))
    assert [f.code for f in missing_field_findings(h)] == list(EXPECTED.values())
