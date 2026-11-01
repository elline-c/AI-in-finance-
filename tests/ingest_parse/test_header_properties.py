"""Property tests for the header reader.

Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed

**Validates: Requirements 2.8, 4.7, 4.8**

Strategy: start from the header of each fixture layout, damage one or more
mandatory fields (remove the label and value, blank the value, insert an
invalid value, write an impossible date), and check that the damaged field
comes back ``None`` (never a value taken from somewhere else), that
``missing_field_findings`` names it, and that every untouched field still reads
exactly as in the undamaged page. Fixtures are fictional.
"""
from __future__ import annotations

import dataclasses
import re

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.nodes.ingest_parse.header import (
    _UEN_VALUE_RE,
    SectionHeader,
    missing_field_findings,
    read_header,
)
from fraud_detect.nodes.ingest_parse.types import Line, PageContent, Word

from .fixtures import ALL_FIXTURES, CHAR_W, GAP, _SUMMARY_X0, _SUMMARY_X1

TAG = "Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed"

SETTINGS = settings(max_examples=100, deadline=None)

LAYOUTS = sorted(ALL_FIXTURES)
SUMMARY_KEYS = ("opening", "withdrawals", "deposits", "closing")
SUMMARY_ATTR = {
    "opening": "opening",
    "withdrawals": "printed_total_withdrawals",
    "deposits": "printed_total_deposits",
    "closing": "closing",
}
CODE_OF = {
    "holder": "DQ_MISSING_HOLDER",
    "uen": "DQ_MISSING_UEN",
    "period": "DQ_MISSING_PERIOD",
    "opening": "DQ_MISSING_OPENING",
}
# attribute groups on SectionHeader owned by each damageable field
ATTRS_OF = {
    "holder": {"holder"},
    "uen": {"uen"},
    "period": {"period_start", "period_end"},
    "opening": {"opening"},
}


# ---- page editing helpers ----------------------------------------------
def _words_for(text: str, x0: float, top: float) -> list[Word]:
    out, x = [], x0
    for tok in text.split():
        w = CHAR_W * len(tok)
        out.append(Word(tok, round(x, 1), round(x + w, 1), top))
        x += w + GAP
    return out


def _find_line(page: PageContent, marker: str) -> int | None:
    for i, ln in enumerate(page.lines):
        if marker in ln.text:
            return i
    return None


def _line_words(page: PageContent, line: Line) -> list[Word]:
    return sorted((w for w in page.words if abs(w.top - line.top) < 0.5), key=lambda w: w.x0)


def _commit(page: PageContent, i: int, old_words: list[Word], new_words: list[Word], text: str) -> PageContent:
    old_ids = {id(w) for w in old_words}
    words = [w for w in page.words if id(w) not in old_ids] + new_words
    lines = list(page.lines)
    lines[i] = Line(text, page.lines[i].top)
    return PageContent(page.page, lines, words)


def replace_from(page: PageContent, marker: str, new_text: str) -> PageContent:
    """Replace everything from ``marker`` to the end of its line with ``new_text``."""
    i = _find_line(page, marker)
    if i is None:
        return page
    ln = page.lines[i]
    pos = ln.text.find(marker)
    left = ln.text[:pos].rstrip()
    ws = _line_words(page, ln)
    n = len(left.split())
    x0 = ws[n].x0 if n < len(ws) else 40.0
    new_words = ws[:n] + _words_for(new_text, x0, ln.top)
    text = " ".join(w.text for w in new_words)
    return _commit(page, i, ws, new_words, text)


def replace_before(page: PageContent, marker: str, new_left: str) -> PageContent:
    """Replace everything before ``marker`` on its line with ``new_left``."""
    i = _find_line(page, marker)
    if i is None:
        return page
    ln = page.lines[i]
    pos = ln.text.find(marker)
    n = len(ln.text[:pos].split())
    ws = _line_words(page, ln)
    x0 = ws[0].x0 if ws else 40.0
    new_words = _words_for(new_left, x0, ln.top) + ws[n:]
    text = " ".join(w.text for w in new_words)
    return _commit(page, i, ws, new_words, text)


def add_line(page: PageContent, text: str, top: float = 33.0) -> PageContent:
    return PageContent(
        page.page, [Line(text, top), *page.lines], [*_words_for(text, 40.0, top), *page.words]
    )


def _summary_cells(page: PageContent):
    """(label line index, value line index) of the four-label summary row, or None."""
    i = _find_line(page, "Opening Balance Total")
    if i is None or i + 1 >= len(page.lines):
        return None
    return i, i + 1


def damage_summary(page: PageContent, key: str, mode: str, garbage: str = "N/A") -> PageContent:
    """Remove (label + value) or garble the value of one summary cell."""
    cells = _summary_cells(page)
    if cells is None:
        return page
    li, vi = cells
    k = SUMMARY_KEYS.index(key)
    lline, vline = page.lines[li], page.lines[vi]
    lws, vws = _line_words(page, lline), _line_words(page, vline)
    lo_x1 = _SUMMARY_X1[k - 1] + 3 if k else float("-inf")
    hi_x1 = _SUMMARY_X1[k] + 3
    cell = [w for w in vws if lo_x1 < w.x1 <= hi_x1]
    new_v = [w for w in vws if w not in cell]
    if mode == "garble":
        new_v += [Word(garbage, cell[0].x0, cell[-1].x1, vline.top)]
        new_v.sort(key=lambda w: w.x0)
        page = _commit(page, vi, vws, new_v, " ".join(w.text for w in new_v))
        return page
    page = _commit(page, vi, vws, new_v, " ".join(w.text for w in new_v))
    lo_x0 = _SUMMARY_X0[k] - 1
    hi_x0 = _SUMMARY_X0[k + 1] - 1 if k < 3 else float("inf")
    label = [w for w in lws if lo_x0 <= w.x0 < hi_x0]
    new_l = [w for w in lws if w not in label]
    return _commit(page, li, lws, new_l, " ".join(w.text for w in new_l))


def read(page: PageContent) -> SectionHeader:
    return read_header(page.lines, page.words, None, page.page)


# ---- field damage -------------------------------------------------------
def _has_line(page: PageContent, marker: str) -> bool:
    return _find_line(page, marker) is not None


def damage_holder(page: PageContent, mode: str) -> PageContent:
    if _has_line(page, "Account Holder"):  # single-column generator layout
        return replace_from(page, "Account Holder", "" if mode == "remove" else "Account Holder:")
    new_left = "" if mode == "remove" else "BUSINESS CURRENT ACCOUNT"
    return replace_before(page, "Account No.", new_left)


def damage_uen(page: PageContent, mode: str, bad: str) -> PageContent:
    marker = "Company UEN" if _has_line(page, "Company UEN") else "UEN"
    if not _has_line(page, marker):
        return add_line(page, f"Company UEN {bad}") if mode == "invalid" else page
    label = "Company UEN" if marker == "Company UEN" else "UEN:"
    return replace_from(page, marker, "" if mode == "remove" else f"{label} {bad}")


def damage_period(page: PageContent, mode: str, bad: tuple[str, str]) -> PageContent:
    if mode == "remove":
        return replace_from(page, "Statement Period", "")
    return replace_from(page, "Statement Period", f"Statement Period {bad[0]} to {bad[1]}")


def damage_opening(page: PageContent, mode: str, garbage: str) -> PageContent:
    if _has_line(page, "Opening Balance:"):  # generator: same-line opening
        return replace_from(page, "Opening Balance:", "" if mode == "remove" else f"Opening Balance: {garbage}")
    return damage_summary(page, "opening", "remove" if mode == "remove" else "garble", garbage)


# ---- strategies ----------------------------------------------------------
_uen_chars = st.text(alphabet="0123456789ABCDEFXYZ-./", min_size=1, max_size=14)
bad_uens = _uen_chars.filter(lambda s: _UEN_VALUE_RE.fullmatch(s.strip(".,;")) is None)

_bad_day = st.integers(32, 99)
_bad_month = st.integers(13, 99)
unreadable_dates = st.one_of(
    st.builds(lambda d, m, y: f"{d}/{m:02d}/{y}", _bad_day, st.integers(1, 12), st.integers(2000, 2099)),
    st.builds(lambda d, m, y: f"{d:02d}/{m}/{y}", st.integers(1, 28), _bad_month, st.integers(2000, 2099)),
    st.builds(lambda d, m, y: f"{y}-{m}-{d}", _bad_day, st.integers(1, 12), st.integers(2000, 2099)),
    st.builds(lambda d, y: f"{d} Foo {y}", st.integers(1, 28), st.integers(2000, 2099)),
)
unreadable_periods = st.tuples(unreadable_dates, unreadable_dates)
garbage_amounts = st.sampled_from(["N/A", "--", "???", "XXXX", "10,000", "1O,OOO.OO", "abc"])
# "10,000" is a legitimate same-line amount (the summary row alone demands two decimals)
non_numeric_garbage = st.sampled_from(["N/A", "--", "???", "XXXX", "1O,OOO.OO", "abc"])
actions = st.sampled_from(["keep", "remove", "corrupt"])


@st.composite
def damage_plan(draw):
    return {
        "holder": draw(actions),
        "uen": draw(actions),
        "period": draw(actions),
        "opening": draw(actions),
        "bad_uen": draw(bad_uens),
        "bad_period": draw(unreadable_periods),
        "garbage": draw(non_numeric_garbage),
    }


def apply_plan(page: PageContent, plan: dict) -> tuple[PageContent, set[str]]:
    """Return the damaged page and the set of fields that must now read as None."""
    damaged: set[str] = set()
    if plan["holder"] != "keep":
        page = damage_holder(page, plan["holder"])
        damaged.add("holder")
    if plan["uen"] != "keep":
        page = damage_uen(page, "remove" if plan["uen"] == "remove" else "invalid", plan["bad_uen"])
        damaged.add("uen")
    if plan["period"] != "keep":
        page = damage_period(page, plan["period"], plan["bad_period"])
        damaged.add("period")
    if plan["opening"] != "keep":
        page = damage_opening(page, plan["opening"], plan["garbage"])
        damaged.add("opening")
    return page, damaged


def is_none(h: SectionHeader, field: str) -> bool:
    if field == "period":
        return h.period_start is None or h.period_start.value is None
    return getattr(h, field) is None


def expected_codes(h: SectionHeader) -> list[str]:
    return [CODE_OF[f] for f in ("holder", "uen", "period", "opening") if is_none(h, f)]


# ---- properties -----------------------------------------------------------
@SETTINGS
@given(layout=st.sampled_from(LAYOUTS), plan=damage_plan())
def test_damaged_fields_are_none_flagged_and_others_untouched(layout, plan):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed
    # Validates: Requirements 2.8, 4.7, 4.8
    page = ALL_FIXTURES[layout]().pages[0]
    base = read(page)
    damaged_page, damaged = apply_plan(page, plan)
    h = read(damaged_page)

    for field in damaged:
        assert is_none(h, field), (field, getattr(h, field, None))
    codes = [f.code for f in missing_field_findings(h)]
    assert codes == expected_codes(h)
    for field in damaged:
        assert CODE_OF[field] in codes

    touched = set().union(*(ATTRS_OF[f] for f in damaged)) if damaged else set()
    for f in dataclasses.fields(SectionHeader):
        if f.name in touched:
            continue
        assert getattr(h, f.name) == getattr(base, f.name), f.name


@SETTINGS
@given(
    layout=st.sampled_from([n for n in LAYOUTS if n != "generator_v1"]),
    key=st.sampled_from(SUMMARY_KEYS),
    mode=st.sampled_from(["remove", "garble"]),
    garbage=garbage_amounts,
)
def test_damaged_summary_value_is_none_and_others_still_read(layout, key, mode, garbage):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed
    # Validates: Requirements 2.8, 4.7, 4.8
    page = ALL_FIXTURES[layout]().pages[0]
    base = read(page)
    h = read(damage_summary(page, key, mode, garbage))

    assert getattr(h, SUMMARY_ATTR[key]) is None
    for other in SUMMARY_KEYS:
        if other != key:
            assert getattr(h, SUMMARY_ATTR[other]) == pytest.approx(getattr(base, SUMMARY_ATTR[other]))
    if key == "opening":
        assert "DQ_MISSING_OPENING" in [f.code for f in missing_field_findings(h)]


@SETTINGS
@given(
    layout=st.sampled_from(LAYOUTS),
    bad=unreadable_periods,
)
def test_unreadable_period_dates_keep_raw_text_with_none_value(layout, bad):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed
    # Validates: Requirements 2.8, 4.7, 4.8
    page = damage_period(ALL_FIXTURES[layout]().pages[0], "corrupt", bad)
    h = read(page)
    assert h.period_start is not None and h.period_end is not None
    assert h.period_start.value is None and h.period_end.value is None
    assert (h.period_start.raw, h.period_end.raw) == bad
    assert "DQ_MISSING_PERIOD" in [f.code for f in missing_field_findings(h)]


_company = st.builds(
    lambda w, s: " ".join(w) + " " + s,
    st.lists(st.text(alphabet="ABCDEFGHIJKLMNOPRSTUVW", min_size=3, max_size=9), min_size=1, max_size=3),
    st.sampled_from(["PTE. LTD.", "TRADING PTE. LTD.", "HOLDINGS PTE. LTD."]),
)


@SETTINGS
@given(a=_company, b=_company)
def test_two_equally_plausible_holder_lines_give_no_holder(a, b):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed
    # Validates: Requirements 2.8, 4.7, 4.8
    texts = ["STATEMENT OF ACCOUNT", "Corvane Trust", "Page 1 of 1", "SPECIMEN", a, b]
    lines = [Line(t, 30.0 + 12 * i) for i, t in enumerate(texts)]
    words = [w for ln in lines for w in _words_for(ln.text, 40.0, ln.top)]
    h = read_header(lines, words, None, 1)
    assert h.holder is None
    assert "DQ_MISSING_HOLDER" in [f.code for f in missing_field_findings(h)]


_fragments = st.sampled_from([
    "Account Holder:", "Account No.", "Account Name", "UEN", "Company UEN", "Co. Reg. No.",
    "Statement Period", "Statement Date", "Currency", "to", "-", ":", "Opening Balance",
    "Closing Balance", "Total Withdrawals", "Total Deposits", "Money in", "Money out",
    "SGD", "S$", "CR", "10,000.00", "01/01/2026", "2026-01-31", "31 Jan 2026", "(SGD)",
])
_noise = st.one_of(
    st.text(max_size=200),
    st.lists(st.one_of(_fragments, st.text(max_size=12)), max_size=14).map(" ".join),
    st.text(max_size=20).map(lambda s: s * 300),  # very long line
)
_boxes = st.builds(
    lambda t, x0, w, top: Word(t, x0, x0 + w, top),
    st.text(min_size=1, max_size=12).filter(lambda s: s.strip() != ""),
    st.floats(0, 600, allow_nan=False),
    st.floats(0, 100, allow_nan=False),
    st.floats(0, 800, allow_nan=False),
)


@SETTINGS
@given(
    texts=st.lists(_noise, max_size=12),
    words=st.lists(_boxes, max_size=30),
    page_no=st.integers(1, 50),
)
def test_arbitrary_text_never_raises(texts, words, page_no):
    # Feature: multi-layout-ingestion-and-fabrication-rules, Property 7: Unreadable means flagged, not guessed
    # Validates: Requirements 2.8, 4.7, 4.8
    lines = [Line(t, 30.0 + 12 * i) for i, t in enumerate(texts)]
    h = read_header(lines, words, None, page_no)
    assert isinstance(h, SectionHeader) and h.page == page_no
    codes = [f.code for f in missing_field_findings(h)]
    assert codes == expected_codes(h)
    assert set(codes) <= set(CODE_OF.values())
    assert re.fullmatch(r"[A-Z]{3}", h.currency or "AAA")
