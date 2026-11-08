"""Tests for layout profiles and selection (task 7.1; Requirement 7).

Written BEFORE ``fraud_detect/nodes/ingest_parse/layouts.py`` exists (task 7.2),
so every test here is expected to fail with ``ModuleNotFoundError`` for now.

API contract assumed by these tests (compatible with design.md Component 6):

* ``LayoutProfile(name, date_styles, period_separators, holder_labels,
  amount_style, two_date_columns, balance_suffix_cr)`` - frozen dataclass.
* ``REGISTRY`` - tuple of exactly six profiles, in the order of
  ``PROFILE_NAMES`` below.
* ``PageParse(rows, coverage)`` - result of parsing ONE page with ONE profile.
  ``coverage`` is the share of table-looking lines that became valid rows
  (date read, amounts read). Selection is by coverage, never by bank name.
* ``parse_page(page, profile) -> PageParse`` - module-level seam that
  ``choose_profile`` MUST call (looked up on the module at call time) for every
  (page, profile) pair. Tests monkeypatch it to build ties and errors.
* ``choose_profile(pages) -> ProfileChoice`` with fields:
    - ``profile``: the winning ``LayoutProfile``, or ``None`` when no page
      could be decided (e.g. a single page with an unresolved tie);
    - ``coverage``: overall coverage (0..1);
    - ``per_page``: ``{page_number: best coverage}`` for every page;
    - ``rows``: the rows read with the chosen profile(s), still returned when
      the layout is unrecognised;
    - ``findings``: ``list[Finding]``. ``DQ_LAYOUT_UNRECOGNISED`` is surfaced
      here: severity MEDIUM, weight 8, ``evidence`` names the page
      (``"page N"``), one finding per affected page.
* Ties: equal best coverage -> earlier profile in ``REGISTRY`` wins, unless the
  tied profiles' ``rows`` differ, in which case the page is not decided and
  gets ``DQ_LAYOUT_UNRECOGNISED``.
* An exception raised by one profile on a page is treated as coverage 0 for
  that profile; selection never raises.
"""
from __future__ import annotations

import dataclasses

import pytest

from fraud_detect.models import Severity
from fraud_detect.nodes.ingest_parse import layouts
from fraud_detect.nodes.ingest_parse.layouts import (
    REGISTRY,
    LayoutProfile,
    PageParse,
    choose_profile,
)
from fraud_detect.nodes.ingest_parse.types import Line, PageContent, Word

from .fixtures import ALL_FIXTURES, PROFILE_NAMES, continuation_fixture, six_page_page1of1_fixture

CODE = "DQ_LAYOUT_UNRECOGNISED"


def _layout_findings(choice):
    return [f for f in choice.findings if f.code == CODE]


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_registry_has_exactly_six_profiles_in_required_order():
    assert isinstance(REGISTRY, tuple)
    assert tuple(p.name for p in REGISTRY) == PROFILE_NAMES
    assert all(isinstance(p, LayoutProfile) for p in REGISTRY)
    assert len(set(p.name for p in REGISTRY)) == 6


def test_registry_profiles_expose_the_required_fields():
    names = {f.name for f in dataclasses.fields(LayoutProfile)}
    assert names >= {
        "name", "date_styles", "period_separators", "holder_labels",
        "amount_style", "two_date_columns", "balance_suffix_cr",
    }
    by_name = {p.name: p for p in REGISTRY}
    assert by_name["two_col_dual_date"].two_date_columns is True
    assert by_name["two_col_ddmmyy_cr"].balance_suffix_cr is True
    assert by_name["signed_amount_iso"].amount_style == "single_signed"
    for n in ("two_col_dmon", "two_col_slash", "two_col_dual_date", "two_col_ddmmyy_cr"):
        assert by_name[n].amount_style in ("two_column", "auto")
    for n in ("two_col_dmon", "two_col_slash", "signed_amount_iso", "generator_v1"):
        assert by_name[n].two_date_columns is False
        assert by_name[n].balance_suffix_cr is False


def test_registry_profiles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        REGISTRY[0].name = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------
# each layout selects its own profile (mapping required by Requirement 7.2)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_each_fixture_layout_selects_its_own_profile_with_full_coverage(name):
    fx = ALL_FIXTURES[name]()
    choice = choose_profile(fx.pages)
    assert choice.profile is not None
    assert choice.profile.name == name
    assert choice.coverage == pytest.approx(1.0)
    assert set(choice.per_page) == {p.page for p in fx.pages}
    assert all(v == pytest.approx(1.0) for v in choice.per_page.values())
    assert _layout_findings(choice) == []
    assert len(choice.rows) == fx.n_txns


def test_continuation_pages_select_profile_with_full_coverage_on_both_pages():
    fx = continuation_fixture()
    choice = choose_profile(fx.pages)
    assert choice.profile is not None and choice.profile.name == "two_col_slash"
    assert set(choice.per_page) == {1, 2}
    assert all(v == pytest.approx(1.0) for v in choice.per_page.values())
    assert _layout_findings(choice) == []


def test_six_page_file_all_marked_page_1_of_1_selects_two_col_dmon():
    fx = six_page_page1of1_fixture()
    choice = choose_profile(fx.pages)
    assert choice.profile is not None and choice.profile.name == "two_col_dmon"
    assert set(choice.per_page) == set(range(1, 7))
    assert choice.coverage == pytest.approx(1.0)
    assert _layout_findings(choice) == []


def test_selection_ignores_the_bank_name():
    """Rename the bank in every line/word: the layout is still recognised."""
    fx = ALL_FIXTURES["two_col_slash"]()
    bank = "Corvane"

    def scrub(t: str) -> str:
        return t.replace(bank, "Zzyzx")

    pages = [
        PageContent(
            page=p.page,
            lines=[Line(scrub(ln.text), ln.top) for ln in p.lines],
            words=[dataclasses.replace(w, text=scrub(w.text)) for w in p.words],
        )
        for p in fx.pages
    ]
    choice = choose_profile(pages)
    assert choice.profile is not None and choice.profile.name == "two_col_slash"
    assert choice.coverage == pytest.approx(1.0)


# --------------------------------------------------------------------------
# ties (use the parse_page seam to construct them deterministically)
# --------------------------------------------------------------------------
def _page(n: int = 1) -> PageContent:
    return PageContent(page=n, lines=[Line("anything", 10.0)], words=[Word("anything", 0, 40, 10.0)])


def test_tie_with_identical_output_goes_to_earlier_registry_profile(monkeypatch):
    same = [("2026-07-02", -1200.0), ("2026-07-04", 3500.0)]
    monkeypatch.setattr(layouts, "parse_page", lambda page, prof: PageParse(rows=list(same), coverage=1.0))
    choice = choose_profile([_page()])
    assert choice.profile is REGISTRY[0]
    assert choice.coverage == pytest.approx(1.0)
    assert _layout_findings(choice) == []
    assert list(choice.rows) == same


def test_tie_between_later_profiles_goes_to_earlier_of_the_tied(monkeypatch):
    tied = {REGISTRY[2].name, REGISTRY[4].name}

    def fake(page, prof):
        if prof.name in tied:
            return PageParse(rows=[("r", 1.0)], coverage=0.95)
        return PageParse(rows=[], coverage=0.2)

    monkeypatch.setattr(layouts, "parse_page", fake)
    choice = choose_profile([_page()])
    assert choice.profile is REGISTRY[2]
    assert choice.coverage == pytest.approx(0.95)


def test_tie_with_different_output_gives_layout_unrecognised(monkeypatch):
    def fake(page, prof):
        if prof.name in (REGISTRY[1].name, REGISTRY[3].name):
            return PageParse(rows=[(prof.name, 1.0)], coverage=1.0)
        return PageParse(rows=[], coverage=0.1)

    monkeypatch.setattr(layouts, "parse_page", fake)
    choice = choose_profile([_page(3)])
    found = _layout_findings(choice)
    assert len(found) == 1
    assert "page 3" in found[0].evidence.lower()
    assert choice.profile is None  # not chosen: the tie was not resolved silently


def test_higher_coverage_beats_earlier_registry_position(monkeypatch):
    def fake(page, prof):
        if prof is REGISTRY[5]:
            return PageParse(rows=[("a", 1.0)], coverage=1.0)
        return PageParse(rows=[("b", 1.0)], coverage=0.93)

    monkeypatch.setattr(layouts, "parse_page", fake)
    choice = choose_profile([_page()])
    assert choice.profile is REGISTRY[5]


# --------------------------------------------------------------------------
# low coverage -> DQ_LAYOUT_UNRECOGNISED, rows still returned
# --------------------------------------------------------------------------
def _corrupt_one_row_date(fx):
    """Replace one printed date with an impossible one on page 1."""
    old, new = "02/07/2026", "32/13/2026"
    page = fx.pages[0]
    assert any(w.text == old for w in page.words)
    return [
        PageContent(
            page=page.page,
            lines=[Line(ln.text.replace(old, new), ln.top) for ln in page.lines],
            words=[dataclasses.replace(w, text=new) if w.text == old else w for w in page.words],
        )
    ] + list(fx.pages[1:])


def test_coverage_below_threshold_emits_medium_weight_8_naming_the_page_and_keeps_rows():
    fx = ALL_FIXTURES["two_col_slash"]()  # 7 rows -> one bad row = 6/7 = 0.857 < 0.9
    choice = choose_profile(_corrupt_one_row_date(fx))
    assert choice.per_page[1] < 0.9
    found = _layout_findings(choice)
    assert len(found) == 1
    f = found[0]
    assert f.severity == Severity.MEDIUM
    assert f.weight == 8
    assert "page 1" in f.evidence.lower()
    assert len(choice.rows) >= 1  # the rows that were read are still returned
    assert len(choice.rows) < fx.n_txns


def test_only_the_low_coverage_page_is_named():
    fx = six_page_page1of1_fixture()
    page = fx.pages[2]
    garbage = PageContent(
        page=page.page,
        lines=[Line("zz qq 99/99/9999 ??? 1,2,3 4.5.6", 100.0 + 12 * i) for i in range(8)],
        words=[Word("99/99/9999", 40, 90, 100.0 + 12 * i) for i in range(8)],
    )
    pages = fx.pages[:2] + [garbage] + fx.pages[3:]
    choice = choose_profile(pages)
    found = _layout_findings(choice)
    assert len(found) == 1
    assert "page 3" in found[0].evidence.lower()
    assert choice.per_page[3] < 0.9
    assert all(choice.per_page[p] == pytest.approx(1.0) for p in (1, 2, 4, 5, 6))
    assert len(choice.rows) >= 5 * 7  # the five good pages are still read


def test_garbage_text_page_gives_low_coverage():
    junk = PageContent(
        page=1,
        lines=[Line(t, 20.0 + 12 * i) for i, t in enumerate(
            ["lorem ipsum 12 dolor", "99/99/9999 !!! ???", "a b c 1.2.3", "--- 00 ---", "x y z"])],
        words=[Word(t, 10.0 * i, 10.0 * i + 8, 20.0) for i, t in enumerate("lorem ipsum 12 dolor 99/99/9999".split())],
    )
    choice = choose_profile([junk])
    assert choice.per_page[1] < 0.9
    assert choice.coverage < 0.9
    assert len(_layout_findings(choice)) == 1


def test_empty_page_list_does_not_crash():
    choice = choose_profile([])
    assert choice.per_page == {}
    assert _layout_findings(choice) == []


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------
def test_error_inside_one_profile_does_not_crash_selection():
    """Patch parse_page so the profile that would win raises; others still work."""
    fx = ALL_FIXTURES["two_col_dual_date"]()
    real = layouts.parse_page

    def flaky(page, prof):
        if prof.name == "two_col_slash":
            raise RuntimeError("boom")
        return real(page, prof)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(layouts, "parse_page", flaky)
        choice = choose_profile(fx.pages)
    finally:
        mp.undo()
    assert choice.profile is not None and choice.profile.name == "two_col_dual_date"
    assert choice.coverage == pytest.approx(1.0)


def test_error_in_the_correct_profile_is_treated_as_zero_coverage(monkeypatch):
    fx = ALL_FIXTURES["signed_amount_iso"]()
    real = layouts.parse_page

    def flaky(page, prof):
        if prof.name == "signed_amount_iso":
            raise ValueError("boom")
        return real(page, prof)

    monkeypatch.setattr(layouts, "parse_page", flaky)
    choice = choose_profile(fx.pages)  # must not raise
    assert choice.profile is None or choice.profile.name != "signed_amount_iso"
    assert choice.per_page[1] < 0.9
    assert len(_layout_findings(choice)) == 1


def test_all_profiles_raising_does_not_crash(monkeypatch):
    def boom(page, prof):
        raise RuntimeError("boom")

    monkeypatch.setattr(layouts, "parse_page", boom)
    choice = choose_profile([_page()])
    assert choice.per_page[1] == 0.0
    assert len(_layout_findings(choice)) == 1


# --------------------------------------------------------------------------
# 'd Mon' dates are checked against the printed Statement Period
# --------------------------------------------------------------------------
from datetime import date  # noqa: E402

from .fixtures import two_col_dmon_fixture  # noqa: E402

def _dmon_page(out_of_period_days=(), drop_period_line=False) -> PageContent:
    """July two_col_dmon page; the given rows get month 'Jun' (outside the period)."""
    page = two_col_dmon_fixture().pages[0]
    words = list(page.words)
    if out_of_period_days:
        by_top: dict[float, list[Word]] = {}
        for w in words:
            by_top.setdefault(w.top, []).append(w)
        for ws in by_top.values():
            ws.sort(key=lambda w: w.x0)
            if len(ws) > 2 and ws[0].text in out_of_period_days and ws[1].text == "Jul":
                words = [dataclasses.replace(w, text="Jun") if w is ws[1] else w for w in words]
    lines = list(page.lines)
    if drop_period_line:
        top = next(ln.top for ln in lines if "Statement Period" in ln.text)
        lines = [ln for ln in lines if "Statement Period" not in ln.text]
        words = [w for w in words if not (w.top == top and w.x0 >= 300)]
    return PageContent(page=page.page, lines=lines, words=words)


def _dmon() -> LayoutProfile:
    return next(p for p in REGISTRY if p.name == "two_col_dmon")


def test_dmon_dates_outside_the_printed_period_are_not_valid_rows():
    base = layouts.parse_page(_dmon_page(), _dmon())
    assert base.coverage == pytest.approx(1.0) and len(base.rows) == 7
    # 4 of the 7 rows now read '<d> Jun' inside a July period
    result = layouts.parse_page(_dmon_page(out_of_period_days=("4", "15", "22", "28")), _dmon())
    assert result.coverage < 1.0
    assert len(result.rows) < 7


def test_page_without_period_line_accepts_calendar_valid_dmon_dates():
    page = _dmon_page(out_of_period_days=("4", "15", "22", "28"), drop_period_line=True)
    result = layouts.parse_page(page, _dmon())
    assert result.coverage == pytest.approx(1.0)
    assert len(result.rows) == 7


def test_continuation_page_uses_the_previous_pages_period():
    page = _dmon_page(drop_period_line=True)
    july = (date(2026, 7, 1), date(2026, 7, 31))
    april = (date(2026, 4, 1), date(2026, 4, 30))
    assert layouts.parse_page(page, _dmon(), july).coverage == pytest.approx(1.0)
    assert layouts.parse_page(page, _dmon(), april).coverage < 1.0


def test_choose_profile_carries_period_onto_page_without_period_line():
    first = _dmon_page()
    second = dataclasses.replace(_dmon_page(drop_period_line=True), page=2)
    choice = choose_profile([first, second])
    assert choice.profile is not None and choice.profile.name == "two_col_dmon"
    assert all(v == pytest.approx(1.0) for v in choice.per_page.values())
