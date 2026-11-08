"""Layout profiles and automatic profile selection (Requirement 7).

A profile is a small bundle of settings. Every profile parses every page and
the profile with the highest coverage wins (never the bank name). Coverage is
the share of table-looking lines that became a valid row: the date was read in
one of the profile's styles (``d Mon`` dates use the section period for the
year), the amount cell(s) match the profile's amount style, and the balance
was read (with a ``CR`` suffix when the profile expects one).

Selection is deterministic and file-level: the profile with the highest mean
page coverage wins; ties go to the earlier registry profile unless the tied
profiles read different rows, in which case nothing is decided and
``DQ_LAYOUT_UNRECOGNISED`` is raised. Pages below ``MIN_PAGE_COVERAGE`` also
get ``DQ_LAYOUT_UNRECOGNISED`` (rows read are still returned). Errors inside a
profile count as coverage 0; selection never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

from ...models import Finding, Severity
from .dates import parse_day_month, parse_full_date
from .header import AGENT, read_header
from .table import (
    RawRow,
    TableLayout,
    _group_lines,
    _header_columns,
    _is_noise,
    _line_text,
    _read_cells,
    _read_date,
    find_table,
    read_rows,
)
from .types import PageContent

CODE = "DQ_LAYOUT_UNRECOGNISED"
MIN_PAGE_COVERAGE = 0.9
_EPS = 1e-9


@dataclass(frozen=True)
class LayoutProfile:
    name: str
    date_styles: tuple[str, ...]
    period_separators: tuple[str, ...]
    holder_labels: tuple[str, ...]
    amount_style: Literal["two_column", "single_signed", "auto"]
    two_date_columns: bool
    balance_suffix_cr: bool
    # Extra date styles accepted only on a page that also has rows in one of
    # ``date_styles``. Lets a profile keep rows whose date format was changed
    # (a tamper signal Node 2 looks for) without claiming pages of a layout
    # that prints only the secondary style.
    secondary_date_styles: tuple[str, ...] = ()


REGISTRY: tuple[LayoutProfile, ...] = (
    # generator_v1 accepts yyyy-mm-dd alongside dd/mm/yyyy, as the old fixed
    # regex did (the generator's "mixed date formats" tamper relies on it).
    LayoutProfile("generator_v1", ("dd/mm/yyyy",), (" - ",), ("Account Holder:", "UEN:"),
                  "single_signed", False, False, ("yyyy-mm-dd",)),
    LayoutProfile("two_col_dmon", ("dd Mon",), (" to ",), ("Account No.",),
                  "two_column", False, False),
    LayoutProfile("two_col_slash", ("dd/mm/yyyy",), (" to ",), ("Account No.",),
                  "two_column", False, False),
    LayoutProfile("two_col_dual_date", ("dd Mon yyyy",), (" to ",), ("Account No.",),
                  "two_column", True, False),
    LayoutProfile("signed_amount_iso", ("yyyy-mm-dd",), (" to ",), ("Account No.",),
                  "single_signed", False, False),
    LayoutProfile("two_col_ddmmyy_cr", ("dd-mm-yy",), (" to ",), ("Account No.",),
                  "two_column", False, True),
)

_STYLE_RE = {
    "dd/mm/yyyy": re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
    "yyyy-mm-dd": re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
    "dd Mon yyyy": re.compile(r"\d{1,2} [A-Za-z]{3,9} \d{4}"),
    "dd Mon": re.compile(r"\d{1,2} [A-Za-z]{3,9}"),
    "dd-mm-yy": re.compile(r"\d{1,2}-\d{1,2}-\d{2}"),
}
_DATELIKE_RE = re.compile(r"\d{1,4}[/-]\d{1,4}[/-]\d{2,4}")
_LEAP = (date(2024, 1, 1), date(2024, 12, 31))


@dataclass
class PageParse:
    rows: list = field(default_factory=list)
    coverage: float = 0.0


@dataclass
class ProfileChoice:
    profile: Optional[LayoutProfile]
    coverage: float
    per_page: dict[int, float]
    rows: list = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


# ---------------------------------------------------------------- one page
def _date_ok(text: str, profile: LayoutProfile, period: tuple[date, date],
             styles: Optional[tuple[str, ...]] = None) -> bool:
    for style in (profile.date_styles if styles is None else styles):
        rx = _STYLE_RE.get(style)
        if rx is None or rx.fullmatch(text) is None:
            continue
        if style == "dd Mon":
            if parse_day_month(text, period[0], period[1]).value is not None:
                return True
        elif parse_full_date(text).value is not None:
            return True
    return False


def _row_ok(row: RawRow, profile: LayoutProfile, layout: Optional[TableLayout],
            period: tuple[date, date], styles: Optional[tuple[str, ...]] = None) -> bool:
    if not _date_ok(row.date_text, profile, period, styles):
        return False
    if profile.two_date_columns:
        if row.value_date_text is None or not _date_ok(row.value_date_text, profile, period, styles):
            return False
    elif row.value_date_text is not None:
        return False
    if len(row.amount_cells) != 1:
        return False
    if layout is not None and profile.amount_style != "auto":
        want = "two_column" if profile.amount_style == "two_column" else "single_signed"
        if layout.kind != want:
            return False
        col = row.amount_cells[0].column
        if want == "two_column" and col not in ("money_out", "money_in"):
            return False
        if want == "single_signed" and col != "amount":
            return False
    if layout is not None and layout.has_value_date != profile.two_date_columns:
        return False
    has_cr = (row.balance.suffix or "").upper() == "CR"
    return has_cr == profile.balance_suffix_cr


def _count_candidates(page: PageContent, layout: Optional[TableLayout]) -> int:
    n = 0
    for line in _group_lines(page.words):
        if layout is not None and line[0].top <= layout.header_y + 1.25:
            continue
        if _header_columns(line) is not None:
            continue
        text = _line_text(line)
        if _is_noise(text):
            continue
        tokens = [w.text for w in line]
        n_date = _read_date(tokens)
        datelike = n_date > 0 or _DATELIKE_RE.fullmatch(tokens[0]) is not None
        first_cell, cells = _read_cells(line[n_date:])
        if layout is not None:
            cells = [c for c in cells if c[1] >= layout.desc_x0 - 1.0]
        if not (datelike or cells):
            continue
        # first_cell indexes into line[n_date:], so the description is
        # line[n_date : n_date + first_cell]
        desc_text = " ".join(w.text for w in line[n_date:n_date + first_cell])
        if n_date and cells and _is_noise(desc_text):
            continue
        n += 1
    return n


def _printed_period(page: PageContent) -> Optional[tuple[date, date]]:
    """The Statement Period printed on this page, or None when there is none."""
    try:
        h = read_header(page.lines, page.words, None, page.page)
        s = h.period_start.value if h.period_start else None
        e = h.period_end.value if h.period_end else None
    except (ValueError, TypeError):  # malformed text only; real bugs must surface
        return None
    if s is not None and e is not None and s <= e:
        return s, e
    return None


def _period_of(page: PageContent, carried: Optional[tuple[date, date]] = None) -> tuple[date, date]:
    """Printed period, else the previous page's period, else accept any calendar day."""
    return _printed_period(page) or carried or _LEAP


def parse_page(page: PageContent, profile: LayoutProfile,
               carried_period: Optional[tuple[date, date]] = None) -> PageParse:
    """Parse one page with one profile.

    ``carried_period`` is the previous page's period, used only when this page
    prints none (continuation pages).
    """
    layout = find_table(page.words, profile)
    period = _period_of(page, carried_period)
    good: list[RawRow] = []
    secondary: list[RawRow] = []
    has_primary = False
    for row in read_rows(page.words, layout, profile):
        if _row_ok(row, profile, layout, period):
            has_primary = True
        elif not (profile.secondary_date_styles and _row_ok(
                row, profile, layout, period, profile.secondary_date_styles)):
            continue
        else:
            secondary.append(row)
        row.page = page.page
        good.append(row)
    if secondary and not has_primary:
        good = []  # secondary styles never claim a page on their own
    total = _count_candidates(page, layout)
    coverage = min(1.0, len(good) / total) if total else 0.0
    return PageParse(rows=good, coverage=coverage)


# ---------------------------------------------------------------- selection
def _safe_parse(page: PageContent, profile: LayoutProfile,
                carried: Optional[tuple[date, date]] = None) -> PageParse:
    try:
        # parse_page is looked up at call time (tests patch it)
        result = (parse_page(page, profile) if carried is None
                  else parse_page(page, profile, carried))
        cov = float(result.coverage)
        if not 0.0 <= cov <= 1.0:
            return PageParse([], 0.0)
        return PageParse(list(result.rows), cov)
    except Exception:
        return PageParse([], 0.0)


def _finding(page_no: int, coverage: float, undecided: bool) -> Finding:
    why = (
        "two layouts read the page equally well but differently"
        if undecided
        else f"only {coverage:.0%} of table-looking lines could be read as rows"
    )
    return Finding(
        agent=AGENT, code=CODE, title="Statement layout not recognised",
        severity=Severity.MEDIUM, weight=8, field="layout", page=page_no,
        evidence=f"Layout not recognised on page {page_no}: {why}. "
                 "The page may be incomplete; rows that were read are kept.",
    )


def choose_profile(pages: list[PageContent]) -> ProfileChoice:
    if not pages:
        return ProfileChoice(None, 0.0, {}, [], [])
    carried: list[Optional[tuple[date, date]]] = []
    last: Optional[tuple[date, date]] = None
    for pg in pages:
        carried.append(last)
        last = _printed_period(pg) or last
    results = {p.name: [_safe_parse(pg, p, carried[i]) for i, pg in enumerate(pages)]
               for p in REGISTRY}
    totals = {name: sum(r.coverage for r in rs) for name, rs in results.items()}
    best = max(totals.values())
    tied = [p for p in REGISTRY if abs(totals[p.name] - best) <= _EPS]

    def rows_of(p: LayoutProfile) -> list:
        return [row for r in results[p.name] for row in r.rows]

    chosen: Optional[LayoutProfile] = tied[0]
    if len(tied) > 1:
        first = rows_of(tied[0])
        if any(rows_of(p) != first for p in tied[1:]):
            chosen = None

    if chosen is None:
        per_page = {pg.page: max(results[p.name][i].coverage for p in tied)
                    for i, pg in enumerate(pages)}
        findings = [_finding(pg.page, per_page[pg.page], True) for pg in pages]
        return ProfileChoice(None, best / len(pages), per_page, [], findings)

    chosen_results = results[chosen.name]
    per_page = {pg.page: chosen_results[i].coverage for i, pg in enumerate(pages)}
    findings = [
        _finding(pg.page, per_page[pg.page], False)
        for pg in pages if per_page[pg.page] < MIN_PAGE_COVERAGE
    ]
    return ProfileChoice(chosen, best / len(pages), per_page, rows_of(chosen), findings)
