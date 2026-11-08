"""Task 4.5: Balanced_Styling (style_version 2) rendered by all six Layouts, read back.

Properties 6 to 11 of the generator-styling-variety design. Each property has fixed
examples (every Layout, parametrised) and a hypothesis test. The hypothesis tests render
PDFs, so they use 12 examples each (10 to 20 as the task allows) with no deadline.
PDFs are read with ``pdfplumber`` and ``pypdf`` only; Node 1 is the real ingestion.

Feature: generator-styling-variety
"""
from __future__ import annotations

import io
import re
from functools import lru_cache

import pdfplumber
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.corpus import finish, styling
from fraud_detect.corpus.schema import LAYOUTS, FileOp
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from tests.corpus.helpers import (
    compare_rows,
    fonts_of,
    is_builtin_font,
    load_renderer,
    make_test_ledger,
    metadata_of,
    page_texts_of,
    replayed_balances,
)

EXAMPLES = 12  # hypothesis examples for the render-based properties
FIXED_SEEDS = (7001, 7002)
_STREET_RE = re.compile(
    r"\b\d{1,4} [A-Z][A-Za-z]+ (?:Road|Street|Avenue|Lane|Drive|Way|Crescent|Close|Walk|Link)\b"
)

_hyp = settings(max_examples=EXAMPLES, deadline=None)
_cases = st.tuples(
    st.sampled_from(LAYOUTS),
    st.integers(min_value=1, max_value=10_000),
    st.integers(min_value=1, max_value=3),
)


@lru_cache(maxsize=None)
def _render(layout: str, seed: int, n_months: int, version: int):
    """(ledger, style, RenderResult) for one document under the given Styling_Version."""
    renderer = load_renderer(layout)
    assert renderer is not None, layout
    ledger = make_test_ledger(n_months, seed, getattr(renderer, "max_rows_per_month", None))
    doc_id = f"{layout}-{seed:05d}"
    style = styling.make_style_v2(seed, doc_id) if version == 2 else finish.make_style(seed, doc_id)
    return ledger, style, renderer.render(ledger, style, seed)


def _fixed():
    return [(layout, seed, 2) for layout in LAYOUTS for seed in FIXED_SEEDS]


# ---------------------------------------------------------------------------
# Property 6: Producer and Creator survive rendering and a metadata operation
# ---------------------------------------------------------------------------
def _check_p6(layout, seed, n_months):
    ledger, style, res = _render(layout, seed, n_months, 2)
    meta = metadata_of(res.pdf)
    assert meta["/Producer"] == style.producer
    assert meta["/Creator"] == style.creator
    op = FileOp(op="set_metadata", params=(("modified_delta_s", "3600"),))
    out = apply_file_ops(res.pdf, [op], res.row_map, seed)
    meta2 = metadata_of(out)
    assert meta2["/Producer"] == style.producer
    assert meta2["/Creator"] == style.creator
    assert meta2["/ModDate"] != meta["/ModDate"]


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p6_producer_creator_fixed(layout, seed, n_months):
    _check_p6(layout, seed, n_months)


@_hyp
@given(_cases)
def test_p6_producer_creator_property(case):
    """**Validates: Requirements 2.4, 3.3, 9.5**"""
    _check_p6(*case)


# ---------------------------------------------------------------------------
# Property 7: the bank street is the first street-pattern match on page 1
# ---------------------------------------------------------------------------
def _check_p7(layout, seed, n_months):
    _, style, res = _render(layout, seed, n_months, 2)
    first = _STREET_RE.search(page_texts_of(res.pdf)[0])
    assert first is not None, "no street on page 1"
    assert first.group(0) == style.address_lines[0]


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p7_first_street_fixed(layout, seed, n_months):
    _check_p7(layout, seed, n_months)


@_hyp
@given(_cases)
def test_p7_first_street_property(case):
    """**Validates: Requirements 5.4**"""
    _check_p7(*case)


# ---------------------------------------------------------------------------
# Property 8: one non-overlapping address line on every page with the bank name
# ---------------------------------------------------------------------------
def _boxes(pdf_bytes):
    """Per page: list of (text, x0, x1, top, bottom) with a tight x tolerance."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append([(w["text"], w["x0"], w["x1"], w["top"], w["bottom"])
                          for w in page.extract_words(x_tolerance=1, y_tolerance=1)])
    return pages


def _find_sequence(words, tokens):
    n = len(tokens)
    hits = []
    for i in range(len(words) - n + 1):
        if [w[0] for w in words[i:i + n]] == tokens:
            hits.append(i)
    return hits


def _overlap(a, b):
    return a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4]


def _check_p8(layout, seed, n_months):
    _, style, res = _render(layout, seed, n_months, 2)
    texts = page_texts_of(res.pdf)
    pages = _boxes(res.pdf)
    bank_pages = [i for i, t in enumerate(texts) if style.bank_name in t]
    assert bank_pages, "bank name printed on no page"
    addr_tokens = styling.address_text(style).split()
    for i in bank_pages:
        words = pages[i]
        hits = _find_sequence(words, addr_tokens)
        assert len(hits) == 1, f"page {i + 1}: address line found {len(hits)} times"
        start = hits[0]
        addr = words[start:start + len(addr_tokens)]
        for j, w in enumerate(words):
            if start <= j < start + len(addr_tokens):
                continue
            for a in addr:
                assert not _overlap(a, w), f"page {i + 1}: {a[0]!r} overlaps {w[0]!r}"
        # table cells from the Row_Map (1-based pages)
        for cells in res.row_map.cells.values():
            for cell in (cells.values() if isinstance(cells, dict) else cells):
                if cell.page != i + 1:
                    continue
                for a in addr:
                    cell_box = (cell.text, cell.x0, cell.x1, cell.top, cell.top + 8.0)
                    assert not _overlap(a, cell_box), f"{a[0]!r} overlaps cell {cell.text!r}"


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p8_address_line_fixed(layout, seed, n_months):
    _check_p8(layout, seed, n_months)


@_hyp
@given(_cases)
def test_p8_address_line_property(case):
    """**Validates: Requirements 5.1, 5.5**"""
    _check_p8(*case)


# ---------------------------------------------------------------------------
# Property 9: fonts follow the style and stay inside the built-in set
# ---------------------------------------------------------------------------
def _check_p9(layout, seed, n_months):
    _, style, res = _render(layout, seed, n_months, 2)
    fonts = fonts_of(res.pdf)
    assert all(is_builtin_font(f) for f in fonts), fonts
    assert style.font_regular in ("Helvetica", "Times-Roman")
    assert fonts <= {"Helvetica", "Helvetica-Bold", style.font_regular}, fonts
    assert style.font_regular in fonts  # the footer is drawn in the drawn font
    assert "Helvetica" in fonts  # table text keeps the legacy fonts


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p9_fonts_fixed(layout, seed, n_months):
    _check_p9(layout, seed, n_months)


@_hyp
@given(_cases)
def test_p9_fonts_property(case):
    """**Validates: Requirements 4.2, 4.3, 4.4**"""
    _check_p9(*case)


# ---------------------------------------------------------------------------
# Property 10: table geometry does not depend on the Styling_Version
# ---------------------------------------------------------------------------
def _check_p10(layout, seed, n_months):
    _, _, r1 = _render(layout, seed, n_months, 1)
    _, _, r2 = _render(layout, seed, n_months, 2)
    assert r1.n_pages == r2.n_pages
    assert r1.row_map.pages_per_month == r2.row_map.pages_per_month
    assert r1.row_map.cells == r2.row_map.cells


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p10_geometry_fixed(layout, seed, n_months):
    _check_p10(layout, seed, n_months)


@_hyp
@given(_cases)
def test_p10_geometry_property(case):
    """**Validates: Requirements 5.6, 10.5**"""
    _check_p10(*case)


# ---------------------------------------------------------------------------
# Property 11: Node 1 reads the same rows under Balanced_Styling
# ---------------------------------------------------------------------------
def _ingest(layout, seed, n_months, version, tmp_path_factory):
    from fraud_detect.nodes.ingestion import ingest
    _, _, res = _render(layout, seed, n_months, version)
    path = tmp_path_factory.mktemp("p11") / f"{layout}-{seed}-v{version}.pdf"
    path.write_bytes(res.pdf)
    return ingest(str(path))


def _check_p11(layout, seed, n_months, tmp_path_factory):
    ledger, _, _ = _render(layout, seed, n_months, 2)
    ing1 = _ingest(layout, seed, n_months, 1, tmp_path_factory)
    ing2 = _ingest(layout, seed, n_months, 2, tmp_path_factory)
    assert compare_rows(ledger.rows, ing2.transactions, replayed_balances(ledger)) == []
    assert [(t.date, t.amount, t.balance) for t in ing1.transactions] == \
           [(t.date, t.amount, t.balance) for t in ing2.transactions]


@pytest.mark.parametrize("layout,seed,n_months", _fixed())
def test_p11_node1_rows_fixed(layout, seed, n_months, tmp_path_factory):
    _check_p11(layout, seed, n_months, tmp_path_factory)


@_hyp
@given(case=_cases)
def test_p11_node1_rows_property(tmp_path_factory, case):
    """**Validates: Requirements 10.4**"""
    _check_p11(*case, tmp_path_factory)
