"""Task 3.1: tests for the renderer base and finishing (Requirements 2, 3, 10.3, 12.8, 18.5).

Written before the modules exist (task 3.2). Until then this module fails at import with
``ModuleNotFoundError`` for ``fraud_detect.corpus.layouts.base`` / ``fraud_detect.corpus.finish``.

The behaviour tests are generic: they run against every layout that has a renderer and
skip the rest with a clear reason, so they cover each layout as it is added.
"""
from __future__ import annotations

import re
from dataclasses import replace

import pytest

import fraud_detect.corpus.finish  # noqa: F401  (task 3.2)
import fraud_detect.corpus.layouts.base as layout_base  # noqa: F401  (task 3.2)
from fraud_detect.corpus.schema import LAYOUTS, CellBox
from tests.corpus.helpers import (
    fonts_of,
    is_builtin_font,
    load_renderer,
    make_test_ledger,
    metadata_of,
    page_texts_of,
    render_ledger,
    text_at,
    words_of,
)

SEED = 4242
N_MONTHS = 2
CELLS = ("date", "description", "amount", "balance")
_cache: dict[str, tuple] = {}


def _renderer_or_skip(layout: str):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout!r} is not implemented yet (renderer registry has no entry)")
    return renderer


def _rendered(layout: str):
    """Ledger + RenderResult for ``layout`` (built once per layout)."""
    if layout not in _cache:
        renderer = _renderer_or_skip(layout)
        ledger = make_test_ledger(N_MONTHS, SEED, getattr(renderer, "max_rows_per_month", None))
        _cache[layout] = (renderer, ledger, render_ledger(renderer, ledger, SEED))
    return _cache[layout]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


pytestmark = pytest.mark.parametrize("layout", LAYOUTS)


def test_base_module_exposes_interface(layout):
    assert hasattr(layout_base, "LayoutRenderer")


def test_row_map_covers_every_ledger_row(layout):
    _, ledger, res = _rendered(layout)
    cells = res.row_map.cells
    assert [r.row_id for r in ledger.rows] == [p.row_id for p in res.printed]
    assert set(cells) == {r.row_id for r in ledger.rows}
    for row_id, entry in cells.items():
        assert set(entry) >= set(CELLS), row_id
        assert all(isinstance(c, CellBox) for c in entry.values())


def test_text_at_mapped_position_equals_printed_cell(layout):
    _, ledger, res = _rendered(layout)
    words = words_of(res.pdf)
    for row_id, entry in res.row_map.cells.items():
        for name in CELLS:
            box = entry[name]
            found = text_at(words, box.page, box.top, box.x0, box.x1)
            assert _norm(found) == _norm(box.text), (row_id, name, found, box)


def test_printed_rows_match_row_map_text(layout):
    _, _, res = _rendered(layout)
    for p in res.printed:
        entry = res.row_map.cells[p.row_id]
        assert _norm(entry["date"].text) == _norm(p.date_text)
        assert _norm(entry["description"].text) == _norm(p.description_text)
        assert _norm(entry["amount"].text) == _norm(p.amount_text)
        assert _norm(entry["balance"].text) == _norm(p.balance_text)


def test_every_page_has_specimen_footer(layout):
    _, _, res = _rendered(layout)
    texts = page_texts_of(res.pdf)
    assert len(texts) == res.n_pages >= 1
    for i, text in enumerate(texts, start=1):
        low = text.lower()
        assert "specimen" in low and "synthetic" in low, f"page {i} lacks the specimen footer"


def test_title_and_subject_say_synthetic(layout):
    _, _, res = _rendered(layout)
    meta = metadata_of(res.pdf)
    assert "synthetic" in meta.get("/Title", "").lower()
    assert "synthetic" in meta.get("/Subject", "").lower()


def test_only_reportlab_builtin_fonts(layout):
    _, _, res = _rendered(layout)
    used = fonts_of(res.pdf)
    assert used, "no fonts found"
    assert all(is_builtin_font(f) for f in used), sorted(used)


def test_printed_holder_is_used_when_set(layout):
    renderer, ledger, _ = _rendered(layout)
    other = "ZZ MISMATCH HOLDINGS PTE. LTD."
    res = render_ledger(renderer, replace(ledger, printed_holder=other), SEED)
    first_page = _norm(page_texts_of(res.pdf)[0]).upper()
    assert other in first_page
    # and the default is the company name
    base_text = _norm(page_texts_of(_rendered(layout)[2].pdf)[0]).upper()
    assert ledger.company.name.upper() in base_text
    assert other not in base_text


def test_same_seed_gives_equal_words_and_metadata(layout):
    renderer, ledger, first = _rendered(layout)
    again = render_ledger(renderer, ledger, SEED)
    assert words_of(first.pdf) == words_of(again.pdf)
    assert metadata_of(first.pdf) == metadata_of(again.pdf)
    assert first.row_map == again.row_map
