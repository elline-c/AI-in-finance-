"""Task 14.3: property tests for overlay and file-level tampers.

Feature: synthetic-corpus-generator, Property 12: Overlay keeps both values; rerender keeps only the new one
Feature: synthetic-corpus-generator, Property 13: File-level tampers leave the pages alone

Rendering is slow, so each layout gets only a handful of examples.
"""
from __future__ import annotations

import hashlib
import io
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pypdf import PdfReader

from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from tests.corpus.helpers import (
    DEFAULT_STYLE,
    load_renderer,
    make_test_ledger,
    metadata_of,
    text_at,
    words_of,
)

PBT = settings(max_examples=5, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
FIELD_TAMPERS = ("edited_amount", "altered_date", "broken_balance")
EDITOR_RE = re.compile(
    r"(?i)acrobat|photoshop|illustrator|indesign|foxit|nitro|xchange|sejda|ilovepdf"
    r"|smallpdf|pdfescape|pdf ?editor|libreoffice|microsoft.*word|canva|gimp|inkscape"
)
EXPORT_RE = re.compile(
    r"(?i)print|export|save as pdf|quartz|skia|cairo|ghostscript|distiller|pdfcreator"
    r"|microsoft|chrome|cups"
)


def _squash(s: str) -> str:
    return s.replace(" ", "")


def _run(name, layout, seed, months):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout} not implemented")
    ledger = make_test_ledger(months, seed)
    result = CATALOG[name].plan(ledger, renderer, rng(seed, "prop-tamper-file", name))
    base = renderer.render(result.ledger, DEFAULT_STYLE, seed)
    out = apply_file_ops(base.pdf, result.file_ops, base.row_map, seed)
    return renderer, ledger, result, base, out


def _text_hash(pdf: bytes) -> str:
    items = [(w.page, w.text, w.x0, w.x1, w.top) for w in words_of(pdf)]
    return hashlib.sha256(repr(items).encode()).hexdigest()


def _dates(pdf: bytes):
    meta = PdfReader(io.BytesIO(pdf)).metadata
    created, modified = meta.creation_date, meta.modification_date
    assert created is not None and modified is not None
    return created.replace(tzinfo=None), modified.replace(tzinfo=None)


cases = st.tuples(st.sampled_from(sorted(LAYOUTS)), st.integers(0, 10_000), st.integers(2, 4))
field_cases = st.tuples(st.sampled_from(FIELD_TAMPERS), cases)


# -- Property 12 -----------------------------------------------------------
@PBT
@given(field_cases)
def test_overlay_keeps_both_values_rerender_keeps_only_new(case):
    """**Validates: Requirements 6.2, 6.4, 6.5**"""
    name, (layout, seed, months) = case
    renderer, ledger, result, base, out = _run(name, layout, seed, months)
    label = result.label
    affected = label.affected_rows[0]
    cell = base.row_map.cells[affected["row_id"]][affected["field"]]

    if label.method == "overlay":
        ops = [o for o in result.file_ops if o.op == "overlay"]
        assert len(ops) == 1
        op = ops[0]
        assert label.hidden_text_layer is True
        cell = base.row_map.cells[op.row_id][op.field]
        before, after = words_of(base.pdf), words_of(out)
        # old value still extractable at the overlaid position
        assert _squash(cell.text) in _squash(
            text_at(after, cell.page, cell.top, cell.x0, cell.x1))
        # new value present near the overlaid position
        near = {_squash(w.text) for w in after
                if w.page == cell.page and abs(w.top - cell.top) <= 4.0
                and w.x1 >= cell.x0 - 80 and w.x0 <= cell.x1 + 80}
        assert _squash(op.new) in near
        # every text element outside the overlaid cell is unchanged
        def outside(ws):
            return sorted(
                (w.page, w.text, round(w.x0, 1), round(w.top, 1)) for w in ws
                if not (w.page == cell.page and abs(w.top - cell.top) <= 4.0
                        and w.x1 >= cell.x0 - 80 and w.x0 <= cell.x1 + 80))
        assert outside(before) == outside(after)
    else:
        assert label.method == "rerender"
        assert label.hidden_text_layer is False
        genuine = renderer.render(ledger, DEFAULT_STYLE, seed)
        old_cell = genuine.row_map.cells[affected["row_id"]][affected["field"]]
        words = words_of(base.pdf)
        shown = _squash(text_at(words, cell.page, cell.top, cell.x0, cell.x1))
        assert shown == _squash(cell.text)
        old = _squash(old_cell.text)
        page_texts = [_squash(w.text) for w in words if w.page == cell.page]
        # if the original was unique on the genuine page, it is gone from the tampered page
        gen_texts = [_squash(w.text) for w in words_of(genuine.pdf) if w.page == old_cell.page]
        if old and gen_texts.count(old) == 1 and old != _squash(cell.text):
            assert old not in page_texts


# -- Property 13 -----------------------------------------------------------
@PBT
@given(cases)
def test_metadata_inconsistency_leaves_pages_alone(case):
    """**Validates: Requirements 5.5, 5.7**"""
    layout, seed, months = case
    _r, _ledger, result, base, out = _run("metadata_inconsistency", layout, seed, months)
    assert result.label.is_tampered is True
    assert _text_hash(out) == _text_hash(base.pdf)
    assert EDITOR_RE.search(metadata_of(out).get("/Producer", ""))
    created, modified = _dates(out)
    assert modified > created


@PBT
@given(cases)
def test_genuine_resaved_leaves_pages_alone(case):
    """**Validates: Requirements 5.5, 5.7**"""
    layout, seed, months = case
    _r, _ledger, result, base, out = _run("genuine_resaved", layout, seed, months)
    assert result.label.is_tampered is False
    assert result.label.genuine_resaved is True
    assert _text_hash(out) == _text_hash(base.pdf)
    producer = metadata_of(out).get("/Producer", "")
    assert EXPORT_RE.search(producer) and not EDITOR_RE.search(producer)
    created, modified = _dates(out)
    assert modified > created
