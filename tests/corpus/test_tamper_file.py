"""File-level tampers: overlay, font, metadata, re-save (task 14.1).

Written first; fails with ImportError until task 14.2 adds
``fraud_detect/corpus/tamper/pdf_ops.py`` (and registers the tampers in the catalog).

Interfaces used (from the design and test_tamper_document.py):
  * ``CATALOG[name].plan(ledger, renderer, rng) -> TamperResult`` (``file_ops``, ``label``)
  * ``apply_file_ops(pdf, file_ops, row_map, seed) -> bytes``
  * pipeline: render ``result.ledger``, then apply the file ops to that PDF with its row map.
Overlay keeps the genuine ledger, so both the old and the new value stay in the text
layer; Node 1 may drop the overlaid row (a blind spot the label notes must state).
Variants (overlay vs rerender, background box, "creation before period") are found by
scanning seeds, as in test_tamper_document.py.
"""
from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from datetime import datetime

import pytest
from pypdf import PdfReader

from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from tests.corpus.helpers import (
    DEFAULT_STYLE,
    fonts_of,
    load_renderer,
    make_test_ledger,
    metadata_of,
    text_at,
    words_of,
)

SEEDS = range(16)
FIELD_TAMPERS = ("edited_amount", "altered_date", "broken_balance")
EDITOR_RE = re.compile(
    r"(?i)acrobat|photoshop|illustrator|indesign|foxit|nitro|xchange|sejda|ilovepdf"
    r"|smallpdf|pdfescape|pdf ?editor|libreoffice|microsoft.*word|canva|gimp|inkscape"
)
EXPORT_RE = re.compile(
    r"(?i)print|export|save as pdf|quartz|skia|cairo|ghostscript|distiller|pdfcreator"
    r"|microsoft|chrome|cups"
)
NODE1_RE = re.compile(r"(?i)node[ _]?1")


# -- helpers ---------------------------------------------------------------
def _renderer():
    r = load_renderer("two_col_slash")
    if r is None:
        pytest.skip("renderer not implemented yet")
    return r


def _run(name: str, seed: int):
    """Plan, render the (possibly edited) ledger and apply the file ops."""
    renderer = _renderer()
    ledger = make_test_ledger(6, 1000 + seed)
    result = CATALOG[name].plan(ledger, renderer, rng(seed, "tamper-file-test", name))
    base = renderer.render(result.ledger, DEFAULT_STYLE, seed)
    out = apply_file_ops(base.pdf, result.file_ops, base.row_map, seed)
    return ledger, result, base, out


def _find(name: str, pred, seeds=SEEDS):
    found = []
    for s in seeds:
        run = _run(name, s)
        if pred(run[1]):
            found.append(run)
    assert found, f"{name}: no matching variant in {len(list(seeds))} seeds"
    return found


def _is_method(method):
    return lambda result: result.label.method == method


def _chars(pdf: bytes) -> Counter:
    import pdfplumber

    c: Counter = Counter()
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        for pno, page in enumerate(doc.pages, start=1):
            for ch in page.chars:
                c[(pno, ch["text"], round(ch["x0"], 1), round(ch["top"], 1))] += 1
    return c


def _text_hash(pdf: bytes) -> str:
    items = [(w.page, w.text, w.x0, w.x1, w.top) for w in words_of(pdf)]
    return hashlib.sha256(repr(items).encode()).hexdigest()


def _dates(pdf: bytes):
    meta = PdfReader(io.BytesIO(pdf)).metadata
    created, modified = meta.creation_date, meta.modification_date
    assert created is not None and modified is not None
    return created.replace(tzinfo=None), modified.replace(tzinfo=None)


def _overlay_op(result):
    ops = [o for o in result.file_ops if o.op == "overlay"]
    assert len(ops) == 1
    return ops[0]


def _cell(base, row_id, field):
    return base.row_map.cells[row_id][field]


# -- overlay ---------------------------------------------------------------
@pytest.mark.parametrize("name", FIELD_TAMPERS)
def test_overlay_keeps_ledger_and_targets_one_affected_row(name):
    for ledger, result, base, out in _find(name, _is_method("overlay")):
        assert result.ledger == ledger
        op = _overlay_op(result)
        assert len(result.label.affected_rows) == 1
        affected = result.label.affected_rows[0]
        assert affected["row_id"] == op.row_id
        assert affected["field"] == op.field
        assert op.row_id in base.row_map.cells and op.field in base.row_map.cells[op.row_id]
        assert op.new
        assert result.label.is_tampered and result.label.tamper_types == (name,)


@pytest.mark.parametrize("name", FIELD_TAMPERS)
def test_overlay_leaves_old_and_new_value_in_text_layer_and_changes_nothing_else(name):
    for ledger, result, base, out in _find(name, _is_method("overlay")):
        op = _overlay_op(result)
        cell = _cell(base, op.row_id, op.field)
        before, after = _chars(base.pdf), _chars(out)
        # every original character is still there: old value and the rest of the page
        assert not (before - after), "overlay removed or moved original text"
        extra = after - before
        drawn = "".join(sorted(k[1] for k in extra.elements()))
        assert drawn == "".join(sorted(op.new.replace(" ", "")))
        # the new value sits at the mapped cell
        for (page, _t, _x, top) in extra:
            assert page == cell.page and abs(top - cell.top) <= 4.0
        # the old value is still extractable at the mapped position
        words = words_of(out)
        assert cell.text.replace(" ", "") in text_at(
            words, cell.page, cell.top, cell.x0, cell.x1
        ).replace(" ", "")
        assert len(words_of(out)) >= len(words_of(base.pdf))


@pytest.mark.parametrize("name", FIELD_TAMPERS)
def test_overlay_label_flags_and_background_box(name):
    seen = set()
    for _ledger, result, _base, _out in _find(name, _is_method("overlay")):
        op = _overlay_op(result)
        assert result.label.hidden_text_layer is True
        assert result.label.overlay_bg_matched is bool(op.background)
        seen.add(bool(op.background))
    assert seen == {True, False}


@pytest.mark.parametrize("name", FIELD_TAMPERS)
def test_overlay_label_says_node1_might_drop_the_row(name):
    for _ledger, result, _base, _out in _find(name, _is_method("overlay")):
        notes = result.label.notes
        assert NODE1_RE.search(notes) and "drop" in notes.lower()
        assert result.label.affected_rows


# -- rerender --------------------------------------------------------------
@pytest.mark.parametrize("name", ["edited_amount", "altered_date"])
def test_rerender_has_only_the_new_value_in_the_text_layer(name):
    renderer = _renderer()
    for ledger, result, base, out in _find(name, _is_method("rerender")):
        assert not result.file_ops
        assert out == base.pdf or _chars(out) == _chars(base.pdf)
        affected = result.label.affected_rows[0]
        new_cell = _cell(base, affected["row_id"], affected["field"])
        genuine = renderer.render(ledger, DEFAULT_STYLE, 0)
        old_cell = _cell(genuine, affected["row_id"], affected["field"])
        assert old_cell.text != new_cell.text
        shown = text_at(words_of(base.pdf), new_cell.page, new_cell.top, new_cell.x0, new_cell.x1)
        assert shown.replace(" ", "") == new_cell.text.replace(" ", "")
        assert old_cell.text.replace(" ", "") not in shown.replace(" ", "")
        assert result.label.hidden_text_layer is False
        assert result.label.overlay_bg_matched is None


# -- font_mismatch ---------------------------------------------------------
def test_font_mismatch_uses_courier_on_the_changed_cell_only():
    renderer = _renderer()
    for ledger, result, base, out in _find("font_mismatch", lambda r: True):
        label = result.label
        assert label.is_tampered and label.tamper_types == ("font_mismatch",)
        assert label.font_inconsistent is True
        assert label.method in ("overlay", "rerender")
        assert len(label.affected_rows) == 1
        affected = label.affected_rows[0]
        cell = _cell(base, affected["row_id"], affected["field"])
        genuine_pdf = renderer.render(ledger, DEFAULT_STYLE, 0).pdf
        assert "Courier" not in fonts_of(genuine_pdf)
        import pdfplumber

        with pdfplumber.open(io.BytesIO(out)) as doc:
            courier = [
                (pno, ch)
                for pno, page in enumerate(doc.pages, start=1)
                for ch in page.chars
                if re.sub(r"^[A-Z]{6}\+", "", ch["fontname"]).startswith("Courier")
            ]
        assert courier, "no Courier text drawn"
        for pno, ch in courier:
            assert pno == cell.page and abs(ch["top"] - cell.top) <= 4.0
        # other text keeps the original fonts
        assert fonts_of(out) - {"Courier"} == fonts_of(genuine_pdf)


# -- metadata_inconsistency ------------------------------------------------
def test_metadata_inconsistency_changes_only_metadata():
    for ledger, result, base, out in _find("metadata_inconsistency", lambda r: True):
        label = result.label
        assert result.ledger == ledger and not result.edits
        assert label.is_tampered and label.tamper_types == ("metadata_inconsistency",)
        assert label.method == "file"
        assert label.metadata_altered is True
        assert label.arithmetic_consistent is True
        assert _text_hash(out) == _text_hash(base.pdf)
        assert _chars(out) == _chars(base.pdf)
        assert EDITOR_RE.search(metadata_of(out).get("/Producer", ""))
        created, modified = _dates(out)
        assert modified > created


def test_metadata_inconsistency_creation_before_period_variant():
    early, normal = [], []
    for ledger, result, _base, out in _find("metadata_inconsistency", lambda r: True):
        created, modified = _dates(out)
        first = datetime(ledger.months[0].year, ledger.months[0].month, 1)
        assert modified > created
        (early if created < first else normal).append(created)
    assert early, "no 'creation before period' variant in the seeds"
    assert normal, "every document is the 'creation before period' variant"


# -- genuine_resaved -------------------------------------------------------
def test_genuine_resaved_keeps_pages_and_is_not_tampered():
    for ledger, result, base, out in _find("genuine_resaved", lambda r: True):
        label = result.label
        assert result.ledger == ledger and not result.edits
        assert label.is_tampered is False
        assert label.genuine_resaved is True
        assert _text_hash(out) == _text_hash(base.pdf)
        assert _chars(out) == _chars(base.pdf)
        producer = metadata_of(out).get("/Producer", "")
        assert EXPORT_RE.search(producer)
        assert not EDITOR_RE.search(producer)
        created, modified = _dates(out)
        assert modified > created
