"""Tampers that change a finished PDF: overlay, font, metadata and re-save.

``apply_file_ops(pdf, file_ops, row_map, seed)`` applies recorded :class:`FileOp` s:

    overlay       draw ``new`` at the mapped cell (optional white box first)
    set_font      same as overlay, in the built-in font named by ``font`` (Courier)
    set_metadata  rewrite Producer / dates; ``params`` keys: producer, created
                  (ISO, optional), modified_delta_s (seconds after the creation moment)
    resave        same as set_metadata (a print/export style re-save)

An overlay keeps the original text: both the old and the new value stay in the text
layer. The new text is drawn a little below the old line (a sloppy forgery) so that
both values stay readable as separate words. Only reportlab built-in fonts are used.
"""
from __future__ import annotations

import io
import random
import re
from datetime import datetime, timedelta

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from fraud_detect.corpus.finish import pdf_date
from fraud_detect.corpus.layouts.base import baseline_from_top
from fraud_detect.corpus.layouts.dataset_family import fmt_date
from fraud_detect.corpus.schema import (
    FileOp,
    Ledger,
    RowMap,
    StyleKnobs,
    TamperLabel,
    TamperResult,
)
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper._registry import info_for, register

OVERLAY_DROP = 3.6          # points the forged text sits below the original line
_LEFT_FIELDS = ("date", "description")
_NUMBER = re.compile(r"\d[\d,]*\.\d{2}")

EDITOR_PRODUCERS = (
    "PixelForge PDF Editor 5.2",
    "DocuMender PDF Editor 11",
    "OpenLeaf PDF Editor",
    "SheetSmith PDF Editor Pro",
)
EXPORT_PRODUCERS = (
    "Mac OS X Quartz PDFContext",
    "Skia/PDF m118",
    "Microsoft: Print To PDF",
    "GPL Ghostscript 9.55",
)

OVERLAY_NOTE = ("Overlay: the old value stays in the text layer under the new one. "
                "Node 1 might drop this row without a warning (known blind spot).")


# ---------------------------------------------------------------------------
# applying the operations
# ---------------------------------------------------------------------------
def _draw_overlay(c: Canvas, op: FileOp, cell, page_h: float) -> None:
    font = op.font or "Helvetica"
    width = stringWidth(cell.text, "Helvetica", 1.0)
    size = round((cell.x1 - cell.x0) / width / 0.05) * 0.05 if width > 0 else 8.0
    size = size if 4.0 <= size <= 16.0 else 8.0
    left = op.field in _LEFT_FIELDS
    new_w = stringWidth(op.new, font, size)
    x0 = cell.x0 if left else cell.x1 - new_w
    base_old = baseline_from_top(page_h, cell.top, size)
    base_new = baseline_from_top(page_h, cell.top + OVERLAY_DROP, size)
    if op.background:
        lo = min(x0, cell.x0) - 1.0
        hi = max(x0 + new_w, cell.x1) + 1.0
        c.setFillColorRGB(1, 1, 1)
        c.rect(lo, base_new - 0.25 * size, hi - lo, (base_old - base_new) + 1.1 * size,
               stroke=0, fill=1)
        c.setFillColorRGB(0, 0, 0)
    c.setFont(font, size)
    c.drawString(x0, base_new, op.new)


def _apply_overlays(pdf: bytes, ops: list[FileOp], row_map: RowMap) -> bytes:
    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter(clone_from=reader)
    meta = dict(reader.metadata or {})
    by_page: dict[int, list[tuple[FileOp, object]]] = {}
    for op in ops:
        cell = row_map.cells[op.row_id][op.field]
        by_page.setdefault(cell.page, []).append((op, cell))
    for pno, items in sorted(by_page.items()):
        page = writer.pages[pno - 1]
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = Canvas(buf, pagesize=(w, h), invariant=1, pageCompression=0)
        for op, cell in items:
            _draw_overlay(c, op, cell, h)
        c.showPage()
        c.save()
        page.merge_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
    if meta:
        writer.add_metadata({str(k): str(v) for k, v in meta.items()})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _apply_metadata(pdf: bytes, op: FileOp) -> bytes:
    params = dict(op.params)
    reader = PdfReader(io.BytesIO(pdf))
    meta = reader.metadata
    orig = meta.creation_date if meta is not None else None
    orig = orig.replace(tzinfo=None) if orig is not None else datetime(2026, 1, 1, 9, 0, 0)
    created = datetime.fromisoformat(params["created"]) if "created" in params else orig
    delta = timedelta(seconds=int(params.get("modified_delta_s", "86400")))
    modified = (orig if "created" in params else created) + delta
    writer = PdfWriter(clone_from=reader)
    info = {str(k): str(v) for k, v in (meta or {}).items()}
    # A missing "producer" keeps the PDF's own Producer (Balanced_Styling ops).
    if "producer" in params:
        info["/Producer"] = params["producer"]
    info.update({"/CreationDate": pdf_date(created), "/ModDate": pdf_date(modified)})
    writer.add_metadata(info)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def apply_file_ops(pdf: bytes, file_ops, row_map: RowMap, seed: int) -> bytes:
    """Apply recorded file operations to a finished PDF (deterministic)."""
    ops = list(file_ops)
    overlays = [o for o in ops if o.op in ("overlay", "set_font")]
    if overlays:
        pdf = _apply_overlays(pdf, overlays, row_map)
    for op in ops:
        if op.op in ("set_metadata", "resave"):
            pdf = _apply_metadata(pdf, op)
        elif op.op not in ("overlay", "set_font"):
            raise ValueError(f"unknown file op {op.op!r}")
    return pdf


# ---------------------------------------------------------------------------
# planning helpers (used here and by ledger_ops for the overlay variants)
# ---------------------------------------------------------------------------
def cell_text(ledger: Ledger, renderer, row_id: str, field: str) -> str:
    """Printed text of one cell of the genuine rendering."""
    res = renderer.render(ledger, StyleKnobs(), 0)
    return res.row_map.cells[row_id][field].text


def replace_number(text: str, value: float) -> str:
    """Replace the number in a money cell, keeping prefix, sign and suffix."""
    return _NUMBER.sub(f"{abs(value):,.2f}", text, count=1)


def date_format_of(renderer) -> str:
    return getattr(getattr(renderer, "spec", None), "row_date_fmt", "%d/%m/%Y")


def format_date(renderer, d) -> str:
    return fmt_date(d, date_format_of(renderer))


def overlay_result(ledger: Ledger, name: str, row_id: str, field: str, original: str,
                   new: str, new_text: str, rng: random.Random, *, op: str = "overlay",
                   font: str | None = None, date_inconsistent: bool = False,
                   arithmetic_consistent: bool = False, notes: str = "") -> TamperResult:
    """Result of an overlay-style tamper: genuine ledger, one file op, label."""
    background = rng.random() < 0.5
    aff = ledger_ops._affected(ledger, row_id, field, original, new)
    info = info_for(name)
    label = TamperLabel(
        is_tampered=True, tamper_types=(name,), method="overlay",
        arithmetic_consistent=arithmetic_consistent, date_inconsistent=date_inconsistent,
        font_inconsistent=font is not None and font != "Helvetica",
        overlay_bg_matched=background, hidden_text_layer=True,
        affected_rows=(aff,), affected_months=(aff["month"],),
        should_be_seen_by=info.should_be_seen_by,
        notes=(notes + " " if notes else "") + OVERLAY_NOTE)
    fop = FileOp(op=op, row_id=row_id, field=field, new=new_text, font=font,
                 background=background)
    return TamperResult(ledger=ledger, edits=(), file_ops=(fop,), label=label)


# ---------------------------------------------------------------------------
# font_mismatch
# ---------------------------------------------------------------------------
@register("font_mismatch")
def plan_font_mismatch(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    row = ledger.rows[rng.randrange(len(ledger.rows))]
    factor = rng.choice((rng.uniform(1.2, 3.0), rng.uniform(0.2, 0.8)))
    new = round(row.amount * factor, 2)
    if abs(new - row.amount) < 0.01:
        new = round(row.amount + (1.0 if row.amount >= 0 else -1.0), 2)
    text = replace_number(cell_text(ledger, renderer, row.row_id, "amount"), new)
    return overlay_result(
        ledger, "font_mismatch", row.row_id, "amount", ledger_ops._money(row.amount),
        ledger_ops._money(new), text, rng, op="set_font", font="Courier",
        notes="The changed amount is drawn in Courier; balances left as printed.")


# ---------------------------------------------------------------------------
# metadata_inconsistency and genuine_resaved
# ---------------------------------------------------------------------------
@register("metadata_inconsistency")
def plan_metadata_inconsistency(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    producer = rng.choice(EDITOR_PRODUCERS)
    params = [("producer", producer),
              ("modified_delta_s", str(rng.randint(2, 40) * 86400 + rng.randint(0, 86399)))]
    early = rng.random() < 0.5
    if early:
        first = ledger.months[0]
        created = datetime(first.year, first.month, 1, rng.randint(6, 20), rng.randint(0, 59), 0)
        created -= timedelta(days=rng.randint(1, 30))
        params.append(("created", created.isoformat()))
    info = info_for("metadata_inconsistency")
    label = TamperLabel(
        is_tampered=True, tamper_types=("metadata_inconsistency",), method="file",
        arithmetic_consistent=True, metadata_altered=True,
        should_be_seen_by=info.should_be_seen_by,
        notes=("Editing-tool Producer and a later ModDate"
               + ("; creation date before the statement period." if early else ".")))
    fop = FileOp(op="set_metadata", params=tuple(sorted(params)))
    return TamperResult(ledger=ledger, edits=(), file_ops=(fop,), label=label)


@register("genuine_resaved")
def plan_genuine_resaved(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    params = (("modified_delta_s", str(rng.randint(60, 3 * 86400))),
              ("producer", rng.choice(EXPORT_PRODUCERS)))
    label = TamperLabel(
        is_tampered=False, tamper_types=("genuine_resaved",), method="file",
        arithmetic_consistent=True, genuine_resaved=True,
        should_be_seen_by=info_for("genuine_resaved").should_be_seen_by,
        notes="Hard negative: content unchanged, print/export Producer and a later ModDate.")
    return TamperResult(ledger=ledger, edits=(), file_ops=(FileOp(op="resave", params=params),),
                        label=label)
