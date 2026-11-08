"""Balanced_Styling (Styling_Version 2): label-independent styling choices.

Every styling value (Producer family and Producer, Creator, decorative font, address
wording) is drawn from the stream ``("style2", *parts)`` using only the Master_Seed and
the document id, never the label. Bank, street, postcode, creation time, row step and the
second-address flag are copied from the legacy ``finish.make_style`` draw. The draw order
on the new stream is fixed: family, producer, creator, font, wording. Changing it would
be a new styling version.

Node 2 consequences (by design, not a bug):
# Node 2 (visual_forensics) raises VF_EDIT_SOFTWARE for the whole word "editor" in the
# Producer or Creator. Editor-family Producers are drawn for 10% of ALL documents, so some
# genuine Balanced_Styling documents trigger VF_EDIT_SOFTWARE.
# Node 2 raises VF_MODIFIED_AFTER_CREATION for a ModDate more than 60 s after creation
# when the Producer is not a print/export tool. A genuine_resaved document that draws a
# base Producer therefore triggers VF_MODIFIED_AFTER_CREATION on a genuine document.
# The Creator pool avoids the editing-software keywords of Node 2 and Node 6b.

All strings are fictional. ``finish`` does not import this module at module level.
"""
from __future__ import annotations

import dataclasses

from fraud_detect.corpus import finish
from fraud_detect.corpus.schema import FileOp, StyleKnobs, TamperResult
from fraud_detect.corpus.seeding import rng as make_rng
from fraud_detect.corpus.tamper import pdf_ops

FAMILY_WEIGHTS = (("base", 0.65), ("export", 0.25), ("editor", 0.10))
PRODUCER_CATALOGUE = {
    "base": finish.PRODUCERS,
    "export": pdf_ops.EXPORT_PRODUCERS,
    "editor": pdf_ops.EDITOR_PRODUCERS,
}
CREATOR_POOL = (
    finish.CREATOR,
    "StatementForge Batch 2.1",
    "LedgerCore Export Service 6",
    "BankDoc Render Node 7",
    "PrintStream Composer 3.4",
)
DECORATIVE_FONTS = (("Helvetica", 0.65), ("Times-Roman", 0.35))
FOREIGN_FONT_POOL = ("Courier", "Courier-Bold", "Times-Bold", "Helvetica-Oblique")
ADDRESS_TEMPLATES = 3

_LEGACY_FONT_PHRASE = "drawn in Courier"
_PRODUCER_SENTENCE = (" In Balanced_Styling the Producer comes from the shared styling draw "
                      "and is not set by the tamper.")


def _weighted(r, weights):
    x = r.random()
    acc = 0.0
    for name, w in weights:
        acc += w
        if x < acc:
            return name
    return weights[-1][0]


def make_style_v2(master_seed: int, *parts: object) -> StyleKnobs:
    """Legacy draw plus Balanced_Styling values; depends on seed and parts only."""
    legacy = finish.make_style(master_seed, *parts)
    r = make_rng(master_seed, "style2", *parts)
    family = _weighted(r, FAMILY_WEIGHTS)
    producer = r.choice(PRODUCER_CATALOGUE[family])
    creator = r.choice(CREATOR_POOL)
    font = _weighted(r, DECORATIVE_FONTS)
    wording = r.randrange(ADDRESS_TEMPLATES)
    return dataclasses.replace(
        legacy, producer=producer, producer_family=family, creator=creator,
        font_regular=font, font_bold="Helvetica-Bold", print_address=True,
        address_wording=wording, style_version=2)


def address_text(style: StyleKnobs) -> str:
    """The printed bank-address line."""
    street = style.address_lines[0]
    if not style.second_address_line or style.address_wording == 2 or len(style.address_lines) < 2:
        return street
    sep = ", " if style.address_wording == 0 else " "
    return f"{street}{sep}{style.address_lines[1]}"


def normalise_tamper_result(result: TamperResult, *, master_seed: int,
                            doc_id: str) -> TamperResult:
    """Remove label-dependent Producer and font choices from a tamper result."""
    types = result.label.tamper_types
    new_ops = []
    changed = False
    font_name = None
    for op in result.file_ops:
        if op.op in ("set_metadata", "resave") and any(k == "producer" for k, _ in op.params):
            op = dataclasses.replace(
                op, params=tuple((k, v) for k, v in op.params if k != "producer"))
            changed = True
        elif op.op == "set_font":
            if font_name is None:
                font_name = make_rng(master_seed, "font2", doc_id).choice(FOREIGN_FONT_POOL)
            op = dataclasses.replace(op, font=font_name)
            changed = True
        new_ops.append(op)
    label = result.label
    if ("genuine_resaved" in types or "metadata_inconsistency" in types) \
            and _PRODUCER_SENTENCE not in label.notes:          # idempotent
        label = dataclasses.replace(label, notes=label.notes + _PRODUCER_SENTENCE)
        changed = True
    if font_name is not None and "font_mismatch" in types and _LEGACY_FONT_PHRASE in label.notes \
            and f"drawn in {font_name}" not in label.notes:                 # idempotent
        label = dataclasses.replace(
            label, notes=label.notes.replace(_LEGACY_FONT_PHRASE, f"drawn in {font_name}"))
    if not changed:
        return result
    return dataclasses.replace(result, file_ops=tuple(new_ops), label=label)


def style_block(style: StyleKnobs) -> dict:
    """The per-document Manifest 'style' object."""
    return {
        "styling_version": style.style_version,
        "producer": style.producer,
        "producer_family": style.producer_family,
        "creator": style.creator,
        "decorative_font": style.font_regular,
        "bank_name": style.bank_name,
        "street": style.address_lines[0],
        "address_wording": style.address_wording,
        "second_address_line": style.second_address_line,
        "creation_time": style.creation_time,
        "row_step": style.row_step,
    }
