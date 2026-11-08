"""Task 3.1: tests for fraud_detect/corpus/styling.py (Balanced_Styling, version 2).

Written before the module exists; they fail until task 3.2 is done.
"""
from __future__ import annotations

import dataclasses
import re
from collections import Counter
from pathlib import Path

import pytest

from fraud_detect.corpus import finish, styling
from fraud_detect.corpus.schema import FileOp, TamperLabel, TamperResult
from fraud_detect.corpus.tamper import pdf_ops
from fraud_detect.ml import statement_features
from fraud_detect.nodes import visual_forensics

SEED = 20260801
N_DRAWS = 20000


# --------------------------------------------------------------------------- catalogues
def test_creator_pool_has_five_distinct_strings_including_legacy():
    pool = styling.CREATOR_POOL
    assert len(pool) == 5 and len(set(pool)) == 5
    assert finish.CREATOR in pool
    assert "synthetic-corpus-generator 1.0.0" in pool


def test_producer_catalogue_has_three_families_of_four():
    cat = styling.PRODUCER_CATALOGUE
    assert set(cat) == {"base", "export", "editor"}
    assert tuple(cat["base"]) == finish.PRODUCERS
    assert tuple(cat["export"]) == pdf_ops.EXPORT_PRODUCERS
    assert tuple(cat["editor"]) == pdf_ops.EDITOR_PRODUCERS
    assert all(len(set(v)) == 4 for v in cat.values())


def test_weights_and_pools_are_pinned():
    assert dict(styling.FAMILY_WEIGHTS) == {"base": 0.65, "export": 0.25, "editor": 0.10}
    assert dict(styling.DECORATIVE_FONTS) == {"Helvetica": 0.65, "Times-Roman": 0.35}
    assert set(styling.FOREIGN_FONT_POOL) == {"Courier", "Courier-Bold", "Times-Bold",
                                              "Helvetica-Oblique"}
    assert styling.ADDRESS_TEMPLATES == 3


def test_creator_pool_avoids_editing_software_keywords():
    keywords = list(visual_forensics._EDITING_SOFTWARE) + list(statement_features._EDITING_SOFTWARE)
    for creator in styling.CREATOR_POOL:
        low = creator.lower()
        for kw in keywords:
            assert kw.lower() not in low, f"{creator!r} contains {kw!r}"
        assert not re.search(r"\beditor\b", low), creator


def test_styling_source_documents_node2_consequences():
    src = Path(styling.__file__).read_text(encoding="utf-8")
    comments = "\n".join(line for line in src.splitlines() if "#" in line)
    # module comment may also be a docstring; accept either place, but both codes must appear
    for code in ("VF_EDIT_SOFTWARE", "VF_MODIFIED_AFTER_CREATION"):
        assert code in src, code
        assert code in comments or '"""' in src


# --------------------------------------------------------------------------- make_style_v2
def test_make_style_v2_is_deterministic_and_marked_v2():
    a = styling.make_style_v2(SEED, "doc-1")
    b = styling.make_style_v2(SEED, "doc-1")
    assert a == b
    assert a.style_version == 2 and a.print_address is True
    assert a.producer in styling.PRODUCER_CATALOGUE[a.producer_family]
    assert a.creator in styling.CREATOR_POOL
    assert a.font_regular in {"Helvetica", "Times-Roman"} and a.font_bold == "Helvetica-Bold"
    assert a.address_wording in (0, 1, 2)


def test_make_style_v2_keeps_legacy_values():
    for i in range(25):
        leg = finish.make_style(SEED, f"doc-{i}")
        new = styling.make_style_v2(SEED, f"doc-{i}")
        for f in ("bank_name", "address_lines", "creation_time", "row_step",
                  "second_address_line"):
            assert getattr(new, f) == getattr(leg, f), f


def test_make_style_v2_takes_only_seed_and_parts():
    import inspect
    params = inspect.signature(styling.make_style_v2).parameters
    assert [p.name for p in params.values()] == ["master_seed", "parts"]
    assert params["parts"].kind is inspect.Parameter.VAR_POSITIONAL


def test_make_style_legacy_output_is_unchanged_by_importing_styling():
    # Values recorded on the unchanged generator.
    s = finish.make_style(20260801, "doc-1")
    assert s.producer in finish.PRODUCERS
    assert s.creator == "" and s.print_address is False and s.style_version == 1
    assert s == finish.make_style(20260801, "doc-1")
    assert styling.make_style_v2(20260801, "doc-1").producer != ""


def _draws():
    return [styling.make_style_v2(SEED, f"d{i}") for i in range(N_DRAWS)]


def test_distribution_shares_over_20000_draws():
    draws = _draws()
    n = len(draws)

    def share(counter, key):
        return 100.0 * counter[key] / n

    fam = Counter(d.producer_family for d in draws)
    for key, target in (("base", 65.0), ("export", 25.0), ("editor", 10.0)):
        assert abs(share(fam, key) - target) <= 2.0, (key, share(fam, key))
    font = Counter(d.font_regular for d in draws)
    assert abs(share(font, "Helvetica") - 65.0) <= 2.0
    assert abs(share(font, "Times-Roman") - 35.0) <= 2.0
    cre = Counter(d.creator for d in draws)
    for c in styling.CREATOR_POOL:
        assert abs(share(cre, c) - 20.0) <= 2.0, c
    wording = Counter(d.address_wording for d in draws)
    for w in (0, 1, 2):
        assert abs(share(wording, w) - 100 / 3) <= 2.0, w
    # producers uniform inside a family
    for family, members in styling.PRODUCER_CATALOGUE.items():
        inside = Counter(d.producer for d in draws if d.producer_family == family)
        total = sum(inside.values())
        for p in members:
            assert abs(100.0 * inside[p] / total - 25.0) <= 4.0, p


# --------------------------------------------------------------------------- address_text
def _style(wording, second):
    base = finish.make_style(SEED, "addr")
    return dataclasses.replace(base, address_lines=("22 Sample Avenue", "Singapore 123456"),
                               second_address_line=second, address_wording=wording,
                               print_address=True, style_version=2)


@pytest.mark.parametrize("wording,expected", [
    (0, "22 Sample Avenue, Singapore 123456"),
    (1, "22 Sample Avenue Singapore 123456"),
    (2, "22 Sample Avenue"),
])
def test_address_text_templates(wording, expected):
    assert styling.address_text(_style(wording, True)) == expected


@pytest.mark.parametrize("wording", [0, 1, 2])
def test_address_text_prints_street_only_without_second_line(wording):
    assert styling.address_text(_style(wording, False)) == "22 Sample Avenue"


# --------------------------------------------------------------------------- style_block
def test_style_block_has_the_eleven_manifest_keys():
    s = styling.make_style_v2(SEED, "blk")
    blk = styling.style_block(s)
    assert set(blk) == {"styling_version", "producer", "producer_family", "creator",
                        "decorative_font", "bank_name", "street", "address_wording",
                        "second_address_line", "creation_time", "row_step"}
    assert blk["styling_version"] == 2
    assert blk["producer"] == s.producer and blk["producer_family"] == s.producer_family
    assert blk["creator"] == s.creator and blk["decorative_font"] == s.font_regular
    assert blk["street"] == s.address_lines[0] and blk["bank_name"] == s.bank_name


# --------------------------------------------------------------------------- normaliser
_LEDGER = object()  # the normaliser must pass the ledger through untouched


def _label(**kw):
    kw.setdefault("method", "file")
    return TamperLabel(is_tampered=kw.pop("is_tampered", True), **kw)


def _resaved():
    return TamperResult(
        ledger=_LEDGER, edits=(),
        file_ops=(FileOp(op="resave", params=(("modified_delta_s", "3600"),
                                              ("producer", "Skia/PDF m118"))),),
        label=_label(is_tampered=False, tamper_types=("genuine_resaved",), genuine_resaved=True,
                     notes="Hard negative: content unchanged, print/export Producer and a later ModDate."))


def _meta(early=False):
    params = [("producer", "OpenLeaf PDF Editor"), ("modified_delta_s", "172800")]
    if early:
        params.append(("created", "2025-12-01T09:00:00"))
    return TamperResult(
        ledger=_LEDGER, edits=(),
        file_ops=(FileOp(op="set_metadata", params=tuple(sorted(params))),),
        label=_label(tamper_types=("metadata_inconsistency",), metadata_altered=True,
                     notes="Editing-tool Producer and a later ModDate."))


def _font(font="Courier"):
    return TamperResult(
        ledger=_LEDGER, edits=(),
        file_ops=(FileOp(op="set_font", row_id="r1", field="amount", new="1,234.00",
                         font=font, background=True),),
        label=_label(tamper_types=("font_mismatch",), font_inconsistent=True,
                     notes="The changed amount is drawn in Courier; balances left as printed."))


def test_normaliser_drops_producer_keeps_other_params():
    out = styling.normalise_tamper_result(_resaved(), master_seed=SEED, doc_id="d1")
    assert dict(out.file_ops[0].params) == {"modified_delta_s": "3600"}
    assert out.file_ops[0].op == "resave"
    out = styling.normalise_tamper_result(_meta(early=True), master_seed=SEED, doc_id="d2")
    assert dict(out.file_ops[0].params) == {"modified_delta_s": "172800",
                                            "created": "2025-12-01T09:00:00"}


@pytest.mark.parametrize("make", [_resaved, _meta])
def test_normaliser_appends_producer_sentence_and_keeps_old_notes(make):
    before = make()
    out = styling.normalise_tamper_result(before, master_seed=SEED, doc_id="d3")
    assert out.label.notes.startswith(before.label.notes)
    added = out.label.notes[len(before.label.notes):].lower()
    assert "producer" in added and "styling" in added
    assert out.label.genuine_resaved == before.label.genuine_resaved
    assert out.label.metadata_altered == before.label.metadata_altered
    assert out.ledger is _LEDGER and out.edits == ()


def test_normaliser_font_mismatch_draws_from_pool_and_fixes_notes():
    seen = set()
    for i in range(200):
        out = styling.normalise_tamper_result(_font(), master_seed=SEED, doc_id=f"f{i}")
        op = out.file_ops[0]
        assert op.op == "set_font" and op.font in styling.FOREIGN_FONT_POOL
        assert (op.row_id, op.field, op.new, op.background) == ("r1", "amount", "1,234.00", True)
        assert op.font in out.label.notes
        if not op.font.startswith("Courier"):
            assert "Courier" not in out.label.notes
        assert out.label.font_inconsistent is True
        seen.add(op.font)
    assert seen == set(styling.FOREIGN_FONT_POOL)


def test_normaliser_font_draw_depends_only_on_seed_and_doc_id():
    a = styling.normalise_tamper_result(_font(), master_seed=SEED, doc_id="same")
    b = styling.normalise_tamper_result(_font(), master_seed=SEED, doc_id="same")
    assert a == b


def test_normaliser_leaves_other_results_unchanged():
    other = TamperResult(
        ledger=_LEDGER, edits=(),
        file_ops=(FileOp(op="overlay", row_id="r1", field="amount", new="9.00", background=False),),
        label=_label(tamper_types=("edited_amount",), method="overlay", notes="x"))
    assert styling.normalise_tamper_result(other, master_seed=SEED, doc_id="o1") == other
