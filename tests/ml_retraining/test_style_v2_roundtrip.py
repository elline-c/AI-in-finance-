"""Round trip of Balanced_Styling through the Retraining_Pipeline reader (task 9.2).

Builds a tiny style_version 2 corpus, reads every PDF back with ``style_profile`` and
compares the values with the Style_Block of the Manifest. Synthetic data only.

**Validates: Requirements 3.3, 5.4, 9.5, 12.5, 12.7**
"""
from __future__ import annotations

import json

import pytest

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import BuildConfig
from fraud_detect.ml.retraining.corpus_gates import capability_report, styling_balance
from fraud_detect.ml.retraining.style_profile import style_profile

from tests.ml_retraining.conftest import MINI_HOLD_OUT_LAYOUT, MINI_TAMPER_MIX

SEED = 20260811
LAYOUTS = ("generator_v1", "two_col_dmon", "two_col_slash", "two_col_dual_date",
           "signed_amount_iso", "two_col_ddmmyy_cr")
BUILTIN_BODY_FONTS = {"Helvetica", "Helvetica-Bold", "Times-Roman"}


@pytest.fixture(scope="module")
def v2_docs(tmp_path_factory):
    out = tmp_path_factory.mktemp("style_v2_roundtrip") / "corpus"
    cfg = BuildConfig(
        out_dir=out, master_seed=SEED, per_layout_genuine=2, per_layout_tampered=2,
        months_min=6, months_max=6, hold_out_layout=MINI_HOLD_OUT_LAYOUT,
        tamper_mix=MINI_TAMPER_MIX, command_line="pytest style_v2_roundtrip", style_version=2)
    CorpusBuilder(cfg).build()
    docs = []
    for mpath in sorted((out / "manifests").glob("*.json")):
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        pdf = (out / "pdfs" / f"{manifest['doc_id']}.pdf").read_bytes()
        docs.append((manifest, style_profile(pdf)))
    return docs


def test_all_layouts_and_documents_present(v2_docs):
    assert len(v2_docs) == 6 * 4
    assert {m["layout"] for m, _ in v2_docs} == set(LAYOUTS)


def test_street_is_bank_street_in_every_layout(v2_docs):
    wrong = [(m["doc_id"], m["layout"], m["style"]["street"], p.street)
             for m, p in v2_docs if p.street != m["style"]["street"]]
    assert not wrong, f"street read back differs from Style_Block: {wrong}"
    assert all(p.street is not None for _, p in v2_docs)


def test_producer_family_and_creator_match_style_block(v2_docs):
    bad = [(m["doc_id"], m["style"]["producer"], p.producer, m["style"]["producer_family"],
            p.producer_family, m["style"]["creator"], p.creator)
           for m, p in v2_docs
           if (p.producer, p.producer_family, p.creator) !=
           (m["style"]["producer"], m["style"]["producer_family"], m["style"]["creator"])]
    assert not bad, f"producer/family/creator mismatch: {bad}"


def test_fonts_contain_decorative_font_and_stay_builtin_for_genuine(v2_docs):
    bad = []
    for m, p in v2_docs:
        fonts = set(p.fonts)
        if m["style"]["decorative_font"] not in fonts:
            bad.append((m["doc_id"], "decorative font missing", sorted(fonts)))
        if not m["label"]["is_tampered"] and not fonts <= BUILTIN_BODY_FONTS:
            bad.append((m["doc_id"], "genuine fonts outside body set", sorted(fonts)))
    assert not bad, f"font mismatch: {bad}"


def test_creation_hour_matches_except_early_creation(v2_docs):
    bad = []
    for m, p in v2_docs:
        if "metadata_inconsistency" in m["label"]["tamper_types"]:
            continue  # keeps its own recorded 'created' parameter (Requirement 6.5)
        expected = int(m["style"]["creation_time"].split(":")[0])
        if p.creation_hour != expected:
            bad.append((m["doc_id"], expected, p.creation_hour))
    assert not bad, f"creation hour mismatch: {bad}"


def test_style_street_column_never_none(v2_docs):
    rows = [{"is_tampered": bool(m["label"]["is_tampered"]), "style_street": p.street,
             "style_producer": p.producer, "style_producer_family": p.producer_family,
             "style_creator": p.creator, "style_fonts": p.fonts,
             "style_bank_name": p.bank_name, "style_row_step_bucket": p.row_step_bucket,
             "style_creation_hour": p.creation_hour}
            for m, p in v2_docs]
    assert all(r["style_street"] is not None for r in rows)
    # Structure only: a 24-document table cannot pass the share rule.
    report = capability_report(rows)
    assert isinstance(report, list) and all(isinstance(x, str) for x in report)
    failures = styling_balance(rows, min_docs=30, share_points=10)
    assert isinstance(failures, list)
