"""Task 2.2: style_version validation, tamper-name check, counts, manifest keys, CLI flag.

Feature: generator-styling-variety. Task 5.2 appends the version-2 build tests here.
"""
from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from fraud_detect.corpus.build import main
from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import (
    ALL_TAMPER_TYPES,
    EXTRA_TAMPER_TYPES,
    TAMPER_TYPES,
    BuildConfig,
)

SEED = 20260811
_MIX = (("edited_amount", 1.0), ("font_mismatch", 1.0))


def _cfg(out, **kw) -> BuildConfig:
    base = dict(out_dir=out, master_seed=SEED, per_layout_genuine=1, per_layout_tampered=1,
                months_min=1, months_max=1, tamper_mix=_MIX, command_line="test")
    base.update(kw)
    return BuildConfig(**base)


def _tiny_args(out, *extra) -> list[str]:
    return ["--out", str(out), "--seed", str(SEED), "--per-layout-genuine", "1",
            "--per-layout-tampered", "1", "--months-min", "1", "--months-max", "1",
            "--tamper-mix", "edited_amount=1,font_mismatch=1", *extra]


# Property 1: style_version validation
@settings(max_examples=100, deadline=None)
@given(v=st.integers().filter(lambda x: x not in (1, 2)))
def test_property_1_bad_style_version_refused(v, tmp_path_factory):
    """Feature: generator-styling-variety, Property 1: only 1 and 2 are accepted.

    **Validates: Requirements 1.1, 1.3**
    """
    out = tmp_path_factory.mktemp("sv") / "o"
    with pytest.raises(ValueError) as exc:
        CorpusBuilder(_cfg(out, style_version=v))
    msg = str(exc.value)
    assert "1" in msg and "2" in msg
    assert not out.exists()


@pytest.mark.parametrize("v", [1, 2])
def test_good_style_versions_accepted(tmp_path, v):
    CorpusBuilder(_cfg(tmp_path / "o", style_version=v))


# Property 16: tamper-mix names
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(name=st.text(min_size=1, max_size=20).filter(lambda s: s not in ALL_TAMPER_TYPES))
def test_property_16_unknown_tamper_name_refused(name, tmp_path):
    """Feature: generator-styling-variety, Property 16: unknown mix names are refused.

    **Validates: Requirements 7.10**
    """
    with pytest.raises(ValueError) as exc:
        CorpusBuilder(_cfg(tmp_path / "o", tamper_mix=((name, 1.0),)))
    assert repr(name) in str(exc.value)


@pytest.mark.parametrize("name", ALL_TAMPER_TYPES)
def test_every_known_tamper_name_accepted(tmp_path, name):
    CorpusBuilder(_cfg(tmp_path / "o", tamper_mix=((name, 1.0),)))


# CLI
def test_cli_style_version_2_writes_manifest_key(tmp_path, capsys):
    out = tmp_path / "o"
    assert main(_tiny_args(out, "--style-version", "2")) == 0
    cm = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert cm["styling_version"] == 2
    assert "Styling version : 2 (balanced)" in (out / "README.txt").read_text(encoding="utf-8")


def test_cli_default_writes_no_styling_key(tmp_path):
    out = tmp_path / "o"
    assert main(_tiny_args(out)) == 0
    cm = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert "styling_version" not in cm
    assert "Styling version" not in (out / "README.txt").read_text(encoding="utf-8")


def test_cli_style_version_3_refused(tmp_path, capsys):
    out = tmp_path / "o"
    assert main(_tiny_args(out, "--style-version", "3")) == 2
    assert "Refused:" in capsys.readouterr().err
    assert not out.exists()


# Counts and warnings
def test_counts_without_extra_type_equal_legacy_keys(tmp_path):
    counts = CorpusBuilder(_cfg(tmp_path / "o"))._counts([
        {"layout": "generator_v1", "split": "train", "label": {"tamper_types": ["edited_amount"]}}])
    for split in counts["per_tamper_type"].values():
        assert tuple(split) == TAMPER_TYPES


def test_counts_with_extra_type_lists_it_for_each_split(tmp_path):
    mix = (("subtle_amount_edit", 1.0), ("edited_amount", 1.0))
    b = CorpusBuilder(_cfg(tmp_path / "o", tamper_mix=mix))
    manifests = [
        {"layout": "generator_v1", "split": s, "label": {"tamper_types": []}}
        for s in ("train", "validation")]
    counts = b._counts(manifests)
    for split in counts["per_tamper_type"].values():
        assert split["subtle_amount_edit"] == 0
        assert tuple(split) == TAMPER_TYPES + EXTRA_TAMPER_TYPES


def test_warnings_name_missing_extra_type(tmp_path):
    mix = (("subtle_amount_edit", 1.0),)
    b = CorpusBuilder(_cfg(tmp_path / "o", tamper_mix=mix))
    counts = b._counts([{"layout": "generator_v1", "split": "train",
                         "label": {"tamper_types": ["edited_amount"]}}])
    assert any("subtle_amount_edit" in w for w in b._warnings(counts, []))


# ---------------------------------------------------------------------------
# Task 5.2: the version-2 build
# ---------------------------------------------------------------------------
import io  # noqa: E402

from pypdf import PdfReader  # noqa: E402

from fraud_detect.corpus import builder as builder_mod  # noqa: E402
from fraud_detect.corpus.schema import LABELS_COLUMNS, ROWS_COLUMNS, StyleKnobs  # noqa: E402
from fraud_detect.corpus.styling import make_style_v2, style_block  # noqa: E402
from tests.corpus import golden_tool  # noqa: E402

_BUILD_MIX = (("genuine_resaved", 1.0), ("metadata_inconsistency", 1.0),
              ("font_mismatch", 1.0), ("edited_amount", 1.0))
_STYLE_KEYS = {"styling_version", "producer", "producer_family", "creator", "decorative_font",
               "bank_name", "street", "address_wording", "second_address_line",
               "creation_time", "row_step"}
_BYTE_FILES = ("labels.csv", "rows.csv", "companies.json", "corpus_manifest.json")
_META_OPS = {"genuine_resaved", "metadata_inconsistency"}


def _build(out, style_version):
    cfg = _cfg(out, style_version=style_version, per_layout_genuine=2, per_layout_tampered=4,
               months_min=1, months_max=2, tamper_mix=_BUILD_MIX)
    summary = CorpusBuilder(cfg).build()
    assert not summary.failed
    return out


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2builds")
    return {"v2a": _build(root / "v2a", 2), "v2b": _build(root / "v2b", 2),
            "v1": _build(root / "v1", 1)}


def _manifests(root):
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / "manifests").glob("*.json"))}


def _pdf_meta(root, doc_id):
    return PdfReader(io.BytesIO((root / "pdfs" / f"{doc_id}.pdf").read_bytes())).metadata


def test_v2_build_twice_is_reproducible(builds):
    a, b = builds["v2a"], builds["v2b"]
    for name in _BYTE_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    for p in sorted((a / "manifests").glob("*.json")):
        assert p.read_bytes() == (b / "manifests" / p.name).read_bytes(), p.name
    # text-layer hashes, metadata and font names
    assert golden_tool.collect(a)["pdfs"] == golden_tool.collect(b)["pdfs"]


def test_v2_label_row_company_files_equal_v1(builds):
    for name in ("labels.csv", "rows.csv", "companies.json"):
        assert (builds["v2a"] / name).read_bytes() == (builds["v1"] / name).read_bytes(), name


def test_v2_csv_headers_unchanged(builds):
    labels = (builds["v2a"] / "labels.csv").read_text(encoding="utf-8").splitlines()[0]
    rows = (builds["v2a"] / "rows.csv").read_text(encoding="utf-8").splitlines()[0]
    assert tuple(labels.split(",")) == LABELS_COLUMNS
    assert tuple(rows.split(",")) == ROWS_COLUMNS


def test_v2_manifests_have_style_block_and_all_old_keys(builds):
    v1, v2 = _manifests(builds["v1"]), _manifests(builds["v2a"])
    assert set(v1) == set(v2) and len(v2) == 36
    for doc_id, m in v2.items():
        assert set(m["style"]) == _STYLE_KEYS
        assert m["style"]["styling_version"] == 2
        assert "style" not in v1[doc_id]
        assert set(v1[doc_id]) | {"style"} == set(m)
        assert m["style"] == style_block(make_style_v2(SEED, doc_id))


def test_v2_pdf_producer_creator_match_style_block(builds):
    checked = 0
    for doc_id, m in _manifests(builds["v2a"]).items():
        if set(m["label"]["tamper_types"]) & _META_OPS:
            continue  # set_metadata / resave documents: Creator only (below)
        meta = _pdf_meta(builds["v2a"], doc_id)
        assert str(meta["/Producer"]) == m["style"]["producer"], doc_id
        assert str(meta["/Creator"]) == m["style"]["creator"], doc_id
        checked += 1
    assert checked >= 24


def test_v2_creator_survives_metadata_operations(builds):
    n = 0
    for doc_id, m in _manifests(builds["v2a"]).items():
        if set(m["label"]["tamper_types"]) & _META_OPS:
            assert str(_pdf_meta(builds["v2a"], doc_id)["/Creator"]) == m["style"]["creator"], doc_id
            n += 1
    assert n >= 6


def test_v2_creation_time_equals_style_unless_early(builds):
    seen = 0
    for doc_id, m in _manifests(builds["v2a"]).items():
        raw = (builds["v2a"] / "manifests" / f"{doc_id}.json").read_text(encoding="utf-8")
        if '"created"' in raw:
            continue  # early creation date set by metadata_inconsistency
        created = str(_pdf_meta(builds["v2a"], doc_id)["/CreationDate"])  # D:YYYYMMDDHHMMSSZ
        assert f"{created[10:12]}:{created[12:14]}:{created[14:16]}" == m["style"]["creation_time"], doc_id
        seen += 1
    assert seen >= 24


def test_renderer_receives_make_style_v2_for_every_document(tmp_path, monkeypatch):
    real_get = builder_mod.get_renderer
    seen: dict[int, object] = {}

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def render(self, ledger, style, seed):
            seen[seed] = style
            return self._inner.render(ledger, style, seed)

    monkeypatch.setattr(builder_mod, "get_renderer", lambda name: Spy(real_get(name)))
    cfg = _cfg(tmp_path / "o", style_version=2, per_layout_genuine=2, per_layout_tampered=4,
               months_min=1, months_max=2, tamper_mix=_BUILD_MIX)
    builder = CorpusBuilder(cfg)
    plans = builder.plan()
    assert not builder.build().failed
    # pdf_ops.cell_text renders once more with StyleKnobs() and seed 0 while planning a
    # tamper; that planning render is not a document render, so it is left out here.
    assert seen.pop(0, None) in (None, StyleKnobs())
    assert len(seen) == len(plans) == 36
    assert {p.tamper_type is None for p in plans} == {True, False}
    for p in plans:
        assert seen[p.doc_seed] == make_style_v2(SEED, p.doc_id), p.doc_id


# Property 12: the plan does not depend on the styling version
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31),
       genuine=st.integers(0, 3), tampered=st.integers(0, 3),
       mix=st.lists(st.sampled_from(ALL_TAMPER_TYPES), min_size=1, max_size=5, unique=True),
       lo=st.integers(1, 3), extra=st.integers(0, 3))
def test_property_12_plan_independent_of_style_version(seed, genuine, tampered, mix, lo, extra):
    """Feature: generator-styling-variety, Property 12: the plan does not depend on the Styling_Version.

    **Validates: Requirements 8.1, 8.2**
    """
    kw = dict(master_seed=seed, per_layout_genuine=genuine, per_layout_tampered=tampered,
              months_min=lo, months_max=lo + extra, tamper_mix=tuple((n, 1.0) for n in mix))
    out = "unused-plan-only"
    assert CorpusBuilder(_cfg(out, style_version=1, **kw)).plan() == \
        CorpusBuilder(_cfg(out, style_version=2, **kw)).plan()


def test_counts_equal_v1_dictionary_without_extra_type(tmp_path):
    ms = [{"layout": "generator_v1", "split": "train", "label": {"tamper_types": ["edited_amount"]}}]
    c1 = CorpusBuilder(_cfg(tmp_path / "a", style_version=1))._counts(ms)
    c2 = CorpusBuilder(_cfg(tmp_path / "b", style_version=2))._counts(ms)
    assert c1 == c2


def test_v2_build_with_extra_type_lists_it_per_split(tmp_path):
    mix = (("subtle_amount_edit", 1.0), ("edited_amount", 1.0))
    cfg = _cfg(tmp_path / "o", style_version=2, tamper_mix=mix, per_layout_genuine=1,
               per_layout_tampered=2, months_min=2, months_max=2)
    summary = CorpusBuilder(cfg).build()
    assert not summary.failed
    for split in summary.counts["per_tamper_type"].values():
        assert "subtle_amount_edit" in split
