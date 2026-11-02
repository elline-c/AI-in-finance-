"""Tests for ``fraud_detect.corpus.builder`` (task 19.1; design: builder and output tree).

They fail with ``ImportError`` until task 19.2 creates the module.

Interface assumed (to be created in task 19.2):

* ``CorpusBuilder(cfg: BuildConfig)`` with
  - ``.plan() -> list[DocumentPlan]``: cheap, deterministic, no PDFs written;
  - ``.build(progress=None) -> CorpusSummary``; the summary has ``.failed`` (list of
    entries naming the ``doc_id`` and the exception type).
* Invalid settings (months, fractions, unknown hold-out layout, non-empty output
  folder without ``force``, output inside ``fraud_detect/``, ``tests/`` or ``.kiro/``)
  raise ``ValueError`` (from the constructor, ``plan()`` or ``build()``) and write nothing.
* ``builder`` imports ``simulate_ledger`` by name, so a test can make one document fail
  by patching ``fraud_detect.corpus.builder.simulate_ledger``.
* Output tree under ``out_dir``: ``pdfs/<doc_id>.pdf``, ``manifests/<doc_id>.json``,
  ``labels.csv``, ``rows.csv``, ``companies.json``, ``corpus_manifest.json``, ``README.txt``.
* ``companies.json``: a list (or ``{"companies": [...]}``) of records with ``uen``,
  ``registered_name`` and ``baseline`` = ``{"avg_monthly_credit", "avg_monthly_debit"}``.
* ``corpus_manifest.json``: ``generator_version``, ``master_seed``, ``command_line``,
  ``failed`` (list), and ``counts`` = ``{"documents", "per_layout", "per_split",
  "per_tamper_type"}`` where ``per_tamper_type`` maps split -> tamper type -> count.
* Per-document manifest ``company`` holds ``company_id``, ``uen``, ``registered_name``.
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.labels import text_layer_sha256
from fraud_detect.corpus.schema import (
    GENERATOR_VERSION,
    LABELS_COLUMNS,
    LAYOUTS,
    ROWS_COLUMNS,
    SPLITS,
    TAMPER_TYPES,
    BuildConfig,
    DocumentPlan,
)
from fraud_detect.corpus.splits import assign_split

from tests.corpus.helpers import metadata_of, words_of

REPO = Path(__file__).resolve().parents[2]
SEED = 20260601
# Cheap-to-render tamper types spanning the four methods (file, rerender, content, behaviour).
MIX = (("edited_amount", 1.0), ("metadata_inconsistency", 1.0),
       ("structuring", 1.0), ("fabricated_receipts", 1.0))
MIX_TYPES = {t for t, _ in MIX}


def _cfg(out: Path, **kw) -> BuildConfig:
    base = dict(out_dir=out, master_seed=SEED, per_layout_genuine=1, per_layout_tampered=1,
                months_min=6, months_max=6, tamper_mix=MIX,
                command_line="python -m fraud_detect.corpus.build --test-marker")
    base.update(kw)
    return BuildConfig(**base)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _companies(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["companies"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------
# shared tiny builds (module scoped)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus_a")
    summary = CorpusBuilder(_cfg(out)).build()
    return out, summary


@pytest.fixture(scope="module")
def built_again(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus_b")
    CorpusBuilder(_cfg(out)).build()
    return out


@pytest.fixture(scope="module")
def built_holdout(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus_h")
    CorpusBuilder(_cfg(out, hold_out_layout="two_col_ddmmyy_cr")).build()
    return out


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def test_plan_has_two_genuine_and_two_tampered_per_layout(tmp_path):
    plan = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=2, per_layout_tampered=2)).plan()
    assert all(isinstance(p, DocumentPlan) for p in plan)
    assert len(plan) == 24
    assert len({p.doc_id for p in plan}) == 24
    for layout in LAYOUTS:
        mine = [p for p in plan if p.layout == layout]
        assert sum(p.tamper_type is None for p in mine) == 2
        assert sum(p.tamper_type is not None for p in mine) == 2


def test_plan_is_deterministic_and_cheap(tmp_path):
    cfg = _cfg(tmp_path / "o", per_layout_genuine=2, per_layout_tampered=2)
    assert CorpusBuilder(cfg).plan() == CorpusBuilder(cfg).plan()
    assert not (tmp_path / "o").exists()          # planning writes nothing


def test_plan_of_a_document_does_not_depend_on_other_documents(tmp_path):
    small = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=2, per_layout_tampered=0)).plan()
    large = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=4, per_layout_tampered=3)).plan()
    by_id = {p.doc_id: p for p in large}
    assert small
    for p in small:
        assert by_id[p.doc_id] == p


def test_plan_tamper_types_follow_the_mix(tmp_path):
    mix = (("structuring", 3.0), ("edited_amount", 1.0))
    plan = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=0, per_layout_tampered=8,
                              tamper_mix=mix)).plan()
    counts = Counter(p.tamper_type for p in plan)
    assert counts == {"structuring": 36, "edited_amount": 12}


def test_plan_spreads_tamper_types_evenly_without_a_mix(tmp_path):
    plan = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=0, per_layout_tampered=16,
                              tamper_mix=None)).plan()
    counts = Counter(p.tamper_type for p in plan)
    assert set(counts) == set(TAMPER_TYPES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_plan_draws_months_from_6_to_12_when_not_given(tmp_path):
    plan = CorpusBuilder(_cfg(tmp_path / "o", per_layout_genuine=4, per_layout_tampered=2,
                              months_min=None, months_max=None)).plan()
    months = {p.n_months for p in plan}
    assert months <= set(range(6, 13))
    assert len(months) > 1


def test_plan_honours_given_month_range(tmp_path):
    plan = CorpusBuilder(_cfg(tmp_path / "o", months_min=8, months_max=8)).plan()
    assert {p.n_months for p in plan} == {8}


@pytest.mark.parametrize("lo,hi", [(0, 6), (6, 25), (9, 7)])
def test_invalid_month_settings_are_refused(tmp_path, lo, hi):
    with pytest.raises(ValueError):
        CorpusBuilder(_cfg(tmp_path / "o", months_min=lo, months_max=hi)).build()
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("fractions", [(0.5, 0.5, 0.5), (1.2, -0.1, -0.1)])
def test_invalid_split_fractions_are_refused(tmp_path, fractions):
    with pytest.raises(ValueError):
        CorpusBuilder(_cfg(tmp_path / "o", split_fractions=fractions)).build()
    assert not (tmp_path / "o").exists()


def test_unknown_hold_out_layout_is_refused(tmp_path):
    with pytest.raises(ValueError):
        CorpusBuilder(_cfg(tmp_path / "o", hold_out_layout="no_such_layout")).build()


# --------------------------------------------------------------------------
# output folder safety
# --------------------------------------------------------------------------
def test_non_empty_output_folder_is_refused_without_force(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("precious", encoding="utf-8")
    with pytest.raises(ValueError, match="force"):
        CorpusBuilder(_cfg(out)).build()
    assert (out / "keep.txt").read_text(encoding="utf-8") == "precious"
    assert not (out / "pdfs").exists()


def test_non_empty_output_folder_is_accepted_with_force(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "old.txt").write_text("old", encoding="utf-8")
    CorpusBuilder(_cfg(out, force=True, per_layout_tampered=0)).build()
    assert (out / "labels.csv").is_file()


@pytest.mark.parametrize("inside", ["fraud_detect", "tests", ".kiro"])
def test_output_inside_protected_folders_is_refused(inside):
    target = REPO / inside / "_corpus_should_not_exist"
    with pytest.raises(ValueError):
        CorpusBuilder(_cfg(target, force=True)).build()
    assert not target.exists()


def test_default_corpus_location_is_git_ignored():
    for path in ("data/corpus/pdfs/x.pdf", "data/corpus/labels.csv"):
        res = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
        assert res.returncode == 0, f"{path} is not git-ignored"


# --------------------------------------------------------------------------
# output tree
# --------------------------------------------------------------------------
def test_output_tree_has_all_files(built):
    out, _ = built
    for name in ("labels.csv", "rows.csv", "companies.json", "corpus_manifest.json", "README.txt"):
        assert (out / name).is_file(), name
    labels = _read_csv(out / "labels.csv")
    assert len(labels) == 12
    for row in labels:
        assert (out / "pdfs" / f"{row['doc_id']}.pdf").is_file()
        assert (out / "manifests" / f"{row['doc_id']}.json").is_file()
        assert row["file"] == f"pdfs/{row['doc_id']}.pdf"
    assert len(list((out / "pdfs").glob("*.pdf"))) == 12
    assert len(list((out / "manifests").glob("*.json"))) == 12


def test_csv_headers_are_exact(built):
    out, _ = built
    assert (out / "labels.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(LABELS_COLUMNS)
    assert (out / "rows.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(ROWS_COLUMNS)


def test_per_layout_counts_and_tamper_mix(built):
    out, summary = built
    labels = _read_csv(out / "labels.csv")
    for layout in LAYOUTS:
        mine = [r for r in labels if r["layout"] == layout]
        assert len(mine) == 2
        assert sum(r["tamper_types"] == "" for r in mine) == 1
    for r in labels:
        if r["tamper_types"]:
            assert set(r["tamper_types"].split(";")) <= MIX_TYPES
        assert r["n_months"] == "6"
        assert r["generator_version"] == GENERATOR_VERSION
    assert not summary.failed


def test_labels_rows_and_manifests_agree(built):
    out, _ = built
    labels = {r["doc_id"]: r for r in _read_csv(out / "labels.csv")}
    rows = _read_csv(out / "rows.csv")
    for doc_id, lab in labels.items():
        manifest = json.loads((out / "manifests" / f"{doc_id}.json").read_text(encoding="utf-8"))
        mine = [r for r in rows if r["doc_id"] == doc_id]
        assert len(mine) == int(lab["n_rows"]) == len(manifest["rows"])
        assert sum(r["is_fraud_row"] == "1" for r in mine) == int(lab["n_affected_rows"])
        assert manifest["synthetic"] is True
        assert manifest["generator_version"] == GENERATOR_VERSION
        assert manifest["split"] == lab["split"]
        pdf = (out / lab["file"]).read_bytes()
        assert manifest["text_layer_sha256"] == text_layer_sha256(pdf)


def test_genuine_documents_have_no_fraud_rows(built):
    out, _ = built
    for lab in _read_csv(out / "labels.csv"):
        if lab["tamper_types"] == "":
            assert lab["is_tampered"] == "0"
            assert lab["n_affected_rows"] == "0"


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def test_split_assignments_follow_assign_split(built):
    out, _ = built
    for lab in _read_csv(out / "labels.csv"):
        assert lab["split"] in SPLITS
        assert lab["split"] == assign_split(SEED, lab["company_id"], lab["layout"], None,
                                            (0.70, 0.15, 0.15))


def test_one_company_one_split(built):
    out, _ = built
    seen: dict[str, str] = {}
    for lab in _read_csv(out / "labels.csv"):
        assert seen.setdefault(lab["company_id"], lab["split"]) == lab["split"]


def test_held_out_layout_goes_to_test_unseen_layout(built_holdout):
    for lab in _read_csv(built_holdout / "labels.csv"):
        if lab["layout"] == "two_col_ddmmyy_cr":
            assert lab["split"] == "test_unseen_layout"
        else:
            assert lab["split"] in ("train", "validation", "test")


# --------------------------------------------------------------------------
# companies file
# --------------------------------------------------------------------------
def test_companies_file_matches_genuine_ledgers(built):
    out, _ = built
    companies = {c["uen"]: c for c in _companies(out / "companies.json")}
    labels = _read_csv(out / "labels.csv")
    assert len(companies) == len({r["company_id"] for r in labels})
    checked = 0
    for lab in labels:
        manifest = json.loads((out / "manifests" / f"{lab['doc_id']}.json").read_text(encoding="utf-8"))
        rec = companies[manifest["company"]["uen"]]
        assert rec["registered_name"] == manifest["company"]["registered_name"]
        if lab["tamper_types"] == "":
            n = len(manifest["months"])
            amounts = [r["amount"] for r in manifest["rows"]]
            credit = sum(a for a in amounts if a > 0) / n
            debit = -sum(a for a in amounts if a < 0) / n
            base = rec["baseline"]
            assert base["avg_monthly_credit"] == pytest.approx(credit, abs=0.01)
            assert base["avg_monthly_debit"] == pytest.approx(debit, abs=0.01)
            checked += 1
    assert checked == 6


# --------------------------------------------------------------------------
# README and corpus manifest
# --------------------------------------------------------------------------
def test_readme_and_corpus_manifest_state_what_they_must(built):
    out, _ = built
    readme = (out / "README.txt").read_text(encoding="utf-8")
    raw = (out / "corpus_manifest.json").read_text(encoding="utf-8")
    cm = json.loads(raw)
    for text in (readme, raw):
        assert "SYNTHETIC DATA" in text
        assert GENERATOR_VERSION in text
        assert str(SEED) in text
        assert "--test-marker" in text
    assert cm["generator_version"] == GENERATOR_VERSION
    assert cm["master_seed"] == SEED
    assert cm["failed"] == []


def test_corpus_manifest_counts_add_up(built):
    out, _ = built
    labels = _read_csv(out / "labels.csv")
    counts = json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))["counts"]
    assert counts["documents"] == len(labels) == 12
    assert counts["per_layout"] == {layout: 2 for layout in LAYOUTS}
    assert counts["per_split"] == dict(Counter(r["split"] for r in labels))
    assert sum(counts["per_split"].values()) == 12
    by_split: dict[str, Counter] = {}
    for r in labels:
        by_split.setdefault(r["split"], Counter())[r["tamper_types"] or "genuine"] += 1
    for split, per_type in counts["per_tamper_type"].items():
        expected = {t: n for t, n in by_split[split].items() if t != "genuine"}
        assert {t: n for t, n in per_type.items() if n} == expected


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def test_same_seed_gives_equal_text_outputs(built, built_again):
    a, _ = built
    b = built_again
    for name in ("labels.csv", "rows.csv", "companies.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    names = sorted(p.name for p in (a / "manifests").glob("*.json"))
    assert names == sorted(p.name for p in (b / "manifests").glob("*.json"))
    for n in names:
        assert (a / "manifests" / n).read_bytes() == (b / "manifests" / n).read_bytes(), n


def test_same_seed_gives_equal_pdf_text_and_metadata(built, built_again):
    a, _ = built
    b = built_again
    for pdf in sorted((a / "pdfs").glob("*.pdf")):
        pa, pb = pdf.read_bytes(), (b / "pdfs" / pdf.name).read_bytes()
        assert words_of(pa) == words_of(pb), pdf.name
        assert metadata_of(pa) == metadata_of(pb), pdf.name


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------
def test_failing_document_is_recorded_and_others_still_written(tmp_path, monkeypatch):
    import fraud_detect.corpus.builder as builder_mod

    cfg = _cfg(tmp_path / "o", per_layout_tampered=0)
    victim = CorpusBuilder(cfg).plan()[0]
    real = builder_mod.simulate_ledger

    def flaky(company, *args, **kwargs):
        if company.company_id == victim.company_id:
            raise RuntimeError("boom")
        return real(company, *args, **kwargs)

    monkeypatch.setattr(builder_mod, "simulate_ledger", flaky)
    summary = CorpusBuilder(cfg).build()

    assert [f["doc_id"] for f in summary.failed] == [victim.doc_id]
    cm = json.loads((tmp_path / "o" / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert [f["doc_id"] for f in cm["failed"]] == [victim.doc_id]
    assert "RuntimeError" in json.dumps(cm["failed"])
    labels = _read_csv(tmp_path / "o" / "labels.csv")
    assert victim.doc_id not in {r["doc_id"] for r in labels}
    assert len(labels) == 5
    assert cm["counts"]["documents"] == 5
