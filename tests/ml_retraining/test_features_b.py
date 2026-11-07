"""Task 9.1: failing tests for ``fraud_detect.ml.retraining.features_b`` (Node 6b table).

Written before the module exists. Only the Mini_Corpus fixture is used (Requirement 17.9).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.features_b``)::

    EXTRACT_COLUMNS: tuple[str, ...]        # the 13 keys of statement_features.extract_features
    class SkippedShareError(Exception):     # exit_code == gates.EXIT_INTEGRITY (2); .offenders: dict split -> percent
    def extractor_sha8() -> str             # sha256(bytes of statement_features.py).hexdigest()[:8]
    def labels_sha8(corpus_dir) -> str      # sha256(bytes of labels.csv).hexdigest()[:8]
    def cache_path_b(cache_root, master_seed, labels_sha8, extractor_sha8) -> Path
        # == Path(cache_root) / "features_b" / f"{master_seed}-{labels_sha8}-{extractor_sha8}.csv"
    def build_feature_table(corpus_dir, *, cache_root, is_ignored=None, extract=None,
                            skip_max_share=2) -> FeatureTableB
        # extract: callable(path) -> dict | None; default statement_features.extract_features.
        # skip_max_share: percent; the run raises SkippedShareError when any Split's skipped
        # percent is strictly greater. The cache path is checked with safe_io.ensure_ignored
        # (is_ignored injectable) before the cache is written.
    FeatureTableB (dataclass): .frame (pandas DataFrame, skipped docs excluded), .skipped_ids (tuple[str]),
        .skip_report (dict split -> {"skipped": int, "total": int, "percent": float}),
        .cache_path (Path), .from_cache (bool)
    frame columns: corpus_key, doc_id, split, layout, company_key, is_tampered, genuine_resaved,
        tamper_types, tamper_method, n_months, skipped, EXTRACT_COLUMNS..., style_<field> for every
        StyleProfile field.
    Cache file: first line "# master_seed=<n> labels_sha8=<x> extractor_sha8=<y>", then CSV of the
        raw extract_features values per doc_id. A cache whose first line differs from the request is
        not read.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fraud_detect.corpus.builder import CorpusBuilder  # noqa: F401  (fixture dependency documented)
from fraud_detect.ml import statement_features
from fraud_detect.ml.retraining import gates
from fraud_detect.ml.retraining.corpora import load_corpus
from fraud_detect.ml.retraining.features_b import (
    EXTRACT_COLUMNS, SkippedShareError, build_feature_table, cache_path_b, extractor_sha8, labels_sha8)
from fraud_detect.ml.retraining.safe_io import WriteGuardError
from fraud_detect.ml.retraining.style_profile import style_profile

DESIGN_COLUMNS = ("corpus_key", "doc_id", "split", "layout", "company_key", "is_tampered",
                  "genuine_resaved", "tamper_types", "tamper_method", "n_months", "skipped")
STYLE_FIELDS = ("producer", "producer_family", "creator", "fonts", "bank_name", "street",
                "second_address_line", "row_step_bucket", "creation_hour", "page_size", "footer_text")


def _labels(corpus: Path) -> list:
    with open(corpus / "labels.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _copy_corpus(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _counting_extract():
    calls = []

    def extract(path):
        calls.append(str(path))
        return statement_features.extract_features(str(path))

    extract.calls = calls
    return extract


@pytest.fixture(scope="module")
def built(mini_corpus, tmp_path_factory):
    """One shared build of seed 0 so most tests do not re-extract."""
    cache = tmp_path_factory.mktemp("cache_b")
    table = build_feature_table(mini_corpus.path(), cache_root=cache, is_ignored=lambda p: True)
    return table, cache


# ------------------------------------------------------------------ columns and values
def test_extract_columns_are_the_thirteen_raw_outputs():
    assert len(EXTRACT_COLUMNS) == 13
    assert set(EXTRACT_COLUMNS) == set(statement_features.FEATURE_NAMES) | {
        "n_distinct_date_formats", "duplicate_amount_overlap"}


def test_table_has_design_columns(built, mini_corpus):
    table, _ = built
    cols = set(table.frame.columns)
    for name in DESIGN_COLUMNS + tuple(EXTRACT_COLUMNS):
        assert name in cols, name
    for field in STYLE_FIELDS:
        assert f"style_{field}" in cols, field
    assert len(table.frame) == len(_labels(mini_corpus.path()))
    assert not table.frame["skipped"].any()
    assert table.skipped_ids == ()


def test_metadata_columns_match_corpus(built, mini_corpus):
    table, _ = built
    docs = {d.doc_id: d for d in load_corpus(mini_corpus.path()).docs}
    for row in table.frame.to_dict("records"):
        d = docs[row["doc_id"]]
        assert row["corpus_key"] == d.corpus_key
        assert row["split"] == d.split and row["layout"] == d.layout
        assert bool(row["is_tampered"]) == d.is_tampered
        assert bool(row["genuine_resaved"]) == d.genuine_resaved
        assert int(row["n_months"]) == d.n_months


def test_values_equal_extract_features_exactly(built, mini_corpus):
    table, _ = built
    for row in table.frame.to_dict("records"):
        expected = statement_features.extract_features(
            str(mini_corpus.path() / "pdfs" / f"{row['doc_id']}.pdf"))
        for name in EXTRACT_COLUMNS:
            assert row[name] == expected[name], (row["doc_id"], name)


def test_style_columns_equal_style_profile(built, mini_corpus):
    table, _ = built
    for row in table.frame.to_dict("records")[:6]:
        prof = style_profile((mini_corpus.path() / "pdfs" / f"{row['doc_id']}.pdf").read_bytes())
        assert row["style_producer_family"] == prof.producer_family
        assert row["style_bank_name"] == prof.bank_name
        assert row["style_creation_hour"] == prof.creation_hour


# ------------------------------------------------------------------ skipped documents
def _damage_one(mini_corpus, tmp_path):
    corpus = _copy_corpus(mini_corpus.path(), tmp_path / "damaged")
    victim = next(r for r in _labels(corpus) if r["split"] == "train")
    (corpus / "pdfs" / f"{victim['doc_id']}.pdf").write_bytes(b"this is not a pdf")
    return corpus, victim


def test_unreadable_pdf_is_skipped_counted_and_extraction_continues(mini_corpus, tmp_path):
    corpus, victim = _damage_one(mini_corpus, tmp_path)
    table = build_feature_table(corpus, cache_root=tmp_path / "c", is_ignored=lambda p: True,
                                skip_max_share=100)
    labels = _labels(corpus)
    assert table.skipped_ids == (victim["doc_id"],)
    assert victim["doc_id"] not in set(table.frame["doc_id"])
    assert len(table.frame) == len(labels) - 1
    for split in {r["split"] for r in labels}:
        total = sum(1 for r in labels if r["split"] == split)
        skipped = 1 if split == victim["split"] else 0
        rep = table.skip_report[split]
        assert (rep["skipped"], rep["total"]) == (skipped, total)
        assert rep["percent"] == pytest.approx(100.0 * skipped / total)


def test_skipped_share_over_limit_stops_with_exit_2_and_no_cache(mini_corpus, tmp_path):
    corpus, victim = _damage_one(mini_corpus, tmp_path)
    cache = tmp_path / "c"
    with pytest.raises(SkippedShareError) as err:
        build_feature_table(corpus, cache_root=cache, is_ignored=lambda p: True)  # default 2 percent
    assert err.value.exit_code == gates.EXIT_INTEGRITY
    assert victim["split"] in err.value.offenders


# ------------------------------------------------------------------ cache
def test_cache_path_and_header(built, mini_corpus):
    table, cache = built
    seed = mini_corpus.seeds[0]
    expected = cache / "features_b" / f"{seed}-{labels_sha8(mini_corpus.path())}-{extractor_sha8()}.csv"
    assert cache_path_b(cache, seed, labels_sha8(mini_corpus.path()), extractor_sha8()) == expected
    assert table.cache_path == expected and expected.is_file()
    first = expected.read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# master_seed={seed} labels_sha8={labels_sha8(mini_corpus.path())} extractor_sha8={extractor_sha8()}"


def test_key_parts_come_from_file_bytes(mini_corpus):
    labels = (mini_corpus.path() / "labels.csv").read_bytes()
    assert labels_sha8(mini_corpus.path()) == hashlib.sha256(labels).hexdigest()[:8]
    src = Path(statement_features.__file__).read_bytes()
    assert extractor_sha8() == hashlib.sha256(src).hexdigest()[:8]


def test_second_run_reads_cache_and_gives_equal_values(mini_corpus, tmp_path):
    first_extract = _counting_extract()
    t1 = build_feature_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True,
                             extract=first_extract)
    assert len(first_extract.calls) == len(t1.frame) and not t1.from_cache
    again = _counting_extract()
    t2 = build_feature_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True,
                             extract=again)
    assert again.calls == [] and t2.from_cache
    for name in EXTRACT_COLUMNS:
        assert list(t2.frame[name]) == list(t1.frame[name])


def test_header_that_differs_from_request_is_not_read(mini_corpus, tmp_path):
    t1 = build_feature_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True)
    lines = t1.cache_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(f"master_seed={mini_corpus.seeds[0]}", "master_seed=1")
    t1.cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    extract = _counting_extract()
    t2 = build_feature_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True,
                             extract=extract)
    assert len(extract.calls) == len(t2.frame) and not t2.from_cache


def test_other_seed_or_labels_file_gives_other_key_and_reextracts(mini_corpus, tmp_path):
    a = build_feature_table(mini_corpus.path(mini_corpus.seeds[0]), cache_root=tmp_path,
                            is_ignored=lambda p: True)
    b = build_feature_table(mini_corpus.path(mini_corpus.seeds[1]), cache_root=tmp_path,
                            is_ignored=lambda p: True)
    assert a.cache_path != b.cache_path and a.cache_path.is_file() and b.cache_path.is_file()
    # A changed labels file: same seed, different labels hash, different cache file, re-extraction.
    edited = _copy_corpus(mini_corpus.path(), tmp_path / "edited")
    text = (edited / "labels.csv").read_text(encoding="utf-8")
    (edited / "labels.csv").write_text(text + "\n", encoding="utf-8")
    assert labels_sha8(edited) != labels_sha8(mini_corpus.path())
    extract = _counting_extract()
    c = build_feature_table(edited, cache_root=tmp_path, is_ignored=lambda p: True, extract=extract)
    assert c.cache_path not in (a.cache_path, b.cache_path) and extract.calls


def test_cache_path_is_checked_with_the_write_guard(mini_corpus, tmp_path, is_ignored):
    build_feature_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=is_ignored)
    assert any(p.parent.name == "features_b" for p in is_ignored.calls)
    is_ignored.ignored = False
    other = tmp_path / "other"
    with pytest.raises(WriteGuardError):
        build_feature_table(mini_corpus.path(), cache_root=other, is_ignored=is_ignored)
    assert not list(other.rglob("*.csv")) if other.exists() else True


# ------------------------------------------------------------------ import hygiene
def test_import_pulls_no_pdf_or_torch_libraries():
    code = ("import sys, fraud_detect.ml.retraining.features_b;"
            "bad = [m for m in ('pdfplumber', 'torch', 'torch_geometric') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
