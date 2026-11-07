"""Tests for ``fraud_detect.ml.retraining.corpora`` (task 4.1; Requirements 1.1, 1.9, 1.10).

Only the in-memory / temporary Mini_Corpus is used. These tests are written before the
module exists and fail until task 4.2 is done.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from fraud_detect.ml.retraining.corpora import CorpusError, DocRef, LoadedCorpus, load_corpus

SPLIT_NAMES = {"train", "validation", "test", "test_unseen_layout"}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _copy_labels_only(src: Path, dst: Path) -> Path:
    """Copy the small metadata files (never PDFs) so a test can damage labels.csv."""
    dst.mkdir(parents=True)
    for name in ("labels.csv", "companies.json", "corpus_manifest.json"):
        shutil.copy(src / name, dst / name)
    return dst


def _rewrite_labels(path: Path, edit) -> list[str]:
    rows = _read_csv(path / "labels.csv")
    fields = list(rows[0].keys())
    touched = edit(rows)
    with open(path / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return touched


# ------------------------------------------------------------------ loading
def test_one_docref_per_label_row_with_file_split(mini_corpus):
    seed = mini_corpus.seeds[0]
    corpus = load_corpus(mini_corpus.path(seed))
    rows = _read_csv(mini_corpus.path(seed) / "labels.csv")
    assert isinstance(corpus, LoadedCorpus)
    assert len(corpus.docs) == len(rows) > 0
    by_id = {d.doc_id: d for d in corpus.docs}
    assert len(by_id) == len(rows)
    for row in rows:
        doc = by_id[row["doc_id"]]
        assert isinstance(doc, DocRef)
        assert doc.split == row["split"]
        assert doc.layout == row["layout"]


def test_docref_fields(mini_corpus):
    seed = mini_corpus.seeds[0]
    path = mini_corpus.path(seed)
    corpus = load_corpus(path)
    rows = {r["doc_id"]: r for r in _read_csv(path / "labels.csv")}
    uens = {c["company_id"]: c["uen"] for c in json.loads((path / "companies.json").read_text())}
    assert corpus.master_seed == seed
    assert corpus.hold_out_layout == mini_corpus.hold_out_layout
    for doc in corpus.docs:
        row = rows[doc.doc_id]
        assert doc.master_seed == seed
        assert doc.company_key == (seed, row["company_id"])
        assert doc.uen == uens[row["company_id"]]
        assert doc.is_tampered == (row["is_tampered"] == "1")
        assert doc.genuine_resaved == (row["genuine_resaved"] == "1")
        assert doc.tamper_types == tuple(t for t in row["tamper_types"].split(";") if t)
        assert doc.tamper_method == row["tamper_method"]
        assert doc.n_months == int(row["n_months"])


def test_resaved_documents_are_genuine_with_resaved_type(mini_corpus):
    corpus = load_corpus(mini_corpus.path())
    resaved = [d for d in corpus.docs if d.genuine_resaved]
    assert resaved, "mini corpus should contain genuine_resaved documents"
    assert all(not d.is_tampered and "genuine_resaved" in d.tamper_types for d in resaved)


def test_counts_per_split_and_layout_sum_to_document_count(mini_corpus):
    corpus = load_corpus(mini_corpus.path())
    n = len(corpus.docs)
    split_counts = corpus.split_counts()
    layout_counts = corpus.layout_counts()
    assert sum(split_counts.values()) == n
    assert sum(layout_counts.values()) == n
    assert split_counts == dict(Counter(d.split for d in corpus.docs))
    assert layout_counts == dict(Counter(d.layout for d in corpus.docs))
    assert set(split_counts) <= SPLIT_NAMES


def test_manifest_counts_and_uen_check_are_read(mini_corpus):
    path = mini_corpus.path()
    corpus = load_corpus(path)
    manifest = json.loads((path / "corpus_manifest.json").read_text())
    assert corpus.manifest_counts == manifest["counts"]
    assert corpus.manifest_counts["documents"] == len(corpus.docs)
    assert corpus.uen_check == manifest["uen_check"]


def test_same_raw_company_id_in_two_corpora_is_two_keys(mini_corpus):
    a, b = (load_corpus(mini_corpus.path(s)) for s in mini_corpus.seeds)
    raw_a = {d.company_key[1] for d in a.docs}
    raw_b = {d.company_key[1] for d in b.docs}
    shared = raw_a & raw_b
    assert shared, "company ids repeat across corpora (design finding F2)"
    keys_a = {d.company_key for d in a.docs}
    keys_b = {d.company_key for d in b.docs}
    assert keys_a.isdisjoint(keys_b)
    assert a.master_seed != b.master_seed


def test_loading_is_read_only(mini_corpus):
    path = mini_corpus.path()
    before = {p.name: p.stat().st_mtime_ns for p in path.iterdir() if p.is_file()}
    load_corpus(path)
    after = {p.name: p.stat().st_mtime_ns for p in path.iterdir() if p.is_file()}
    assert before == after


# ------------------------------------------------------------ bad split data
def test_missing_split_column_raises_naming_corpus_and_all_doc_ids(mini_corpus, tmp_path):
    src = mini_corpus.path()
    dst = _copy_labels_only(src, tmp_path / "no_split_corpus")
    rows = _read_csv(dst / "labels.csv")
    fields = [f for f in rows[0].keys() if f != "split"]
    with open(dst / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(CorpusError) as err:
        load_corpus(dst)
    text = str(err.value)
    assert "no_split_corpus" in text
    assert "split" in text.lower()
    for row in rows:
        assert row["doc_id"] in text


@pytest.mark.parametrize("bad_value", ["", "holdout", "Train", "val"])
def test_empty_or_unknown_split_value_names_exactly_the_offenders(mini_corpus, tmp_path, bad_value):
    dst = _copy_labels_only(mini_corpus.path(), tmp_path / "bad_split_corpus")

    def edit(rows):
        picked = [rows[0]["doc_id"], rows[3]["doc_id"]]
        for r in rows:
            if r["doc_id"] in picked:
                r["split"] = bad_value
        return picked

    offenders = _rewrite_labels(dst, edit)
    good = [r["doc_id"] for r in _read_csv(dst / "labels.csv") if r["doc_id"] not in offenders]
    with pytest.raises(CorpusError) as err:
        load_corpus(dst)
    text = str(err.value)
    assert "bad_split_corpus" in text
    for doc_id in offenders:
        assert doc_id in text
    for doc_id in good:
        assert doc_id not in text


def test_missing_labels_file_raises_corpus_error(tmp_path):
    (tmp_path / "empty_corpus").mkdir()
    with pytest.raises(CorpusError) as err:
        load_corpus(tmp_path / "empty_corpus")
    assert "empty_corpus" in str(err.value)
