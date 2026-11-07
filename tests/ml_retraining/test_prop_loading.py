"""Property 1: Loading preserves the Split column and counts add up.

Feature: ml-retraining, Property 1
**Validates: Requirements 1.1, 1.9, 1.10**

Small label tables are written to temporary folders (never real corpus files).
"""
from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.corpora import CorpusError, VALID_SPLITS, load_corpus

SPLITS = sorted(VALID_SPLITS)
LAYOUTS = ["lay_a", "lay_b", "lay_c", "lay_d"]
BAD_SPLITS = ["", "holdout", "Train", "val", "test_unseen", " train", "TEST"]
FIELDS = ["doc_id", "split", "layout", "company_id", "is_tampered",
          "genuine_resaved", "tamper_types", "tamper_method", "n_months"]


@st.composite
def label_tables(draw, split_strategy=st.sampled_from(SPLITS)):
    n = draw(st.integers(min_value=1, max_value=25))
    rows = []
    for i in range(n):
        tampered = draw(st.booleans())
        rows.append({
            "doc_id": f"DOC-{i:04d}",
            "split": draw(split_strategy),
            "layout": draw(st.sampled_from(LAYOUTS)),
            "company_id": f"C{draw(st.integers(0, 6)):03d}",
            "is_tampered": "1" if tampered else "0",
            "genuine_resaved": "0",
            "tamper_types": "edited_amount" if tampered else "",
            "tamper_method": "pdf_edit" if tampered else "",
            "n_months": str(draw(st.integers(1, 12))),
        })
    return rows


def _write_corpus(root: Path, rows: list[dict], seed: int, drop_split: bool = False) -> Path:
    path = root / "corpus_under_test"
    path.mkdir()
    fields = [f for f in FIELDS if not (drop_split and f == "split")]
    with open(path / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    companies = sorted({r["company_id"] for r in rows})
    (path / "companies.json").write_text(json.dumps(
        [{"company_id": c, "uen": f"UEN-{c}"} for c in companies]))
    (path / "corpus_manifest.json").write_text(json.dumps(
        {"master_seed": seed, "hold_out_layout": LAYOUTS[0],
         "counts": {"documents": len(rows)}}))
    return path


@settings(max_examples=100, deadline=None)
@given(rows=label_tables(), seed=st.integers(0, 10**6))
def test_loaded_split_equals_file_and_counts_add_up(rows, seed):
    with tempfile.TemporaryDirectory() as tmp:
        corpus = load_corpus(_write_corpus(Path(tmp), rows, seed))
    assert len(corpus.docs) == len(rows)
    by_id = {d.doc_id: d for d in corpus.docs}
    for r in rows:
        assert by_id[r["doc_id"]].split == r["split"]
    splits, layouts = corpus.split_counts(), corpus.layout_counts()
    assert sum(splits.values()) == len(rows)
    assert sum(layouts.values()) == len(rows)
    assert splits == dict(Counter(r["split"] for r in rows))
    assert layouts == dict(Counter(r["layout"] for r in rows))


@settings(max_examples=100, deadline=None)
@given(data=st.data(), seed=st.integers(0, 10**6))
def test_bad_split_values_raise_naming_corpus_and_exact_offenders(data, seed):
    rows = data.draw(label_tables())
    idx = data.draw(st.sets(st.integers(0, len(rows) - 1), min_size=1, max_size=len(rows)))
    for i in idx:
        rows[i]["split"] = data.draw(st.sampled_from(BAD_SPLITS))
    offenders = {rows[i]["doc_id"] for i in idx}
    good = {r["doc_id"] for r in rows} - offenders
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(Path(tmp), rows, seed)
        with pytest.raises(CorpusError) as err:
            load_corpus(path)
    text = str(err.value)
    assert "corpus_under_test" in text
    assert all(d in text for d in offenders)
    assert not any(d in text for d in good)


@settings(max_examples=50, deadline=None)
@given(rows=label_tables(), seed=st.integers(0, 10**6))
def test_missing_split_column_raises_naming_corpus_and_all_docs(rows, seed):
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(Path(tmp), rows, seed, drop_split=True)
        with pytest.raises(CorpusError) as err:
            load_corpus(path)
    text = str(err.value)
    assert "corpus_under_test" in text
    assert all(r["doc_id"] in text for r in rows)
