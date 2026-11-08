"""Task 9.4: property tests for feature caches and skipped documents.

Property 26: Cache keys.
Property 27: Skipped-document accounting.

**Validates: Requirements 10.3, 10.4, 10.9, 10.10**

Tiny synthetic corpora (labels.csv, rows.csv, companies.json, corpus_manifest.json, no PDFs) and
injected fake ``extract`` / ``build`` callables keep every example fast. For Node 6b the style
profile cannot be read from a missing PDF; ``build_feature_table`` tolerates that.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining import features_a, features_b
from fraud_detect.ml.retraining.safe_io import WriteGuardError

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

SPLITS = ("train", "validation", "test")
HEX8 = st.text(alphabet="0123456789abcdef", min_size=8, max_size=8)
# Small pools so that equal and unequal parts both occur often.
SEEDS = st.sampled_from([1, 2, 20260801])
SHAS = st.sampled_from(["aaaaaaaa", "bbbbbbbb", "00000000"])
REQUEST = st.tuples(SEEDS, SHAS, SHAS)


# ------------------------------------------------------------------ synthetic corpus
def _write_corpus(root: Path, seed: int, splits: list, n_months: int = 6, rows_per_doc: int = 2) -> Path:
    """One document per entry of ``splits``; each document is its own company."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "split", "layout", "company_id", "is_tampered", "genuine_resaved",
                    "tamper_types", "tamper_method", "n_months"])
        for i, s in enumerate(splits):
            w.writerow([f"d{i:03d}", s, "L1", f"c{i:03d}", "0", "0", "", "", n_months])
    with open(root / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "row_index", "date", "amount", "balance", "is_fraud_row"])
        for i in range(len(splits)):
            for r in range(rows_per_doc):
                w.writerow([f"d{i:03d}", r, f"2026-01-{r + 1:02d}", 10.0 + r, 100.0 + r, "0"])
    (root / "companies.json").write_text(
        json.dumps([{"company_id": f"c{i:03d}", "uen": f"U{i}"} for i in range(len(splits))]),
        encoding="utf-8")
    (root / "corpus_manifest.json").write_text(json.dumps({"master_seed": seed}), encoding="utf-8")
    return root


def _good_features() -> dict:
    return {c: 1.0 for c in features_b.EXTRACT_COLUMNS}


class FakeExtract:
    """Fails (raises or returns None) for doc ids in ``failing``; counts every call."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, path):
        doc_id = Path(path).stem
        self.calls.append(doc_id)
        if doc_id in self.failing:
            if int(doc_id[1:]) % 2:
                raise ValueError("unreadable")
            return None
        return _good_features()


def _fake_build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in tf.PDF_FEATURE_COLUMNS:
        out[col] = 0.0
    return out


def _always_ignored(_path) -> bool:
    return True


# ------------------------------------------------------------------ Property 26
@SETTINGS
@given(REQUEST, REQUEST)
def test_cache_keys_equal_iff_all_three_parts_equal(req1, req2):
    """Property 26: keys are equal iff seed, file hash and extractor hash are all equal (6b and 6a)."""
    root = Path("cache")
    b1 = features_b.cache_path_b(root, *req1)
    b2 = features_b.cache_path_b(root, *req2)
    a1 = features_a.cache_path_a(root, *req1)
    a2 = features_a.cache_path_a(root, *req2)
    assert (b1 == b2) == (req1 == req2)
    assert (a1 == a2) == (req1 == req2)
    # Paths live in the node's own folder under the cache root, never elsewhere.
    assert b1.parent == root / "features_b" and a1.parent == root / "features_a"
    assert b1 != a1


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=6), SEEDS, SEEDS,
       st.integers(1, 12), st.integers(1, 12))
def test_cache_key_follows_seed_and_labels_file_b(splits, seed1, seed2, m1, m2):
    """Property 26: a different seed or labels file gives a different file and re-extracts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        c1 = _write_corpus(tmp / "c1", seed1, splits, n_months=m1)
        c2 = _write_corpus(tmp / "c2", seed2, splits, n_months=m2)
        cache = tmp / "cache"
        e1 = FakeExtract()
        t1 = features_b.build_feature_table(c1, cache_root=cache, is_ignored=_always_ignored, extract=e1)
        assert not t1.from_cache and len(e1.calls) == len(splits)
        e2 = FakeExtract()
        t2 = features_b.build_feature_table(c2, cache_root=cache, is_ignored=_always_ignored, extract=e2)
        same_key = (seed1 == seed2) and (m1 == m2)
        assert (t1.cache_path == t2.cache_path) == same_key
        assert t2.from_cache == same_key
        assert (len(e2.calls) == 0) == same_key
        assert t1.cache_path.parent == cache / "features_b"
        # The header repeats the three key parts.
        first = t1.cache_path.read_text(encoding="utf-8").splitlines()[0]
        assert first == (f"# master_seed={seed1} labels_sha8={features_b.labels_sha8(c1)} "
                         f"extractor_sha8={features_b.extractor_sha8()}")


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=6), SEEDS,
       st.sampled_from(["master_seed", "labels_sha8", "extractor_sha8"]), HEX8)
def test_header_mismatch_is_not_read_b(splits, seed, field, junk):
    """Property 26: a 6b cache file whose header differs from the request is not read."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = _write_corpus(tmp / "c", seed, splits)
        cache = tmp / "cache"
        t1 = features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                            extract=FakeExtract())
        lines = t1.cache_path.read_text(encoding="utf-8").splitlines()
        parts = dict(p.split("=") for p in lines[0][2:].split())
        new_value = junk if field != "master_seed" else str(seed + 1)
        if parts[field] == new_value:
            new_value = new_value + "x"
        parts[field] = new_value
        lines[0] = "# " + " ".join(f"{k}={v}" for k, v in parts.items())
        t1.cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        e = FakeExtract()
        t2 = features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored, extract=e)
        assert not t2.from_cache and len(e.calls) == len(splits)
        # The file was rewritten with the correct header, so a third run reads it.
        e3 = FakeExtract()
        t3 = features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored, extract=e3)
        assert t3.from_cache and e3.calls == []


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=6), SEEDS)
def test_cache_hit_gives_same_values_b(splits, seed):
    """Property 26 (cache round trip): values read back equal the extracted values."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = _write_corpus(tmp / "c", seed, splits)
        t1 = features_b.build_feature_table(corpus, cache_root=tmp / "k", is_ignored=_always_ignored,
                                            extract=FakeExtract())
        t2 = features_b.build_feature_table(corpus, cache_root=tmp / "k", is_ignored=_always_ignored,
                                            extract=FakeExtract())
        assert t2.from_cache
        for col in features_b.EXTRACT_COLUMNS:
            assert list(t2.frame[col]) == list(t1.frame[col])


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=4), SEEDS, SEEDS,
       st.integers(1, 3), st.integers(1, 3))
def test_cache_key_follows_seed_and_rows_file_a(splits, seed1, seed2, r1, r2):
    """Property 26 (Node 6a): same key scheme with the rows.csv hash."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        c1 = _write_corpus(tmp / "c1", seed1, splits, rows_per_doc=r1)
        c2 = _write_corpus(tmp / "c2", seed2, splits, rows_per_doc=r2)
        cache = tmp / "cache"
        t1 = features_a.load_row_table(c1, cache_root=cache, is_ignored=_always_ignored, build=_fake_build)
        assert not t1.from_cache
        t2 = features_a.load_row_table(c2, cache_root=cache, is_ignored=_always_ignored, build=_fake_build)
        same_key = (seed1 == seed2) and (r1 == r2)
        assert (t1.cache_path == t2.cache_path) == same_key
        assert t2.from_cache == same_key
        assert t1.cache_path.parent == cache / "features_a"
        assert t1.cache_path.read_text(encoding="utf-8").splitlines()[0] == (
            f"# master_seed={seed1} rows_sha8={features_a.rows_sha8(c1)} "
            f"extractor_sha8={features_a.extractor_sha8()}")


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=4), SEEDS,
       st.sampled_from(["master_seed", "rows_sha8", "extractor_sha8"]))
def test_header_mismatch_is_not_read_a(splits, seed, field):
    """Property 26 (Node 6a): a cache file whose header differs from the request is not read."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = _write_corpus(tmp / "c", seed, splits)
        cache = tmp / "cache"
        t1 = features_a.load_row_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                       build=_fake_build)
        lines = t1.cache_path.read_text(encoding="utf-8").splitlines()
        parts = dict(p.split("=") for p in lines[0][2:].split())
        parts[field] = parts[field] + "0"
        lines[0] = "# " + " ".join(f"{k}={v}" for k, v in parts.items())
        t1.cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        t2 = features_a.load_row_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                       build=_fake_build)
        assert not t2.from_cache


@SETTINGS
@given(st.lists(st.sampled_from(SPLITS), min_size=1, max_size=3), SEEDS)
def test_cache_path_must_be_git_ignored(splits, seed):
    """Property 26: a cache folder that is not ignored is refused and nothing is written (6b and 6a)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = _write_corpus(tmp / "c", seed, splits)
        cache = tmp / "cache"
        with pytest.raises(WriteGuardError):
            features_b.build_feature_table(corpus, cache_root=cache, is_ignored=lambda p: False,
                                           extract=FakeExtract())
        with pytest.raises(WriteGuardError):
            features_a.load_row_table(corpus, cache_root=cache, is_ignored=lambda p: False,
                                      build=_fake_build)
        assert not cache.exists() or not list(cache.rglob("*.csv"))


# ------------------------------------------------------------------ Property 27
@st.composite
def _outcomes(draw):
    """A list of (split, fails) per document."""
    n = draw(st.integers(1, 30))
    splits = draw(st.lists(st.sampled_from(SPLITS), min_size=n, max_size=n))
    fails = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return splits, fails


@SETTINGS
@given(_outcomes(), st.integers(0, 100), SEEDS)
def test_skipped_document_accounting(outcome, limit, seed):
    """Property 27: skipped = exactly the unreadable documents; counts and percentages are
    the direct values; the run stops if and only if some Split is over the limit."""
    splits, fails = outcome
    failing = {f"d{i:03d}" for i, f in enumerate(fails) if f}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = _write_corpus(tmp / "c", seed, splits)
        cache = tmp / "cache"
        totals = {s: splits.count(s) for s in set(splits)}
        skipped = {s: sum(1 for i, sp in enumerate(splits) if sp == s and fails[i]) for s in totals}
        over = any(100.0 * skipped[s] / totals[s] > limit for s in totals)

        extract = FakeExtract(failing)
        if over:
            with pytest.raises(features_b.SkippedShareError) as err:
                features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                               extract=extract, skip_max_share=limit)
            assert err.value.exit_code == 2
            assert set(err.value.offenders) == {s for s in totals
                                                if 100.0 * skipped[s] / totals[s] > limit}
            # No cache is written for a run that stops.
            assert not list(cache.rglob("*.csv")) if cache.exists() else True
            return

        table = features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                               extract=extract, skip_max_share=limit)
        assert set(table.skipped_ids) == failing
        assert list(table.skipped_ids) == sorted(table.skipped_ids)
        kept = set(table.frame["doc_id"])
        assert kept.isdisjoint(failing)
        assert kept == {f"d{i:03d}" for i in range(len(splits))} - failing
        assert not table.frame["skipped"].any()
        assert set(table.skip_report) == set(totals)
        for s in totals:
            rep = table.skip_report[s]
            assert (rep["skipped"], rep["total"]) == (skipped[s], totals[s])
            assert rep["percent"] == pytest.approx(100.0 * skipped[s] / totals[s])

        # A cache hit reports the same skipped documents and none enters the table.
        again = features_b.build_feature_table(corpus, cache_root=cache, is_ignored=_always_ignored,
                                               extract=FakeExtract(), skip_max_share=limit)
        assert again.from_cache
        assert set(again.skipped_ids) == failing
        assert set(again.frame["doc_id"]) == kept
