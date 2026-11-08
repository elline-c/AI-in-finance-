"""Task 9.1: failing tests for ``fraud_detect.ml.retraining.features_a`` (Node 6a table).

Written before the module exists. Only the Mini_Corpus fixture is used (Requirement 17.9).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.features_a``)::

    class ForbiddenFeatureError(Exception)  # exit_code == gates.EXIT_INTEGRITY; .offenders: sorted list[str]
    class RowExclusionError(Exception)      # exit_code == gates.EXIT_INTEGRITY; .share: percent
    def extractor_sha8() -> str             # sha256(bytes of transaction_features.py).hexdigest()[:8]
    def rows_sha8(corpus_dir) -> str        # sha256(bytes of rows.csv).hexdigest()[:8]
    def cache_path_a(cache_root, master_seed, rows_sha8, extractor_sha8) -> Path
        # == Path(cache_root) / "features_a" / f"{master_seed}-{rows_sha8}-{extractor_sha8}.csv"
    def account_id_for(company_key) -> str  # f"{master_seed}-{company_id}" (qualified; never the raw id)
    def rows_to_frame(corpus_dir) -> tuple[DataFrame, int]
        # (included rows in rows.csv order, number excluded for a missing / non-binary is_fraud_row).
        # columns: corpus_key, doc_id, row_index (int), account_id, timestamp (= date, ISO string),
        # direction ("CR" if amount >= 0 else "DR"), amount_sgd (= abs amount), balance_after_sgd
        # (= balance), y (int 0/1 = is_fraud_row), split (value of labels.csv for the document).
        # Same mapping as ml_transaction._rows_from_ingestion. Company comes from labels.csv.
    def build_pdf_table(rows, build=None) -> DataFrame
        # rows: a frame like rows_to_frame's. Returns, in the SAME row order, corpus_key, doc_id,
        # row_index, account_id, then transaction_features.PDF_FEATURE_COLUMNS, then y, split.
        # Calls build_pdf_features (injectable via ``build``). Never uses y as an input.
    def load_row_table(corpus_dir, *, cache_root, columns=None, is_ignored=None, build=None,
                       row_exclusion_max_share=1) -> RowTableA
        # columns: optional requested feature columns; any not in PDF_FEATURE_COLUMNS raises
        # ForbiddenFeatureError (all offenders listed) BEFORE anything is written.
        # row_exclusion_max_share: percent; RowExclusionError when excluded share is strictly greater.
    RowTableA (dataclass): .rows (DataFrame), .features (DataFrame as build_pdf_table, minus
        unrequested feature columns), .excluded_count (int), .excluded_share (percent float),
        .cache_path (Path), .from_cache (bool)
    Cache file: first line "# master_seed=<n> rows_sha8=<x> extractor_sha8=<y>", then CSV of .features.
        A cache whose first line differs from the request is not read. Cache path is checked with
        safe_io.ensure_ignored (is_ignored injectable) before writing.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining import gates
from fraud_detect.ml.retraining.corpora import load_corpus
from fraud_detect.ml.retraining.features_a import (
    ForbiddenFeatureError, RowExclusionError, account_id_for, build_pdf_table, cache_path_a,
    extractor_sha8, load_row_table, rows_sha8, rows_to_frame)
from fraud_detect.ml.retraining.safe_io import WriteGuardError
from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction
from fraud_detect.nodes import ml_transaction

KEY_COLS = ["corpus_key", "doc_id", "row_index", "account_id"]
FORBIDDEN_NAMES = {"is_fraud_row", "fraud_type", "kind", "is_fraud", "is_tampered", "tamper_types",
                   "tamper_method", "genuine_resaved", "layout", "fraud_pattern", "ring_id"}


def _csv_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _copy_meta(src: Path, dst: Path) -> Path:
    """Copy only the small metadata files (never PDFs)."""
    dst.mkdir(parents=True)
    for name in ("labels.csv", "rows.csv", "companies.json", "corpus_manifest.json"):
        shutil.copy(src / name, dst / name)
    return dst


def _set_labels(corpus: Path, indices, value: str) -> None:
    rows = _csv_rows(corpus / "rows.csv")
    fields = list(rows[0].keys())
    for i in indices:
        rows[i]["is_fraud_row"] = value
    with open(corpus / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def frame(mini_corpus):
    rows, excluded = rows_to_frame(mini_corpus.path())
    return rows, excluded


# ------------------------------------------------------------------ adapter
def test_adapter_columns_and_mapping(mini_corpus, frame):
    rows, excluded = frame
    src = _csv_rows(mini_corpus.path() / "rows.csv")
    assert excluded == 0 and len(rows) == len(src) > 0
    assert list(rows.columns)[:4] == KEY_COLS
    for name in ("timestamp", "direction", "amount_sgd", "balance_after_sgd", "y", "split"):
        assert name in rows.columns
    first = src[0]
    r0 = rows.iloc[0]
    assert r0["doc_id"] == first["doc_id"] and int(r0["row_index"]) == int(first["row_index"])
    assert r0["timestamp"] == first["date"]
    assert r0["amount_sgd"] == pytest.approx(abs(float(first["amount"])))
    assert r0["direction"] == ("CR" if float(first["amount"]) >= 0 else "DR")
    assert r0["balance_after_sgd"] == pytest.approx(float(first["balance"]))
    assert int(r0["y"]) == int(first["is_fraud_row"])


def test_adapter_split_and_account_come_from_labels_not_recomputed(mini_corpus, frame):
    rows, _ = frame
    docs = {d.doc_id: d for d in load_corpus(mini_corpus.path()).docs}
    for doc_id, grp in rows.groupby("doc_id"):
        assert set(grp["split"]) == {docs[doc_id].split}
        assert set(grp["account_id"]) == {account_id_for(docs[doc_id].company_key)}


def test_account_id_is_qualified_by_seed(mini_corpus):
    assert account_id_for((20260801, "C001")) != account_id_for((20260802, "C001"))
    assert "C001" in account_id_for((20260801, "C001"))


def test_adapter_matches_rows_from_ingestion_on_a_fake_result(mini_corpus, frame):
    rows, _ = frame
    account = rows["account_id"].iloc[0]
    mine = rows[rows["account_id"] == account].reset_index(drop=True)
    ing = IngestionResult(
        source_path="fake.pdf", page_count=1, raw_text="", metadata=DocumentMetadata(), uen=account,
        transactions=[Transaction(index=int(r.row_index), date=date.fromisoformat(r.timestamp),
                                  amount=float(r.amount_sgd) if r.direction == "CR" else -float(r.amount_sgd),
                                  balance=float(r.balance_after_sgd))
                      for r in mine.itertuples()])
    theirs = ml_transaction._rows_from_ingestion(ing)
    for col in ("account_id", "timestamp", "direction", "amount_sgd", "balance_after_sgd"):
        assert list(mine[col]) == list(theirs[col]), col


# ------------------------------------------------------------------ PDF feature columns
def test_pdf_features_equal_build_pdf_features_exactly(frame):
    rows, _ = frame
    table = build_pdf_table(rows)
    assert list(table.columns) == KEY_COLS + list(tf.PDF_FEATURE_COLUMNS) + ["y", "split"]
    assert len(table) == len(rows)
    assert list(table["doc_id"]) == list(rows["doc_id"]) and list(table["row_index"]) == list(rows["row_index"])
    direct = rows.rename(columns={}).copy()
    for col, default in (("channel", "NA"), ("counterparty_type", "NA"), ("counterparty_country", "NA"),
                         ("merchant_category", "NA"), ("segment", "NA"), ("age_band", "NA"),
                         ("business_type", "NA"), ("kyc_risk_rating", "NA"),
                         ("declared_monthly_income_sgd", 0.0), ("declared_monthly_turnover_sgd", 0.0)):
        direct[col] = default
    direct["open_date"] = None
    direct["is_fraud"] = direct["y"]
    expected = tf.build_pdf_features(direct)
    pd.testing.assert_frame_equal(
        table[list(tf.PDF_FEATURE_COLUMNS)].reset_index(drop=True),
        expected[list(tf.PDF_FEATURE_COLUMNS)].reset_index(drop=True), check_exact=True)


def test_no_label_or_corpus_column_in_the_feature_table(frame):
    table = build_pdf_table(frame[0])
    assert not FORBIDDEN_NAMES & set(table.columns)


def test_feature_values_do_not_depend_on_labels(frame):
    rows, _ = frame
    flipped = rows.copy()
    flipped["y"] = 1 - flipped["y"].astype(int)
    a, b = build_pdf_table(rows), build_pdf_table(flipped)
    pd.testing.assert_frame_equal(a[list(tf.PDF_FEATURE_COLUMNS)], b[list(tf.PDF_FEATURE_COLUMNS)],
                                  check_exact=True)


# ------------------------------------------------------------------ history locality
def _company_block(table, account):
    sub = table[table["account_id"] == account]
    return sub.set_index(["corpus_key", "doc_id", "row_index"])[list(tf.PDF_FEATURE_COLUMNS)]


def test_history_features_are_local_to_the_company(mini_corpus):
    rows_a, _ = rows_to_frame(mini_corpus.path(mini_corpus.seeds[0]))
    rows_b, _ = rows_to_frame(mini_corpus.path(mini_corpus.seeds[1]))
    account = rows_a["account_id"].iloc[0]
    block = rows_a[rows_a["account_id"] == account]
    others = rows_a[rows_a["account_id"] != account]
    base = _company_block(build_pdf_table(block), account)
    assert len(base) > 0
    variants = {
        "all_rows_of_corpus": pd.concat([rows_a]),
        "other_companies_first": pd.concat([others, block]),
        "other_companies_after": pd.concat([block, others]),
        "plus_other_corpus": pd.concat([rows_a, rows_b]),
        "other_corpus_first": pd.concat([rows_b, rows_a]),
    }
    for name, variant in variants.items():
        got = _company_block(build_pdf_table(variant.reset_index(drop=True)), account)
        pd.testing.assert_frame_equal(base, got, check_exact=True, obj=name)


# ------------------------------------------------------------------ exclusion accounting
def test_rows_with_missing_or_non_binary_label_are_excluded_and_counted(mini_corpus, tmp_path):
    corpus = _copy_meta(mini_corpus.path(), tmp_path / "c")
    _set_labels(corpus, [0], "")
    _set_labels(corpus, [1], "2")
    total = len(_csv_rows(corpus / "rows.csv"))
    rows, excluded = rows_to_frame(corpus)
    assert excluded == 2 and len(rows) == total - 2
    ids = {(r["doc_id"], int(r["row_index"])) for r in _csv_rows(corpus / "rows.csv")[:2]}
    assert not ids & set(zip(rows["doc_id"], rows["row_index"].astype(int)))
    table = load_row_table(corpus, cache_root=tmp_path / "cache", is_ignored=lambda p: True,
                           row_exclusion_max_share=100)
    assert table.excluded_count == 2
    assert table.excluded_share == pytest.approx(100.0 * 2 / total)
    assert len(table.rows) == len(table.features) == total - 2


def test_excluded_share_over_limit_aborts_with_exit_2_and_writes_nothing(mini_corpus, tmp_path):
    corpus = _copy_meta(mini_corpus.path(), tmp_path / "c")
    total = len(_csv_rows(corpus / "rows.csv"))
    bad = int(total * 0.01) + 2
    _set_labels(corpus, list(range(bad)), "")
    cache = tmp_path / "cache"
    with pytest.raises(RowExclusionError) as err:
        load_row_table(corpus, cache_root=cache, is_ignored=lambda p: True)  # default 1 percent
    assert err.value.exit_code == gates.EXIT_INTEGRITY
    assert err.value.share > 1.0
    assert not cache.exists() or not list(cache.rglob("*.csv"))


# ------------------------------------------------------------------ refusal of non-PDF columns
def test_non_pdf_columns_are_refused_with_all_names_and_nothing_written(mini_corpus, tmp_path):
    cache = tmp_path / "cache"
    requested = ["amount_sgd", "kyc_risk_rating", "channel", "is_fraud_row", "day_of_week"]
    with pytest.raises(ForbiddenFeatureError) as err:
        load_row_table(mini_corpus.path(), cache_root=cache, columns=requested, is_ignored=lambda p: True)
    assert err.value.exit_code == gates.EXIT_INTEGRITY
    assert err.value.offenders == ["channel", "is_fraud_row", "kyc_risk_rating"]
    assert not cache.exists() or not list(cache.rglob("*"))


def test_pdf_columns_are_accepted(mini_corpus, tmp_path):
    wanted = ["amount_sgd", "day_of_week", "direction"]
    table = load_row_table(mini_corpus.path(), cache_root=tmp_path, columns=wanted, is_ignored=lambda p: True)
    for name in wanted:
        assert name in table.features.columns
    assert "n_txn_24h" not in table.features.columns


# ------------------------------------------------------------------ cache
def test_cache_path_header_and_key_parts(mini_corpus, tmp_path):
    seed = mini_corpus.seeds[0]
    table = load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True)
    rows_bytes = (mini_corpus.path() / "rows.csv").read_bytes()
    assert rows_sha8(mini_corpus.path()) == hashlib.sha256(rows_bytes).hexdigest()[:8]
    assert extractor_sha8() == hashlib.sha256(Path(tf.__file__).read_bytes()).hexdigest()[:8]
    expected = tmp_path / "features_a" / f"{seed}-{rows_sha8(mini_corpus.path())}-{extractor_sha8()}.csv"
    assert cache_path_a(tmp_path, seed, rows_sha8(mini_corpus.path()), extractor_sha8()) == expected
    assert table.cache_path == expected and expected.is_file()
    first = expected.read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# master_seed={seed} rows_sha8={rows_sha8(mini_corpus.path())} extractor_sha8={extractor_sha8()}"


def _counting_build():
    calls = []

    def build(df):
        calls.append(len(df))
        return tf.build_pdf_features(df)

    build.calls = calls
    return build


def test_second_run_reads_the_cache(mini_corpus, tmp_path):
    b1 = _counting_build()
    t1 = load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True, build=b1)
    assert b1.calls and not t1.from_cache
    b2 = _counting_build()
    t2 = load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True, build=b2)
    assert b2.calls == [] and t2.from_cache
    for name in tf.PDF_FEATURE_COLUMNS:
        if name != "direction":
            assert np.allclose(t1.features[name].astype(float), t2.features[name].astype(float))
    assert list(t1.features["doc_id"]) == list(t2.features["doc_id"])


def test_header_that_differs_is_not_read(mini_corpus, tmp_path):
    t1 = load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True)
    lines = t1.cache_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(f"master_seed={mini_corpus.seeds[0]}", "master_seed=1")
    t1.cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build = _counting_build()
    t2 = load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=lambda p: True, build=build)
    assert build.calls and not t2.from_cache


def test_other_seed_or_rows_file_gives_other_key(mini_corpus, tmp_path):
    a = load_row_table(mini_corpus.path(mini_corpus.seeds[0]), cache_root=tmp_path, is_ignored=lambda p: True)
    b = load_row_table(mini_corpus.path(mini_corpus.seeds[1]), cache_root=tmp_path, is_ignored=lambda p: True)
    assert a.cache_path != b.cache_path
    edited = _copy_meta(mini_corpus.path(), tmp_path / "edited")
    with open(edited / "rows.csv", "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert rows_sha8(edited) != rows_sha8(mini_corpus.path())
    build = _counting_build()
    c = load_row_table(edited, cache_root=tmp_path, is_ignored=lambda p: True, build=build)
    assert c.cache_path not in (a.cache_path, b.cache_path) and build.calls


def test_cache_path_is_checked_with_the_write_guard(mini_corpus, tmp_path, is_ignored):
    load_row_table(mini_corpus.path(), cache_root=tmp_path, is_ignored=is_ignored)
    assert any(p.parent.name == "features_a" for p in is_ignored.calls)
    is_ignored.ignored = False
    other = tmp_path / "other"
    with pytest.raises(WriteGuardError):
        load_row_table(mini_corpus.path(), cache_root=other, is_ignored=is_ignored)
    assert not other.exists() or not list(other.rglob("*.csv"))


def test_import_pulls_no_pdf_or_torch_libraries():
    code = ("import sys, fraud_detect.ml.retraining.features_a;"
            "bad = [m for m in ('pdfplumber', 'torch', 'torch_geometric') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
