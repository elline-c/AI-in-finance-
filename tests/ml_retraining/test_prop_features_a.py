"""Task 9.5: property tests for the Node 6a adapter and history locality.

Property 5: Account-history features are local.
Property 28: Node 6a row adapter.

**Validates: Requirements 1.8, 11.2, 11.3, 11.4, 11.5**

Small synthetic rows and tiny synthetic corpora (no PDFs) keep every example fast. The real
``build_pdf_features`` is used for Property 5; Property 28 uses it too through ``load_row_table``.
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
from fraud_detect.ml.retraining import features_a, gates

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

SPLITS = ("train", "validation", "test")
FEATURES = list(tf.PDF_FEATURE_COLUMNS)
FORBIDDEN_NAMES = {"is_fraud_row", "fraud_type", "kind", "is_fraud", "is_tampered", "tamper_types",
                   "tamper_method", "genuine_resaved", "layout", "fraud_pattern", "ring_id",
                   "doc_label", "company_id"}


# ------------------------------------------------------------------ Property 5
@st.composite
def company_rows(draw, account, seed):
    n = draw(st.integers(1, 6))
    split = draw(st.sampled_from(SPLITS))
    recs = []
    for i in range(n):
        day = draw(st.integers(1, 28))
        month = draw(st.integers(1, 3))
        recs.append({
            "corpus_key": f"corpus-{seed}", "doc_id": f"{account}-d", "row_index": i,
            "account_id": account, "timestamp": f"2026-{month:02d}-{day:02d}",
            "direction": draw(st.sampled_from(["CR", "DR"])),
            "amount_sgd": float(draw(st.integers(1, 5000))),
            "balance_after_sgd": float(draw(st.integers(0, 20000))),
            "y": draw(st.integers(0, 1)), "split": split})
    return pd.DataFrame(recs)


@st.composite
def locality_case(draw):
    n = draw(st.integers(2, 4))
    blocks = [draw(company_rows(f"{1000 + i}-C{i}", 1000 + i)) for i in range(n)]
    return blocks, draw(st.randoms(use_true_random=False))


def _block(table, account):
    sub = table[table["account_id"] == account]
    return sub.set_index(["doc_id", "row_index"])[FEATURES]


@SETTINGS
@given(locality_case())
def test_history_features_unchanged_by_other_companies(case):
    """Property 5: adding, removing or reordering other companies' rows leaves a company's features unchanged."""
    blocks, rnd = case
    target, others = blocks[0], blocks[1:]
    account = target["account_id"].iloc[0]
    base = _block(features_a.build_pdf_table(target), account)
    assert len(base) == len(target)

    other_rows = pd.concat(others, ignore_index=True)
    # Interleave: keep the target's own order, shuffle everything else around it.
    mixed = pd.concat([target, other_rows], ignore_index=True)
    order = list(range(len(mixed)))
    rnd.shuffle(order)
    tgt_positions = [i for i in order if i < len(target)]
    tgt_positions_sorted = sorted(tgt_positions)
    it = iter(tgt_positions_sorted)
    order = [next(it) if i < len(target) else i for i in order]  # restore own relative order
    variants = {
        "others_after": pd.concat([target, other_rows], ignore_index=True),
        "others_before": pd.concat([other_rows, target], ignore_index=True),
        "interleaved": mixed.iloc[order].reset_index(drop=True),
        "one_other_removed": pd.concat([target] + others[:-1], ignore_index=True),
    }
    for name, variant in variants.items():
        got = _block(features_a.build_pdf_table(variant), account)
        pd.testing.assert_frame_equal(base, got, check_exact=True, obj=name)


# ------------------------------------------------------------------ Property 28
LABEL = st.sampled_from(["0", "1", "0", "1", "", "2", "-1", "x", "0.5"])


def _write_corpus(root: Path, seed: int, labels: list, splits: list) -> Path:
    """One document per label list; each document is its own company."""
    root.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "split", "layout", "company_id", "is_tampered", "genuine_resaved",
                    "tamper_types", "tamper_method", "n_months"])
        for i in range(n):
            w.writerow([f"d{i:03d}", splits[i], "L1", f"c{i:03d}", "1", "0", "", "", 6])
    with open(root / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "row_index", "date", "amount", "balance", "is_fraud_row",
                    "fraud_type", "kind"])
        for i, doc_labels in enumerate(labels):
            for r, lab in enumerate(doc_labels):
                w.writerow([f"d{i:03d}", r, f"2026-01-{r + 1:02d}", (-1) ** r * (10.0 + r),
                            100.0 + r, lab, "structuring", "tampered"])
    (root / "companies.json").write_text(
        json.dumps([{"company_id": f"c{i:03d}", "uen": f"U{i}"} for i in range(n)]), encoding="utf-8")
    (root / "corpus_manifest.json").write_text(json.dumps({"master_seed": seed}), encoding="utf-8")
    return root


@st.composite
def corpus_case(draw):
    n_docs = draw(st.integers(1, 3))
    labels = [draw(st.lists(LABEL, min_size=1, max_size=4)) for _ in range(n_docs)]
    splits = [draw(st.sampled_from(SPLITS)) for _ in range(n_docs)]
    return labels, splits


def _always_ignored(_path) -> bool:
    return True


@SETTINGS
@given(corpus_case())
def test_row_adapter_labels_exclusion_and_columns(case):
    """Property 28: label vector equals is_fraud_row, bad labels excluded and counted, no label-derived columns."""
    labels, splits = case
    flat = [lab for doc in labels for lab in doc]
    good = [lab for lab in flat if lab in ("0", "1")]
    total, excluded = len(flat), len(flat) - len(good)
    share = 100.0 * excluded / total
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_corpus(Path(tmp) / "c", 7, labels, splits)
        cache = Path(tmp) / "cache"
        rows, got_excluded = features_a.rows_to_frame(root)
        assert got_excluded == excluded
        assert list(rows["y"]) == [int(x) for x in good]

        if share > 1.0:
            with pytest.raises(features_a.RowExclusionError) as err:
                features_a.load_row_table(root, cache_root=cache, is_ignored=_always_ignored)
            assert err.value.exit_code == gates.EXIT_INTEGRITY
            assert err.value.share == pytest.approx(share)
            assert not cache.exists() or not list(cache.rglob("*.csv"))
            return

        table = features_a.load_row_table(root, cache_root=cache, is_ignored=_always_ignored)
        assert table.excluded_count == excluded
        assert table.excluded_share == pytest.approx(share)
        feats = table.features
        assert len(feats) == len(table.rows) == len(good)
        assert list(feats["y"]) == [int(x) for x in good] == list(table.rows["y"])
        assert list(feats.columns) == features_a.KEY_COLS + FEATURES + ["y", "split"]
        assert not FORBIDDEN_NAMES & set(feats.columns)

        # Features never depend on the label: flip every label and compare.
        flipped = table.rows.copy()
        flipped["y"] = 1 - flipped["y"]
        other = features_a.build_pdf_table(flipped)
        pd.testing.assert_frame_equal(feats[FEATURES].reset_index(drop=True),
                                      other[FEATURES].reset_index(drop=True), check_exact=True)


REQUESTED = st.lists(
    st.sampled_from(FEATURES[:4] + ["is_fraud_row", "fraud_type", "kind", "channel",
                                    "kyc_risk_rating", "is_tampered", "layout", "segment"]),
    min_size=0, max_size=6, unique=True)


@SETTINGS
@given(REQUESTED)
def test_non_pdf_columns_refused_with_all_names_and_nothing_written(requested):
    """Property 28: any column outside PDF_FEATURE_COLUMNS is refused, all offenders named, nothing written."""
    offenders = sorted(c for c in requested if c not in set(FEATURES))
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_corpus(Path(tmp) / "c", 7, [["0", "1"]], ["train"])
        cache = Path(tmp) / "cache"
        if offenders:
            with pytest.raises(features_a.ForbiddenFeatureError) as err:
                features_a.load_row_table(root, cache_root=cache, columns=requested,
                                          is_ignored=_always_ignored)
            assert err.value.offenders == offenders
            assert err.value.exit_code == gates.EXIT_INTEGRITY
            assert not cache.exists() or not list(cache.rglob("*"))
        else:
            table = features_a.load_row_table(root, cache_root=cache, columns=requested,
                                              is_ignored=_always_ignored)
            assert set(FEATURES) & set(table.features.columns) == set(requested)
