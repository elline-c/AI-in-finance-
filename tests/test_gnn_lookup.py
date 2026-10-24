"""Fail-safe tests for the live-path GNN lookup adapter (Node 6a, Task 6.1).

These exercise ``fraud_detect.ml.gnn.gnn_lookup`` — the ONLY GNN module the live
pipeline imports (joblib only, no torch). The adapter must mirror the existing
model loaders' fail-safe contract: a missing file, an unknown key, or a corrupt /
malformed store all yield ``None`` so Node 6a simply emits no finding. A
well-formed stub store must return the expected score for a known account.

Stub stores are built in-test with ``joblib.dump`` to a tmp path (same pattern as
the other offline-safe GNN tests — no torch required).
"""
from __future__ import annotations

import os

import joblib
import pytest

from fraud_detect.ml.gnn import gnn_lookup


def _well_formed_store():
    """A minimal store matching the Task 5 format."""
    return {
        "account_scores": {"ACC000123": 0.87, "ACC000456": 0.12},
        "edge_scores": {"offshore nominee a pte ltd|~149k|credit": 0.91},
        "meta": {"model": "gin+reverse_mp+ego+ports", "pr_auc_test": 0.93},
    }


# --- positive path ----------------------------------------------------------

def test_well_formed_store_returns_expected_account_score(tmp_path):
    path = os.path.join(str(tmp_path), "gnn_scores.joblib")
    joblib.dump(_well_formed_store(), path)

    scores = gnn_lookup.load_scores(path)
    assert scores is not None
    assert gnn_lookup.score_account(scores, "ACC000123") == pytest.approx(0.87)
    assert gnn_lookup.score_account(scores, "ACC000456") == pytest.approx(0.12)


def test_well_formed_store_returns_expected_edge_score(tmp_path):
    path = os.path.join(str(tmp_path), "gnn_scores.joblib")
    joblib.dump(_well_formed_store(), path)

    scores = gnn_lookup.load_scores(path)
    key = "offshore nominee a pte ltd|~149k|credit"
    assert gnn_lookup.score_edge(scores, key) == pytest.approx(0.91)


# --- fail-safe: missing file ------------------------------------------------

def test_missing_file_returns_none(tmp_path):
    missing = os.path.join(str(tmp_path), "does_not_exist.joblib")
    assert gnn_lookup.load_scores(missing) is None


# --- fail-safe: unknown key -------------------------------------------------

def test_unknown_account_key_returns_none(tmp_path):
    path = os.path.join(str(tmp_path), "gnn_scores.joblib")
    joblib.dump(_well_formed_store(), path)

    scores = gnn_lookup.load_scores(path)
    assert gnn_lookup.score_account(scores, "ACC_UNKNOWN") is None


def test_unknown_edge_key_returns_none(tmp_path):
    path = os.path.join(str(tmp_path), "gnn_scores.joblib")
    joblib.dump(_well_formed_store(), path)

    scores = gnn_lookup.load_scores(path)
    assert gnn_lookup.score_edge(scores, "no|such|edge") is None


def test_lookup_on_none_store_returns_none():
    """A caller may pass the None from a failed load straight through."""
    assert gnn_lookup.score_account(None, "ACC000123") is None
    assert gnn_lookup.score_edge(None, "a|b|c") is None


# --- fail-safe: corrupt / malformed store -----------------------------------

def test_corrupt_file_returns_none(tmp_path):
    """A file that isn't a valid joblib pickle must fail safe to None."""
    path = os.path.join(str(tmp_path), "corrupt.joblib")
    with open(path, "wb") as fh:
        fh.write(b"this is not a valid joblib pickle \x00\x01\x02")
    assert gnn_lookup.load_scores(path) is None


def test_store_of_wrong_type_returns_none(tmp_path):
    """A loaded object that isn't a dict is treated as absent."""
    path = os.path.join(str(tmp_path), "wrong_type.joblib")
    joblib.dump(["not", "a", "dict"], path)
    assert gnn_lookup.load_scores(path) is None


def test_store_missing_expected_keys_returns_none(tmp_path):
    """A dict lacking the required account_scores section is malformed -> None."""
    path = os.path.join(str(tmp_path), "no_keys.joblib")
    joblib.dump({"meta": {"model": "gin"}}, path)
    assert gnn_lookup.load_scores(path) is None


def test_account_scores_wrong_type_returns_none(tmp_path):
    """account_scores present but not a dict is malformed -> None."""
    path = os.path.join(str(tmp_path), "bad_section.joblib")
    joblib.dump({"account_scores": [0.1, 0.2], "edge_scores": {}, "meta": {}}, path)
    assert gnn_lookup.load_scores(path) is None


def test_non_numeric_account_value_returns_none(tmp_path):
    """A stored value that can't be read as a float fails safe to None."""
    path = os.path.join(str(tmp_path), "bad_value.joblib")
    joblib.dump(
        {"account_scores": {"ACC000123": "not-a-number"}, "edge_scores": {}, "meta": {}},
        path,
    )
    scores = gnn_lookup.load_scores(path)
    assert scores is not None  # store shape is valid
    assert gnn_lookup.score_account(scores, "ACC000123") is None  # value unreadable
