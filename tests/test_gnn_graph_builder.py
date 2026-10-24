"""Tests for the split-safe internal-transfer graph builder (Node 6a GNN).

These exercise the three invariants the design's testing strategy calls out for
the graph builder:

    1. No edge connects nodes from different splits.
    2. No leaky / label-derived column appears in the node or edge feature
       tensors.
    3. Only ``counterparty_type == "INTERNAL"`` transfers become edges.

The builder returns a lightweight ``GraphBundle`` (pandas/numpy only), so these
run under plain pytest with NO torch / torch-geometric installed — matching the
offline-only dependency boundary in the design.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from fraud_detect.ml.gnn.graph_builder import (
    ACCOUNT_LABEL_COLUMNS,
    EDGE_FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    NODE_FEATURE_COLUMNS,
    build_graph,
    build_graph_from_frames,
)

# Columns that must NEVER appear in a node or edge feature frame.
_FORBIDDEN = set(LEAKY_COLUMNS) | set(ACCOUNT_LABEL_COLUMNS)

# The real dataset, if present, lets us assert the invariants hold end-to-end.
_REAL_DATA = os.path.expanduser("~/Downloads/transaction_fraud_dataset")


def _toy_frames():
    """Two accounts per split; a mix of INTERNAL / non-INTERNAL transfers,
    including an INTERNAL edge that crosses a split boundary (must be dropped)."""
    accts = pd.DataFrame(
        {
            "account_id": ["A1", "A2", "B1", "B2"],
            "segment": ["RETAIL", "SME", "RETAIL", "SME"],
            "open_date": ["2020-01-01"] * 4,
            "age_band": ["30-39", "40-49", "30-39", "50-59"],
            "declared_monthly_income_sgd": [5000, 8000, 6000, 9000],
            "declared_monthly_turnover_sgd": [0, 20000, 0, 30000],
            "business_type": ["NA", "F&B", "NA", "RETAIL"],
            "kyc_risk_rating": ["LOW", "MEDIUM", "LOW", "HIGH"],
            # label-derived + split columns that must stay out of features
            "is_suspicious_account": [0, 1, 0, 1],
            "account_fraud_pattern": ["NA", "ROUND_TRIPPING", "NA", "MULE"],
            "split": ["train", "train", "test", "test"],
            "split_group": ["g1", "g1", "g2", "g2"],
        }
    )
    txns = pd.DataFrame(
        {
            "txn_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "account_id": ["A1", "A1", "A2", "B1", "A1", "B1"],
            "timestamp": ["2021-01-01T10:00:00Z"] * 6,
            "direction": ["DR", "CR", "DR", "DR", "DR", "DR"],
            "amount_sgd": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "channel": ["FAST", "FAST", "GIRO", "FAST", "FAST", "GIRO"],
            "counterparty_id": ["A2", "A2", "A1", "B2", "B1", "X9"],
            "counterparty_type": [
                "INTERNAL",   # T1 A1->A2 (train, kept)
                "INTERNAL",   # T2 A1->A2 (train, kept)
                "INTERNAL",   # T3 A2->A1 (train, kept)
                "INTERNAL",   # T4 B1->B2 (test, kept in test graph)
                "EXTERNAL_LOCAL",  # T5 not INTERNAL -> dropped
                # T6 B1->X9: cp not an account in test split -> cross/dangling, dropped
                "INTERNAL",
            ],
            "counterparty_country": ["SG"] * 6,
            "merchant_category": ["NA"] * 6,
            "description": [""] * 6,
            "balance_after_sgd": [0.0] * 6,
            "split": ["train", "train", "train", "test", "train", "test"],
            "is_fraud": [1, 0, 0, 1, 0, 0],
            "fraud_pattern": ["ROUND_TRIPPING", "NA", "NA", "MULE", "NA", "NA"],
            "ring_id": ["R1", "NA", "NA", "R2", "NA", "NA"],
        }
    )
    return txns, accts


def test_only_internal_transfers_become_edges():
    """The non-INTERNAL transfer (T5) must not appear as an edge; the three
    valid INTERNAL train transfers must."""
    txns, accts = _toy_frames()
    g = build_graph_from_frames(txns, accts, "train")

    # T1, T2, T3 are INTERNAL within train; T5 is EXTERNAL_LOCAL (dropped).
    assert g.num_edges == 3
    # Every retained edge connects two train accounts (A1/A2 -> indices 0/1).
    assert set(g.node_ids) == {"A1", "A2"}
    for s, d in zip(g.edge_index[0], g.edge_index[1]):
        assert 0 <= s < g.num_nodes
        assert 0 <= d < g.num_nodes
    # Edge labels come from is_fraud (T1=1, T2=0, T3=0) in row order.
    assert sorted(g.edge_labels.tolist()) == [0, 0, 1]


def test_no_edge_crosses_a_split():
    """Both endpoints of every edge must be nodes within the SAME split, and
    a dangling/cross INTERNAL transfer (T6 B1->X9) is dropped."""
    txns, accts = _toy_frames()

    train_ids = set(accts.loc[accts.split == "train", "account_id"])
    test_ids = set(accts.loc[accts.split == "test", "account_id"])

    for split, ids in (("train", train_ids), ("test", test_ids)):
        g = build_graph_from_frames(txns, accts, split)
        node_set = set(g.node_ids)
        # Nodes are exactly the accounts in this split.
        assert node_set == ids
        # Every edge endpoint resolves to a node IN THIS split.
        for s, d in zip(g.edge_index[0], g.edge_index[1]):
            assert g.node_ids[s] in ids
            assert g.node_ids[d] in ids

    # The test split keeps only T4 (B1->B2); T6 (B1->X9, dangling) is dropped.
    g_test = build_graph_from_frames(txns, accts, "test")
    assert g_test.num_edges == 1


def test_no_leaky_column_in_node_or_edge_features():
    """No label-derived or split column may appear in the feature frames."""
    txns, accts = _toy_frames()
    g = build_graph_from_frames(txns, accts, "train")

    node_cols = set(g.node_features.columns)
    edge_cols = set(g.edge_features.columns)

    assert not (node_cols & _FORBIDDEN), f"leak in node features: {node_cols & _FORBIDDEN}"
    assert not (edge_cols & _FORBIDDEN), f"leak in edge features: {edge_cols & _FORBIDDEN}"
    # is_fraud is the edge LABEL, never a feature.
    assert "is_fraud" not in node_cols
    assert "is_fraud" not in edge_cols
    # Features are exactly the whitelisted columns.
    assert node_cols == set(NODE_FEATURE_COLUMNS)
    assert edge_cols == set(EDGE_FEATURE_COLUMNS)


def test_edge_labels_align_with_edges():
    """One label per edge, drawn from is_fraud."""
    txns, accts = _toy_frames()
    g = build_graph_from_frames(txns, accts, "train")
    assert g.edge_labels.shape[0] == g.num_edges
    assert set(np.unique(g.edge_labels).tolist()) <= {0, 1}


def test_empty_edge_set_is_well_formed():
    """A split with INTERNAL transfers removed still yields a valid empty graph."""
    txns, accts = _toy_frames()
    txns_no_internal = txns[txns.counterparty_type != "INTERNAL"].copy()
    g = build_graph_from_frames(txns_no_internal, accts, "train")
    assert g.num_edges == 0
    assert g.edge_index.shape == (2, 0)
    assert g.edge_labels.shape == (0,)
    # Nodes still present.
    assert g.num_nodes == 2


# --- End-to-end against the real dataset, when available --------------------

_real_available = (
    os.path.exists(os.path.join(_REAL_DATA, "transactions.csv"))
    and os.path.exists(os.path.join(_REAL_DATA, "accounts.csv"))
)


@pytest.mark.skipif(not _real_available, reason="real dataset not present")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_real_dataset_invariants(split):
    accts = pd.read_csv(os.path.join(_REAL_DATA, "accounts.csv"))
    g = build_graph(_REAL_DATA, split)

    split_ids = set(accts.loc[accts.split == split, "account_id"].astype(str))

    # No edge crosses a split: both endpoints are in this split.
    for s, d in zip(g.edge_index[0], g.edge_index[1]):
        assert g.node_ids[s] in split_ids
        assert g.node_ids[d] in split_ids

    # No leaky columns anywhere in the features.
    assert not (set(g.node_features.columns) & _FORBIDDEN)
    assert not (set(g.edge_features.columns) & _FORBIDDEN)
    assert "is_fraud" not in g.node_features.columns
    assert "is_fraud" not in g.edge_features.columns

    # Edge labels align with edges; graph has at least the nodes for the split.
    assert g.edge_labels.shape[0] == g.num_edges
    assert g.num_nodes == len(split_ids)
