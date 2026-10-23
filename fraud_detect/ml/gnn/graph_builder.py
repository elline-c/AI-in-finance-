"""Split-safe internal-transfer graph builder for the offline GNN (Node 6a).

This module turns the dataset's ``transactions.csv`` / ``accounts.csv`` into a
graph the Multi-GNN models can train on, while enforcing the SAME data-card
leakage guards the tabular path uses (``transaction_features``):

    * Nodes = accounts. Node features reuse ONLY the safe static KYC columns
      whitelisted in ``transaction_features.load_joined`` (segment,
      open_date/account_age, age_band, declared income/turnover, business_type,
      kyc_risk_rating). The account-level labels ``is_suspicious_account`` and
      ``account_fraud_pattern`` are NEVER used as features.
    * Edges = ``counterparty_type == "INTERNAL"`` transfers ONLY. External
      counterparties are randomized hashes per the data card and carry no graph
      signal. Edge features = amount, direction, channel, timestamp. Edge label
      = ``is_fraud``.
    * Leakage guards (verbatim from ``transaction_features.LEAKY_COLUMNS``):
      ``is_fraud``, ``fraud_pattern``, ``ring_id``, ``split`` never enter the
      node/edge FEATURE tensors; ``eval_tags`` is never joined. (``is_fraud`` is
      kept separately as the edge *label*, not a feature.)
    * Split-safe: one graph is built PER SPLIT. Internal transfers never cross
      splits (data-card guarantee), so a per-split graph has no train/val/test
      leakage and no edge connects accounts from two different splits.

Heavy-dependency boundary
--------------------------
``torch`` / ``torch_geometric`` are offline-only deps (``requirements-gnn.txt``)
and are imported LAZILY inside :func:`to_pyg_data`. Importing this module — and
building / testing the split-safe, leakage-free, INTERNAL-only core — requires
only ``pandas`` + ``numpy``. :func:`build_graph` returns a plain
:class:`GraphBundle` dataclass; conversion to a PyTorch-Geometric ``Data`` object
happens only when the caller explicitly asks and PyG is installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch import
    from torch_geometric.data import Data

# Leakage guards carried over VERBATIM from the tabular feature builder.
# These never become node or edge *features*. ``is_fraud`` additionally serves
# as the edge label and is surfaced separately in ``GraphBundle.edge_labels``.
from ..transaction_features import LEAKY_COLUMNS

# Account-level label columns from accounts.csv — label-derived, never features.
ACCOUNT_LABEL_COLUMNS = {"is_suspicious_account", "account_fraud_pattern"}

# Safe static KYC node features — identical whitelist to
# ``transaction_features.load_joined`` (minus the join key ``account_id``).
NODE_FEATURE_COLUMNS: List[str] = [
    "segment",
    "open_date",
    "age_band",
    "declared_monthly_income_sgd",
    "declared_monthly_turnover_sgd",
    "business_type",
    "kyc_risk_rating",
]

# Edge features for INTERNAL transfers (per the design: amount, direction,
# channel, timestamp). ``is_fraud`` is the LABEL, surfaced separately.
EDGE_FEATURE_COLUMNS: List[str] = [
    "amount_sgd",
    "direction",
    "channel",
    "timestamp",
]

INTERNAL_COUNTERPARTY = "INTERNAL"
SPLIT_COL = "split"
TARGET = "is_fraud"


@dataclass
class GraphBundle:
    """A split-safe, leakage-free graph for ONE dataset split.

    This is a lightweight, framework-agnostic container so the split-safety /
    leakage / INTERNAL-only invariants can be tested without torch installed.
    Convert to a PyTorch-Geometric ``Data`` object with :meth:`to_pyg_data`
    only in the offline training path where PyG is available.

    Attributes
    ----------
    split:
        The dataset split this graph belongs to ("train" / "val" / "test").
    node_ids:
        Account ids, one per node, indexed by node position.
    node_features:
        Per-node safe KYC features. Columns are exactly ``NODE_FEATURE_COLUMNS``
        — no leaky / label-derived columns. Row order matches ``node_ids``.
    edge_index:
        ``(2, num_edges)`` int array of [source_node_idx, dest_node_idx] for the
        INTERNAL transfers. Both endpoints are nodes in THIS split.
    edge_features:
        Per-edge features. Columns are exactly ``EDGE_FEATURE_COLUMNS``.
    edge_labels:
        ``(num_edges,)`` int array of ``is_fraud`` (0/1) for each edge. This is
        the training label and is intentionally kept OUT of ``edge_features``.
    """

    split: str
    node_ids: List[str]
    node_features: pd.DataFrame
    edge_index: np.ndarray
    edge_features: pd.DataFrame
    edge_labels: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_pyg_data(self) -> "Data":
        """Convert to a PyTorch-Geometric ``Data`` object.

        ``torch`` / ``torch_geometric`` are imported LAZILY here so the live
        path and the test suite never require them. Categorical feature columns
        are integer-encoded (stable, per-column) before tensor conversion.
        """
        import torch  # lazy, offline-only
        from torch_geometric.data import Data  # lazy, offline-only

        x = torch.tensor(_encode_frame(self.node_features), dtype=torch.float)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(_encode_frame(self.edge_features),
                                 dtype=torch.float)
        y = torch.tensor(self.edge_labels, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _encode_frame(frame: pd.DataFrame) -> np.ndarray:
    """Integer/float-encode a feature frame to a dense float matrix.

    Numeric columns pass through (coerced, NaN->0). Non-numeric columns get a
    stable per-column integer code. Used only inside the lazy PyG conversion.
    """
    cols = []
    for name in frame.columns:
        s = frame[name]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any() and num.notna().mean() >= 0.5:
            cols.append(num.fillna(0.0).to_numpy(dtype="float64"))
        else:
            codes = pd.Categorical(s.astype("string").fillna("NA")).codes
            cols.append(codes.astype("float64"))
    if not cols:
        return np.zeros((len(frame), 0), dtype="float64")
    return np.column_stack(cols)


def _assert_no_leaky_columns(frame: pd.DataFrame, where: str) -> None:
    """Fail loudly (offline) if any guarded column leaked into a feature frame."""
    forbidden = (set(LEAKY_COLUMNS) | ACCOUNT_LABEL_COLUMNS) - {TARGET}
    # ``is_fraud`` (TARGET) is only ever the edge LABEL, never in a feature frame.
    present = (set(frame.columns) & forbidden) | ({TARGET} & set(frame.columns))
    if present:
        raise ValueError(
            f"leakage guard violated: {sorted(present)} present in {where} "
            f"features"
        )


def build_graph(data_dir: str, split: str) -> GraphBundle:
    """Build the split-safe INTERNAL-transfer graph for a single ``split``.

    Parameters
    ----------
    data_dir:
        Folder containing ``transactions.csv`` and ``accounts.csv``.
    split:
        One of ``"train"`` / ``"val"`` / ``"test"``. Only accounts and edges in
        this split are included — no cross-split edges are ever created.

    Returns
    -------
    GraphBundle
        Nodes = accounts in ``split``; edges = INTERNAL transfers whose BOTH
        endpoints are in ``split``; node features = safe KYC columns; edge
        features = amount/direction/channel/timestamp; edge labels = is_fraud.
    """
    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    accts = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    return build_graph_from_frames(txns, accts, split)


def build_graph_from_frames(
    txns: pd.DataFrame, accts: pd.DataFrame, split: str
) -> GraphBundle:
    """Frame-level core of :func:`build_graph` (no disk IO).

    Separated out so tests can feed small in-memory frames without writing CSVs.
    Enforces the same split-safety, leakage, and INTERNAL-only guarantees.
    """
    # --- Nodes: accounts in THIS split only -------------------------------
    acct_split = accts[accts[SPLIT_COL] == split].copy()
    node_ids: List[str] = acct_split["account_id"].astype(str).tolist()
    node_pos = {aid: i for i, aid in enumerate(node_ids)}
    node_in_split = set(node_ids)

    # Safe KYC node features only — explicit whitelist, never label columns.
    node_cols = [c for c in NODE_FEATURE_COLUMNS if c in acct_split.columns]
    node_features = acct_split[node_cols].reset_index(drop=True).copy()
    _assert_no_leaky_columns(node_features, where="node")

    # --- Edges: INTERNAL transfers whose BOTH endpoints are in this split --
    internal = txns[txns["counterparty_type"] == INTERNAL_COUNTERPARTY].copy()
    # Restrict to this split up front (the txn split == source account split).
    internal = internal[internal[SPLIT_COL] == split]
    internal["account_id"] = internal["account_id"].astype(str)
    internal["counterparty_id"] = internal["counterparty_id"].astype(str)

    # Keep only edges where BOTH endpoints are nodes in this split — the hard
    # guarantee that no edge connects nodes from different splits.
    both_in_split = (
        internal["account_id"].isin(node_in_split)
        & internal["counterparty_id"].isin(node_in_split)
    )
    internal = internal[both_in_split].reset_index(drop=True)

    if len(internal) > 0:
        src = internal["account_id"].map(node_pos).to_numpy(dtype="int64")
        dst = internal["counterparty_id"].map(node_pos).to_numpy(dtype="int64")
        edge_index = np.vstack([src, dst])
        edge_labels = (
            pd.to_numeric(internal[TARGET], errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy()
        )
    else:
        edge_index = np.empty((2, 0), dtype="int64")
        edge_labels = np.empty((0,), dtype="int64")

    edge_cols = [c for c in EDGE_FEATURE_COLUMNS if c in internal.columns]
    edge_features = internal[edge_cols].reset_index(drop=True).copy()
    _assert_no_leaky_columns(edge_features, where="edge")

    return GraphBundle(
        split=split,
        node_ids=node_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_labels=edge_labels,
        meta={
            "data_card": "internal transfers never cross splits",
            "node_feature_columns": node_cols,
            "edge_feature_columns": edge_cols,
        },
    )
