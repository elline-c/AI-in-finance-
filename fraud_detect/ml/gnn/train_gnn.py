"""Offline trainer for the Multi-GNN transaction-fraud model (Node 6a).

Mirrors ``fraud_detect.ml.train_transaction`` so results are directly comparable
to the tabular GBM baseline: it reads the dataset straight from disk, honours the
dataset-provided split, trains on the train-split graph, early-stops on the
val-split graph, and reports PR-AUC + precision/recall@top-1% on the test-split
graph using the SAME ``report.imbalanced_report`` / ``print_report`` helpers.

Unlike the GBM (which scores single rows), this model does **edge-level**
classification on the split-safe INTERNAL-transfer graph built by
``graph_builder.build_graph``: each INTERNAL transfer is an edge, and the label is
``is_fraud``. After reporting the headline metrics it also prints
**recall-by-pattern** among the top-1% alerts so the relational typologies the GNN
is meant to catch — ``ROUND_TRIPPING`` and ``DORMANT_REACTIVATION`` — are visible
next to the others.

Usage::

    python -m fraud_detect.ml.gnn.train_gnn \
        --data ~/Downloads/transaction_fraud_dataset \
        --model gin --reverse_mp --ego --ports \
        --out models/txn_fraud_gnn.pt \
        --scores-out models/gnn_scores.joblib

Heavy-dependency boundary
-------------------------
``torch`` / ``torch_geometric`` are OFFLINE-ONLY deps (``requirements-gnn.txt``).
They are imported LAZILY inside the functions that need them (never at module
import time), so this module can be imported — and ``--help`` rendered — without
torch installed, matching the discipline used by ``graph_builder`` and the vendor
package. ``--scores-out`` is accepted here but the scores-store is written by
Task 5 (``produce the precomputed scores store``); this module stays focused on
training + reporting.
"""
from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ``store`` is pure stdlib + pydantic models (NO torch), so importing its
# fingerprint helpers at module top is safe and keeps the edge-key format
# IDENTICAL to the live cross-case memory path (``TxnFingerprint.key()``).
from ...store import _amount_band, normalize_counterparty
from ..report import imbalanced_report, print_report
from .graph_builder import (
    EDGE_FEATURE_COLUMNS,
    INTERNAL_COUNTERPARTY,
    SPLIT_COL,
    GraphBundle,
    build_graph,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime torch import
    import torch
    from torch_geometric.data import Data

# Patterns we explicitly surface in the recall-by-pattern breakdown. The GNN is
# motivated by the relational typologies the tabular GBM is weakest at.
HIGHLIGHT_PATTERNS = ("ROUND_TRIPPING", "DORMANT_REACTIVATION")

# Default model hyper-parameters — small, CPU-friendly, enough to be comparable
# to the GBM on this synthetic dataset without a tuning sweep.
DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 2
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_LR = 5e-3


# ---------------------------------------------------------------------------
# Per-edge pattern recovery (for recall-by-pattern reporting)
# ---------------------------------------------------------------------------
def edge_patterns_for_split(data_dir: str, split: str) -> np.ndarray:
    """Return the ``fraud_pattern`` label for each edge of ``split``'s graph.

    The order matches ``build_graph(data_dir, split).edge_index`` column order
    exactly: it replays the SAME edge selection the graph builder performs
    (INTERNAL transfers, in ``split``, both endpoints in ``split``) and reads the
    aligned ``fraud_pattern`` column. ``fraud_pattern`` is a LABEL — it is NEVER a
    model feature; it is used here only to attribute recall to each typology.
    """
    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    accts = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    return edge_patterns_from_frames(txns, accts, split)


def edge_patterns_from_frames(
    txns: pd.DataFrame, accts: pd.DataFrame, split: str
) -> np.ndarray:
    """Frame-level core of :func:`edge_patterns_for_split` (no disk IO).

    Replays the graph builder's edge filter so the returned ``fraud_pattern``
    array lines up 1:1 with the graph's edges.
    """
    node_in_split = set(
        accts.loc[accts[SPLIT_COL] == split, "account_id"].astype(str)
    )
    internal = txns[txns["counterparty_type"] == INTERNAL_COUNTERPARTY].copy()
    internal = internal[internal[SPLIT_COL] == split]
    internal["account_id"] = internal["account_id"].astype(str)
    internal["counterparty_id"] = internal["counterparty_id"].astype(str)
    both_in_split = (
        internal["account_id"].isin(node_in_split)
        & internal["counterparty_id"].isin(node_in_split)
    )
    internal = internal[both_in_split].reset_index(drop=True)
    if "fraud_pattern" in internal.columns:
        pat = internal["fraud_pattern"].astype("string").fillna("NA")
    else:
        pat = pd.Series(["NA"] * len(internal), dtype="string")
    return pat.to_numpy()


def recall_by_pattern(
    y_true: np.ndarray,
    scores: np.ndarray,
    patterns: np.ndarray,
    k_fraction: float = 0.01,
) -> List[Dict[str, Any]]:
    """Per-pattern recall among the top ``k_fraction`` highest-scoring edges.

    For each distinct fraud ``fraud_pattern`` present on the FRAUD edges, report
    how many of that pattern's fraud edges land in the alert budget (top-1%).
    This mirrors ``report.topk_precision_recall`` but attributes the catch to the
    typology, so ``ROUND_TRIPPING`` / ``DORMANT_REACTIVATION`` coverage is visible.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    patterns = np.asarray(patterns, dtype=object)
    n = len(scores)
    rows: List[Dict[str, Any]] = []
    if n == 0:
        return rows
    k = max(1, int(round(n * k_fraction)))
    top_idx = set(np.argsort(-scores)[:k].tolist())

    fraud_mask = y_true == 1
    # Patterns that actually appear on fraud edges (ignore the "NA" of clean ones).
    fraud_patterns = sorted(
        {
            str(p)
            for p, f in zip(patterns, fraud_mask)
            if f and str(p) not in ("NA", "nan", "<NA>", "")
        }
    )
    for pat in fraud_patterns:
        pat_fraud = [
            i for i in range(n) if fraud_mask[i] and str(patterns[i]) == pat
        ]
        total = len(pat_fraud)
        caught = sum(1 for i in pat_fraud if i in top_idx)
        rows.append(
            {
                "pattern": pat,
                "fraud_edges": total,
                "caught_in_top1pct": caught,
                "recall_at_top1pct": (caught / total) if total else float("nan"),
            }
        )
    return rows


def print_recall_by_pattern(rows: List[Dict[str, Any]], label: str = "TEST") -> None:
    """Pretty-print the recall-by-pattern table, highlighting the key typologies."""
    print(f"--- {label} recall-by-pattern @ top-1% ---")
    if not rows:
        print("    (no fraud edges with a labelled pattern in this split)")
        return
    for r in rows:
        star = " *" if r["pattern"] in HIGHLIGHT_PATTERNS else ""
        print(
            f"    {r['pattern']:24}: recall "
            f"{r['recall_at_top1pct']:.4f}  "
            f"({r['caught_in_top1pct']}/{r['fraud_edges']} fraud edges){star}"
        )
    # Make the highlighted typologies explicit even if absent from this split.
    present = {r["pattern"] for r in rows}
    for pat in HIGHLIGHT_PATTERNS:
        if pat not in present:
            print(f"    {pat:24}: (no fraud edges with this pattern in split)")
    print("    (* = relational typology the GNN is specifically meant to catch)")


# ---------------------------------------------------------------------------
# Graph preparation (torch / PyG — imported lazily)
# ---------------------------------------------------------------------------
def prepare_pyg_graph(
    bundle: GraphBundle,
    *,
    reverse_mp: bool,
    ego: bool,
    ports: bool,
) -> Tuple["Data", int, int]:
    """Turn a :class:`GraphBundle` into a PyG graph with the requested Multi-GNN
    adaptations applied (ports -> ego -> reverse-MP), returning the graph and the
    ``(num_node_features, num_edge_features)`` the model should be sized for.

    ``torch`` / ``torch_geometric`` are imported LAZILY here so importing this
    module needs no heavy deps.
    """
    import torch  # lazy, offline-only

    from .vendor.adaptations import AddEgoIds, GraphData, create_hetero_obj

    data = bundle.to_pyg_data()  # z-normalise the (encoded) features for stability
    x = _z_norm(data.x)
    edge_attr = _z_norm(data.edge_attr)
    edge_index = data.edge_index
    y = data.y
    timestamps = edge_attr[:, -1].clone() if edge_attr.shape[1] else None

    if reverse_mp:
        hetero = create_hetero_obj(x, y, edge_index, edge_attr, timestamps,
                                   ports=False)
        if ports:
            hetero.add_ports()
        if ego:
            hetero = AddEgoIds()(_with_edge_label_index_hetero(hetero))
        num_node_feats = hetero["node"].x.shape[1]
        num_edge_feats = hetero["node", "to", "node"].edge_attr.shape[1]
        return hetero, num_node_feats, num_edge_feats

    graph = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if ports:
        graph.add_ports()
    if ego:
        graph = AddEgoIds()(_with_edge_label_index(graph))
    num_node_feats = graph.x.shape[1]
    num_edge_feats = graph.edge_attr.shape[1]
    return graph, num_node_feats, num_edge_feats


def _with_edge_label_index(graph: "Data") -> "Data":
    """``AddEgoIds`` marks the centre nodes of the seed edges via
    ``edge_label_index``; for full-graph (transductive) training every edge is a
    seed edge, so set it to the full ``edge_index``."""
    graph.edge_label_index = graph.edge_index
    return graph


def _with_edge_label_index_hetero(hetero: "Data") -> "Data":
    hetero["node", "to", "node"].edge_label_index = \
        hetero["node", "to", "node"].edge_index
    return hetero


def _z_norm(t: "torch.Tensor") -> "torch.Tensor":
    """Column-wise z-normalisation; zero-variance columns pass through."""
    import torch  # lazy, offline-only

    if t.numel() == 0 or t.shape[0] == 0:
        return t
    mean = t.mean(0, keepdim=True)
    std = t.std(0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (t - mean) / std


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(
    model_name: str,
    num_node_feats: int,
    num_edge_feats: int,
    *,
    reverse_mp: bool,
    n_hidden: int = DEFAULT_HIDDEN,
    n_layers: int = DEFAULT_LAYERS,
    deg: Optional["torch.Tensor"] = None,
):
    """Instantiate a vendored Multi-GNN model. ``--model`` selects gin/gat/pna.

    When ``reverse_mp`` is set the homogeneous model is wrapped by PyG's
    ``to_hetero`` so it runs on the ('node','to','node') / ('node','rev_to',
    'node') relation graph produced by ``create_hetero_obj``.
    """
    import torch  # lazy, offline-only
    from torch_geometric.nn import to_hetero

    from .vendor import models as vm

    name = model_name.lower()
    if name == "gin":
        model = vm.GINe(
            num_features=num_node_feats, num_gnn_layers=n_layers,
            n_hidden=n_hidden, edge_updates=True, edge_dim=num_edge_feats,
        )
    elif name == "gat":
        model = vm.GATe(
            num_features=num_node_feats, num_gnn_layers=n_layers,
            n_hidden=n_hidden, edge_updates=True, edge_dim=num_edge_feats,
        )
    elif name == "pna":
        if deg is None:
            raise ValueError("PNA requires a degree histogram (deg) tensor")
        model = vm.PNA(
            num_features=num_node_feats, num_gnn_layers=n_layers,
            n_hidden=n_hidden, edge_updates=True, edge_dim=num_edge_feats,
            deg=deg,
        )
    else:
        raise ValueError(
            f"unknown --model {model_name!r}; choose one of gin/gat/pna"
        )

    if reverse_mp:
        metadata = (
            ["node"],
            [("node", "to", "node"), ("node", "rev_to", "node")],
        )
        model = to_hetero(model, metadata, aggr="mean")
    return model


def _pna_degree_histogram(edge_index: "torch.Tensor", num_nodes: int) -> "torch.Tensor":
    """In-degree histogram PNAConv needs for its scalers."""
    import torch  # lazy, offline-only
    from torch_geometric.utils import degree

    d = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    deg = torch.zeros(int(d.max()) + 1, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
    return deg


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _forward_edge_logits(model, graph, *, reverse_mp: bool) -> "torch.Tensor":
    """Run the model and return per-edge logits (n_edges, n_classes)."""
    if reverse_mp:
        x_dict = {"node": graph["node"].x}
        edge_index_dict = {
            ("node", "to", "node"): graph["node", "to", "node"].edge_index,
            ("node", "rev_to", "node"): graph["node", "rev_to", "node"].edge_index,
        }
        edge_attr_dict = {
            ("node", "to", "node"): graph["node", "to", "node"].edge_attr,
            ("node", "rev_to", "node"): graph["node", "rev_to", "node"].edge_attr,
        }
        out = model(x_dict, edge_index_dict, edge_attr_dict)
        # to_hetero returns the readout for the forward ('to') relation.
        return out if not isinstance(out, dict) else out[("node", "to", "node")]
    return model(graph.x, graph.edge_index, graph.edge_attr)


def _edge_scores(model, graph, *, reverse_mp: bool) -> np.ndarray:
    """Return the per-edge P(fraud) as a numpy array."""
    import torch  # lazy, offline-only

    model.eval()
    with torch.no_grad():
        logits = _forward_edge_logits(model, graph, reverse_mp=reverse_mp)
        probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    graphs: Dict[str, Any],
    labels: Dict[str, np.ndarray],
    model,
    *,
    reverse_mp: bool,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
) -> Dict[str, Any]:
    """Transductive full-graph training: train on the train graph, early-stop on
    the val graph's PR-AUC. Returns the best-val model state + per-epoch history.
    """
    import copy

    import torch
    import torch.nn.functional as F

    y_tr = torch.tensor(labels["train"], dtype=torch.long)
    # class weights: rare-positive task, upweight fraud like the GBM's balanced.
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    w = torch.tensor(
        [
            1.0,
            (n_neg / max(n_pos, 1)) if n_pos else 1.0,
        ],
        dtype=torch.float,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = _forward_edge_logits(model, graphs["train"], reverse_mp=reverse_mp)
        loss = F.cross_entropy(logits, y_tr, weight=w)
        loss.backward()
        opt.step()

        val_scores = _edge_scores(model, graphs["val"], reverse_mp=reverse_mp)
        val_rep = imbalanced_report(labels["val"], val_scores, "VAL")
        val_metric = val_rep["pr_auc"]
        val_metric = -1.0 if val_metric != val_metric else val_metric  # NaN guard
        history.append({"epoch": epoch, "loss": float(loss.item()),
                        "val_pr_auc": float(val_metric)})

        if val_metric > best_val:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(best val PR-AUC {best_val:.4f} @ epoch {best_epoch}).")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  "
                  f"val PR-AUC {val_metric:.4f}")

    model.load_state_dict(best_state)
    return {"best_val_pr_auc": best_val, "best_epoch": best_epoch,
            "history": history}


# ---------------------------------------------------------------------------
# Precomputed scores store (Task 5)
# ---------------------------------------------------------------------------
# Direction normalisation: the dataset encodes a transfer's direction as "DR"
# (debit) / "CR" (credit); the store/memory convention (``TxnFingerprint``) uses
# the lower-case words "debit" / "credit" / "na". Mapping here — rather than in
# the live path — keeps the OFFLINE edge key byte-for-byte identical to the key
# the live lookup (Tasks 6/7) reconstructs from parsed statement rows.
_DIRECTION_MAP = {
    "DR": "debit",
    "CR": "credit",
    "DEBIT": "debit",
    "CREDIT": "credit",
}


def normalize_direction(direction: Any) -> str:
    """Map a raw transfer direction to the store convention (credit|debit|na).

    Accepts the dataset's "DR"/"CR" as well as already-normalised
    "debit"/"credit"; anything unknown/missing collapses to "na" so the key is
    still well-formed.
    """
    if direction is None:
        return "na"
    key = str(direction).strip().upper()
    if not key or key in ("NAN", "<NA>", "NA", "NONE"):
        return "na"
    return _DIRECTION_MAP.get(key, "na")


def edge_key(counterparty: Any, amount: Any, direction: Any) -> str:
    """Build the edge-store key, mirroring ``TxnFingerprint.key()`` EXACTLY.

    ``TxnFingerprint.key()`` is ``f"{counterparty}|{amount_band}|{direction}"``
    where ``counterparty`` is ``store.normalize_counterparty(...)``,
    ``amount_band`` is ``store._amount_band(...)`` and ``direction`` is one of
    ``credit | debit | na``. We reuse the SAME ``store`` helpers so the offline
    edge keys and the live-lookup keys collapse to the same string for the same
    inputs — the whole point of the lookup being auditable and degrading
    gracefully.

    This function is PURE Python (no torch), so it is unit-testable offline.
    """
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        amt = None
    return "|".join(
        (
            normalize_counterparty(None if counterparty is None else str(counterparty)),
            _amount_band(amt),
            normalize_direction(direction),
        )
    )


def _edge_rows_for_split(data_dir: str, split: str) -> pd.DataFrame:
    """Replay the graph builder's edge selection to recover, per edge and in the
    SAME order as ``build_graph(...).edge_index``, the raw columns needed to
    rebuild store keys: source account, counterparty, amount, direction.

    Mirrors ``edge_patterns_from_frames`` / the builder's filter so row ``i``
    here lines up 1:1 with edge ``i``'s score from ``_edge_scores``.
    """
    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    accts = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    node_in_split = set(
        accts.loc[accts[SPLIT_COL] == split, "account_id"].astype(str)
    )
    internal = txns[txns["counterparty_type"] == INTERNAL_COUNTERPARTY].copy()
    internal = internal[internal[SPLIT_COL] == split]
    internal["account_id"] = internal["account_id"].astype(str)
    internal["counterparty_id"] = internal["counterparty_id"].astype(str)
    both_in_split = (
        internal["account_id"].isin(node_in_split)
        & internal["counterparty_id"].isin(node_in_split)
    )
    internal = internal[both_in_split].reset_index(drop=True)
    cols = {
        "account_id": internal["account_id"],
        "counterparty_id": internal["counterparty_id"],
        "amount_sgd": internal.get("amount_sgd"),
        "direction": internal.get("direction"),
    }
    return pd.DataFrame(cols)


def build_scores_store(
    data_dir: str,
    graphs: Dict[str, Any],
    model,
    *,
    reverse_mp: bool,
    model_name: str,
    pr_auc_test: float,
) -> Dict[str, Any]:
    """Run inference over ALL splits and assemble the precomputed scores store.

    For every split's graph we compute per-edge P(fraud) with ``_edge_scores``,
    then map each edge back to its raw transaction row (same order) to build:

    * ``edge_scores``  — ``{edge_key: score}`` keyed like ``TxnFingerprint.key()``
      (``counterparty|amount_band|direction``). When several edges collapse to
      the same key (the banding is intentionally coarse), we keep the MAX score,
      so the lookup reflects the riskiest matching transfer.
    * ``account_scores`` — ``{account_id: score}`` aggregated as the MAX edge
      score over the SOURCE account's outgoing edges, per the design ("max /
      aggregated edge risk per account").
    * ``meta`` — model name (+ enabled adaptations), ISO ``trained_at`` and the
      test PR-AUC.

    Returns the store dict (the caller persists it with joblib).
    """
    from datetime import datetime

    account_scores: Dict[str, float] = {}
    edge_scores: Dict[str, float] = {}

    for split in ("train", "val", "test"):
        graph = graphs[split]
        scores = _edge_scores(model, graph, reverse_mp=reverse_mp)
        rows = _edge_rows_for_split(data_dir, split)
        if len(rows) != len(scores):
            raise ValueError(
                f"{split}: edge-row/score misalignment "
                f"({len(rows)} rows vs {len(scores)} scores) — the scores store "
                f"would be keyed incorrectly"
            )
        for i in range(len(scores)):
            s = float(scores[i])
            src = str(rows["account_id"].iloc[i])
            cp = rows["counterparty_id"].iloc[i]
            amt = rows["amount_sgd"].iloc[i] if "amount_sgd" in rows else None
            direction = rows["direction"].iloc[i] if "direction" in rows else None

            k = edge_key(cp, amt, direction)
            if s > edge_scores.get(k, float("-inf")):
                edge_scores[k] = s

            # Aggregate per SOURCE account via max of its outgoing edge scores.
            if s > account_scores.get(src, float("-inf")):
                account_scores[src] = s

    return {
        "account_scores": account_scores,
        "edge_scores": edge_scores,
        "meta": {
            "model": model_name,
            "trained_at": datetime.utcnow().isoformat(),
            "pr_auc_test": float(pr_auc_test),
        },
    }


def model_name_with_adaptations(
    model: str, *, reverse_mp: bool, ego: bool, ports: bool
) -> str:
    """Human-readable model name reflecting enabled adaptations,
    e.g. ``"gin+reverse_mp+ego+ports"``."""
    parts = [model.lower()]
    if reverse_mp:
        parts.append("reverse_mp")
    if ego:
        parts.append("ego")
    if ports:
        parts.append("ports")
    return "+".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train the offline Multi-GNN transaction-fraud model "
        "(edge-level) and report PR-AUC + recall-by-pattern, comparable to the "
        "tabular GBM baseline.",
    )
    ap.add_argument("--data", required=True,
                    help="Path to transaction_fraud_dataset folder "
                         "(transactions.csv + accounts.csv)")
    ap.add_argument("--model", default="gin", choices=["gin", "gat", "pna"],
                    help="Which vendored Multi-GNN backbone to use")
    ap.add_argument("--reverse_mp", action="store_true",
                    help="Enable reverse message passing (hetero to/rev_to graph)")
    ap.add_argument("--ego", action="store_true",
                    help="Add ego IDs to the centre nodes of seed edges")
    ap.add_argument("--ports", action="store_true",
                    help="Add port numberings to the edge features")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                    help="Early-stopping patience on val PR-AUC")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--out", default="models/txn_fraud_gnn.pt",
                    help="Where to save the trained model checkpoint")
    ap.add_argument("--scores-out", dest="scores_out",
                    default="models/gnn_scores.joblib",
                    help="Where the precomputed scores store will be written "
                         "(produced by the scores-store step; this trainer "
                         "focuses on training + reporting)")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    # argparse first so ``--help`` works WITHOUT importing torch.
    args = build_arg_parser().parse_args(argv)

    # Heavy deps are imported only now (after argparse), so --help never needs them.
    import torch  # lazy, offline-only

    from pathlib import Path

    data_dir = os.path.expanduser(args.data)
    print(f"Building split-safe INTERNAL-transfer graphs from {data_dir} ...")
    bundles = {split: build_graph(data_dir, split)
               for split in ("train", "val", "test")}
    labels = {split: bundles[split].edge_labels for split in bundles}
    for split, b in bundles.items():
        rate = float(np.mean(b.edge_labels)) if b.num_edges else float("nan")
        print(f"  {split:5}: nodes={b.num_nodes:,}  edges={b.num_edges:,}  "
              f"fraud rate {rate:.4f}")

    cfg = dict(reverse_mp=args.reverse_mp, ego=args.ego, ports=args.ports)
    print(f"Preparing PyG graphs (model={args.model}, reverse_mp={args.reverse_mp}, "
          f"ego={args.ego}, ports={args.ports}) ...")
    graphs: Dict[str, Any] = {}
    feat_dims: Dict[str, Tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        g, nf, ef = prepare_pyg_graph(bundles[split], **cfg)
        graphs[split] = g
        feat_dims[split] = (nf, ef)
    num_node_feats, num_edge_feats = feat_dims["train"]

    deg = None
    if args.model == "pna":
        deg = _pna_degree_histogram(
            bundles["train"].to_pyg_data().edge_index,
            bundles["train"].num_nodes,
        )

    model = build_model(
        args.model, num_node_feats, num_edge_feats,
        reverse_mp=args.reverse_mp, n_hidden=args.hidden, n_layers=args.layers,
        deg=deg,
    )
    # to_hetero is lazily initialised on first forward; prime it so params exist.
    if args.reverse_mp:
        with torch.no_grad():
            _forward_edge_logits(model, graphs["train"], reverse_mp=True)

    print(f"Training {args.model.upper()} (node_feats={num_node_feats}, "
          f"edge_feats={num_edge_feats}) ...")
    train_info = train(
        graphs, labels, model,
        reverse_mp=args.reverse_mp, epochs=args.epochs,
        patience=args.patience, lr=args.lr,
    )

    # Report on val + test identically to train_transaction.py.
    val_scores = _edge_scores(model, graphs["val"], reverse_mp=args.reverse_mp)
    test_scores = _edge_scores(model, graphs["test"], reverse_mp=args.reverse_mp)
    val_rep = imbalanced_report(labels["val"], val_scores, "VAL")
    test_rep = imbalanced_report(labels["test"], test_scores, "TEST")
    print_report(val_rep)
    print_report(test_rep)

    # Recall-by-pattern on the test split so ROUND_TRIPPING /
    # DORMANT_REACTIVATION coverage is visible (fraud_pattern is a LABEL, never a
    # feature — recovered here only for attribution).
    test_patterns = edge_patterns_for_split(data_dir, "test")
    rbp = recall_by_pattern(labels["test"], test_scores, test_patterns)
    print_recall_by_pattern(rbp, label="TEST")

    print("(GBM baseline for reference: test PR-AUC ~0.90 from Stage 2 "
          "single-row + static KYC + per-account rolling history. This GNN is "
          "edge-level on the INTERNAL-transfer graph — a different, "
          "relational signal; see the offline eval write-up for the honest "
          "live account-level vs offline edge-level gap.)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "model": args.model,
                "reverse_mp": args.reverse_mp,
                "ego": args.ego,
                "ports": args.ports,
                "hidden": args.hidden,
                "layers": args.layers,
                "num_node_feats": num_node_feats,
                "num_edge_feats": num_edge_feats,
            },
            "metrics": {"val": val_rep, "test": test_rep,
                        "recall_by_pattern": rbp},
            "train_info": {"best_val_pr_auc": train_info["best_val_pr_auc"],
                           "best_epoch": train_info["best_epoch"]},
        },
        out,
    )
    print(f"Saved model -> {out}")

    # --- Scores store: inference over ALL splits -> precomputed lookup -----
    import joblib  # stdlib-light; already a core dependency via the ML path

    model_label = model_name_with_adaptations(
        args.model, reverse_mp=args.reverse_mp, ego=args.ego, ports=args.ports
    )
    print(f"Running inference over all splits to build the scores store "
          f"(model='{model_label}') ...")
    store = build_scores_store(
        data_dir, graphs, model,
        reverse_mp=args.reverse_mp,
        model_name=model_label,
        pr_auc_test=test_rep["pr_auc"],
    )
    scores_out = Path(args.scores_out)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(store, scores_out)
    print(f"Saved scores store -> {scores_out}  "
          f"({len(store['account_scores']):,} accounts, "
          f"{len(store['edge_scores']):,} edge keys, "
          f"test PR-AUC {store['meta']['pr_auc_test']:.4f})")


if __name__ == "__main__":
    main()
