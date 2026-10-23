"""Live-path lookup adapter for the precomputed GNN scores store (Node 6a).

This is the ONLY GNN module the live screening pipeline imports. It loads the
offline-produced scores store (``models/gnn_scores.joblib``) and performs an O(1)
lookup of the precomputed per-account (and, optionally, per-edge) graph-fraud
score. It imports only ``joblib`` plus stdlib/typing — NO ``torch`` /
``torch_geometric`` / ``pandas`` — so the live path stays lightweight and
auditable (see Design: Components §4).

Fail-safe contract (mirrors the existing model loaders, e.g.
``ml_transaction.load_model``): any failure — a missing file, an unreadable or
corrupt store, a store missing the expected keys / of the wrong type, or an
unknown lookup key — yields ``None``. The caller then simply emits no finding;
the GBM path and the rest of the pipeline are unaffected. The GNN never fabricates
a score and never auto-rejects.

The store format (produced by Task 5)::

    {
        "account_scores": { account_id: float, ... },
        "edge_scores":    { edge_key: float, ... },   # keyed like TxnFingerprint.key()
        "meta":           { "model": ..., "trained_at": ..., "pr_auc_test": ... },
    }
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# The live path depends ONLY on joblib (already a core dependency). No torch.
import joblib

# The serialized store is a plain dict; typing it as such keeps the live path
# free of any GNN/torch types.
GnnScores = Dict[str, Any]

DEFAULT_SCORES_PATH = "models/gnn_scores.joblib"


def load_scores(path: str = DEFAULT_SCORES_PATH) -> Optional[GnnScores]:
    """Load the precomputed GNN scores store, or ``None`` on any failure.

    A well-formed store is a ``dict`` carrying ``account_scores``, ``edge_scores``
    and ``meta``. A missing file, an unreadable / corrupt pickle, or a loaded
    object that is not a dict with the expected keys all fail safe to ``None`` —
    matching the fail-safe contract of the existing model loaders.
    """
    try:
        store = joblib.load(path)
    except Exception:
        # Missing file, corrupt / non-joblib file, permission error, etc.
        return None

    # Guard against a store of the wrong shape (e.g. a bare list, or a dict
    # missing the required sections). A malformed store is treated as absent.
    if not isinstance(store, dict):
        return None
    if not isinstance(store.get("account_scores"), dict):
        return None

    return store


def score_account(scores: Optional[GnnScores], account_id: str) -> Optional[float]:
    """Return the precomputed GNN fraud score for ``account_id``, or ``None``.

    ``None`` is returned for a missing/malformed store, an unknown account key, or
    a value that cannot be read as a float. The lookup never raises.
    """
    try:
        table = scores["account_scores"]  # type: ignore[index]
        value = table[account_id]
        if value is None:
            return None
        return float(value)
    except Exception:
        # None store, wrong type, unknown key, non-numeric value -> fail safe.
        return None


def score_edge(scores: Optional[GnnScores], edge_key: str) -> Optional[float]:
    """Return the precomputed GNN fraud score for an edge key, or ``None``.

    Optional convenience mirroring :func:`score_account`. ``edge_key`` follows the
    ``TxnFingerprint.key()`` convention (``counterparty|amount_band|direction``).
    Same fail-safe behaviour: unknown key / malformed store -> ``None``.
    """
    try:
        table = scores["edge_scores"]  # type: ignore[index]
        value = table[edge_key]
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
