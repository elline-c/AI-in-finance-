"""Graph-neural-network (GNN) extension for the transaction-fraud agent (Node 6a).

This subpackage adapts the IBM Multi-GNN anti-money-laundering models to capture
the *relational* AML typologies the tabular GBM is weakest at (``ROUND_TRIPPING``,
``DORMANT_REACTIVATION``, ``MULE_FAN_IN_OUT``). It does NOT replace the existing
``HistGradientBoostingClassifier`` — the GNN is a second, independent scorer whose
output is reconciled into Node 6a's findings. Governance is preserved: the GNN
only adds explainable evidence and NEVER auto-rejects.

Offline-batch design
--------------------
A GNN needs the whole transfer **graph** (accounts + edges) to compute
embeddings, but the live screening path sees a single statement in isolation with
no stable internal counterparty IDs — there is no graph to run inference over at
request time. Consequently the GNN runs **offline, in batch**, on the full
transaction graph; its per-account / per-edge risk signal is **precomputed** and
persisted to a lookup store (``models/gnn_scores.joblib``). At screening time
Node 6a reads the precomputed signal by key instead of running the model.

Module layout and the heavy-dependency boundary
------------------------------------------------
    * ``graph_builder``  — offline. Builds a split-safe PyTorch Geometric graph.
    * ``vendor``         — offline. Minimal vendored Multi-GNN model classes
      (Apache-2.0; see ``vendor/LICENSE`` and ``vendor/PROVENANCE.md``).
    * ``train_gnn``      — offline. Trains the model and writes the scores store.
    * ``gnn_lookup``     — LIVE. The ONLY module the live pipeline imports. It
      loads the precomputed store with ``joblib`` and performs an O(1) lookup;
      it imports NO PyTorch / PyTorch-Geometric.

The heavy training dependencies (``torch``, ``torch-geometric``) are isolated to
``requirements-gnn.txt`` and imported lazily inside the offline modules only, so
the live screening path stays lightweight and auditable.
"""
