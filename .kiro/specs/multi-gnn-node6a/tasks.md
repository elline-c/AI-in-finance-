# Tasks — Multi-GNN Integration into Node 6a

Implementation plan. Each task is incremental, ends in something runnable or
testable, and references the design component it builds. Governance holds
throughout: the GNN only adds explainable evidence and never auto-rejects.

- [x] 1. Scaffold the GNN subpackage and isolate heavy dependencies
  - Create `fraud_detect/ml/gnn/__init__.py` with a module docstring stating the
    offline-batch design and that the live path imports only `gnn_lookup`.
  - Add `requirements-gnn.txt` (`torch`, `torch-geometric`) separate from the core
    `requirements.txt`; add a note in the README that these are offline-only.
  - Create `fraud_detect/ml/gnn/vendor/` with upstream `LICENSE` (Apache-2.0) and
    `PROVENANCE.md` recording the Multi-GNN source URL and commit hash.
  - _Design: Components §1, §2 (vendoring), Dependencies_

- [x] 2. Build the split-safe internal transfer graph
  - Implement `fraud_detect/ml/gnn/graph_builder.py::build_graph(data_dir, split)`.
  - Nodes = accounts; node features = the same safe KYC columns whitelisted in
    `transaction_features.load_joined` (never `is_suspicious_account` /
    `account_fraud_pattern`).
  - Edges = `counterparty_type == "INTERNAL"` transfers only; edge features =
    amount, direction, channel, timestamp; edge label = `is_fraud`.
  - Enforce leakage guards: `fraud_pattern`, `ring_id`, `split`, `eval_tags` never
    enter features; build one graph per split (no cross-split edges).
  - _Design: Components §1; Governance; Open Questions #1_

- [x] 2.1 Tests for the graph builder
  - Assert no edge connects nodes from different splits.
  - Assert no leaky column appears in node or edge feature tensors.
  - Assert only INTERNAL transfers become edges.
  - _Design: Testing strategy (graph-builder tests)_

- [x] 3. Vendor the minimal Multi-GNN model classes
  - Copy only the needed model classes (GIN/GAT/PNA and the reverse-MP / ego /
    ports adaptations) into `fraud_detect/ml/gnn/vendor/`, adjusting imports to the
    local package; keep attribution in `PROVENANCE.md`.
  - Do NOT vendor the upstream data pipeline — we feed graphs from `graph_builder`.
  - _Design: Components §2_

- [x] 4. Offline trainer with comparable reporting
  - Implement `fraud_detect/ml/gnn/train_gnn.py` with a CLI mirroring
    `train_transaction.py` (`--data`, `--model`, `--reverse_mp/--ego/--ports`,
    `--out`, `--scores-out`).
  - Train on the train-split graph, early-stop on val, report on test using
    `report.imbalanced_report` / `print_report` (same PR-AUC + top-1% metrics).
  - Print recall-by-pattern so `ROUND_TRIPPING` / `DORMANT_REACTIVATION` are visible.
  - _Design: Components §2; Testing strategy (offline eval)_

- [x] 5. Produce and persist the precomputed scores store
  - After training, run inference over all splits to produce per-edge and
    aggregated per-account fraud scores.
  - Write `models/gnn_scores.joblib` with `account_scores`, `edge_scores` (keyed
    like `TxnFingerprint.key()`: counterparty|amount_band|direction), and `meta`
    (model name, trained_at, test PR-AUC).
  - _Design: Components §3_

- [x] 6. Live-path lookup adapter (no PyTorch)
  - Implement `fraud_detect/ml/gnn/gnn_lookup.py` with `load_scores(path)` and
    `score_account(scores, account_id)`; import only `joblib`.
  - Fail-safe: missing file / unknown key / corrupt store → `None`.
  - _Design: Components §4; Error handling_

- [x] 6.1 Fail-safe tests for the lookup adapter
  - Missing file, unknown key, and corrupt store each return `None`.
  - _Design: Testing strategy (lookup fail-safe tests)_

- [x] 7. Wire the GNN finding into Node 6a
  - Extend `fraud_detect/nodes/ml_transaction.py::analyze` with an optional
    `gnn_scores` arg; keep the existing GBM path unchanged.
  - When the account lookup is elevated, append one `ML_TXN_GNN_RISK` finding
    (HIGH ≥ 0.80, MEDIUM ≥ 0.50) tagged with agent sub-label
    `Node6a:MLTransaction:GNN`, with evidence citing the graph score and stating
    it is reviewer evidence, never an auto-rejection.
  - _Design: Components §5; Governance_

- [x] 7.1 Node 6a tests with a stub scores store

  - Elevated account → exactly one `ML_TXN_GNN_RISK` finding at the right severity.
  - No store → current behavior unchanged.
  - Node never produces an auto-reject.
  - _Design: Testing strategy (Node 6a tests)_

- [x] 8. Thread the store through the pipeline
  - Add an optional `gnn_scores=None` parameter to `run_pipeline`; pass it to
    `ml_transaction.analyze`. When `None`, behavior is identical to today.
  - Load the store once in `webapp.py` / `demo.py` (fail-safe if absent).
  - _Design: Components §6_

- [x] 9. Offline evaluation write-up vs the GBM baseline
  - Record GNN test PR-AUC, precision/recall@top-1%, and recall-by-pattern next to
    the current ~0.90 GBM baseline; note the live account-level vs offline
    edge-level gap honestly (Open Question #1) and the synthetic upper-bound caveat.
  - _Design: Testing strategy (offline eval); Open Questions #1, #2_

- [x] 10. Documentation
  - Update `FRAUD_DETECT_README.md`: new Node 6a GNN signal, offline-vs-live split,
    `requirements-gnn.txt`, how to retrain and regenerate the scores store, and the
    governance stance (adds evidence, never auto-rejects).
  - _Design: Overview; Governance_
```
