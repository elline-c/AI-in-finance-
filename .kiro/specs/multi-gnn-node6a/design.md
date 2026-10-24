# Design — Multi-GNN Integration into Node 6a (Transaction-Fraud Agent)

## Overview

This design integrates a graph neural network (GNN) based anti-money-laundering
model — adapted from [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN) (Apache-2.0)
— into the transaction-fraud detection path (Node 6a). The goal is to capture the
relational AML typologies the current single-row + rolling-window model is weakest
at: `ROUND_TRIPPING` and `DORMANT_REACTIVATION`, plus `MULE_FAN_IN_OUT`. The
dataset data card states explicitly that graph features (cycles, shared
counterparties) are where a model beats the rolling-window baseline.

The current Node 6a (`fraud_detect/nodes/ml_transaction.py`) scores each parsed
statement row with a scikit-learn `HistGradientBoostingClassifier` and emits an
explainable `Finding` for rows above a probability threshold. The current best
offline result is test PR-AUC ~0.90 from Stage 2 account-history features.

This design does **not** replace that model. The GNN is added as a second,
independent scorer whose output is reconciled into Node 6a's findings. The
rule-based nodes, the fail-safe contract, and the "adds evidence, never
auto-rejects" governance stance are all preserved.

### The central architectural constraint

A GNN needs the whole transaction **graph** (accounts and their transfer edges) to
compute node/edge embeddings. The live screening path (`run_pipeline` → Node 6a)
sees exactly one statement in isolation, and parsed statement rows do **not** carry
stable internal counterparty IDs — so there is no graph to run inference over at
screening time. This is the single most important fact shaping the design.

Consequence: the GNN runs as an **offline, batch model** trained on the full
`transactions.csv` graph. Its per-account/per-transaction risk signal is
**precomputed** and persisted to a lookup store. At screening time, Node 6a reads
the precomputed signal by key (account / counterparty / amount band) instead of
running the GNN. This keeps the live pipeline lightweight (no PyTorch at request
time) while still delivering graph-derived evidence.

### Scope

In scope:
- Offline GNN training on the dataset's internal transfer graph.
- A precomputed-score store and a lookup adapter for Node 6a.
- Node 6a emitting GNN-derived `Finding`s alongside the existing GBM findings.
- Honest evaluation vs the current 0.90 baseline, split-safe.

Out of scope (explicitly deferred):
- Live, per-request GNN inference (no graph available at screening time).
- DocTamper / Node 6b (separate spec, blocked on PDF→image + licensing).
- Real-time embedding updates / streaming graph.

## Architecture

```
OFFLINE (batch, has the full graph)                 LIVE (per-statement, no graph)
────────────────────────────────────                ──────────────────────────────
transactions.csv ─┐                                  run_pipeline(pdf)
accounts.csv ─────┤                                       │
                  ▼                                        ▼
         graph_builder.py                            Node 1 Ingestion
   (internal transfer graph:                              │  parsed Transaction rows
    accounts = nodes,                                     ▼
    INTERNAL transfers = edges)                     Node 6a ml_transaction.analyze()
                  │                                        │
                  ▼                                        ├─ existing GBM path (unchanged)
         train_gnn.py (PyTorch Geometric)                  │
   GIN/GAT/PNA + reverse MP, ego, ports                    └─ GNN lookup path:
                  │                                            gnn_lookup.score(account,
                  ▼                                               counterparty, amount)
      per-node / per-edge fraud scores                          │   reads precomputed store
                  │                                             ▼
                  ▼                                        reconcile → Finding(s)
      gnn_scores store (joblib/parquet)  ───────────────►  (ML_TXN_GNN_RISK)
      keyed by account_id / edge key
```

### Why offline-batch, not live

| Option | Live GNN inference | Offline batch + precomputed lookup (chosen) |
|---|---|---|
| Graph at inference | Not available — one statement, no internal IDs | Full graph available at train time |
| Request-time deps | PyTorch + PyG in the serving path | None — a simple store lookup |
| Latency | Graph assembly + forward pass per request | O(1) dict lookup |
| Freshness | Real-time | As fresh as the last batch run (acceptable: screening is not HFT) |
| Fail-safe | Hard — model/graph can fail mid-request | Easy — missing key → no finding (matches current node) |

The offline approach is the only one compatible with the existing input pipeline
and the fail-safe contract. If a future phase adds an internal graph store keyed by
real counterparty IDs, live inference can be revisited — the lookup interface below
is designed so that swap is possible without touching Node 6a's logic.

## Components and interfaces

### 1. `fraud_detect/ml/gnn/graph_builder.py` (new)

Builds a PyTorch Geometric graph from the dataset, split-safe.

```python
def build_graph(data_dir: str, split: str) -> "GraphBundle":
    """Return nodes (accounts), edges (INTERNAL transfers only), node features,
    edge features, edge timestamps, and the is_fraud edge labels — for ONE split.
    Internal transfers never cross splits (data card guarantee), so a per-split
    graph has no leakage across train/val/test."""
```

Key decisions:
- **Nodes = accounts.** Node features reuse the safe static KYC attributes already
  whitelisted in `transaction_features.load_joined` (segment, age_band,
  kyc_risk_rating, declared income/turnover, account_age) — never label-derived
  columns (`is_suspicious_account`, `account_fraud_pattern`).
- **Edges = `counterparty_type == "INTERNAL"` transfers only.** External
  counterparties are randomized hashes per the data card, so they carry no graph
  signal; including them would only add noise. Edge features: amount, direction,
  channel, timestamp. Edge label: `is_fraud`.
- **Leakage guards carried over verbatim** from `transaction_features`:
  `fraud_pattern`, `ring_id`, `split`, `eval_tags` never become features.
- **Split-safe graph:** build one graph per split; do not connect nodes across
  splits. This matches the data card ("internal transfers never cross splits").

### 2. `fraud_detect/ml/gnn/train_gnn.py` (new)

Adapts Multi-GNN's model classes (GIN/GAT/PNA/RGCN) and adaptations (reverse
message passing, ego IDs, port numberings). Mirrors `train_transaction.py`'s CLI
and reporting so results are comparable.

```python
# python -m fraud_detect.ml.gnn.train_gnn \
#     --data ~/Downloads/transaction_fraud_dataset \
#     --model gin --reverse_mp --ego --ports \
#     --out models/txn_fraud_gnn.pt \
#     --scores-out models/gnn_scores.joblib
```

- Reuses `report.imbalanced_report` / `print_report` so PR-AUC and
  precision/recall@top-1% are reported identically to the GBM.
- Trains on the train-split graph, early-stops on val, reports on test.
- After training, runs inference over **all** splits to produce per-edge (and
  aggregated per-account) fraud scores, written to the scores store.
- Vendored Multi-GNN code lives under `fraud_detect/ml/gnn/vendor/` with the
  upstream `LICENSE` (Apache-2.0) and a `PROVENANCE.md` noting commit + source.

### 3. `fraud_detect/ml/gnn/gnn_scores` store (new artifact)

A serialized lookup (joblib dict or parquet) produced offline:

```
{
  "account_scores": { "ACC000123": 0.87, ... },   # max/aggregated edge risk per account
  "edge_scores":    { "ACC000123|ACC000456|CR|140000-150000": 0.91, ... },
  "meta": { "model": "gin+reverse_mp+ego+ports", "trained_at": ..., "pr_auc_test": ... }
}
```

Keying mirrors the existing `TxnFingerprint.key()` convention (counterparty +
amount band + direction) so the lookup degrades gracefully and is auditable.

### 4. `fraud_detect/ml/gnn/gnn_lookup.py` (new)

The only GNN component the live pipeline imports. No PyTorch dependency — pure
stdlib + joblib load.

```python
def load_scores(path: str = "models/gnn_scores.joblib") -> Optional["GnnScores"]:
    """Load the precomputed store, or None (fail-safe)."""

def score_account(scores, account_id: str) -> Optional[float]:
    """Return the precomputed GNN fraud score for an account, or None."""
```

Fail-safe exactly like the current model loaders: any failure → `None` → no finding.

### 5. `fraud_detect/nodes/ml_transaction.py` (modified)

`analyze()` gains an optional precomputed-scores argument and emits a new,
distinctly-coded finding when the GNN signal is elevated. The existing GBM path is
untouched.

```python
def analyze(ing: IngestionResult, model=None, gnn_scores=None) -> AgentReport:
    # ... existing GBM scoring produces ML_TXN_FRAUD_LIKELIHOOD findings ...
    # NEW: if gnn_scores is available, look up the account (ing.uen as the key the
    # live path has) and, when elevated, append an ML_TXN_GNN_RISK finding citing
    # the graph-derived score. Fail-safe: no store / no key -> no finding.
```

New reason code: `ML_TXN_GNN_RISK` (HIGH ≥ 0.80, MEDIUM ≥ 0.50), with evidence
that names the graph signal and states it is evidence for a reviewer, never an
automatic rejection — identical governance wording to the existing node.

Reconciliation rule: the GBM and GNN findings are **independent evidence** and both
flow into Node 4. Because Node 4 already gives a corroboration bonus when ≥2
distinct agents raise HIGH findings, GBM+GNN agreement naturally strengthens the
risk call without any special-casing. We will tag the GNN finding with a distinct
`agent` sub-label (`Node6a:MLTransaction:GNN`) so corroboration counts it as a
distinct signal while keeping it within Node 6a.

### 6. `fraud_detect/pipeline.py` (modified)

`run_pipeline` gains an optional `gnn_scores=None` parameter, loaded once by the
caller (webapp/demo) and threaded into `ml_transaction.analyze`. When `None`, the
pipeline behaves exactly as today.

## Data model

No change to `models.py`. The GNN path reuses the existing `Finding` and
`AgentReport` contracts. The scores store is an ML artifact, not a pipeline model.

## Dependencies

New (offline/training only — NOT required to run the live pipeline):
- `torch` (CPU build acceptable for this dataset size)
- `torch-geometric`

These are isolated to `fraud_detect/ml/gnn/` and imported lazily inside
`train_gnn.py` / `graph_builder.py`. To keep the base install light, they go in a
separate `requirements-gnn.txt` rather than the core `requirements.txt`. The live
path (`gnn_lookup.py`) imports only `joblib`, which is already a dependency.

## Error handling / fail-safe

Every new live-path entry point follows the node's existing contract:
- Missing scores file, unreadable store, unknown key, or malformed data → the
  lookup returns `None` and Node 6a emits no GNN finding. The GBM path and the rest
  of the pipeline are unaffected.
- The offline trainer validates the graph (no cross-split edges, no leaky columns)
  and fails loudly if a leakage guard is violated — offline failure is acceptable
  and preferable to silently shipping a leaky model.

## Testing strategy

- **Graph-builder tests:** no edge crosses a split; no leaky column present in node
  or edge features; INTERNAL-only edges.
- **Lookup fail-safe tests:** missing file / unknown key / corrupt store all yield
  `None` and therefore no finding (mirrors existing `test_detection.py` style).
- **Node 6a tests:** with a stub scores store, an elevated account yields exactly
  one `ML_TXN_GNN_RISK` finding at the right severity; an absent store yields the
  current behavior unchanged; the node never produces an auto-reject.
- **Offline eval (not a unit test):** report GNN test PR-AUC and recall-by-pattern
  vs the 0.90 GBM baseline, with specific attention to `ROUND_TRIPPING` and
  `DORMANT_REACTIVATION`.

## Open questions / risks

1. **Live-path key fidelity.** The live path only has `ing.uen` as an account key,
   and parsed rows lack internal counterparty IDs. So the live GNN contribution is
   coarse (account-level lookup), not edge-level. The full edge-level benefit is
   realized only in offline batch scoring. This caps the realistic live lift and
   should be stated plainly in the evaluation write-up.
2. **Synthetic upper bound.** Per the data card, scores are an upper bound; the
   graph lift may shrink on real data.
3. **Dependency weight.** PyTorch + PyG is a large addition; keeping it offline-only
   and behind a separate requirements file is the mitigation.
4. **Vendoring vs dependency.** Multi-GNN is a research repo, not a published
   package. Design assumes vendoring the needed model classes under `vendor/` with
   its Apache-2.0 license retained, rather than a pip dependency.

## Governance (MAS FEAT) alignment

- The GNN only **adds explainable evidence** and never auto-rejects — identical to
  every other node.
- Every GNN finding carries a stable reason code (`ML_TXN_GNN_RISK`) and
  human-readable evidence citing the graph-derived score.
- Leakage guards from the data card are enforced in the graph builder, not just the
  tabular feature builder.
- Offline-only heavy dependencies keep the auditable live path simple and
  inspectable.
