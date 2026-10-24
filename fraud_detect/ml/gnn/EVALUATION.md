# Offline evaluation — Multi-GNN (Node 6a) vs the GBM baseline

> **Status: template / reproduction report, not measured results.**
> Every metric cell below marked `TBD` is a **placeholder** to be filled from an
> actual `train_gnn` run. `torch` / `torch-geometric` are offline-only
> dependencies (`requirements-gnn.txt`) and are not installed in every
> environment, and the dataset (`~/Downloads/transaction_fraud_dataset`) may not
> be present. **No number in this document has been measured** — do not cite any
> cell as a result until it has been replaced with output from the commands in
> the next section.

This write-up compares the offline, edge-level Multi-GNN model to the current
tabular gradient-boosted (GBM) transaction-fraud model. It is the "offline eval"
deliverable from the design's **Testing strategy** section, and it states the
honest limitations from **Open Questions #1 and #2**.

The two models are **not** measuring the same thing:

- The **GBM** (`fraud_detect.ml.train_transaction`) scores **single statement
  rows** with static KYC + per-account rolling-window history. Its current best
  offline result is **test PR-AUC ~0.90** (per the design Overview and the
  reference note printed by `train_transaction.py`).
- The **GNN** (`fraud_detect.ml.gnn.train_gnn`) does **edge-level**
  classification on the split-safe INTERNAL-transfer graph: each INTERNAL
  transfer is an edge and the label is `is_fraud`. It targets the relational AML
  typologies the GBM is weakest at — `ROUND_TRIPPING` and
  `DORMANT_REACTIVATION` (and `MULE_FAN_IN_OUT`).

Because they operate on different units (rows vs graph edges), the headline
numbers are **directional context, not a like-for-like leaderboard**. Read the
recall-by-pattern table and the caveats section alongside the headline PR-AUC.

---

## How to reproduce these numbers

Both trainers read the dataset straight from disk, honour the dataset-provided
split, train on the train split, early-stop/validate on the val split, and
report on the **test** split using the same `report.imbalanced_report` /
`print_report` helpers — so PR-AUC and precision/recall@top-1% are computed
identically for both models.

### GBM baseline

```bash
python -m fraud_detect.ml.train_transaction \
    --data ~/Downloads/transaction_fraud_dataset \
    --out models/txn_fraud_clf.joblib
```

Prints `VAL` and `TEST` reports and the reference line:
`single-row PR-AUC ~0.57; +account history ~0.84` (Stage 2 here is single-row +
static KYC + per-account rolling history, i.e. ~0.90).

### GNN (edge-level)

```bash
python -m fraud_detect.ml.gnn.train_gnn \
    --data ~/Downloads/transaction_fraud_dataset \
    --model gin --reverse_mp --ego --ports \
    --out models/txn_fraud_gnn.pt \
    --scores-out models/gnn_scores.joblib
```

Real `train_gnn.py` CLI flags used above:

| Flag | Meaning |
|---|---|
| `--data` | Path to the `transaction_fraud_dataset` folder (`transactions.csv` + `accounts.csv`). Required. |
| `--model` | Vendored Multi-GNN backbone: `gin` (default), `gat`, or `pna`. |
| `--reverse_mp` | Enable reverse message passing (hetero `to` / `rev_to` graph). |
| `--ego` | Add ego IDs to the centre nodes of the seed edges. |
| `--ports` | Add port numberings to the edge features. |
| `--out` | Where to save the trained model checkpoint (default `models/txn_fraud_gnn.pt`). |
| `--scores-out` | Where the precomputed scores store is written (default `models/gnn_scores.joblib`). |

(Additional tuning flags also exist: `--epochs`, `--patience`, `--lr`,
`--hidden`, `--layers`. The command above uses their defaults.)

`train_gnn.py` prints a `TEST` report (via `print_report`) and then a
recall-by-pattern breakdown (via `print_recall_by_pattern`). Copy the printed
values into the tables below, replacing each `TBD`.

---

## Headline results (fill from the runs above)

Metric names match `report.imbalanced_report` keys exactly, so each row maps
directly to a line in the printed `TEST` report.

| Metric (test split) | GBM baseline | GNN (`gin+reverse_mp+ego+ports`) |
|---|---|---|
| PR-AUC (`pr_auc`) | ~0.90 (reference; re-run to confirm exact value) | TBD — fill from `train_gnn` TEST report |
| ROC-AUC (`roc_auc`) | TBD — fill from `train_transaction` TEST report | TBD — fill from `train_gnn` TEST report |
| Precision @ top-1% (`precision_at_top1pct`) | TBD — fill from `train_transaction` TEST report | TBD — fill from `train_gnn` TEST report |
| Recall @ top-1% (`recall_at_top1pct`) | TBD — fill from `train_transaction` TEST report | TBD — fill from `train_gnn` TEST report |
| Alerts @ top-1% (`alerts_at_top1pct`, the queue size `k`) | TBD — fill from `train_transaction` TEST report | TBD — fill from `train_gnn` TEST report |

GBM reference points (from the dataset's own baseline, printed by
`train_transaction.py`):

- single-row only: **PR-AUC ~0.57**
- single-row + account history: **PR-AUC ~0.84**
- Stage 2 (single-row + static KYC + per-account rolling history): **~0.90**

> Note: the GBM alert budget (top-1% of **rows**) and the GNN alert budget
> (top-1% of graph **edges**) are different populations, so compare the two
> columns with the unit difference in mind.

---

## Recall by pattern @ top-1% (fill from the GNN run)

`train_gnn.py` prints this table via `print_recall_by_pattern`: for each fraud
`fraud_pattern` present on the fraud edges, it reports how many of that pattern's
fraud edges land in the top-1% alert budget. `fraud_pattern` is a **label**, used
here only for attribution — it is never a model feature. The two starred rows
are the relational typologies the GNN is specifically meant to catch
(`HIGHLIGHT_PATTERNS` in `train_gnn.py`).

| Fraud pattern | Recall @ top-1% | Caught / fraud edges |
|---|---|---|
| `ROUND_TRIPPING` * | TBD — fill from `print_recall_by_pattern` | TBD |
| `DORMANT_REACTIVATION` * | TBD — fill from `print_recall_by_pattern` | TBD |
| `MULE_FAN_IN_OUT` | TBD — fill from `print_recall_by_pattern` | TBD |
| (other patterns printed by the run) | TBD — fill from `print_recall_by_pattern` | TBD |

`*` = relational typology the GNN is specifically meant to catch.

If a highlighted pattern has no fraud edges in the test split,
`print_recall_by_pattern` prints an explicit
`(no fraud edges with this pattern in split)` line instead of a recall value —
record that as-is rather than inventing a number.

For the GBM, recall-by-pattern is not emitted by `train_transaction.py`; it is
reported here for the GNN only, which is the point of comparison (the GBM is the
single-row baseline these relational patterns are hard for).

---

## Honest caveats

These are the design's **Open Questions #1 and #2**, stated plainly as required
by the Testing strategy. They bound how the offline numbers above should be read.

### 1. Live account-level vs offline edge-level gap (Open Question #1)

The numbers above are **offline, edge-level** results. The live screening path
cannot reproduce them:

- The live path (Node 6a) only has `ing.uen` — an **account-level** key — to look
  up a precomputed score.
- Parsed statement rows do **not** carry stable internal counterparty IDs, so
  there is no edge to look up and no graph to run inference over at screening
  time.

Consequently the live GNN contribution is **coarse** (account-level lookup),
while the full **edge-level** benefit measured in this write-up is realized only
in **offline batch scoring**. This caps the realistic **live** lift: the headline
and recall-by-pattern numbers here are an upper bound on what the live path can
contribute, not a promise of live performance. Report the offline numbers as
offline numbers.

### 2. Synthetic upper-bound caveat (Open Question #2)

The dataset is synthetic, and per its data card the reported scores are an
**upper bound**. The graph lift — the margin the GNN adds over the tabular
baseline on the relational typologies — **may shrink on real data**. Treat any
improvement shown in the tables as the optimistic end of the range, not the
expected production delta.

---

## Checklist before citing this document as results

- [ ] Ran `train_transaction` and recorded the GBM `TEST` report values.
- [ ] Ran `train_gnn` with the flags above and recorded the GNN `TEST` report values.
- [ ] Filled every `TBD` cell in the headline and recall-by-pattern tables.
- [ ] Recorded `print_recall_by_pattern` output verbatim (including any
      "no fraud edges with this pattern in split" lines).
- [ ] Removed the "template / not measured results" status banner once real
      numbers are in, and noted the dataset snapshot / commit used.
