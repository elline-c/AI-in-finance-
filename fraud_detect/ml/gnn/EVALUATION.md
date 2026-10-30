# Offline evaluation — Multi-GNN (Node 6a) vs the GBM baseline

> **Status: measured, with caveats.** Every number below comes from a real run on
> the project dataset (`~/Downloads/transaction_fraud_dataset`): `torch` 2.15.0,
> `torch_geometric` 2.8.1, Python 3.13.6, macOS arm64 (CPU); scikit-learn 1.9.1
> for the GBM. Each GNN configuration was trained **once**, and `train_gnn.py`
> sets **no random seed**, so a re-run can give different numbers - especially on
> a graph this small (336 test edges). Treat the results as one draw, not a
> distribution. The fraud counts are tiny (27 fraud accounts in the test split),
> so read counts and intervals, not just percentages.

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
  transfer is an edge and the label is `is_fraud`. It was designed to target the
  relational AML typologies the GBM is weakest at — `ROUND_TRIPPING`,
  `DORMANT_REACTIVATION` and `MULE_FAN_IN_OUT`. **Measured: the graph as built can
  only see `ROUND_TRIPPING`.** Only INTERNAL transfers become edges, and in this
  dataset all 322 INTERNAL fraud transactions are round-tripping; dormant
  reactivation, mule fan-in/out, structuring and rapid pass-through ride on
  external counterparties and never enter the graph.

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
recall-by-pattern breakdown (via `print_recall_by_pattern`). The tables below
were filled from these printed values.

---

## Headline results (measured)

Metric names match `report.imbalanced_report` keys exactly, so each row maps
directly to a line in the printed `TEST` report.

| Metric (test split) | GBM baseline | GNN (`gin+reverse_mp+ego+ports`) | GNN ablation (`gin`, no extras) |
|---|---|---|---|
| Population | 23,220 rows, 261 fraud (1.1%) | 336 edges, 58 fraud (17.3%) | same 336 edges |
| PR-AUC (`pr_auc`) | 0.9087 | 1.0000 | 0.9801 |
| ROC-AUC (`roc_auc`) | 0.9984 | 1.0000 | 0.9955 |
| Precision @ top-1% (`precision_at_top1pct`) | 0.8621 | 1.0000 | 1.0000 |
| Recall @ top-1% (`recall_at_top1pct`) | 0.7663 | 0.0517 | 0.0517 |
| Alerts @ top-1% (`alerts_at_top1pct`) | 232 | 3 | 3 |

Validation PR-AUC (used to choose between the two GNN configurations): full
config **1.0000** (early-stopped at epoch 25, best at epoch 10), ablation
**0.8272** (early-stopped at epoch 38, best at epoch 23). The full configuration
was chosen on validation, not on test.

**These columns are not comparable like-for-like - do not read this as "the GNN
beats the GBM".**

- Different populations: the GBM ranks 23,220 rows at a 1.1% fraud rate; the GNN
  ranks 336 internal-transfer edges at a 17.3% fraud rate. PR-AUC rises with the
  base rate, so a PR-AUC of 1.0 on a 17% base rate and 0.91 on a 1% base rate are
  different achievements.
- "Top-1%" of 336 edges is **3 alerts**, so precision/recall @ top-1% for the GNN
  is essentially meaningless (recall 0.0517 is simply 3 of 58).
- A perfect score on 336 edges from a synthetic generator is an **upper bound**
  (see caveat 2), not an expected production number.

**Is the GNN just reading an easy edge feature?** Checked, and no. On the same
test edges, with no graph structure at all: the single feature `amount_sgd` gives
ROC-AUC 0.742 / PR-AUC 0.285, and gradient boosting on all four edge features
(`amount_sgd`, `direction`, `channel`, `timestamp`) gives ROC-AUC 0.739 /
PR-AUC 0.410. The GNN's 0.98-1.00 therefore comes from graph structure. This does
not rule out every possible leak, and the structure itself may be an artifact of
how the synthetic generator plants rings.

GBM reference points (from the dataset's own baseline, printed by
`train_transaction.py`):

- single-row only: **PR-AUC ~0.57**
- single-row + account history: **PR-AUC ~0.84**
- Stage 2 (single-row + static KYC + per-account rolling history): **0.9087** (measured)

---

## Recall by pattern @ top-1% (measured)

`train_gnn.py` prints this table via `print_recall_by_pattern`. `fraud_pattern` is
a **label**, used here only for attribution - never a model feature. Starred rows
are the typologies the GNN was designed for (`HIGHLIGHT_PATTERNS`).

| Fraud pattern | Recall @ top-1% | Caught / fraud edges |
|---|---|---|
| `ROUND_TRIPPING` * | 0.0517 | 3 / 58 (both configurations) |
| `DORMANT_REACTIVATION` * | n/a | no fraud edges with this pattern in split |
| `MULE_FAN_IN_OUT` | n/a | not printed: no fraud edges in the graph |
| `STRUCTURING`, `RAPID_PASS_THROUGH` | n/a | not printed: no fraud edges in the graph |

The `ROUND_TRIPPING` figure only says the 3 alert slots all landed on fraud edges.
With a 3-alert budget it cannot say anything about how well round-tripping is
detected overall - use the account-level comparison below for that.

---

## Node 3 vs the GNN, per account (measured)

**Provenance: this comparison is a dated snapshot and cannot be regenerated from
this repository.** The figures below came from a one-off comparison script that is
**not included here**, run once against the trained models. The method is written
out below (unit, adaptation, operating points) so it can be rebuilt; until then,
treat the numbers as a single-run measurement, not a reproducible benchmark.

**Unit:** an account is flagged by Node 3 if any of its AML-behaviour findings
fires (`FL_SUDDEN_DEPOSIT`, `FL_UNVERIFIED_COUNTERPARTY`, `FL_STRUCTURING`; the
tamper/identity codes are excluded), and by the GNN if its precomputed account
score is >= 0.50, the live Node 6a threshold (not tuned on test). An account is a
true positive if it has at least one `is_fraud` transaction in the split.

**Adaptation needed to run Node 3 on this dataset.** Node 3's sudden-deposit check
reads a baseline from a three-company fixture (`CASHFLOW_BASELINE`) that never
matches dataset accounts, so as shipped it would flag nothing here. The comparison
derived each account's baseline from its **KYC-declared** monthly turnover (else
income). The Node 3 numbers below are "Node 3 with a declared-income baseline",
not "Node 3 as it behaves live today".

### Test split: 1,201 accounts, 27 fraud accounts (95% Wilson intervals)

| Detector | Alerts | Fraud caught | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| Node 3, all AML rules | 68 | 14 | 54 | 20.6% [13-32%] | 51.9% [34-69%] |
| - `FL_SUDDEN_DEPOSIT` | 61 | 14 | 47 | 23.0% [14-35%] | 51.9% [34-69%] |
| - `FL_UNVERIFIED_COUNTERPARTY` | 0 | 0 | 0 | n/a | 0% [0-12%] |
| - `FL_STRUCTURING` | 8 | 1 | 7 | 12.5% [2-47%] | 3.7% [1-18%] |
| GNN (full config, >= 0.50) | 10 | 10 | 0 | 100% [72-100%] | 37.0% [22-56%] |
| GNN (ablation, >= 0.50) | 6 | 6 | 0 | 100% [61-100%] | 22.2% [11-41%] |

**Coverage:** the store has a score for only **62 of 1,201 accounts** and **10 of
the 27 fraud accounts**. The other 17 fraud accounts can never be flagged by the
GNN because they have no score. (A second GNN operating point, flagging the top 68
scored accounts to match Node 3's alert count, was also run. It is uninformative:
68 exceeds the 62 scored accounts, so it flags every scored account.)

**Overlap (full-config GNN vs Node 3):**

| | Both | Node 3 only | GNN only | Neither |
|---|---|---|---|---|
| Fraud accounts (27) | 5 | 9 | 5 | 8 |
| All alerts | 5 | 63 | 5 | n/a |

Jaccard overlap of the two alert sets is 6.8%. Together they catch 19 of 27 fraud
accounts (70%), against 52% for Node 3 alone and 37% for the GNN alone.

**By fraud typology (test split, fraud accounts caught):**

| Typology | Fraud accounts | Node 3 | GNN (full) | Both | Neither |
|---|---|---|---|---|---|
| `ROUND_TRIPPING` | 10 | 5 | 10 | 5 | 0 |
| `DORMANT_REACTIVATION` | 4 | 2 | 0 | 0 | 2 |
| `RAPID_PASS_THROUGH` | 5 | 4 | 0 | 0 | 1 |
| `STRUCTURING` | 4 | 3 | 0 | 0 | 1 |
| `MULE_FAN_IN_OUT` | 4 | 0 | 0 | 0 | 4 |

### Larger sample for Node 3 only: all splits, 8,000 accounts, 160 fraud accounts

Node 3 is rule-based and fits nothing to the labels, so using every split does not
leak. The GNN is left out here because its scores on train/val are optimistic.

| Detector | Alerts | Fraud caught | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| Node 3, all AML rules | 471 | 85 | 386 | 18.0% [15-22%] | 53.1% [45-61%] |
| - `FL_SUDDEN_DEPOSIT` | 430 | 85 | 345 | 19.8% [16-24%] | 53.1% [45-61%] |
| - `FL_STRUCTURING` | 42 | 1 | 41 | 2.4% [0-12%] | 0.6% [0-3%] |

By typology: `RAPID_PASS_THROUGH` 29/30, `DORMANT_REACTIVATION` 18/27,
`STRUCTURING` 14/28, `ROUND_TRIPPING` 24/53, `MULE_FAN_IN_OUT` 0/22.

**Structuring window mismatch.** Node 3 looks for amounts in [9,500, 10,000).
None of the 315 `STRUCTURING` fraud transactions fall in that window (median
amount 18,160). Node 3 does flag 14 of the 28 structuring accounts, but the
structuring rule itself has only 1 true positive across all 160 fraud accounts; the
rest are caught by the sudden-deposit rule. A sensitivity run with the
threshold raised to 20,000 puts 56 of 315 transactions in the window and moves
overall recall only from 53.1% to 55.0%. `FL_UNVERIFIED_COUNTERPARTY` never fires
because the dataset's descriptions are generic ("FAST TRANSFER IN") and carry no
counterparty names, so that rule cannot be evaluated on this data.

### What this says

- **Not redundant on round-tripping.** The GNN catches 5 round-tripping accounts
  that Node 3 misses, with no false alarms at the live threshold. Node 3 reaches
  round-tripping only through large-amount sudden deposits (it cannot see the
  loop), at much lower precision.
- **Narrow, not general.** The GNN covers one of five typologies and scores 62 of
  1,201 accounts. It adds nothing on the other four.
- **A shared blind spot.** Neither catches any of the mule fan-in/out accounts
  (0/4 test, 0/22 overall for Node 3, and the GNN cannot see them at all).
- **Sample size.** 10 round-tripping accounts support the GNN result; the intervals
  above are wide.

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

**Measured effect.** At the account level the live lookup has a score for only 62
of 1,201 test accounts (10 of 27 fraud accounts), all of them accounts with
outgoing internal transfers. The account-level figures in the Node 3 comparison
above are the ones that reflect what the live path can actually do.

### 2. Synthetic upper-bound caveat (Open Question #2)

The dataset is synthetic, and per its data card the reported scores are an
**upper bound**. The graph lift — the margin the GNN adds over the tabular
baseline on the relational typologies — **may shrink on real data**. Treat any
improvement shown in the tables as the optimistic end of the range, not the
expected production delta.

---

## Checklist

- [x] Ran `train_transaction` and recorded the GBM `TEST` report values.
- [x] Ran `train_gnn` (full configuration and the plain ablation) and recorded the
      GNN `TEST` report values.
- [x] Filled every `TBD` cell in the headline and recall-by-pattern tables.
- [x] Recorded `print_recall_by_pattern` output, including the "no fraud edges"
      line.
- [x] Removed the "template / not measured" banner.
- [ ] Repeat the GNN runs across several random seeds (the trainer sets none) and
      report the spread.
- [ ] Re-run on a larger or real dataset; 27 fraud test accounts is too few for
      firm conclusions.
- [ ] Note the dataset snapshot used (the dataset folder is not versioned here).
