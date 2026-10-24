# Bank Statement Tampering Screening — 4-Agent Prototype

Screens corporate-loan bank statements for signs of **tampering** and
**suspicious activity** *before* a credit analyst touches them, auto-clears clean
files, and attaches **readable, reason-coded evidence** to every flag so each
decision can be explained and audited. **The machine never rejects a customer on
its own** — the most severe automated outcome is a hand-off to a human reviewer.

---

## Problem & end goals

Tampered statements reach credit assessment because catching them relies on slow,
inconsistent manual review. This system:

- Screens **every** submitted statement automatically.
- **Auto-clears clean files** so analysts only spend time on flagged ones.
- Attaches **explainable evidence** to every flag (aligns with MAS **FEAT**).
- **Never auto-rejects** — humans make the reject/approve call.

---

## Architecture — 4 worker nodes

```
 PDF ─► Node 1 Ingestion ─┬─► Node 2 Visual Forensics ─┐
                          └─► Node 3 Financial Logic ──┴─► Node 4 Risk Synthesis
                                                                  │
                                             risk + confidence ───┤
                                                                  ▼
                                     ┌──────────── routing ───────────┐
                              AUTO_CLEAR (low risk,          HUMAN_REVIEW (medium/high
                              high confidence,               risk, low confidence, or
                              no material findings)          poor data quality)
                              → approval recommendation      → credit analyst reviews
```

| Node | File | What it does |
|------|------|--------------|
| **1 Ingestion & Preprocessing** | `fraud_detect/nodes/ingestion.py` | PDF → text (OCR stand-in), extracts file **metadata**, captures per-page **font fingerprints**, parses the statement **header** + **transaction table**, emits explicit **data-quality** findings for missing/unreadable fields (never guesses). Output is a strict, validated `IngestionResult`. |
| **2 Visual Forensics** | `fraud_detect/nodes/visual_forensics.py` | Localised **font inconsistency** (altered text), mixed **date/currency formats**, **editing-software** traces in metadata (Photoshop/Illustrator/GIMP…), and **creation-date-before-period**. |
| **3 Financial Logic & Validation** | `fraud_detect/nodes/financial_logic.py` | Row-by-row **running-balance reconciliation**, closing-balance check, **period-continuity** gaps, **sudden deposits** vs a historical baseline, **unverified counterparties**, **structuring** below the reporting threshold, and **UEN↔ACRA** entity cross-check. |
| **4 Risk Scoring & Synthesis** | `fraud_detect/nodes/risk_synthesis.py` | Aggregates all findings into a **risk score (1–100)** + band, a **confidence score (1–100)** + band, a **routing** decision with reason, and a plain-language summary. |

Orchestration lives in `fraud_detect/pipeline.py` (`run_pipeline(pdf_path) -> RiskReport`).

### Structured output contract
Every agent emits typed, validated Pydantic models (`fraud_detect/models.py`).
Missing mandatory fields become explicit `null` values surfaced as data-quality
findings — the model does not hallucinate values.

---

## Scoring & routing

**Risk bands:** 1–25 Low · 26–50 Medium · 51–75 High · 76+ Very High
**Confidence bands:** same cut-offs (how certain the system is of the risk call).

- **Risk** = sum of finding weights (clamped 1–100), plus a corroboration bonus
  when independent agents both raise high-severity flags. Any material (MEDIUM+)
  finding floors the score into at least the Medium band.
- **Confidence** starts high for a clean digital PDF and is **reduced** by
  data-quality problems, thin evidence near a band boundary, and single-dimension
  (only visual *or* only financial) evidence; it is **raised** by corroboration.
- **Routing:**
  - `AUTO_CLEAR` only when risk ≤ 25 **and** confidence ≥ 60 **and** no material findings.
  - `HUMAN_REVIEW` otherwise — including **low confidence** or **poor data quality**,
    even when the numeric risk looks low.

Conditional decisions (per the design):
- No significant issues → **Low** → auto-clear.
- Some inconsistencies → **Medium** → human review.
- Strong tampering indicators → **High/Very High** → human review.
- Low AI confidence / poor data / conflicting evidence → human review regardless of band.

---

## Human-in-the-loop review UI

FastAPI app (`fraud_detect/webapp.py`) where a credit analyst can:
- see the **screening queue** (auto-cleared vs escalated),
- open a case and read the **risk score, confidence score, every detected issue
  with its evidence and the relevant page/row**,
- **view the statement PDF**,
- **confirm** the concern (reject application) or **clear** it (approve), recording
  a **reason** and optional **feedback for future model training**,
- everything is written to a **case store + audit log** (`fraud_detect/store.py`).

---

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) Generate synthetic samples (clean + tampered)
python -m fraud_detect.synthetic

# 2) End-to-end demo (statement → 4 agents → risk + confidence → evidence → human review)
python demo.py

# 3) Run the reviewer web UI
uvicorn fraud_detect.webapp:app --port 8000
#    then open http://127.0.0.1:8000/

# 4) Run the test suite (Phase 5)
python -m pytest -q
```

---

## Synthetic data & planted issues (Phases 2–3)

`fraud_detect/synthetic.py` generates PDFs **with a ground-truth manifest** of what
was planted, so the tests can assert detection. Planted-issue kinds → detected
reason codes:

| Planted issue | Detected by |
|---------------|-------------|
| `ALTERED_BALANCE` | `FL_BALANCE_BREAK` |
| `ALTERED_NUMBER` | `FL_BALANCE_BREAK`, `VF_FONT_MISMATCH` |
| `FORMAT_INCONSISTENCY` | `VF_DATE_FORMAT_MIX`, `VF_CURRENCY_MIX` |
| `METADATA_MISMATCH` | `VF_EDIT_SOFTWARE`, `VF_CREATION_BEFORE_PERIOD` |
| `PERIOD_GAP` | `FL_PERIOD_GAP` |
| `SUSPICIOUS_TXN` | `FL_SUDDEN_DEPOSIT`, `FL_UNVERIFIED_COUNTERPARTY` |
| `STRUCTURING` | `FL_STRUCTURING` |
| `UEN_MISMATCH` | `FL_UEN_NAME_MISMATCH` |

`tests/test_detection.py` (13 tests) verifies each planted issue is caught, that a
clean statement auto-clears, that borderline/unreadable inputs escalate, and that
the machine never auto-rejects.

---

## Governance alignment

- **MAS FEAT (Fairness, Ethics, Accountability, Transparency):** every flag carries
  a stable **reason code** and human-readable **evidence**; the final decision is
  always made by a **human**, and both the AI output and the human decision are
  recorded.
- **PDPA:** the prototype uses only **synthetic** data. In production, statement
  contents and PII would be access-controlled, encrypted at rest/in transit, and
  retained per a defined policy.
- **MAS TRM (Technology Risk Management):** a full **audit log** records screening
  results and every reviewer decision. The JSON store is a stand-in for a
  production, access-controlled datastore.

---

## Limitations (be explicit in the demo)

This is a **prototype** with deterministic, rule-based detectors — not production AI:

1. **No real AI models.** Visual forensics uses structural signals (fonts, formats,
   metadata), **not** pixel-level ELA / copy-move / deep-learning image forensics.
   OCR is stubbed by digital-PDF text extraction; **real scanned images are not
   OCR'd** here.
2. **Synthetic-tuned thresholds.** Weights and thresholds (structuring window,
   deposit multiples, gap days) are illustrative and would need calibration on real,
   labelled data — with false-positive/false-negative analysis.
3. **Toy reference data.** ACRA registry, known-vendor list and cash-flow baselines
   are small in-memory fixtures, not live sources; no cross-application reuse
   detection across a real corpus.
4. **Statement-format-specific parser.** The transaction/header parser targets the
   generator's layout; real statements vary widely by bank and would need per-bank
   templates or a learned layout model.
5. **Not a fairness-audited model.** No bias testing, no adversarial-robustness
   testing, and the confidence score is a heuristic, not a calibrated probability.
6. **Prototype persistence & security.** JSON file store, no auth, no encryption —
   fine for a demo, not for production.

The value shown is the **end-to-end shape**: ingestion → multi-agent detection →
explainable risk + confidence → routed human review with an audit trail — with
clear seams where production OCR, image-forensics, ML scoring, and real reference
data would slot in.

---

## Node 5 — Historical Fraud Pattern Intelligence & Feedback Loop

By default each statement is screened in isolation. The optional **Cross-Case
Memory Agent** gives the pipeline a *memory* across submissions: before a case is
finalised it compares the current statement's **suspicious** transactions against
prior cases **a human reviewer has already decided on** (recalled from the case
store) and flags repeats.

This answers a question a single-case screener cannot: *"have we seen this
suspicious transaction/entity before — and did a reviewer clear it or stop it?"*

**Learns only from authorised historical evidence.** Matching uses only cases a
reviewer has explicitly **REJECTED** or **APPROVED**. Cases still awaiting review
(an AI screening with no human sign-off) are ignored — so an unverified AI
suspicion never becomes a "fact" used against a later applicant. This is the
feedback loop: verified reviewer outcomes feed forward to make future flags more
informative.

| Situation | Reason code | Severity |
|-----------|-------------|----------|
| A near-identical transaction was **rejected** by a reviewer in a prior case | `MEM_RECURRING_KNOWN_BAD` | HIGH |
| A similar suspicious transaction was **approved/cleared** by a reviewer before | `MEM_SIMILAR_PASSED_BEFORE` | MEDIUM |
| The same counterparty/amount recurs across **many** reviewed prior cases | `MEM_HIGH_FREQUENCY_PATTERN` | MEDIUM |

**How it stays precise and auditable**
- Only transactions already flagged by Node 2/Node 3 are fingerprinted, so routine
  payroll/utility payments never enter memory or create noise.
- A fingerprint = normalized counterparty + bucketed amount + direction, so a
  lightly-tweaked repeat (e.g. 149,500 vs 150,000, or "Shell Co" vs "Shell Co Pte Ltd")
  still matches.
- Every finding **cites the prior case id**, so a reviewer can trace the link.
- It only *adds evidence and escalates* — it never auto-rejects (MAS FEAT preserved).

**How it is wired**
`run_pipeline(pdf_path, store=my_store)` activates the memory agent (it needs a
`CaseStore` to recall prior cases from). Called without a store, the pipeline
behaves exactly as the 4-node version.

**See it in action**
```bash
python demo_memory.py
```
This screens a tampered statement, records a human REJECT, then re-submits the
same doctored statement as a new application — and Node 5 reports that each
suspicious transaction was seen (and rejected) before, citing the earlier case.

**Tests:** `tests/test_memory.py` covers recurrence detection (rejected + cleared
paths), fingerprint normalization, auditability (prior case id cited), and that
the machine never auto-rejects.

---

## Node 6a — GNN relational-AML signal (offline-only dependencies)

The transaction-fraud agent (Node 6a) has an optional graph-neural-network
extension under `fraud_detect/ml/gnn/`, adapted from
[IBM/Multi-GNN](https://github.com/IBM/Multi-GNN) (Apache-2.0, vendored under
`fraud_detect/ml/gnn/vendor/` with attribution in `PROVENANCE.md`). It captures
the relational AML typologies the tabular model is weakest at (round-tripping,
dormant reactivation, mule fan-in/out). It only **adds explainable evidence** and
**never auto-rejects** — same governance stance as every other node.

**The GNN is offline-batch, not live.** It trains on the full transaction graph
and precomputes a per-account / per-edge risk store; at screening time Node 6a
does an O(1) lookup instead of running the model.

**Heavy dependencies are offline-only.** `torch` and `torch-geometric` live in a
separate `requirements-gnn.txt` and are **not** needed to run the live screening
pipeline — install them only to train the GNN or regenerate the scores store:

```bash
# Offline / training only — NOT required for the live pipeline:
pip install -r requirements-gnn.txt
```

The live lookup path (`fraud_detect/ml/gnn/gnn_lookup.py`) imports only `joblib`,
which is already in the core `requirements.txt`.

**Offline vs live, concretely.**

- *Offline (batch, has the whole graph).* `fraud_detect/ml/gnn/graph_builder.py`
  builds a split-safe **INTERNAL-transfer** graph (accounts = nodes, INTERNAL
  transfers = edges; leaky columns and cross-split edges are excluded).
  `fraud_detect/ml/gnn/train_gnn.py` trains the vendored Multi-GNN backbone on that
  graph **and** writes the precomputed scores store by running inference over all
  splits.
- *Live (per-statement, no graph).* Node 6a
  (`ml_transaction.analyze(ing, model, gnn_scores=...)`) does an **O(1)
  account-level lookup** via `gnn_lookup.score_account`, keyed on `ing.uen`. The
  store is loaded once in `webapp.py` / `demo.py`; if it is absent the lookup
  returns `None` and the pipeline behaves exactly as before (fail-safe).

The live path only has `ing.uen` as a key, so the live contribution is coarse
(account-level) while the offline model scores at the edge level — an honest gap.
The full evaluation write-up (GNN test PR-AUC, recall-by-pattern vs the ~0.90 GBM
baseline, and this live-vs-offline caveat) lives in
`fraud_detect/ml/gnn/EVALUATION.md`.

**Retrain and regenerate the scores store.** Install the offline deps first
(`pip install -r requirements-gnn.txt`), then:

```bash
# Trains the GNN AND writes the precomputed models/gnn_scores.joblib store
# (via inference over all splits):
python -m fraud_detect.ml.gnn.train_gnn \
    --data ~/Downloads/transaction_fraud_dataset \
    --model gin --reverse_mp --ego --ports \
    --out models/txn_fraud_gnn.pt \
    --scores-out models/gnn_scores.joblib

# GBM baseline trainer, for comparison:
python -m fraud_detect.ml.train_transaction \
    --data ~/Downloads/transaction_fraud_dataset \
    --out models/txn_fraud_clf.joblib
```

**The new finding.** When the account lookup is elevated, Node 6a emits one
`ML_TXN_GNN_RISK` finding (HIGH ≥ 0.80, MEDIUM ≥ 0.50) tagged with the agent
sub-label `Node6a:MLTransaction:GNN`. It is **independent evidence** that flows
into Node 4 corroboration alongside the GBM's `ML_TXN_FRAUD_LIKELIHOOD` — so
GBM+GNN agreement naturally strengthens the risk call. Like every other node it
only adds evidence for a reviewer and never auto-rejects.

**Tests:** `tests/test_gnn_graph_builder.py` (split-safety, leaky-column and
INTERNAL-only edge guards), `tests/test_gnn_lookup.py` (missing file / unknown key
/ corrupt store all fail-safe to `None`), `tests/test_gnn_scores_store.py` (store
shape + keying), and `tests/test_node6a_gnn.py` (elevated account → one finding at
the right severity; no store → behavior unchanged; never auto-rejects).
