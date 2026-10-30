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
| **1 Ingestion & Preprocessing** | `fraud_detect/nodes/ingestion.py` | PDF → text (OCR stand-in), extracts file **metadata**, captures per-page **font fingerprints**, parses the statement **header** + **transaction table** (several layouts, see below), emits explicit **data-quality** findings for missing/unreadable fields (never guesses). Output is a strict, validated `IngestionResult`. |
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

FastAPI app (`fraud_detect/webapp.py`) where a credit analyst works through each
stage of a case:

- **Screening queue** (`/`) — all cases (auto-cleared vs escalated); still-open
  cases (awaiting a human, or awaiting the customer) sort to the top, highest risk
  first. A status badge shows `PENDING` / `NEEDS INFO` / `APPROVED` / `REJECTED`.
- **Case overview** (`/case/{id}`) — risk score, confidence score, a
  **findings-by-pipeline-stage** summary, and the decision panel.
- **Flagged transactions** (`/case/{id}/transactions`) — every parsed row joined
  with the findings that cite it; flagged rows highlighted and listed first, each
  with its reason code and evidence.
- **Pipeline stages** (`/case/{id}/stages`) — what each agent produced
  (Ingestion / Visual forensics / Financial logic / Cross-case memory / ML), with
  a per-stage "clear" or finding count — so the analyst can see *which stage* fired.
- **Statement PDF** (`/pdf/{id}`).
- **Three analyst actions** on a case:
  - **Request more information** → sets the case to **`NEEDS_INFO`** and stores a
    *customer-safe* message that the customer portal relays (no scores/evidence).
    A non-terminal state: the applicant can re-submit and the case returns to review.
  - **Approve** (clear the concern) or **Reject** (confirm), each requiring a
    **reason** and capturing optional **feedback for future model training**.
- Everything is written to a **case store + audit log** (`fraud_detect/store.py`),
  including `INFO_REQUESTED` events.

> The machine never rejects on its own — only a human can approve, reject, or ask
> for more information. See `CUSTOMER_PORTAL_README.md` for how these states appear
> to the applicant.

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

## Reading statements in Node 1 (multiple layouts)

Node 1 no longer assumes one statement layout. The parser lives in
`fraud_detect/nodes/ingest_parse/`. How it reads a statement now:

- It tries every **layout profile** on the file. The profile that reads the most
  of each page (best coverage) wins; on a tie the earlier profile in the registry
  wins, unless the tied profiles read different rows, in which case nothing is
  decided and the file is flagged.
- **Direction** (money in or out) comes from the column the amount sits in, or the
  printed sign, or the change in running balance, in that order of preference.
- The **printed amount is always kept** as printed. If a row's direction disagrees
  with the balance change, the amount stays and the row is flagged.
- A **self-check** compares printed section totals with the rows it read, so a
  reading mistake of ours is not mistaken for document fraud.

New data-quality reason codes (all from Node 1; they ask a human to look, they
never reject anyone):

| Code | Severity (weight) | Meaning |
|------|-------------------|---------|
| `DQ_LAYOUT_UNRECOGNISED` | MEDIUM (8) | No profile read a page well enough, tied profiles disagreed, or an unexpected error stopped a page. Rows that were read are kept. |
| `DQ_DIRECTION_UNCERTAIN` | LOW (3) | The credit/debit direction of a row disagrees with the balance change. |
| `DQ_PARSE_INCONSISTENT` | LOW (3) | Every row's running balance reconciles, but a printed section total differs from the parsed sum. |
| `DQ_UNREADABLE_ROW` | LOW (3) | A row looked like a transaction but could not be fully read. |

A layout we cannot read is **never auto-cleared**. It is flagged
`DQ_LAYOUT_UNRECOGNISED` and goes to a human.

### The six layout profiles

Registered in `REGISTRY` in `fraud_detect/nodes/ingest_parse/layouts.py`:

| Profile | Layout it covers |
|---------|------------------|
| `generator_v1` | Our own generator (`fraud_detect/synthetic.py`) |
| `two_col_dmon` | `statement_E` / `statement_F` style, and dataset layout 5 |
| `two_col_slash` | Dataset layout 1 |
| `two_col_dual_date` | Dataset layout 2 |
| `signed_amount_iso` | Dataset layout 3 |
| `two_col_ddmmyy_cr` | Dataset layout 4 |

### How to add a layout profile

1. **Build a word-box fixture** for the new layout in
   `tests/ingest_parse/fixtures.py` (fictional values only).
2. **Write a failing test** that parses the fixture and expects its header, rows
   and balances.
3. **Add the `LayoutProfile`** in `fraud_detect/nodes/ingest_parse/layouts.py`.
4. **Register it in `REGISTRY`.** Order matters: when two profiles tie, the earlier
   one wins.
5. **Run the evaluation script:**
   ```bash
   python -m fraud_detect.nodes.ingest_parse.eval_layouts            # 20 genuine docs per layout
   python -m fraud_detect.nodes.ingest_parse.eval_layouts --sample 50
   python -m fraud_detect.nodes.ingest_parse.eval_layouts --all-genuine
   ```
6. **Run the full test suite:** `python -m pytest -q`

### What this does and does not prove

- With only synthetic layouts, success **proves the parser design**. It does **not**
  prove that real Singapore bank statements will parse.
- Real-bank profiles are added later, when real data can be handled safely
  (on-premises, anonymised, with consent).
- The external evaluation files (`statement_forgery_dataset`, `statement_E` /
  `statement_F`, the answer key) are **not stored in the repository**. Tests that
  need them are skipped when the files are missing.

### Results (synthetic-only)

Measured on synthetic layouts, 20 genuine documents per layout by default. Reading
all 1,000 genuine documents also gives 100%.

| Layout | Profile | Rows read | Balance replays | Header match |
|--------|---------|-----------|-----------------|--------------|
| 1 | `two_col_slash` | 100.0% | 100.0% | 100.0% |
| 2 | `two_col_dual_date` | 100.0% | 100.0% | 100.0% |
| 3 | `signed_amount_iso` | 100.0% | 100.0% | 100.0% |
| 4 | `two_col_ddmmyy_cr` | 100.0% | 100.0% | 100.0% |
| 5 | `two_col_dmon` | 100.0% | 100.0% | 100.0% |

- `statement_E` and `statement_F`: all six months match the answer key to the cent.
- `statement_E`: 12 of 12 inserted rows are read as transactions.
- Longest file took 0.36 s.
- All 1,000 genuine dataset documents still route to `HUMAN_REVIEW`. They are
  single-month personal accounts, so they raise `DQ_MISSING_UEN` and
  `FL_PERIOD_LENGTH` (plus `VF_CURRENCY_MIX` on layout 5). This is expected and
  not a parser problem.

**Known limitation.** Overlay-style tampers (a new amount drawn over the old one)
can garble the printed amount, so the reader may drop that row instead of reading
it. 200 of 1,000 tampered dataset documents drop at least one row. This is not yet
flagged per row.

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
4. **Synthetic-layout parser.** The transaction/header parser reads six layouts
   (our generator plus five dataset layouts), all synthetic. Real statements vary
   widely by bank and each real bank would need its own profile (see
   [Reading statements in Node 1](#reading-statements-in-node-1-multiple-layouts)).
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

**Node 6b (statement forgery) is a document-forensics check with a size guard.**
Its balance-reconciliation feature now reconciles each transaction row (the earlier
version paired every money value on the page and reported breaks on genuine
documents), and two features that only reflected how the training dataset's
generator prints documents were removed from its inputs. It abstains on documents
larger than anything it was trained on (training statements are single months), and
its severity cutoffs live in the model file. Retrain with
`python -m fraud_detect.ml.train_statement --data <statement_forgery_dataset>`; it
caches features in `data/private/statement_features_v2.csv` (do not reuse the older
`statement_features.csv`, which holds the earlier, incorrect feature).

**The GBM transaction model is off in the live app by default.** On clean business
statements it raised false alarms without adding detection: the dataset is mostly
small retail payments, and the fields that let it tell businesses apart (segment,
KYC, channel, counterparty) are not on a PDF statement. The web app and `demo.py`
load it only when `FRAUD_ENABLE_TXN_MODEL=1` is set. `train_transaction.py
--feature-set pdf` trains a version restricted to statement-only fields, for when
representative statements are available.

**Tests:** `tests/test_gnn_graph_builder.py` (split-safety, leaky-column and
INTERNAL-only edge guards), `tests/test_gnn_lookup.py` (missing file / unknown key
/ corrupt store all fail-safe to `None`), `tests/test_gnn_scores_store.py` (store
shape + keying), and `tests/test_node6a_gnn.py` (elevated account → one finding at
the right severity; no store → behavior unchanged; never auto-rejects).
---

## Immutable audit trail (signed, hash-chained)

Audit events — `SCREENED`, `HUMAN_APPROVED`, `HUMAN_REJECTED` — are written to a
dedicated **append-only, hash-chained, Ed25519-signed** log at
`data/audit_chain.jsonl`, kept separate from the mutable case snapshot in
`data/cases.json`. The snapshot legitimately changes as a review moves from
`PENDING` to a decision; the event history must not. Co-locating them in one
rewritten file is exactly what made the old embedded `audit_log` forgeable, so the
two are split: each event is one newline-delimited JSON object, links to its
predecessor by a hash chain (`prev_hash` → the prior event's `entry_hash`), and is
signed over a payload that commits to that `entry_hash`. Editing, deleting,
reordering, inserting, or a full rewrite-with-rehash all become **detectable**,
because an attacker cannot produce valid signatures without the private key.

**How to verify**

```bash
# Verify the chain is intact and signed (exit code is the contract):
python -m fraud_detect.audit.verify [--path data/audit_chain.jsonl]
```

The exit code is the machine-readable contract:

| Exit | Result | Meaning |
|------|--------|---------|
| `0` | `INTACT_SIGNED` | hash chain walks cleanly **and** every event is validly signed (the only OK) |
| `1` | `BROKEN_CHAIN` / `SIGNATURE_INVALID` | a broken/edited/reordered chain, or a signature that did not validate |
| `2` | `INTACT_UNSIGNED` | chain intact but events carry no signature — surfaced **distinctly** so it is never mistaken for signed |
| `3` | `ERROR` | verification could not run (e.g. a configured signing key could not be loaded) |

**Signing key config.** Set `AUDIT_SIGNING_KEY_PATH` to an Ed25519 PEM private key
located **outside** the `data/` directory (so the key is never co-located with the
log it protects). If the variable is **unset**, the chain still records events but
they are **UNSIGNED**, and `verify` reports `INTACT_UNSIGNED` (exit 2) — never
"signed". If the variable is **set but the key is unreadable or invalid**, the
system **fails loudly** rather than silently falling back to unsigned. The key path
is pluggable behind a `Signer` interface so production can swap the local-file key
for a KMS/HSM backend without touching the chain.

**The honest security boundary.** Stated plainly so no one overstates the
prototype:

- The prototype is **tamper-evident and forgery-resistant *while the signing key is
  secure*.** Editing, deleting, reordering, inserting, and a full rewrite-with-rehash
  are all detectable, because the attacker cannot forge valid signatures without the
  private key.
- It is **not** proof against signing-key **theft.** An attacker who steals the
  private key can forge. Production **MUST** move key custody to a **KMS/HSM so the
  private key never leaves the security boundary** — the app process would then hold
  no private-key material at all.
- It is **not** proof against a full **wipe-and-restart** of the chain (delete
  everything, begin a fresh genesis). Detecting that requires external
  anchoring/checkpointing, which is a **production requirement and out of scope for
  the prototype.**

**Governance (never auto-repair, never auto-reject).** A broken, invalid,
missing-signature, or unexpectedly-UNSIGNED chain is surfaced to a human as an
incident. The system never auto-repairs or rewrites the chain (auto-editing an
append-only log would itself be a tamper vector), and an integrity failure never
triggers an automated reject on a customer — consistent with the whole system's
never-auto-reject stance.
