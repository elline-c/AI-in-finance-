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
| **2 Visual Forensics** | `fraud_detect/nodes/visual_forensics.py` | Localised **font inconsistency** (altered text), mixed **date formats**, mixed **currency notation in the transaction amounts** (the account-header label is ignored), **editing-software** traces in metadata (Photoshop, GIMP, PDF editors…), a file **modified after it was created** by something other than a print/export tool, and **creation-date-before-period**. See [Node 2 checks](#node-2-checks-metadata-and-currency). |
| **3 Financial Logic & Validation** | `fraud_detect/nodes/financial_logic.py` | Row-by-row **running-balance reconciliation**, closing-balance check, **period-continuity** gaps, **sudden deposits** vs a historical baseline, **unverified counterparties**, **structuring** (3+ near-threshold credits within 14 days), **UEN↔ACRA** entity cross-check, and the **fabrication rules** (repeated invoice references, invoice numbers out of date order, turnover jumps against flat fixed costs; see [Node 3 fabrication rules](#node-3-fabrication-rules)). |
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

## Node 3 fabrication rules

Node 3 now has three extra checks aimed at **fabricated business income**: fake
customer receipts pasted into a statement. They live in
`fraud_detect/nodes/fabrication.py` and run inside `financial_logic.analyze`.
Like every other finding, they **never reject anyone**. They add a reason and
evidence, and route the file to a human.

### Reason codes

| Code | Severity (weight) | Meaning |
|------|-------------------|---------|
| `FL_DUP_INVOICE_REF` | MEDIUM (10) per repeated reference; one HIGH (22) summary when 3 or more references repeat **within a month**; across months: one MEDIUM (10) for repeats too far apart or too frequent to be an instalment, one weak LOW (3) note for close ones | The same customer and invoice reference appears on more than one credit in the same calendar month. |
| `FL_INVOICE_SEQUENCE_BREAK` | LOW (5) | Invoice numbers run out of date order. |
| `FL_TURNOVER_COST_MISMATCH` | MEDIUM (8) | Income jumps in several months while payroll, CPF and rent stay flat. |
| `DQ_RULE_SKIPPED` | LOW (2) | One of the checks hit an internal error and was skipped; the others still ran. |

### What each rule looks for

The example is `statement_E`, which has 12 fake receipts inserted.

- **Duplicate invoice reference (`FL_DUP_INVOICE_REF`).** The 12 fakes reuse a
  customer name and invoice number that already appear on a genuine receipt, in the
  same month. A monthly statement has no reason to show one invoice paid twice, so the
  rule groups credits by customer, invoice reference **and calendar month** and names
  every row in a group. One same-month repeat gives a MEDIUM finding per reference.
  Three or more repeated references give a single HIGH summary, so the score is not
  flooded. The same invoice appearing in **different** months (often an instalment that
  crosses a month end) is judged on distance: **close in time and few payments** (45 days
  or less and at most 3 payments, the instalment limits) is a single weak LOW note of
  weight 3; **far apart or too many payments** (a customer rarely pays one invoice again
  months later) is a MEDIUM of weight 10, reported as one finding for the whole file.
  Neither counts toward the HIGH summary. A credit with no date counts as inside every
  month, and a group that cannot be dated gets the benefit of the doubt (LOW). The evidence
  notes when a group looks like an instalment (few payments, close together) or an
  identical amount (possible duplicate posting).
- **Invoice sequence break (`FL_INVOICE_SEQUENCE_BREAK`).** Invoice numbers
  normally rise with the date. Fakes are often pasted in with numbers that do not
  fit. The rule finds the longest in-order run and lists the credits outside it.
  It only needs 6 references and 2 breaks to speak. It also gives the duplicate
  rule a hint about which row of a pair is the likelier copy.
- **Turnover against fixed costs (`FL_TURNOVER_COST_MISMATCH`).** On `statement_E`
  income is up 25–30% in three months, yet payroll, CPF and rent stay within 10%
  of their median. Real growth usually brings costs with it. It needs at least 4
  full months.

Every finding says what it saw and asks a person to check. None is proof.

### Tuning: `FabricationConfig`

All settings are in one frozen dataclass, `FabricationConfig`. `DEFAULT_CONFIG` is
used when nothing is passed. Bad values raise `ValueError` at creation.

| Field | Default | Raising it | Lowering it |
|-------|---------|------------|-------------|
| `invoice_patterns` | `INV` + 3 or more digits | n/a (a list of regexes: group 1 prefix, group 2 number; add patterns for other reference styles) | n/a |
| `rail_prefixes` | `FAST FROM`, `PAYNOW FROM`, `GIRO`, `TT FROM` | n/a (payment-rail words stripped from the front of a description so the customer name matches across rails) | n/a |
| `dup_escalate_min_refs` | 3 | More repeated references needed before the HIGH summary; fewer HIGH findings | HIGH summary appears sooner |
| `dup_min_payments_same_month` | 2 | One invoice must be paid this many times in one calendar month to count as a same-month repeat (minimum 2). 3 would miss a plain original-plus-copy pair | Not allowed below 2 |
| `dup_far_apart_weight` | 10 | Far-apart cross-month repeats move the score more | Moves it less |
| `dup_cross_month_weight` | 3 | Weak cross-month note moves the score more | Moves it less (0 keeps the note but adds nothing) |
| `dup_isolated_weight` | 10 | Each lone repeat moves the score more | Moves it less |
| `dup_escalated_weight` | 22 | HIGH summary moves the score more | Moves it less |
| `instalment_max_days` | 45 | More groups get the "may be an instalment" note | Fewer get it |
| `instalment_max_count` | 3 | Larger groups still get the instalment note | Only very small groups do |
| `evidence_max_groups` | 20 | Longer evidence text in the summary | Shorter text; the rest is counted as "and N more" |
| `seq_min_refs` | 6 | Rule B needs more references; quieter on short files | Speaks on shorter files |
| `seq_min_breaks` | 2 | Needs more out-of-order credits; quieter | A single break is enough; noisier |
| `seq_weight` | 5 | Moves the score more | Moves it less |
| `fixed_cost_keywords` | `CPF`, `PAYROLL`, `SALARY`, `RENT` | n/a (debit words that count as fixed costs) | n/a |
| `turnover_exclude_keywords` | `LOAN`, `CAPITAL`, `TRANSFER`, `INTEREST` | n/a (credit words left out of turnover) | n/a |
| `credit_jump_ratio` | 1.2 | A month must beat the median of the other months by more; fewer jump months, quieter | Smaller rises count as jumps; noisier |
| `fixed_cost_stability_pct` | 0.10 | Wider "flat" band; more months count as stable; noisier | Stricter flat band; quieter |
| `turnover_min_months` | 4 | Needs a longer statement before Rule C runs | Runs on shorter statements |
| `turnover_min_stable_share` | 2/3 | Needs most months to have stable costs; quieter | Tolerates more uneven costs; noisier |
| `turnover_min_jump_months` | 2 | Needs more jump months; quieter | One jump month is enough; noisier |
| `turnover_weight` | 8 | Moves the score more | Moves it less |

**Worked example: a business with seasonal sales.** A shop whose sales spike
every few months while rent and payroll stay put will look like Rule C's pattern.
Ask for a bigger jump and more jump months before it speaks:

```python
from fraud_detect.nodes import financial_logic
from fraud_detect.nodes.fabrication import FabricationConfig
from fraud_detect.nodes.ingestion import ingest

ing = ingest("statement.pdf")
seasonal = FabricationConfig(credit_jump_ratio=1.4, turnover_min_jump_months=3)
report = financial_logic.analyze(ing, fabrication_config=seasonal)
```

I ran this on `statement_E` (read-only). With defaults it reports
`FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK` and `FL_TURNOVER_COST_MISMATCH`.
With the settings above `FL_TURNOVER_COST_MISMATCH` goes quiet (E's jumps are
1.25x to 1.30x) and the other two remain. That is the trade-off: the same setting
that spares a seasonal shop would also mute a modest fabricated rise.

### Running the evaluation report

```bash
python -m fraud_detect.nodes.fabrication_eval                       # defaults
python -m fraud_detect.nodes.fabrication_eval --sample 200          # first 200 genuine dataset documents
python -m fraud_detect.nodes.fabrication_eval --dataset ~/Downloads/statement_forgery_dataset
python -m fraud_detect.nodes.fabrication_eval --statements ~/Downloads   # folder with statement_E/F.pdf
python -m fraud_detect.nodes.fabrication_eval --generator-count 0        # skip generator statements
```

It counts how many genuine documents fire each rule, shows the share of documents
carrying an invoice reference, and prints risk score, band, routing and codes for
`statement_E` and `statement_F`. It is read-only: it writes nothing to
`data/cases.json` or the audit chain (the full pipeline run on E/F may still append
audit events). A missing folder is skipped, not an error. Results are recorded in
`.kiro/specs/node3-fabrication-rules/evaluation-results.md`.

### Structuring calibration

`FL_STRUCTURING` (HIGH, weight 20) used to fire on any 3 near-threshold credits
anywhere in a statement. It now needs **3 near-threshold credits within 14 days**
of each other. `statement_F`, a genuine file, had 3 such credits on 28 Jul, 1 Aug
and 28 Aug; it raised a false alarm before and no longer does. The trade-off:
**slow structuring spread over weeks is missed**, for example one deposit every
3–4 weeks. Undated near-threshold credits count as inside any window.

### Honest limits

- Only **two labelled files** exist (`statement_E` tampered, `statement_F`
  genuine). Everything here is indicative.
- The Rule C jump ratio default (1.2) was **informed by both files**, so E and F
  separating under it is a sanity check, not proof. F's highest ratio is 1.14x and
  E's lowest jump is 1.25x, a thin gap.
- A fake with a **brand-new invoice number** evades the duplicate rule.
- **Genuine instalments can look like duplicates.** The evidence says so when it
  can.
- About **half of the rows the duplicate rule names are genuine** (the twins the
  fakes copied). On E: 24 rows named, 12 fake and 12 genuine. The rule cannot tell
  which of a pair was copied.
- The order hint names only **9 of the 12 fakes** on `statement_E`; it stays silent
  when it cannot single out one row.
- **No finding means no signal, not verified.** The 1,000-document dataset has no
  invoice references and is single-month, so zero firings there prove nothing.
- Weights were set from first principles, not fitted. **Node 4 (risk synthesis)
  weights and thresholds were NOT changed.**
- Findings that cover several rows appear in the findings list, but are **not
  highlighted on a table row** in the reviewer UI.

### Results (indicative)

| Measure | Before | After |
|---------|--------|-------|
| `statement_E` (tampered) risk | 26 | 55, HIGH, `HUMAN_REVIEW` |
| `statement_F` (genuine) risk | 40 | 26, MEDIUM, `HUMAN_REVIEW` |
| Clean synthetic sample risk | n/a | 1, `AUTO_CLEAR` |
| Genuine dataset documents firing a new rule | n/a | 0 of 1,000 (no signal, see limits) |
| Generator statements firing a new rule | n/a | 0 of 30 (no signal, see limits) |

`statement_F` still goes to `HUMAN_REVIEW` because of `FL_UEN_NOT_FOUND` and
`VF_CURRENCY_MIX`, both MEDIUM. Nothing was tuned after seeing these figures.

### Known follow-ups

- The two Node 2 gaps found on `statement_E` (a metadata check that missed it, and a
  currency warning that fired on a whole layout) are fixed. See
  [Node 2 checks](#node-2-checks-metadata-and-currency).

---

## Company registry lookup (UEN cross-check)

Node 3 checks the UEN that Node 1 read against a company registry
(`fraud_detect/registry.py`, function `lookup_entity`). There are two sources, asked
in this order:

1. **Local registry** (`ACRA_REGISTRY` in `fraud_detect/reference_data.py`). Always on.
   It holds the fictional demo and test companies, so those never need the network.
   Local entries win.
2. **Live registry** (data.gov.sg, "Registered Entities with Unique Entity Number",
   published by ACRA). **On while the web apps run** (`webapp.py` and
   `customer_app.py`), so real companies can be shown end to end. It is **off** in
   the test suite, in `demo.py` and when you call the code as a library. Override it
   either way with an environment variable, which always wins:

   ```bash
   export ACRA_LIVE_LOOKUP=1   # force on (also for demo.py and scripts)
   export ACRA_LIVE_LOOKUP=0   # force off, even for the web apps (fully offline demo)
   ```

   Only the UEN (a public identifier) is sent. No statement text, name or amount
   leaves the machine. The request is verified HTTPS using the `certifi` CA bundle.

What you get back:

| Outcome | Finding |
|---------|---------|
| UEN found, holder name matches | none. Names are compared ignoring case, punctuation and spacing (`PTE. LTD.` = `Pte Ltd`). |
| UEN found, holder name differs | `FL_UEN_NAME_MISMATCH` (HIGH, 22) |
| Every enabled source answered and none knows the UEN | `FL_UEN_NOT_FOUND` (MEDIUM, 10) |
| Live source enabled but could not answer (network error, timeout, bad reply) | `DQ_REGISTRY_UNAVAILABLE` (LOW, 2). This says nothing about the company and is never reported as "not found". |

Answers are cached in memory (found 24 h, not found 1 h, unavailable 1 min).

**Things to know**
- The test statements use **fictional companies**, so the real registry does not know
  them. That is why they are also listed locally.
- The dataset is published monthly, so a company registered this month may not be in
  it yet.
- The live record also carries a status (for example `Deregistered`). It is not used
  for a finding yet; a deregistered company submitting a statement is a sensible
  follow-up rule.
- Tests never call the network (a fake service stands in). To try the real service:
  `RUN_LIVE_TESTS=1 python -m pytest tests/test_registry.py`.

---

## Node 2 checks: metadata and currency

Two Node 2 checks were reworked after `.kiro/specs/node3-fabrication-rules/investigation.md`
showed they carried no signal on the test data.

**Currency notation (`VF_CURRENCY_MIX`, MEDIUM, 10).** A forger who retypes an amount
may write it differently from the bank (`SGD 1,234.00` among `S$` rows). The check now
looks only at the amounts in the **transaction rows**. The account-header label
(`Currency SGD`) and the summary box are labels, not amounts, so they are no longer
counted. Before the change it fired on every document of one bank layout, genuine or
tampered (200 of 200 each); now it fires on 0 of the 2,000 dataset documents. It still
catches the generator's planted mixed-notation case.

**File changed after it was created (`VF_MODIFIED_AFTER_CREATION`, MEDIUM, 14).** A bank's
statement system writes the PDF once, so its "last modified" time equals its "created"
time. The new check fires when the file was modified **more than 60 seconds** after it
was created **and** the program that made it is not a print or export tool. A plain
re-save through print-to-PDF or "save as PDF" is not flagged (64 genuine re-saved
dataset documents are the test for that). The time of day is read, so a same-day edit
is caught. It does not depend on knowing the editor's name, which is how
`statement_E` ("OpenPage Writer 3.8") is caught.

**Editing software (`VF_EDIT_SOFTWARE`, HIGH, 24).** The list of named tools grew from
image editors to common PDF editors (Foxit, PDF-XChange, Sejda, PDFescape, Smallpdf,
iLovePDF, PDFelement, Nitro) and any producer with the whole word "editor". Adobe's PDF
libraries are deliberately not listed because real banks use them. When this fires, the
modified-after-created finding is not added on top.

Measured on the synthetic dataset (indicative only):

| Group | Documents | `VF_CURRENCY_MIX` | modified-after-created | editing software |
|-------|-----------|-------------------|------------------------|------------------|
| Genuine | 936 | 0 | 0 | 0 |
| Genuine, re-saved by a print/export tool | 64 | 0 | 0 | 0 |
| Tampered: metadata changed | 200 | 0 | 152 | 48 |
| Tampered: other types (800) | 800 | 0 | 286 | 55 |

`statement_F` now has no Node 2 findings; `statement_E` gets the
modified-after-created finding.

**Limits.** The dataset's producer names are invented, so real print drivers and real
editors may behave differently; the lists of re-save tools and editors would need
checking against real files. More than half of the other tampered documents (459 of 800)
leave the metadata untouched, and this check cannot see those.

---

## Synthetic corpus generator

A small tool that makes **fictional** company bank statements as PDFs, in the six
layouts Node 1 already reads, and writes the "answer key" for each one: which rows
are fake, what was changed, how, and in which layout. It can make clean ("genuine")
statements and 16 kinds of tampered ones. A second command then feeds the PDFs
through Nodes 1 to 4 and reports what was read and what was flagged. Use it to
test and demo the pipeline on more data than the hand-made samples.

**All data is synthetic.** Companies, accounts, counterparties and amounts are made
up. The generated corpus is **never committed**: the default output folder
`data/corpus` is under `data/`, which is git-ignored.

### Build a corpus

```bash
python -m fraud_detect.corpus.build --out data/corpus --seed 20260601
```

| Option | Meaning (default) |
|--------|-------------------|
| `--out` | Output folder (`data/corpus`) |
| `--seed` | Master seed (`20260601`) |
| `--per-layout-genuine` | Genuine documents per layout (`2`) |
| `--per-layout-tampered` | Tampered documents per layout (`2`) |
| `--months-min` / `--months-max` | Fewest / most months per statement, 1 to 24 |
| `--hold-out-layout LAYOUT` | Keep one layout for the `test_unseen_layout` split |
| `--tamper-mix TYPE=WEIGHT,...` | Tamper types by weight, e.g. `metadata_inconsistency=1,font_mismatch=2` (even mix) |
| `--split-fractions TRAIN,VAL,TEST` | Split sizes, must sum to 1 (`0.70,0.15,0.15`) |
| `--force` | Build into a non-empty folder (existing files are never deleted) |

Run `python -m fraud_detect.corpus.build --help` for the full list of tamper type names.

**Output folder**

- `pdfs/<doc_id>.pdf`: the statements
- `manifests/<doc_id>.json`: per-document answer key (tamper type, method, changed rows)
- `labels.csv`: one row per document (layout, tampered or not, tamper type, split)
- `rows.csv`: one row per statement line, with the fake ones marked
- `companies.json`: the fictional companies
- `corpus_manifest.json`: settings, counts and generator version
- `README.txt`: a plain summary of the build, with any warnings

**Reproducible by seed.** The same options and `--seed` give the same corpus. A
different seed gives a different one.

### Verify a corpus

```bash
python -m fraud_detect.corpus.verify --corpus data/corpus --sample 5 \
    --json data/corpus/report.json --markdown data/corpus/report.md
```

`--sample N` measures at most N documents per layout (faster). `--json` and
`--markdown` write the report to those files; nothing else is written, and the
corpus is only read. The report covers:

- **Node 1:** share of rows read (genuine and tampered separately), share where
  the replayed balance equals the printed closing balance, share with matching
  header, the layout profile chosen, finding codes, and overlay rows dropped.
- **Nodes 2 to 4:** false alarms on genuine documents, detection per tamper type,
  blind spots, figures per layout and per split, decision breakdown, and
  follow-up suggestions (text only).

Screening runs in memory with no case store, no audit chain and the live registry
lookup off.

### The six layouts

`generator_v1`, `two_col_dmon`, `two_col_slash`, `two_col_dual_date`,
`signed_amount_iso`, `two_col_ddmmyy_cr` (see
[The six layout profiles](#the-six-layout-profiles)).

### The 16 tamper types

| Type | Group | What it does |
|------|-------|--------------|
| `edited_amount` | Document | One amount changed; balances recomputed or left as printed |
| `altered_date` | Document | One date changed |
| `broken_balance` | Document | One printed balance changed, amounts untouched |
| `font_mismatch` | Document | The edited amount is drawn in Courier |
| `metadata_inconsistency` | Document | Editor-style Producer and later ModDate |
| `genuine_resaved` | Hard negative | Print/export Producer and later ModDate; **not** tampered |
| `fabricated_receipts` | Content | 6 to 12 extra credits copying an existing receipt |
| `cross_month_copy` | Content | Receipt copy landing in a different month |
| `turnover_inflation` | Content | Extra credits with new invoice numbers in 2+ months |
| `structuring` | Behaviour | 3 to 5 credits of 9,500 to 9,999.99 within 14 days |
| `sudden_large_deposit` | Behaviour | One credit far above the average monthly credit |
| `shell_counterparty` | Behaviour | One credit of 50,000+ from a name like UNKNOWN TRADING LLC |
| `rapid_pass_through` | Behaviour | Money in, almost the same out within 1 to 3 days (expected blind spot) |
| `dormant_reactivation` | Behaviour | Months of only bank charges, then large activity (expected blind spot) |
| `period_gap` | Behaviour | One middle month removed |
| `uen_mismatch` | Behaviour | Printed holder name differs from the registered name |

### Known limits and blind spots

Results are indicative only: small samples, our own tamper designs, and a
generator and detectors written by the same team (possible circularity). Some
tampers are not caught by any rule (for example `rapid_pass_through`). Document
level only; row-level accuracy is not measured. See
[`evaluation-results.md`](.kiro/specs/synthetic-corpus-generator/evaluation-results.md)
for the measured numbers and the full list.

### What is unchanged

Nodes 1 to 4 and the reference data are **unchanged**. The fictional companies are
registered only temporarily, in memory, while `verify` measures, and are not
written anywhere in the repo.

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
