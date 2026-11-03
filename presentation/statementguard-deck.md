---
marp: true
theme: default
paginate: true
title: StatementGuard — Project Presentation
---

<!-- _class: lead -->

# StatementGuard

**AI screening of corporate-loan bank statements for _tampering_ — before a credit analyst ever sees them.**

A multi-agent pipeline that auto-clears clean files, routes suspicious ones to a human, and attaches readable evidence to every flag.

Built to MAS **FEAT** principles — the machine never approves or rejects a customer on its own.

`github.com/elline-c/AI-in-finance-` · ~25.7k lines of Python · 71 test modules

---

## The problem

**Today:** Tampered statements slip into credit assessment because catching them relies on slow, inconsistent **manual review**. Fakes that look plausible get through.

**What we need:**
- Screen **every** submitted statement automatically
- **Auto-clear** the clean ones so analysts focus only on flagged files
- Attach **explainable evidence** to each flag

> The guiding rule: the system only gathers evidence and, at worst, routes a case to a human — **it never rejects a customer automatically.**

---

## Designed for governance (MAS FEAT)

| Principle | How StatementGuard honours it |
|---|---|
| **Fairness** | No protected attributes in scoring; flag rates tested across segments |
| **Ethics** | Fail-safe: poor data or low confidence escalates — never auto-clears |
| **Accountability** | Every AI output and human decision recorded to an audit log |
| **Transparency** | Every flag carries a stable reason code + plain-English evidence |

Also: **PDPA** (synthetic-only data; encryption + access control in production) · **MAS TRM** (immutable signed audit chain) · human owns the final call.

---

## The pipeline at a glance

```
📄 PDF ─► Node 1 Ingestion ─► Node 2 Visual Forensics ─► Node 3 Financial Logic ─► Node 4 Risk Synthesis
                │                                                                        │
       Security layer (prompt-injection guard)                      risk + confidence ───┤
       Node 5 Cross-case memory                                                          ▼
       Node 6a GNN transaction / Node 6b ML statement forgery          ┌──── routing ────┐
                                                                 AUTO_CLEAR          HUMAN_REVIEW
                                                        (low risk, confident,   (any material finding,
                                                         no material findings)   low confidence, poor data)
```

All agents feed **Node 4**. Independent agents that agree raise a corroboration bonus.

---

## Node 1 — Ingestion (the mailroom)

- Extracts text, hidden **metadata** (author, software, dates), **font fingerprints**, header (holder, UEN, period, balances) and the transaction table.
- Missing/unreadable fields are reported **honestly** as data-quality findings — never guessed.
- Tries **six layout profiles**; best page coverage wins; ties that disagree are flagged.
- A **self-check** compares printed section totals with parsed rows.

**Synthetic eval:** 100% rows read & balances replayed across all layouts · longest file 0.36s.

---

## Node 2 — Visual Forensics (the document examiner)

Signs the file was *edited*, independent of the numbers:

| Signal | Reason code | Severity |
|---|---|---|
| Localised font mismatch (overwritten number) | `VF_FONT_MISMATCH` | MED |
| Mixed date / currency notation in rows | `VF_DATE_FORMAT_MIX` · `VF_CURRENCY_MIX` | MED |
| Image/PDF editor in metadata | `VF_EDIT_SOFTWARE` | HIGH |
| Modified >60s after creation by non-export tool | `VF_MODIFIED_AFTER_CREATION` | MED |
| Creation date before claimed period | `VF_CREATION_BEFORE_PERIOD` | MED |

Adobe's PDF libraries are deliberately **not** flagged (real banks use them). Currency check tuned from 200/200 false fires on one layout to **0 / 2,000**.

---

## Node 3 — Financial Logic (the accountant)

**Core checks:** running-balance reconciliation · closing-balance match · period-continuity gaps · sudden deposits vs baseline · unverified counterparties · structuring (3+ near-threshold credits within 14 days) · **UEN ↔ ACRA** cross-check.

**Fabrication rules (income faking):** duplicate invoice refs in one month · invoice numbers out of date order · turnover jumps while payroll/CPF/rent stay flat.

Live UEN cross-check against data.gov.sg (ACRA) — only the public UEN leaves the machine, over verified HTTPS, cached; fully overridable for offline demos. Every finding cites evidence; none is proof.

---

## Nodes 5 & 6 — Memory & Machine Learning

- **Node 5 — Cross-case memory:** matches suspicious rows against prior **human-decided** cases; always cites the prior case. Learns only from human-verified decisions.
- **Node 6b — ML statement forgery (active):** scores the whole document from fonts, editor traces and balance math; **abstains** on documents larger than its training.
- **Node 6a — GNN transaction (GNN on, GBM off):** graph neural network catches relational AML patterns (round-tripping, mule fan-in/out) via offline-precomputed score lookup. Row-level GBM off by default (false alarms on clean business files).
- **Security layer:** scans for prompt-injection; **never obeys** — files the attempt itself as evidence.

All ML nodes are **fail-safe**: a missing model yields no findings, never a crash.

---

## Node 4 — Risk & Synthesis (the senior officer)

| Risk score | Band | | Confidence |
|---|---|---|---|
| 1–25 | Low | | Starts high for a clean digital PDF |
| 26–50 | Medium | | Lowered by data-quality gaps, thin/single-dimension evidence |
| 51–75 | High | | Raised by corroboration |
| 76–100 | Very High | | |

**Routing**
- `AUTO_CLEAR` only when risk ≤ 25 **and** confidence ≥ 60 **and** no material findings.
- `HUMAN_REVIEW` otherwise — including low confidence or poor data, even if risk looks low.

> The ceiling is always "send to a human." **The machine never rejects a customer.**

---

## Three roles, separated by design

- **👤 Customer portal** (`customer_app.py` :8001) — apply, upload, track. Sees **only neutral status** (Verified / Being reviewed / Action needed). Never sees scores or evidence.
- **🔍 Fraud analyst** (`webapp.py` :8000) — screening queue, case overview, flagged rows, per-stage view, PDF. Request info / Approve / Reject, each with a mandatory reason.
- **🏦 Loan officer** (`banker_app.py` :8002) — only sees **fraud-cleared** cases; makes the lending call; **cannot override** a fraud outcome.

All three share one case store & audit log. The application reference number *is* the case id.

---

## Live demonstration

```bash
# generate synthetic samples (clean + deliberately tampered)
python -m fraud_detect.synthetic

# run the full pipeline end-to-end
python demo.py

# or run any statement through the CLI
python3 -m statement_guard.cli samples/clean_statement.json
python3 -m statement_guard.cli samples/tampered_statement.json

# the three web apps (shared case store)
uvicorn fraud_detect.customer_app:app --port 8001   # applicant
uvicorn fraud_detect.webapp:app        --port 8000   # analyst
uvicorn fraud_detect.banker_app:app    --port 8002   # loan officer
```

Core pipeline has **no external dependencies** — pure Python 3.

---

## Demo A — a clean statement

```
Node 1 Ingestion ........ OK   balances reconcile, all fields present
Node 2 Forensics ........ OK   no editor traces, consistent fonts/formats
Node 3 Financial ........ OK   running balance replays row-by-row
Node 6b ML forgery ...... OK   low forgery likelihood
------------------------------------------------------------
Risk score ......... 1 / 100  (Low)
Confidence ......... High
Decision ........... ✅ AUTO_CLEAR → approval recommendation
```

Clean + confident + no material findings → the analyst never touches it.

---

## Demo B — a tampered statement

```
Node 1 Ingestion ........ 1 data-quality note
Node 2 Forensics ........ VF_EDIT_SOFTWARE (OpenPage Writer 3.8)
                          VF_MODIFIED_AFTER_CREATION
Node 3 Financial ........ FL_BALANCE_BREAK  row 14 doesn't reconcile
                          FL_STRUCTURING    3 credits @ ~$9.8k / 14 days
                          FL_UEN_NAME_MISMATCH
Node 6b ML forgery ...... elevated likelihood
------------------------------------------------------------
Risk score ......... High  (+corroboration: visual AND financial agree)
Confidence ......... lowered by data-quality gap
Decision ........... 🧑‍⚖️ HUMAN_REVIEW — 10 pieces of evidence attached
```

Every flag carries a reason code + the raw data it came from — fully auditable.

---

## Every flag is auditable evidence

```json
{
  "reason_code": "FL_BALANCE_BREAK",
  "severity":    "HIGH",
  "stage":       "financial_logic",
  "explanation": "Running balance at row 14 is 48,210.00 but the printed balance is 58,210.00 — a 10,000.00 break.",
  "evidence": { "row": 14, "printed_balance": 58210.00, "computed_balance": 48210.00, "delta": 10000.00 }
}
```

Findings are typed, validated objects — the analyst sees *which stage fired, on which row, and why*. Missing fields become explicit nulls, never hallucinated.

---

## Does it actually catch fakes?

| Planted tamper | Detected by |
|---|---|
| Altered balance / number | `FL_BALANCE_BREAK` · `VF_FONT_MISMATCH` |
| Format inconsistency | `VF_DATE_FORMAT_MIX` · `VF_CURRENCY_MIX` |
| Metadata mismatch | `VF_EDIT_SOFTWARE` · `VF_CREATION_BEFORE_PERIOD` |
| Period gap · suspicious txn · structuring | `FL_PERIOD_GAP` · `FL_SUDDEN_DEPOSIT` · `FL_STRUCTURING` |
| UEN mismatch | `FL_UEN_NAME_MISMATCH` |

**71** test modules · **13** detection tests (every planted issue caught, clean auto-clears, nothing auto-rejects) · **0 / 2000** false fires after tuning.

---

## GNN evaluation — honest, measured

**Where it adds value:** on round-tripping accounts the GNN catches **5 that rules miss**, at the live threshold with **no false alarms**. GNN + Node 3 together catch 19 / 27 fraud accounts (70%) vs 52% rules-alone.

**Stated limits:**
- Narrow — covers 1 of 5 AML typologies; scores only 62 / 1,201 accounts
- Live path is account-level only — offline numbers are an **upper bound**
- Synthetic data → graph lift may shrink on real data
- Only 27 fraud test accounts — wide confidence intervals

> "A green build is not success." We separate measured results from aspiration.

---

## What's real vs. simulated

**✅ Done & working**
- The entire 4-agent decision engine, end-to-end
- Scoring, confidence, routing, corroboration
- Readable evidence on every flag
- 3 web apps · cross-case memory · ML nodes
- Immutable signed audit chain · test suite

**🟠 Simulated / remaining**
- Real PDF + OCR reading (input is typed data today)
- Real pixel/layer image forensics (metadata + rules now)
- Historical baselines from real cash-flow data
- Production DB, integration, security hardening

> The *brain* is complete; real-world inputs and production hardening remain.

---

## Roadmap — 6 staged phases

| Phase | In five words |
|---|---|
| 0 · Foundations | Gather examples, agree the goal |
| 1 · Harden core | Make the prototype sturdy |
| 2 · Real ingestion | Read real PDF statements (OCR) |
| 3 · Real forensics | Detect real image tampering |
| 4 · Analyst UI | Build the staff review screen |
| 5 · Shadow pilot | Safely test alongside humans |
| 6 · Rollout | Turn it on gradually |

Recommended strategy: **hybrid** — buy OCR + pixel forensics, build the orchestration, financial-logic, scoring & audit layer we own.

---

## How we'll measure success

- **Recall on tampered docs** — the catch rate that protects the bank
- **Auto-clear precision** — protects analyst time & customers
- **% auto-cleared** — the efficiency win (only counts if precision holds)
- **Time-to-decision** vs. the current manual baseline
- **Analyst override rate** — calibration signal
- **Explainability** — % of flags analysts rate "clear evidence" (FEAT)

Targets are set on a held-out, labelled eval set — the single most valuable asset.

---

<!-- _class: lead -->

# Computer flags. Human decides.

- A working, end-to-end **multi-agent screening pipeline** for tampered bank statements
- Explainable evidence on every flag; **never** auto-rejects — MAS FEAT by design
- Three separated roles, an immutable audit trail, honest real-vs-simulated
- A clear, staged path from prototype to production

**Thank you — questions & live demo welcome.**

---

<!-- Notes -->

*Sources drawn from the project's own documentation: `README.md`, `PROJECT_PLAN.md`, `WORKFLOW.md`, `STATEMENTGUARD_README.md`, `FRAUD_DETECT_README.md`, `CUSTOMER_PORTAL_README.md`, `BANKER_PORTAL_README.md`, and `fraud_detect/ml/gnn/EVALUATION.md`.*
