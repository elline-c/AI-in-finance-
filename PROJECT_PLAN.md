# StatementGuard — Project Plan

How we take the prototype in this repo to a production-grade system that screens
corporate-loan bank statements for tampering before a credit analyst reviews them.

This plan is deliberately staged so that **each phase ships something usable** and
de-risks the next. It is written to satisfy MAS **FEAT** (Fairness, Ethics,
Accountability, Transparency) from day one, not as an afterthought.

---

## 0. What's already built (the prototype) ✅

A **working proof-of-concept** already lives in this repo. It runs today with one
command (`python3 -m statement_guard.cli --all`) and needs no installation — pure
Python. It is a **finished prototype**, not a finished product.

**What is DONE and works:**

- **The full 4-agent pipeline**, wired together and running end-to-end:
  - **Node 1 — Ingestion:** takes a statement, enforces a strict data format,
    returns explicit "missing" flags instead of guessing, and scores data quality
    (OCR confidence, unreadable pages, bad date/number formats).
  - **Node 2 — Visual Forensics:** flags editing-software traces (e.g. Photoshop),
    files modified after creation, dates that don't match the statement period,
    altered-text hints, and irrelevant/mismatched supporting documents.
  - **Node 3 — Financial Logic:** checks running balances add up row by row,
    reconciles the statement total, checks the dates flow correctly, and detects
    structuring (deposits just under limits), sudden abnormal deposits, and rapid
    in-then-out "pass-through" money movements.
  - **Node 4 — Risk & Synthesis:** combines everything into a **risk score
    (1–100, banded Low / Medium / High / Very High)** and an **inverted confidence
    score (High / Medium / Low / Very Low)**.
- **The decision & routing logic:** clean + confident files are **auto-cleared**;
  anything suspicious, low-confidence, or poor-quality is **routed to a human**.
  The system **never approves or rejects a customer on its own.**
- **Readable evidence on every flag** (MAS FEAT): each finding has a plain-English
  explanation plus the raw data it came from — fully auditable.
- **A command-line runner and a readable report**, plus a machine-readable JSON
  report for downstream systems.
- **3 sample statements** proving it works: a clean one (auto-cleared), a tampered
  one (10 pieces of evidence, routed to human), and a poor-quality scan (routed on
  data-quality grounds).

**What is SIMULATED or NOT yet built** (this is what the phases below deliver):

- Reading **real PDF files** with real OCR — currently the input is typed-out data.
- **Real image forensics** (pixel/layer analysis) — currently done with metadata +
  rules, not true image inspection.
- The **analyst review screen** (human-in-the-loop UI) and the **feedback loop**.
- **Historical baselines** (comparing to expected cash flow / known vendors),
  database, integration, tests, security, and deployment.

> In short: **the entire "brain" (decision engine) is complete; the real-world
> inputs, the human review screen, and production hardening are what remain.**

---

## 1. Guiding principles (non-negotiables)

1. **Human owns the decision.** The system auto-clears clean files and escalates
   everything else. It never rejects or approves a customer autonomously.
2. **Every flag carries evidence.** No score without a readable, auditable "why".
   (Already true in the prototype: each finding has severity + explanation + raw
   evidence.)
3. **Explainable over clever.** Prefer deterministic, inspectable checks. Use ML
   only where it clearly beats rules, and always wrap ML output in an explanation.
4. **Fail safe, not silent.** Poor data quality / low model confidence → escalate,
   never auto-clear.
5. **Privacy & retention by design.** Statements are sensitive PII/financial data;
   scope access, encrypt, and set retention from the start.

---

## 2. Build vs. buy vs. hybrid (decide first)

The market already has forensics engines (Resistant AI, Inscribe, Ocrolus, ABBYY,
Docsumo). This is a real decision, not a formality:

| Option | Pros | Cons | When it wins |
|--------|------|------|--------------|
| **Buy** a forensics API | Fast, battle-tested models, maintained | Recurring cost, less control, data leaves premises | Speed to market, small team |
| **Build** everything | Full control, IP, data stays in-house | Slow, needs ML + forensics expertise | Strong differentiation / data-residency needs |
| **Hybrid** (recommended) | Buy OCR + pixel forensics; **build the orchestration, financial-logic, scoring, routing, audit layer** | Integration work | Most teams — you own the parts that encode *your* credit policy |

> **Recommendation:** start **hybrid**. The prototype in this repo already *is* the
> orchestration + financial-logic + scoring + audit layer we'd own. Treat OCR and
> pixel-level image forensics as pluggable providers behind an interface, so we can
> swap a vendor for an in-house model later without touching the pipeline.

**Decision needed from stakeholders before Phase 1:** budget for a forensics vendor
vs. in-house ML headcount, and any data-residency constraint (does customer data
have to stay on our infra?).

---

## 3. Architecture target

```
          Loan Origination System
                    │  (statement + supporting docs)
                    ▼
        ┌───────────────────────┐
        │  Intake API / queue    │   auth, virus scan, dedupe, store raw
        └───────────┬───────────┘
                    ▼
   ┌────────────────────────────────────────────┐
   │            Screening Pipeline                │
   │  N1 Ingestion → N2 Forensics → N3 Financial  │   ← this repo, hardened
   │           → N4 Risk & Synthesis              │
   └───────────────┬──────────────────────────────┘
                   ▼
        risk band + confidence + evidence
         ├── AUTO-CLEAR → recommendation summary → back to LOS
         └── ESCALATE   → Analyst Review UI (queue)
                              │
                              ▼
                   Analyst decision + reason
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              Feedback store      Audit log (immutable)
              (tune thresholds)   (FEAT accountability)
```

**Interfaces to define early** (so components stay swappable):
- `OcrProvider.extract(pdf) -> NormalizedStatement`  (vendor or in-house)
- `ImageForensics.inspect(pages) -> list[Finding]`   (vendor or in-house)
- `Pipeline.screen(case) -> Report`                  (owned — already exists)
- `ReviewQueue` / `FeedbackStore` / `AuditLog`

---

## 4. The 6 phases in plain English (quick overview)

If the detailed version below is too technical, here is the whole roadmap in
everyday language:

| Phase | In plain words | In 5 words |
|-------|----------------|------------|
| **0 — Get ready** 📋 | Gather test examples (some clean statements, some made-up tampered ones) and agree what "success" looks like. Pack your bags before the trip. | Gather examples, agree the goal |
| **1 — Tidy up what we have** 🧹 | Clean up and strengthen the prototype we already built so it doesn't break. No new features yet — just make it solid. | Make the prototype sturdy |
| **2 — Read real documents** 📄 | Teach it to read an actual PDF bank statement by itself (scanner + reader), instead of being handed typed-out data. | Read real PDF statements |
| **3 — Spot real fakes** 🔍 | Give it the ability to truly inspect the document image and detect tampering — edited numbers, Photoshopped bits, copied logos. | Detect real image tampering |
| **4 — Build the staff screen** 🖥️ | Create the web page where a bank officer sees flagged statements, reads the evidence, and clicks approve or reject. Computer flags; human decides. | Build the staff review screen |
| **5 — Test quietly alongside humans** 🧪 | Run it next to the current manual process without affecting real decisions, and check it catches what humans catch. A safe trial run. | Safely test alongside humans |
| **6 — Switch on gradually** 🚀 | Once trusted, let it auto-clear the obviously clean ones to save time — carefully, a bit at a time — while watching for new fraud tricks. | Turn it on gradually |

> Everyday test data note: you do **not** need real customer bank statements —
> **made-up sample statements (clean + deliberately tampered) are the right way to
> start** and can be created today.

The same phases with full technical detail, time estimates, and deliverables:

## 4a. Delivery phases (detailed)

### Phase 0 — Foundations & decisions (1–2 weeks)
- Confirm build/buy/hybrid + vendor shortlist; sign data-processing terms if buying.
- Data governance: classify data, define retention (statements are 6–12 months old
  per policy; decide how long we keep them post-decision), access control, encryption.
- Assemble a **labelled evaluation set**: genuine statements + known-tampered
  examples across banks/formats. This is the single most valuable asset — everything
  is measured against it.
- Define success metrics (see §6).
- **Deliverable:** signed-off scope, data-handling policy, eval dataset v0.

### Phase 1 — Harden the pipeline core (2–3 weeks)
_Start from the prototype in this repo._
- Add a test suite (unit tests per node + golden-file pipeline tests on the eval set).
- Formalise the input contract as a versioned JSON Schema; validate at the boundary.
- Config-drive the thresholds (structuring band, σ multiplier, gap days, auto-clear
  ceiling) instead of hard-coding — analysts will want to tune these.
- Structured logging + a stable machine-readable report (already emitted as JSON).
- **Deliverable:** tested, configurable screening engine with CI.

### Phase 2 — Real ingestion (OCR + PDF/metadata) (3–4 weeks)
- Implement `OcrProvider` against the chosen vendor (or Textract/DocAI/ABBYY etc.),
  mapping their output to `NormalizedStatement`.
- Real PDF metadata extraction (author, creator/producer tool, creation/mod dates).
- Handle multi-page, multi-format, scanned vs. digital; carry OCR confidence through.
- **Deliverable:** upload a real PDF → get a real normalized statement + data-quality
  report. Node 1 stops being simulated.

### Phase 3 — Real visual forensics (3–5 weeks)
- Integrate pixel/layer/compression analysis (vendor API first; in-house model later).
- Keep the metadata + format heuristics we already have — they're cheap and explainable.
- Ensure every forensic signal returns a **human-readable finding**, not just a score.
- **Deliverable:** Node 2 detects real image manipulation with evidence.

### Phase 4 — Analyst Review UI + human-in-the-loop (3–4 weeks)
- Queue of escalated cases sorted by risk; each opens the statement, the evidence
  findings, and the supporting docs side by side.
- Analyst records **approve / reject + reason**; reason is mandatory and stored.
- Wire the feedback loop: analyst outcomes feed the feedback store for threshold
  tuning and (later) model retraining — this is the "let the AI know why" step.
- **Deliverable:** analysts work entirely in-tool; nothing auto-rejects.

### Phase 5 — Integration, pilot & hardening (3–4 weeks)
- Integrate with the loan-origination system (intake + return recommendation).
- Shadow mode: run alongside the current manual process, compare outcomes, no impact
  on live decisions. Measure precision/recall vs. analyst ground truth.
- Security review, load testing, DR/runbooks, monitoring & alerting.
- **Deliverable:** pilot in shadow mode with a measured accuracy report.

### Phase 6 — Controlled rollout (ongoing)
- Turn on auto-clear only above a confidence bar proven in the pilot; start
  conservative (escalate more), loosen as metrics justify.
- Monitor for drift (new statement formats, new tampering techniques); scheduled
  re-evaluation against a refreshed eval set.
- Model/threshold governance: version everything, log every decision, periodic FEAT
  review.

---

## 5. Workstreams (run in parallel, not strictly sequential)

- **Engineering:** pipeline, integrations, UI, infra.
- **Data science / forensics:** OCR + image models, threshold calibration, drift.
- **Risk & Compliance:** FEAT alignment, audit requirements, retention, sign-off gates.
- **Ops / Credit:** analyst workflow, training, feedback discipline.

---

## 6. Success metrics (define targets in Phase 0)

- **Recall on tampered docs** (catch rate) — the metric that protects the bank.
- **False-positive rate / auto-clear precision** — protects analyst time & customers.
- **% auto-cleared** — the efficiency win; only counts if precision stays high.
- **Time-to-decision** vs. current manual baseline.
- **Analyst override rate** — how often humans disagree with the risk band (calibration).
- **Explainability:** % of flags an analyst rates as "clear evidence" (FEAT transparency).

> A green build is not success. Success = these numbers on the held-out eval set.

---

## 7. Key risks & mitigations

| Risk | Mitigation |
|------|------------|
| Not enough tampered examples to evaluate | Curate + synthesise controlled tampering; partner with fraud team |
| Overfitting to today's forgery techniques | Keep rules explainable; scheduled drift review; refresh eval set |
| False positives frustrate analysts | Start conservative, tune thresholds from real overrides, show evidence |
| Vendor lock-in / data residency | Providers sit behind interfaces; keep orchestration + policy in-house |
| Bias / fairness (FEAT) | Test flag rates across customer segments; no protected attributes in scoring |
| Sensitive-data exposure | Encrypt at rest/in transit, least-privilege access, retention limits, audit log |

---

## 8. Immediate next steps (this week)

1. Stakeholders decide **build/buy/hybrid** and any data-residency constraint (§2).
2. Kick off assembling the **labelled eval dataset** (§4 Phase 0) — longest lead time.
3. Approve **Phase 1** (harden the existing prototype: tests, config, schema) — this
   is low-risk and can start immediately against the code already in this repo.
4. Agree the **success-metric targets** (§6) so "done" is measurable.

---

### Sources
- Industry pattern (forensics + data checks + cross-doc + route to reviewer with evidence):
  [Docsumo](https://www.docsumo.com/blog/document-fraud-detection-systems),
  [Inscribe](https://www.inscribe.ai/blog/thinking-of-buying-document-fraud-detection-software-heres-what-you-need-to-know),
  [KlearStack](https://klearstack.com/blogs/automated-document-tampering-detection).
- MAS FEAT principles (Fairness, Ethics, Accountability, Transparency):
  [MAS/OECD.ai](https://oecd.ai/en/dashboards/policy-initiatives/principles-to-promote-fairness-ethics-accountability-and-transparency-in-the-use-of-artificial-intelligence-and-data-analytics-in-singapores-financial-sector-3706).

_Content from external sources was rephrased for compliance with licensing restrictions._
