# Approval console — the banker / loan-officer perspective

The **third** role in the system, alongside the customer portal
(`customer_app.py`) and the fraud-analyst console (`webapp.py`).

- **Customer** submits a statement and tracks status.
- **Fraud analyst** screens the document for tampering and clears or confirms the
  fraud concern.
- **Loan officer** (this app, `banker_app.py`) makes the **final lending
  decision** on applications the fraud screening has cleared.

## Separation of duties (the core control)

Screening decides whether a document can be **trusted**; the banker decides
whether to **lend**. These are deliberately separate roles:

- The approval queue only shows cases whose fraud concern is **cleared** — the
  analyst `APPROVED` the case, or it `AUTO_CLEAR`ed with no concern.
- A statement that is still under review, `NEEDS_INFO`, or `REJECTED` by the
  analyst **never** appears to the banker, and the app refuses to record a lending
  decision on such a case (HTTP 403 on view, 409 on decide).
- The banker **cannot override a fraud outcome** — it is a lending call, not a
  fraud call.

## What the loan officer sees

| Page | Route | Purpose |
|------|-------|---------|
| **Approval queue** | `GET /` | Fraud-cleared applications awaiting a decision, plus recent decisions. |
| **Application** | `GET /application/{id}` | Trust signal (fraud cleared + screening confidence), headline **financials** (opening/closing, total credits/debits, net change), and the decision form. |
| **Decide** | `POST /application/{id}/decide` | Record **Approve / Decline / Refer** with a mandatory rationale, optional approved amount + conditions. |

The banker view is credit-oriented: it surfaces the numbers a lender cares about
and shows the fraud result only as a trust signal — it does not re-expose the
forensic evidence (that is the analyst's screen).

## Data model

- `CreditDecision` enum: `PENDING / APPROVED / DECLINED / REFERRED`.
- `CreditRecord` on each `Case`: officer, rationale, approved amount, conditions,
  timestamp. Backward-compatible (older stored cases validate with a default).
- `CaseStore.fraud_is_cleared(case)`, `list_credit_queue()`, and
  `record_credit_decision(...)` (which enforces the clearance guardrail).

All lending decisions are written to the shared `audit_log` as `CREDIT_*` events.

## Run it

```bash
pip install -r requirements.txt
python -m fraud_detect.synthetic            # first run only: sample PDFs
uvicorn fraud_detect.banker_app:app --port 8002
# open http://127.0.0.1:8002/
```

Run it alongside the other two apps (all three share one case store):

```bash
uvicorn fraud_detect.customer_app:app --port 8001   # customer
uvicorn fraud_detect.webapp:app        --port 8000   # fraud analyst
uvicorn fraud_detect.banker_app:app    --port 8002   # loan officer
```

End-to-end flow: a customer submits a statement → the analyst clears the fraud
concern (or it auto-clears) → the case appears in the banker's approval queue →
the loan officer approves, declines, or refers it.

**Tests:** `tests/test_banker.py` covers the queue filtering, the view/decision
guardrails, the recorded decision + audit trail, and that a fraud rejection keeps
a case out of the banker stage entirely.
