# UI demo — static, clickable snapshots

Fully-rendered snapshots of both user interfaces, generated from **real cases run
through the actual screening pipeline**. No server needed — open the files in a
browser and click through; the links work offline.

**Start here:** open [`index.html`](index.html).

## What's inside

### Customer portal (the loan applicant's view)
- `apply.html` — application form (company details + statement upload)
- `cust_submitted_clean.html` — confirmation for a clean statement → "verified"
- `cust_status_clean.html` — status lookup showing "verified"
- `cust_submitted_tampered.html` — confirmation for an escalated statement → "being reviewed"
- `cust_status_needinfo.html` — status after the analyst requested more info → neutral "action needed"
- `status_lookup.html` — track-by-reference form

### Analyst console (the credit reviewer's view)
- `dashboard.html` — screening queue with status badges
- `an_overview_*.html` — case overview: scores + findings-by-stage + decision panel
- `an_transactions_*.html` — flagged transactions, each row with its evidence
- `an_stages_*.html` — pipeline stages (what each agent produced)

(`*` is the case id — one clean case, one tampered case.)

## The scenario these snapshots capture

1. **Harbourline Logistics** uploads a clean statement → auto-cleared → customer sees **Verified**.
2. **Meridian Trading** uploads a tampered statement → the customer sees a calm **"being reviewed"**; the analyst sees the VERY_HIGH risk, the flagged rows, and every stage's evidence.
3. The analyst **requests more information** → the customer now sees a neutral **"action needed"** with the reviewer's message, and can re-submit.

Note the privacy boundary: the customer pages never show risk scores, forensic
evidence, reason codes, or any "fraud" language — only neutral status.

## Regenerating

```bash
.venv/bin/python build_ui_demo.py     # rewrites everything in ui_demo/
```

## Caveats (static demo)

Forms don't submit (buttons are inert) and the "Statement PDF" link points back to
the index. For the live, interactive apps:

```bash
uvicorn fraud_detect.customer_app:app --port 8001   # customer portal
uvicorn fraud_detect.webapp:app --port 8000         # analyst console
```
