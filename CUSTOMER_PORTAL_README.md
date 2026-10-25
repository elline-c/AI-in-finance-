# Customer Portal — the applicant's view

A customer-facing loan-application portal for the bank-statement screening system.
It is the **applicant's** side of the product, distinct from the analyst review UI
(`fraud_detect/webapp.py`).

Where the analyst app shows risk scores, confidence scores, forensic evidence and a
reject/approve control, the customer portal deliberately shows **none** of that. A
business applying for a corporate loan only ever sees neutral, plain-language status.

## What a customer can do

| Page | Route | Purpose |
|------|-------|---------|
| **Apply** | `GET /` → `POST /apply` | Enter company name, UEN, contact email, optional loan amount, and upload a bank-statement PDF. |
| **Confirmation** | `GET /submitted/{ref}` | Shows the application reference number, a 3-step progress tracker, and what happens next. |
| **Track** | `GET /status?ref=...` | Look up an application by reference and see its current status. |

The application **reference number is the case id**, so the same submission appears
in the analyst dashboard automatically (both apps share one `CaseStore`).

## Privacy & FEAT stance (why the customer sees so little)

The customer is **never** shown the internal risk score, confidence score, forensic
findings, reason codes, or the raw `AUTO_CLEAR` / `HUMAN_REVIEW` routing. Doing so
would tip off bad actors and would misrepresent an internal screening signal as a
decision. Internal state is mapped to four neutral customer statuses:

| Internal state | Customer sees |
|----------------|---------------|
| `AUTO_CLEAR` (no human decision yet) | **Verified** — passed automated checks, moving forward |
| `HUMAN_REVIEW` (no human decision yet) | **Being reviewed** — routine specialist review, no action needed (never "suspected tampering") |
| Human `APPROVED` | **Verified** |
| Human `REJECTED` | **More information needed** — neutral request to re-submit / contact us (never a fraud accusation) |

The machine never approves or rejects the customer; a person always makes the final
call. This mirrors the MAS **FEAT** stance the rest of the system is built around.

## Run it

```bash
pip install -r requirements.txt
python -m fraud_detect.synthetic          # generate sample PDFs (first run only)
uvicorn fraud_detect.customer_app:app --port 8001
# open http://127.0.0.1:8001/
```

Run it alongside the analyst app (`uvicorn fraud_detect.webapp:app --port 8000`) to
see both views of the same submission.

## Files

- `fraud_detect/customer_app.py` — the FastAPI portal (upload hardening + text
  sanitising reuse the same guards as `webapp.py`; the pipeline and store are shared).
- `fraud_detect/templates_customer/` — `apply.html`, `submitted.html`, `status.html`,
  a shared `base.html`, and the `_status_block.html` partial.
