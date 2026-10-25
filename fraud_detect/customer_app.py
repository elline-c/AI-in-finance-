"""Customer-facing loan-application portal (the applicant's perspective).

This is the **customer** side of the system — distinct from the analyst
review UI in ``webapp.py``. A business applying for a corporate loan uses it to:

    * start an application and upload their bank statement (PDF),
    * receive a reference number and a clear explanation of what happens next,
    * track the status of their application in plain language.

Privacy / FEAT stance (important — this is NOT the analyst view):
    * The customer is **never** shown the internal fraud *risk score*, the
      *confidence score*, the forensic *evidence*, reason codes, or the raw
      routing label. Exposing those would tip off bad actors and would
      misrepresent an internal screening signal as a decision.
    * The machine never approves or rejects the customer. The customer only ever
      sees neutral, human-centred statuses: *Received*, *Being verified*,
      *Verified*, *More information needed*, or *A specialist will contact you*.
    * A ``HUMAN_REVIEW`` routing is surfaced simply as "being reviewed by a
      specialist" — never as "suspected tampering".

It reuses the exact same screening ``pipeline`` and ``CaseStore`` as the analyst
app, so a single uploaded statement flows to both views consistently.
"""
from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .ml.gnn import gnn_lookup
from .models import Case, ReviewDecision, Routing
from .pipeline import run_pipeline
from .store import CaseStore

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE.parent / "data" / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Upload hardening (mirrors the analyst app — customers upload files too)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES = 25 * 1024 * 1024          # 25 MB ceiling (DoS guard)
_PDF_MAGIC = b"%PDF-"                         # a real PDF starts with this
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")  # everything else is stripped

# Length caps for customer free-text fields (XSS / log-injection guard).
_MAX_NAME = 160
_MAX_EMAIL = 160
_MAX_UEN = 40
_MAX_AMOUNT = 40
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _scrub(text: str, max_len: int) -> str:
    """Cap length, strip control chars, HTML-escape at the boundary."""
    text = (text or "")[:max_len]
    text = _CONTROL_RE.sub("", text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _safe_upload_destination(filename: str | None) -> Path:
    """Destination path that cannot escape UPLOAD_DIR (see webapp.py)."""
    raw = (filename or "statement.pdf").replace("\x00", "")
    base = Path(raw).name
    base = _SAFE_NAME_RE.sub("_", base) or "statement.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    dest = (UPLOAD_DIR / f"{secrets.token_hex(8)}_{base}").resolve()
    if UPLOAD_DIR not in dest.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    return dest


app = FastAPI(title="Corporate Loan Application Portal")
templates = Jinja2Templates(directory=str(BASE / "templates_customer"))
store = CaseStore(str(BASE.parent / "data" / "cases.json"))

# Load the precomputed GNN scores once at startup (fail-safe to None if absent).
GNN_SCORES = gnn_lookup.load_scores()


# --------------------------------------------------------------------------- #
# Customer-facing status model — maps internal routing/decision to plain language
# WITHOUT leaking scores, evidence, or any "fraud" framing.
# --------------------------------------------------------------------------- #
def customer_status(case: Case) -> dict:
    """Translate the internal case into a neutral, customer-safe status.

    Returns a dict: {key, label, tone, headline, detail, step} where ``step`` is
    the 1-3 index into the progress tracker and ``tone`` drives colour only.
    """
    decision = case.review.decision
    routing = case.report.routing

    # A human has made the final call — surface it neutrally.
    if decision == ReviewDecision.APPROVED:
        return {
            "key": "verified",
            "label": "Verified",
            "tone": "ok",
            "headline": "Your bank statement has been verified",
            "detail": ("Your document passed our verification checks. Your loan "
                       "application will now proceed to credit assessment. No "
                       "further action is needed from you at this stage."),
            "step": 3,
        }
    if decision == ReviewDecision.REJECTED:
        return {
            "key": "action_needed",
            "label": "More information needed",
            "tone": "warn",
            "headline": "We need to verify some details",
            "detail": ("We were unable to fully verify the document you "
                       "submitted. This can happen for a number of reasons, "
                       "including scan quality or a mismatch in the account "
                       "details. Please contact your relationship manager to "
                       "re-submit or provide an alternative statement."),
            "step": 3,
        }

    # No human decision yet — base the message on the automated routing, but
    # never reveal scores or that anything was "flagged".
    if routing == Routing.AUTO_CLEAR:
        return {
            "key": "verified",
            "label": "Verified",
            "tone": "ok",
            "headline": "Your bank statement has been verified",
            "detail": ("Your document passed our automated verification checks "
                       "and your application is moving forward. We will be in "
                       "touch about the next steps of your loan assessment."),
            "step": 3,
        }
    # HUMAN_REVIEW — surfaced simply as a specialist review, no alarm language.
    return {
        "key": "in_review",
        "label": "Being reviewed",
        "tone": "info",
        "headline": "Your application is being reviewed by a specialist",
        "detail": ("Thank you — we have received your bank statement. A member "
                   "of our team is completing a standard verification review. "
                   "This is a routine step for corporate loan applications and "
                   "does not require any action from you right now. We will "
                   "contact you if we need anything further."),
        "step": 2,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Landing + application form."""
    return templates.TemplateResponse("apply.html", {"request": request})


@app.post("/apply")
async def apply(
    request: Request,
    company_name: str = Form(...),
    uen: str = Form(...),
    contact_email: str = Form(...),
    loan_amount: str = Form(""),
    file: UploadFile | None = None,
):
    """Accept the application + statement, screen it, and confirm to the customer."""
    company_name = _scrub(company_name, _MAX_NAME)
    uen = _scrub(uen, _MAX_UEN)
    contact_email = _scrub(contact_email, _MAX_EMAIL)
    loan_amount = _scrub(loan_amount, _MAX_AMOUNT)

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A bank statement PDF is required")

    # Stream to disk with the same hardening as the analyst upload path.
    dest = _safe_upload_destination(file.filename)
    written = 0
    first_chunk = True
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            if first_chunk:
                if not chunk.startswith(_PDF_MAGIC):
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail="The file you uploaded is not a valid PDF",
                    )
                first_chunk = False
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            fh.write(chunk)

    if first_chunk:  # empty upload
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file was empty")

    # Screen through the exact same pipeline the analyst app uses.
    report = run_pipeline(str(dest), store=store, gnn_scores=GNN_SCORES)
    store.upsert_report(report)

    # The case_id doubles as the customer's application reference number.
    return RedirectResponse(url=f"/submitted/{report.case_id}", status_code=303)


@app.get("/submitted/{case_id}", response_class=HTMLResponse)
def submitted(request: Request, case_id: str):
    """Post-submission confirmation with the reference number + next steps."""
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Application not found", status_code=404)
    return templates.TemplateResponse("submitted.html", {
        "request": request, "case": case, "status": customer_status(case),
    })


@app.get("/status", response_class=HTMLResponse)
def status_form(request: Request, ref: str | None = None):
    """Status-lookup form; if ?ref= is present, show the status directly."""
    case = store.get(ref) if ref else None
    return templates.TemplateResponse("status.html", {
        "request": request,
        "ref": _scrub(ref or "", 40),
        "case": case,
        "not_found": bool(ref) and case is None,
        "status": customer_status(case) if case else None,
    })
