"""Human-in-the-loop review web app (Phase 4).

A credit analyst can:
    * see all screened statements (auto-cleared vs escalated),
    * open a flagged case and read the risk score, confidence score, every
      detected issue with its evidence and the relevant page/transaction,
    * view the underlying statement PDF,
    * confirm (reject application) or clear (approve) the fraud concern, recording
      a reason and optional feedback for future model training.

The machine never rejects on its own — only a human can reject here.
"""
from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .models import ReviewDecision
from .pipeline import run_pipeline
from .store import CaseStore

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE.parent / "data" / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Anti-hack upload constraints
# --------------------------------------------------------------------------- #
MAX_UPLOAD_BYTES = 25 * 1024 * 1024          # 25 MB ceiling (DoS guard)
_PDF_MAGIC = b"%PDF-"                         # a real PDF starts with this
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")  # everything else is stripped


def _safe_upload_destination(filename: str | None) -> Path:
    """Produce a destination path that cannot escape UPLOAD_DIR.

    Defeats path-traversal (``../``), absolute paths, and NUL-byte tricks by
    taking only the final path component, stripping it to a safe charset, and
    prefixing a random token so uploads can never collide or overwrite.
    """
    raw = (filename or "statement.pdf").replace("\x00", "")
    base = Path(raw).name                      # drop any directory part
    base = _SAFE_NAME_RE.sub("_", base) or "statement.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    dest = (UPLOAD_DIR / f"{secrets.token_hex(8)}_{base}").resolve()
    # Belt-and-braces: confirm the resolved path is still inside UPLOAD_DIR.
    if UPLOAD_DIR not in dest.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    return dest

app = FastAPI(title="Bank Statement Tampering Screening")
templates = Jinja2Templates(directory=str(BASE / "templates"))
store = CaseStore(str(BASE.parent / "data" / "cases.json"))

_BAND_COLOR = {
    "LOW": "#2e7d32", "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00", "VERY_HIGH": "#c62828",
}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    cases = store.list_cases()
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "cases": cases, "band_color": _BAND_COLOR,
    })


@app.post("/upload")
async def upload(file: UploadFile):
    # 1. Sanitize the destination (path-traversal / overwrite guard).
    dest = _safe_upload_destination(file.filename)

    # 2. Stream to disk with a hard size cap (DoS guard) and verify the PDF
    #    magic bytes on the first chunk (content-type spoofing guard).
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
                        detail="Uploaded file is not a valid PDF",
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
        raise HTTPException(status_code=400, detail="Empty upload")

    report = run_pipeline(str(dest))
    store.upsert_report(report)
    return RedirectResponse(url=f"/case/{report.case_id}", status_code=303)


@app.post("/screen-sample")
def screen_sample(name: str = Form(...)):
    """Screen one of the bundled sample PDFs *by name only*.

    The previous version accepted an arbitrary filesystem path, allowing a
    client to read/serve any file on the host. We now accept only a bare
    filename and resolve it strictly inside the samples directory.
    """
    safe = Path(_SAFE_NAME_RE.sub("_", Path(name).name))
    samples_dir = (BASE.parent / "samples").resolve()
    target = (samples_dir / safe).resolve()
    if samples_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    report = run_pipeline(str(target))
    store.upsert_report(report)
    return RedirectResponse(url=f"/case/{report.case_id}", status_code=303)


@app.get("/case/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: str):
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Case not found", status_code=404)
    return templates.TemplateResponse("case.html", {
        "request": request, "case": case, "r": case.report,
        "band_color": _BAND_COLOR, "ReviewDecision": ReviewDecision,
    })


@app.get("/pdf/{case_id}")
def case_pdf(case_id: str):
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Case not found", status_code=404)
    # Only serve PDFs that live inside UPLOAD_DIR (prevent stored-path
    # manipulation from leaking arbitrary host files).
    src = Path(case.report.source_path).resolve()
    if UPLOAD_DIR not in src.parents or not src.is_file():
        return HTMLResponse("PDF not accessible", status_code=403)
    return FileResponse(str(src), media_type="application/pdf")


# Length caps for human-review free-text fields (XSS / log-injection guard).
_MAX_REVIEWER = 120
_MAX_REASON = 2_000
_MAX_FEEDBACK = 4_000


@app.post("/case/{case_id}/decide")
def decide(case_id: str, decision: str = Form(...), reviewer: str = Form(...),
           reason: str = Form(...), feedback: str = Form("")):
    # Sanitize free-text from the review form: cap length, strip control chars,
    # HTML-escape. These values end up in the JSON case store and potentially
    # in reports / templates — we neutralize injection at the boundary.
    reviewer = _scrub(reviewer, _MAX_REVIEWER)
    reason = _scrub(reason, _MAX_REASON)
    feedback = _scrub(feedback, _MAX_FEEDBACK)

    # decision must be an expected value; reject anything unexpected.
    if decision not in ("reject", "approve"):
        raise HTTPException(status_code=400, detail="Invalid decision value")

    dec = ReviewDecision.REJECTED if decision == "reject" else ReviewDecision.APPROVED
    store.record_decision(case_id, dec, reviewer=reviewer, reason=reason,
                          feedback_for_training=feedback or None)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


# --------------------------------------------------------------------------- #
# Free-text sanitizer (mirrors the pattern in input_integrity.py)
# --------------------------------------------------------------------------- #
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _scrub(text: str, max_len: int) -> str:
    text = text[:max_len]
    text = _CONTROL_RE.sub("", text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
