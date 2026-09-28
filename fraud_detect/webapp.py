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

import shutil
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .models import ReviewDecision
from .pipeline import run_pipeline
from .store import CaseStore

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    report = run_pipeline(str(dest))
    store.upsert_report(report)
    return RedirectResponse(url=f"/case/{report.case_id}", status_code=303)


@app.post("/screen-sample")
def screen_sample(path: str = Form(...)):
    report = run_pipeline(path)
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
    return FileResponse(case.report.source_path, media_type="application/pdf")


@app.post("/case/{case_id}/decide")
def decide(case_id: str, decision: str = Form(...), reviewer: str = Form(...),
           reason: str = Form(...), feedback: str = Form("")):
    dec = ReviewDecision.REJECTED if decision == "reject" else ReviewDecision.APPROVED
    store.record_decision(case_id, dec, reviewer=reviewer, reason=reason,
                          feedback_for_training=feedback or None)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)
