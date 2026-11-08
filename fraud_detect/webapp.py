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
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .ml.gnn import gnn_lookup
from .nodes import ml_statement, ml_transaction
from .nodes.checklist import CHECKLIST_SIDE_NOTE, build_checklist, stage_confidence_map
from .models import ReviewDecision, display_company_name
from .registry import live_lookup_lifespan
from .reviewers import (
    is_valid_reviewer,
    is_valid_senior,
    reviewer_choices,
    senior_choices,
)
from .store import CaseStore, revert_window_remaining, REVERT_WINDOW_MINUTES

BASE = Path(__file__).resolve().parent
# UPLOAD_DIR is still required by GET /pdf/{id} to confirm a stored statement
# PDF lives inside the uploads directory before serving it.
UPLOAD_DIR = (BASE.parent / "data" / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bank Statement Tampering Screening", lifespan=live_lookup_lifespan)
templates = Jinja2Templates(directory=str(BASE / "templates"))
store = CaseStore(str(BASE.parent / "data" / "cases.json"))

# Load the precomputed GNN graph-fraud scores once at startup (fail-safe: returns
# None if the store is absent). gnn_lookup imports only joblib (no torch), so this
# is safe in the live app and avoids re-reading the joblib file on every request.
GNN_SCORES = gnn_lookup.load_scores()

# Load the trained ML models once at startup, mirroring the GNN_SCORES pattern.
# Both loaders are fail-safe (return None if the .joblib file is absent), so the
# optional ML nodes (6a/6b) simply stay dormant when a model is missing.
# Off by default: see ml_transaction.ENABLE_ENV. GNN_SCORES above still feed Node 6a.
TXN_MODEL = ml_transaction.load_live_model()
STATEMENT_MODEL = ml_statement.load_model()

_BAND_COLOR = {
    "LOW": "#2e7d32", "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00", "VERY_HIGH": "#c62828",
}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    cases = store.list_cases()
    # Resolve the display company name per case (applicant-typed name preferred
    # over the statement-parsed account_holder), keyed by case_id so the queue
    # template can look it up without re-deriving the preference inline.
    company_names = {c.case_id: display_company_name(c) for c in cases}
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "cases": cases, "band_color": _BAND_COLOR,
        "company_names": company_names,
    })


# ITEM 7: the analyst console no longer screens statements on its own. The
# POST /upload and POST /screen-sample handlers were removed as an internal
# control - a fraud analyst only reviews cases that reach them through the
# customer portal, never ad-hoc statements of their own choosing. Hitting those
# paths now returns 404/405 naturally because the routes are gone. GET /pdf and
# UPLOAD_DIR remain because the PDF viewer still needs them.


# --------------------------------------------------------------------------- #
# Per-node (pipeline-stage) grouping of findings, for the analyst stage views.
# Maps the agent label prefix each node emits to a human-readable stage.
# --------------------------------------------------------------------------- #
_STAGE_META = [
    ("Node1", "DQ_", "1 · Ingestion & data quality",
     "Reads the statement and flags missing/unreadable fields. Never guesses."),
    ("Node2", "VF_", "2 · Visual forensics",
     "Font/format inconsistencies and editing-software traces in the document."),
    ("Node3", "FL_", "3 · Financial logic",
     "Balance reconciliation, period gaps, structuring and counterparty checks."),
    ("Node5", "MEM_", "5 · Cross-case memory",
     "Matches suspicious rows against prior human-decided cases."),
    ("Node6", "ML_", "6 · ML / GNN signals",
     "Optional machine-learning and graph-fraud signals (when models are loaded)."),
]


def _group_by_stage(findings):
    """Return [(title, blurb, [findings])] in pipeline order, incl. empty stages."""
    groups = []
    for prefix, code_prefix, title, blurb in _STAGE_META:
        matched = [
            f for f in findings
            if f.agent.startswith(prefix) or f.code.startswith(code_prefix)
        ]
        groups.append((title, blurb, matched))
    # Anything that didn't match a known stage (e.g. security layer) -> "Other".
    known = {id(f) for _, _, g in groups for f in g}
    other = [f for f in findings if id(f) not in known]
    if other:
        groups.append(("Other signals", "Security layer and uncategorised findings.", other))
    return groups


def _transactions_with_findings(report):
    """Join each parsed transaction row with the findings that cite it.

    Returns [(transaction, [findings])] with flagged rows (most severe first)
    preceding clean rows, so the analyst reviews what matters first.
    """
    by_row = {}
    for f in report.findings:
        if f.field and f.field.startswith("row "):
            try:
                row = int(f.field.split()[1])
            except (ValueError, IndexError):
                continue
            by_row.setdefault(row, []).append(f)
    rows = [(t, by_row.get(t.index, [])) for t in report.transactions]
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    rows.sort(key=lambda tf: (
        0 if tf[1] else 1,
        min((sev_rank.get(f.severity.value, 9) for f in tf[1]), default=9),
    ))
    return rows


@app.get("/case/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: str):
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Case not found", status_code=404)

    # Group findings by pipeline stage once; the per-stage confidence map is
    # keyed by the SAME stage titles so the template can show a confidence band
    # next to each inlined stage (ITEM 2).
    stages = _group_by_stage(case.report.findings)
    stage_confidence = stage_confidence_map(stages, case.report)

    # Human-decision checklist (ITEM 2): the key data points the screening
    # looks for, each with a tick/cross and an explainable confidence band, plus
    # the recommended side-note (single source of truth imported from the
    # checklist module).
    checklist = build_checklist(case.report)

    # ITEM 1 (server gate): the number of pipeline stages the officer must
    # acknowledge having reviewed before deciding. Passed to the template so the
    # acknowledgement control can reference "all N pipeline stages" and so the
    # client echoes the same count the server enforces.
    stage_count = len(stages)

    # ITEM 4: whether the case can still be reverted (final decision within the
    # 30-minute window) so the template shows/hides the revert control.
    revert_allowed, revert_seconds_remaining = revert_window_remaining(case)

    return templates.TemplateResponse("case.html", {
        "request": request, "case": case, "r": case.report,
        "company_name": display_company_name(case),
        "band_color": _BAND_COLOR, "ReviewDecision": ReviewDecision,
        "stages": stages,
        "stage_confidence": stage_confidence,
        "checklist": checklist,
        "checklist_side_note": CHECKLIST_SIDE_NOTE,
        "stage_count": stage_count,
        "reviewer_choices": reviewer_choices(),
        "senior_choices": senior_choices(),
        "revert_allowed": revert_allowed,
        "revert_seconds_remaining": revert_seconds_remaining,
        "revert_window_minutes": REVERT_WINDOW_MINUTES,
    })


@app.get("/case/{case_id}/transactions", response_class=HTMLResponse)
def case_transactions(request: Request, case_id: str):
    """Stage: review the flagged transactions row by row."""
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Case not found", status_code=404)
    rows = _transactions_with_findings(case.report)
    flagged = sum(1 for _, fs in rows if fs)
    return templates.TemplateResponse("transactions.html", {
        "request": request, "case": case, "r": case.report,
        "rows": rows, "flagged": flagged, "band_color": _BAND_COLOR,
    })


@app.get("/case/{case_id}/stages", response_class=HTMLResponse)
def case_stages(request: Request, case_id: str):
    """Stage: see what each pipeline node produced."""
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Case not found", status_code=404)
    return templates.TemplateResponse("stages.html", {
        "request": request, "case": case, "r": case.report,
        "stages": _group_by_stage(case.report.findings),
    })


@app.post("/case/{case_id}/request-info")
def request_info(case_id: str, reviewer: str = Form(...),
                 message: str = Form(...), internal_note: str = Form("")):
    """Stage: reviewer asks the applicant for more information (NEEDS_INFO)."""
    reviewer = _scrub(reviewer, _MAX_REVIEWER)
    message = _scrub(message, _MAX_REASON)
    internal_note = _scrub(internal_note, _MAX_FEEDBACK)
    # ITEM 5: the reviewer must be a known officer from the roster. _scrub HTML-
    # escapes the submitted string, and reviewer_choices()/is_valid_reviewer()
    # operate on the same escaped display strings, so an exact match holds.
    if not is_valid_reviewer(reviewer):
        raise HTTPException(status_code=400, detail="Unknown reviewer")
    if not store.get(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    store.request_info(case_id, reviewer=reviewer,
                       message_to_customer=message,
                       internal_note=internal_note or None)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


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
           reason: str = Form(...), feedback: str = Form(""),
           stages_reviewed: str = Form("")):
    # Sanitize free-text from the review form: cap length, strip control chars,
    # HTML-escape. These values end up in the JSON case store and potentially
    # in reports / templates — we neutralize injection at the boundary.
    reviewer = _scrub(reviewer, _MAX_REVIEWER)
    reason = _scrub(reason, _MAX_REASON)
    feedback = _scrub(feedback, _MAX_FEEDBACK)

    # decision must be an expected value; reject anything unexpected.
    if decision not in ("reject", "approve"):
        raise HTTPException(status_code=400, detail="Invalid decision value")

    # ITEM 5: the reviewer signing off must be a known roster officer.
    if not is_valid_reviewer(reviewer):
        raise HTTPException(status_code=400, detail="Unknown reviewer")

    # The case must exist before we can gate on its pipeline stages / decide.
    case = store.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # ITEM 1 (server gate): the officer MUST confirm they have reviewed every
    # pipeline stage before any decision is recorded. The checkbox only POSTs a
    # value when ticked, so an absent/falsy `stages_reviewed` means the officer
    # did not acknowledge the review. This is a REAL server-side control: the
    # gate runs BEFORE store.record_decision, independent of the client. The
    # stage count is computed server-side from the actual findings grouping.
    stage_count = len(_group_by_stage(case.report.findings))
    if not stages_reviewed.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "You must confirm you have reviewed all "
                f"{stage_count} pipeline stages before deciding"
            ),
        )

    dec = ReviewDecision.REJECTED if decision == "reject" else ReviewDecision.APPROVED
    store.record_decision(case_id, dec, reviewer=reviewer, reason=reason,
                          feedback_for_training=feedback or None)
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


@app.post("/case/{case_id}/revert")
def revert(case_id: str, reviewer: str = Form(...), note: str = Form(...)):
    """ITEM 4: undo an APPROVED/REJECTED decision within the 30-minute window.

    The officer made a mistake (e.g. rejected an application that should have
    been approved). Reverting re-opens the case to PENDING and appends a NEW
    audit event; prior decision history is preserved (append-only). The store
    raises ValueError when the case is not in a final state or the 30-minute
    window has elapsed, which we surface to the client as HTTP 400.
    """
    reviewer = _scrub(reviewer, _MAX_REVIEWER)
    note = _scrub(note, _MAX_FEEDBACK)
    if not is_valid_reviewer(reviewer):
        raise HTTPException(status_code=400, detail="Unknown reviewer")
    if not store.get(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    try:
        store.revert_decision(case_id, reviewer=reviewer, note=note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RedirectResponse(url=f"/case/{case_id}", status_code=303)


@app.post("/case/{case_id}/escalate")
def escalate(case_id: str, reviewer: str = Form(...),
             senior_officer: str = Form(...), note: str = Form(...)):
    """ITEM 6: route a complex case up to a senior officer.

    A junior officer who lacks the right experience reroutes the case to a
    senior officer and MUST justify the reroute with an internal note. Both the
    junior reviewer and the chosen senior officer are validated against their
    rosters; a blank note is rejected.
    """
    reviewer = _scrub(reviewer, _MAX_REVIEWER)
    senior_officer = _scrub(senior_officer, _MAX_REVIEWER)
    note = _scrub(note, _MAX_FEEDBACK)
    if not is_valid_reviewer(reviewer):
        raise HTTPException(status_code=400, detail="Unknown reviewer")
    if not is_valid_senior(senior_officer):
        raise HTTPException(status_code=400, detail="Unknown senior officer")
    if not note.strip():
        raise HTTPException(
            status_code=400,
            detail="An internal note is required when escalating to a senior officer",
        )
    if not store.get(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    store.escalate_to_senior(case_id, reviewer=reviewer,
                             senior_officer=senior_officer, note=note)
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
