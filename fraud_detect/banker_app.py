"""Approval / banker web app — the loan-officer's perspective.

This is the **third** role in the system, distinct from the two existing apps:

    * ``customer_app.py``  — the applicant submits a statement, tracks status.
    * ``webapp.py``        — the fraud ANALYST screens the document for tampering
                             and clears or confirms the fraud concern.
    * ``banker_app.py``    — (this file) the loan OFFICER makes the final LENDING
                             decision on files the fraud screening has cleared.

Separation of duties (a core control): screening decides whether the document can
be *trusted*; the banker decides whether to *lend*. A banker only ever sees cases
whose fraud concern is resolved — the analyst APPROVED it, or it auto-cleared. A
statement still suspected of tampering, still under review, or REJECTED by the
analyst never reaches this queue. The banker cannot override a fraud rejection.

The banker's view is **credit-oriented**: it surfaces the trust signal (fraud
cleared + screening confidence), the headline financials parsed from the
statement, and the requested amount — then records APPROVE / DECLINE / REFER with
a mandatory rationale. Everything is written to the shared audit log.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .models import Case, CreditDecision, ReviewDecision, Routing
from .store import CaseStore

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates_banker"))

app = FastAPI(title="Loan Approval Console")
store = CaseStore(str(BASE.parent / "data" / "cases.json"))

# --------------------------------------------------------------------------- #
# Free-text sanitising (mirrors webapp.py: cap length, strip control chars,
# HTML-escape at the boundary).
# --------------------------------------------------------------------------- #
_MAX_OFFICER = 120
_MAX_RATIONALE = 2_000
_MAX_CONDITIONS = 2_000
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _scrub(text: str, max_len: int) -> str:
    text = (text or "")[:max_len]
    text = _CONTROL_RE.sub("", text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _parse_amount(raw: str) -> float | None:
    """Best-effort parse of a currency amount like 'SGD 500,000' -> 500000.0."""
    if not raw:
        return None
    cleaned = re.sub(r"[^0-9.]", "", raw)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Credit summary: the numbers a loan officer cares about, derived from the
# already-parsed statement. No fraud scores are "decided" here — the fraud
# outcome is shown only as a trust signal.
# --------------------------------------------------------------------------- #
def credit_summary(case: Case) -> dict:
    r = case.report
    txns = r.transactions or []
    credits = sum(t.amount for t in txns if t.amount and t.amount > 0)
    debits = sum(-t.amount for t in txns if t.amount and t.amount < 0)
    net = None
    if r.opening_balance is not None and r.closing_balance is not None:
        net = r.closing_balance - r.opening_balance

    # Trust signal derived from the fraud stage (shown, not re-decided).
    if case.review.decision == ReviewDecision.APPROVED:
        trust = ("Cleared by analyst", "ok")
    elif case.report.routing == Routing.AUTO_CLEAR:
        trust = ("Auto-cleared (no concerns)", "ok")
    else:
        trust = ("Not cleared", "warn")

    return {
        "currency": r.declared_currency or "",
        "opening": r.opening_balance,
        "closing": r.closing_balance,
        "total_credits": credits,
        "total_debits": debits,
        "net_change": net,
        "txn_count": len(txns),
        "trust_label": trust[0],
        "trust_tone": trust[1],
        "confidence": r.confidence_score,
        "confidence_band": r.confidence_band.value,
    }


@app.get("/", response_class=HTMLResponse)
def queue(request: Request):
    """The approval queue: fraud-cleared cases awaiting a lending decision."""
    cases = store.list_credit_queue()
    pending = [c for c in cases if c.credit.decision == CreditDecision.PENDING]
    decided = [c for c in cases if c.credit.decision != CreditDecision.PENDING]
    return templates.TemplateResponse("queue.html", {
        "request": request, "pending": pending, "decided": decided,
        "CreditDecision": CreditDecision,
    })


@app.get("/application/{case_id}", response_class=HTMLResponse)
def application(request: Request, case_id: str):
    """The full credit view of one application + the decision form."""
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Application not found", status_code=404)
    if not store.fraud_is_cleared(case):
        # Defence in depth: never expose a non-cleared file to the banker view.
        return templates.TemplateResponse("blocked.html", {
            "request": request, "case": case,
        }, status_code=403)
    return templates.TemplateResponse("application.html", {
        "request": request, "case": case, "r": case.report,
        "summary": credit_summary(case), "CreditDecision": CreditDecision,
    })


@app.post("/application/{case_id}/decide")
def decide(case_id: str, decision: str = Form(...), officer: str = Form(...),
           rationale: str = Form(...), approved_amount: str = Form(""),
           conditions: str = Form("")):
    officer = _scrub(officer, _MAX_OFFICER)
    rationale = _scrub(rationale, _MAX_RATIONALE)
    conditions = _scrub(conditions, _MAX_CONDITIONS)

    mapping = {
        "approve": CreditDecision.APPROVED,
        "decline": CreditDecision.DECLINED,
        "refer": CreditDecision.REFERRED,
    }
    if decision not in mapping:
        raise HTTPException(status_code=400, detail="Invalid decision value")

    amount = _parse_amount(approved_amount) if decision == "approve" else None
    try:
        case = store.record_credit_decision(
            case_id, mapping[decision], officer=officer, rationale=rationale,
            approved_amount=amount, conditions=conditions or None,
        )
    except ValueError as exc:
        # Fraud not cleared -> refuse (keeps separation of duties intact).
        raise HTTPException(status_code=409, detail=str(exc))
    if case is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return RedirectResponse(url=f"/application/{case_id}", status_code=303)
