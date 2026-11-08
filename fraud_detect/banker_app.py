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

from .models import Case, CreditDecision, CreditMemo, ReviewDecision, Routing
from .nodes.cashflow import cashflow_highlights, cashflow_metrics
from .nodes.loan_sizing import (
    CUSHION_MAX,
    CUSHION_MIN,
    DEFAULT_ANNUAL_RATE,
    DEFAULT_SIZING_DSCR,
    DEFAULT_TENOR_YEARS,
    RATE_MAX,
    RATE_MIN,
    TENOR_MAX_YEARS,
    TENOR_MIN_YEARS,
    indicative_loan_capacity,
)
from .nodes.ratios import lending_ratios
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
_MAX_COLLATERAL = 2_000
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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sizing_rate(raw: str | None) -> float:
    """Parse the officer-supplied annual rate (as a %), clamp to a sane range.

    The form submits a percentage (e.g. "8" for 8% p.a.); we store/compute the
    rate as a fraction. Bad input falls back to the default. Clamped to
    RATE_MIN..RATE_MAX (0%..100%).
    """
    if raw is None or raw.strip() == "":
        return DEFAULT_ANNUAL_RATE
    val = _parse_amount(raw)
    if val is None:
        return DEFAULT_ANNUAL_RATE
    return _clamp(val / 100.0, RATE_MIN, RATE_MAX)


def _sizing_tenor(raw: str | None) -> int:
    """Parse the officer-supplied tenor in years, clamp, fall back to default."""
    if raw is None or raw.strip() == "":
        return DEFAULT_TENOR_YEARS
    val = _parse_amount(raw)
    if val is None:
        return DEFAULT_TENOR_YEARS
    return int(_clamp(round(val), TENOR_MIN_YEARS, TENOR_MAX_YEARS))


def _sizing_cushion(raw: str | None) -> float:
    """Parse the officer-supplied DSCR sizing cushion, clamp, fall back."""
    if raw is None or raw.strip() == "":
        return DEFAULT_SIZING_DSCR
    val = _parse_amount(raw)
    if val is None:
        return DEFAULT_SIZING_DSCR
    return _clamp(val, CUSHION_MIN, CUSHION_MAX)


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
def application(request: Request, case_id: str, rate: str | None = None,
                tenor: str | None = None, cushion: str | None = None):
    """The full credit view of one application + the decision form.

    ``rate`` / ``tenor`` / ``cushion`` are OPTIONAL query params that let the
    officer re-run the indicative loan-capacity sizing with different assumptions
    (kept GET so it is idempotent and shareable). They are parsed defensively and
    clamped to sane ranges; bad input falls back to the documented defaults. The
    assumptions are NOT persisted to the store.
    """
    case = store.get(case_id)
    if not case:
        return HTMLResponse("Application not found", status_code=404)
    if not store.fraud_is_cleared(case):
        # Defence in depth: never expose a non-cleared file to the banker view.
        return templates.TemplateResponse("blocked.html", {
            "request": request, "case": case,
        }, status_code=403)

    # Cash-flow lens (derived from the parsed statement) + the officer-entered
    # memo + the indicative lending ratios that blend the two. Compute metrics
    # once; the ratio engine needs the closing balance, so pass it through on
    # the metrics dict rather than recomputing it.
    metrics = cashflow_metrics(case.report)
    metrics["closing_balance"] = case.report.closing_balance
    highlights = cashflow_highlights(case.report)
    memo = case.credit.memo if case.credit else None
    ratios = lending_ratios(memo, metrics)

    # Indicative loan capacity (screening aid — not a decision or an offer).
    # Officer-adjustable assumptions come from optional query params, parsed
    # defensively and clamped; the requested amount is read from the applicant's
    # typed application (reusing _parse_amount).
    sizing_rate = _sizing_rate(rate)
    sizing_tenor = _sizing_tenor(tenor)
    sizing_cushion = _sizing_cushion(cushion)
    requested_raw = case.application.loan_amount if case.application else None
    requested_amount = _parse_amount(requested_raw) if requested_raw else None
    capacity = indicative_loan_capacity(
        memo, metrics,
        annual_rate=sizing_rate, tenor_years=sizing_tenor,
        dscr_cushion=sizing_cushion,
        requested_amount=requested_amount,
        currency=case.report.declared_currency or "",
    )

    return templates.TemplateResponse("application.html", {
        "request": request, "case": case, "r": case.report,
        "summary": credit_summary(case), "CreditDecision": CreditDecision,
        "highlights": highlights, "metrics": metrics, "memo": memo,
        "ratios": ratios, "capacity": capacity,
        "requested_amount_raw": requested_raw,
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


@app.post("/application/{case_id}/memo")
def save_memo(
    case_id: str,
    officer: str = Form(""),
    accounts_receivable: str = Form(""),
    accounts_payable: str = Form(""),
    inventory: str = Form(""),
    total_debt: str = Form(""),
    monthly_debt_repayments: str = Form(""),
    reported_annual_revenue: str = Form(""),
    reported_net_profit: str = Form(""),
    equity: str = Form(""),
    collateral_description: str = Form(""),
    collateral_value: str = Form(""),
):
    """Capture the OFFICER-ENTERED credit memo (figures the statement lacks).

    Mirrors ``/decide``: scrub free text, best-effort parse the numeric fields,
    build a CreditMemo, and persist via ``store.save_credit_memo`` — which is
    itself guarded by ``fraud_is_cleared``. A non-cleared case raises ValueError
    in the store, which we map to 409 so separation of duties holds. On success
    redirect (303) back to the application so the ratios refresh.
    """
    memo = CreditMemo(
        accounts_receivable=_parse_amount(accounts_receivable),
        accounts_payable=_parse_amount(accounts_payable),
        inventory=_parse_amount(inventory),
        total_debt=_parse_amount(total_debt),
        monthly_debt_repayments=_parse_amount(monthly_debt_repayments),
        reported_annual_revenue=_parse_amount(reported_annual_revenue),
        reported_net_profit=_parse_amount(reported_net_profit),
        equity=_parse_amount(equity),
        collateral_description=(
            _scrub(collateral_description, _MAX_COLLATERAL) or None),
        collateral_value=_parse_amount(collateral_value),
    )
    officer = _scrub(officer, _MAX_OFFICER) or None
    try:
        case = store.save_credit_memo(case_id, memo, officer=officer)
    except ValueError as exc:
        # Fraud not cleared -> refuse (keeps separation of duties intact).
        raise HTTPException(status_code=409, detail=str(exc))
    if case is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return RedirectResponse(url=f"/application/{case_id}", status_code=303)
