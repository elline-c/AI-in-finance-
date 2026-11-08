"""Indicative loan-capacity sizing for the loan-officer console.

Pure, deterministic, explainable helper (rules-first; NO LLM, NO network, NO new
dependency) that turns the figures already on the loan-officer view — the
officer-entered ``CreditMemo`` plus the statement-derived cash-flow ``metrics`` —
into an INDICATIVE maximum loan the bank could extend, and compares it to the
amount the applicant actually requested.

This is a SCREENING AID, not a credit decision and not an offer. Every number is
derived straight from the inputs via a documented formula; when an input is
missing the result is a clearly-flagged "insufficient inputs" dict rather than a
fabricated figure.

The headline figure is the LOWEST of up to three caps (mirroring real practice —
capacity AND leverage both constrain a loan):

  * Affordability / DSCR cap (primary): how much new debt the surplus operating
    cash flow can service, converted to a principal via the standard fixed-rate
    amortisation (present value of an annuity) formula.
  * Revenue-multiple cap: a conservative illustrative ceiling at a multiple of
    reported annual revenue (when revenue is present).
  * Leverage cap: the extra debt that keeps total debt within a target
    debt-to-equity ceiling (when equity and total debt are present).

The three default assumptions are OFFICER-ADJUSTABLE in the UI (exposed as named
constants + form inputs that default to these), never magic numbers buried in the
logic:

  * ``DEFAULT_ANNUAL_RATE`` = 8.0% p.a. (indicative APR, not a quoted rate).
  * ``DEFAULT_TENOR_YEARS`` = 5 years (60 monthly payments).
  * ``DEFAULT_SIZING_DSCR`` = 1.25 (minimum DSCR cushion for SIZING — deliberately
    a bit more conservative than ``ratios.DSCR_CONCERN_THRESHOLD`` = 1.2, which is
    the REPORTING flag; both are kept and named clearly).
"""
from __future__ import annotations

from typing import Optional

from ..models import CreditMemo
from .ratios import MONTHS_PER_YEAR

# --------------------------------------------------------------------------- #
# Officer-adjustable default assumptions (exposed to the UI as form defaults).
# --------------------------------------------------------------------------- #
# Indicative annual interest rate (APR) used to amortise the affordable payment
# into a principal. NOT a quoted rate — a screening assumption the officer can
# change. Expressed as a fraction (0.08 == 8.0% p.a.).
DEFAULT_ANNUAL_RATE = 0.08

# Indicative tenor in years. 5 years == 60 monthly payments.
DEFAULT_TENOR_YEARS = 5

# Minimum DSCR cushion used for SIZING (distinct from the reporting flag in
# ratios.DSCR_CONCERN_THRESHOLD = 1.2). Sizing is deliberately more conservative:
# we only lend against surplus cash flow divided by this cushion.
DEFAULT_SIZING_DSCR = 1.25

# Illustrative conservative ceiling: indicative principal capped at this multiple
# of reported annual revenue. Documented as illustrative / adjustable.
REVENUE_MULTIPLE_CAP = 1.0

# Target maximum debt-to-equity after the new facility. The leverage cap is the
# additional debt that keeps total debt within equity * this ceiling.
TARGET_MAX_DEBT_TO_EQUITY = 2.0

# Sane clamp ranges for officer-supplied assumptions (used by the web layer when
# parsing query params, re-exported here so the bounds live beside the defaults).
RATE_MIN, RATE_MAX = 0.0, 1.0          # 0% .. 100% p.a. (as a fraction)
TENOR_MIN_YEARS, TENOR_MAX_YEARS = 1, 30
CUSHION_MIN, CUSHION_MAX = 1.0, 3.0

_CAVEAT = (
    "Indicative only — a screening aid, not a credit decision or an offer. "
    "It depends on the officer-entered memo figures plus an ASSUMED rate and "
    "tenor (not a quoted rate). The revenue-multiple and leverage ceilings are "
    "illustrative, adjustable defaults."
)

_FORMULA = (
    "principal = A * (1 - (1 + i)^(-n)) / i, where A = affordable monthly "
    "payment = max(annual_operating_cash_flow - existing_annual_debt_service, 0) "
    "/ dscr_cushion / 12, i = annual_rate / 12, n = tenor_years * 12 "
    "(when annual_rate == 0, principal = A * n)."
)


def _annuity_principal(monthly_payment: float, annual_rate: float,
                       tenor_years: int) -> float:
    """Present value of an annuity: principal a monthly payment can amortise.

    ``P = A * (1 - (1 + i)^(-n)) / i`` with ``i = annual_rate / 12`` and
    ``n = tenor_years * 12``. The ``i == 0`` (zero-rate) edge case degenerates to
    ``P = A * n``. All inputs are assumed already validated/clamped by the caller.
    """
    n = tenor_years * MONTHS_PER_YEAR
    if monthly_payment <= 0 or n <= 0:
        return 0.0
    i = annual_rate / MONTHS_PER_YEAR
    if i == 0:
        return monthly_payment * n
    return monthly_payment * (1 - (1 + i) ** (-n)) / i


def indicative_loan_capacity(
    memo: Optional[CreditMemo],
    metrics: Optional[dict],
    *,
    annual_rate: float = DEFAULT_ANNUAL_RATE,
    tenor_years: int = DEFAULT_TENOR_YEARS,
    dscr_cushion: float = DEFAULT_SIZING_DSCR,
    requested_amount: Optional[float] = None,
    currency: Optional[str] = None,
) -> dict:
    """Compute an indicative maximum loan and a fit-vs-requested verdict.

    Returns a dict that is always safe to render. When the core inputs are
    missing (no memo, or no net monthly cash flow to proxy operating cash flow),
    ``available`` is False and ``indicative_max`` is None with an
    ``insufficient_inputs`` message — no number is fabricated.

    ``requested_amount`` (already parsed to a float by the caller via
    ``banker_app._parse_amount``) drives the verdict; ``currency`` is carried
    through purely for display.
    """
    assumptions = {
        "annual_rate": annual_rate,
        "annual_rate_pct": annual_rate * 100,
        "tenor_years": tenor_years,
        "tenor_months": tenor_years * MONTHS_PER_YEAR,
        "dscr_cushion": dscr_cushion,
        "revenue_multiple_cap": REVENUE_MULTIPLE_CAP,
        "target_max_debt_to_equity": TARGET_MAX_DEBT_TO_EQUITY,
    }

    metrics = metrics or {}
    net_monthly_cashflow = metrics.get("net_monthly_cashflow")

    # Core requirement: a cash-flow proxy for operating cash flow. Without it (or
    # with no memo at all) we cannot size affordability — return a flagged
    # insufficient-inputs result rather than inventing a figure.
    if memo is None or net_monthly_cashflow is None:
        return {
            "available": False,
            "indicative_max": None,
            "currency": currency or "",
            "assumptions": assumptions,
            "formula": _FORMULA,
            "caveat": _CAVEAT,
            "insufficient_inputs": True,
            "message": (
                "Enter revenue, existing debt and monthly repayments in the "
                "credit memo (and ensure the statement yields a net monthly "
                "cash flow) to see an indicative capacity."
            ),
            "requested_amount": requested_amount,
            "verdict": "cannot assess (need more inputs)",
            "verdict_tone": "warn",
            "verdict_line": (
                "Cannot assess an indicative capacity yet — more inputs needed."
            ),
            "caps": [],
            "binding_constraint": None,
        }

    # ---- Step 1: surplus annual cash flow available for NEW debt service ---- #
    annual_ocf = net_monthly_cashflow * MONTHS_PER_YEAR
    monthly_repay = memo.monthly_debt_repayments
    existing_annual_debt_service = (
        monthly_repay * MONTHS_PER_YEAR if monthly_repay is not None else 0.0
    )
    surplus_annual = annual_ocf - existing_annual_debt_service

    caps: list[dict] = []

    # ---- Step 2-4: affordability / DSCR cap (primary) ---------------------- #
    if surplus_annual <= 0:
        affordable_annual_payment = 0.0
        affordable_monthly_payment = 0.0
        affordability_principal = 0.0
        affordability_note = (
            "No surplus cash flow to service new debt — annual operating cash "
            "flow does not exceed existing debt service, so the indicative "
            "affordability capacity is 0."
        )
    else:
        affordable_annual_payment = surplus_annual / dscr_cushion
        affordable_monthly_payment = affordable_annual_payment / MONTHS_PER_YEAR
        affordability_principal = _annuity_principal(
            affordable_monthly_payment, annual_rate, tenor_years)
        affordability_note = (
            "Largest principal the surplus operating cash flow can amortise at "
            "the assumed rate and tenor, after applying the DSCR sizing cushion."
        )
    caps.append({
        "key": "affordability",
        "label": "Affordability / DSCR cap",
        "value": affordability_principal,
        "note": affordability_note,
    })

    # ---- Revenue-multiple cap (illustrative ceiling) ----------------------- #
    annual_revenue = memo.reported_annual_revenue
    if annual_revenue is not None and annual_revenue > 0:
        revenue_cap = annual_revenue * REVENUE_MULTIPLE_CAP
        caps.append({
            "key": "revenue_multiple",
            "label": "Revenue-multiple cap",
            "value": revenue_cap,
            "note": (
                f"Illustrative ceiling at {REVENUE_MULTIPLE_CAP:g}x reported "
                f"annual revenue ({annual_revenue:,.0f})."
            ),
        })

    # ---- Leverage cap (debt-to-equity ceiling) ----------------------------- #
    equity = memo.equity
    total_debt = memo.total_debt
    if equity is not None and total_debt is not None and equity > 0:
        max_total_debt = equity * TARGET_MAX_DEBT_TO_EQUITY
        headroom = max(max_total_debt - total_debt, 0.0)
        caps.append({
            "key": "leverage",
            "label": "Leverage cap",
            "value": headroom,
            "note": (
                f"Extra debt that keeps total debt within "
                f"{TARGET_MAX_DEBT_TO_EQUITY:g}x equity: equity {equity:,.0f} * "
                f"{TARGET_MAX_DEBT_TO_EQUITY:g} = {max_total_debt:,.0f} max "
                f"total debt, less existing debt {total_debt:,.0f} "
                f"(floored at 0)."
            ),
        })

    # ---- Indicative figure = LOWEST available cap; report the binding one --- #
    binding = min(caps, key=lambda c: c["value"])
    indicative_max = binding["value"]
    for c in caps:
        c["binding"] = c is binding

    binding_phrases = {
        "affordability": "limited by debt-service affordability",
        "revenue_multiple": "limited by the revenue multiple",
        "leverage": "limited by the leverage ceiling",
    }
    binding_constraint = binding_phrases.get(binding["key"], "limited by "
                                             + binding["label"])

    if surplus_annual <= 0 and binding["key"] == "affordability":
        explanation = (
            "Indicative capacity is 0: there is no surplus operating cash flow "
            "to service new debt after existing repayments."
        )
    else:
        explanation = (
            f"Indicative maximum of {indicative_max:,.0f} {currency or ''}".strip()
            + f" is {binding_constraint}. It is the lowest of the "
            f"{len(caps)} cap(s) computed from the available inputs, amortising "
            f"an affordable monthly payment of {affordable_monthly_payment:,.0f} "
            f"over {tenor_years} year(s) at {annual_rate * 100:g}% p.a."
        )

    # ---- Fit-vs-requested verdict ------------------------------------------ #
    verdict, verdict_tone, verdict_line = _verdict(
        indicative_max, requested_amount, currency)

    return {
        "available": True,
        "indicative_max": indicative_max,
        "currency": currency or "",
        "surplus_annual": surplus_annual,
        "annual_operating_cashflow": annual_ocf,
        "existing_annual_debt_service": existing_annual_debt_service,
        "affordable_monthly_payment": affordable_monthly_payment,
        "affordable_annual_payment": affordable_annual_payment,
        "assumptions": assumptions,
        "caps": caps,
        "binding_constraint": binding_constraint,
        "binding_key": binding["key"],
        "formula": _FORMULA,
        "explanation": explanation,
        "caveat": _CAVEAT,
        "insufficient_inputs": False,
        "requested_amount": requested_amount,
        "verdict": verdict,
        "verdict_tone": verdict_tone,
        "verdict_line": verdict_line,
    }


# Requested amount within this fraction OVER the indicative max is "at the edge".
EDGE_BAND = 0.10


def _verdict(indicative_max: Optional[float], requested: Optional[float],
             currency: Optional[str]) -> tuple[str, str, str]:
    """Compare the requested amount against the indicative max.

    Returns ``(verdict, tone, readable_line)``. Tone is a page style class:
    ``ok`` (green) when within capacity, ``warn`` (amber) at the edge or above,
    ``warn`` when it cannot be assessed.
    """
    cur = (currency or "").strip()

    def _fmt(x: float) -> str:
        return (f"{cur} {x:,.0f}".strip()) if cur else f"{x:,.0f}"

    if indicative_max is None:
        return (
            "cannot assess (need more inputs)", "warn",
            "Cannot assess fit — the indicative capacity could not be computed.",
        )
    if requested is None:
        return (
            "no requested amount captured", "warn",
            "No requested loan amount was captured on the application, so fit "
            f"against the indicative maximum of {_fmt(indicative_max)} cannot "
            "be judged.",
        )
    if requested <= indicative_max:
        return (
            "within indicative capacity", "ok",
            f"Requested {_fmt(requested)} is within the indicative maximum of "
            f"{_fmt(indicative_max)}.",
        )
    if requested <= indicative_max * (1 + EDGE_BAND):
        return (
            "at the edge", "warn",
            f"Requested {_fmt(requested)} is just above the indicative maximum "
            f"of {_fmt(indicative_max)} (within ~{EDGE_BAND * 100:g}%).",
        )
    return (
        "above indicative capacity", "warn",
        f"Requested {_fmt(requested)} exceeds the indicative maximum of "
        f"{_fmt(indicative_max)}.",
    )
