"""Lending ratios for the loan-officer console (FEAT-001, Part A).

Pure, deterministic helper that blends the OFFICER-ENTERED ``CreditMemo``
figures (balance-sheet / P&L items the statement does not contain) with the
cash-flow ``metrics`` derived from the parsed statement, to produce the handful
of ratios a loan officer reads when sizing a facility:

  * current ratio          — short-term liquidity
  * quick (acid-test)      — liquidity excluding inventory
  * debt-to-equity         — leverage
  * DSCR                   — can operating cash cover debt service? (flag < 1.2)
  * debt-to-revenue        — leverage against turnover

Every ratio is returned as a dict entry with:
  * ``value``          float, or None when a needed input is missing/zero;
  * ``formula``        the exact expression used (with the approximations named);
  * ``interpretation`` plain-English reading of the number (or what is missing);
  * ``caveat``         a standing reminder these are indicative, not audited;
  * (DSCR only) ``concern`` bool and ``concern_note`` when below the threshold.

CAVEATS / APPROXIMATIONS (documented here AND in each entry's strings):
  * current_assets ≈ accounts_receivable + inventory + closing cash balance
    (closing balance is passed in via ``metrics['closing_balance']``);
  * current_liabilities ≈ accounts_payable + one year of debt repayments
    (monthly_debt_repayments * 12) as a proxy for the short-term debt portion;
  * annual operating cash flow ≈ net monthly cash flow * 12 (from the statement).
These are deliberate simplifications so an officer gets an indicative read from
the inputs available; they are NOT audited financial-statement ratios.

NO new dependency, NO network, NO fabrication: when an input is absent or a
denominator is zero/None, the ratio ``value`` is None and the ``interpretation``
says which input is needed.
"""
from __future__ import annotations

from typing import Optional

from ..models import CreditMemo

# Below this DSCR a loan officer should treat debt-service coverage as a
# concern: operating cash is too thin against the debt service. 1.2x is a common
# minimum covenant floor. Named so the UI and tests reference one constant.
DSCR_CONCERN_THRESHOLD = 1.2

MONTHS_PER_YEAR = 12

_CAVEAT = ("Indicative, based on provided/derived inputs — not audited "
           "financials.")


def _entry(value: Optional[float], formula: str, interpretation: str,
           caveat: str = _CAVEAT) -> dict:
    return {
        "value": value,
        "formula": formula,
        "interpretation": interpretation,
        "caveat": caveat,
    }


def _safe_div(numerator: Optional[float],
              denominator: Optional[float]) -> Optional[float]:
    """Divide, returning None if either side is None or the denominator is 0."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def lending_ratios(memo: Optional[CreditMemo], metrics: dict) -> dict:
    """Compute indicative lending ratios from the memo + cash-flow metrics.

    ``memo`` carries the officer-entered figures; ``metrics`` is the dict from
    ``cashflow.cashflow_metrics`` (for net monthly cash flow) plus an optional
    ``closing_balance`` the caller passes through from the report. Both are
    treated defensively: a None memo or missing keys yield None-valued ratios
    with an explanatory interpretation, never an exception.
    """
    memo = memo or CreditMemo()
    closing_balance = metrics.get("closing_balance")
    net_monthly_cashflow = metrics.get("net_monthly_cashflow")

    ar = memo.accounts_receivable
    ap = memo.accounts_payable
    inventory = memo.inventory
    total_debt = memo.total_debt
    monthly_repay = memo.monthly_debt_repayments
    annual_revenue = memo.reported_annual_revenue
    equity = memo.equity

    # ---- current assets / current liabilities (approximations) --------- #
    # current_assets ≈ AR + inventory + closing cash balance
    ca_parts = [ar, inventory, closing_balance]
    current_assets = (sum(p for p in ca_parts if p is not None)
                      if any(p is not None for p in ca_parts) else None)
    # current_liabilities ≈ AP + 12 * monthly debt repayments (short-term proxy)
    annual_debt_service = (monthly_repay * MONTHS_PER_YEAR
                           if monthly_repay is not None else None)
    cl_parts = [ap, annual_debt_service]
    current_liabilities = (sum(p for p in cl_parts if p is not None)
                           if any(p is not None for p in cl_parts) else None)

    ratios: dict[str, dict] = {}

    # ---- current ratio ------------------------------------------------- #
    cr_formula = ("current_ratio = current_assets / current_liabilities, "
                  "where current_assets ≈ accounts_receivable + inventory + "
                  "closing cash balance and current_liabilities ≈ "
                  "accounts_payable + monthly_debt_repayments * 12")
    cr_caveat = (_CAVEAT + " current_liabilities uses 12 months of debt "
                 "repayments as a short-term-debt proxy.")
    current_ratio = _safe_div(current_assets, current_liabilities)
    if current_ratio is None:
        ratios["current_ratio"] = _entry(
            None, cr_formula,
            "Need accounts_receivable, inventory, closing balance, "
            "accounts_payable and monthly_debt_repayments "
            "(and current_liabilities must be non-zero).",
            cr_caveat)
    else:
        reading = ("above 1.0 — short-term assets cover short-term obligations"
                   if current_ratio >= 1.0
                   else "below 1.0 — short-term obligations exceed short-term "
                        "assets")
        ratios["current_ratio"] = _entry(
            current_ratio, cr_formula,
            f"Current ratio {current_ratio:.2f}: {reading}.", cr_caveat)

    # ---- quick (acid-test) ratio --------------------------------------- #
    qr_formula = ("quick_ratio = (current_assets - inventory) / "
                  "current_liabilities")
    qr_caveat = cr_caveat
    quick_assets = (current_assets - inventory
                    if current_assets is not None and inventory is not None
                    else (current_assets if current_assets is not None
                          and inventory is None else None))
    quick_ratio = _safe_div(quick_assets, current_liabilities)
    if quick_ratio is None:
        ratios["quick_ratio"] = _entry(
            None, qr_formula,
            "Need current assets (AR + inventory + closing balance), inventory "
            "and a non-zero current_liabilities.", qr_caveat)
    else:
        reading = ("above 1.0 — liquid assets cover short-term obligations "
                   "without relying on inventory"
                   if quick_ratio >= 1.0
                   else "below 1.0 — liquidity is thin once inventory is "
                        "excluded")
        ratios["quick_ratio"] = _entry(
            quick_ratio, qr_formula,
            f"Quick ratio {quick_ratio:.2f}: {reading}.", qr_caveat)

    # ---- debt-to-equity ------------------------------------------------ #
    de_formula = "debt_to_equity = total_debt / equity"
    debt_to_equity = _safe_div(total_debt, equity)
    if debt_to_equity is None:
        ratios["debt_to_equity"] = _entry(
            None, de_formula,
            "Need total_debt and a non-zero equity.")
    else:
        reading = ("higher leverage — debt exceeds equity"
                   if debt_to_equity > 1.0
                   else "conservative leverage — equity exceeds debt")
        ratios["debt_to_equity"] = _entry(
            debt_to_equity, de_formula,
            f"Debt-to-equity {debt_to_equity:.2f}: {reading}.")

    # ---- DSCR ---------------------------------------------------------- #
    dscr_formula = ("dscr = annual_operating_cash_flow / annual_debt_service, "
                    "where annual_operating_cash_flow ≈ net_monthly_cashflow * "
                    "12 (from the statement) and annual_debt_service = "
                    "monthly_debt_repayments * 12")
    dscr_caveat = (_CAVEAT + " Operating cash flow is proxied by the statement's "
                   "net monthly cash flow annualised.")
    annual_ocf = (net_monthly_cashflow * MONTHS_PER_YEAR
                  if net_monthly_cashflow is not None else None)
    dscr = _safe_div(annual_ocf, annual_debt_service)
    if dscr is None:
        ratios["dscr"] = _entry(
            None, dscr_formula,
            "Need net monthly cash flow (from the statement) and a non-zero "
            "monthly_debt_repayments.", dscr_caveat)
        ratios["dscr"]["concern"] = False
        ratios["dscr"]["concern_note"] = (
            "DSCR could not be computed; coverage cannot be assessed.")
    else:
        concern = dscr < DSCR_CONCERN_THRESHOLD
        if concern:
            interp = (f"DSCR {dscr:.2f} is below the {DSCR_CONCERN_THRESHOLD:g} "
                      "threshold — operating cash may not comfortably cover "
                      "debt service.")
            note = (f"Below {DSCR_CONCERN_THRESHOLD:g}x: debt-service coverage "
                    "is a concern.")
        else:
            interp = (f"DSCR {dscr:.2f} is at or above the "
                      f"{DSCR_CONCERN_THRESHOLD:g} threshold — operating cash "
                      "covers debt service.")
            note = (f"At or above {DSCR_CONCERN_THRESHOLD:g}x: debt-service "
                    "coverage is adequate.")
        entry = _entry(dscr, dscr_formula, interp, dscr_caveat)
        entry["concern"] = concern
        entry["concern_note"] = note
        ratios["dscr"] = entry

    # ---- debt-to-revenue ----------------------------------------------- #
    dr_formula = "debt_to_revenue = total_debt / reported_annual_revenue"
    debt_to_revenue = _safe_div(total_debt, annual_revenue)
    if debt_to_revenue is None:
        ratios["debt_to_revenue"] = _entry(
            None, dr_formula,
            "Need total_debt and a non-zero reported_annual_revenue.")
    else:
        ratios["debt_to_revenue"] = _entry(
            debt_to_revenue, dr_formula,
            f"Debt-to-revenue {debt_to_revenue:.2f}: total debt is "
            f"{debt_to_revenue:.2f}x reported annual revenue.")

    return ratios
