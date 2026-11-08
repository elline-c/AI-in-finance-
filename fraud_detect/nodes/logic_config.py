"""Settings for the Node 3 financial-logic checks (everything except the fabrication
rules, which have their own ``FabricationConfig``).

Every number that decides whether a Node 3 check fires lives here, so it can be tuned
and tested without editing code. ``DEFAULT_LOGIC_CONFIG`` holds the values the
prototype ships with. None of them has been fitted to real bank data.

Use ``dataclasses.replace(DEFAULT_LOGIC_CONFIG, gap_max_days=30)`` to change a few.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..reference_data import STRUCTURING_THRESHOLD


def _positive_int(name: str, v: object, minimum: int = 1) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {v!r}")


@dataclass(frozen=True)
class FinancialLogicConfig:
    # --- balance checks ---
    balance_tolerance: float = 0.01          # rounding tolerance, in currency units

    # --- statement period (corporate-loan policy, not statistics) ---
    min_period_months: int = 6
    max_period_months: int = 12

    # --- FL_PERIOD_GAP: silence limit learned from the company's own rhythm ---
    gap_default_days: int = 25               # used when there are too few gaps to learn from
    gap_min_days: int = 10                   # never flag a silence shorter than this (holiday-length)
    gap_max_days: int = 35                   # never excuse a silence longer than this
    gap_min_observations: int = 10           # gaps needed before trusting a learned pattern
    gap_mad_mult: float = 6.0                # robust spreads above the median that count as unusual

    # --- credit baseline: what is a normal credit and a normal month for this company ---
    credit_min_obs: int = 10                 # credits needed to describe a normal credit size
    credit_min_months: int = 3               # complete months needed to describe a normal month
    declared_baseline_mult: float = 3.0      # last resort: credit above this x the declared month

    # --- FL_SUDDEN_DEPOSIT ---
    spike_month_share: float = 0.5           # at least this share of a normal month of credits
    spike_mad_mult: float = 6.0              # ... and above normal size by this many robust spreads

    # --- FL_RAPID_PASS_THROUGH ---
    pass_window_days: int = 3                # money out within this many days of the credit
    pass_match_ratio: float = 0.9            # ... totalling at least this share of the credit
    pass_max_ratio: float = 1.05             # ... but at most this share ("almost the same money")
    pass_max_payments: int = 1               # ... paid out in at most this many payments
    pass_max_candidates: int = 12            # largest outflows considered per credit
    pass_large_mult: float = 3.0             # only credits this x the normal credit are considered
    pass_min_credits: int = 5                # credits needed to know what a normal credit is
    pass_escalate_pairs: int = 3             # this many pairs or more: HIGH instead of MEDIUM
    pass_max_shown: int = 5                  # pairs quoted in the evidence
    # Scheduled outflows that naturally follow receipts are not pass-through.
    pass_exclude_words: tuple[str, ...] = ("CPF", "PAYROLL", "SALARY", "RENT", "GST", "IRAS", "TAX")

    # --- FL_STRUCTURING ---
    # Reference levels: a band just under each one is watched. Splitting to stay under a
    # reporting level is the offence at any amount, so add the levels that apply to you.
    structuring_levels: tuple[float, ...] = (STRUCTURING_THRESHOLD,)
    structuring_band: float = 0.05           # "just under" = within this share below a level
    structuring_cluster_days: int = 14       # credits this close together form a cluster
    structuring_min_credits: int = 3         # credits needed in a cluster
    # Relative checks, applied only when a baseline for the company exists:
    structuring_month_share: float = 0.10    # cluster must total this share of a normal month
    structuring_routine_share: float = 0.25  # ... and near-level credits must not already be
    structuring_min_others: int = 10         #     this share of the company's other credits

    # --- FL_UNVERIFIED_COUNTERPARTY ---
    counterparty_fallback_amount: float = 50_000.0   # "large" when no baseline exists at all
    counterparty_shell_words: tuple[str, ...] = ("UNKNOWN", "LLC", "HOLDINGS", "TT")

    def __post_init__(self) -> None:
        for name in ("min_period_months", "max_period_months", "gap_default_days", "gap_min_days",
                     "gap_max_days", "gap_min_observations", "credit_min_obs", "credit_min_months",
                     "pass_window_days", "pass_max_payments", "pass_max_candidates",
                     "pass_min_credits", "pass_escalate_pairs", "pass_max_shown",
                     "structuring_cluster_days", "structuring_min_credits",
                     "structuring_min_others"):
            _positive_int(name, getattr(self, name))
        if self.min_period_months > self.max_period_months:
            raise ValueError("min_period_months must not exceed max_period_months")
        if self.gap_min_days > self.gap_max_days:
            raise ValueError("gap_min_days must not exceed gap_max_days")
        for name in ("balance_tolerance", "gap_mad_mult", "declared_baseline_mult",
                     "spike_month_share", "spike_mad_mult", "pass_large_mult",
                     "structuring_month_share", "structuring_routine_share",
                     "counterparty_fallback_amount"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 < self.pass_match_ratio <= self.pass_max_ratio:
            raise ValueError("pass_match_ratio must be > 0 and not above pass_max_ratio")
        if not 0 < self.structuring_band < 1:
            raise ValueError("structuring_band must be between 0 and 1")
        if not self.structuring_levels or any(not (lv > 0) for lv in self.structuring_levels):
            raise ValueError("structuring_levels must be a non-empty tuple of positive amounts")


DEFAULT_LOGIC_CONFIG = FinancialLogicConfig()
