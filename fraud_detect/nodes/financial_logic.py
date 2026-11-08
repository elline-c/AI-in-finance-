"""Node 3 — Financial Logic & Validation Agent.

Validates the *numbers and behaviour* of the statement:

    * Mathematical continuity — every running balance must equal the previous
      balance plus the row amount (row-by-row reconciliation).
    * Opening/closing consistency — first/last balances match the header.
    * Period continuity — no silence longer than the company's own rhythm allows.
    * Suspicious transactions — deposits unusually large for THIS company, money that
      leaves again within days, and large inflows from names never seen before.
    * Structuring — clusters of credits just below a reporting level.
    * Entity cross-check — account-holder name vs the ACRA record for the UEN
      (catches statements reused under a different company).

Findings are explainable and carry the offending page/row where possible. Every
setting is in ``logic_config.FinancialLogicConfig``.
"""
from __future__ import annotations

import itertools
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Sequence

from ..history import TrailingHistory, counterparty_key, credit_sizes, monthly_credit_totals
from ..models import AgentReport, Finding, IngestionResult, Severity, Transaction
from . import fabrication
from .fabrication import DEFAULT_CONFIG
from .logic_config import DEFAULT_LOGIC_CONFIG, FinancialLogicConfig
from .coverage import (
    duplicate_months,
    missing_months,
    month_name,
    months_in_period,
    out_of_order_segments,
)
from ..registry import NOT_FOUND as REGISTRY_NOT_FOUND
from ..registry import UNAVAILABLE as REGISTRY_UNAVAILABLE
from ..registry import lookup_entity, names_match
from ..reference_data import CASHFLOW_BASELINE, KNOWN_VENDORS

AGENT = "Node3:FinancialLogic"


def _reconcile(ing: IngestionResult,
               cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    findings: list[Finding] = []
    txns = ing.transactions
    if not txns:
        return findings

    prev_balance = ing.opening_balance
    for t in txns:
        if prev_balance is None or t.amount is None or t.balance is None:
            prev_balance = t.balance
            continue
        expected = round(prev_balance + t.amount, 2)
        if abs(expected - t.balance) > cfg.balance_tolerance:
            findings.append(Finding(
                agent=AGENT, code="FL_BALANCE_BREAK",
                title="Running balance does not reconcile",
                severity=Severity.HIGH, weight=26, page=t.page,
                field=f"row {t.index}",
                evidence=(f"Row {t.index} ({t.date} {t.description}): "
                          f"expected balance {expected:,.2f} "
                          f"(prev {prev_balance:,.2f} {'+' if t.amount>=0 else '-'} "
                          f"{abs(t.amount):,.2f}) but statement shows {t.balance:,.2f}. "
                          f"Discrepancy of {t.balance - expected:,.2f} indicates an "
                          f"altered amount or balance."),
            ))
        prev_balance = t.balance
    return findings


def _closing_check(ing: IngestionResult,
                   cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    findings: list[Finding] = []
    if ing.closing_balance is not None and ing.transactions:
        last = ing.transactions[-1].balance
        if last is not None and abs(last - ing.closing_balance) > cfg.balance_tolerance:
            findings.append(Finding(
                agent=AGENT, code="FL_CLOSING_MISMATCH",
                title="Closing balance disagrees with last row",
                severity=Severity.MEDIUM, weight=12,
                evidence=(f"Header closing balance {ing.closing_balance:,.2f} does not "
                          f"match the final row balance {last:,.2f}."),
            ))
    return findings


def gaps_between_active_days(dates: Sequence[date]) -> list[int]:
    """Days of silence between consecutive distinct transaction dates."""
    days = sorted(set(dates))
    return [(b - a).days for a, b in zip(days, days[1:])]


def gap_limit(baseline_gaps: Sequence[int],
              cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> tuple[float, str]:
    """Learn how long this company normally goes without a transaction.

    Returns ``(limit_in_days, basis)``. A busy account that posts nearly every day
    gets a short limit; a sparse account gets a longer one. The limit is
    ``median + gap_mad_mult * robust spread`` of the observed gaps, so a single
    long silence (the thing we are hunting) cannot stretch its own limit. The result
    is clamped to ``[gap_min_days, gap_max_days]``. With fewer than
    ``gap_min_observations`` gaps there is nothing to learn from, so the fixed
    ``gap_default_days`` is used.
    """
    if len(baseline_gaps) < cfg.gap_min_observations:
        return float(cfg.gap_default_days), "default (too few transactions to learn a pattern)"
    med = statistics.median(baseline_gaps)
    mad = statistics.median(abs(g - med) for g in baseline_gaps)
    limit = med + cfg.gap_mad_mult * 1.4826 * mad
    limit = min(max(limit, cfg.gap_min_days), cfg.gap_max_days)
    return limit, f"learned (median gap {med:g} days, spread {mad:g} days)"


def _period_continuity(
    ing: IngestionResult,
    trailing_gap_days: Optional[Sequence[int]] = None,
    trailing_source: Optional[str] = None,
    cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG,
) -> list[Finding]:
    """Flag unusually long silences, judged against the company's own rhythm.

    The baseline is the company's trailing statements when supplied
    (``trailing_gap_days``), otherwise the gaps in this statement itself.
    ``trailing_source`` is a short human-readable description of where the trailing
    history came from; it is quoted in the evidence so a reviewer can audit it.
    """
    findings: list[Finding] = []
    dated = [t for t in ing.transactions if t.date]
    own_gaps = gaps_between_active_days([t.date for t in dated])
    if trailing_gap_days is not None and len(trailing_gap_days) >= cfg.gap_min_observations:
        limit, basis = gap_limit(list(trailing_gap_days), cfg)
        basis = "trailing statements" + (f" ({trailing_source})" if trailing_source else "") \
            + ", " + basis
    else:
        limit, basis = gap_limit(own_gaps, cfg)
    for a, b in zip(dated, dated[1:]):
        gap = (b.date - a.date).days
        if gap > limit:
            findings.append(Finding(
                agent=AGENT, code="FL_PERIOD_GAP",
                title="Unexplained gap in statement period",
                severity=Severity.MEDIUM, weight=12, page=b.page,
                evidence=(f"{gap}-day gap between {a.date} and {b.date} with no "
                          f"transactions. This account's normal limit is about "
                          f"{limit:.0f} days ({basis}). A silence this long may "
                          f"indicate removed pages/rows."),
            ))
    return findings


def period_months(start: date, end: date) -> int:
    """Whole calendar months covered by a statement period, END DATE INCLUSIVE.

    Real statements end on the last day of the month (1 Apr to 30 Sep is six
    months of data), so we measure from ``start`` up to the day AFTER ``end``.
    Counting the end date as inclusive also keeps the older "start + N months"
    style correct: 1 Jan to 1 Jul still counts as 6. A final month that is not
    complete (end day before the start day) is not counted.
    """
    try:
        stop = end + timedelta(days=1)
    except OverflowError:  # end == date.max; no real statement gets here
        stop = end
    months = (stop.year - start.year) * 12 + (stop.month - start.month)
    if stop.day < start.day:
        months -= 1
    return months


def _period_length(ing: IngestionResult,
                   cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    """Flag when the declared statement period falls outside the required
    6-12 month window. Missing dates are handled elsewhere (DQ_MISSING_PERIOD),
    so we emit nothing when either bound is absent."""
    findings: list[Finding] = []
    start = ing.statement_period_start
    end = ing.statement_period_end
    if start is None or end is None:
        return findings

    months = period_months(start, end)

    if months < cfg.min_period_months or months > cfg.max_period_months:
        too_short = months < cfg.min_period_months
        direction = "too short" if too_short else "too long"
        consequence = ("selectively removed pages" if too_short
                       else "padded pages")
        findings.append(Finding(
            agent=AGENT, code="FL_PERIOD_LENGTH",
            title="Statement period outside the required 6\u201312 month window",
            severity=Severity.MEDIUM, weight=12,
            evidence=(f"Statement covers {start} to {end} ({months} months); "
                      f"corporate-loan policy requires "
                      f"{cfg.min_period_months}\u2013{cfg.max_period_months} months of "
                      f"statements. A {direction} period may indicate "
                      f"{consequence}."),
        ))
    return findings


def _segment_periods(ing: IngestionResult):
    """Return the per-segment periods used by the month-coverage checks.

    Falls back to the single top-level statement period when the ingestion has
    no explicit segments (older results) or only one, so single-period
    statements behave exactly as before. Each item is
    ``(period_start, period_end, segment_label)`` where the label is used only
    for human-readable evidence, never for month identity.
    """
    periods = []
    if ing.segments:
        # A MULTI-FILE submission has segments originating from distinct files
        # (distinct ``source`` basenames); a single-file / combined PDF has one
        # source shared by every segment. When the sources genuinely differ we
        # label each segment by its file name so duplicate-month evidence can
        # say WHICH files collide; otherwise we keep the existing page-based
        # wording for the single-file combined-PDF case.
        distinct_sources = {
            seg.source for seg in ing.segments if seg.source is not None
        }
        use_source_labels = len(distinct_sources) > 1
        for i, seg in enumerate(ing.segments, start=1):
            if use_source_labels and seg.source is not None:
                label = seg.source
            elif seg.page_start is not None and seg.page_end is not None:
                if seg.page_start == seg.page_end:
                    label = f"page {seg.page_start}"
                else:
                    label = f"pages {seg.page_start}\u2013{seg.page_end}"
            else:
                label = f"segment {i}"
            periods.append((seg.period_start, seg.period_end, label))
    else:
        periods.append((
            ing.statement_period_start,
            ing.statement_period_end,
            "statement period",
        ))
    return periods


def _month_missing(ing: IngestionResult) -> list[Finding]:
    """Flag calendar months absent from the interior of the covered range."""
    periods = _segment_periods(ing)
    bounds = [(s, e) for s, e, _ in periods]
    gaps = missing_months(bounds)
    if not gaps:
        return []

    covered = sorted(
        ym for s, e in bounds if s is not None and e is not None
        for ym in months_in_period(s, e)
    )
    covered_set = {ym for ym in covered}
    findings: list[Finding] = []
    for gap in gaps:
        gap_idx = gap[0] * 12 + (gap[1] - 1)
        before = max((ym for ym in covered_set
                      if ym[0] * 12 + (ym[1] - 1) < gap_idx), default=None,
                     key=lambda ym: ym[0] * 12 + (ym[1] - 1))
        after = min((ym for ym in covered_set
                     if ym[0] * 12 + (ym[1] - 1) > gap_idx), default=None,
                    key=lambda ym: ym[0] * 12 + (ym[1] - 1))
        between = ""
        if before is not None and after is not None:
            between = f" between {month_name(before)} and {month_name(after)}"
        findings.append(Finding(
            agent=AGENT, code="FL_MONTH_MISSING",
            title="Missing month in statement coverage",
            severity=Severity.MEDIUM, weight=12,
            evidence=(f"{month_name(gap)} is missing{between}. The submitted "
                      f"statements skip a calendar month; a removed month can hide "
                      f"adverse activity and must be explained."),
        ))
    return findings


def _month_duplicate(ing: IngestionResult) -> list[Finding]:
    """Flag a calendar month covered by more than one segment/file (or
    overlapping date ranges, which always share at least one month)."""
    periods = _segment_periods(ing)
    labelled = [(s, e, label) for s, e, label in periods]
    dupes = duplicate_months(labelled)
    findings: list[Finding] = []
    for ym, sources in dupes:
        # Sort the colliding source labels so the evidence (and therefore the
        # finding) is identical regardless of the order the files/segments were
        # submitted in — this keeps FL_MONTH_DUPLICATE order-independent.
        where = ", ".join(sorted(str(s) for s in sources))
        findings.append(Finding(
            agent=AGENT, code="FL_MONTH_DUPLICATE",
            title="Duplicate / overlapping month in statement coverage",
            severity=Severity.MEDIUM, weight=12,
            evidence=(f"{month_name(ym)} is covered more than once ({where}). "
                      f"Duplicated or overlapping statement periods may indicate a "
                      f"resubmitted or doctored month."),
        ))
    return findings


def _month_out_of_order(ing: IngestionResult) -> list[Finding]:
    """Flag segments within a single combined PDF that appear in a
    non-chronological page order (a later page covers an earlier month).

    Page-order inversion is only meaningful for a genuine in-file layout where
    the segments occupy DISTINCT, physically-ordered pages (a combined PDF). In
    a multi-file submission each standalone monthly file parses to a single
    segment with the same ``page_start`` (all == 1), so list position is just
    the arbitrary upload order and must NEVER drive a finding; that is the
    whole point of the order-independence requirement. We therefore only run the
    check when there are >= 2 segments whose ``page_start`` values are all
    present and not all identical, and we iterate in true physical page order
    (sorted by ``page_start``) rather than list-insertion order."""
    if len(ing.segments) < 2:
        return []
    page_starts = [seg.page_start for seg in ing.segments]
    if any(ps is None for ps in page_starts) or len(set(page_starts)) < 2:
        # No distinct physical page positions: multi-file aggregate (all the
        # same page) or missing page info. Coverage then depends only on the
        # parsed dates, keeping multi-file screening order-independent.
        return []
    ordered_segments = sorted(ing.segments, key=lambda seg: seg.page_start)
    ordered = [
        (seg.period_start, seg.period_end, seg.page_start)
        for seg in ordered_segments
    ]
    inversions = out_of_order_segments(ordered)
    findings: list[Finding] = []
    for prev_page, prev_ym, curr_page, curr_ym in inversions:
        findings.append(Finding(
            agent=AGENT, code="FL_MONTH_OUT_OF_ORDER",
            title="Statement months out of chronological order",
            severity=Severity.MEDIUM, weight=12, page=curr_page,
            evidence=(f"Page {curr_page} covers {month_name(curr_ym)} but appears "
                      f"after page {prev_page} which covers the later month "
                      f"{month_name(prev_ym)}. Out-of-order monthly sections in a "
                      f"combined PDF may indicate inserted or reordered pages."),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Credit baseline: what is a normal credit, and a normal month, for THIS company?
# --------------------------------------------------------------------------- #
# A bank judges a deposit against the customer's own moving history, not against one
# fixed number. The baseline comes from, in order of preference:
#   1. the company's earlier human-approved statements (``history``);
#   2. this statement's own complete months (median, so one spike cannot move it);
#   3. a declared per-UEN monthly figure in ``CASHFLOW_BASELINE`` (old 3x rule).
_DOCS_TO_REQUEST = ("Suggested request: the invoice or contract (or purchase order) behind "
                    "this payment, and who the payer is and why they paid.")


@dataclass(frozen=True)
class CreditBaseline:
    typical_monthly: float                 # median total credits of a complete month
    sizes: tuple[float, ...]               # sizes of the company's credits (empty if unknown)
    source: str                            # where this came from, quoted in the evidence
    declared_only: bool = False            # True when only a declared monthly figure exists
    from_statement: bool = False           # True when ``sizes`` come from the statement itself
    typical_credit: float = 0.0            # median credit size
    fence: float = 0.0                     # upper end of normal credit sizes


def _make_baseline(monthly: float, sizes: Sequence[float], source: str,
                   cfg: FinancialLogicConfig, **flags) -> CreditBaseline:
    if not sizes:
        return CreditBaseline(float(monthly), (), source, **flags)
    med = statistics.median(sizes)
    mad = statistics.median(abs(x - med) for x in sizes)
    return CreditBaseline(float(monthly), tuple(sizes), source, typical_credit=med,
                          fence=med + cfg.spike_mad_mult * 1.4826 * mad, **flags)


def _credit_baseline(ing: IngestionResult, history: Optional[TrailingHistory],
                     cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> Optional[CreditBaseline]:
    if history is not None and len(history.credits) >= cfg.credit_min_obs \
            and len(history.monthly_credits) >= cfg.credit_min_months:
        return _make_baseline(statistics.median(history.monthly_credits), history.credits,
                              f"trailing statements ({history.describe()})", cfg)
    sizes = credit_sizes(ing.transactions)
    monthly = monthly_credit_totals(
        ing.transactions, ing.statement_period_start, ing.statement_period_end)
    if len(sizes) >= cfg.credit_min_obs and len(monthly) >= cfg.credit_min_months:
        return _make_baseline(statistics.median(monthly), sizes,
                              f"this statement's {len(monthly)} complete months", cfg,
                              from_statement=True)
    declared = CASHFLOW_BASELINE.get(ing.uen or "", {}).get("avg_monthly_credit")
    if declared and declared > 0:
        return _make_baseline(declared, (), "the declared per-UEN monthly baseline", cfg,
                              declared_only=True)
    return None


def _large_how(amount: float, base: CreditBaseline, cfg: FinancialLogicConfig,
               relaxed: bool = False) -> Optional[str]:
    """None unless ``amount`` is large for this company; else a phrase saying how large.

    Large means above the company's normal credit sizes AND a big share of a normal
    month of credits. ``relaxed`` drops the month-share condition (used when another
    clue, such as a shell-like name, already points at the credit). With only a declared
    monthly figure, the old rule applies: more than ``declared_baseline_mult`` times it.
    """
    if base.typical_monthly <= 0:
        return None
    if base.declared_only:
        if amount <= cfg.declared_baseline_mult * base.typical_monthly:
            return None
        return (f"{amount / base.typical_monthly:.1f}x the declared monthly credit "
                f"({base.typical_monthly:,.0f})")
    if amount <= base.fence:
        return None
    if not relaxed and amount < cfg.spike_month_share * base.typical_monthly:
        return None
    return (f"{amount / base.typical_credit:.1f}x this company's normal credit "
            f"({base.typical_credit:,.0f}) and {amount / base.typical_monthly:.0%} of its "
            f"normal month of credits ({base.typical_monthly:,.0f})")


def _sudden_deposits(ing: IngestionResult, base: Optional[CreditBaseline],
                     passed_through: set[int],
                     cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    """Credits that are large for THIS company, as a prompt to ask for documents.

    That is a request for invoices or contracts, not an accusation, so the severity is
    MEDIUM. It is raised to HIGH only when the money also left again within days
    (``passed_through``, see FL_RAPID_PASS_THROUGH).
    """
    if base is None:
        return []
    findings: list[Finding] = []
    for t in ing.transactions:
        if t.amount is None or t.amount <= 0:
            continue
        how = _large_how(t.amount, base, cfg)
        if how is None:
            continue
        also = t.index in passed_through
        findings.append(Finding(
            agent=AGENT, code="FL_SUDDEN_DEPOSIT",
            title="Deposit unusually large for this company",
            severity=Severity.HIGH if also else Severity.MEDIUM,
            weight=20 if also else 12, page=t.page, field=f"row {t.index}",
            evidence=(f"Credit of {t.amount:,.2f} on {t.date} ({t.description}) is {how}, "
                      f"judged against {base.source}. "
                      + ("The money also left the account within days (see "
                         "FL_RAPID_PASS_THROUGH). " if also else "")
                      + _DOCS_TO_REQUEST),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Velocity: large money in, almost the same money out again within days
# --------------------------------------------------------------------------- #
def _pass_through_pairs(ing: IngestionResult, base: Optional[CreditBaseline],
                        cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG):
    """[(credit, [debits], total_out)] for large credits that leave again within days.

    "Large" is relative to the company: a multiple of its normal credit (from the
    baseline when there is one, otherwise this statement). Each debit is matched to at
    most one credit, earliest credit first.
    """
    sizes = list(base.sizes) if base is not None and base.sizes else credit_sizes(ing.transactions)
    if len(sizes) < cfg.pass_min_credits:
        return []
    cutoff = cfg.pass_large_mult * statistics.median(sizes)
    exclude = (re.compile(r"\b(?:" + "|".join(re.escape(w) for w in cfg.pass_exclude_words)
                          + r")\b", re.IGNORECASE) if cfg.pass_exclude_words else None)
    dated = sorted((t for t in ing.transactions if t.date is not None and t.amount is not None),
                   key=lambda t: (t.date, t.index))
    credits = [t for t in dated if t.amount > 0 and t.amount >= cutoff]
    debits = [t for t in dated if t.amount < 0
              and not (exclude and exclude.search(t.description or ""))]
    window = timedelta(days=cfg.pass_window_days)
    used: set[int] = set()
    pairs = []
    for c in credits:
        # The money must leave AFTER it arrived: a debit printed above the credit
        # (same day, earlier row) cannot have been paid out of it.
        cands = [d for d in debits
                 if d.index not in used and (d.date, d.index) > (c.date, c.index)
                 and d.date <= c.date + window]
        cands = sorted(cands, key=lambda d: d.amount)[:cfg.pass_max_candidates]   # biggest outflows
        best, best_gap = None, None
        for k in range(1, cfg.pass_max_payments + 1):
            for combo in itertools.combinations(cands, k):
                total = -sum(d.amount for d in combo)
                if cfg.pass_match_ratio * c.amount <= total <= cfg.pass_max_ratio * c.amount:
                    gap = abs(total - c.amount)
                    if best_gap is None or gap < best_gap:
                        best, best_gap = (combo, total), gap
        if best is not None:
            combo, total = best
            used.update(d.index for d in combo)
            pairs.append((c, sorted(combo, key=lambda d: (d.date, d.index)), total))
    return pairs


def _pass_through(pairs, cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    if not pairs:
        return []
    shown = pairs[:cfg.pass_max_shown]
    parts = []
    for c, out, total in shown:
        days = (max(d.date for d in out) - c.date).days
        parts.append(f"row {c.index}: {c.date.strftime('%d %b %Y')} credit {c.amount:,.2f} "
                     f"({c.description}), then {total:,.2f} out within {days} day(s) "
                     f"(row{'s' if len(out) > 1 else ''} {', '.join(str(d.index) for d in out)})")
    more = f", and {len(pairs) - len(shown)} more" if len(pairs) > len(shown) else ""
    rows = sorted({c.index for c, _, _ in pairs} | {d.index for _, o, _ in pairs for d in o})
    many = len(pairs) >= cfg.pass_escalate_pairs
    return [Finding(
        agent=AGENT, code="FL_RAPID_PASS_THROUGH",
        title="Large credits leave the account again within days",
        severity=Severity.HIGH if many else Severity.MEDIUM, weight=20 if many else 12,
        page=pairs[0][0].page, field="rows " + ", ".join(str(r) for r in rows),
        evidence=(f"{len(pairs)} large credit(s) were followed within {cfg.pass_window_days} "
                  f"days by payments out of {cfg.pass_match_ratio:.0%}-{cfg.pass_max_ratio:.0%} "
                  f"of the amount: "
                  + "; ".join(parts) + more + ". Money that arrives and leaves this fast can "
                  "be pass-through activity, but refunds and agency or trading flows can look "
                  "the same. Suggested request: the contract or invoice for both the incoming "
                  "and the outgoing payment, and the relationship with each counterparty."),
    )]


# --------------------------------------------------------------------------- #
# Large inflow from a name never seen before
# --------------------------------------------------------------------------- #
def _unverified_counterparty(ing: IngestionResult, base: Optional[CreditBaseline],
                             history: Optional[TrailingHistory],
                             cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    """A large credit from a payer that appears nowhere else in the history.

    "Seen before" means the payer's name (rail words, references and company suffixes
    removed, whole words only) occurs on another row of this statement or in the
    company's approved earlier statements. A credit is "large" for this company by the
    same test as FL_SUDDEN_DEPOSIT; a shell-like word in the name (UNKNOWN, LLC, ...)
    relaxes that to "above the company's normal credit sizes". With no baseline at all
    there is nothing to measure against, so a shell-like name AND the fallback amount
    are both required.
    """
    known_names = frozenset(history.counterparties) if history is not None else frozenset()
    keys = [counterparty_key(t.description) for t in ing.transactions]
    seen_here = Counter(k for k in keys if k)
    shell_re = (re.compile(r"\b(?:" + "|".join(re.escape(w) for w in cfg.counterparty_shell_words)
                           + r")\b", re.IGNORECASE) if cfg.counterparty_shell_words else None)
    vendor_res = [re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE) for v in KNOWN_VENDORS]
    where = ("the company's approved earlier statements or on any other row of this statement"
             if known_names else "any other row of this statement")
    findings: list[Finding] = []
    for t, key in zip(ing.transactions, keys):
        if t.amount is None or t.amount <= 0 or not key:
            continue
        if key in known_names or seen_here[key] > 1:
            continue
        desc = (t.description or "").strip()
        if any(r.search(desc) for r in vendor_res):
            continue
        shell = sorted({m.group(0).upper() for m in shell_re.finditer(desc)}) if shell_re else []
        if base is None:
            if not shell or t.amount < cfg.counterparty_fallback_amount:
                continue
            how = (f"at least the fixed {cfg.counterparty_fallback_amount:,.0f} used when no "
                   f"baseline for the company exists")
        else:
            how = _large_how(t.amount, base, cfg, relaxed=bool(shell))
            if how is None:
                continue
        clue = (f" The name also contains {', '.join(shell)}." if shell else "")
        findings.append(Finding(
            agent=AGENT, code="FL_UNVERIFIED_COUNTERPARTY",
            title="Large inflow from a payer not seen before",
            severity=Severity.MEDIUM, weight=12, page=t.page, field=f"row {t.index}",
            evidence=(f"Credit of {t.amount:,.2f} on {t.date} from '{desc}' is {how}. The payer "
                      f"'{key}' does not appear on {where}.{clue} Suggested request: who the "
                      f"payer is (registration and relationship to the company) and the "
                      f"invoice or contract behind the payment."),
        ))
    return findings


# --------------------------------------------------------------------------- #
# Structuring: clusters of credits just under a reporting level
# --------------------------------------------------------------------------- #
# FL_STRUCTURING fires only on a CLUSTER: ``structuring_min_credits`` or more credits
# just under a reporting level (lo <= amount < level) inside a span of
# ``structuring_cluster_days`` days (last - first <= 14, inclusive). An undated
# near-level credit cannot be placed in time, so it counts as inside every span; 3+
# undated credits fire on their own. Spans start at each dated near-level credit in
# date order and the earliest qualifying span is reported (independent of file row
# order). Accepted trade-off: slow structuring spread over weeks is missed.
#
# When a baseline for the company exists, two more tests apply, because the same cluster
# means different things to different companies:
#   * the cluster must total at least ``structuring_month_share`` of a normal month of
#     credits (three 9,700 credits are noise to a company that banks millions a month);
#   * near-level credits must not already be ``structuring_routine_share`` of the
#     company's OTHER credits (a business with fixed 9,800 fees is not structuring).
# With no baseline the absolute rule above is all there is.
def _structuring_span(near: list[Transaction],
                      cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG):
    """Return (rows, start, end) for the earliest qualifying span, or None."""
    undated = [t for t in near if t.date is None]
    dated = sorted((t for t in near if t.date is not None),
                   key=lambda t: (t.date, t.index))
    limit = timedelta(days=cfg.structuring_cluster_days)
    for first in dated:
        inside = [t for t in dated if first.date <= t.date <= first.date + limit]
        if len(inside) + len(undated) >= cfg.structuring_min_credits:
            rows = sorted(inside + undated, key=lambda t: t.index)
            return rows, first.date, inside[-1].date
    if len(undated) >= cfg.structuring_min_credits:
        return sorted(undated, key=lambda t: t.index), None, None
    return None


def _structuring(ing: IngestionResult, base: Optional[CreditBaseline] = None,
                 cfg: FinancialLogicConfig = DEFAULT_LOGIC_CONFIG) -> list[Finding]:
    findings: list[Finding] = []
    credits = [t for t in ing.transactions if t.amount is not None and t.amount > 0]
    for level in sorted(set(cfg.structuring_levels)):
        lo = level * (1 - cfg.structuring_band)
        span = _structuring_span([t for t in credits if lo <= t.amount < level], cfg)
        if span is None:
            continue
        near, start, end = span
        relative = ""
        if base is not None and base.typical_monthly > 0:
            total = sum(t.amount for t in near)
            if total < cfg.structuring_month_share * base.typical_monthly:
                continue                                  # small next to this company's size
            others = list(base.sizes)
            if base.from_statement:                       # do not let the cluster vouch for itself
                for t in near:
                    if t.amount in others:
                        others.remove(t.amount)
            if len(others) >= cfg.structuring_min_others and \
                    sum(1 for x in others if lo <= x < level) / len(others) \
                    >= cfg.structuring_routine_share:
                continue                                  # near-level credits are this company's norm
            relative = (f" The cluster totals {total:,.0f}, {total / base.typical_monthly:.0%} of "
                        f"this company's normal month of credits ({base.typical_monthly:,.0f}), "
                        f"judged against {base.source}.")
        when = (f" within {start.isoformat()} to {end.isoformat()}"
                if start is not None else " (undated rows)")
        rows = ", ".join(f"#{t.index}({t.amount:,.0f})" for t in near)
        findings.append(Finding(
            agent=AGENT, code="FL_STRUCTURING",
            title="Possible structuring below reporting threshold",
            severity=Severity.HIGH, weight=20, page=near[0].page,
            evidence=(f"{len(near)} deposits fall in {lo:,.0f}-{level:,.0f}{when} "
                      f"(just under the {level:,.0f} reporting threshold): "
                      f"{rows}. Repeated near-threshold amounts suggest deliberate splitting."
                      f"{relative}"),
        ))
    return findings


def _entity_crosscheck(ing: IngestionResult) -> list[Finding]:
    """Check the statement's UEN and holder name against the company registry.

    The lookup goes through ``registry.lookup_entity``: local fixtures first, then the
    live data.gov.sg registry when ``ACRA_LIVE_LOOKUP`` is on. A live lookup that could
    not answer is reported as a LOW data-quality note, never as "UEN not found".
    """
    findings: list[Finding] = []
    if not ing.uen:
        return findings
    result = lookup_entity(ing.uen)
    if result.state == REGISTRY_NOT_FOUND:
        where = "the local registry and the live ACRA registry (data.gov.sg)" \
            if result.live_checked else "the registry"
        findings.append(Finding(
            agent=AGENT, code="FL_UEN_NOT_FOUND",
            title="UEN not found in registry",
            severity=Severity.MEDIUM, weight=10,
            evidence=f"UEN {ing.uen} has no matching ACRA record in {where}.",
        ))
    elif result.state == REGISTRY_UNAVAILABLE:
        findings.append(Finding(
            agent=AGENT, code="DQ_REGISTRY_UNAVAILABLE",
            title="Company registry could not be checked",
            severity=Severity.LOW, weight=2,
            evidence=(f"The live ACRA registry could not be reached ({result.detail}), so UEN "
                      f"{ing.uen} was not verified. This says nothing about the company."),
        ))
    elif result.record is not None and ing.account_holder \
            and not names_match(result.record.name, ing.account_holder):
        findings.append(Finding(
            agent=AGENT, code="FL_UEN_NAME_MISMATCH",
            title="Account holder does not match ACRA record",
            severity=Severity.HIGH, weight=22,
            evidence=(f"Statement holder '{ing.account_holder}' does not match the ACRA "
                      f"record '{result.record.name}' for UEN {ing.uen}. May indicate a "
                      f"statement reused under a different company."),
        ))
    return findings


def analyze(ing: IngestionResult, fabrication_config=None,
            trailing_gap_days: Optional[Sequence[int]] = None,
            trailing_source: Optional[str] = None,
            history: Optional[TrailingHistory] = None,
            config: Optional[FinancialLogicConfig] = None) -> AgentReport:
    """Run all Node 3 checks.

    ``history`` is what the case store knows about this company from earlier,
    human-approved statements. It feeds the gap, deposit, pass-through, counterparty
    and structuring checks. ``trailing_gap_days`` / ``trailing_source`` override the gap
    part only. ``config`` holds every threshold (default: ``DEFAULT_LOGIC_CONFIG``).
    """
    cfg = DEFAULT_LOGIC_CONFIG if config is None else config
    if history is not None and history.case_ids:
        if trailing_gap_days is None:
            trailing_gap_days = history.gap_days
        if trailing_source is None:
            trailing_source = history.describe()
    else:
        history = None
    base = _credit_baseline(ing, history, cfg)
    findings: list[Finding] = []
    findings += _reconcile(ing, cfg)
    findings += _closing_check(ing, cfg)
    findings += _period_continuity(ing, trailing_gap_days, trailing_source, cfg)
    findings += _period_length(ing, cfg)
    findings += _month_missing(ing)
    findings += _month_duplicate(ing)
    findings += _month_out_of_order(ing)
    pairs = _pass_through_pairs(ing, base, cfg)
    passed_through = {c.index for c, _, _ in pairs}
    findings += _sudden_deposits(ing, base, passed_through, cfg)
    findings += _pass_through(pairs, cfg)
    findings += _unverified_counterparty(ing, base, history, cfg)
    findings += _structuring(ing, base, cfg)
    findings += _entity_crosscheck(ing)
    findings += fabrication.analyze_fabrication(
        ing, DEFAULT_CONFIG if fabrication_config is None else fabrication_config
    )
    return AgentReport(agent=AGENT, findings=findings)
