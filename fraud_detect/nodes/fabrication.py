"""Node 3 fabrication rules: configuration and invoice-reference reading.

Task 2.2 covers only ``FabricationConfig``, ``DEFAULT_CONFIG``, ``CreditRef``
and the reference/counterparty readers. The rules themselves come later.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
import logging
from datetime import date
from functools import lru_cache
from typing import Optional, Sequence

from ..models import Finding, Severity, Transaction

AGENT = "Node3:FinancialLogic"

_MAX_DESCRIPTION_CHARS = 500


@dataclass(frozen=True)
class FabricationConfig:
    # --- reading references and customers ---
    # Each pattern has group 1 = prefix and group 2 = number. Case-insensitive.
    invoice_patterns: tuple[str, ...] = (r"\b(INV)[\s\-#]*0*(\d{3,})\b",)
    rail_prefixes: tuple[str, ...] = ("FAST FROM", "PAYNOW FROM", "GIRO", "TT FROM")

    # --- Rule A: duplicate invoice reference ---
    dup_escalate_min_refs: int = 3
    dup_isolated_weight: float = 10.0
    dup_escalated_weight: float = 22.0
    dup_min_payments_same_month: int = 2      # payments of one invoice in ONE calendar month that count as a repeat
    dup_cross_month_weight: float = 3.0       # LOW note for repeats that do not meet that test (e.g. across a month end)
    dup_far_apart_weight: float = 10.0        # MEDIUM for cross-month repeats that cannot be an instalment (too far apart / too many)
    instalment_max_days: int = 45
    instalment_max_count: int = 3
    evidence_max_groups: int = 20

    # --- Rule B: invoice sequence break ---
    seq_min_refs: int = 6
    seq_min_breaks: int = 2
    seq_weight: float = 5.0

    # --- Rule C: turnover against fixed costs ---
    fixed_cost_keywords: tuple[str, ...] = ("CPF", "PAYROLL", "SALARY", "RENT")
    turnover_exclude_keywords: tuple[str, ...] = ("LOAN", "CAPITAL", "TRANSFER", "INTEREST")
    credit_jump_ratio: float = 1.2
    fixed_cost_stability_pct: float = 0.10
    turnover_min_months: int = 4
    turnover_min_stable_share: float = 2 / 3
    turnover_min_jump_months: int = 2
    turnover_weight: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "dup_escalate_min_refs",
            "instalment_max_count",
            "seq_min_refs",
            "seq_min_breaks",
            "turnover_min_months",
            "turnover_min_jump_months",
            "evidence_max_groups",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}")
        if isinstance(self.dup_min_payments_same_month, bool) or \
                not isinstance(self.dup_min_payments_same_month, int) or \
                self.dup_min_payments_same_month < 2:
            raise ValueError("dup_min_payments_same_month must be an integer >= 2")
        if self.instalment_max_days < 0:
            raise ValueError("instalment_max_days must be >= 0")
        if not self.credit_jump_ratio > 1:
            raise ValueError("credit_jump_ratio must be greater than 1")
        if not 0 <= self.fixed_cost_stability_pct < 1:
            raise ValueError("fixed_cost_stability_pct must be >= 0 and < 1")
        if not 0 < self.turnover_min_stable_share <= 1:
            raise ValueError("turnover_min_stable_share must be > 0 and <= 1")
        for name in ("dup_isolated_weight", "dup_escalated_weight", "dup_cross_month_weight", "dup_far_apart_weight",
                     "seq_weight", "turnover_weight"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0")
        for p in self.invoice_patterns:  # fail early on bad patterns
            re.compile(p, re.IGNORECASE)


DEFAULT_CONFIG = FabricationConfig()


@dataclass(frozen=True)
class CreditRef:
    row: int
    page: Optional[int]
    date: Optional[date]
    amount: float
    ref: str
    ref_number: int
    counterparty: str


@lru_cache(maxsize=64)
def _compile(patterns: tuple[str, ...]) -> tuple["re.Pattern[str]", ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def _clean(description: object) -> str:
    if not isinstance(description, str):
        return ""
    return description[:_MAX_DESCRIPTION_CHARS]


def _match(description: object, cfg: FabricationConfig):
    text = _clean(description)
    if not text:
        return text, None
    for pat in _compile(cfg.invoice_patterns):
        m = pat.search(text)
        if m:
            return text, m
    return text, None


def extract_invoice_ref(description: Optional[str], cfg: FabricationConfig) -> Optional[tuple[str, int]]:
    """Return ``(UPPER_PREFIX, int_number)`` or ``None``."""
    _, m = _match(description, cfg)
    if m is None:
        return None
    try:
        return m.group(1).upper(), int(m.group(2))
    except (ValueError, IndexError):
        return None


def normalise_counterparty(description: Optional[str], cfg: FabricationConfig) -> str:
    """Description without the reference and leading rail, upper-cased, spaces collapsed."""
    text, m = _match(description, cfg)
    if m is not None:
        text = text[: m.start()] + " " + text[m.end():]
    text = " ".join(text.upper().split())
    for rail in cfg.rail_prefixes:
        r = " ".join(rail.upper().split())
        if text == r:
            text = ""
            break
        if r and text.startswith(r + " "):
            text = text[len(r):].strip()
            break
    return text


def extract_credit_refs(txns: Sequence[Transaction], cfg: FabricationConfig) -> list[CreditRef]:
    """Credits with a description and an invoice reference, in statement order."""
    out: list[CreditRef] = []
    for t in txns:
        amount = getattr(t, "amount", None)
        desc = getattr(t, "description", None)
        if amount is None or not amount > 0 or not isinstance(desc, str) or not desc:
            continue
        got = extract_invoice_ref(desc, cfg)
        if got is None:
            continue
        prefix, number = got
        out.append(
            CreditRef(
                row=t.index,
                page=getattr(t, "page", None),
                date=getattr(t, "date", None),
                amount=amount,
                ref=f"{prefix}{number}",
                ref_number=number,
                counterparty=normalise_counterparty(desc, cfg),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Rule B: invoice sequence break
# --------------------------------------------------------------------------- #
def _dated_sorted(refs: Sequence[CreditRef]) -> list[CreditRef]:
    """Dated references ordered by (date, reference number, row)."""
    dated = [r for r in refs if r.date is not None]
    return sorted(dated, key=lambda r: (r.date, r.ref_number, r.row))


def _kept_positions(values: Sequence[int]) -> list[int]:
    """Positions of the longest non-decreasing subsequence.

    Among equal-length longest subsequences, the one whose positions are
    lexicographically smallest. O(n log n).
    """
    n = len(values)
    if n == 0:
        return []
    # lens[i] = longest non-decreasing subsequence starting at i
    lens = [0] * n
    tails: list[int] = []
    for i in range(n - 1, -1, -1):
        x = -values[i]
        k = bisect_right(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
        lens[i] = k + 1
    need = len(tails)
    kept: list[int] = []
    prev: Optional[int] = None
    for i in range(n):
        if need == 0:
            break
        if lens[i] == need and (prev is None or values[i] >= prev):
            kept.append(i)
            prev = values[i]
            need -= 1
    return kept


def find_order_breaks(refs: Sequence[CreditRef], cfg: FabricationConfig) -> set[int]:
    """Row numbers of dated credits outside the longest in-order run."""
    ordered = _dated_sorted(refs)
    kept = set(_kept_positions([r.ref_number for r in ordered]))
    return {r.row for i, r in enumerate(ordered) if i not in kept}


def check_invoice_sequence(txns: Sequence[Transaction], cfg: FabricationConfig) -> list[Finding]:
    """One LOW finding per file when invoice numbers run out of date order."""
    ordered = _dated_sorted(extract_credit_refs(txns, cfg))
    if len(ordered) < cfg.seq_min_refs:
        return []
    kept = set(_kept_positions([r.ref_number for r in ordered]))
    named = [r for i, r in enumerate(ordered) if i not in kept]
    if len(named) < cfg.seq_min_breaks:
        return []

    by_row = sorted(named, key=lambda r: r.row)
    latest = max(named, key=lambda r: (r.date, r.row))
    field = f"row {by_row[0].row}" if len(by_row) == 1 else "rows " + ", ".join(str(r.row) for r in by_row)
    parts = [
        f"row {r.row}: {r.date.strftime('%d %b %Y')} {r.ref} S${r.amount:,.2f}"
        for r in by_row
    ]
    evidence = (
        f"{len(by_row)} credit(s) carry invoice numbers that go out of date order: "
        + "; ".join(parts)
        + ". This is a supporting hint only, because customers can pay late or out of order."
    )
    return [
        Finding(
            agent=AGENT,
            code="FL_INVOICE_SEQUENCE_BREAK",
            title="Invoice numbers out of date order",
            severity=Severity.LOW,
            weight=cfg.seq_weight,
            evidence=evidence,
            page=latest.page,
            field=field,
        )
    ]


# --------------------------------------------------------------------------- #
# Rule A: duplicate invoice reference
# --------------------------------------------------------------------------- #
_INSTALMENT_NOTE = "may be an instalment payment; check the invoice"
_IDENTICAL_NOTE = "identical amount, possible duplicate posting"


def _occurrence(r: CreditRef) -> str:
    when = r.date.strftime("%d %b %Y") if r.date is not None else "no date"
    where = f"page {r.page}, row {r.row}" if r.page is not None else f"row {r.row}"
    return f"{when} S${r.amount:,.2f} ({where})"


def _join_and(items: Sequence[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _group_text(rows: Sequence[CreditRef], cfg: FabricationConfig, likely: frozenset) -> str:
    first = rows[0]
    label = f"{first.counterparty} {first.ref}".strip()
    text = f"{label}: {_join_and([_occurrence(r) for r in rows])}."
    if len({r.amount for r in rows}) == 1:
        text += f" Note: {_IDENTICAL_NOTE}."
    dates = [r.date for r in rows if r.date is not None]
    if len(dates) >= 2 and len(rows) <= cfg.instalment_max_count:
        if (max(dates) - min(dates)).days <= cfg.instalment_max_days:
            text += f" Note: this {_INSTALMENT_NOTE}."
    flagged = [r.row for r in rows if r.row in likely]
    if len(flagged) == 1:
        text += f" Hint only: row {flagged[0]} breaks the invoice number order, so it is the likelier copy."
    return text


def _month_key(r: CreditRef):
    return (r.date.year, r.date.month) if r.date is not None else None


def _split_by_month(rows: Sequence[CreditRef], minimum: int):
    """Split the credits of one (customer, reference) into same-month groups and leftovers.

    A same-month group is ``minimum`` or more payments in one calendar month. A credit
    with no date cannot be placed in time, so it counts as inside every month.
    """
    undated = [r for r in rows if r.date is None]
    months = sorted({_month_key(r) for r in rows if r.date is not None})
    groups: list[list[CreditRef]] = []
    used: set[int] = set()
    for m in months:
        members = [r for r in rows if r.date is not None and _month_key(r) == m] + undated
        if len(members) >= minimum:
            members = sorted(members, key=lambda r: r.row)
            groups.append(members)
            used.update(r.row for r in members)
    if not months and len(undated) >= minimum:
        groups.append(sorted(undated, key=lambda r: r.row))
        used.update(r.row for r in undated)
    leftover = [r for r in rows if r.row not in used]
    return groups, leftover


def _instalment_like(rows: Sequence[CreditRef], cfg: FabricationConfig) -> bool:
    """Could these payments of one invoice be instalments? Few payments, close in time.

    Same limits as the "may be an instalment" note. A group with fewer than two dated
    rows cannot be judged on distance, so it gets the benefit of the doubt.
    """
    if len(rows) > cfg.instalment_max_count:
        return False
    dates = [r.date for r in rows if r.date is not None]
    if len(dates) < 2:
        return True
    return (max(dates) - min(dates)).days <= cfg.instalment_max_days


def check_duplicate_invoice_refs(
    txns: Sequence[Transaction],
    cfg: FabricationConfig,
    likely_copies: frozenset = frozenset(),
) -> list[Finding]:
    """Credits that repeat the same invoice reference under the same customer.

    The main signal is a repeat inside ONE calendar month (``dup_min_payments_same_month``
    payments or more): a monthly statement has no reason to show the same invoice paid
    twice. Repeats that do not meet that test are judged on distance: close in time and
    few payments (a payment that crosses a month end) is only a weak LOW note; far apart
    or too many payments cannot be an instalment and is MEDIUM. Neither kind counts toward
    the HIGH escalation.
    """
    likely = frozenset(likely_copies or ())
    buckets: dict[tuple[str, str], list[CreditRef]] = {}
    for r in extract_credit_refs(txns, cfg):
        buckets.setdefault((r.counterparty, r.ref), []).append(r)
    groups: list[list[CreditRef]] = []
    spread: list[list[CreditRef]] = []
    for rows in buckets.values():
        rows = sorted(rows, key=lambda r: r.row)
        if len(rows) < 2:
            continue
        same, left = _split_by_month(rows, cfg.dup_min_payments_same_month)
        groups.extend(same)
        if len(left) >= 2:
            spread.append(left)
    groups.sort(key=lambda g: g[0].row)
    spread.sort(key=lambda g: g[0].row)
    if not groups and not spread:
        return []
    cannot_tell = "This statement cannot tell which row of a group was copied from the other."

    def build(sel: Sequence[Sequence[CreditRef]], severity: Severity, weight: float,
              evidence: str, title: str = "Invoice reference used on more than one credit") -> Finding:
        named = sorted({r.row: r for g in sel for r in g}.values(), key=lambda r: r.row)
        return Finding(
            agent=AGENT,
            code="FL_DUP_INVOICE_REF",
            title=title,
            severity=severity,
            weight=weight,
            evidence=evidence,
            page=named[-1].page,
            field="rows " + ", ".join(str(r.row) for r in named),
        )

    findings: list[Finding] = []
    if groups and len(groups) < cfg.dup_escalate_min_refs:
        findings += [
            build([g], Severity.MEDIUM, cfg.dup_isolated_weight,
                  f"Same invoice paid more than once in one month. {_group_text(g, cfg, likely)} "
                  f"{cannot_tell}")
            for g in groups
        ]
    elif groups:
        shown = groups[: cfg.evidence_max_groups]
        hidden = len(groups) - len(shown)
        body = " ".join(_group_text(g, cfg, likely) for g in shown)
        head = (f"{len(groups)} invoice references are each used on several credits within "
                f"one month. {cannot_tell} ")
        evidence = head + (body.rstrip(".") + f", and {hidden} more." if hidden else body)
        findings.append(build(groups, Severity.HIGH, cfg.dup_escalated_weight, evidence))
    far = [g for g in spread if not _instalment_like(g, cfg)]
    near = [g for g in spread if _instalment_like(g, cfg)]
    if far:
        shown = far[: cfg.evidence_max_groups]
        hidden = len(far) - len(shown)
        body = " ".join(_group_text(g, cfg, frozenset()) for g in shown)
        head = (f"{len(far)} invoice reference(s) are repeated too far apart (more than "
                f"{cfg.instalment_max_days} days) or too many times (more than "
                f"{cfg.instalment_max_count} payments) to look like an instalment. "
                f"A customer rarely pays one invoice again months later. ")
        evidence = head + (body.rstrip(".") + f", and {hidden} more." if hidden else body)
        findings.append(build(far, Severity.MEDIUM, cfg.dup_far_apart_weight, evidence,
                              title="Invoice reference repeated months apart"))
    if near:
        shown = near[: cfg.evidence_max_groups]
        hidden = len(near) - len(shown)
        body = " ".join(_group_text(g, cfg, frozenset()) for g in shown)
        head = (f"{len(near)} invoice reference(s) also appear on several credits, but not "
                f"enough within one calendar month to count as a same-month repeat, and close "
                f"enough in time to be an instalment (often a payment that crosses a month "
                f"end). Weak signal only. ")
        evidence = head + (body.rstrip(".") + f", and {hidden} more." if hidden else body)
        findings.append(build(near, Severity.LOW, cfg.dup_cross_month_weight, evidence,
                              title="Invoice reference reused across months"))
    return findings


# --------------------------------------------------------------------------- #
# Rule C: turnover against fixed costs
# --------------------------------------------------------------------------- #
def _keyword_regex(words: Sequence[str]) -> Optional["re.Pattern[str]"]:
    words = [w for w in words if isinstance(w, str) and w.strip()]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w.strip()) for w in words) + r")\b", re.IGNORECASE)


def _full_months(
    seen: set[tuple[int, int]], start: Optional[date], end: Optional[date]
) -> list[tuple[int, int]]:
    """Months to judge: covered by the declared period from first to last day."""
    if start is None or end is None:
        return sorted(seen)
    out = []
    for y, m in sorted(seen):
        first = date(y, m, 1)
        last = date(y + (m == 12), m % 12 + 1, 1).toordinal() - 1
        if start <= first and end.toordinal() >= last:
            out.append((y, m))
    return out


def check_turnover_vs_costs(ing, cfg: FabricationConfig) -> list[Finding]:
    """One MEDIUM finding when turnover jumps in several months while fixed costs stay flat."""
    import statistics

    txns = getattr(ing, "transactions", None) or []
    exclude = _keyword_regex(cfg.turnover_exclude_keywords)
    fixed_re = _keyword_regex(cfg.fixed_cost_keywords)

    seen: set[tuple[int, int]] = set()
    turnover: dict[tuple[int, int], float] = {}
    fixed: dict[tuple[int, int], float] = {}
    for t in txns:
        d = getattr(t, "date", None)
        amount = getattr(t, "amount", None)
        desc = getattr(t, "description", None)
        if d is None or amount is None or not isinstance(desc, str) or not desc:
            continue
        key = (d.year, d.month)
        seen.add(key)
        text = _clean(desc)
        if amount > 0:
            if exclude is None or not exclude.search(text):
                turnover[key] = turnover.get(key, 0.0) + amount
        elif amount < 0 and fixed_re is not None and fixed_re.search(text):
            fixed[key] = fixed.get(key, 0.0) + abs(amount)

    months = _full_months(
        seen, getattr(ing, "statement_period_start", None), getattr(ing, "statement_period_end", None)
    )
    costed = [m for m in months if fixed.get(m, 0.0) > 0]
    if len(costed) < cfg.turnover_min_months or len(months) < 2:
        return []

    median_fixed = statistics.median(fixed[m] for m in costed)
    pct = cfg.fixed_cost_stability_pct
    lo, hi = median_fixed * (1 - pct), median_fixed * (1 + pct)

    def stable(m: tuple[int, int]) -> bool:
        return lo <= fixed.get(m, 0.0) <= hi

    if sum(1 for m in months if stable(m)) / len(months) < cfg.turnover_min_stable_share:
        return []

    jumps = []
    for m in months:
        if not stable(m):
            continue
        others = statistics.median(turnover.get(o, 0.0) for o in months if o != m)
        if others <= 0:
            continue
        mine = turnover.get(m, 0.0)
        if mine > cfg.credit_jump_ratio * others:
            jumps.append((m, mine, others, mine / others, fixed[m]))
    if len(jumps) < cfg.turnover_min_jump_months:
        return []

    parts = [
        f"{y}-{mo:02d}: turnover {mine:,.2f} against a median of {others:,.2f} for the other months "
        f"({ratio:.2f}x), while fixed costs were {fc:,.2f} against a median of {median_fixed:,.2f}"
        for (y, mo), mine, others, ratio, fc in jumps
    ]
    evidence = (
        f"In {len(jumps)} full month(s) credits jump while fixed costs stay flat: "
        + "; ".join(parts)
        + ". Seasonal sales or one-off contracts can look like this, so treat it as a hint for a reviewer."
    )
    return [
        Finding(
            agent=AGENT,
            code="FL_TURNOVER_COST_MISMATCH",
            title="Turnover jumps while fixed costs stay flat",
            severity=Severity.MEDIUM,
            weight=cfg.turnover_weight,
            evidence=evidence,
        )
    ]


# --------------------------------------------------------------------------- #
# Entry point with per-rule failure isolation
# --------------------------------------------------------------------------- #
_log = logging.getLogger(__name__)

_RULE_SEQ = "FL_INVOICE_SEQUENCE_BREAK"
_RULE_DUP = "FL_DUP_INVOICE_REF"
_RULE_TURN = "FL_TURNOVER_COST_MISMATCH"


def _skipped(rule_code: str, exc: BaseException) -> Finding:
    """LOW finding for a rule that failed. Carries the exception type only, never its message."""
    type_name = type(exc).__name__
    _log.warning("fabrication rule %s skipped: %s", rule_code, type_name)
    return Finding(
        agent=AGENT,
        code="DQ_RULE_SKIPPED",
        title="A fabrication check was skipped",
        severity=Severity.LOW,
        weight=2,
        evidence=(
            f"Rule {rule_code} was skipped because of an internal error ({type_name}); "
            "the other checks still ran."
        ),
        field=rule_code,
    )


def analyze_fabrication(ing, cfg: FabricationConfig = DEFAULT_CONFIG) -> list[Finding]:
    """Run rules B, A, C in that order, each isolated. Never raises.

    Findings are reported in the same order: B, A, C.
    """
    findings: list[Finding] = []
    txns = getattr(ing, "transactions", None) or []
    likely: frozenset = frozenset()

    # Rule B (and the order-break rows that Rule A uses as a hint)
    seq_failed = False
    try:
        findings += check_invoice_sequence(txns, cfg)
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        seq_failed = True
        findings.append(_skipped(_RULE_SEQ, exc))
    if not seq_failed:
        try:
            likely = frozenset(find_order_breaks(extract_credit_refs(txns, cfg), cfg))
        except Exception as exc:  # noqa: BLE001
            likely = frozenset()
            findings.append(_skipped(_RULE_SEQ, exc))

    # Rule A
    try:
        findings += check_duplicate_invoice_refs(txns, cfg, likely)
    except Exception as exc:  # noqa: BLE001
        findings.append(_skipped(_RULE_DUP, exc))

    # Rule C
    try:
        findings += check_turnover_vs_costs(ing, cfg)
    except Exception as exc:  # noqa: BLE001
        findings.append(_skipped(_RULE_TURN, exc))

    return findings
