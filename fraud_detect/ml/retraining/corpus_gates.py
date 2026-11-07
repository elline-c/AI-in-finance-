"""Corpus-composition gates (Task 6.3). Standard library only.

Every function works on in-memory tables (lists of dicts) and never writes anything.
Style columns carry the ``style_`` prefix of the Node 6b feature table. A style value of
``None`` (for example the street, which cannot be recovered from a PDF) means "unknown"
and is ignored everywhere.
"""
from __future__ import annotations

import dataclasses
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .gates import GateResult

SPLITS_CHECKED = ("train", "validation", "test")
SUBTLE_CATEGORIES = ("single_amount", "single_date", "amount_le_5pct")

# Styling choices checked by the capability gate: (name, table column).
CAPABILITY_CHOICES = (
    ("producer", "style_producer"),
    ("creator", "style_creator"),
    ("fonts", "style_fonts"),
    ("bank_name", "style_bank_name"),
    ("street", "style_street"),
    ("row_step_bucket", "style_row_step_bucket"),
    ("creation_hour", "style_creation_hour"),
)
FAMILY_COLUMN = "style_producer_family"


# ------------------------------------------------------------------------------ helpers
def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, set, frozenset)):
        return tuple(sorted(value, key=repr)) if not isinstance(value, list) else tuple(value)
    return value


def _types(row: Mapping[str, Any]) -> Tuple[str, ...]:
    value = row.get("tamper_types") or ()
    if isinstance(value, str):
        return tuple(t.strip() for t in re.split(r"[|;,]", value) if t.strip())
    return tuple(str(t) for t in value)


def _gate(gate_id: str, requirement: str, offenders: Sequence[str], waivable: bool) -> GateResult:
    return GateResult(gate_id, requirement, not offenders, tuple(offenders), waivable, None)


def _fmt(x: float) -> str:
    return f"{x:.1f}"


# ------------------------------------------------------------------------ styling balance
@dataclasses.dataclass(frozen=True)
class StyleFailure:
    column: str
    value: Any
    genuine_share: float
    tampered_share: float
    n_docs: int


def styling_balance(rows: Sequence[Mapping[str, Any]], *, min_docs: int,
                    share_points: float) -> List[StyleFailure]:
    """Values in at least ``min_docs`` documents whose genuine and tampered shares differ by
    more than ``share_points`` percentage points."""
    n_gen = sum(1 for r in rows if not _truthy(r.get("is_tampered")))
    n_tam = len(rows) - n_gen
    columns = sorted({k for r in rows for k in r if k.startswith("style_")})
    failures: List[StyleFailure] = []
    for col in columns:
        gen: Counter = Counter()
        tam: Counter = Counter()
        for r in rows:
            v = r.get(col)
            if v is None:
                continue
            v = _hashable(v)
            (tam if _truthy(r.get("is_tampered")) else gen)[v] += 1
        for value in sorted(set(gen) | set(tam), key=repr):
            total = gen[value] + tam[value]
            if total < min_docs:
                continue
            gs = 100.0 * gen[value] / n_gen if n_gen else 0.0
            ts = 100.0 * tam[value] / n_tam if n_tam else 0.0
            if abs(gs - ts) > share_points + 1e-9:
                failures.append(StyleFailure(col, value, gs, ts, total))
    return failures


def styling_balance_gate(rows: Sequence[Mapping[str, Any]], *, min_docs: int,
                         share_points: float) -> GateResult:
    offenders = [f"{f.column}={f.value}: genuine {_fmt(f.genuine_share)}%, "
                 f"tampered {_fmt(f.tampered_share)}% ({f.n_docs} documents)"
                 for f in styling_balance(rows, min_docs=min_docs, share_points=share_points)]
    return _gate("styling_balance", "5.2, 5.3", offenders, False)


# -------------------------------------------------------------------------- resaved share
def resaved_share_gate(rows: Sequence[Mapping[str, Any]], *, min_share: float) -> GateResult:
    totals: Counter = Counter()
    resaved: Counter = Counter()
    for r in rows:
        if _truthy(r.get("is_tampered")):
            continue
        split = r.get("split")
        totals[split] += 1
        if _truthy(r.get("genuine_resaved")):
            resaved[split] += 1
    offenders = []
    for split in sorted(totals, key=str):
        share = 100.0 * resaved[split] / totals[split]
        if share + 1e-9 < float(min_share):
            offenders.append(f"{split}: genuine_resaved share {_fmt(share)}% "
                             f"is below {_fmt(float(min_share))}% "
                             f"({resaved[split]} of {totals[split]} genuine documents)")
    return _gate("resaved_share", "5.5", offenders, False)


# ---------------------------------------------------------------------- tamper-type minimum
def tamper_type_minimum_gate(rows: Sequence[Mapping[str, Any]], *, min_docs: int,
                             pooled: bool = True) -> GateResult:
    counts: Counter = Counter()
    for r in rows:
        if r.get("split") != "train" or not _truthy(r.get("is_tampered")):
            continue
        for t in _types(r):
            counts[(None if pooled else r.get("corpus_key"), t)] += 1
    offenders = []
    for (corpus, t), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if n < min_docs:
            where = "pooled train" if pooled else f"train of {corpus}"
            offenders.append(f"{t}: {n} documents in {where}, minimum {min_docs}")
    return _gate("tamper_type_minimum", "6.1", offenders, False)


# ---------------------------------------------------------------------------------- coverage
def coverage_gaps(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    seen = set()
    types = set()
    for r in rows:
        if not _truthy(r.get("is_tampered")) or r.get("split") not in SPLITS_CHECKED:
            continue
        for t in _types(r):
            types.add(t)
            seen.add((t, r["split"]))
    return [(t, s) for t in sorted(types) for s in SPLITS_CHECKED if (t, s) not in seen]


def coverage_gate(rows: Sequence[Mapping[str, Any]]) -> GateResult:
    offenders = [f"{t}: no documents in {s}" for t, s in coverage_gaps(rows)]
    return _gate("tamper_type_coverage", "6.2", offenders, False)


# ----------------------------------------------------------------------------------- subtle
def _dec(text: Any) -> Optional[Decimal]:
    if text is None:
        return None
    try:
        return Decimal(str(text).replace(",", "").strip())
    except InvalidOperation:
        return None


def _changed(entry: Mapping[str, Any]) -> bool:
    orig, new = entry.get("original"), entry.get("new")
    if orig in (None, "") or new in (None, ""):
        return False
    a, b = _dec(orig), _dec(new)
    if a is not None and b is not None:
        return a != b
    return str(orig).strip() != str(new).strip()


def classify_subtle(affected_rows: Iterable[Mapping[str, Any]],
                    max_relative_change: float) -> FrozenSet[str]:
    """Categories met by one document: single changed amount, single changed date, and a
    single changed amount whose relative change is at most ``max_relative_change``."""
    amounts = []
    dates = []
    for e in affected_rows or ():
        if not _changed(e):
            continue
        field = e.get("field")
        if field == "amount":
            amounts.append(e)
        elif field == "date":
            dates.append(e)
    cats = set()
    if len(amounts) == 1:
        cats.add("single_amount")
        a, b = _dec(amounts[0]["original"]), _dec(amounts[0]["new"])
        if a is not None and b is not None and a != 0:
            if abs(b - a) / abs(a) <= Decimal(str(max_relative_change)):
                cats.add("amount_le_5pct")
    if len(dates) == 1:
        cats.add("single_date")
    return frozenset(cats)


def subtle_gate(docs: Sequence[Mapping[str, Any]], *, max_relative_change: float) -> GateResult:
    present: Dict[str, set] = defaultdict(set)
    for d in docs:
        if not _truthy(d.get("is_tampered")):
            continue
        present[d.get("split")] |= classify_subtle(d.get("affected_rows") or (), max_relative_change)
    offenders = [f"{split}: no document with {cat}"
                 for split in sorted(present, key=str)
                 for cat in SUBTLE_CATEGORIES if cat not in present[split]]
    return _gate("subtle", "6.6", offenders, True)


# ------------------------------------------------------------------------------ capability
def capability_report(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Styling choices the generator does not vary, plus class-exclusive Producer families.
    Unknown (None) values are ignored; a choice with no known value at all is not assessable
    and is not reported."""
    report: List[str] = []
    for name, col in CAPABILITY_CHOICES:
        values = {False: set(), True: set()}
        for r in rows:
            v = r.get(col)
            if v is not None:
                values[_truthy(r.get("is_tampered"))].add(_hashable(v))
        if not values[False] and not values[True]:
            continue
        for tampered, label in ((False, "genuine"), (True, "tampered")):
            n = len(values[tampered])
            if n < 2:
                report.append(f"{name}: {label} documents show {n} distinct value(s), need at least 2")
    fam = {False: set(), True: set()}
    for r in rows:
        v = r.get(FAMILY_COLUMN)
        if v is not None:
            fam[_truthy(r.get("is_tampered"))].add(v)
    for f in sorted(fam[False] - fam[True]):
        report.append(f"family {f}: appears in genuine documents only")
    for f in sorted(fam[True] - fam[False]):
        report.append(f"family {f}: appears in tampered documents only")
    return report


def capability_stop_message(report: Sequence[str]) -> str:
    lines = ["Training stopped: the corpus generator cannot produce these styling choices "
             "in both classes:"]
    lines += [f"  - {item}" for item in report]
    return "\n".join(lines)


def capability_gate(rows: Sequence[Mapping[str, Any]]) -> GateResult:
    return _gate("capability", "5.4, 5.6", capability_report(rows), True)
