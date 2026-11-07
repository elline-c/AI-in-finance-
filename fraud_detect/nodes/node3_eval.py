"""Evaluation script for the Node 3 financial-logic rules.

Run as ``python -m fraud_detect.nodes.node3_eval --companies 200``.

For each synthetic company it builds a genuine statement and the same statement with
one planted tamper, runs Node 3 on every one of them, and reports how often each rule
fires, with a 95% Wilson confidence interval:

* on GENUINE statements (every firing is a false alarm), and
* on each PLANTED tamper (does the rule that should see it, see it?).

Presets let two settings be compared on identical documents. ``learned`` is the shipped
configuration. ``fixed`` switches off what could be switched off: the learned gap limit
(back to 25 days), the statement's own baseline (so only the declared per-UEN figure is
left, the old 3x rule), pass-through, and the relative structuring checks. The
counterparty rule needs a baseline to say what "large" means, so ``fixed`` weakens it
too. Its other change (whole words, "payer not seen before") is not a setting and can
only be compared against an earlier commit.

Read-only: nothing is written unless ``--out`` or ``--json`` is given. No PDF is drawn and
Node 1 is not involved: ledgers are turned straight into ingestion results, so this
measures the rules, not the parser.

HONEST LIMITS. The documents and the tampers come from our own generator, which was
written by the same team as the rules (possible circularity). Every company is fictional.
Nothing here says how the rules behave on real bank statements.
"""
from __future__ import annotations

import argparse
import calendar
import dataclasses
import json
import math
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional, TextIO

from .. import reference_data
from ..corpus.company import make_company
from ..corpus.ledger import simulate_ledger
from ..corpus.schema import Ledger, SimulatorKnobs
from ..corpus.seeding import rng
from ..corpus.tamper.catalog import CATALOG
from ..models import DocumentMetadata, IngestionResult, StatementSegment, Transaction
from . import financial_logic
from .logic_config import DEFAULT_LOGIC_CONFIG, FinancialLogicConfig

DEFAULT_COMPANIES = 200
DEFAULT_MONTHS = 9
DEFAULT_SEED = 2000
START = date(2026, 1, 1)

# The rules a banker would act on; any firing on a genuine statement is a false alarm.
WATCHED = (
    "FL_SUDDEN_DEPOSIT", "FL_RAPID_PASS_THROUGH", "FL_UNVERIFIED_COUNTERPARTY",
    "FL_STRUCTURING", "FL_PERIOD_GAP", "FL_MONTH_MISSING",
)

# Scenario -> the rule codes that should see it. ``short_silence`` is made by this script
# (a 12-day stretch of rows removed); the others come from the corpus tamper catalogue.
EXPECTED: dict[str, tuple[str, ...]] = {
    "structuring": ("FL_STRUCTURING",),
    "sudden_large_deposit": ("FL_SUDDEN_DEPOSIT",),
    "shell_counterparty": ("FL_UNVERIFIED_COUNTERPARTY",),
    "rapid_pass_through": ("FL_RAPID_PASS_THROUGH",),
    "period_gap": ("FL_PERIOD_GAP", "FL_MONTH_MISSING"),
    "short_silence": ("FL_PERIOD_GAP",),
}
DEFAULT_TAMPERS = tuple(EXPECTED)
SILENCE_DAYS = 12                 # length of the stretch ``short_silence`` removes


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
_NEVER = 10 ** 9
PRESETS: dict[str, FinancialLogicConfig] = {
    "learned": DEFAULT_LOGIC_CONFIG,
    "fixed": dataclasses.replace(
        DEFAULT_LOGIC_CONFIG,
        gap_min_observations=_NEVER,         # always the fixed 25-day limit
        credit_min_months=_NEVER,            # no baseline from the statement: declared figure only
        pass_large_mult=1e12,                # pass-through off
        structuring_month_share=0.0,         # relative structuring checks off
        structuring_routine_share=2.0,
    ),
}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials; (0, 1) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    rate = (k / n) if n else 0.0
    return f"{k}/{n} ({rate:.1%}) [{lo:.1%}-{hi:.1%}]"


def ledger_to_ingestion(ledger: Ledger) -> IngestionResult:
    """What Node 1 would hand to Node 3 for this ledger (balances replayed from amounts)."""
    bal, txns = ledger.opening_balance, []
    for i, r in enumerate(ledger.rows, start=1):
        bal = round(bal + r.amount, 2)
        txns.append(Transaction(index=i, date=r.date, description=r.description,
                                amount=r.amount, balance=bal, page=1))
    first = ledger.months[0].replace(day=1)
    last = ledger.months[-1]
    end = last.replace(day=calendar.monthrange(last.year, last.month)[1])
    # One statement section per month that still has rows, like the printed monthly headers.
    present = sorted({(r.date.year, r.date.month) for r in ledger.rows})
    segments = [StatementSegment(
        period_start=date(y, m, 1), period_end=date(y, m, calendar.monthrange(y, m)[1]))
        for y, m in present]
    return IngestionResult(
        source_path="eval.pdf", page_count=1, raw_text="", metadata=DocumentMetadata(),
        uen=ledger.company.uen, account_holder=ledger.company.name,
        statement_period_start=first, statement_period_end=end,
        opening_balance=ledger.opening_balance, closing_balance=bal,
        transactions=txns, segments=segments)


def _changed(cfg: FinancialLogicConfig) -> dict:
    """Settings that differ from the shipped defaults (shown in the report)."""
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
            if getattr(cfg, f.name) != getattr(DEFAULT_LOGIC_CONFIG, f.name)}


def apply_overrides(presets: dict[str, FinancialLogicConfig],
                    overrides: Iterable[str]) -> dict[str, FinancialLogicConfig]:
    """Apply ``name=value`` settings (numbers only) to every preset, e.g. gap_min_days=14."""
    fields = {f.name: getattr(DEFAULT_LOGIC_CONFIG, f.name) for f in dataclasses.fields(DEFAULT_LOGIC_CONFIG)}
    changes: dict = {}
    for item in overrides:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep or name not in fields:
            raise ValueError(f"unknown setting in {item!r}; use name=value with a FinancialLogicConfig field")
        kind = type(fields[name])
        if kind not in (int, float):
            raise ValueError(f"{name} is not a plain number and cannot be set here")
        try:
            changes[name] = kind(raw)
        except ValueError:
            raise ValueError(f"{raw!r} is not a valid {kind.__name__} for {name}") from None
    return {p: dataclasses.replace(c, **changes) for p, c in presets.items()}


def make_ledger(seed: int, months: int) -> Ledger:
    company = make_company(seed, "C0001")
    return simulate_ledger(company, START, months, seed, SimulatorKnobs(), None)


def remove_silence(ledger: Ledger, r: random.Random, days: int = SILENCE_DAYS) -> Ledger:
    """Drop every row in a random ``days``-long stretch inside the middle of the statement."""
    first, last = ledger.months[0], ledger.months[-1]
    span = (last - first).days
    lo, hi = 30, max(31, span - 30 - days)
    start = first + timedelta(days=r.randint(lo, hi))
    end = start + timedelta(days=days - 1)
    kept = tuple(row for row in ledger.rows if not (start <= row.date <= end))
    return dataclasses.replace(ledger, rows=kept)


def tampered_ledger(name: str, ledger: Ledger, seed: int,
                    silence_days: int = SILENCE_DAYS) -> Ledger:
    if name == "short_silence":
        return remove_silence(ledger, rng(seed, "eval", name), silence_days)
    return CATALOG[name].plan(ledger, None, rng(seed, "eval", name)).ledger


class _DeclaredBaseline:
    """Register each company's average monthly credit as its declared baseline, like the
    corpus builder does, and take it away again afterwards (the dictionary is shared)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._uen: Optional[str] = None

    def set(self, ledger: Ledger) -> None:
        if not self.enabled:
            return
        self._uen = ledger.company.uen
        n = len(ledger.months)
        reference_data.CASHFLOW_BASELINE[self._uen] = {
            "avg_monthly_credit": sum(r.amount for r in ledger.rows if r.amount > 0) / n,
            "avg_monthly_debit": 0.0,
        }

    def clear(self) -> None:
        if self._uen is not None:
            reference_data.CASHFLOW_BASELINE.pop(self._uen, None)
            self._uen = None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def codes_for(ledger: Ledger, config: FinancialLogicConfig, declared: bool) -> set[str]:
    guard = _DeclaredBaseline(declared)
    guard.set(ledger)
    try:
        report = financial_logic.analyze(ledger_to_ingestion(ledger), config=config)
    finally:
        guard.clear()
    return {f.code for f in report.findings}


def run(companies: int, months: int, seed: int, tampers: Iterable[str],
        presets: dict[str, FinancialLogicConfig], declared: bool = True,
        progress: Optional[Callable[[int, int], None]] = None,
        silence_days: int = SILENCE_DAYS) -> dict:
    """Counts of documents that fired each code, per preset and scenario.

    Result: ``{"settings": {...}, "results": {preset: {scenario: {"n": int, "codes": {code: k}}}}}``
    where ``scenario`` is ``"genuine"`` or a tamper name. A document counts once per code.
    """
    tampers = tuple(tampers)
    results = {p: {s: {"n": 0, "codes": Counter()} for s in ("genuine", *tampers)} for p in presets}
    for i in range(companies):
        s = seed + i
        genuine = make_ledger(s, months)
        scenarios = {"genuine": genuine}
        for name in tampers:
            scenarios[name] = tampered_ledger(name, genuine, s, silence_days)
        for preset, cfg in presets.items():
            for scenario, ledger in scenarios.items():
                cell = results[preset][scenario]
                cell["n"] += 1
                cell["codes"].update(codes_for(ledger, cfg, declared))
        if progress:
            progress(i + 1, companies)
    return {
        "settings": {"companies": companies, "months": months, "seed": seed,
                     "declared_baseline": declared, "tampers": list(tampers),
                     "presets": list(presets), "silence_days": silence_days,
                     "overrides": {p: _changed(c) for p, c in presets.items()}},
        "results": {p: {s: {"n": c["n"], "codes": dict(c["codes"])} for s, c in sc.items()}
                    for p, sc in results.items()},
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _detected(cell: dict, expected: tuple[str, ...]) -> int:
    """Documents where at least one expected code fired. A document counts once.

    ``codes`` only holds per-code counts, so this is a lower bound on the union when more
    than one code is expected; for single-code scenarios it is exact.
    """
    return max((cell["codes"].get(c, 0) for c in expected), default=0)


def format_report(data: dict) -> str:
    st, res = data["settings"], data["results"]
    lines = [
        "Node 3 evaluation on the synthetic corpus",
        f"companies={st['companies']} months={st['months']} seed={st['seed']} "
        f"declared_baseline={'yes' if st['declared_baseline'] else 'no'} "
        f"short_silence_days={st.get('silence_days', SILENCE_DAYS)}",
        "SYNTHETIC DATA. The documents and tampers come from our own generator, written by",
        "the same team as the rules, so a high detection rate shows the rules find what the",
        "generator plants, not that they find real fraud. Intervals are 95% Wilson.",
    ]
    for preset in st["presets"]:
        r = res[preset]
        g = r["genuine"]
        changed = st.get("overrides", {}).get(preset) or {}
        lines += ["", f"== preset: {preset} =="]
        if changed:
            lines.append("settings changed from the shipped defaults: "
                         + ", ".join(f"{k}={v}" for k, v in sorted(changed.items())))
        lines.append(f"Genuine statements (n={g['n']}): each firing is a false alarm")
        for code in WATCHED:
            lines.append(f"  {code:28} {fmt_rate(g['codes'].get(code, 0), g['n'])}")
        other = {c: k for c, k in g["codes"].items() if c not in WATCHED}
        if other:
            lines.append("  other Node 3 codes seen on genuine statements: "
                         + ", ".join(f"{c} {k}" for c, k in sorted(other.items())))
            if "FL_UEN_NOT_FOUND" in other:
                lines.append("  (FL_UEN_NOT_FOUND is expected: the fictional companies are not in "
                             "the registry.)")
        lines.append("Planted tampers: does the rule that should see it, see it?")
        for name in st["tampers"]:
            cell = r[name]
            exp = EXPECTED.get(name, ())
            lines.append(f"  {name:22} -> {'|'.join(exp) or '(none expected)':44}"
                         f" {fmt_rate(_detected(cell, exp), cell['n'])}")
    if len(st["presets"]) > 1:
        names = st["presets"]
        lines += ["", "== side by side (rate, with 95% interval) =="]
        head = f"  {'':40}" + "".join(f"{p:>30}" for p in names)
        lines.append(head)
        for code in WATCHED:
            lines.append(f"  {'genuine ' + code:40}" + "".join(
                f"{fmt_rate(res[p]['genuine']['codes'].get(code, 0), res[p]['genuine']['n']):>30}"
                for p in names))
        for name in st["tampers"]:
            exp = EXPECTED.get(name, ())
            lines.append(f"  {'planted ' + name:40}" + "".join(
                f"{fmt_rate(_detected(res[p][name], exp), res[p][name]['n']):>30}" for p in names))
        lines.append("  'fixed' leaves only the declared per-UEN figure as a baseline, which also")
        lines.append("  weakens the counterparty rule. Its whole-word change is not a setting:")
        lines.append("  compare against an earlier commit for that.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None, out: TextIO = sys.stdout) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the Node 3 rules on synthetic statements.")
    ap.add_argument("--companies", type=int, default=DEFAULT_COMPANIES)
    ap.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--presets", default="learned,fixed", help="comma list of: " + ", ".join(PRESETS))
    ap.add_argument("--tampers", default=",".join(DEFAULT_TAMPERS))
    ap.add_argument("--silence-days", type=int, default=SILENCE_DAYS,
                    help="length of the stretch the short_silence scenario removes")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                    help="override a numeric FinancialLogicConfig setting for every preset "
                         "(repeatable), e.g. --set gap_min_days=14")
    ap.add_argument("--no-declared", action="store_true",
                    help="do not register a declared per-UEN baseline for each company")
    ap.add_argument("--out", type=Path, help="also write the text report here")
    ap.add_argument("--json", type=Path, help="also write the raw counts here")
    args = ap.parse_args(argv)

    names = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in names if p not in PRESETS]
    tampers = [t.strip() for t in args.tampers.split(",") if t.strip()]
    bad = [t for t in tampers if t not in EXPECTED]
    if unknown or bad or args.companies < 1 or args.months < 3 or args.silence_days < 1:
        print(f"unknown preset(s) {unknown}, tamper(s) {bad}, or a count below its minimum",
              file=sys.stderr)
        return 2
    try:
        chosen = apply_overrides({p: PRESETS[p] for p in names}, args.overrides)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    def tick(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"  {done}/{total} companies", file=sys.stderr)

    data = run(args.companies, args.months, args.seed, tampers, chosen,
               declared=not args.no_declared, progress=tick, silence_days=args.silence_days)
    text = format_report(data)
    out.write(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
