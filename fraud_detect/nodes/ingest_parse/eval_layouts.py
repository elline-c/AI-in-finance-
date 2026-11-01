"""Evaluation script for the multi-layout Node 1 parser (Requirement 13).

Run as ``python -m fraud_detect.nodes.ingest_parse.eval_layouts``.

Read-only: it ingests PDFs from the external synthetic dataset and the
statement_E / statement_F files, prints a report, and writes nothing. The default
pipeline runs Node 1 plus the in-memory detection nodes and risk synthesis; it
passes no CaseStore, so ``data/`` and the audit chain are never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TextIO

DEFAULT_DATASET_DIR = Path.home() / "Downloads" / "statement_forgery_dataset"
DEFAULT_STATEMENTS_DIR = Path.home() / "Downloads"
ANSWER_KEY_NAME = "ANSWER_KEY_6M_v2.json"
DEFAULT_SAMPLE = 20
TOL = 0.01

_MONEY = r"(?:S\$|SGD\s)?-?[\d,]+\.\d{2}"
_ACCT_RE = re.compile(r"^(?P<holder>.+?)\s+Account No\.\s+(?P<acct>.+?)\s*$")
_PERIOD_RE = re.compile(r"Statement Period\s+(\S.*?)\s+to\s+(\S.*?)\s*$")
_DATE_FORMATS = ("%d/%m/%Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%y")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def summarise_layout(results: list[dict]) -> dict:
    """Aggregate per-document results of one layout."""
    n = len(results)
    expected = sum(int(r.get("rows_expected") or 0) for r in results)
    read = sum(int(r.get("rows_read") or 0) for r in results)
    codes = sorted({c for r in results for c in (r.get("finding_codes") or [])})

    def share(key: str) -> float:
        return sum(1 for r in results if r.get(key)) / n if n else 0.0

    return {
        "profile": (results[0].get("profile") if results else None) or "",
        "documents": n,
        "rows_share": (read / expected) if expected else 1.0,
        "balance_share": share("balance_ok"),
        "header_share": share("header_ok"),
        "finding_codes": codes,
    }


def _money(raw) -> float:
    return float(re.sub(r"[^\d.\-]", "", str(raw)))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _close(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= TOL + 1e-9


# --------------------------------------------------------------------------- #
# Default (real) pipeline
# --------------------------------------------------------------------------- #
def _parse_date(s: str):
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _plain_header(path: Path) -> Optional[dict]:
    """Independent plain-pdfplumber read of the header block on page 1."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        lines = (pdf.pages[0].extract_text() or "").split("\n")
    out: dict = {}
    for i, ln in enumerate(lines):
        m = _ACCT_RE.match(ln)
        if m and "holder" not in out:
            out["holder"], out["account"] = m.group("holder"), m.group("acct")
        p = _PERIOD_RE.search(ln)
        if p and "start" not in out:
            s, e = _parse_date(p.group(1)), _parse_date(p.group(2))
            if s and e:
                out["start"], out["end"] = s, e
        if ln.startswith("Opening Balance") and "Closing Balance" in ln and i + 1 < len(lines):
            vals = re.findall(_MONEY, lines[i + 1])
            if len(vals) == 4:
                out["opening"], out["closing"] = _money(vals[0]), _money(vals[3])
    need = {"holder", "account", "start", "end", "opening", "closing"}
    return out if need <= out.keys() else None


def _norm(s) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _default_ingest(path: Path, rows_expected: int = 0) -> dict:
    """Real Node 1 + detection nodes + synthesis; in memory only (no store)."""
    from fraud_detect.models import AgentReport
    from fraud_detect.nodes import financial_logic, risk_synthesis, security, visual_forensics
    from fraud_detect.nodes.ingestion import ingest

    ing = ingest(str(path))
    reports = [
        AgentReport(agent=security.AGENT, findings=security.scan(ing)),
        visual_forensics.analyze(ing),
        financial_logic.analyze(ing),
    ]
    report = risk_synthesis.synthesize(ing, reports)

    res: dict = {
        "profile": ing.layout_profile or "",
        "routing": report.routing.value,
        "finding_codes": sorted({f.code for f in report.findings}),
        "rows_read": len(ing.transactions),
    }
    replay_open = ing.opening_balance
    replay_close = ing.closing_balance
    ref = None
    try:
        ref = _plain_header(path)
    except Exception:
        ref = None
    if ref is not None:
        replay_open, replay_close = ref["opening"], ref["closing"]
        res["header_ok"] = bool(
            _norm(ing.account_holder) == _norm(ref["holder"])
            and _norm(ing.account_number) == _norm(ref["account"])
            and ing.statement_period_start == ref["start"]
            and ing.statement_period_end == ref["end"]
            and _close(ing.opening_balance, ref["opening"])
            and _close(ing.closing_balance, ref["closing"])
        )
    if replay_open is not None and replay_close is not None:
        replay = replay_open + sum(t.amount or 0.0 for t in ing.transactions)
        res["balance_ok"] = abs(replay - replay_close) <= TOL
    if path.stem in ("statement_E", "statement_F"):
        monthly = {}
        for s in ing.sections:
            if s.period_start is None:
                continue
            monthly[s.period_start.strftime("%Y-%m")] = {
                "opening": s.opening, "total_deposits": s.total_deposits,
                "total_withdrawals": s.total_withdrawals, "closing": s.closing,
            }
        res["monthly"] = monthly
        res["parsed_rows"] = [
            {"month": t.date.strftime("%Y-%m"), "date": t.date.isoformat(), "amount": t.amount}
            for t in ing.transactions if t.date is not None and t.amount is not None
        ]
    return res


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _read_genuine(dataset_dir: Path) -> dict[int, list[dict]]:
    by_layout: dict[int, list[dict]] = {}
    with (dataset_dir / "labels.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("is_tampered") != "0":
                continue
            try:
                layout = int(r["bank_layout"])
            except (KeyError, ValueError):
                continue
            by_layout.setdefault(layout, []).append(r)
    return dict(sorted(by_layout.items()))


def _run_dataset(dataset_dir: Path, sample: Optional[int], ingest: Callable,
                 use_default: bool, out: TextIO, timings: list[float]) -> None:
    if not (dataset_dir.is_dir() and (dataset_dir / "labels.csv").is_file()):
        print(f"dataset section skipped: no labels.csv in {dataset_dir}", file=out)
        return
    by_layout = _read_genuine(dataset_dir)
    routing: Counter = Counter()
    findings: Counter = Counter()
    for layout, rows in by_layout.items():
        chosen = rows if sample is None else rows[:sample]
        results = []
        for r in chosen:
            path = dataset_dir / r["file"]
            t0 = time.perf_counter()
            res = dict(ingest(path))
            timings.append(time.perf_counter() - t0)
            try:
                res.setdefault("rows_expected", int(r.get("n_transactions") or 0))
            except ValueError:
                res.setdefault("rows_expected", 0)
            results.append(res)
            if res.get("routing"):
                routing[res["routing"]] += 1
            for c in set(res.get("finding_codes") or []):
                findings[c] += 1
        s = summarise_layout(results)
        print(
            f"layout {layout}: profile={s['profile']} documents={s['documents']} "
            f"rows={_pct(s['rows_share'])} balance={_pct(s['balance_share'])} "
            f"header={_pct(s['header_share'])} findings={','.join(s['finding_codes'])}",
            file=out,
        )
    if sample is None:
        print("false-alarm table (all genuine documents)", file=out)
        for k in sorted(routing):
            print(f"routing {k}: {routing[k]}", file=out)
        for k in sorted(findings):
            print(f"finding {k}: {findings[k]}", file=out)


def _fmt(v) -> str:
    return "None" if v is None else f"{float(v):.2f}"


def _compare_months(tag: str, key_months: list[dict], monthly: dict, out: TextIO) -> None:
    for km in sorted(key_months, key=lambda m: m["month"]):
        want = {k: _money(km[k]) for k in ("opening", "total_deposits", "total_withdrawals", "closing")}
        got = (monthly or {}).get(km["month"]) or {}
        ok = all(_close(got.get(k), want[k]) for k in want)
        cols = " ".join(f"{k}={_fmt(got.get(k))}/{_fmt(want[k])}" for k in want)
        print(f"{tag} {km['month']} {cols} {'MATCH' if ok else 'DIFF'}", file=out)


def _run_statements(statements_dir: Path, ingest: Callable, out: TextIO,
                    timings: list[float]) -> None:
    key_path = statements_dir / ANSWER_KEY_NAME
    if not key_path.is_file():
        print(f"statement section skipped: no {ANSWER_KEY_NAME} in {statements_dir}", file=out)
        return
    try:
        key = json.loads(key_path.read_text())
    except (OSError, ValueError):
        print(f"statement section skipped: unreadable {ANSWER_KEY_NAME}", file=out)
        return
    for tag, stem, month_key in (("E", "statement_E", "monthly_as_shown"), ("F", "statement_F", "monthly")):
        pdf = statements_dir / f"{stem}.pdf"
        if not pdf.is_file() or stem not in key:
            print(f"{stem} skipped: {pdf.name} or its answer-key entry is missing", file=out)
            continue
        t0 = time.perf_counter()
        res = ingest(pdf)
        timings.append(time.perf_counter() - t0)
        _compare_months(tag, key[stem].get(month_key) or [], res.get("monthly") or {}, out)
        if tag == "E":
            inserted = key[stem].get("inserted_rows") or []
            parsed = res.get("parsed_rows") or []
            found = 0
            for ins in inserted:
                amt = _money(ins["fabricated_amount"])
                if any(p.get("month") == ins["month"] and p.get("date") == ins["date"]
                       and _close(p.get("amount"), amt) for p in parsed):
                    found += 1
            print(f"E inserted rows found: {found} of {len(inserted)}", file=out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_layouts", description=__doc__.splitlines()[0])
    p.add_argument("--sample", type=_positive_int, default=DEFAULT_SAMPLE,
                   help="genuine documents per layout (default 20)")
    p.add_argument("--all-genuine", action="store_true",
                   help="use every genuine document and print the false-alarm table")
    return p


def main(argv=None, *, dataset_dir=None, statements_dir=None, out=None, ingest=None) -> int:
    out = out if out is not None else sys.stdout
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) and exc.code else 2
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    statements_dir = Path(statements_dir) if statements_dir is not None else DEFAULT_STATEMENTS_DIR
    use_default = ingest is None
    fn = ingest if ingest is not None else _default_ingest

    timings: list[float] = []
    _run_dataset(dataset_dir, None if args.all_genuine else args.sample, fn, use_default, out, timings)
    _run_statements(statements_dir, fn, out, timings)
    print(f"elapsed (longest file): {max(timings):.2f}s" if timings else "elapsed (longest file): 0.00s",
          file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
