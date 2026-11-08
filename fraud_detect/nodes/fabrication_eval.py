"""Evaluation script for the Node 3 fabrication rules (Requirements 12, 13, 16, 17).

Run as ``python -m fraud_detect.nodes.fabrication_eval``.

Read-only report: counts rule firings on the genuine documents of the external
synthetic dataset, on freshly generated clean statements, and reports the scores
of statement_E / statement_F. The default pipeline runs ingestion, the detection
nodes and risk synthesis in memory; it builds no case store and writes no file
(generator statements live in a temporary folder that is removed afterwards).
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, TextIO

DEFAULT_DATASET_DIR = Path.home() / "Downloads" / "statement_forgery_dataset"
DEFAULT_STATEMENTS_DIR = Path.home() / "Downloads"
DEFAULT_GENERATOR_COUNT = 30
TARGET_SCORE = 25

RULE_CODES = (
    "FL_DUP_INVOICE_REF",
    "FL_INVOICE_SEQUENCE_BREAK",
    "FL_TURNOVER_COST_MISMATCH",
    "DQ_RULE_SKIPPED",
)
_META_KEYS = ("Producer", "Creator", "CreationDate", "ModDate")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def count_code(results: list[dict], code: str) -> int:
    return sum(1 for r in results if code in (r.get("finding_codes") or []))


def count_rule_firings(results: list[dict], rule_codes=RULE_CODES) -> dict[str, int]:
    return {c: count_code(results, c) for c in rule_codes}


def invoice_ref_share(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("has_invoice_ref")) / len(results)


def format_firing_table(label: str, counts: dict[str, int], total: int) -> list[str]:
    return [f"{label} firing {code}: {n} of {total}" for code, n in counts.items()]


def resaved_metadata_counts(results: list[dict]) -> dict:
    producers: dict[str, int] = {}
    mod = created = 0
    for r in results:
        meta = r.get("metadata") or {}
        p = meta.get("Producer")
        if p:
            producers[p] = producers.get(p, 0) + 1
        if meta.get("ModDate"):
            mod += 1
        if meta.get("CreationDate"):
            created += 1
    return {"documents": len(results), "Producer": producers,
            "ModDate": mod, "CreationDate": created}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Default (real) pipeline
# --------------------------------------------------------------------------- #
def _default_screen(path: Path) -> dict:
    """Ingest + detection nodes + synthesis, in memory only."""
    from fraud_detect.models import AgentReport
    from fraud_detect.nodes import (
        fabrication, financial_logic, risk_synthesis, security, visual_forensics,
    )
    from fraud_detect.nodes.ingestion import ingest

    ing = ingest(str(path))
    reports = [
        AgentReport(agent=security.AGENT, findings=security.scan(ing)),
        visual_forensics.analyze(ing),
        financial_logic.analyze(ing),
    ]
    report = risk_synthesis.synthesize(ing, reports)
    codes = [f.code for rep in reports for f in rep.findings]
    refs = fabrication.extract_credit_refs(ing.transactions, fabrication.DEFAULT_CONFIG)
    m = ing.metadata
    return {
        "finding_codes": codes,
        "risk_score": report.risk_score,
        "risk_band": getattr(report.risk_band, "value", report.risk_band),
        "routing": getattr(report.routing, "value", report.routing),
        "has_invoice_ref": bool(refs),
        "metadata": {"Producer": m.producer, "Creator": m.creator,
                     "CreationDate": m.creation_date, "ModDate": m.mod_date},
    }


def _default_generator_screen(count: int) -> list[dict]:
    from fraud_detect.synthetic import generate_clean

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, count + 1):
            stmt = generate_clean(tmp, seed=seed)
            results.append(_default_screen(Path(stmt.path)))
    return results


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #
def _read_labels(dataset_dir: Path) -> tuple[list[dict], bool]:
    """Genuine rows in label order, and whether a genuine_resaved column exists."""
    with (dataset_dir / "labels.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        has_col = "genuine_resaved" in (reader.fieldnames or [])
        rows = [r for r in reader if r.get("is_tampered") == "0"]
    return rows, has_col


def _run_dataset(dataset_dir: Path, sample: Optional[int], screen: Callable,
                 out: TextIO, timings: list[float]) -> None:
    if not (dataset_dir.is_dir() and (dataset_dir / "labels.csv").is_file()):
        print(f"dataset section skipped: no labels.csv in {dataset_dir}", file=out)
        return
    rows, has_col = _read_labels(dataset_dir)
    if sample is not None:
        rows = rows[:sample]
    results, resaved = [], []
    for r in rows:
        t0 = time.perf_counter()
        res = dict(screen(dataset_dir / r["file"]))
        timings.append(time.perf_counter() - t0)
        results.append(res)
        if has_col and (r.get("genuine_resaved") or "").strip() == "1":
            resaved.append(res)
    for line in format_firing_table("dataset", count_rule_firings(results), len(results)):
        print(line, file=out)
    print(f"dataset invoice reference share: {_pct(invoice_ref_share(results))}", file=out)
    print(f"dataset VF_CURRENCY_MIX: {count_code(results, 'VF_CURRENCY_MIX')}", file=out)
    meta = resaved_metadata_counts(resaved)
    print(f"resaved documents: {meta['documents']}", file=out)
    for value in sorted(meta["Producer"]):
        print(f"resaved Producer {value}: {meta['Producer'][value]}", file=out)
    print(f"resaved ModDate present: {meta['ModDate']}", file=out)
    print(f"resaved CreationDate present: {meta['CreationDate']}", file=out)


def _run_generator(count: int, generator_screen: Callable, out: TextIO) -> None:
    if count == 0:
        print("generator section skipped: --generator-count is 0", file=out)
        return
    results = list(generator_screen(count))
    for line in format_firing_table("generator", count_rule_firings(results), len(results)):
        print(line, file=out)


def _run_statements(statements_dir: Path, screen: Callable, out: TextIO,
                    timings: list[float]) -> None:
    paths = {t: statements_dir / f"statement_{t}.pdf" for t in ("E", "F")}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        print(f"statements section skipped: missing {', '.join(missing)} in {statements_dir}",
              file=out)
        return
    res = {}
    for tag, path in paths.items():
        t0 = time.perf_counter()
        res[tag] = dict(screen(path))
        timings.append(time.perf_counter() - t0)
    for tag in ("E", "F"):
        meta = res[tag].get("metadata") or {}
        fields = " ".join(f"{k}={meta.get(k) or ''}" for k in _META_KEYS)
        print(f"metadata {tag} {fields}", file=out)
    for tag in ("E", "F"):
        r = res[tag]
        codes = ",".join(r.get("finding_codes") or [])
        print(f"{tag} score={int(r.get('risk_score') or 0)} band={r.get('risk_band') or ''} "
              f"routing={r.get('routing') or ''} codes={codes}", file=out)
    e, f = int(res["E"].get("risk_score") or 0), int(res["F"].get("risk_score") or 0)
    print(f"E above {TARGET_SCORE} and above F: {'yes' if e > TARGET_SCORE and e > f else 'no'}",
          file=out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _non_negative_int(text: str) -> int:
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError("must be an integer >= 0")
    return n


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fabrication_eval", description=__doc__.splitlines()[0])
    p.add_argument("--sample", type=_positive_int, default=None,
                   help="genuine dataset documents to use, in label order (default: all)")
    p.add_argument("--dataset", default=None, help="synthetic forgery dataset folder")
    p.add_argument("--statements", default=None, help="folder holding statement_E/F.pdf")
    p.add_argument("--generator-count", type=_non_negative_int, default=DEFAULT_GENERATOR_COUNT,
                   help="generated clean statements to screen (0 skips; default 30)")
    return p


def main(argv=None, *, dataset_dir=None, statements_dir=None, out=None,
         screen=None, generator_screen=None) -> int:
    out = out if out is not None else sys.stdout
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) and exc.code else 2
    ds = dataset_dir if dataset_dir is not None else args.dataset
    st = statements_dir if statements_dir is not None else args.statements
    ds = Path(ds) if ds is not None else DEFAULT_DATASET_DIR
    st = Path(st) if st is not None else DEFAULT_STATEMENTS_DIR
    screen_fn = screen if screen is not None else _default_screen
    gen_fn = generator_screen if generator_screen is not None else _default_generator_screen

    timings: list[float] = []
    _run_dataset(ds, args.sample, screen_fn, out, timings)
    _run_generator(args.generator_count, gen_fn, out)
    _run_statements(st, screen_fn, out, timings)
    print(f"elapsed (longest file): {max(timings):.2f}s" if timings else "elapsed (longest file): 0.00s",
          file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
