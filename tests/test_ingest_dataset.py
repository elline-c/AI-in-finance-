"""Task 11.1: dataset acceptance test for Node 1 (Requirements 10.1-10.4, 14.2).

Reads the external synthetic dataset at ~/Downloads/statement_forgery_dataset
(never copied into the repo; the whole module skips when it is missing).
For each of the five layouts the Sample is the first 20 genuine documents of
that layout in labels.csv order. The expected header values are read straight
from the PDF text with a plain pdfplumber regex that is independent of the
parser under test (labels.csv carries no header values).

Path constants are re-defined here (tests/ is not a package, so importing
tests/ingest_parse/conftest.py is awkward); the values match that conftest.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pdfplumber
import pytest

from fraud_detect.nodes.ingestion import ingest

DATASET_DIR = Path.home() / "Downloads" / "statement_forgery_dataset"
DATASET_LABELS = DATASET_DIR / "labels.csv"

requires_dataset = pytest.mark.skipif(
    not (DATASET_DIR.is_dir() and DATASET_LABELS.is_file()),
    reason=f"forgery dataset not found at {DATASET_DIR}",
)
pytestmark = requires_dataset

SAMPLE_SIZE = 20
TOL = 0.01

# bank_layout id -> (expected profile name, strptime format of the period dates)
LAYOUTS = {
    "1": ("two_col_slash", "%d/%m/%Y"),
    "2": ("two_col_dual_date", "%d %b %Y"),
    "3": ("signed_amount_iso", "%Y-%m-%d"),
    "4": ("two_col_ddmmyy_cr", "%d-%m-%y"),
    "5": ("two_col_dmon", "%d %b %Y"),
}

_MONEY = r"(?:S\$|SGD\s)?-?[\d,]+\.\d{2}"
_ACCT_RE = re.compile(r"^(?P<holder>.+?)\s+Account No\.\s+(?P<acct>.+?)\s*$")
_PERIOD_RE = re.compile(r"Statement Period\s+(\S.*?)\s+to\s+(\S.*?)\s*$")


def _num(s: str) -> float:
    return float(re.sub(r"[^\d.\-]", "", s))


def expected_header(pdf_path: Path, fmt: str) -> dict:
    """Independent plain-text read of the header block printed on page 1."""
    with pdfplumber.open(pdf_path) as pdf:
        lines = (pdf.pages[0].extract_text() or "").split("\n")
    out: dict = {}
    for i, ln in enumerate(lines):
        m = _ACCT_RE.match(ln)
        if m and "holder" not in out:
            out["holder"] = m.group("holder")
            out["account"] = m.group("acct")
        p = _PERIOD_RE.search(ln)
        if p and "start" not in out:
            out["start"] = datetime.strptime(p.group(1), fmt).date()
            out["end"] = datetime.strptime(p.group(2), fmt).date()
        if ln.startswith("Opening Balance") and "Closing Balance" in ln and i + 1 < len(lines):
            vals = re.findall(_MONEY, lines[i + 1])
            assert len(vals) == 4, f"summary row unreadable: {lines[i + 1]!r}"
            out["opening"] = _num(vals[0])
            out["closing"] = _num(vals[3])
    missing = {"holder", "account", "start", "end", "opening", "closing"} - out.keys()
    assert not missing, f"test helper could not read {missing} from {pdf_path.name}"
    return out


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


@lru_cache(maxsize=None)
def sample(layout: str) -> tuple[dict, ...]:
    with DATASET_LABELS.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["bank_layout"] == layout and r["is_tampered"] == "0"]
    return tuple(rows[:SAMPLE_SIZE])


@lru_cache(maxsize=None)
def run(layout: str) -> tuple[tuple[dict, object, dict], ...]:
    fmt = LAYOUTS[layout][1]
    out = []
    for r in sample(layout):
        path = DATASET_DIR / r["file"]
        out.append((r, ingest(str(path)), expected_header(path, fmt)))
    return tuple(out)


def test_sample_is_twenty_per_layout():
    for layout in LAYOUTS:
        assert len(sample(layout)) == SAMPLE_SIZE, layout


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_rows_read_at_least_99_percent(layout):
    expected = sum(int(r["n_transactions"]) for r, _, _ in run(layout))
    got = sum(len(res.transactions) for _, res, _ in run(layout))
    ratio = got / expected
    print(f"\n[layout {layout}] rows read {got}/{expected} = {ratio:.4f}")
    assert ratio >= 0.99, f"layout {layout}: read {got} of {expected} rows ({ratio:.2%})"


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_replayed_balance_equals_printed_closing(layout):
    bad = []
    for r, res, exp in run(layout):
        replay = exp["opening"] + sum(t.amount or 0.0 for t in res.transactions)
        if abs(replay - exp["closing"]) > TOL:
            bad.append((r["doc_id"], round(replay, 2), exp["closing"], len(res.transactions), r["n_transactions"]))
    print(f"\n[layout {layout}] balance replay mismatches: {len(bad)}/{SAMPLE_SIZE}")
    assert not bad, f"layout {layout}: (doc, replayed, printed, rows read, rows expected) {bad}"


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_header_fields_match_printed_values(layout):
    bad = []
    for r, res, exp in run(layout):
        got = {
            "holder": _norm(res.account_holder),
            "account": _norm(res.account_number),
            "start": res.statement_period_start,
            "end": res.statement_period_end,
        }
        want = {k: (_norm(exp[k]) if k in ("holder", "account") else exp[k]) for k in got}
        for k in got:
            if got[k] != want[k]:
                bad.append((r["doc_id"], k, got[k], want[k]))
        for k, attr in (("opening", "opening_balance"), ("closing", "closing_balance")):
            v = getattr(res, attr)
            if v is None or abs(v - exp[k]) > TOL:
                bad.append((r["doc_id"], k, v, exp[k]))
    print(f"\n[layout {layout}] header field mismatches: {len(bad)}")
    assert not bad, f"layout {layout}: (doc, field, parsed, printed) {bad}"


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_expected_profile_chosen(layout):
    want = LAYOUTS[layout][0]
    got = Counter(res.layout_profile for _, res, _ in run(layout))
    print(f"\n[layout {layout}] profiles chosen: {dict(got)}")
    assert set(got) == {want}, f"layout {layout}: expected {want}, got {dict(got)}"


def test_print_finding_codes_across_sample():
    """Summary only (no assertion): Finding codes fired per layout."""
    print()
    for layout in sorted(LAYOUTS):
        codes = Counter(f.code for _, res, _ in run(layout) for f in res.data_quality_findings)
        print(f"[layout {layout}] ({LAYOUTS[layout][0]}) Finding codes: {dict(codes) or 'none'}")
