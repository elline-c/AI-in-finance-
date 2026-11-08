"""Task 9.1: fabrication rules on statement_E / statement_F (Requirements 14, 17).

Reads external files from ~/Downloads (never copied into the repo; every test
skips when any of them is missing). Path constants are defined locally because
tests/ is not a package. Run with -s to see the reported figures.
"""
from __future__ import annotations

import json
import re
import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from fraud_detect.nodes.financial_logic import analyze
from fraud_detect.nodes.fabrication import DEFAULT_CONFIG
from fraud_detect.nodes.ingestion import ingest

DOWNLOADS = Path.home() / "Downloads"
STATEMENT_E = DOWNLOADS / "statement_E.pdf"
STATEMENT_F = DOWNLOADS / "statement_F.pdf"
ANSWER_KEY = DOWNLOADS / "ANSWER_KEY_6M_v2.json"

pytestmark = pytest.mark.skipif(
    not (STATEMENT_E.is_file() and STATEMENT_F.is_file() and ANSWER_KEY.is_file()),
    reason="statement_E.pdf, statement_F.pdf or ANSWER_KEY_6M_v2.json not found in ~/Downloads",
)

TOL = 0.01
NEW_CODES = ("FL_DUP_INVOICE_REF", "FL_INVOICE_SEQUENCE_BREAK", "FL_TURNOVER_COST_MISMATCH", "DQ_RULE_SKIPPED")
LIKELIER = re.compile(r"row (\d+) breaks the invoice number order, so it is the likelier copy")


def money(s: str) -> float:
    return float(re.sub(r"[^\d.\-]", "", s))


@lru_cache(maxsize=None)
def key() -> dict:
    return json.loads(ANSWER_KEY.read_text())


@lru_cache(maxsize=None)
def ing(name: str):
    return ingest(str({"E": STATEMENT_E, "F": STATEMENT_F}[name]))


@lru_cache(maxsize=None)
def findings(name: str):
    return tuple(analyze(ing(name)).findings)


def by_code(name: str, code: str):
    return [f for f in findings(name) if f.code == code]


def rows_of(f) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", f.field or "")}


def inserted_rows() -> list[int]:
    """Row numbers of the 12 inserted rows, matched to parsed credits by date and amount."""
    credits = [t for t in ing("E").transactions if t.amount is not None and t.amount > 0 and t.date]
    rows = []
    for item in key()["statement_E"]["inserted_rows"]:
        amt = money(item["fabricated_amount"])
        hit = [t for t in credits if t.date.isoformat() == item["date"] and abs(t.amount - amt) <= TOL]
        assert hit, f"no parsed credit matches inserted row {item['date']} {amt}"
        rows.append(hit[0].index)
    return rows


def monthly_figures(name: str):
    """Turnover / fixed costs per month, using the same keyword lists as Rule C."""
    cfg = DEFAULT_CONFIG
    ex = re.compile(r"\b(?:" + "|".join(cfg.turnover_exclude_keywords) + r")\b", re.I)
    fx = re.compile(r"\b(?:" + "|".join(cfg.fixed_cost_keywords) + r")\b", re.I)
    turn: dict[str, float] = {}
    fixed: dict[str, float] = {}
    for t in ing(name).transactions:
        if t.date is None or t.amount is None or not t.description:
            continue
        k = t.date.strftime("%Y-%m")
        if t.amount > 0 and not ex.search(t.description):
            turn[k] = turn.get(k, 0.0) + t.amount
        elif t.amount < 0 and fx.search(t.description):
            fixed[k] = fixed.get(k, 0.0) + -t.amount
    return turn, fixed


def report(name: str) -> None:
    fs = findings(name)
    dup = [f for f in fs if f.code == "FL_DUP_INVOICE_REF"]
    seq = [f for f in fs if f.code == "FL_INVOICE_SEQUENCE_BREAK"]
    turn_f = [f for f in fs if f.code == "FL_TURNOVER_COST_MISMATCH"]
    named = set().union(*(rows_of(f) for f in dup)) if dup else set()
    print(f"\n===== statement_{name} =====")
    print("new-rule codes reported:", sorted({f.code for f in fs if f.code in NEW_CODES}) or "none")
    if name == "E":
        fakes = set(inserted_rows())
        hinted = {int(n) for f in dup for n in LIKELIER.findall(f.evidence)}
        print(f"likelier-copy hint names {len(hinted & fakes)} of {len(fakes)} fakes "
              f"(hint rows: {sorted(hinted)}; hinted genuine rows: {sorted(hinted - fakes)})")
        print(f"Dup finding names {len(named)} rows; genuine (named minus fakes): {len(named - fakes)}; "
              f"fakes missed: {sorted(fakes - named)}")
    else:
        print(f"Dup finding names {len(named)} rows; likelier-copy hints: "
              f"{len({n for f in dup for n in LIKELIER.findall(f.evidence)})} (expected none)")
    print("FL_INVOICE_SEQUENCE_BREAK fired:", bool(seq),
          "rows:", sorted(set().union(*(rows_of(f) for f in seq))) if seq else "none")
    if name == "E" and seq:
        fakes = set(inserted_rows())
        seq_rows = set().union(*(rows_of(f) for f in seq))
        print(f"  of those rows, {len(seq_rows & fakes)} are fakes, {len(seq_rows - fakes)} genuine")
    print("FL_TURNOVER_COST_MISMATCH fired:", bool(turn_f))
    for f in turn_f:
        for m in re.finditer(r"(\d{4}-\d{2}): turnover ([\d,.]+) against a median of ([\d,.]+) "
                             r"for the other months \(([\d.]+)x\)", f.evidence):
            print(f"  Rule C jump month {m.group(1)}: turnover {m.group(2)}, median {m.group(3)}, ratio {m.group(4)}x")
    turn, fixed = monthly_figures(name)
    print("monthly figures (turnover | fixed costs | ratio vs median of other months):")
    for k in sorted(turn):
        others = statistics.median(v for o, v in turn.items() if o != k)
        print(f"  {k}: turnover {turn[k]:,.2f} | fixed {fixed.get(k, 0.0):,.2f} | {turn[k] / others:.2f}x")
    if fixed:
        print(f"  median fixed costs: {statistics.median(fixed.values()):,.2f}")


def test_report_figures():
    for name in ("E", "F"):
        report(name)


def test_e_dup_invoice_ref_names_all_inserted_rows():
    dup = by_code("E", "FL_DUP_INVOICE_REF")
    assert dup, "FL_DUP_INVOICE_REF not reported on statement_E"
    fakes = inserted_rows()
    assert len(fakes) == len(set(fakes)) == 12
    named = set().union(*(rows_of(f) for f in dup))
    assert set(fakes) <= named, f"inserted rows not named: {sorted(set(fakes) - named)}"
    # evidence names each row too
    text = " ".join(f.evidence for f in dup)
    for r in fakes:
        assert f"row {r})" in text


def test_e_turnover_cost_mismatch_reported():
    assert by_code("E", "FL_TURNOVER_COST_MISMATCH"), "FL_TURNOVER_COST_MISMATCH not reported on statement_E"


def test_e_no_rule_skipped():
    assert not by_code("E", "DQ_RULE_SKIPPED")


@pytest.mark.parametrize("code", NEW_CODES)
def test_f_no_fabrication_finding(code):
    assert not by_code("F", code), f"{code} fired on genuine statement_F"
