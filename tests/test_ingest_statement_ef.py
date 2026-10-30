"""Task 11.3: Statement_E / Statement_F accuracy against the answer key (Requirement 11.1-11.6).

Reads external files from ~/Downloads (never copied into the repo; every test
skips when any of them is missing). Path constants are defined locally because
tests/ is not a package.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pytest

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
PERIOD = (date(2026, 4, 1), date(2026, 9, 30))
CASES = [("statement_E", STATEMENT_E, "monthly_as_shown"), ("statement_F", STATEMENT_F, "monthly")]


def money(s: str) -> float:
    return float(re.sub(r"[^\d.\-]", "", s))


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


@lru_cache(maxsize=None)
def key() -> dict:
    return json.loads(ANSWER_KEY.read_text())


@lru_cache(maxsize=None)
def result(name: str):
    path = {"statement_E": STATEMENT_E, "statement_F": STATEMENT_F}[name]
    return ingest(str(path))


def key_period() -> tuple[date, date]:
    m = re.match(r"(\d{1,2} \w{3} \d{4}) to (\d{1,2} \w{3} \d{4})", key()["statement_E"]["period"])
    a, b = (datetime.strptime(g, "%d %b %Y").date() for g in m.groups())
    return a, b


@pytest.mark.parametrize("name,_p,monthly_key", CASES)
def test_header_fields_match_answer_key(name, _p, monthly_key):
    k, res = key()[name], result(name)
    assert _norm(res.account_holder) == _norm(k["company"])
    assert _norm(res.uen) == _norm(k["uen"])
    assert _norm(res.account_number) == _norm(k["account_no"])
    assert key_period() == PERIOD
    assert (res.statement_period_start, res.statement_period_end) == PERIOD
    assert res.opening_balance is not None
    assert abs(res.opening_balance - money(k[monthly_key][0]["opening"])) <= TOL


@pytest.mark.parametrize("name,_p,monthly_key", CASES)
def test_monthly_values_match_answer_key(name, _p, monthly_key):
    months = key()[name][monthly_key]
    res = result(name)
    assert len(res.sections) == len(months) == 6
    for sec, m in zip(res.sections, months):
        assert sec.period_start is not None
        assert sec.period_start.strftime("%Y-%m") == m["month"]
        for attr, field in (("opening", "opening"), ("total_deposits", "total_deposits"),
                            ("total_withdrawals", "total_withdrawals"), ("closing", "closing")):
            got = getattr(sec, attr)
            assert got is not None, (m["month"], attr)
            assert abs(got - money(m[field])) <= TOL, (m["month"], attr, got, m[field])


@pytest.mark.parametrize("name,_p,monthly_key", CASES)
def test_file_closing_is_last_section_closing(name, _p, monthly_key):
    res = result(name)
    last_key = money(key()[name][monthly_key][-1]["closing"])
    assert res.closing_balance is not None
    assert abs(res.closing_balance - res.sections[-1].closing) <= TOL
    assert abs(res.closing_balance - last_key) <= TOL


@pytest.mark.parametrize("name,_p,_m", CASES)
def test_six_segments_with_distinct_periods(name, _p, _m):
    res = result(name)
    assert len(res.segments) == 6
    periods = {(s.period_start, s.period_end) for s in res.segments}
    assert len(periods) == 6
    assert all(a and b for a, b in periods)


@pytest.mark.parametrize("name,_p,_m", CASES)
def test_transactions_replay_to_file_closing(name, _p, _m):
    res = result(name)
    assert len(res.transactions) > 0
    replay = res.opening_balance + sum(t.amount or 0.0 for t in res.transactions)
    assert abs(replay - res.closing_balance) <= TOL, (replay, res.closing_balance)


def test_statement_e_inserted_rows_found_and_no_dq_no_txns():
    res = result("statement_E")
    rows = key()["statement_E"]["inserted_rows"]
    assert len(rows) == 12
    missing = []
    for r in rows:
        want_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        want_amt = money(r["fabricated_amount"])
        hit = any(
            t.date == want_date
            and t.date.strftime("%Y-%m") == r["month"]
            and t.amount is not None
            and t.amount > 0
            and abs(t.amount - want_amt) <= TOL
            for t in res.transactions
        )
        if not hit:
            missing.append(r)
    assert not missing, f"inserted rows not found as parsed credits: {missing}"
    assert "DQ_NO_TXNS" not in {f.code for f in res.data_quality_findings}
