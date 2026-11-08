"""Regression sweep for the Node 3 fabrication rules (Requirements 15.1-15.4).

Generator statements carry no invoice references and no fixed-cost pattern, so
none of the fabrication codes may fire on them. Everything runs in memory
(ingest + nodes); no CaseStore is created and nothing is written to data/.
"""
from __future__ import annotations

import pytest

from fraud_detect import synthetic
from fraud_detect.models import AgentReport, Routing
from fraud_detect.nodes import financial_logic, risk_synthesis, security, visual_forensics
from fraud_detect.nodes.ingestion import ingest

FABRICATION_CODES = {
    "FL_DUP_INVOICE_REF",
    "FL_INVOICE_SEQUENCE_BREAK",
    "FL_TURNOVER_COST_MISMATCH",
    "DQ_RULE_SKIPPED",
}

# Every issue kind used in tests/test_detection.py::PLANTED_TO_CODES.
ISSUE_KINDS = [
    "ALTERED_BALANCE", "ALTERED_NUMBER", "FORMAT_INCONSISTENCY",
    "METADATA_MISMATCH", "PERIOD_GAP", "SUSPICIOUS_TXN",
    "STRUCTURING", "UEN_MISMATCH",
]

# Generation is fast (~0.1s per statement), so every kind runs for all 20 seeds.
TAMPERED_SEEDS = range(1, 21)
CLEAN_SEEDS = range(1, 21)


def _fab_codes(ing) -> set[str]:
    return {f.code for f in financial_logic.analyze(ing).findings} & FABRICATION_CODES


def _screen(path):
    ing = ingest(str(path))
    reports = [
        AgentReport(agent=security.AGENT, findings=security.scan(ing)),
        visual_forensics.analyze(ing),
        financial_logic.analyze(ing),
    ]
    return ing, risk_synthesis.synthesize(ing, reports)


@pytest.mark.parametrize("seed", CLEAN_SEEDS)
def test_clean_generator_has_no_fabrication_findings(tmp_path, seed):
    gen = synthetic.generate_clean(str(tmp_path), seed=seed)
    assert _fab_codes(ingest(gen.path)) == set(), f"clean seed={seed}"


@pytest.mark.parametrize("seed", TAMPERED_SEEDS)
@pytest.mark.parametrize("issue", ISSUE_KINDS)
def test_tampered_generator_has_no_fabrication_findings(tmp_path, seed, issue):
    gen = synthetic.generate_tampered(str(tmp_path), seed=seed, issues=[issue])
    assert _fab_codes(ingest(gen.path)) == set(), f"tampered seed={seed} issue={issue}"


@pytest.mark.parametrize("seed", TAMPERED_SEEDS)
def test_tampered_default_all_issues_has_no_fabrication_findings(tmp_path, seed):
    gen = synthetic.generate_tampered(str(tmp_path), seed=seed)
    assert _fab_codes(ingest(gen.path)) == set(), f"tampered(all) seed={seed}"


@pytest.mark.parametrize("seed", TAMPERED_SEEDS)
def test_medium_generator_has_no_fabrication_findings(tmp_path, seed):
    gen = synthetic.generate_medium(str(tmp_path), seed=seed)
    assert _fab_codes(ingest(gen.path)) == set(), f"medium seed={seed}"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 20])
def test_clean_sample_auto_clears_through_pipeline(tmp_path, seed):
    gen = synthetic.generate_clean(str(tmp_path), seed=seed)
    _, report = _screen(gen.path)
    assert report.routing == Routing.AUTO_CLEAR
    assert report.risk_score <= 5
    assert not ({f.code for f in report.findings} & FABRICATION_CODES)


def test_clean_default_seed_auto_clears_through_pipeline(tmp_path):
    gen = synthetic.generate_clean(str(tmp_path))
    _, report = _screen(gen.path)
    assert report.routing == Routing.AUTO_CLEAR
    assert report.risk_score <= 5
