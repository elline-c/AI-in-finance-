"""Characterisation test (task 1.2, Requirement 15).

Node 3 must raise none of the fabrication codes on clean input. This keeps
passing after the rules exist: it asserts the codes are ABSENT on the clean
fixture, not that the file has no findings at all.

Baseline recorded before the rules: 571 passed, 1 xfailed.
"""
from fraud_detect.nodes.financial_logic import analyze
from fraud_detect.nodes.ingestion import ingest
from fraud_detect.synthetic import generate_clean

from tests.fabrication.fixtures import business_file

FABRICATION_CODES = {
    "FL_DUP_INVOICE_REF",
    "FL_INVOICE_SEQUENCE_BREAK",
    "FL_TURNOVER_COST_MISMATCH",
    "DQ_RULE_SKIPPED",
}


def test_fixture_business_file_has_no_fabrication_codes():
    _txns, ing = business_file()
    codes = {f.code for f in analyze(ing).findings}
    assert not (codes & FABRICATION_CODES)


def test_generator_clean_sample_has_no_node3_findings(tmp_path):
    gen = generate_clean(str(tmp_path))
    ing = ingest(gen.path)
    assert analyze(ing).findings == []
