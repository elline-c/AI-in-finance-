"""Step 2: the deterministic risk / confidence / routing explanation.

The breakdown must (a) add up to Node 4's scores exactly, (b) derive the same routing,
(c) say so loudly when it does not, and (d) leave Node 4 itself untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.models import (
    DocumentMetadata, Finding, IngestionResult, RiskReport, Routing, Severity,
)
from fraud_detect.nodes import risk_synthesis as node4
from fraud_detect.pipeline import run_pipeline
from fraud_detect.reviewer import breakdown
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import (
    generate_clean, generate_medium, generate_tampered, generate_unreadable,
)

PINNED = json.loads((Path(__file__).parent / "ml_retraining" / "protected_hashes.json")
                    .read_text())["node4_pinned"]


def _finding(agent="Node3:FinancialLogic", sev=Severity.MEDIUM, weight=12.0, code="FL_X"):
    return Finding(agent=agent, code=code, title=f"title {code}", severity=sev,
                   weight=weight, evidence="e")


def _ing(dq=0, txns=True):
    from fraud_detect.models import Transaction
    return IngestionResult(
        source_path="x.pdf", page_count=1, raw_text="", metadata=DocumentMetadata(),
        transactions=[Transaction(index=1, amount=1.0, page=1)] if txns else [],
        data_quality_findings=[_finding("Node1:Ingestion", Severity.LOW, 3.0, f"DQ_{i}")
                               for i in range(dq)])


# --------------------------------------------------------------------------- #
# Numbers kept inside Node 4's private functions must match the pinned values
# --------------------------------------------------------------------------- #
def test_constants_match_the_values_pinned_for_node4():
    risk_literals = PINNED["numeric_literals_by_function"]["_compute_risk"]
    conf_literals = PINNED["numeric_literals_by_function"]["_compute_confidence"]
    assert breakdown.CORROBORATION_BONUS in risk_literals
    assert breakdown.MEDIUM_FLOOR in risk_literals
    assert breakdown.SCORE_MIN in risk_literals and breakdown.SCORE_MAX in risk_literals
    assert conf_literals[0] == breakdown.CONFIDENCE_BASE
    assert PINNED["module_constants"]["AUTO_CLEAR_MAX_RISK"] == node4.AUTO_CLEAR_MAX_RISK
    assert PINNED["module_constants"]["MIN_CONFIDENCE_FOR_AUTO"] == node4.MIN_CONFIDENCE_FOR_AUTO


# --------------------------------------------------------------------------- #
# Hand-built cases
# --------------------------------------------------------------------------- #
def test_no_findings_scores_the_minimum_with_an_explicit_line():
    lines = breakdown.risk_lines([])
    assert [l.id for l in lines] == ["LIMIT"]
    assert sum(l.points for l in lines) == node4._compute_risk([]) == 1


def test_bonus_needs_two_independent_agents_with_a_high_finding():
    one = [_finding("A", Severity.HIGH, 30), _finding("A", Severity.HIGH, 30)]
    two = [_finding("A", Severity.HIGH, 30), _finding("B", Severity.HIGH, 30)]
    assert "BONUS" not in {l.id for l in breakdown.risk_lines(one)}
    assert "BONUS" in {l.id for l in breakdown.risk_lines(two)}


def test_floor_line_appears_only_when_a_material_finding_sits_below_the_floor():
    low = [_finding(sev=Severity.MEDIUM, weight=5)]
    lines = {l.id: l for l in breakdown.risk_lines(low)}
    assert lines["FLOOR"].points == breakdown.MEDIUM_FLOOR - 5
    assert "FLOOR" not in {l.id for l in breakdown.risk_lines([_finding(weight=40)])}
    assert "FLOOR" not in {l.id for l in breakdown.risk_lines(
        [_finding(sev=Severity.LOW, weight=5)])}


def test_fractional_weights_get_a_rounding_line():
    fs = [_finding(sev=Severity.LOW, weight=5.4), _finding(sev=Severity.LOW, weight=5.4)]
    lines = breakdown.risk_lines(fs)
    assert "ROUND" in {l.id for l in lines}
    assert sum(l.points for l in lines) == pytest.approx(node4._compute_risk(fs))


def test_total_over_100_is_capped_with_a_readable_reason():
    fs = [_finding("A", Severity.HIGH, 40), _finding("B", Severity.HIGH, 40),
          _finding("C", Severity.HIGH, 40)]
    lines = {l.id: l for l in breakdown.risk_lines(fs)}
    assert lines["LIMIT"].points < 0
    assert "cannot exceed 100" in lines["LIMIT"].reading
    assert sum(l.points for l in lines.values()) == 100


@pytest.mark.parametrize("ing, code", [
    (_ing(dq=2), "CONF_DATA_QUALITY"),
    (_ing(txns=False), "CONF_NO_TRANSACTIONS"),
])
def test_data_quality_and_no_transaction_adjustments_are_named(ing, code):
    lines, problems = breakdown.confidence_lines(ing, ing.data_quality_findings, 10)
    assert code in {l.code for l in lines}
    assert problems == []


def test_corroboration_single_dimension_and_thin_evidence_are_named():
    both = [_finding("Node2:VisualForensics", Severity.MEDIUM), _finding("Node3:FinancialLogic")]
    lines, _ = breakdown.confidence_lines(_ing(), both, 40)
    assert "CONF_CORROBORATION" in {l.code for l in lines}

    visual_only = [_finding("Node2:VisualForensics", Severity.LOW, 25)]
    lines, _ = breakdown.confidence_lines(_ing(), visual_only, 30)
    codes = {l.code for l in lines}
    assert {"CONF_SINGLE_DIMENSION", "CONF_THIN_EVIDENCE"} <= codes


def test_every_adjustment_has_a_plain_language_reading():
    both = [_finding("Node2:VisualForensics", Severity.LOW, 25)]
    lines, _ = breakdown.confidence_lines(_ing(dq=3, txns=False), both, 30)
    assert all(len(l.reading) > 25 for l in lines)
    assert lines[0].code == "CONF_BASE" and lines[0].points == 85


# --------------------------------------------------------------------------- #
# Properties: parity with Node 4 on random inputs
# --------------------------------------------------------------------------- #
_agents = st.sampled_from(["Node1:Ingestion", "Node2:VisualForensics", "Node3:FinancialLogic",
                           "Node5:Memory", "SecurityLayer:InstructionScan"])
_findings = st.lists(st.builds(
    Finding, agent=_agents, code=st.just("X_CODE"), title=st.just("t"),
    severity=st.sampled_from(list(Severity)),
    weight=st.one_of(st.integers(0, 40).map(float), st.floats(0, 40, allow_nan=False)),
    evidence=st.just("e")), max_size=14)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(findings=_findings)
def test_risk_lines_always_add_up_to_node4(findings):
    lines = breakdown.risk_lines(findings)
    assert sum(l.points for l in lines) == pytest.approx(node4._compute_risk(findings), abs=1e-6)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(findings=_findings, dq=st.integers(0, 8), txns=st.booleans(), risk=st.integers(1, 100))
def test_confidence_lines_always_add_up_to_node4(findings, dq, txns, risk):
    ing = _ing(dq=dq, txns=txns)
    expected, _ = node4._compute_confidence(ing, findings, risk)
    lines, problems = breakdown.confidence_lines(ing, findings, risk)
    assert problems == []
    assert sum(l.points for l in lines) == pytest.approx(expected, abs=1e-6)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(findings=_findings, risk=st.integers(1, 100), conf=st.integers(1, 100))
def test_routing_always_matches_node4(findings, risk, conf):
    cause, routing = breakdown.routing_cause(findings, risk, conf)
    assert routing == node4._route(risk, conf, findings)[0]
    assert (cause.primary == breakdown.AUTO_CLEAR) == (routing == Routing.AUTO_CLEAR)
    if cause.primary != breakdown.AUTO_CLEAR:
        assert cause.primary not in cause.also_applies


# --------------------------------------------------------------------------- #
# Real statements: reconciled end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("bd"))
    return {n: run_pipeline(g(d).path) for n, g in {
        "clean": generate_clean, "tampered": generate_tampered,
        "medium": generate_medium, "unreadable": generate_unreadable}.items()}


@pytest.mark.parametrize("name", ["clean", "tampered", "medium", "unreadable"])
def test_pipeline_attaches_a_reconciled_explanation(reports, name):
    r = reports[name]
    assert r.explanation is not None
    assert r.explanation.reconciled, r.explanation.notes
    assert r.explanation.risk_total == r.risk_score
    assert r.explanation.confidence_total == r.confidence_score


def test_routing_cause_names_what_really_drove_the_routing(reports):
    assert reports["clean"].explanation.routing.primary == breakdown.AUTO_CLEAR
    assert reports["tampered"].explanation.routing.primary == breakdown.MATERIAL_FINDING
    assert reports["medium"].explanation.routing.primary in (
        breakdown.MATERIAL_FINDING, breakdown.LOW_CONFIDENCE, breakdown.RISK_THRESHOLD)
    # low-confidence path: nothing material, confidence under the auto-clear minimum
    cause, routing = breakdown.routing_cause([], 5, node4.MIN_CONFIDENCE_FOR_AUTO - 1)
    assert cause.primary == breakdown.LOW_CONFIDENCE and routing == Routing.HUMAN_REVIEW
    assert str(node4.MIN_CONFIDENCE_FOR_AUTO) in cause.statement


def test_material_findings_are_listed_by_citation_id(reports):
    r = reports["tampered"]
    ids = set(r.explanation.routing.finding_ids)
    expected = {breakdown.finding_id(i) for i, f in enumerate(r.findings)
                if f.severity in (Severity.MEDIUM, Severity.HIGH)}
    assert ids == expected and ids


def test_a_mismatch_is_reported_not_hidden(reports):
    r = reports["tampered"].model_copy(deep=True)
    r.risk_score = r.risk_score - 7
    ing = _ing()
    # rebuild only the score-dependent parts against the doctored report
    expl = breakdown.explain_scores(ing, r)
    assert not expl.reconciled
    assert any("risk lines add up" in n for n in expl.notes)


def test_explanation_survives_the_case_store_and_old_reports_still_load(tmp_path, reports):
    store = CaseStore(str(tmp_path / "cases.json"), str(tmp_path / "chain.jsonl"))
    store.upsert_report(reports["tampered"])
    back = store.get(reports["tampered"].case_id).report
    assert back.explanation == reports["tampered"].explanation

    old = json.loads(reports["clean"].model_dump_json())
    old.pop("explanation")
    old.pop("reviewer_notes")
    assert RiskReport.model_validate(old).explanation is None


def test_format_breakdown_shows_totals_and_warns_when_unreconciled(reports):
    r = reports["unreadable"]
    text = breakdown.format_breakdown(r.explanation, r)
    assert f"= {r.risk_score}" in text
    assert "WARNING" not in text
    bad = r.explanation.model_copy(update={"reconciled": False, "notes": ["x"]})
    assert "WARNING" in breakdown.format_breakdown(bad, r)


def test_node4_file_is_not_modified_by_this_feature():
    import hashlib
    pinned = json.loads((Path(__file__).parent / "ml_retraining" / "protected_hashes.json")
                        .read_text())["sha256"]
    path = Path(node4.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned["fraud_detect/nodes/risk_synthesis.py"]


# --------------------------------------------------------------------------- #
# What the confidence score means for the analyst
# --------------------------------------------------------------------------- #
def _cause(primary):
    from fraud_detect.models import RoutingCause
    return RoutingCause(primary=primary, statement="s")


def test_meaning_when_confidence_is_below_the_auto_clear_minimum():
    text = breakdown.confidence_meaning(52, _cause(breakdown.LOW_CONFIDENCE))
    assert "At 52" in text and "below the auto-clear minimum of 60" in text
    assert "a person must review this case whatever the risk score is" in text
    assert "does not measure how likely it is that the statement was tampered with" in text


def test_meaning_when_a_finding_not_confidence_sends_the_case_to_a_person():
    text = breakdown.confidence_meaning(79, _cause(breakdown.MATERIAL_FINDING))
    assert "meets the auto-clear minimum of 60" in text and "not what sends this case to a person" in text


def test_meaning_for_a_clean_case_that_clears():
    text = breakdown.confidence_meaning(85, _cause(breakdown.AUTO_CLEAR))
    assert "lets a clean, low-risk case clear without a person" in text


def test_meaning_uses_no_internal_labels_and_no_verdict_words():
    from fraud_detect.reviewer import validator
    for primary in (breakdown.MATERIAL_FINDING, breakdown.LOW_CONFIDENCE, breakdown.RISK_THRESHOLD,
                    breakdown.AUTO_CLEAR):
        for conf in (10, 59, 60, 100):
            text = breakdown.confidence_meaning(conf, _cause(primary))
            assert not validator._RAW_LABEL_RE.search(text)
            assert not any(rx.search(text) for rx in validator._VERDICT_RES)


def test_every_real_explanation_carries_the_meaning_for_its_own_score(reports):
    for name, r in reports.items():
        ex = r.explanation
        assert ex.confidence_meaning.startswith(f"At {r.confidence_score},"), name
        below = r.confidence_score < 60
        assert ("below the auto-clear minimum" in ex.confidence_meaning) == below


def test_old_explanations_without_the_meaning_still_load(reports):
    data = json.loads(reports["clean"].model_dump_json())
    data["explanation"].pop("confidence_meaning")
    assert RiskReport.model_validate(data).explanation.confidence_meaning == ""
