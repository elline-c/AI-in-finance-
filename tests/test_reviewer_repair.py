"""The citation-repair stage: it fixes two slips, changes no wording, and hides nothing."""
from __future__ import annotations

import json

import pytest
from reviewer_support import PerfectClient, finding, make_report

from fraud_detect.models import Claim, ReviewDecision, Severity
from fraud_detect.reviewer import baseline, context, repair, validator
from fraud_detect.reviewer.context import SimilarCase
from fraud_detect.reviewer.evaluation import ScriptedClient
from fraud_detect.reviewer.explainer import Explainer
from fraud_detect.store import CaseStore


@pytest.fixture()
def report():
    return make_report([
        finding("FL_BALANCE_BREAK", page=1, field="row 8"),
        finding("FL_BALANCE_BREAK", page=1, field="row 9"),
        finding("VF_FONT_MISMATCH", Severity.HIGH, 22, agent="Node2:VisualForensics", page=2)])


@pytest.fixture()
def ctx(report):
    return context.build_context(report)


@pytest.fixture()
def good(ctx):
    return baseline.build_baseline(ctx)


def with_claim(n, field, claim, index=None):
    n = n.model_copy(deep=True)
    if index is None:
        setattr(n, field, claim)
    else:
        getattr(n, field)[index] = claim
    return n


def with_checks(n, *claims):
    n = n.model_copy(deep=True)
    n.suggested_checks = list(claims)
    return n


def kinds(vs):
    return {v.kind for v in vs}


# --------------------------------------------------------------------------- #
# The routing citation
# --------------------------------------------------------------------------- #
def test_a_routing_explanation_without_the_route_gets_it_added(good, ctx):
    slip = with_claim(good, "routing_explanation",
                      Claim(text="A person must review this case because of the findings.", cites=["F1"]))
    assert validator.CITATION_ROLE in kinds(validator.validate(slip, ctx))        # the validator stays strict
    fixed, repairs = repair.repair_narrative(slip, ctx)
    assert fixed.routing_explanation.cites == ["ROUTE", "F1"] and fixed.routing_explanation.text == slip.routing_explanation.text
    assert repairs == ["routing_explanation: added the routing decision (ROUTE) as a citation"]
    assert validator.validate(fixed, ctx) == []


def test_nothing_is_repaired_when_nothing_is_wrong(good, ctx):
    fixed, repairs = repair.repair_narrative(good, ctx)
    assert repairs == [] and fixed == good


def test_the_input_is_not_modified(good, ctx):
    slip = with_claim(good, "routing_explanation", Claim(text="A person must review it.", cites=["F1"]))
    before = slip.model_dump()
    repair.repair_narrative(slip, ctx)
    assert slip.model_dump() == before


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def test_a_row_in_another_listed_finding_gets_that_finding_cited(good, ctx):
    slip = with_checks(good, Claim(
        text="Recalculate the balance around row 8 and row 9 to verify the discrepancy.", cites=["F1"]))
    assert validator.FAITHFUL_ROW in kinds(validator.validate(slip, ctx))
    fixed, repairs = repair.repair_narrative(slip, ctx)
    assert fixed.suggested_checks[0].cites == ["F1", "F2"]
    assert repairs == ["suggested_checks[0]: added F2 as a citation for row 9"]
    assert fixed.suggested_checks[0].text == slip.suggested_checks[0].text
    assert validator.FAITHFUL_ROW not in kinds(validator.validate(fixed, ctx))


def test_an_invented_row_is_not_repaired_and_is_still_rejected(good, ctx):
    slip = with_checks(good, Claim(text="Look at row 4321 closely.", cites=["F1"]))
    fixed, repairs = repair.repair_narrative(slip, ctx)
    assert repairs == [] and fixed.suggested_checks[0].cites == ["F1"]
    assert validator.FAITHFUL_ROW in kinds(validator.validate(fixed, ctx))


def test_pages_are_never_repaired(good, ctx):
    slip = with_checks(good, Claim(text="Open page 2 and compare the font.", cites=["F1"]))
    fixed, repairs = repair.repair_narrative(slip, ctx)
    assert repairs == [] and validator.FAITHFUL_PAGE in kinds(validator.validate(fixed, ctx))


def test_the_citation_limit_is_respected(good, ctx):
    full = ["F1"] + ["A1"] * (validator.MAX_CITES - 1)
    slip = with_checks(good, Claim(text="Check row 9.", cites=full))
    fixed, repairs = repair.repair_narrative(slip, ctx)
    assert len(fixed.suggested_checks[0].cites) == validator.MAX_CITES and repairs == []


def test_a_row_inside_a_verified_quote_is_not_repaired(reports_with_case):
    ctx, n = reports_with_case
    text = "A person rejected it. \u201cRow 9 was edited by hand.\u201d"
    quoted = with_claim(n, "similar_cases", Claim(text=text, cites=["P1"]), 0)
    _, repairs = repair.repair_narrative(quoted, ctx)
    assert repairs == []


@pytest.fixture()
def reports_with_case(report):
    case = SimilarCase(case_id="CASE-OLD", decision="REJECTED", score=0.8, shared_codes=("FL_BALANCE_BREAK",),
                       reason="Row 9 was edited by hand.")
    ctx = context.build_context(report, cases=[case])
    return ctx, baseline.build_baseline(ctx)


# --------------------------------------------------------------------------- #
# In the explainer: recorded, audited, shown, never counted as clean
# --------------------------------------------------------------------------- #
class SlippyClient(PerfectClient):
    """Right answer, two citation slips."""
    model = "slippy:1b"

    def chat_json(self, messages, schema, **kw):
        res = super().chat_json(messages, schema, **kw)
        data = json.loads(res.content)
        data["routing_explanation"]["cites"] = ["A1"]
        data["suggested_checks"] = [{"text": "Recalculate the balance around row 8 and row 9.", "cites": ["F1"]}]
        return type(res)(json.dumps(data), res.seconds, res.model)


def test_an_answer_with_two_slips_is_accepted_after_repair_and_says_so(report):
    out = Explainer(SlippyClient()).explain_detailed(report)
    assert out.notes.source == "LLM" and out.violations == []
    assert len(out.notes.repairs) == 2 and out.repairs == out.notes.repairs
    assert out.notes.narrative.routing_explanation.cites[0] == "ROUTE"
    assert out.notes.narrative.suggested_checks[0].cites == ["F1", "F2"]


def test_a_clean_answer_has_no_repairs(report):
    assert Explainer(PerfectClient()).explain_detailed(report).notes.repairs == []


def test_an_invented_row_still_falls_back_to_the_template(report):
    class Invents(PerfectClient):
        def chat_json(self, messages, schema, **kw):
            res = super().chat_json(messages, schema, **kw)
            data = json.loads(res.content)
            data["suggested_checks"] = [{"text": "Recalculate the balance around row 999.", "cites": ["F1"]}]
            return type(res)(json.dumps(data), res.seconds, res.model)
    out = Explainer(Invents()).explain_detailed(report)
    assert out.notes.source == "BASELINE" and "row 999" in out.notes.fallback_reason
    assert out.notes.repairs == []                      # nothing from a rejected answer is recorded


def test_repairs_reach_the_audit_chain_and_the_page(tmp_path, monkeypatch, report):
    from fastapi.testclient import TestClient
    from fraud_detect import webapp
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    monkeypatch.setattr(webapp, "store", store)
    report.reviewer_notes = Explainer(SlippyClient()).explain(report)
    store.upsert_report(report)
    detail = store._audit.events_for(report.case_id)[-1].detail
    assert "repairs=2" in detail and "added F2 as a citation for row 9" in detail
    html = TestClient(webapp.app).get(f"/case/{report.case_id}").text
    assert "Citations fixed automatically before checking (2)" in html
    assert "The wording was not changed." in html


def test_no_repair_note_on_the_page_when_there_were_none(tmp_path, monkeypatch, report):
    from fastapi.testclient import TestClient
    from fraud_detect import webapp
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    monkeypatch.setattr(webapp, "store", store)
    report.reviewer_notes = Explainer(PerfectClient()).explain(report)
    store.upsert_report(report)
    assert "fixed automatically" not in TestClient(webapp.app).get(f"/case/{report.case_id}").text
    assert "repairs=" not in store._audit.events_for(report.case_id)[-1].detail


def test_old_stored_notes_without_a_repairs_field_still_load(report):
    from fraud_detect.models import ReviewerNotes
    notes = Explainer(PerfectClient()).explain(report)
    data = json.loads(notes.model_dump_json())
    data.pop("repairs")
    assert ReviewerNotes.model_validate(data).repairs == []


# --------------------------------------------------------------------------- #
# The evaluation never lets repairs hide
# --------------------------------------------------------------------------- #
class RouteSlipClient(PerfectClient):
    """Right answer; the routing explanation cites a confidence line instead of the routing decision."""
    model = "routeslip:1b"

    def chat_json(self, messages, schema, **kw):
        res = super().chat_json(messages, schema, **kw)
        data = json.loads(res.content)
        data["routing_explanation"]["cites"] = ["A1"]
        return type(res)(json.dumps(data), res.seconds, res.model)


def test_the_evaluation_counts_answers_accepted_only_after_repair(tmp_path):
    from fraud_detect.reviewer import evaluation
    docs = [d for d in evaluation.build_eval_set(str(tmp_path)) if d.name in ("tampered", "plain")]
    rep = evaluation.evaluate(docs, Explainer(RouteSlipClient()), live=True)
    m = rep.metrics
    assert m["accepted_rate"] == 1.0
    assert m["accepted_only_after_repair"] >= 1 and m["accepted_unrepaired_rate"] < m["accepted_rate"]
    assert "accepted only after a citation repair" in evaluation.render(rep)
    clean = evaluation.evaluate(docs, Explainer(PerfectClient()), live=True)
    assert clean.metrics["accepted_only_after_repair"] == 0 and clean.metrics["accepted_unrepaired_rate"] == 1.0
