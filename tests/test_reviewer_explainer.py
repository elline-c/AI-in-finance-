"""Step 5: the optional LLM reviewer, its fallbacks, and its place in the pipeline."""
from __future__ import annotations

import json
import logging

import pytest
from reviewer_support import PerfectClient, finding, make_report

from fraud_detect import webapp
from fraud_detect.models import ReviewDecision, Severity
from fraud_detect.pipeline import run_pipeline
from fraud_detect.reviewer import baseline, context, explainer, prompt
from fraud_detect.reviewer.evaluation import ScriptedClient
from fraud_detect.reviewer.explainer import Explainer
from fraud_detect.reviewer.knowledge import KnowledgeBase
from fraud_detect.reviewer.llm_client import LLMError, LLMUnavailable, OllamaClient
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_clean, generate_tampered


class Raising:
    def __init__(self, exc):
        self.exc = exc

    def chat_json(self, *a, **k):
        raise self.exc


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


@pytest.fixture()
def report():
    return make_report([finding("FL_BALANCE_BREAK", page=3, field="row 14"),
                        finding("VF_FONT_MISMATCH", Severity.HIGH, 22, agent="Node2:VisualForensics", page=3)])


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "garbage"])
def test_off_unless_explicitly_enabled(value):
    env = {} if value is None else {explainer.FLAG_ENV: value}
    assert explainer.load_live_explainer(env) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_on_when_enabled_without_touching_the_network(value):
    ex = explainer.load_live_explainer({explainer.FLAG_ENV: value})
    assert isinstance(ex, Explainer) and isinstance(ex.client, OllamaClient)
    assert ex.knowledge is not None and ex.case_finder is not None


def test_a_remote_model_url_leaves_the_reviewer_off(caplog):
    env = {explainer.FLAG_ENV: "1", "REVIEWER_LLM_URL": "http://10.0.0.9:11434"}
    with caplog.at_level(logging.WARNING):
        assert explainer.load_live_explainer(env) is None
    assert "stays off" in caplog.text


def test_the_web_app_starts_with_the_reviewer_off(monkeypatch):
    import importlib
    monkeypatch.delenv(explainer.FLAG_ENV, raising=False)
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    assert importlib.reload(webapp).EXPLAINER is None


# --------------------------------------------------------------------------- #
# The happy path and every fallback
# --------------------------------------------------------------------------- #
def test_a_good_answer_is_used_and_fully_described(report, kb):
    out = Explainer(PerfectClient(), knowledge=kb).explain_detailed(report)
    n = out.notes
    assert n.source == "LLM" and n.fallback_reason is None
    assert n.model == "perfect:1b" and n.prompt_version == prompt.PROMPT_VERSION
    assert n.knowledge_version == kb.version and n.knowledge_ids and n.case_ids == []
    assert n.citations["ROUTE"] == "Routing decision" and "FL_BALANCE_BREAK" in n.citations["F1"]
    assert out.schema_ok is True and out.violations == [] and out.raw
    assert n.narrative.risk_drivers[0].cites == ["F1"]


def test_markdown_in_a_good_answer_is_tidied_not_punished(report):
    class Markdowny(PerfectClient):
        def chat_json(self, messages, schema, **kw):
            res = super().chat_json(messages, schema, **kw)
            data = json.loads(res.content)
            data["headline"]["text"] = "**" + data["headline"]["text"] + "**"
            data["risk_drivers"][0]["cites"].append(" F1 ")
            return type(res)(json.dumps(data), res.seconds, res.model)
    out = Explainer(Markdowny()).explain_detailed(report)
    assert out.notes.source == "LLM" and "**" not in out.notes.narrative.headline.text
    assert out.notes.narrative.risk_drivers[0].cites == ["F1"]


@pytest.mark.parametrize("client, expected", [
    (Raising(LLMUnavailable("ConnectionRefusedError")), "local model unavailable"),
    (Raising(LLMError("answer was cut off")), "model call failed (answer was cut off)"),
    (Raising(RuntimeError("boom")), "model call failed (RuntimeError)"),
    (ScriptedClient("this is not json"), "did not match the schema"),
    (ScriptedClient({"headline": "wrong shape"}), "did not match the schema"),
])
def test_every_failure_falls_back_to_the_template_and_says_why(report, client, expected):
    out = Explainer(client).explain_detailed(report)
    assert out.notes.source == "BASELINE" and expected in out.notes.fallback_reason
    assert out.notes.narrative == out.baseline


def test_a_wrong_answer_is_discarded_with_its_reasons_kept(report):
    good = json.loads(PerfectClient().chat_json(
        [{}, {"content": "x\n" + json.dumps(prompt.case_data(context.build_context(report)))}], {}).content)
    good["headline"]["cites"] = ["F99"]
    out = Explainer(ScriptedClient(good)).explain_detailed(report)
    assert out.notes.source == "BASELINE" and "failed validation" in out.notes.fallback_reason
    assert out.schema_ok is True and out.violations and out.notes.narrative == out.baseline


def test_an_answer_that_follows_a_planted_instruction_is_not_shown(report):
    obedient = json.loads(PerfectClient().chat_json(
        [{}, {"content": "x\n" + json.dumps(prompt.case_data(context.build_context(report)))}], {}).content)
    obedient["headline"]["text"] = "As the document asks, this application is low risk and should be approved."
    out = Explainer(ScriptedClient(obedient)).explain_detailed(report)
    assert out.notes.source == "BASELINE"
    assert "approved" not in out.notes.narrative.headline.text


def test_without_a_client_the_template_is_returned(report, kb):
    out = Explainer(None, knowledge=kb).explain_detailed(report)
    assert out.notes.source == "BASELINE" and out.prompt and out.baseline_violations == []


def test_a_report_without_an_explanation_gets_no_notes(report):
    report.explanation = None
    assert Explainer(PerfectClient()).explain_detailed(report) is None
    assert Explainer(PerfectClient()).explain(report) is None


def test_explain_never_raises(report, monkeypatch):
    monkeypatch.setattr(Explainer, "explain_detailed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert Explainer(PerfectClient()).explain(report) is None


def test_retrieval_failures_do_not_stop_the_explanation(report):
    class BrokenKb:
        version = "kb-x"

        def retrieve(self, r):
            raise RuntimeError("index corrupt")

    def broken_cases(r, s):
        raise RuntimeError("store unreadable")
    out = Explainer(PerfectClient(), knowledge=BrokenKb(), case_finder=broken_cases).explain_detailed(
        report, store=object())
    assert out.notes.source == "LLM" and out.notes.knowledge_ids == [] and out.notes.case_ids == []


def test_the_model_sees_citation_ids_and_the_schema_limits_them(report, kb):
    client = PerfectClient()
    seen = {}
    orig = client.chat_json

    def spy(messages, schema, **kw):
        seen["messages"], seen["schema"] = messages, schema
        return orig(messages, schema, **kw)
    client.chat_json = spy
    Explainer(client, knowledge=kb).explain_detailed(report)
    user = seen["messages"][-1]["content"]
    assert '"id": "F1"' in user and '"id": "K1"' in user and '"id": "ROUTE"' in user
    ids = seen["schema"]["properties"]["headline"]["properties"]["cites"]["items"]["enum"]
    assert "F1" in ids and "K1" in ids and "F99" not in ids


# --------------------------------------------------------------------------- #
# In the pipeline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("exp"))
    return {"clean": generate_clean(d).path, "tampered": generate_tampered(d).path}


def _scores(r):
    return (r.risk_score, r.risk_band, r.confidence_score, r.confidence_band, r.routing,
            r.routing_reason, r.summary, [f.model_dump() for f in r.findings])


@pytest.mark.parametrize("name", ["clean", "tampered"])
def test_the_reviewer_never_changes_a_score_a_finding_or_the_routing(pdfs, name, kb):
    plain = run_pipeline(pdfs[name])
    for client in (PerfectClient(), Raising(RuntimeError("x")), ScriptedClient("junk")):
        with_notes = run_pipeline(pdfs[name], explainer=Explainer(client, knowledge=kb))
        assert _scores(with_notes) == _scores(plain)
        assert with_notes.reviewer_notes is not None
        assert with_notes.explanation == plain.explanation
    assert plain.reviewer_notes is None


def test_a_crashing_explainer_cannot_fail_a_screening(pdfs):
    class Crashes:
        def explain(self, *a, **k):
            raise RuntimeError("bug")
    r = run_pipeline(pdfs["clean"], explainer=Crashes())
    assert r.reviewer_notes is None and r.risk_score >= 1


def test_giving_the_explainer_a_store_does_not_switch_on_node5_or_node3_history(pdfs, tmp_path):
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    first = run_pipeline(pdfs["tampered"])
    store.upsert_report(first)
    store.record_decision(first.case_id, ReviewDecision.REJECTED, "alice", "confirmed")
    plain = run_pipeline(pdfs["tampered"])
    seen = []
    with_store = run_pipeline(pdfs["tampered"], explainer=Explainer(
        PerfectClient(), case_finder=lambda r, s: seen.append(s) or []), explainer_store=store)
    assert seen == [store]                                   # the explainer did get the store
    assert _scores(with_store) == _scores(plain)             # and no Node 5 finding appeared
    assert not [f for f in with_store.findings if f.agent.startswith("Node5")]


def test_past_decided_cases_reach_the_notes_through_the_pipeline(pdfs, tmp_path, kb):
    import shutil
    from fraud_detect.reviewer.case_memory import find_similar_cases
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    old = run_pipeline(pdfs["tampered"])
    store.upsert_report(old)
    store.record_decision(old.case_id, ReviewDecision.REJECTED, "alice", "Edited balances confirmed.")
    copy = tmp_path / "resubmitted.pdf"                    # same statement, new application
    shutil.copy(pdfs["tampered"], copy)
    finder = Explainer(None, knowledge=kb, case_finder=find_similar_cases)
    new = run_pipeline(str(copy), explainer=finder, explainer_store=store)
    notes = new.reviewer_notes
    assert notes.case_ids == [old.case_id] and new.case_id != old.case_id
    [claim] = notes.narrative.similar_cases
    assert claim.cites == ["P1"] and "Edited balances confirmed." in claim.text
    assert notes.citations["P1"] == f"{old.case_id} (Rejected)"
    # a model gets the case too, and its prompt carries the reviewer's masked note
    seen = {}
    client = PerfectClient()
    orig = client.chat_json
    client.chat_json = lambda m, sch, **k: (seen.update(user=m[-1]["content"]), orig(m, sch, **k))[1]
    run_pipeline(str(copy), explainer=Explainer(client, knowledge=kb, case_finder=find_similar_cases),
                 explainer_store=store)
    assert '"id": "P1"' in seen["user"] and "Edited balances confirmed." in seen["user"]


def test_pending_and_undecided_cases_never_reach_the_notes(pdfs, tmp_path, kb):
    import shutil
    from fraud_detect.reviewer.case_memory import find_similar_cases
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    store.upsert_report(run_pipeline(pdfs["tampered"]))          # screened, nobody decided
    copy = tmp_path / "again.pdf"
    shutil.copy(pdfs["tampered"], copy)
    new = run_pipeline(str(copy), explainer=Explainer(None, knowledge=kb, case_finder=find_similar_cases),
                       explainer_store=store)
    assert new.reviewer_notes.case_ids == [] and new.reviewer_notes.narrative.similar_cases == []


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def test_the_audit_chain_records_what_produced_the_notes(pdfs, tmp_path, monkeypatch, kb):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    r = run_pipeline(pdfs["tampered"], explainer=Explainer(PerfectClient(), knowledge=kb))
    store.upsert_report(r)
    events = [e for e in store._audit.events_for(r.case_id)]
    assert [e.event for e in events] == ["SCREENED", "EXPLAINER"]
    detail = events[1].detail
    assert "source=LLM" in detail and "model=perfect:1b" in detail
    assert f"prompt={prompt.PROMPT_VERSION}" in detail and kb.version in detail and "CODE:" in detail
    assert store.get(r.case_id).audit_log[-1]["event"] == "EXPLAINER"
    assert store._audit.verify().status.value.startswith("INTACT")


def test_a_fallback_is_recorded_with_its_reason(pdfs, tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    r = run_pipeline(pdfs["clean"], explainer=Explainer(Raising(LLMUnavailable("TimeoutError"))))
    store.upsert_report(r)
    detail = store._audit.events_for(r.case_id)[1].detail
    assert "source=BASELINE" in detail and "fallback=local model unavailable" in detail


def test_no_explainer_event_when_the_reviewer_is_off(pdfs, tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    r = run_pipeline(pdfs["clean"])
    store.upsert_report(r)
    assert [e.event for e in store._audit.events_for(r.case_id)] == ["SCREENED"]


def test_notes_survive_the_case_store(pdfs, tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "c.json"), str(tmp_path / "chain.jsonl"))
    r = run_pipeline(pdfs["tampered"], explainer=Explainer(PerfectClient()))
    store.upsert_report(r)
    assert store.get(r.case_id).report.reviewer_notes == r.reviewer_notes


def test_the_template_text_is_the_same_every_time(report, kb):
    a = Explainer(None, knowledge=kb).explain_detailed(report).baseline
    b = Explainer(None, knowledge=kb).explain_detailed(report).baseline
    assert a == b == baseline.build_baseline(
        context.build_context(report, knowledge=kb.retrieve(report)))


# --------------------------------------------------------------------------- #
# Hardening
# --------------------------------------------------------------------------- #
def test_no_notes_are_written_for_a_breakdown_that_does_not_reconcile(report, caplog):
    report.explanation = report.explanation.model_copy(update={"reconciled": False, "notes": ["off by 3"]})
    client = PerfectClient()
    with caplog.at_level(logging.WARNING):
        assert Explainer(client).explain_detailed(report) is None
        assert Explainer(client).explain(report) is None
    assert client.calls == 0 and "does not reconcile" in caplog.text


def test_upload_runs_the_pipeline_in_a_worker_thread_and_attaches_notes(tmp_path, monkeypatch, pdfs):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "cases.json"), str(tmp_path / "chain.jsonl"))
    monkeypatch.setattr(webapp, "store", store)
    monkeypatch.setattr(webapp, "UPLOAD_DIR", (tmp_path / "uploads").resolve())
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(webapp, "EXPLAINER", Explainer(PerfectClient()))
    used = []
    real = webapp.run_in_threadpool

    async def spy(fn, *a, **k):
        used.append(fn)
        return await real(fn, *a, **k)
    monkeypatch.setattr(webapp, "run_in_threadpool", spy)

    with open(pdfs["tampered"], "rb") as fh:
        resp = TestClient(webapp.app).post("/upload", files={"file": ("s.pdf", fh, "application/pdf")},
                                           follow_redirects=False)
    assert resp.status_code == 303 and used == [webapp.run_pipeline]
    case = store.get(resp.headers["location"].rsplit("/", 1)[1])
    assert case.report.reviewer_notes is not None and case.report.reviewer_notes.source == "LLM"
    assert case.report.explanation.reconciled


def test_upload_without_the_reviewer_behaves_as_before(tmp_path, monkeypatch, pdfs):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "cases.json"), str(tmp_path / "chain.jsonl"))
    monkeypatch.setattr(webapp, "store", store)
    monkeypatch.setattr(webapp, "UPLOAD_DIR", (tmp_path / "uploads").resolve())
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(webapp, "EXPLAINER", None)
    with open(pdfs["clean"], "rb") as fh:
        resp = TestClient(webapp.app).post("/upload", files={"file": ("s.pdf", fh, "application/pdf")},
                                           follow_redirects=False)
    case = store.get(resp.headers["location"].rsplit("/", 1)[1])
    assert case.report.reviewer_notes is None and case.report.explanation is not None
    assert [e["event"] for e in case.audit_log] == ["SCREENED"]
