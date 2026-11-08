"""Step 3: the breakdown (and later the reviewer notes) in the analyst case view."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_detect import webapp
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_clean, generate_tampered


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_SIGNING_KEY_PATH", raising=False)
    store = CaseStore(str(tmp_path / "cases.json"), str(tmp_path / "chain.jsonl"))
    monkeypatch.setattr(webapp, "store", store)
    return store, TestClient(webapp.app), str(tmp_path)


def _screen(store, path):
    report = run_pipeline(path)
    store.upsert_report(report)
    return report


def test_case_page_shows_the_breakdown(env):
    store, client, tmp = env
    r = _screen(store, generate_tampered(tmp).path)
    html = client.get(f"/case/{r.case_id}").text
    assert "Why this score" in html
    assert f"Risk score: how {r.risk_score} was reached" in html
    assert f"Confidence: how {r.confidence_score} was reached" in html
    assert "Corroboration bonus" in html
    assert "FL_BALANCE_BREAK" in html
    assert "Human review." in html
    assert "Also true: a risk score above the auto-clear limit" in html
    assert "does not match the scores" not in html


def test_auto_cleared_case_says_why_it_cleared(env):
    store, client, tmp = env
    r = _screen(store, generate_clean(tmp).path)
    html = client.get(f"/case/{r.case_id}").text
    assert "Auto-clear." in html
    assert "no MEDIUM or HIGH finding" in html


def test_table_totals_equal_the_scores(env):
    store, client, tmp = env
    r = _screen(store, generate_tampered(tmp).path)
    html = client.get(f"/case/{r.case_id}").text
    assert f"<b>{r.risk_score}</b></td><td><b>Risk score</b>" in html
    assert f"<b>{r.confidence_score}</b></td><td><b>Confidence score</b>" in html


def test_old_report_without_a_breakdown_gets_a_notice_not_an_error(env):
    store, client, tmp = env
    r = _screen(store, generate_clean(tmp).path)
    stored = store.get(r.case_id)
    stored.report.explanation = None
    data = store._read()
    import json
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    resp = client.get(f"/case/{r.case_id}")
    assert resp.status_code == 200
    assert "screened" in resp.text and "before the breakdown existed" in resp.text


def test_unreconciled_breakdown_is_flagged_loudly(env):
    store, client, tmp = env
    r = _screen(store, generate_clean(tmp).path)
    stored = store.get(r.case_id)
    stored.report.explanation = stored.report.explanation.model_copy(
        update={"reconciled": False, "notes": ["risk lines add up to 3, but the report says 1"]})
    import json
    data = store._read()
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    html = client.get(f"/case/{r.case_id}").text
    assert "does not match the scores" in html
    assert "risk lines add up to 3" in html


def test_finding_text_is_escaped_in_the_breakdown(env):
    store, client, tmp = env
    r = _screen(store, generate_tampered(tmp).path)
    stored = store.get(r.case_id)
    line = stored.report.explanation.risk_lines[0]
    line.label = "<script>alert(1)</script>"
    import json
    data = store._read()
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    html = client.get(f"/case/{r.case_id}").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# --------------------------------------------------------------------------- #
# Reviewer notes panel
# --------------------------------------------------------------------------- #
def _store_with_notes(store, tmp, explainer, name="tampered"):
    gen = generate_tampered(tmp) if name == "tampered" else generate_clean(tmp)
    report = run_pipeline(gen.path, explainer=explainer)
    store.upsert_report(report)
    return report


def test_no_notes_panel_while_the_reviewer_is_off(env):
    store, client, tmp = env
    r = _screen(store, generate_tampered(tmp).path)
    html = client.get(f"/case/{r.case_id}").text
    assert 'id="reviewer-notes"' not in html and "AI-generated" not in html


def test_ai_notes_are_labelled_advisory_and_show_their_evidence(env):
    from reviewer_support import PerfectClient
    from fraud_detect.reviewer.explainer import Explainer
    store, client, tmp = env
    r = _store_with_notes(store, tmp, Explainer(PerfectClient()))
    html = client.get(f"/case/{r.case_id}").text
    assert 'id="reviewer-notes"' in html and "AI-generated · advisory" in html
    assert "never change a score or a routing" in html
    assert "FL_BALANCE_BREAK" in html                       # a citation chip with its label
    assert "model perfect:1b" in html and "prompt reviewer-v7" in html
    assert "Template text" not in html


def test_template_notes_say_no_ai_was_used_and_why(env):
    from fraud_detect.reviewer.evaluation import ScriptedClient
    from fraud_detect.reviewer.explainer import Explainer
    from fraud_detect.reviewer.knowledge import KnowledgeBase
    store, client, tmp = env
    r = _store_with_notes(store, tmp, Explainer(ScriptedClient("junk"), knowledge=KnowledgeBase.load()))
    html = client.get(f"/case/{r.case_id}").text
    assert "Template text · no AI used" in html and "AI-generated" not in html
    assert "AI text not used: model answer did not match the schema" in html
    assert "Suggested checks" in html and "documents: CODE:" in html


def test_note_text_is_escaped(env):
    from reviewer_support import PerfectClient
    from fraud_detect.reviewer.explainer import Explainer
    store, client, tmp = env
    r = _store_with_notes(store, tmp, Explainer(PerfectClient()))
    stored = store.get(r.case_id)
    stored.report.reviewer_notes.narrative.headline.text = "<img src=x onerror=alert(1)>"
    stored.report.reviewer_notes.citations["ROUTE"] = "<script>x</script>"
    import json
    data = store._read()
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    html = client.get(f"/case/{r.case_id}").text
    assert "<img src=x" not in html and "<script>x</script>" not in html
    assert "&lt;img src=x" in html


def test_turning_the_reviewer_on_does_not_change_the_breakdown_or_the_scores(env):
    from reviewer_support import PerfectClient
    from fraud_detect.reviewer.explainer import Explainer
    store, client, tmp = env
    plain = _screen(store, generate_clean(tmp).path)
    with_notes = run_pipeline(generate_clean(tmp).path, explainer=Explainer(PerfectClient()))
    assert with_notes.explanation == plain.explanation
    assert (with_notes.risk_score, with_notes.confidence_score, with_notes.routing) == (
        plain.risk_score, plain.confidence_score, plain.routing)


def test_repeated_citations_show_one_chip_per_distinct_label(env):
    from reviewer_support import PerfectClient
    from fraud_detect.reviewer.explainer import Explainer
    store, client, tmp = env
    r = _store_with_notes(store, tmp, Explainer(PerfectClient()))
    stored = store.get(r.case_id)
    n = stored.report.reviewer_notes
    n.citations.update({"F3": "SAME LABEL", "F4": "SAME LABEL", "F5": "OTHER LABEL"})
    n.narrative.headline.cites = ["F3", "F4", "F5", "F3"]
    import json
    data = store._read()
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    html = client.get(f"/case/{r.case_id}").text
    assert html.count(">SAME LABEL<") == 1 and html.count(">OTHER LABEL<") == 1


def test_the_meaning_of_confidence_is_shown_in_both_panels(env):
    from reviewer_support import PerfectClient
    from fraud_detect.reviewer.explainer import Explainer
    store, client, tmp = env
    r = _store_with_notes(store, tmp, Explainer(PerfectClient()))
    html = client.get(f"/case/{r.case_id}").text
    meaning = r.explanation.confidence_meaning
    assert meaning and html.count(meaning) == 2           # once under the table, once in the notes
    assert "What it means:" in html
    notes_panel = html.split('id="reviewer-notes"')[1].split("Statement details")[0]
    assert "VERY_HIGH" not in notes_panel and "HUMAN_REVIEW" not in notes_panel


def test_an_old_explanation_without_the_meaning_falls_back_to_the_generic_note(env):
    store, client, tmp = env
    r = _screen(store, generate_clean(tmp).path)
    stored = store.get(r.case_id)
    stored.report.explanation.confidence_meaning = ""
    import json
    data = store._read()
    data[r.case_id] = json.loads(stored.model_dump_json())
    store._write(data)
    html = client.get(f"/case/{r.case_id}").text
    assert "It is not a probability of fraud" in html
