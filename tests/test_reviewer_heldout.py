"""The held-out evaluation: its set, its seeded store, its extra criteria and its stability probe."""
from __future__ import annotations

import json

import pytest
from reviewer_support import PerfectClient, PrecedentAwareClient

from fraud_detect.models import ReviewDecision
from fraud_detect.reviewer import evaluation, heldout, llm_client
from fraud_detect.reviewer.case_memory import find_similar_cases
from fraud_detect.reviewer.evaluation import ScriptedClient, wilson_interval
from fraud_detect.reviewer.explainer import Explainer
from fraud_detect.reviewer.knowledge import KnowledgeBase
from fraud_detect.store import CaseStore


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


@pytest.fixture(scope="module")
def held(tmp_path_factory):
    return heldout.build_heldout_set(str(tmp_path_factory.mktemp("heldout")), per_layout=1)


def make(client, kb):
    return Explainer(client, knowledge=kb, case_finder=find_similar_cases)


@pytest.fixture(scope="module")
def small(held):
    """A seven-document slice with every kind of document, for tests of client behaviour."""
    import dataclasses
    keep = ["clean_0", "tampered_0", "plain_h", "inj_h_hidden"]
    tampered = [d for d in held.docs if d.kind == "corpus_tampered"][:2]
    genuine = [d for d in held.docs if d.kind == "corpus_genuine"][:1]
    chosen = tampered + genuine + [d for d in held.docs if d.name in keep]
    return dataclasses.replace(held, docs=chosen, precedents={}, decoy_ids=[])


@pytest.fixture(scope="module")
def offline(held, kb):
    return heldout.evaluate_heldout(held, make(None, kb), live=False)


# --------------------------------------------------------------------------- #
# The set
# --------------------------------------------------------------------------- #
def test_the_set_is_varied_and_labelled(held):
    names = [d.name for d in held.docs]
    assert len(names) == len(set(names)) >= 20
    kinds = {d.kind for d in held.docs}
    assert {"corpus_genuine", "corpus_tampered", "clean", "tampered", "plain", "injected"} <= kinds
    layouts = {d.layout for d in held.docs if d.layout}
    assert len(layouts) == 6                                   # every statement layout is present
    assert any(d.tags for d in held.docs if d.kind == "corpus_tampered")


def test_nothing_in_the_set_was_used_for_tuning(held, tmp_path):
    tuning = evaluation.build_eval_set(str(tmp_path))
    assert not {d.name for d in held.docs} & {d.name for d in tuning}
    assert evaluation.HIDDEN_INSTRUCTION not in {d.injected for d in held.docs}
    fresh = {d.injected for d in held.docs if d.injected}
    lowered = {x.lower() for x in fresh}               # one variant is deliberately in capitals
    assert len(lowered) >= 3 and lowered == {x.lower() for x in heldout.FRESH_INJECTIONS}
    assert not lowered & {d.injected.lower() for d in tuning if d.injected}


def test_the_planted_instructions_are_real_ones_the_security_layer_catches(held):
    from fraud_detect.nodes import ingestion, security
    for d in (x for x in held.docs if x.kind == "injected"):
        found = security.scan(ingestion.ingest(d.path))
        assert any(f.code == "SEC_EMBEDDED_INSTRUCTION" for f in found), d.name


def test_the_fingerprint_describes_the_set_not_the_pdf_bytes(tmp_path):
    a = heldout.build_heldout_set(str(tmp_path / "a"), per_layout=1, extras=False)
    b = heldout.build_heldout_set(str(tmp_path / "b"), per_layout=1, extras=False)
    assert a.set_id == b.set_id and len(a.set_id) == 12            # rebuilt PDFs differ in bytes, not in id
    assert heldout.build_heldout_set(str(tmp_path / "c"), seed=heldout.HELDOUT_SEED + 1, per_layout=1,
                                     extras=False).set_id != a.set_id
    assert heldout.build_heldout_set(str(tmp_path / "d"), per_layout=1, extras=True).set_id != a.set_id


# --------------------------------------------------------------------------- #
# The seeded store
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seeded(held, tmp_path_factory):
    from fraud_detect.corpus.scope import registered_companies
    d = tmp_path_factory.mktemp("seeded")
    store = CaseStore(str(d / "c.json"), str(d / "chain.jsonl"))
    held.precedents.clear()
    held.decoy_ids.clear()
    with registered_companies(held.companies):
        heldout.seed_store(held, store)
    return store


def test_precedents_are_decided_cases_confirmed_by_the_chain(held, seeded):
    assert 1 <= len(held.precedents) <= heldout.MAX_PRECEDENTS
    decided = {d.case_id: d for d in seeded.recall_decided_cases()}
    assert set(decided) == {p.case_id for p in held.precedents.values()}
    assert {d.decision for d in decided.values()} <= {"APPROVED", "REJECTED"}
    assert any(d.decision == "REJECTED" for d in decided.values())
    assert all(d.reason for d in decided.values())


def test_precedents_are_different_cases_from_the_documents_they_resemble(held, seeded):
    from fraud_detect.pipeline import run_pipeline
    by_name = {d.name: d for d in held.docs}
    for name, p in held.precedents.items():
        assert run_pipeline(by_name[name].path).case_id != p.case_id


def test_reviewer_reasons_contain_the_things_masking_has_to_handle(held, seeded):
    reasons = " ".join(p.reason for p in held.precedents.values())
    assert "@" in reasons and "Ignore all previous instructions" in reasons


def test_decoys_exist_but_are_not_decided(held, seeded):
    assert len(held.decoy_ids) == 3
    decided = {d.case_id for d in seeded.recall_decided_cases()}
    for cid in held.decoy_ids:
        case = seeded.get(cid)
        assert case is not None and case.review.decision in (ReviewDecision.PENDING, ReviewDecision.NEEDS_INFO)
        assert cid not in decided
    assert {seeded.get(c).review.decision for c in held.decoy_ids} == {
        ReviewDecision.PENDING, ReviewDecision.NEEDS_INFO}


# --------------------------------------------------------------------------- #
# Evaluating it
# --------------------------------------------------------------------------- #
def test_offline_run_passes_every_deterministic_check(offline):
    assert offline.passed, offline.checks
    assert offline.precedent_recall == 1.0 and offline.decoys_retrieved == 0
    assert offline.precedent_docs >= 1
    m = offline.base.metrics
    assert m["docs_with_cases"] >= offline.precedent_docs and m["identifier_leaks"] == 0
    assert m["baseline_valid_rate"] == 1.0


def test_the_template_is_valid_on_every_held_out_document(offline):
    assert all(not r.baseline_violations for r in offline.base.results)


def test_the_report_records_what_makes_it_comparable(offline):
    assert offline.prompt_version and offline.set_id
    data = offline.to_dict()
    assert data["set_id"] == offline.set_id and data["precedent_recall"] == 1.0
    json.dumps(data, default=str)


def test_a_good_model_passes_and_uses_the_precedents(small, kb):
    rep = heldout.evaluate_heldout(small, make(PrecedentAwareClient(), kb), live=True)
    assert rep.passed, rep.checks
    m = rep.base.metrics
    assert m["accepted_rate"] == 1.0 and m["similar_cases_use_rate"] == 1.0
    assert m["accepted_by_kind"]["corpus_tampered"]["accepted"] == m["accepted_by_kind"]["corpus_tampered"]["n"]
    n = len(small.docs)                                  # a perfect score on 7 documents proves little
    lo, hi = m["accepted_ci95"]
    assert hi == 1.0 and lo == pytest.approx(wilson_interval(n, n)[0]) and 0.55 < lo < 0.75


def test_similar_case_use_is_a_soft_signal_not_a_gate(small, kb):
    rep = heldout.evaluate_heldout(small, make(PerfectClient(), kb), live=True)   # never writes similar cases
    assert rep.base.metrics["similar_cases_use_rate"] == 0.0 and rep.passed


def test_junk_answers_fail_acceptance_but_never_break_the_retrieval_checks(small, kb):
    rep = heldout.evaluate_heldout(small, make(ScriptedClient("junk"), kb), live=True)
    assert not rep.passed and not rep.checks["model_answers_pass_validation"]
    assert rep.checks["seeded_precedent_retrieved"] and rep.checks["undecided_cases_never_retrieved"]
    assert rep.base.metrics["identifier_leaks"] == 0


def test_a_name_in_the_output_is_counted_as_a_leak(held, kb):
    class Leaky(PerfectClient):
        def chat_json(self, messages, schema, **kw):
            res = super().chat_json(messages, schema, **kw)
            data = json.loads(res.content)
            data["headline"]["text"] = "Meridian Logistics Pte Ltd was screened and sent to a human reviewer."
            return type(res)(json.dumps(data), res.seconds, res.model)
    docs = [d for d in held.docs if d.name == "plain_h"]
    rep = evaluation.evaluate(docs, make(Leaky(), kb), live=True)
    assert rep.results[0].identifier_leak and rep.metrics["identifier_leaks"] == 1
    assert not rep.checks["no_identifier_in_output"]


def test_a_decoy_that_leaks_into_retrieval_fails_the_check(small, kb, monkeypatch):
    def leaky_finder(report, store):
        from fraud_detect.reviewer.context import SimilarCase
        return [SimilarCase(case_id=small.decoy_ids[0], decision="REJECTED", score=1.0,
                            shared_codes=("FL_BALANCE_BREAK",), reason="x")] if small.decoy_ids else []
    rep = heldout.evaluate_heldout(small, Explainer(None, knowledge=kb, case_finder=leaky_finder), live=False)
    assert rep.decoys_retrieved > 0 and not rep.checks["undecided_cases_never_retrieved"]


def test_missing_precedent_retrieval_fails_the_check(small, kb):
    rep = heldout.evaluate_heldout(small, Explainer(None, knowledge=kb, case_finder=lambda r, s: []), live=False)
    assert rep.precedent_recall == 0.0 and not rep.checks["seeded_precedent_retrieved"]


# --------------------------------------------------------------------------- #
# Statistics, stability and plumbing
# --------------------------------------------------------------------------- #
def test_wilson_interval_values():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(8, 8)
    assert hi == 1.0 and 0.65 < lo < 0.70                 # 8 of 8 only shows "at least ~68%"
    lo36, _ = wilson_interval(36, 36)
    assert 0.90 < lo36 < 0.93 and lo36 > lo               # more documents, tighter bound
    lo, hi = wilson_interval(18, 36)
    assert lo < 0.5 < hi and hi - lo < 0.35


def test_the_stability_probe_reports_the_spread_across_seeds(held, kb):
    def factory(seed):
        # seed 22 is "unlucky": its model returns junk
        return make(ScriptedClient("junk") if seed == 22 else PerfectClient(), kb)
    out = heldout.stability_probe(held, factory, seeds=(11, 22, 33), n_docs=5)
    assert out["documents"] == 5 and out["temperature"] == heldout.STABILITY_TEMPERATURE
    assert out["accepted_by_seed"] == {11: 1.0, 22: 0.0, 33: 1.0}
    assert (out["accepted_min"], out["accepted_max"], out["accepted_spread"]) == (0.0, 1.0, 1.0)


def test_the_stability_subset_is_spread_across_the_set(held, kb):
    seen = []
    heldout.stability_probe(held, lambda s: make(PerfectClient(), kb), seeds=(1,), n_docs=6,
                            progress=lambda sd, r: seen.append(r.name))
    kinds = {next(d.kind for d in held.docs if d.name == n) for n in seen}
    assert len(seen) == 6 and len(kinds) >= 2


def test_the_client_takes_temperature_and_seed_but_defaults_stay_deterministic():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):          # noqa: N802
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"model": "m", "message": {"content": "{}"}, "done_reason": "stop"}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        llm_client.OllamaClient(base_url=url).chat_json([{"role": "user", "content": "x"}], {})
        llm_client.OllamaClient(base_url=url, temperature=0.3, seed=5).chat_json(
            [{"role": "user", "content": "x"}], {})
    finally:
        srv.shutdown()
        srv.server_close()
    assert (got[0]["options"]["temperature"], got[0]["options"]["seed"]) == (0.0, 7)
    assert (got[1]["options"]["temperature"], got[1]["options"]["seed"]) == (0.3, 5)


def test_render_shows_the_interval_the_reasons_and_the_stability(small, kb):
    rep = heldout.evaluate_heldout(small, make(PrecedentAwareClient(), kb), live=True)
    rep.stability = {"temperature": 0.3, "documents": 4, "accepted_by_seed": {1: 1.0, 2: 0.75},
                     "parsed_by_seed": {1: 1.0, 2: 1.0}, "accepted_min": 0.75, "accepted_max": 1.0,
                     "accepted_spread": 0.25}
    text = heldout.render(rep)
    for needle in ("95% interval", "accepted by kind", "why answers were rejected", "Stability at temperature 0.3",
                   "spread 25%", f"set {rep.set_id}", "RESULT: PASS"):
        assert needle in text, needle


def test_cli_offline_exits_zero_and_writes_a_report(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert heldout.main(["--per-layout", "1", "--no-extras", "--dir", str(tmp_path / "d"),
                      "--json", str(out)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["passed"] and data["precedent_recall"] == 1.0


def test_cli_live_mode_reports_a_missing_server(capsys, monkeypatch):
    monkeypatch.setenv("REVIEWER_LLM_URL", "http://127.0.0.1:9")
    assert heldout.main(["--live"]) == 2
    assert "Cannot reach the local model" in capsys.readouterr().err


def test_the_same_folder_gives_the_same_precedent_case_ids_so_runs_are_comparable(tmp_path):
    """Case ids hash the file path, and ties between similar past cases are broken by id, so a
    random working folder would change the prompt on every run."""
    from fraud_detect.corpus.scope import registered_companies
    ids = []
    for i in range(2):
        import shutil
        shutil.rmtree(tmp_path / "w", ignore_errors=True)
        held = heldout.build_heldout_set(str(tmp_path / "w"), per_layout=1, extras=False)
        store = CaseStore(str(tmp_path / f"c{i}.json"), str(tmp_path / f"chain{i}.jsonl"))
        with registered_companies(held.companies):
            heldout.seed_store(held, store)
        ids.append(sorted(p.case_id for p in held.precedents.values()) + sorted(held.decoy_ids))
    assert ids[0] == ids[1] and len(ids[0]) >= 4
