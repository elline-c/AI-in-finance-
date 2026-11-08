"""Step 6a: the curated knowledge base (policy text and finding-code catalogue)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from reviewer_support import finding, make_report

from fraud_detect.models import Severity
from fraud_detect.reviewer import baseline, context, knowledge, validator
from fraud_detect.reviewer.knowledge import KnowledgeBase

REPO = Path(__file__).resolve().parents[1]
PINNED = json.loads((REPO / "tests/ml_retraining/protected_hashes.json").read_text())["node4_pinned"]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


def _codes_in_source() -> set[str]:
    pat = re.compile(r"[\"']((?:DQ|FL|VF|MEM|ML|SEC)_[A-Z0-9_]+)[\"']")
    found: set[str] = set()
    for path in (REPO / "fraud_detect").rglob("*.py"):
        parts = path.relative_to(REPO / "fraud_detect").parts
        if parts[0] in ("corpus", "reviewer", "ml"):
            continue
        found |= set(pat.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_code_the_detectors_emit_has_a_catalogue_entry(kb):
    missing = _codes_in_source() - set(kb.codes)
    assert not missing, f"add these to finding_codes.json: {sorted(missing)}"


def test_the_catalogue_has_no_stale_entries(kb):
    stale = set(kb.codes) - _codes_in_source()
    # The ML codes are built in fraud_detect/nodes/ml_*.py, which the scan above covers.
    assert not stale, f"no detector emits these any more: {sorted(stale)}"


def test_every_entry_is_complete_and_readable(kb):
    for code, e in kb.codes.items():
        for key in ("title", "meaning", "why", "confirm"):
            assert e.get(key, "").strip(), f"{code} lacks {key}"
        doc = kb.code_doc(code)
        assert doc.doc_id == f"CODE:{code}" and "How to confirm:" in doc.text


def test_curated_text_never_trips_the_output_validator_rules(kb):
    texts = [d.text for d in kb.policy] + [kb.code_doc(c).text for c in kb.codes]
    for text in texts:
        for _cat, _label, rx in validator._OUTPUT_INSTRUCTION_PATTERNS:
            assert not rx.search(text), (rx.pattern, text[:80])
        for rx in validator._VERDICT_RES:
            assert not rx.search(text), (rx.pattern, text[:80])
        assert not validator._BAD_TEXT_RE.search(text)


def test_how_to_confirm_is_extracted_exactly_for_every_code(kb):
    for code, e in kb.codes.items():
        m = baseline._HOW_TO_RE.search(kb.code_doc(code).text)
        assert m and " ".join(m.group(1).split()) == e["confirm"], code


@pytest.mark.parametrize("code", sorted(KnowledgeBase.load().codes))
def test_a_baseline_with_the_catalogue_entry_passes_the_validator_for_every_code(kb, code):
    report = make_report([finding(code, Severity.MEDIUM, 12, page=2, field="row 5"),
                          finding("VF_FONT_MISMATCH", Severity.HIGH, 22, agent="Node2:VisualForensics")])
    ctx = context.build_context(report, knowledge=kb.retrieve(report))
    n = baseline.build_baseline(ctx)
    assert validator.validate(n, ctx) == []
    assert any(c.text.startswith(code) for c in n.suggested_checks) or len(n.suggested_checks) >= 1


def test_policy_numbers_come_from_the_code_not_from_the_file(kb):
    text = " ".join(d.text for d in kb.policy)
    assert "{" not in text and "}" not in text
    from fraud_detect.nodes import risk_synthesis as n4
    routing = next(d for d in kb.policy if d.doc_id == "POL-ROUTING").text
    assert f"below {n4.MIN_CONFIDENCE_FOR_AUTO}" in routing and f"above {n4.AUTO_CLEAR_MAX_RISK}" in routing
    assert "10,000" in next(d for d in kb.policy if d.doc_id == "POL-STRUCTURING").text


def test_confidence_rules_in_the_policy_match_the_numbers_pinned_for_node4():
    pinned = set(PINNED["numeric_literals_by_function"]["_compute_confidence"])
    assert set(knowledge.CONFIDENCE_RULES.values()) <= pinned


def test_retrieval_puts_the_heaviest_codes_first_then_routing_policy(kb):
    report = make_report([
        finding("FL_BALANCE_BREAK", Severity.HIGH, 26), finding("FL_BALANCE_BREAK", Severity.HIGH, 26),
        finding("VF_CURRENCY_MIX", Severity.MEDIUM, 10, agent="Node2:VisualForensics"),
        finding("FL_PERIOD_GAP", Severity.MEDIUM, 12),
        finding("DQ_NO_TXNS", Severity.LOW, 3, agent="Node1:Ingestion")])
    docs = kb.retrieve(report)
    ids = [d.doc_id for d in docs]
    assert ids[:3] == ["CODE:FL_BALANCE_BREAK", "CODE:FL_PERIOD_GAP", "CODE:VF_CURRENCY_MIX"]
    assert "POL-ROUTING" in ids and len(ids) <= knowledge.MAX_DOCS
    assert len(set(ids)) == len(ids)


def test_retrieval_is_deterministic(kb):
    report = make_report([finding("FL_STRUCTURING"), finding("FL_SUDDEN_DEPOSIT", Severity.MEDIUM, 12)])
    assert kb.retrieve(report) == kb.retrieve(report)


def test_a_case_with_no_findings_gets_policy_only(kb):
    docs = kb.retrieve(make_report([]))
    assert docs and all(not d.doc_id.startswith("CODE:") for d in docs)


def test_low_confidence_cases_get_the_confidence_policy(kb):
    report = make_report([], dq=7)       # no transactions and many data-quality issues
    assert report.explanation.routing.primary == "LOW_CONFIDENCE"
    assert "POL-CONFIDENCE" in [d.doc_id for d in kb.retrieve(report)]


def test_policy_search_finds_the_relevant_passage(kb):
    top = kb.search_policy("deposits just under the reporting threshold structuring", 2, set())
    assert top[0].doc_id == "POL-STRUCTURING"
    assert kb.search_policy("zzzz qqqq", 3, set()) == []
    assert "POL-STRUCTURING" not in [d.doc_id for d in
                                     kb.search_policy("structuring threshold", 5, {"POL-STRUCTURING"})]


def test_version_is_a_hash_of_the_files_and_changes_with_them(tmp_path, kb):
    assert re.fullmatch(r"kb-[0-9a-f]{12}", kb.version) and KnowledgeBase.load().version == kb.version
    copy = tmp_path / "kb"
    shutil.copytree(knowledge.KNOWLEDGE_DIR, copy)
    assert KnowledgeBase.load(copy).version == kb.version
    (copy / "policy.md").write_text((copy / "policy.md").read_text() + "\n")
    assert KnowledgeBase.load(copy).version != kb.version


def test_known_codes_only_get_catalogue_docs(kb):
    assert kb.code_doc("NOT_A_REAL_CODE") is None
