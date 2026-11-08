"""Property 40: The UEN manifest gate (task 5.3).

**Validates: Requirements 17.11, 17.12**

For any corpus manifest the gate refuses if and only if there is no ``uen_check`` object or
its ``uens_checked`` is lower than the number of companies; and a UEN shared by two corpora
is reported through ``UenChecker`` as taken.
"""
from __future__ import annotations

import urllib.request

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.uen_check import UenChecker
from fraud_detect.ml.retraining.corpora import DocRef, LoadedCorpus
from fraud_detect.ml.retraining.decisions import DecisionLog
from fraud_detect.ml.retraining.gates import Gate, run_gates
from fraud_detect.ml.retraining.group_checks import held_out_seed_gate, uen_manifest_gate


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def _doc(seed, company, uen, n):
    return DocRef(
        corpus_key=f"seed-{seed}", master_seed=seed, doc_id=f"d-{company}-{n}", split="train",
        layout="generator_v1", company_key=(seed, company), uen=uen, is_tampered=False,
        genuine_resaved=False, tamper_types=(), tamper_method="", n_months=6)


def _corpus(seed, companies, docs_per_company, uen_check):
    docs = [_doc(seed, f"C{i}", f"2026{seed % 100:02d}{i:04d}Z", n)
            for i in range(companies) for n in range(docs_per_company)]
    return LoadedCorpus(key=f"seed-{seed}", master_seed=seed, hold_out_layout=None,
                        docs=tuple(docs), manifest_counts={"documents": len(docs)},
                        uen_check=uen_check)


# Manifests: absent, non-object, object with an unusable count, or a usable integer count.
_other_values = st.one_of(st.none(), st.text(max_size=5), st.booleans(), st.floats(),
                          st.lists(st.integers(), max_size=2))
_bad_checks = st.one_of(
    st.none(), st.text(max_size=5), st.integers(), st.lists(st.integers(), max_size=2),
    st.fixed_dictionaries({}, optional={"source": st.text(max_size=5)}),
    st.fixed_dictionaries({"uens_checked": _other_values}),
)


@settings(max_examples=100, deadline=None)
@given(companies=st.integers(1, 8), docs=st.integers(1, 4),
       count=st.integers(-3, 20), extra=st.dictionaries(
           st.sampled_from(["source", "snapshot_file", "redraws", "snapshot_uens"]),
           st.one_of(st.none(), st.integers(0, 5), st.text(max_size=5)), max_size=4))
def test_integer_count_passes_iff_not_below_company_count(companies, docs, count, extra):
    check = {**extra, "uens_checked": count}
    res = uen_manifest_gate(_corpus(1, companies, docs, check))
    assert res.gate_id == "uen_manifest"
    assert res.passed == (count >= companies)
    assert (res.offenders == ()) == res.passed
    if not res.passed:
        assert any("seed-1" in o for o in res.offenders)
        assert any("uens_checked" in o for o in res.offenders)


@settings(max_examples=100, deadline=None)
@given(companies=st.integers(1, 6), docs=st.integers(1, 3), check=_bad_checks)
def test_missing_or_unusable_manifest_is_refused_and_named(companies, docs, check):
    res = uen_manifest_gate(_corpus(1, companies, docs, check))
    assert not res.passed and res.offenders
    text = " | ".join(res.offenders)
    assert "seed-1" in text
    assert "uen_check" in text


@settings(max_examples=50, deadline=None)
@given(companies=st.integers(1, 6), docs=st.integers(1, 4), extra=st.integers(0, 5))
def test_company_count_is_distinct_companies_not_documents(companies, docs, extra):
    c = _corpus(1, companies, docs, {"uens_checked": companies + extra})
    assert uen_manifest_gate(c).passed
    if companies > 1:
        c = _corpus(1, companies, docs, {"uens_checked": companies - 1})
        assert not uen_manifest_gate(c).passed


@settings(max_examples=30, deadline=None)
@given(companies=st.integers(1, 5), count=st.integers(0, 8),
       has_check=st.booleans())
def test_refusal_is_integrity_failure_and_not_waivable(tmp_path_factory, companies, count,
                                                       has_check):
    tmp = tmp_path_factory.mktemp("gate")
    log = DecisionLog(tmp / "decisions.json", is_ignored=lambda p: True)
    log.append("waive-gate", target="uen_manifest", decision="waive", reason="r", author="a")
    check = {"uens_checked": count} if has_check else None
    c = _corpus(1, companies, 2, check)
    gate = Gate("uen_manifest", "17.11, 17.12", False,
                lambda ctx: uen_manifest_gate(ctx).offenders)
    (res,) = run_gates([gate], c, log)
    should_pass = has_check and count >= companies
    assert res.passed == should_pass
    if not should_pass:
        assert res.waived_by is None
    assert not [p for p in tmp.rglob("*") if p.suffix in {".joblib", ".pkl", ".onnx"}]


@settings(max_examples=50, deadline=None)
@given(companies=st.integers(1, 5), count=st.integers(-2, 8), has_check=st.booleans())
def test_gate_does_not_change_the_corpus(companies, count, has_check):
    check = {"uens_checked": count} if has_check else None
    c = _corpus(1, companies, 2, check)
    before = (c.docs, None if check is None else dict(c.uen_check))
    uen_manifest_gate(c)
    assert (c.docs, None if check is None else dict(c.uen_check)) == before


@settings(max_examples=50, deadline=None)
@given(digits=st.integers(0, 9999), lower=st.booleans(), pad=st.sampled_from(["", " "]))
def test_uen_shared_by_two_corpora_is_taken_through_checker(digits, lower, pad):
    shared = f"2027{digits:04d}Z"
    a = _corpus(1, 2, 1, {"uens_checked": 2})
    b = _corpus(2, 2, 1, {"uens_checked": 2})
    a = LoadedCorpus(**{**a.__dict__, "docs": a.docs + (_doc(1, "CX", shared, 0),)})
    b = LoadedCorpus(**{**b.__dict__, "docs": b.docs + (_doc(2, "CY", shared, 0),)})
    checker = UenChecker({d.uen for d in a.docs})
    probe = pad + (shared.lower() if lower else shared) + pad
    assert checker.is_taken(probe) is True
    res = held_out_seed_gate(b, [a])
    assert not res.passed
    assert any(shared in o for o in res.offenders)
