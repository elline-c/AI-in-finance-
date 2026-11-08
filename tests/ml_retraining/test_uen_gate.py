"""Tests for ``group_checks.uen_manifest_gate`` (task 5.1; Requirements 17.11, 17.12, 1.5).

They fail until task 5.2 adds the gate.

Interface fixed by these tests (task 5.2 must provide it):
    uen_manifest_gate(corpus) -> GateResult
        * ``corpus`` is a ``LoadedCorpus``; the gate reads only ``corpus.uen_check`` (a dict
          or ``None``) and the number of distinct ``company_key`` values in ``corpus.docs``.
        * ``gate_id == "uen_manifest"``; ``waivable is False`` (integrity gate);
          ``requirement`` mentions 17.11.
        * passes iff ``uen_check`` is a dict with an integer ``uens_checked`` that is
          >= the company count.
        * a missing/None/non-dict ``uen_check`` gives one offender naming ``uen_check`` and
          the corpus key; a low count gives an offender naming ``uens_checked``, the count
          found and the company count, and the corpus key.
        * pure: no file, no network, no ``live=True`` lookup.
A failed gate maps to ``EXIT_INTEGRITY`` (2) through ``run_gates``; it cannot be waived.
"""
from __future__ import annotations

import urllib.request

import pytest

from fraud_detect.corpus.uen_check import UenChecker
from fraud_detect.ml.retraining import group_checks
from fraud_detect.ml.retraining.corpora import DocRef, LoadedCorpus, load_corpus
from fraud_detect.ml.retraining.decisions import DecisionLog
from fraud_detect.ml.retraining.gates import EXIT_INTEGRITY, Gate, GateResult, run_gates
from fraud_detect.ml.retraining.group_checks import held_out_seed_gate, uen_manifest_gate

GOOD_CHECK = {"source": "in-repo registry only", "snapshot_file": None, "snapshot_sha256": None,
              "snapshot_uens": 0, "uens_checked": 3, "redraws": 0,
              "real_registry_checked": False}


def doc(seed, company, uen, n=0):
    return DocRef(
        corpus_key=f"seed-{seed}", master_seed=seed, doc_id=f"d-{company}-{n}", split="train",
        layout="generator_v1", company_key=(seed, company), uen=uen, is_tampered=False,
        genuine_resaved=False, tamper_types=(), tamper_method="", n_months=6)


def corpus(seed=1, companies=3, uen_check=GOOD_CHECK, docs_per_company=2):
    docs = [doc(seed, f"C{i}", f"2026{seed % 100:02d}{i:04d}Z", n)
            for i in range(companies) for n in range(docs_per_company)]
    return LoadedCorpus(key=f"seed-{seed}", master_seed=seed, hold_out_layout=None,
                        docs=tuple(docs), manifest_counts={"documents": len(docs)},
                        uen_check=uen_check)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def joined(result):
    return " | ".join(result.offenders)


# ------------------------------------------------------------------ the gate
def test_gate_result_shape_and_integrity_kind():
    res = uen_manifest_gate(corpus())
    assert isinstance(res, GateResult)
    assert res.gate_id == "uen_manifest"
    assert res.waivable is False
    assert "17.11" in res.requirement


def test_good_manifest_passes():
    res = uen_manifest_gate(corpus(companies=3, uen_check={**GOOD_CHECK, "uens_checked": 3}))
    assert res.passed and res.offenders == ()


def test_more_checks_than_companies_passes():
    # redraws make uens_checked larger than the company count
    assert uen_manifest_gate(corpus(companies=3, uen_check={**GOOD_CHECK, "uens_checked": 9})).passed


def test_company_count_is_distinct_company_keys_not_documents():
    # 3 companies x 4 documents = 12 documents; 3 checks are enough
    c = corpus(companies=3, docs_per_company=4, uen_check={**GOOD_CHECK, "uens_checked": 3})
    assert uen_manifest_gate(c).passed


def test_missing_uen_check_is_refused_and_named():
    res = uen_manifest_gate(corpus(uen_check=None))
    assert not res.passed
    assert len(res.offenders) == 1
    assert "uen_check" in joined(res)
    assert "seed-1" in joined(res)


@pytest.mark.parametrize("bad", ["yes", [], 3, ""])
def test_non_object_uen_check_is_refused(bad):
    res = uen_manifest_gate(corpus(uen_check=bad))
    assert not res.passed and "uen_check" in joined(res)


@pytest.mark.parametrize("bad", [{}, {"source": "snapshot"}, {"uens_checked": None},
                                 {"uens_checked": "3"}])
def test_object_without_usable_count_is_refused(bad):
    res = uen_manifest_gate(corpus(companies=3, uen_check=bad))
    assert not res.passed and "uens_checked" in joined(res)


def test_count_lower_than_company_count_is_refused():
    res = uen_manifest_gate(corpus(companies=4, uen_check={**GOOD_CHECK, "uens_checked": 3}))
    assert not res.passed
    text = joined(res)
    assert "uens_checked" in text and "3" in text and "4" in text and "seed-1" in text


def test_zero_checks_is_refused():
    res = uen_manifest_gate(corpus(companies=2, uen_check={**GOOD_CHECK, "uens_checked": 0}))
    assert not res.passed


def test_boundary_one_below_and_equal():
    below = uen_manifest_gate(corpus(companies=5, uen_check={**GOOD_CHECK, "uens_checked": 4}))
    equal = uen_manifest_gate(corpus(companies=5, uen_check={**GOOD_CHECK, "uens_checked": 5}))
    assert not below.passed and equal.passed


def test_gate_does_not_modify_the_corpus():
    c = corpus(uen_check=dict(GOOD_CHECK))
    before = (c.docs, dict(c.uen_check))
    uen_manifest_gate(c)
    assert (c.docs, c.uen_check) == before


# ------------------------------------------------- refusal: exit 2, no Candidate
def test_refusal_is_exit_2_cannot_be_waived_and_writes_nothing(tmp_path):
    log = DecisionLog(tmp_path / "decisions.json", is_ignored=lambda p: True)
    log.append("waive-gate", target="uen_manifest", decision="waive", reason="r", author="a")
    models = tmp_path / "models"
    models.mkdir()
    gate = Gate("uen_manifest", "17.11, 17.12", False,
                lambda ctx: uen_manifest_gate(ctx).offenders)
    (res,) = run_gates([gate], corpus(uen_check=None), log)
    assert not res.passed and res.waived_by is None
    assert EXIT_INTEGRITY == 2
    assert list(models.iterdir()) == []                      # zero Candidate_Model files
    assert not [p for p in tmp_path.rglob("*") if p.suffix in {".joblib", ".pkl", ".onnx"}]


# --------------------------------------------- shared UENs and the real checker
def test_uen_shared_by_two_corpora_is_taken_through_uen_checker():
    a, b = corpus(seed=1), corpus(seed=2)
    shared = "202699999Z"
    a = LoadedCorpus(**{**a.__dict__, "docs": a.docs + (doc(1, "CX", shared),)})
    b = LoadedCorpus(**{**b.__dict__, "docs": b.docs + (doc(2, "CY", shared),)})
    checker = UenChecker({d.uen for d in a.docs})
    assert checker.is_taken(shared) is True
    assert checker.is_taken(shared.lower()) is True          # normalised
    res = held_out_seed_gate(b, [a])
    assert not res.passed and shared in joined(res)


def test_group_checks_reuses_uen_checker_and_offers_no_live_option():
    assert group_checks.UenChecker is UenChecker
    import inspect
    src = inspect.getsource(group_checks)
    assert "live=True" not in src
    assert "check-live-uen" not in src and "check_live_uen" not in src


def test_cli_offers_no_check_live_uen_option():
    from fraud_detect.ml.retraining import cli
    import inspect
    assert "check-live-uen" not in inspect.getsource(cli)


# ----------------------------------------------------------- Mini_Corpus happy path
def test_mini_corpus_manifest_passes(mini_corpus):
    for seed, path in mini_corpus.paths.items():
        res = uen_manifest_gate(load_corpus(path))
        assert res.passed, (seed, res.offenders)
