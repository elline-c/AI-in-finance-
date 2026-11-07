"""Tests for ``fraud_detect.ml.retraining.group_checks`` (task 4.1).

Requirements 1.2 to 1.7, 8.4. The checks run on small in-memory ``LoadedCorpus`` objects,
plus the Mini_Corpus for the happy path. They fail until task 4.3 is done.

Interface fixed by these tests (task 4.3 must provide it):

    company_split_gate(corpora) -> GateResult
    layout_gate(corpora, held_out_layout) -> GateResult
    held_out_seed_gate(held_out, training) -> GateResult
    awkward_overlap_gate(training, *, doc_ids, company_ids, master_seed) -> GateResult
    name_sharing_warnings(training, evaluation, names) -> list[str]
    grouped_folds(docs, n_folds, seed=0) -> list[list[DocRef]]

Each ``GateResult`` has ``gate_id`` (the check name), ``passed``, ``offenders`` (strings) and
``waivable`` (always False for these integrity gates).
"""
from __future__ import annotations

import dataclasses

import pytest

from fraud_detect.ml.retraining.corpora import DocRef, LoadedCorpus, load_corpus
from fraud_detect.ml.retraining.group_checks import (
    awkward_overlap_gate,
    company_split_gate,
    grouped_folds,
    held_out_seed_gate,
    layout_gate,
    name_sharing_warnings,
)

HELD_OUT = "two_col_ddmmyy_cr"


def doc(seed, company, split, layout="generator_v1", doc_id=None, uen=None, **kw):
    return DocRef(
        corpus_key=f"seed-{seed}", master_seed=seed,
        doc_id=doc_id or f"{layout}-{company}-{split}",
        split=split, layout=layout, company_key=(seed, company),
        uen=uen or f"U{seed % 1000:03d}{company}",
        is_tampered=kw.get("is_tampered", False), genuine_resaved=kw.get("genuine_resaved", False),
        tamper_types=kw.get("tamper_types", ()), tamper_method=kw.get("tamper_method", ""),
        n_months=kw.get("n_months", 6))


def corpus(seed, docs, hold_out=HELD_OUT):
    return LoadedCorpus(key=f"seed-{seed}", master_seed=seed, hold_out_layout=hold_out,
                        docs=tuple(docs), manifest_counts={"documents": len(docs)}, uen_check=None)


def joined(result):
    return " | ".join(result.offenders)


# ------------------------------------------------------------ company check
def test_company_split_gate_lists_exactly_the_straddling_keys():
    c = corpus(1, [
        doc(1, "CA", "train", doc_id="a1"), doc(1, "CA", "validation", doc_id="a2"),   # offender
        doc(1, "CB", "train", doc_id="b1"), doc(1, "CB", "train", doc_id="b2"),        # fine
        doc(1, "CC", "test", doc_id="c1"), doc(1, "CC", "validation", doc_id="c2"),
        doc(1, "CC", "train", doc_id="c3"),                                            # offender
    ])
    result = company_split_gate([c])
    assert result.gate_id and not result.passed and not result.waivable
    text = joined(result)
    assert len(result.offenders) == 2
    assert "CA" in text and "CC" in text and "CB" not in text
    assert "1" in text  # qualified by the Master_Seed


def test_same_raw_company_id_in_two_corpora_is_not_an_offender():
    a = corpus(1, [doc(1, "C1G0001", "train")])
    b = corpus(2, [doc(2, "C1G0001", "validation")])
    result = company_split_gate([a, b])
    assert result.passed and result.offenders == ()


def test_company_split_gate_passes_on_mini_corpus(mini_corpus):
    corpora = [load_corpus(mini_corpus.path(s)) for s in mini_corpus.seeds]
    assert company_split_gate(corpora).passed


# ------------------------------------------------------------- layout check
def test_layout_gate_passes_when_unseen_split_is_exactly_the_held_out_layout():
    c = corpus(1, [
        doc(1, "C1", "train", layout="generator_v1", doc_id="d1"),
        doc(1, "C2", "test_unseen_layout", layout=HELD_OUT, doc_id="d2"),
        doc(1, "C3", "test_unseen_layout", layout=HELD_OUT, doc_id="d3"),
    ])
    assert layout_gate([c], HELD_OUT).passed


def test_layout_gate_lists_every_violating_document():
    c = corpus(1, [
        doc(1, "C1", "train", layout=HELD_OUT, doc_id="held-out-in-train"),            # violation
        doc(1, "C2", "test_unseen_layout", layout="generator_v1", doc_id="wrong-unseen"),  # violation
        doc(1, "C3", "test_unseen_layout", layout=HELD_OUT, doc_id="fine-unseen"),
        doc(1, "C4", "validation", layout="two_col_slash", doc_id="fine-validation"),
    ])
    result = layout_gate([c], HELD_OUT)
    assert not result.passed and not result.waivable
    text = joined(result)
    assert "held-out-in-train" in text and "wrong-unseen" in text
    assert "fine-unseen" not in text and "fine-validation" not in text
    assert len(result.offenders) == 2


def test_layout_gate_passes_on_mini_corpus(mini_corpus):
    corpora = [load_corpus(mini_corpus.path(s)) for s in mini_corpus.seeds]
    assert layout_gate(corpora, mini_corpus.hold_out_layout).passed


def test_layout_gate_fails_for_the_wrong_held_out_layout_name(mini_corpus):
    corpora = [load_corpus(mini_corpus.path(s)) for s in mini_corpus.seeds]
    assert not layout_gate(corpora, "two_col_slash").passed


# ---------------------------------------------------------- held-out seed
def test_held_out_seed_gate_passes_for_a_separate_seed_and_distinct_uens():
    train = corpus(1, [doc(1, "C1", "train", uen="UEN-A")])
    valid = corpus(2, [doc(2, "C1", "validation", uen="UEN-B")])
    held = corpus(99, [doc(99, "C1", "test", uen="UEN-Z")])
    assert held_out_seed_gate(held, [train, valid]).passed


def test_held_out_seed_gate_fails_when_the_seed_is_reused():
    train = corpus(1, [doc(1, "C1", "train", uen="UEN-A")])
    valid = corpus(2, [doc(2, "C1", "validation", uen="UEN-B")])
    held = corpus(2, [doc(2, "C9", "test", uen="UEN-Z")])
    result = held_out_seed_gate(held, [train, valid])
    assert not result.passed and not result.waivable
    assert "2" in joined(result)


def test_held_out_seed_gate_lists_shared_uens_and_shared_qualified_keys():
    train = corpus(1, [doc(1, "C1", "train", uen="UEN-SHARED"), doc(1, "C2", "train", uen="UEN-OWN")])
    held = corpus(99, [doc(99, "C1", "test", uen="UEN-SHARED"), doc(99, "C5", "test", uen="UEN-FREE")])
    result = held_out_seed_gate(held, [train])
    assert not result.passed
    text = joined(result)
    assert "UEN-SHARED" in text
    assert "UEN-OWN" not in text and "UEN-FREE" not in text

    # a shared qualified company key (same seed and company id) is also listed
    clash = corpus(1, [doc(1, "C2", "test", uen="UEN-X")])
    result = held_out_seed_gate(clash, [train])
    assert not result.passed
    assert "C2" in joined(result)


def test_held_out_seed_gate_passes_on_mini_corpus_with_distinct_seeds(mini_corpus):
    train = load_corpus(mini_corpus.path(mini_corpus.seeds[0]))
    held = load_corpus(mini_corpus.path(mini_corpus.seeds[1]))
    assert held_out_seed_gate(held, [train]).passed


# --------------------------------------------------------- Awkward_Set overlap
def test_awkward_overlap_gate_passes_when_nothing_overlaps():
    train = corpus(1, [doc(1, "C1", "train", doc_id="d1")])
    result = awkward_overlap_gate([train], doc_ids={"AW-seasonal-1"},
                                  company_ids={"AW-seasonal-1"}, master_seed=20260701)
    assert result.passed


def test_awkward_overlap_gate_reports_id_company_and_seed():
    train = corpus(20260701, [
        doc(20260701, "AW-resave-1", "train", doc_id="AW-resave-1-g"),
        doc(20260701, "C7", "validation", doc_id="plain"),
    ])
    result = awkward_overlap_gate([train], doc_ids={"AW-resave-1-g", "never-present"},
                                  company_ids={"AW-resave-1"}, master_seed=20260701)
    assert not result.passed and not result.waivable
    text = joined(result)
    assert "AW-resave-1-g" in text      # document id
    assert "AW-resave-1" in text        # company id
    assert "20260701" in text           # Master_Seed
    assert "never-present" not in text and "plain" not in text


# ------------------------------------------------------------ grouped folds
def _many_docs():
    docs = []
    for ci in range(12):
        for k in range(1 + ci % 3):
            docs.append(doc(1, f"C{ci}", "train", doc_id=f"D{ci}-{k}"))
    return docs


def test_grouped_folds_keep_a_company_in_one_fold_and_cover_every_doc_once():
    docs = _many_docs()
    folds = grouped_folds(docs, n_folds=4, seed=0)
    assert len(folds) == 4
    flat = [d.doc_id for fold in folds for d in fold]
    assert sorted(flat) == sorted(d.doc_id for d in docs)
    where = {}
    for i, fold in enumerate(folds):
        for d in fold:
            where.setdefault(d.company_key, set()).add(i)
    assert all(len(v) == 1 for v in where.values())


def test_grouped_folds_are_deterministic_and_seed_sensitive():
    docs = _many_docs()
    first = [[d.doc_id for d in f] for f in grouped_folds(docs, n_folds=3, seed=7)]
    again = [[d.doc_id for d in f] for f in grouped_folds(list(docs), n_folds=3, seed=7)]
    assert first == again
    assert first == [[d.doc_id for d in f] for f in grouped_folds(list(reversed(docs)), n_folds=3, seed=7)]


def test_grouped_folds_same_company_id_in_two_corpora_may_split_across_folds():
    docs = [doc(1, "C1", "train", doc_id="a"), doc(2, "C1", "train", doc_id="b")]
    folds = grouped_folds(docs, n_folds=2, seed=0)
    assert sorted(len(f) for f in folds) == [1, 1]   # two qualified keys, two groups


# ------------------------------------------------------------ name sharing
def test_shared_company_name_is_a_warning_not_a_failure():
    train = corpus(1, [doc(1, "C1", "train"), doc(1, "C2", "train")])
    evaluation = corpus(99, [doc(99, "C1", "test"), doc(99, "C2", "test")])
    names = {(1, "C1"): "IRONWOOD SYSTEMS PTE. LTD.", (1, "C2"): "QUARTZEL WORKS PTE. LTD.",
             (99, "C1"): "IRONWOOD SYSTEMS PTE. LTD.", (99, "C2"): "OTHER NAME PTE. LTD."}
    warnings = name_sharing_warnings([train], [evaluation], names)
    assert isinstance(warnings, list) and len(warnings) == 1
    assert "IRONWOOD SYSTEMS PTE. LTD." in warnings[0]
    # being a warning, the gates that guard training are unaffected
    assert company_split_gate([train, evaluation]).passed


def test_no_warning_when_names_differ():
    train = corpus(1, [doc(1, "C1", "train")])
    evaluation = corpus(99, [doc(99, "C1", "test")])
    names = {(1, "C1"): "A PTE. LTD.", (99, "C1"): "B PTE. LTD."}
    assert name_sharing_warnings([train], [evaluation], names) == []


# ----------------------------------------------------------- failure shape
def test_failures_carry_check_name_and_write_no_model_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = corpus(1, [doc(1, "CA", "train"), doc(1, "CA", "test")])
    result = company_split_gate([c])
    assert not result.passed and result.gate_id
    assert dataclasses.is_dataclass(result)
    assert list(tmp_path.iterdir()) == []   # nothing written (no model, no artefact)
