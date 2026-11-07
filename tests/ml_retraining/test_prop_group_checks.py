"""Properties 2, 3 and 4: group-overlap checks, layout and seed separation, grouped folds.

Feature: ml-retraining, Properties 2, 3, 4
**Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 8.4**

All corpora are small in-memory ``LoadedCorpus`` objects; nothing is read or written.
"""
from __future__ import annotations

import random
from collections import defaultdict

from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.corpora import DocRef, LoadedCorpus
from fraud_detect.ml.retraining.group_checks import (
    awkward_overlap_gate,
    company_split_gate,
    grouped_folds,
    held_out_seed_gate,
    layout_gate,
)

SPLITS = ["train", "validation", "test", "test_unseen_layout"]
LAYOUTS = ["lay_a", "lay_b", "lay_c"]
HELD = "lay_held"
COMPANIES = [f"C{i:02d}" for i in range(8)]
UENS = [f"UEN{i:02d}" for i in range(8)]


def make_doc(seed, company, split, doc_id, layout="lay_a", uen=None):
    return DocRef(
        corpus_key=f"seed-{seed}", master_seed=seed, doc_id=doc_id, split=split,
        layout=layout, company_key=(seed, company), uen=uen or f"U-{seed}-{company}",
        is_tampered=False, genuine_resaved=False, tamper_types=(), tamper_method="", n_months=6)


def make_corpus(seed, docs):
    return LoadedCorpus(key=f"seed-{seed}", master_seed=seed, hold_out_layout=HELD,
                        docs=tuple(docs), manifest_counts={}, uen_check=None)


# ------------------------------------------------------------------ Property 2
@st.composite
def company_corpora(draw):
    seeds = draw(st.lists(st.integers(1, 4), min_size=1, max_size=3, unique=True))
    corpora, n = [], 0
    for s in seeds:
        docs = []
        for _ in range(draw(st.integers(0, 12))):
            docs.append(make_doc(s, draw(st.sampled_from(COMPANIES)),
                                 draw(st.sampled_from(SPLITS)), f"D{n:05d}"))
            n += 1
        corpora.append(make_corpus(s, docs))
    return corpora


@settings(max_examples=100, deadline=None)
@given(corpora=company_corpora())
def test_company_gate_reports_exactly_keys_in_more_than_one_split(corpora):
    splits = defaultdict(set)
    for c in corpora:
        for d in c.docs:
            splits[d.company_key].add(d.split)
    expected = {k for k, v in splits.items() if len(v) > 1}
    result = company_split_gate(corpora)
    assert result.passed == (not expected)
    assert not result.waivable
    assert len(result.offenders) == len(expected)
    for seed, cid in {k for k in splits}:
        listed = any(o.startswith(f"company {cid} (Master_Seed {seed}) ") for o in result.offenders)
        assert listed == ((seed, cid) in expected)


@settings(max_examples=100, deadline=None)
@given(cid=st.sampled_from(COMPANIES), s1=st.integers(1, 5), s2=st.integers(6, 9),
       sp1=st.sampled_from(SPLITS), sp2=st.sampled_from(SPLITS))
def test_same_raw_company_id_in_two_corpora_is_never_an_offender(cid, s1, s2, sp1, sp2):
    a = make_corpus(s1, [make_doc(s1, cid, sp1, "A1")])
    b = make_corpus(s2, [make_doc(s2, cid, sp2, "B1")])
    assert company_split_gate([a, b]).passed


@st.composite
def seed_scenarios(draw):
    train_seeds = draw(st.lists(st.integers(1, 6), min_size=0, max_size=4))
    held_seed = draw(st.integers(1, 6))
    # company ids and UENs drawn from small pools so overlaps happen often
    def docs_for(seed, tag, prefix):
        out = []
        for i in range(draw(st.integers(1, 5))):
            out.append(make_doc(seed, draw(st.sampled_from(COMPANIES)), "train",
                                f"{prefix}{tag}-{i}", uen=draw(st.sampled_from(UENS))))
        return out
    training = [make_corpus(s, docs_for(s, j, "T")) for j, s in enumerate(train_seeds)]
    held = make_corpus(held_seed, docs_for(held_seed, 99, "H"))
    return held, training


@settings(max_examples=150, deadline=None)
@given(sc=seed_scenarios())
def test_held_out_gate_reports_exactly_shared_seed_keys_and_uens(sc):
    held, training = sc
    seed_clashes = sum(1 for t in training if t.master_seed == held.master_seed)
    shared_keys = {d.company_key for d in held.docs} & {d.company_key for t in training for d in t.docs}
    shared_uens = {d.uen for d in held.docs} & {d.uen for t in training for d in t.docs}
    result = held_out_seed_gate(held, training)
    assert result.passed == (seed_clashes + len(shared_keys) + len(shared_uens) == 0)
    assert not result.waivable
    assert len(result.offenders) == seed_clashes + len(shared_keys) + len(shared_uens)
    offenders = list(result.offenders)
    for uen in UENS:
        assert (f"shared UEN: {uen}" in offenders) == (uen in shared_uens)
    for seed, cid in {(s, c) for s in range(1, 7) for c in COMPANIES}:
        marker = f"company {cid} (Master_Seed {seed})"
        assert any(marker in o for o in offenders) == ((seed, cid) in shared_keys)


@st.composite
def awkward_scenarios(draw):
    awk_seed = draw(st.integers(1, 4))
    corpora, all_ids, n = [], [], 0
    for s in draw(st.lists(st.integers(1, 4), min_size=0, max_size=3, unique=True)):
        docs = []
        for _ in range(draw(st.integers(0, 6))):
            did = f"D{n:04d}"
            n += 1
            all_ids.append(did)
            docs.append(make_doc(s, draw(st.sampled_from(COMPANIES)), "train", did))
        corpora.append(make_corpus(s, docs))
    doc_ids = set(draw(st.sets(st.sampled_from(all_ids or ["none"]), max_size=4))) | {"AW-never"}
    company_ids = set(draw(st.sets(st.sampled_from(COMPANIES + ["AW-co"]), max_size=3)))
    return corpora, doc_ids, company_ids, awk_seed


@settings(max_examples=150, deadline=None)
@given(sc=awkward_scenarios())
def test_awkward_gate_reports_exactly_id_company_and_seed_overlaps(sc):
    corpora, doc_ids, company_ids, awk_seed = sc
    expected = 0
    for c in corpora:
        expected += c.master_seed == awk_seed
        expected += sum(1 for d in c.docs if d.doc_id in doc_ids)
        expected += len({d.company_key[1] for d in c.docs} & company_ids)
    result = awkward_overlap_gate(corpora, doc_ids=doc_ids, company_ids=company_ids,
                                  master_seed=awk_seed)
    assert result.passed == (expected == 0)
    assert len(result.offenders) == expected
    assert not result.waivable


# ------------------------------------------------------------------ Property 3
@st.composite
def layout_tables(draw):
    docs = []
    for i in range(draw(st.integers(0, 20))):
        docs.append(make_doc(1, draw(st.sampled_from(COMPANIES)), draw(st.sampled_from(SPLITS)),
                             f"D{i:04d}", layout=draw(st.sampled_from(LAYOUTS + [HELD]))))
    return docs


@settings(max_examples=150, deadline=None)
@given(docs=layout_tables())
def test_layout_gate_passes_iff_unseen_split_is_exactly_held_out_layout(docs):
    violators = {d.doc_id for d in docs if (d.split == "test_unseen_layout") != (d.layout == HELD)}
    result = layout_gate([make_corpus(1, docs)], HELD)
    assert result.passed == (not violators)
    assert not result.waivable
    text = " | ".join(result.offenders)
    assert len(result.offenders) == len(violators)
    for d in docs:
        assert (d.doc_id in text) == (d.doc_id in violators)


@settings(max_examples=100, deadline=None)
@given(n_unseen=st.integers(0, 6), n_other=st.integers(0, 6))
def test_layout_gate_passes_on_a_clean_partition(n_unseen, n_other):
    docs = [make_doc(1, "C00", "test_unseen_layout", f"U{i}", layout=HELD) for i in range(n_unseen)]
    docs += [make_doc(1, "C01", "train", f"T{i}", layout="lay_a") for i in range(n_other)]
    assert layout_gate([make_corpus(1, docs)], HELD).passed


@settings(max_examples=150, deadline=None)
@given(train_seeds=st.lists(st.integers(1, 8), max_size=5), held_seed=st.integers(1, 8))
def test_seed_check_fails_iff_held_out_seed_is_in_the_list(train_seeds, held_seed):
    # companies and UENs are unique per corpus so only the seed can cause a failure
    training = [make_corpus(s, [make_doc(s, "CT", "train", f"T{j}", uen=f"UT{j}")])
                for j, s in enumerate(train_seeds)]
    held = make_corpus(held_seed, [make_doc(held_seed, "CH", "test", "H0", uen="UH")])
    result = held_out_seed_gate(held, training)
    assert result.passed == (held_seed not in train_seeds)
    if not result.passed:
        assert any(str(held_seed) in o for o in result.offenders)


# ------------------------------------------------------------------ Property 4
@st.composite
def fold_inputs(draw):
    keys = draw(st.lists(st.tuples(st.integers(1, 3), st.sampled_from(COMPANIES)), max_size=25))
    docs = [make_doc(s, c, "train", f"D{i:04d}") for i, (s, c) in enumerate(keys)]
    return docs, draw(st.integers(2, 6)), draw(st.integers(0, 50))


def _ids(folds):
    return [[d.doc_id for d in f] for f in folds]


@settings(max_examples=150, deadline=None)
@given(inp=fold_inputs())
def test_grouped_folds_keep_each_company_key_in_one_fold(inp):
    docs, k, seed = inp
    folds = grouped_folds(docs, k, seed)
    assert len(folds) == k
    assert sorted(d.doc_id for f in folds for d in f) == sorted(d.doc_id for d in docs)
    where = defaultdict(set)
    for i, f in enumerate(folds):
        for d in f:
            where[d.company_key].add(i)
    assert all(len(v) == 1 for v in where.values())


@settings(max_examples=100, deadline=None)
@given(inp=fold_inputs(), shuffle_seed=st.integers(0, 1000))
def test_grouped_folds_are_deterministic(inp, shuffle_seed):
    docs, k, seed = inp
    first = _ids(grouped_folds(docs, k, seed))
    assert first == _ids(grouped_folds(list(docs), k, seed))
    shuffled = list(docs)
    random.Random(shuffle_seed).shuffle(shuffled)
    assert first == _ids(grouped_folds(shuffled, k, seed))
