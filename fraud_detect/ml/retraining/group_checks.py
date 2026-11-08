"""Group-aware split checks (Task 4.3; Requirements 1.2 to 1.7, 8.4, 17.11).

Every gate here is an integrity gate (not waivable). Each returns a ``GateResult`` whose
``offenders`` name every violation. Nothing is written to disk. Standard library only,
plus the existing ``corpus.uen_check.UenChecker`` for UEN sharing.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from fraud_detect.corpus.uen_check import UenChecker
from fraud_detect.ml.retraining.gates import GateResult

UNSEEN_SPLIT = "test_unseen_layout"


def _result(gate_id: str, requirement: str, offenders: Iterable[str]) -> GateResult:
    offs = tuple(offenders)
    return GateResult(gate_id, requirement, not offs, offs, False, None)


def _fmt_key(key) -> str:
    return f"company {key[1]} (Master_Seed {key[0]})"


def company_split_gate(corpora: Sequence) -> GateResult:
    """A qualified company key (Master_Seed, company_id) must sit in one Split only."""
    splits: dict = defaultdict(set)
    for c in corpora:
        for d in c.docs:
            splits[d.company_key].add(d.split)
    offenders = [f"{_fmt_key(k)} appears in Splits: {', '.join(sorted(s))}"
                 for k, s in sorted(splits.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
                 if len(s) > 1]
    return _result("company_split", "1.2", offenders)


def layout_gate(corpora: Sequence, held_out_layout: str) -> GateResult:
    """The held-out layout is in test_unseen_layout and nowhere else."""
    offenders = []
    for c in corpora:
        for d in c.docs:
            in_unseen = d.split == UNSEEN_SPLIT
            is_held = d.layout == held_out_layout
            if in_unseen != is_held:
                offenders.append(f"{c.key}: document {d.doc_id} has layout {d.layout!r} "
                                 f"in Split {d.split!r} (held-out layout is {held_out_layout!r})")
    return _result("layout_separation", "1.3", offenders)


def held_out_seed_gate(held_out, training: Sequence) -> GateResult:
    """The held-out corpus has its own seed, its own companies and its own UENs."""
    offenders = []
    for t in training:
        if t.master_seed == held_out.master_seed:
            offenders.append(f"held-out Master_Seed {held_out.master_seed} is also used by "
                             f"corpus {t.key}")
    train_keys = {d.company_key for t in training for d in t.docs}
    for k in sorted({d.company_key for d in held_out.docs} & train_keys,
                    key=lambda k: (str(k[0]), str(k[1]))):
        offenders.append(f"shared qualified key: {_fmt_key(k)}")
    train_uens = {d.uen for t in training for d in t.docs if d.uen}
    checker = UenChecker(train_uens)
    seen = set()
    for d in held_out.docs:
        if d.uen and d.uen not in seen:
            seen.add(d.uen)
            if d.uen.strip().upper() in checker.snapshot:
                offenders.append(f"shared UEN: {d.uen}")
    return _result("held_out_seed", "1.4, 1.5", offenders)


def uen_manifest_gate(corpus) -> GateResult:
    """The corpus manifest must record a UEN collision check covering every company.

    Reads only ``corpus.uen_check`` and the distinct company keys. Pure: no file, no network.
    """
    check = corpus.uen_check
    if not isinstance(check, dict):
        return _result("uen_manifest", "17.11, 17.12",
                       [f"{corpus.key}: manifest has no uen_check object"])
    n = check.get("uens_checked")
    if isinstance(n, bool) or not isinstance(n, int):
        return _result("uen_manifest", "17.11, 17.12",
                       [f"{corpus.key}: uen_check.uens_checked is missing or not an integer"])
    companies = len({d.company_key for d in corpus.docs})
    if n < companies:
        return _result("uen_manifest", "17.11, 17.12",
                       [f"{corpus.key}: uens_checked is {n} but the corpus has "
                        f"{companies} companies"])
    return _result("uen_manifest", "17.11, 17.12", [])


def awkward_overlap_gate(training: Sequence, *, doc_ids, company_ids, master_seed) -> GateResult:
    """No Awkward_Set document id, company id or seed in a training or validation corpus."""
    doc_ids, company_ids = set(doc_ids), set(company_ids)
    offenders = []
    for c in training:
        if c.master_seed == master_seed:
            offenders.append(f"corpus {c.key} uses the Awkward_Set Master_Seed {master_seed}")
        for d in c.docs:
            if d.doc_id in doc_ids:
                offenders.append(f"{c.key}: Awkward_Set document id {d.doc_id}")
        for cid in sorted({d.company_key[1] for d in c.docs} & company_ids):
            offenders.append(f"{c.key}: Awkward_Set company id {cid}")
    return _result("awkward_overlap", "8.4", offenders)


def name_sharing_warnings(training: Sequence, evaluation: Sequence,
                          names: Mapping) -> list:
    """Warnings (never failures) for a registered name used in training and evaluation."""
    def collect(corpora):
        out = defaultdict(set)
        for c in corpora:
            for d in c.docs:
                n = names.get(d.company_key)
                if n:
                    out[n].add(d.company_key)
        return out
    train, ev = collect(training), collect(evaluation)
    return [f"warning: company name {n!r} is used in both training and evaluation corpora"
            for n in sorted(set(train) & set(ev))]


def grouped_folds(docs: Sequence, n_folds: int, seed: int = 0) -> list:
    """Split documents into folds keeping each qualified company in one fold.

    Deterministic and independent of input order.
    """
    groups: dict = defaultdict(list)
    for d in docs:
        groups[d.company_key].append(d)
    keys = sorted(groups, key=lambda k: (str(k[0]), str(k[1])))
    random.Random(seed).shuffle(keys)
    order = sorted(range(len(keys)), key=lambda i: (-len(groups[keys[i]]), i))
    folds = [[] for _ in range(n_folds)]
    for i in order:
        target = min(range(n_folds), key=lambda f: (len(folds[f]), f))
        folds[target].extend(groups[keys[i]])
    return [sorted(f, key=lambda d: (d.corpus_key, d.doc_id)) for f in folds]
