"""Property tests for the corpus builder.

Feature: synthetic-corpus-generator, Property 8: Same seed, same corpus (whole build)
Feature: synthetic-corpus-generator, Property 17: Counts add up
Feature: synthetic-corpus-generator, Property 19: The companies file matches the ledger
**Validates: Requirements 10.2, 10.3, 10.5, 11.6, 12.10, 12.11, 13.1**

Planning is cheap, so most properties use ``plan()``. Real builds are slow, so the
build properties use very small corpora (one genuine document per layout, one month)
and few examples.
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import LAYOUTS, TAMPER_TYPES, BuildConfig

from tests.corpus.helpers import words_of

_seed = st.integers(min_value=0, max_value=2**31 - 1)
_count = st.integers(min_value=0, max_value=3)


@st.composite
def _mix(draw):
    """None (even mix) or a non-empty weighted subset of the tamper types."""
    if draw(st.booleans()):
        return None
    names = draw(st.lists(st.sampled_from(TAMPER_TYPES), min_size=1, max_size=4, unique=True))
    weights = [draw(st.integers(1, 5)) for _ in names]
    return tuple((n, float(w)) for n, w in zip(names, weights))


def _cfg(seed, genuine, tampered, mix=None, out="unused", **kw) -> BuildConfig:
    base = dict(out_dir=Path(out), master_seed=seed, per_layout_genuine=genuine,
                per_layout_tampered=tampered, tamper_mix=mix,
                months_min=1, months_max=1, command_line="prop-test")
    base.update(kw)
    return BuildConfig(**base)


# ---------------------------------------------------------------- Property 17 (plan)
@settings(max_examples=60, deadline=None)
@given(seed=_seed, genuine=_count, tampered=_count, mix=_mix())
def test_plan_counts_and_mix(seed, genuine, tampered, mix):
    plan = CorpusBuilder(_cfg(seed, genuine, tampered, mix)).plan()
    assert len(plan) == (genuine + tampered) * len(LAYOUTS)
    assert len({p.doc_id for p in plan}) == len(plan)
    for layout in LAYOUTS:
        mine = [p for p in plan if p.layout == layout]
        assert sum(p.tamper_type is None for p in mine) == genuine
        assert sum(p.tamper_type is not None for p in mine) == tampered

    types = Counter(p.tamper_type for p in plan if p.tamper_type is not None)
    n = tampered * len(LAYOUTS)
    assert sum(types.values()) == n
    if n:
        if mix is None:
            weights = {t: 1.0 for t in TAMPER_TYPES}
        else:
            weights = {t: w for t, w in mix}
        total = sum(weights.values())
        assert set(types) <= set(weights)
        for t, w in weights.items():
            # largest-remainder quotas: within 1 of the exact share
            assert abs(types.get(t, 0) - n * w / total) < 1.0


# ---------------------------------------------------------------- Property 8 (plan)
@settings(max_examples=40, deadline=None)
@given(seed=_seed, genuine=st.integers(1, 3), tampered=st.integers(0, 2), mix=_mix())
def test_plan_is_repeatable_and_a_document_does_not_depend_on_others(seed, genuine, tampered, mix):
    cfg = _cfg(seed, genuine, tampered, mix)
    plan = CorpusBuilder(cfg).plan()
    assert plan == CorpusBuilder(cfg).plan()

    # Genuine documents are independent of how many other documents are planned.
    bigger = CorpusBuilder(_cfg(seed, genuine + 1, tampered + 1, mix)).plan()
    by_id = {p.doc_id: p for p in bigger}
    for p in plan:
        if p.tamper_type is None:
            assert by_id[p.doc_id] == p


@settings(max_examples=25, deadline=None)
@given(seed=_seed)
def test_plan_months_are_drawn_from_six_to_twelve_when_not_given(seed):
    cfg = BuildConfig(out_dir=Path("unused"), master_seed=seed,
                      per_layout_genuine=2, per_layout_tampered=0)
    assert all(6 <= p.n_months <= 12 for p in CorpusBuilder(cfg).plan())


# ---------------------------------------------------------------- real builds (slow)
def _build(seed, root: Path, name: str):
    out = root / name
    summary = CorpusBuilder(_cfg(seed, 1, 0, out=out)).build()
    return out, summary


@settings(max_examples=2, deadline=None)
@given(seed=_seed)
def test_same_seed_same_corpus_and_consistent_counts_and_companies(seed):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, summary = _build(seed, root, "a")
        b, _ = _build(seed, root, "b")

        # Property 8: byte-equal tables and manifests, equal text layers.
        names = ["labels.csv", "rows.csv", "companies.json"]
        names += [f"manifests/{p.name}" for p in sorted((a / "manifests").glob("*.json"))]
        for rel in names:
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
        for pdf in sorted((a / "pdfs").glob("*.pdf")):
            assert words_of((a / "pdfs" / pdf.name).read_bytes()) == \
                words_of((b / "pdfs" / pdf.name).read_bytes())

        # Property 17: manifest counts add up.
        assert summary.failed == []
        counts = json.loads((a / "corpus_manifest.json").read_text(encoding="utf-8"))["counts"]
        assert counts["documents"] == len(LAYOUTS)
        assert sum(counts["per_layout"].values()) == counts["documents"]
        assert sum(counts["per_split"].values()) == counts["documents"]
        assert counts["per_layout"] == {lay: 1 for lay in LAYOUTS}
        for split, per_type in counts["per_tamper_type"].items():
            assert sum(per_type.values()) == 0   # genuine only

        # Property 19: baseline equals the genuine ledger's monthly averages.
        data = json.loads((a / "companies.json").read_text(encoding="utf-8"))
        companies = {c["uen"]: c for c in (data["companies"] if isinstance(data, dict) else data)}
        assert len(companies) == len(LAYOUTS)
        for mf in sorted((a / "manifests").glob("*.json")):
            m = json.loads(mf.read_text(encoding="utf-8"))
            rec = companies[m["company"]["uen"]]
            n = len(m["months"])
            amounts = [r["amount"] for r in m["rows"]]
            assert rec["baseline"]["avg_monthly_credit"] == pytest.approx(
                sum(x for x in amounts if x > 0) / n, abs=0.01)
            assert rec["baseline"]["avg_monthly_debit"] == pytest.approx(
                -sum(x for x in amounts if x < 0) / n, abs=0.01)


def test_different_master_seed_changes_company_names(tmp_path):
    a, _ = _build(11, tmp_path, "a")
    b, _ = _build(12, tmp_path, "b")

    def names(out):
        data = json.loads((out / "companies.json").read_text(encoding="utf-8"))
        return {c["registered_name"] for c in (data["companies"] if isinstance(data, dict) else data)}

    assert len(LAYOUTS) >= 6
    assert names(a) != names(b)
