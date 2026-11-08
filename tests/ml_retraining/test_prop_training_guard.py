"""Task 14.3: Property 24, Real_Style paths never reach a training or validation list.

For any training or validation list and any spelling of a Real_Style path (dot segments,
redundant separators, trailing separators, relative form), the guard reports the path if and
only if its normalised absolute form equals a Real_Style path; on detection the run aborts
before anything is fitted and existing model and report files are unchanged. With generator
corpus only training, the lists hold zero Forgery_Dataset documents.

**Validates: Requirements 7.3, 7.8, 7.9**
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.ml.retraining.sources import (
    RealStyleInTraining,
    assert_not_in_training,
    build_training_lists,
    normalise_path,
)

_SEG = st.text(alphabet="abcdefghij0123456789_-", min_size=1, max_size=5)
_BASE = st.lists(_SEG, min_size=1, max_size=4).map(
    lambda parts: "/" + "/".join(["gt_root"] + parts) + ".pdf")


def _respell(path: str, style: int, junk: str) -> str:
    """Return another spelling of the same absolute path."""
    head, tail = os.path.split(path)
    if style == 0:
        return f"{head}/./{tail}"
    if style == 1:
        return f"{head}//{tail}"
    if style == 2:
        return f"{head}/{junk}/../{tail}"
    if style == 3:
        return path + "/"
    if style == 4:
        return os.path.relpath(path, os.getcwd())
    return path


def _different(path: str, kind: int, junk: str) -> str:
    """A spelling that normalises to a different path."""
    head, tail = os.path.split(path)
    if kind == 0:
        return f"{head}/{junk}_x/{tail}"
    if kind == 1:
        return f"{head}/{tail}.bak"
    return f"{head}/{junk}/{tail}"


@settings(max_examples=100, deadline=None)
@given(
    protected=st.lists(_BASE, min_size=1, max_size=4, unique=True),
    picks=st.lists(st.tuples(st.integers(0, 3), st.integers(0, 5), _SEG), max_size=6),
    decoys=st.lists(st.tuples(st.integers(0, 3), st.integers(0, 2), _SEG), max_size=6),
)
def test_guard_detects_iff_normalised_form_matches(protected, picks, decoys):
    protected_norm = {normalise_path(p) for p in protected}
    paths, expected = [], set()
    for idx, style, junk in picks:
        base = protected[idx % len(protected)]
        paths.append(_respell(base, style, junk))
        expected.add(normalise_path(base))
    for idx, kind, junk in decoys:
        d = _different(protected[idx % len(protected)], kind, junk)
        paths.append(d)
        if normalise_path(d) in protected_norm:
            expected.add(normalise_path(d))

    if expected:
        with pytest.raises(RealStyleInTraining) as exc:
            assert_not_in_training(paths, protected=protected)
        reported = set(exc.value.offenders)
        assert reported == expected
    else:
        assert_not_in_training(paths, protected=protected)


@settings(max_examples=100, deadline=None)
@given(
    protected=st.lists(_BASE, min_size=1, max_size=3, unique=True),
    style=st.integers(0, 5),
    junk=_SEG,
)
def test_every_spelling_normalises_to_same_absolute_path(protected, style, junk):
    for p in protected:
        n = normalise_path(_respell(p, style, junk))
        assert n.is_absolute()
        assert n == normalise_path(p)


@settings(max_examples=30, deadline=None)
@given(style=st.integers(0, 5), junk=_SEG, base=_BASE)
def test_abort_leaves_existing_files_unchanged(style, junk, base):
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model.joblib"
        report = Path(tmp) / "report.json"
        model.write_bytes(b"model-bytes")
        report.write_text("{}")
        before = (model.read_bytes(), report.read_bytes(), sorted(os.listdir(tmp)))
        fitted = []
        with pytest.raises(RealStyleInTraining):
            assert_not_in_training([_respell(base, style, junk)], protected=[base])
            fitted.append(1)  # stands for the fit; must not be reached
        assert fitted == []
        assert (model.read_bytes(), report.read_bytes(), sorted(os.listdir(tmp))) == before


@settings(max_examples=20, deadline=None)
@given(n=st.integers(1, 6), seed=st.integers(0, 10_000))
def test_generator_corpus_lists_have_no_forgery_documents(n, seed):
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        forgery = Path(tmp) / "forgery"
        corpus.mkdir()
        forgery.mkdir()
        rows = ["doc_id,split,file"]
        for i in range(n):
            split = "train" if (i + seed) % 2 == 0 else "validation"
            rows.append(f"d{i},{split},pdfs/d{i}.pdf")
        (corpus / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        lists = build_training_lists([corpus])
        forgery_root = normalise_path(forgery)
        for p in (*lists.train, *lists.validation):
            assert forgery_root not in normalise_path(p).parents
            assert corpus.resolve() in Path(os.path.realpath(p)).parents or \
                normalise_path(corpus) in normalise_path(p).parents
