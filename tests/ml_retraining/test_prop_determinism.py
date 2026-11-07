"""Task 23.3: property test for determinism through the command line.

Feature: ml-retraining, Property 41: Determinism.

For fixed inputs and Master_Seeds, two plumbing runs (``init``, ``check``, ``audit``, ``leak-test``,
``train`` and ``compare``) on the Mini_Corpus give exactly equal cutoffs and metrics.

Example count: the design asks for 10 to 20 examples for the Mini_Corpus training part. Each example
here is two complete plumbing runs (feature extraction, fitting five-or-so models, evaluation), which
takes minutes, so this test runs only 3 examples (``deadline=None``). The varied input is the pair of
cutoff targets (open decision 7); the Mini_Corpus and seeds are fixed. The wider example count is
left to a manual or overnight run via ``DETERMINISM_EXAMPLES``.

**Validates: Requirements 17.14**
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from .conftest import MINI_SEEDS
from .test_cli import (
    Env, _carve_validation, _copy_corpora, _fake_models, _read_json, _results,
    HELD_OUT_SEED, TRAIN_SEED,
)

EXAMPLES = int(os.environ.get("DETERMINISM_EXAMPLES", "3"))


@pytest.fixture(scope="module")
def prepared_corpora(mini_corpus, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("det_prepared")
    shutil.copytree(mini_corpus.path(TRAIN_SEED), root / f"seed-{TRAIN_SEED}")
    shutil.copytree(mini_corpus.path(HELD_OUT_SEED), root / f"heldout-seed-{HELD_OUT_SEED}")
    _carve_validation(root / f"seed-{TRAIN_SEED}")
    return root


def _plumbing(root: Path, prepared: Path, cache: Path, extra: list) -> Env:
    env = Env(root, cache)
    _fake_models(env.models)
    _copy_corpora(prepared, env)
    res = env.init(*extra)
    assert res.code == 0, res.out
    for command in ("check", "audit", "leak-test"):
        res = env.invoke(command)
        assert res.code == 0, (command, res.out)
    assert env.decide("waive-gate", "capability", reason="plumbing").code == 0
    assert env.decide("waive-gate", "subtle", reason="plumbing").code == 0
    res = env.invoke("train")
    assert res.code == 0, res.out
    res = env.invoke("compare")
    assert res.code == 0, res.out
    return env


@settings(max_examples=EXAMPLES, deadline=None, derandomize=True,
          suppress_health_check=list(HealthCheck))
@given(targets=st.tuples(st.sampled_from([3.0, 5.0, 8.0]), st.sampled_from([0.5, 1.0, 2.0])))
def test_two_plumbing_runs_give_equal_cutoffs_and_metrics(targets, prepared_corpora):
    """Property 41: Determinism. Runs 3 examples (see module docstring)."""
    assert len(MINI_SEEDS) == 2
    extra = ["--target-fa-medium", str(targets[0]), "--target-fa-high", str(targets[1])]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = _plumbing(tmp / "a", prepared_corpora, tmp / "cache_a", extra)
        b = _plumbing(tmp / "b", prepared_corpora, tmp / "cache_b", extra)
        ra, rb = a.run_dir, b.run_dir
        assert _read_json(ra / "frozen.json")["cutoffs"] == _read_json(rb / "frozen.json")["cutoffs"]
        rel_a = [p.relative_to(ra) for p in _results(ra)]
        assert rel_a, "compare wrote no result files"
        assert rel_a == [p.relative_to(rb) for p in _results(rb)]
        for rel in rel_a:
            assert _read_json(ra / rel) == _read_json(rb / rel), rel
