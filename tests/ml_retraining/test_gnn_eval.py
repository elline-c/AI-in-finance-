"""Task 16.1: failing tests for ``fraud_detect.ml.retraining.gnn_eval`` (GNN lookup accounting).
Written before the module exists. No model file, no torch. Fake score stores are plain dicts in the
shape ``gnn_lookup`` reads (``{"account_scores": {uen: score}, "edge_scores": {}, "meta": {}}``).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.gnn_eval``)::

    GNN_NOT_RETRAINED_LINE == "GNN: not retrained, corpus has no transfer graph"
    NO_SCORE == "no score"

    def lookup_coverage(heldout_uens, store) -> GnnCoverage
        # heldout_uens: iterable of UEN strings (duplicates collapse; order irrelevant).
        # store: None or a gnn_lookup-style dict; each UEN is looked up with
        # fraud_detect.ml.gnn.gnn_lookup.score_account(store, uen). None store == every UEN absent.
    GnnCoverage (frozen dataclass):
        .total: int            distinct held-out UENs
        .found: int            |held-out ∩ store|
        .no_score: int         total - found
        .no_score_uens: tuple  sorted absent UENs
        .scores: dict          uen -> float for found UENs, uen -> None for absent UENs (never a substitute)
        .share: float | None   100 * found / total (percent); None when total == 0
        .text: str             "<share, one decimal>% (<found>/<total>)"; "not computable (0/0)" when total == 0

    def gnn_flags(scores, cutoff) -> list   # scores: sequence of float|None -> [None if s is None else s >= cutoff]
    def combine_flags(gbm_flags, gnn_flags) -> list[bool]
        # gbm OR gnn; a gnn flag of None (no score) counts as "not flagged" so the result equals gbm.
    def compare_models(gbm_flags, gnn_flags, is_genuine, is_positive) -> dict
        # Parallel per-statement sequences. gnn_flags entries may be None (no score).
        # is_genuine: statement is genuine (false-alarm denominator).
        # is_positive: tampered statement with at least one fraud row (detection denominator).
        # Returns {"gbm": ModelRates, "gnn": ModelRates, "combined": ModelRates}
    ModelRates (frozen dataclass): .false_alarm and .detection are metrics.Rate objects
        (value is a fraction, .numer, .denom, .text), .status is "ok" or "not computable".
        gbm and combined are computed over all statements. gnn is computed over the SCORED
        statements only (denominators count only statements whose gnn flag is not None); with no
        scored statement gnn.status == "not computable" and both rates have value None.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from fraud_detect.ml.retraining import gnn_eval
from fraud_detect.ml.retraining.gnn_eval import (
    GNN_NOT_RETRAINED_LINE, NO_SCORE, combine_flags, compare_models, gnn_flags, lookup_coverage)

REPO = Path(__file__).resolve().parents[2]


def _store(uens):
    return {"account_scores": {u: 0.9 for u in uens}, "edge_scores": {}, "meta": {}}


def test_constants():
    assert GNN_NOT_RETRAINED_LINE == "GNN: not retrained, corpus has no transfer graph"
    assert NO_SCORE == "no score"


def test_share_found_is_intersection_over_heldout_one_decimal():
    held = [f"U{i}" for i in range(7)]
    cov = lookup_coverage(held, _store(["U0", "U1", "OTHER"]))
    assert (cov.total, cov.found, cov.no_score) == (7, 2, 5)
    assert cov.share == pytest.approx(100 * 2 / 7)
    assert cov.text == "28.6% (2/7)"
    assert cov.no_score_uens == tuple(f"U{i}" for i in range(2, 7))


def test_absent_uens_have_no_substitute_number():
    cov = lookup_coverage(["A", "B"], _store(["A"]))
    assert cov.scores == {"A": 0.9, "B": None}
    assert cov.no_score == 1


def test_missing_store_and_empty_store_give_zero_coverage():
    for store in (None, _store([])):
        cov = lookup_coverage(["A", "B", "A"], store)  # duplicates collapse
        assert (cov.total, cov.found, cov.no_score) == (2, 0, 2)
        assert cov.share == 0.0 and cov.text == "0.0% (0/2)"
        assert all(v is None for v in cov.scores.values())


def test_empty_heldout_is_not_computable():
    cov = lookup_coverage([], _store(["A"]))
    assert cov.total == 0 and cov.share is None
    assert cov.text == "not computable (0/0)"


def test_gnn_flags_none_stays_none():
    assert gnn_flags([None, 0.2, 0.8, 0.5], 0.5) == [None, False, True, True]


def test_combined_is_or_and_equals_gbm_without_gnn_score():
    gbm = [True, False, True, False]
    gnn = [None, True, False, None]
    assert combine_flags(gbm, gnn) == [True, True, True, False]
    assert combine_flags(gbm, [None] * 4) == gbm


def test_compare_models_rates_on_same_statements():
    #            g1     g2     g3     p1     p2     p3
    genuine = [True, True, True, False, False, False]
    positive = [False, False, False, True, True, True]
    gbm = [True, False, False, True, True, False]
    gnn = [False, True, None, True, None, False]
    out = compare_models(gbm, gnn, genuine, positive)
    assert set(out) == {"gbm", "gnn", "combined"}
    assert (out["gbm"].false_alarm.numer, out["gbm"].false_alarm.denom) == (1, 3)
    assert (out["gbm"].detection.numer, out["gbm"].detection.denom) == (2, 3)
    comb = combine_flags(gbm, gnn)
    assert comb == [True, True, False, True, True, False]
    assert (out["combined"].false_alarm.numer, out["combined"].false_alarm.denom) == (2, 3)
    assert (out["combined"].detection.numer, out["combined"].detection.denom) == (2, 3)
    # GNN alone only over scored statements: g1, g2, p1, p3
    assert (out["gnn"].false_alarm.numer, out["gnn"].false_alarm.denom) == (1, 2)
    assert (out["gnn"].detection.numer, out["gnn"].detection.denom) == (1, 2)
    assert out["gnn"].status == "ok" and out["gbm"].status == "ok"


def test_gnn_alone_not_computable_without_scored_statement_and_combined_equals_gbm():
    genuine = [True, True, False, False]
    positive = [False, False, True, True]
    gbm = [True, False, True, False]
    out = compare_models(gbm, [None] * 4, genuine, positive)
    g = out["gnn"]
    assert g.status == "not computable"
    assert g.false_alarm.value is None and g.detection.value is None
    for field in ("false_alarm", "detection"):
        a, b = getattr(out["combined"], field), getattr(out["gbm"], field)
        assert (a.numer, a.denom) == (b.numer, b.denom)


def test_compare_models_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compare_models([True], [None, None], [True], [False])


def test_no_torch_in_module_or_after_import():
    src = Path(gnn_eval.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        assert not any(n.split(".")[0] in ("torch", "torch_geometric") for n in names)
    code = ("import sys; import fraud_detect.ml.retraining.gnn_eval; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','torch_geometric')]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0
