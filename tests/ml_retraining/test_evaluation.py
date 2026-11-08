"""Task 18.1: failing tests for ``fraud_detect.ml.retraining.evaluation`` (per-dataset evaluation).

Written before the module exists. Only in-memory datasets, tiny fake models fitted here and pytest
temporary folders are used; nothing under ``models/`` and no Real_Style or Forgery file is touched
(Requirement 17.9). File names such as ``fake_E.pdf`` are labels only.

PINNED INTERFACE (module ``fraud_detect.ml.retraining.evaluation``)::

    SOURCES == ("forgery_dataset", "generator_corpus", "real_style", "awkward_set")
    DATASET_IDS == tuple(sources.DATASETS)                      # the seven dataset ids
    DIFFICULTY_LINE == "difficulty: not recorded in the corpus manifest (generator 1.0.0)"
    FORGERY_ASYMMETRY: str          # contains comparison_report.FORGERY_NOTE
    class MixedSourcesError(ValueError)

    @dataclass(frozen=True) EvalDoc:
        doc_id: str; is_tampered: bool
        features: dict | None = None        # Node 6b: extract_features output
        rows: DataFrame | None = None       # Node 6a: features_a.rows_to_frame schema (needs "y")
        layout=None; n_months=None; tamper_types=(); tamper_method=None
        genuine_resaved=False; file=None; case_type=None
    @dataclass(frozen=True) EvalDataset: dataset_id, source, docs (tuple[EvalDoc])
        __post_init__ raises MixedSourcesError when source != sources.DATASETS[dataset_id], and
        ValueError for an unknown dataset id.
    aggregate(datasets) -> raises MixedSourcesError when the datasets differ in Source or dataset id.

    evaluate_dataset(node, dataset, *, candidate, current, cutoffs, matched=None, seed,
                     size_guard=True) -> dict
        node "6b" | "6a"; candidate: bundle dict; current: baseline.ModelHandle;
        cutoffs: object with .medium and .high (the Candidate's frozen cutoffs);
        matched: same shape, the Current model's matched cutoffs (required when current is ok).
        Candidate and Current are scored through ``baseline.score_b`` / ``baseline.score_a`` (looked
        up as attributes of the baseline module at call time) on the SAME document objects in the
        SAME order. Result keys:
          node, dataset, source, seed, doc_ids (list, input order),
          current_state ("ok" | "no baseline" | "baseline failed"), current_detail (str | None),
          candidate / current: {"matched": BASIS, "deployed": BASIS}   (current is None unless ok)
          difference: {"matched": {"roc_auc","catch_rate","false_alarm_rate"}} candidate minus
                      current, or None unless current is ok
          notes: list[str] (always contains DIFFICULTY_LINE; contains FORGERY_ASYMMETRY for
                 forgery_test / forgery_ood only)
          per_file: {file: {"label": 0|1, "candidate_flag": bool, "current_flag": bool | None}}
                 for real_style, else None
        BASIS keys: cutoffs {"medium","high"}, n_pos, n_neg, n_abstained, abstained_share (percent),
          n_zero_fraud_tampered (Node 6a, else 0), roc_auc, pr_auc (None when a class is empty),
          flag (metrics.metric_flag), catch_rate / false_alarm_rate / hard_negative_fa /
          ordinary_fa: {"value","numer","denom"}, by_type / by_method: {key: rate dict},
          fa_by_months: {"12": rate, "6": rate} for "test" and "heldout_seed", else None.
        Candidate basis: its own frozen cutoffs, guard per `size_guard`; matched = deployed.
        Current "matched": `matched` cutoffs, guard off. Current "deployed": baseline.deployed_cutoffs
        and the stored guard (Node 6a at 0.5 and 0.8).
        Flag: Node 6b probability >= medium cutoff; abstention is "not flagged" and excluded from AUC.
        Node 6a: statement score = max row probability; tampered statements with no y == 1 row are
        counted in n_zero_fraud_tampered and excluded from the detection denominator and from n_pos.
        Hard_Negative = genuine_resaved genuine documents, except dataset "awkward" where it is ALL
        genuine documents; ordinary = the other genuine documents.

    run_unit(vault, node, dataset_id, seed, *, candidate, current, matched, results_root,
             monitor=None, is_ignored=None, size_guard=True) -> UnitOutcome(path, skipped, result)
        vault: an OpenVault. Dataset via vault.load(dataset_id); cutoffs via vault.cutoffs(node).
        File: <results_root>/<node>/<dataset_id>/<seed>.json, written once with safe_io.atomic_write;
        an existing file is not touched and the unit is reported skipped=True (result read from it).
        Before writing, monitor.observe(Measurement(...)) is called for the Candidate (name
        "roc_auc", model "candidate", roc_auc=<auc>, source=<Source>, master_seed=seed, split
        "test_unseen_layout" for dataset heldout_layout else the dataset id); if it raises nothing is
        written. After writing, vault.record_score("<node>/<dataset_id>/<seed>", values).
    run_units(vault, units, **kw) -> list[UnitOutcome]    # units: [(node, dataset_id, seed), ...] in order;
        written units are skipped without loading their dataset (resume at the first unwritten unit).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml.retraining import baseline, comparison_report, metrics, sources
from fraud_detect.ml.retraining.config import RunConfig
from fraud_detect.ml.retraining.monitor import Measurement, MeasurementStop, Monitor
from fraud_detect.ml.retraining.safe_io import WriteGuardError
from fraud_detect.ml.retraining.vault import (
    FrozenCutoffs, HeldOutVault, code_fingerprint, write_freeze, write_run_json)

from fraud_detect.ml.retraining import evaluation as ev  # noqa: E402  (module does not exist yet)

RULE_HASH = "a" * 64
CAND_CUT = FrozenCutoffs(medium=0.5, high=0.8)
MATCHED = FrozenCutoffs(medium=0.5, high=0.8)


# ----------------------------------------------------------------------------- fakes
def _feat(breaks, pages=2):
    f = {name: 0.0 for name in sf.FEATURE_NAMES}
    f.update(n_pages=pages, n_text_objects=60, n_money_tokens=70, balance_reconcile_breaks=breaks)
    return f


def _bundle_b(seed=0, thresholds=None):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(120, len(sf.FEATURE_NAMES))), columns=sf.FEATURE_NAMES)
    y = (X["balance_reconcile_breaks"] > 0).astype(int)
    return {"model": LogisticRegression().fit(X, y), "feature_names": list(sf.FEATURE_NAMES),
            "thresholds": thresholds or {"medium": 0.5, "high": 0.8},
            "size_ranges": {"n_pages": [1, 3], "n_text_objects": [10, 100], "n_money_tokens": [10, 120]}}


class _Spy:
    """Pipeline stand-in: probability = sigmoid(first feature column / 100)."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-X.iloc[:, 0].to_numpy(dtype=float) / 100.0))
        return np.column_stack([1 - p, p])


def _bundle_a():
    return {"pipeline": _Spy(), "feature_set": "pdf", "feature_columns": ["amount_sgd"]}


def _doc_b(i, tampered, breaks, **kw):
    return ev.EvalDoc(doc_id=f"b{i}", is_tampered=tampered, features=_feat(breaks, kw.pop("pages", 2)), **kw)


def _docs_b():
    """6 genuine (g0 g1 resaved; g0-g2 12 months, g3-g5 6 months; g1 g4 g5 raise a false alarm)
    and 6 tampered (t5 is outside the size range and abstains; t2 is missed)."""
    g = [(0, -3, True, 12), (1, 3, True, 12), (2, -3, False, 12),
         (3, -3, False, 6), (4, 3, False, 6), (5, 3, False, 6)]
    docs = [_doc_b(i, False, br, genuine_resaved=rs, n_months=m) for i, br, rs, m in g]
    t = [(6, 3, "edited_amount", "overlay", 2), (7, 3, "edited_amount", "resave", 2),
         (8, -3, "edited_amount", "resave", 2), (9, 3, "font_mismatch", "overlay", 2),
         (10, 3, "font_mismatch", "resave", 2), (11, 3, "font_mismatch", "resave", 20)]
    docs += [_doc_b(i, True, br, tamper_types=(ty,), tamper_method=me, pages=pg, n_months=6)
             for i, br, ty, me, pg in t]
    return tuple(docs)


def _dataset(dataset_id="test", docs=None):
    return ev.EvalDataset(dataset_id=dataset_id, source=sources.DATASETS[dataset_id],
                          docs=docs if docs is not None else _docs_b())


def _current_b(**kw):
    bundle = _bundle_b(seed=1, **kw)
    # The randomly fitted model saturates at probability 1.0 on these unscaled fixture features
    # (60 text objects, 70 money tokens), flagging every document; give it a sane, deterministic
    # fit: probability driven by balance_reconcile_breaks alone.
    model = bundle["model"]
    model.coef_ = np.zeros_like(model.coef_)
    model.coef_[0, list(sf.FEATURE_NAMES).index("balance_reconcile_breaks")] = 1.0
    model.intercept_ = np.zeros_like(model.intercept_)
    return baseline.wrap("6b", bundle)


def _eval_b(dataset=None, **over):
    kw = dict(candidate=_bundle_b(seed=0), current=_current_b(thresholds={"medium": 0.35, "high": 0.75}),
              cutoffs=CAND_CUT, matched=MATCHED, seed=20260801)
    kw.update(over)
    return ev.evaluate_dataset("6b", dataset or _dataset(), **kw)


def _rate(d):
    return (d["numer"], d["denom"])


# ----------------------------------------------------------------------------- sources and datasets
def test_dataset_ids_and_sources_are_the_seven_of_the_registry():
    assert ev.DATASET_IDS == tuple(sources.DATASETS) and len(ev.DATASET_IDS) == 7
    assert set(ev.SOURCES) == set(sources.DATASETS.values())


def test_every_result_row_has_exactly_one_source_and_dataset():
    for ds_id, src in sources.DATASETS.items():
        r = _eval_b(_dataset(ds_id))
        assert (r["dataset"], r["source"]) == (ds_id, src)


def test_dataset_with_the_wrong_source_is_rejected():
    with pytest.raises(ev.MixedSourcesError):
        ev.EvalDataset(dataset_id="real_style", source="generator_corpus", docs=())
    with pytest.raises(ValueError):
        ev.EvalDataset(dataset_id="not_a_dataset", source="generator_corpus", docs=())


def test_aggregating_across_sources_or_datasets_is_rejected():
    a, b = _dataset("test"), _dataset("forgery_test")
    with pytest.raises(ev.MixedSourcesError):
        ev.aggregate([a, b])
    with pytest.raises(ev.MixedSourcesError):
        ev.aggregate([_dataset("test"), _dataset("heldout_seed")])
    assert ev.aggregate([a, _dataset("test")]).dataset_id == "test"


def test_candidate_and_current_are_scored_on_identical_document_lists(monkeypatch):
    calls = []
    real = baseline.score_b

    def spy(handle, feats, **kw):
        calls.append([id(f) for f in feats])
        return real(handle, feats, **kw)

    monkeypatch.setattr(baseline, "score_b", spy)
    ds = _dataset()
    r = _eval_b(ds)
    expected = [id(d.features) for d in ds.docs]
    assert len(calls) >= 2 and all(c == expected for c in calls)
    assert r["doc_ids"] == [d.doc_id for d in ds.docs]


# ----------------------------------------------------------------------------- Node 6b metrics
def test_candidate_flags_abstention_and_auc():
    r = _eval_b()
    c = r["candidate"]["matched"]
    assert c["cutoffs"] == {"medium": 0.5, "high": 0.8}
    assert _rate(c["false_alarm_rate"]) == (3, 6)
    assert _rate(c["catch_rate"]) == (4, 6)                 # t5 abstained: not flagged
    assert c["n_abstained"] == 1 and c["abstained_share"] == pytest.approx(100 / 12)
    assert (c["n_pos"], c["n_neg"]) == (5, 6)               # abstained document left out of AUC counts
    scored = baseline.score_b(baseline.wrap("6b", _bundle_b(seed=0)), [d.features for d in _docs_b()],
                              dataset="test")
    keep = ~scored.abstain
    labels = np.array([d.is_tampered for d in _docs_b()])
    assert c["roc_auc"] == pytest.approx(metrics.roc_auc(labels[keep], scored.probs[keep]))
    assert c["flag"] == "indicative only"                   # fewer than 30 per class
    assert r["candidate"]["deployed"] == c


def test_size_guard_off_removes_abstention():
    c = _eval_b(size_guard=False)["candidate"]["matched"]
    assert c["n_abstained"] == 0 and _rate(c["catch_rate"]) == (5, 6)


def test_catch_rate_by_type_and_method_with_counts():
    c = _eval_b()["candidate"]["matched"]
    assert {k: _rate(v) for k, v in c["by_type"].items()} == {"edited_amount": (2, 3), "font_mismatch": (2, 3)}
    assert {k: _rate(v) for k, v in c["by_method"].items()} == {"overlay": (2, 2), "resave": (2, 4)}


def test_hard_negative_and_ordinary_false_alarms_are_separate():
    c = _eval_b()["candidate"]["matched"]
    assert _rate(c["hard_negative_fa"]) == (1, 2)           # g0, g1 are resaved; g1 flagged
    assert _rate(c["ordinary_fa"]) == (2, 4)                # g4, g5 flagged


def test_awkward_hard_negatives_are_all_genuine_documents():
    r = _eval_b(_dataset("awkward"))
    c = r["candidate"]["matched"]
    assert _rate(c["hard_negative_fa"]) == (3, 6)           # every genuine document, resaved or not
    assert c["ordinary_fa"]["denom"] == 0 or c["ordinary_fa"]["value"] is None


def test_twelve_and_six_month_false_alarm_rates_for_test_and_heldout_seed_only():
    for ds_id in ("test", "heldout_seed"):
        m = _eval_b(_dataset(ds_id))["candidate"]["matched"]["fa_by_months"]
        assert _rate(m["12"]) == (1, 3) and _rate(m["6"]) == (2, 3)
    assert _eval_b(_dataset("forgery_test"))["candidate"]["matched"]["fa_by_months"] is None


def test_difficulty_line_replaces_difficulty_rows_and_forgery_carries_the_asymmetry():
    assert ev.DIFFICULTY_LINE == "difficulty: not recorded in the corpus manifest (generator 1.0.0)"
    assert comparison_report.FORGERY_NOTE in ev.FORGERY_ASYMMETRY
    for ds_id in ev.DATASET_IDS:
        notes = _eval_b(_dataset(ds_id))["notes"]
        assert ev.DIFFICULTY_LINE in notes
        assert (ev.FORGERY_ASYMMETRY in notes) == (ds_id in ("forgery_test", "forgery_ood"))


def test_real_style_is_reported_per_file():
    docs = (ev.EvalDoc(doc_id="e", is_tampered=True, features=_feat(3), file="fake_E.pdf"),
            ev.EvalDoc(doc_id="f", is_tampered=False, features=_feat(-3), file="fake_F.pdf"))
    r = _eval_b(_dataset("real_style", docs))
    assert r["per_file"] == {"fake_E.pdf": {"label": 1, "candidate_flag": True, "current_flag": True},
                             "fake_F.pdf": {"label": 0, "candidate_flag": False, "current_flag": False}}
    assert r["candidate"]["matched"]["flag"] == "indicative only"
    assert _eval_b()["per_file"] is None


# ----------------------------------------------------------------------------- Current model and bases
def test_current_matched_basis_is_guard_off_and_deployed_basis_keeps_stored_cutoffs_and_guard():
    r = _eval_b()
    cur = r["current"]
    assert cur["matched"]["cutoffs"] == {"medium": 0.5, "high": 0.8}
    assert cur["matched"]["n_abstained"] == 0 and _rate(cur["matched"]["catch_rate"]) == (5, 6)
    assert cur["deployed"]["cutoffs"] == {"medium": 0.35, "high": 0.75}
    assert cur["deployed"]["n_abstained"] == 1 and _rate(cur["deployed"]["catch_rate"]) == (4, 6)


def test_difference_is_candidate_minus_current_on_the_matched_basis():
    r = _eval_b()
    cand, cur, diff = r["candidate"]["matched"], r["current"]["matched"], r["difference"]["matched"]
    assert r["current_state"] == "ok" and r["current_detail"] is None
    assert diff["roc_auc"] == pytest.approx(cand["roc_auc"] - cur["roc_auc"])
    assert diff["catch_rate"] == pytest.approx(cand["catch_rate"]["value"] - cur["catch_rate"]["value"])
    assert diff["false_alarm_rate"] == pytest.approx(
        cand["false_alarm_rate"]["value"] - cur["false_alarm_rate"]["value"])


def test_no_baseline_still_reports_candidate_and_leaves_current_cells_empty(tmp_path):
    missing = baseline.load_current("6b", tmp_path / "absent.joblib")
    r = _eval_b(current=missing, matched=None)
    assert r["current_state"] == "no baseline"
    assert r["current"] is None and r["difference"] is None
    assert r["candidate"]["matched"]["catch_rate"]["denom"] == 6


def test_baseline_failed_names_node_and_dataset_and_keeps_candidate(tmp_path):
    p = tmp_path / "bad.joblib"
    p.write_bytes(b"garbage")
    r = _eval_b(_dataset("forgery_ood"), current=baseline.load_current("6b", p), matched=None)
    assert r["current_state"] == "baseline failed"
    assert "6b" in r["current_detail"] and "forgery_ood" in r["current_detail"]
    assert r["current"] is None and r["difference"] is None
    assert r["candidate"]["matched"]["n_pos"] == 5


# ----------------------------------------------------------------------------- Node 6a (per statement)
def _stmt(doc_id, tampered, amounts, fraud_idx=()):
    n = len(amounts)
    rows = pd.DataFrame({
        "corpus_key": "c", "doc_id": doc_id, "row_index": range(n), "account_id": f"acct-{doc_id}",
        "timestamp": [f"2026-03-{i + 1:02d}" for i in range(n)],
        "direction": ["CR"] * n, "amount_sgd": [float(a) for a in amounts],
        "balance_after_sgd": [5000.0 + i for i in range(n)],
        "y": [1 if i in fraud_idx else 0 for i in range(n)], "split": "test"})
    return ev.EvalDoc(doc_id=doc_id, is_tampered=tampered, rows=rows)


def _docs_a():
    return (_stmt("s0", False, [10, 10, 10]), _stmt("s1", False, [10, 10, 10]),
            _stmt("s2", False, [10, 10, 10]), _stmt("s3", False, [10, 1000, 10]),
            _stmt("s4", True, [10, 1000, 10], fraud_idx=(1,)),
            _stmt("s5", True, [10, 10, 10], fraud_idx=(2,)),
            _stmt("s6", True, [10, 10, 10]))                       # tampered, zero fraud rows


def _eval_a(**over):
    kw = dict(candidate=_bundle_a(), current=baseline.wrap("6a", _bundle_a()),
              cutoffs=FrozenCutoffs(medium=0.9, high=0.99), matched=FrozenCutoffs(medium=0.9, high=0.99),
              seed=20260801)
    kw.update(over)
    return ev.evaluate_dataset("6a", _dataset("test", _docs_a()), **kw)


def test_node_6a_is_judged_per_statement_and_zero_fraud_tampered_are_excluded():
    c = _eval_a()["candidate"]["matched"]
    assert _rate(c["false_alarm_rate"]) == (1, 4)           # s3 only, by its maximum row score
    assert _rate(c["catch_rate"]) == (1, 2)                 # s4 caught, s5 missed; s6 not in the denominator
    assert c["n_zero_fraud_tampered"] == 1 and c["n_pos"] == 2 and c["n_neg"] == 4


def test_node_6a_as_deployed_uses_fixed_cutoffs_0_5_and_0_8():
    cur = _eval_a()["current"]
    assert cur["deployed"]["cutoffs"] == {"medium": 0.5, "high": 0.8}
    assert _rate(cur["deployed"]["false_alarm_rate"]) == (4, 4)   # sigmoid(0.1) = 0.525 >= 0.5
    assert _rate(cur["matched"]["false_alarm_rate"]) == (1, 4)    # matched cutoff 0.9
    assert _eval_a()["candidate"]["deployed"] == _eval_a()["candidate"]["matched"]


# ----------------------------------------------------------------------------- vault, files, monitor
@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "runs" / "r1"


@pytest.fixture
def opened_vault(run_dir, is_ignored, clock):
    """An opened vault whose loaders count their calls."""
    calls = {}

    def loader(ds_id):
        def load():
            calls[ds_id] = calls.get(ds_id, 0) + 1
            return _dataset(ds_id)
        return load

    write_run_json(run_dir, config=RunConfig(), switch_rule_sha256=RULE_HASH,
                   cutoff_rule={"target_fa_medium": 5.0, "target_fa_high": 1.0, "level": "document"},
                   fingerprint=code_fingerprint(run_dir.parent), is_ignored=is_ignored)
    write_freeze(run_dir, cutoffs={"6b": {"medium": 0.5, "high": 0.8}, "6a": {"medium": 0.9, "high": 0.99}},
                 candidate_sha256={"6b": "b" * 64, "6a": "c" * 64}, switch_rule_sha256=RULE_HASH,
                 is_ignored=is_ignored)
    loaders = {d: loader(d) for d in ev.DATASET_IDS}
    vault = HeldOutVault(run_dir, loaders=loaders, is_ignored=is_ignored, clock=clock).open()
    return vault, calls


def _unit_kw(run_dir, is_ignored, **over):
    kw = dict(candidate=_bundle_b(seed=0), current=_current_b(), matched=MATCHED,
              results_root=run_dir / "results", is_ignored=is_ignored)
    kw.update(over)
    return kw


def test_held_out_datasets_are_read_only_through_the_vault(opened_vault, run_dir, is_ignored):
    vault, calls = opened_vault
    out = ev.run_unit(vault, "6b", "heldout_seed", 20260801, **_unit_kw(run_dir, is_ignored))
    assert calls == {"heldout_seed": 1}
    assert out.result["dataset"] == "heldout_seed" and out.result["source"] == "generator_corpus"
    src = Path(ev.__file__).read_text()
    for token in ("load_corpus", "labels.csv", "ReadOnlySource", "load_forgery", "load_real_style",
                  "torch"):
        assert token not in src


def test_result_is_written_once_atomically_at_the_pinned_path(opened_vault, run_dir, is_ignored):
    vault, calls = opened_vault
    out = ev.run_unit(vault, "6b", "test", 20260801, **_unit_kw(run_dir, is_ignored))
    path = run_dir / "results" / "6b" / "test" / "20260801.json"
    assert out.path == path and path.is_file() and out.skipped is False
    assert json.loads(path.read_text()) == json.loads(json.dumps(out.result))
    assert not list(path.parent.glob("*.tmp")) and [p.name for p in path.parent.iterdir()] == ["20260801.json"]
    before = path.read_bytes()
    again = ev.run_unit(vault, "6b", "test", 20260801, **_unit_kw(run_dir, is_ignored))
    assert again.skipped is True and path.read_bytes() == before and calls["test"] == 1
    assert vault.results()["6b/test/20260801"]["held_out"] == "held out"


def test_write_guard_blocks_a_non_ignored_results_folder(opened_vault, run_dir, is_ignored):
    vault, _ = opened_vault
    is_ignored.ignored = False
    with pytest.raises(WriteGuardError):
        ev.run_unit(vault, "6b", "test", 20260801, **_unit_kw(run_dir, is_ignored))
    assert not (run_dir / "results").exists()


def test_rerun_resumes_at_the_first_unwritten_unit(opened_vault, run_dir, is_ignored):
    vault, calls = opened_vault
    kw = _unit_kw(run_dir, is_ignored)
    ev.run_unit(vault, "6b", "test", 20260801, **kw)
    first = run_dir / "results" / "6b" / "test" / "20260801.json"
    before = first.read_bytes()
    units = [("6b", "test", 20260801), ("6b", "heldout_seed", 20260801), ("6b", "forgery_test", 20260801)]
    outs = ev.run_units(vault, units, **kw)
    assert [o.skipped for o in outs] == [True, False, False]
    assert calls == {"test": 1, "heldout_seed": 1, "forgery_test": 1}   # dataset 1 not loaded again
    assert first.read_bytes() == before
    assert (run_dir / "results" / "6b" / "forgery_test" / "20260801.json").is_file()


def test_monitor_is_called_before_the_result_file_exists(opened_vault, run_dir, is_ignored):
    vault, _ = opened_vault
    seen = []
    target = run_dir / "results" / "6b" / "heldout_layout" / "20260803.json"

    class Recorder:
        def observe(self, m):
            seen.append((m, target.exists()))

    ev.run_unit(vault, "6b", "heldout_layout", 20260803, monitor=Recorder(), **_unit_kw(run_dir, is_ignored))
    assert target.is_file() and len(seen) >= 1
    m, existed = seen[0]
    assert isinstance(m, Measurement) and existed is False
    assert (m.name, m.model, m.source, m.master_seed) == ("roc_auc", "candidate", "generator_corpus", 20260803)
    assert m.split == "test_unseen_layout" and m.roc_auc is not None


def test_implausible_measurement_stops_before_anything_is_written(opened_vault, run_dir, is_ignored, clock):
    vault, _ = opened_vault
    perfect = ev.EvalDataset(
        dataset_id="heldout_layout", source="generator_corpus",
        docs=tuple(_doc_b(i, i >= 3, 3 if i >= 3 else -3, tamper_types=("edited_amount",) if i >= 3 else ())
                   for i in range(6)))
    vault._loaders["heldout_layout"] = lambda: perfect
    mon = Monitor(run_dir, is_ignored=is_ignored, clock=clock)
    with pytest.raises(MeasurementStop):
        ev.run_unit(vault, "6b", "heldout_layout", 20260801, monitor=mon, **_unit_kw(run_dir, is_ignored))
    assert not (run_dir / "results").exists() and mon.is_stopped
    assert "6b/heldout_layout/20260801" not in vault.results()


def test_evaluate_dataset_signature_is_pinned():
    params = inspect.signature(ev.evaluate_dataset).parameters
    assert list(params)[:3] == ["node", "dataset", "candidate"] or "candidate" in params
    for name in ("current", "cutoffs", "matched", "seed", "size_guard"):
        assert name in params
