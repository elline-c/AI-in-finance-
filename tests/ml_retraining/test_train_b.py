"""Task 15.1: failing tests for ``fraud_detect.ml.retraining.train_b`` (Node 6b candidates).

Written before the module exists. Only the Mini_Corpus fixture and pytest temporary folders are
used; nothing under the real ``models/`` folder is read or written (Requirement 17.9).

PINNED INTERFACE (module ``fraud_detect.ml.retraining.train_b``)::

    MODEL_LABEL == "model trained on synthetic data"
    GNN_LINE    == "GNN: not retrained, corpus has no transfer graph"
    ABLATION_NOTE == "not used by the Switch_Rule"
    class TrainRefused(Exception):
        .open_items: tuple[str, ...]   # gate ids and Leak_Test suspect names that are still open
        .exit_code: int                # gates.EXIT_INTEGRITY (2) if an integrity gate is open,
                                       # else gates.EXIT_WAITING (3)

    train_candidate(train, valid, *, master_seed, held_out_layout, out_path, audit=AUDIT_V1,
                    rule=CutoffRule(), variant="audited", gate_results=(), leak=None,
                    is_ignored=None, verify=None) -> CandidateResult
        train / valid: pandas DataFrames shaped like ``features_b.FeatureTableB.frame``
            (``split``, ``is_tampered`` and the EXTRACT_COLUMNS). Every ``train`` row must have
            split == "train" and every ``valid`` row split == "validation", else ValueError.
        variant: "audited" (inputs = audit.model_input_columns(audit, "6b"), neutralised features
            transformed inside the pipeline) or "all11" (the informational ablation: inputs =
            statement_features.FEATURE_NAMES, bundle["informational"] is True and
            ABLATION_NOTE is in bundle["note"]).
        gate_results: sequence of gates.GateResult; leak: leak.LeakResult or None.
            Any result with passed False, or leak.blocked True, raises TrainRefused before any
            file is written or any model is fitted.
        master_seed: int, or None for a pooled release candidate.
        verify: callable(bundle, feats_dict) used to prove the written file loads and scores;
            default loads the file with joblib and calls ``fraud_detect.nodes.ml_statement.assess``.
            When it raises, the file stays and a marker ``Path(str(path) + ".failed")`` is written.
        The file is written with safe_io.atomic_write (is_ignored injectable, so a non-ignored
        path raises WriteGuardError and writes nothing). Fitting runs inside
        ``threadpoolctl.threadpool_limits(1)`` (looked up as an attribute of the threadpoolctl
        module at call time).
    CandidateResult (dataclass): .path (Path), .bundle (dict), .failed (bool), .marker_path (Path|None)

    train_run(tables, *, held_out_layout, out_dir, audit=AUDIT_V1, rule=CutoffRule(),
              gate_results=(), leak=None, is_ignored=None, verify=None) -> TrainOutputs
        tables: Mapping[int master_seed, DataFrame with every Split of that seed].
        Writes  <out_dir>/models/6b/seed-<seed>.joblib          (audited, one per seed)
                <out_dir>/models/6b/seed-<seed>-all11.joblib    (ablation, one per seed)
                <out_dir>/release/6b.joblib, <out_dir>/release/6b-all11.joblib
        The release candidates are fitted on the pooled "train" rows of all seeds, with cutoffs
        chosen on the pooled "validation" rows; their bundle has master_seed None and
        ``pooled_seeds`` (sorted list). Prints GNN_LINE and, per candidate, the validation operating
        curve (cutoffs.format_operating_curve).
        TrainOutputs (dataclass): .per_seed {seed: CandidateResult}, .ablation {seed: CandidateResult},
        .release CandidateResult, .release_ablation CandidateResult, .report_lines list[str]
        (contains GNN_LINE).

    Bundle keys (all candidates): model, thresholds{medium, high}, size_ranges, feature_names,
        audit_version, audit_digest, master_seed, held_out_layout, size_guard, cutoff_rule,
        validation_curve, model_label, synthetic_data, n_train.
    bundle["model"]: sklearn Pipeline with steps named "prep" (a ColumnTransformer) and "clf"
        (HistGradientBoostingClassifier with the settings of train_statement.py).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml.retraining import gates
from fraud_detect.ml.retraining.audit import AUDIT_V1, model_input_columns
from fraud_detect.ml.retraining.cutoffs import CutoffRule, choose_cutoffs
from fraud_detect.ml.retraining.features_b import build_feature_table
from fraud_detect.ml.retraining.leak import LeakResult
from fraud_detect.ml.retraining.safe_io import WriteGuardError

from fraud_detect.ml.retraining import train_b as tb  # noqa: E402  (module does not exist yet)

REPO = Path(__file__).resolve().parents[2]
RULE = CutoffRule(target_fa_medium=20.0, target_fa_high=20.0)  # small Mini data: loose targets


# ----------------------------------------------------------------------------- fixtures
def _split_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Carve a validation Split out of the Mini train rows (the Mini validation has no genuine)."""
    tr = frame[frame["split"] == "train"]
    valid = pd.concat([g.iloc[::2] for _, g in tr.groupby("is_tampered")]).copy()
    valid["split"] = "validation"
    rest = tr.drop(valid.index)
    rest = rest.assign(split="train")
    other = frame[~frame["split"].isin(["train", "validation"])]
    return pd.concat([rest, valid, other], ignore_index=True)


@pytest.fixture(scope="session")
def tables(mini_corpus, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache_b")
    out = {}
    for seed in mini_corpus.seeds:
        t = build_feature_table(mini_corpus.path(seed), cache_root=cache, is_ignored=lambda p: True)
        out[seed] = _split_table(t.frame)
    return out


@pytest.fixture
def frames(tables):
    seed = sorted(tables)[0]
    t = tables[seed]
    return seed, t[t["split"] == "train"].copy(), t[t["split"] == "validation"].copy()


def _fit(frames, tmp_path, name="cand.joblib", is_ignored=None, **kw):
    seed, train, valid = frames
    kw.setdefault("rule", RULE)
    return tb.train_candidate(
        train, valid, master_seed=seed, held_out_layout="two_col_ddmmyy_cr",
        out_path=tmp_path / name, is_ignored=is_ignored or (lambda p: True), **kw)


@pytest.fixture
def candidate(frames, tmp_path):
    return _fit(frames, tmp_path)


def _models_snapshot():
    folder = REPO / "models"
    if not folder.is_dir():
        return {}
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------- settings
def test_hgb_settings_equal_train_statement(candidate):
    tree = ast.parse((REPO / "fraud_detect/ml/train_statement.py").read_text())
    kws = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "HistGradientBoostingClassifier":
            kws = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    assert kws, "HistGradientBoostingClassifier call not found in train_statement.py"
    clf = candidate.bundle["model"].named_steps["clf"]
    assert type(clf).__name__ == "HistGradientBoostingClassifier"
    for k, v in kws.items():
        assert getattr(clf, k) == v, k


def test_fitted_inside_one_thread_limit(frames, tmp_path, monkeypatch):
    import threadpoolctl
    from sklearn.ensemble import HistGradientBoostingClassifier
    state = {"active": 0, "calls": [], "fit_active": []}

    class Recorder:
        def __init__(self, *a, **k):
            state["calls"].append((a, k))

        def __enter__(self):
            state["active"] += 1
            return self

        def __exit__(self, *exc):
            state["active"] -= 1

    orig_fit = HistGradientBoostingClassifier.fit

    def fit(self, *a, **k):
        state["fit_active"].append(state["active"])
        return orig_fit(self, *a, **k)

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", Recorder)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", fit)
    _fit(frames, tmp_path)
    assert state["fit_active"] and all(n >= 1 for n in state["fit_active"])
    a, k = state["calls"][0]
    assert 1 in a or k.get("limits") == 1


# ----------------------------------------------------------------------------- inputs
def test_input_columns_equal_audited_ordered_list(candidate):
    expected = model_input_columns(AUDIT_V1, "6b")
    assert expected and len(expected) < len(sf.FEATURE_NAMES)
    assert candidate.bundle["feature_names"] == expected
    assert list(candidate.bundle["model"].feature_names_in_) == expected
    for dropped in ("n_pages", "n_text_objects", "n_distinct_fonts", "n_content_streams",
                    "n_money_tokens", "balance_reconcile_rate", "n_currency_styles"):
        assert dropped not in candidate.bundle["feature_names"]


def test_audit_decides_inputs_when_decision_changes(frames, tmp_path):
    entries = tuple(
        dataclasses.replace(e, decision="keep") if (e.node == "6b" and e.name == "n_pages")
        else dataclasses.replace(e, decision="drop", neutralised_form=None)
        if (e.node == "6b" and e.name == "minor_font_share") else e
        for e in AUDIT_V1.entries)
    audit = dataclasses.replace(AUDIT_V1, entries=entries)
    res = _fit(frames, tmp_path, audit=audit)
    names = res.bundle["feature_names"]
    assert "n_pages" in names and "minor_font_share" not in names
    assert names == model_input_columns(audit, "6b")
    assert res.bundle["audit_digest"] == audit.digest()


def test_neutralise_step_is_inside_the_pipeline(candidate):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer
    pipe = candidate.bundle["model"]
    assert [n for n, _ in pipe.steps] == ["prep", "clf"]
    prep = pipe.named_steps["prep"]
    assert isinstance(prep, ColumnTransformer)
    assert any(isinstance(t, FunctionTransformer) for _, t, _ in prep.transformers_)
    base = {c: 0.0 for c in candidate.bundle["feature_names"]}
    lo, hi = dict(base, minor_font_share=0.2), dict(base, minor_font_share=0.5)
    p = pipe.predict_proba(pd.DataFrame([lo, hi])[candidate.bundle["feature_names"]])[:, 1]
    assert p[0] == p[1]  # shares at or above 0.10 collapse to one value


# ----------------------------------------------------------------------------- fitting data
def test_fitted_on_train_only_and_cutoffs_from_validation_only(frames, tmp_path):
    seed, train, valid = frames
    a = _fit(frames, tmp_path, name="a.joblib")
    # a different validation Split changes the cutoffs but must not change the fitted model
    valid2 = valid.iloc[::-1].iloc[: max(4, len(valid) - 2)]
    b = tb.train_candidate(train, valid2, master_seed=seed, held_out_layout="two_col_ddmmyy_cr",
                           out_path=tmp_path / "b.joblib", rule=RULE, is_ignored=lambda p: True)
    probe = train[a.bundle["feature_names"]]
    np.testing.assert_array_equal(a.bundle["model"].predict_proba(probe),
                                  b.bundle["model"].predict_proba(probe))
    assert a.bundle["n_train"] == len(train)
    # cutoffs are exactly the cutoff rule applied to this model's validation scores
    scores = a.bundle["model"].predict_proba(valid[a.bundle["feature_names"]])[:, 1]
    cut = choose_cutoffs(scores, ~valid["is_tampered"].astype(bool).to_numpy(), RULE)
    assert a.bundle["thresholds"] == {"medium": cut.medium, "high": cut.high}
    assert a.bundle["thresholds"]["high"] >= a.bundle["thresholds"]["medium"]


def test_rows_outside_their_split_are_refused(frames, tmp_path):
    seed, train, valid = frames
    bad = pd.concat([train, valid.iloc[:1]])
    with pytest.raises(ValueError):
        tb.train_candidate(bad, valid, master_seed=seed, held_out_layout="x",
                           out_path=tmp_path / "bad.joblib", rule=RULE, is_ignored=lambda p: True)
    with pytest.raises(ValueError):
        tb.train_candidate(train, train, master_seed=seed, held_out_layout="x",
                           out_path=tmp_path / "bad2.joblib", rule=RULE, is_ignored=lambda p: True)
    assert not list(tmp_path.glob("bad*"))


# ----------------------------------------------------------------------------- bundle
def test_bundle_contents(frames, candidate):
    seed, train, _ = frames
    b = candidate.bundle
    for key in ("model", "thresholds", "size_ranges", "feature_names", "audit_version", "audit_digest",
                "master_seed", "held_out_layout", "size_guard", "cutoff_rule", "validation_curve",
                "model_label", "synthetic_data", "n_train"):
        assert key in b, key
    assert set(b["thresholds"]) == {"medium", "high"}
    assert b["audit_version"] == AUDIT_V1.version == "audit-v1"
    assert b["audit_digest"] == AUDIT_V1.digest()
    assert b["master_seed"] == seed
    assert b["held_out_layout"] == "two_col_ddmmyy_cr"
    assert b["model_label"] == "model trained on synthetic data" == tb.MODEL_LABEL
    assert b["synthetic_data"] is True
    assert b["size_guard"] == {"mode": "candidate_quantiles", "quantiles": [0.01, 0.99]}
    assert b["cutoff_rule"]["target_fa_medium"] == RULE.target_fa_medium
    assert b["cutoff_rule"]["target_fa_high"] == RULE.target_fa_high
    assert isinstance(b["validation_curve"], list) and b["validation_curve"]
    from fraud_detect.ml.train_statement import GUARD_QUANTILES
    lo, hi = GUARD_QUANTILES
    assert set(b["size_ranges"]) == set(sf.SIZE_FEATURES)
    for f in sf.SIZE_FEATURES:  # quantiles of the training documents, not of validation
        assert b["size_ranges"][f] == [float(train[f].quantile(lo)), float(train[f].quantile(hi))]
    on_disk = joblib.load(candidate.path)
    assert on_disk["feature_names"] == b["feature_names"] and on_disk["thresholds"] == b["thresholds"]


# ----------------------------------------------------------------------------- writing
def test_written_through_safe_io_and_models_folder_untouched(frames, tmp_path, is_ignored):
    before = _models_snapshot()
    res = _fit(frames, tmp_path, is_ignored=is_ignored)
    assert res.path.is_file() and res.path.parent == tmp_path
    assert any(c == res.path or c.parent == res.path.parent for c in is_ignored.calls)
    assert REPO / "models" not in res.path.parents
    assert _models_snapshot() == before


def test_non_ignored_path_writes_nothing(frames, tmp_path, is_ignored):
    is_ignored.ignored = False
    with pytest.raises(WriteGuardError):
        _fit(frames, tmp_path, is_ignored=is_ignored)
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------- live compatibility
_CLEAN = r"""
import json, sys, joblib
bundle = joblib.load(sys.argv[1])
feats = json.loads(sys.argv[2])
from fraud_detect.nodes import ml_statement
bad = sorted(m for m in sys.modules if m.startswith("fraud_detect.ml.retraining"))
assert not bad, bad
ok = ml_statement.assess(feats, bundle)
far = dict(feats, n_pages=10_000.0)
guarded = ml_statement.assess(far, bundle)
off = ml_statement.assess(far, dict(bundle, size_ranges={}))
print(json.dumps({"prob": ok["prob"], "abstain": ok["abstain"], "guard_abstain": guarded["abstain"],
                  "outside": guarded["outside"], "off_abstain": off["abstain"], "off_prob": off["prob"]}))
"""


def test_loads_in_clean_interpreter_and_assess_honours_size_ranges(frames, candidate):
    _, train, _ = frames
    row = train.iloc[0]
    feats = {c: float(row[c]) for c in sf.FEATURE_NAMES}
    # keep the probe inside the guard so that only the n_pages change trips it
    for f, (lo, hi) in candidate.bundle["size_ranges"].items():
        feats[f] = min(max(feats[f], lo), hi)
    out = subprocess.run([sys.executable, "-c", _CLEAN, str(candidate.path), json.dumps(feats)],
                         capture_output=True, text=True, cwd=REPO, timeout=180)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["abstain"] is False and 0.0 <= r["prob"] <= 1.0
    assert r["guard_abstain"] is True and "n_pages" in r["outside"]
    assert r["off_abstain"] is False and 0.0 <= r["off_prob"] <= 1.0


# ----------------------------------------------------------------------------- ablation
def test_all11_ablation_is_informational(frames, tmp_path):
    res = _fit(frames, tmp_path, name="all11.joblib", variant="all11")
    b = res.bundle
    assert b["feature_names"] == list(sf.FEATURE_NAMES) and len(b["feature_names"]) == 11
    assert b["informational"] is True
    assert tb.ABLATION_NOTE == "not used by the Switch_Rule" and tb.ABLATION_NOTE in b["note"]
    assert b["n_train"] == len(frames[1])
    audited = _fit(frames, tmp_path, name="audited.joblib")
    assert not audited.bundle.get("informational")


# ----------------------------------------------------------------------------- refusal
def _result(gate_id, passed, waivable, waived_by=None):
    return gates.GateResult(gate_id, "5.x", passed, () if passed else ("offender",), waivable, waived_by)


def _leak(blocked):
    return LeakResult(rows=(), simple={}, suspects=(), decided=(), forgery="skipped",
                      forgery_flagged=(), notice="", blocked=blocked,
                      exit_code=3 if blocked else 0)


def test_refused_while_integrity_gate_open(frames, tmp_path):
    with pytest.raises(tb.TrainRefused) as e:
        _fit(frames, tmp_path, gate_results=[_result("layout-check", False, False)])
    assert "layout-check" in e.value.open_items and e.value.exit_code == gates.EXIT_INTEGRITY
    assert list(tmp_path.iterdir()) == []


def test_refused_while_waivable_gate_unwaived_but_allowed_when_waived(frames, tmp_path):
    with pytest.raises(tb.TrainRefused) as e:
        _fit(frames, tmp_path, gate_results=[_result("capability", False, True)])
    assert e.value.open_items == ("capability",) and e.value.exit_code == gates.EXIT_WAITING
    assert list(tmp_path.iterdir()) == []
    res = _fit(frames, tmp_path, gate_results=[_result("capability", True, True, waived_by="D-0001"),
                                               _result("layout-check", True, False)])
    assert res.path.is_file() and not res.failed


def test_refused_while_leak_suspect_open(frames, tmp_path):
    with pytest.raises(tb.TrainRefused) as e:
        _fit(frames, tmp_path, leak=_leak(True))
    assert e.value.exit_code == gates.EXIT_WAITING
    assert list(tmp_path.iterdir()) == []
    assert _fit(frames, tmp_path, leak=_leak(False)).path.is_file()


# ----------------------------------------------------------------------------- failed marker
def test_candidate_that_fails_to_score_gets_failed_marker(frames, tmp_path):
    def broken(bundle, feats):
        raise RuntimeError("cannot score")

    res = _fit(frames, tmp_path, verify=broken)
    assert res.failed is True
    assert res.path.is_file()
    assert res.marker_path == Path(str(res.path) + ".failed") and res.marker_path.is_file()
    ok = _fit(frames, tmp_path, name="ok.joblib")
    assert ok.failed is False and ok.marker_path is None
    assert not Path(str(ok.path) + ".failed").exists()


# ----------------------------------------------------------------------------- run: per seed and pooled
def test_train_run_per_seed_and_pooled_release(tables, tmp_path, capsys):
    out = tb.train_run(tables, held_out_layout="two_col_ddmmyy_cr", out_dir=tmp_path, rule=RULE,
                       is_ignored=lambda p: True)
    seeds = sorted(tables)
    assert sorted(out.per_seed) == seeds and sorted(out.ablation) == seeds
    for s in seeds:
        assert out.per_seed[s].path == tmp_path / "models" / "6b" / f"seed-{s}.joblib"
        assert out.ablation[s].path == tmp_path / "models" / "6b" / f"seed-{s}-all11.joblib"
        assert out.per_seed[s].bundle["master_seed"] == s
        assert out.ablation[s].bundle["informational"] is True
    rel = out.release
    assert rel.path == tmp_path / "release" / "6b.joblib" and rel.path.is_file()
    assert out.release_ablation.path == tmp_path / "release" / "6b-all11.joblib"
    pooled_train = pd.concat([t[t["split"] == "train"] for t in tables.values()])
    pooled_valid = pd.concat([t[t["split"] == "validation"] for t in tables.values()])
    b = rel.bundle
    assert b["master_seed"] is None and b["pooled_seeds"] == seeds
    assert b["n_train"] == len(pooled_train)
    scores = b["model"].predict_proba(pooled_valid[b["feature_names"]])[:, 1]
    cut = choose_cutoffs(scores, ~pooled_valid["is_tampered"].astype(bool).to_numpy(), RULE)
    assert b["thresholds"] == {"medium": cut.medium, "high": cut.high}
    text = capsys.readouterr().out
    assert tb.GNN_LINE == "GNN: not retrained, corpus has no transfer graph"
    assert tb.GNN_LINE in text and tb.GNN_LINE in out.report_lines
    assert "medium cutoff" in text and "high cutoff" in text  # validation operating curve printed
    assert hashlib.sha256(rel.path.read_bytes()).hexdigest()  # file is real and readable
