"""Task 15.3: property test for the audit-decides-input rule (Node 6b candidates).

**Property 12: The audit decides the model input exactly**
**Validates: Requirements 3.4, 3.5, 3.7, 10.2, 10.7, 11.1**

Tiny synthetic frames and few examples: every example fits a real HistGradientBoosting pipeline.
Files go to pytest temporary folders only; nothing under ``models/`` is touched.
"""
from __future__ import annotations

import dataclasses
import io

import joblib
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.ml import statement_features as sf
from fraud_detect.ml.retraining import train_b as tb
from fraud_detect.ml.retraining.audit import AUDIT_V1, model_input_columns
from fraud_detect.ml.retraining.cutoffs import CutoffRule

RULE = CutoffRule(target_fa_medium=100.0, target_fa_high=100.0)  # loose: random tiny data
NAMES_6B = [e.name for e in AUDIT_V1.entries if e.section == "model_feature" and e.node == "6b"]

forms = st.one_of(
    st.tuples(st.floats(0, 1), st.floats(0, 3)).map(
        lambda t: ("clip", f"clip({min(t):.3f}, {max(t) + 0.001:.3f})")),
    st.floats(0, 1).map(lambda t: ("binarise", f"binarise({t:.3f})")),
)
decision_st = st.one_of(st.just(("keep", None)), st.just(("drop", None)),
                        forms.map(lambda f: ("neutralise", f[1])))


@st.composite
def scenarios(draw):
    decisions = draw(st.lists(decision_st, min_size=len(NAMES_6B), max_size=len(NAMES_6B)))
    if all(d == "drop" for d, _ in decisions):
        decisions[draw(st.integers(0, len(NAMES_6B) - 1))] = ("keep", None)
    seed = draw(st.integers(0, 10_000))
    return decisions, seed


def _audit(decisions):
    by_name = dict(zip(NAMES_6B, decisions))
    entries = tuple(
        dataclasses.replace(e, decision=by_name[e.name][0], neutralised_form=by_name[e.name][1])
        if (e.section == "model_feature" and e.node == "6b") else e
        for e in AUDIT_V1.entries)
    return dataclasses.replace(AUDIT_V1, version="audit-prop", entries=entries)


def _frames(seed):
    rng = np.random.default_rng(seed)
    n = 24
    frame = pd.DataFrame({c: rng.uniform(0, 4, n) for c in sf.FEATURE_NAMES})
    frame["is_tampered"] = np.tile([False, True], n // 2)
    frame["split"] = "train"
    train = frame.iloc[:16].copy()
    valid = frame.iloc[16:].copy()
    valid["split"] = "validation"
    return train, valid


def _expected(value: pd.Series, form: str) -> np.ndarray:
    kind, arg = form.split("(")
    nums = [float(x) for x in arg.rstrip(")").split(",")]
    v = value.to_numpy(float)
    if kind == "clip":
        return np.clip(v, nums[0], nums[1])
    return (v > nums[0]).astype(float)


@settings(max_examples=6, deadline=None, derandomize=True,
          suppress_health_check=list(HealthCheck))
@given(scenarios())
def test_audit_decides_model_input_exactly(tmp_path_factory, scenario):
    decisions, seed = scenario
    audit = _audit(decisions)
    train, valid = _frames(seed)
    out = tmp_path_factory.mktemp("p12") / "cand.joblib"
    res = tb.train_candidate(train, valid, master_seed=seed, held_out_layout="x", out_path=out,
                             audit=audit, rule=RULE, is_ignored=lambda p: True,
                             verify=lambda b, f: None)

    expected = [n for n, (d, _) in zip(NAMES_6B, decisions) if d != "drop"]
    dropped = [n for n, (d, _) in zip(NAMES_6B, decisions) if d == "drop"]
    assert expected == model_input_columns(audit, "6b")

    # the saved file, once loaded, carries the ordered list, audit version and digest
    loaded = joblib.load(out)
    for bundle in (res.bundle, loaded):
        assert bundle["feature_names"] == expected
        assert list(bundle["model"].feature_names_in_) == expected
        assert not set(dropped) & set(bundle["feature_names"])
        assert bundle["audit_version"] == audit.version
        assert bundle["audit_digest"] == audit.digest()

    # the transformed model input: keep columns identical, neutralised columns = toolbox function
    prep = loaded["model"].named_steps["prep"]
    out_arr = prep.transform(train[expected].astype(float))
    neutral = [n for n, (d, _) in zip(NAMES_6B, decisions) if d == "neutralise"]
    plain = [n for n in expected if n not in neutral]
    assert out_arr.shape[1] == len(expected)
    for i, n in enumerate(plain):
        np.testing.assert_array_equal(out_arr[:, i], train[n].to_numpy(float))
    forms_by_name = {n: f for n, (d, f) in zip(NAMES_6B, decisions) if d == "neutralise"}
    for j, n in enumerate(neutral):
        col = out_arr[:, len(plain) + j]
        np.testing.assert_array_equal(col, _expected(train[n], forms_by_name[n]))
        if forms_by_name[n].startswith("clip"):
            hi = float(forms_by_name[n].rstrip(")").split(",")[1])
            above = train[n].to_numpy(float) >= hi
            assert len(set(col[above])) <= 1  # all values at or above the bound collapse
