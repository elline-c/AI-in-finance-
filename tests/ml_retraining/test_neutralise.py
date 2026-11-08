"""Tests for the neutralisation toolbox (task 8.1; written before the module exists).

Expected API of ``fraud_detect.ml.retraining.neutralise`` (task 8.2):

* ``make_form(spec)``: parse a recorded form string and return an unfitted single-column
  scikit-learn transformer. Supported specs: ``"clip(lo, hi)"`` (``FunctionTransformer``
  around ``np.clip``) and ``"binarise(threshold)"`` (``Binarizer``).
* ``NeutraliseError``: raised for a spec outside the toolbox (a feature with no
  single-column stable form is not accepted). It is a ``ValueError`` subclass.

The transformers pickle only as scikit-learn / NumPy references so the live Node 6b can
load a saved model without importing ``fraud_detect.ml.retraining`` (design D5).

**Validates: Requirements 3.5, 10.2**
"""
from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("fraud_detect.ml.retraining.neutralise")


def _col(values):
    return np.asarray(values, dtype=float).reshape(-1, 1)


def test_clip_collapses_values_at_or_above_upper_bound(mod):
    tf = mod.make_form("clip(0, 0.10)")
    out = tf.fit_transform(_col([0.0, 0.03, 0.0999, 0.10, 0.11, 0.25, 0.9, 1.0])).ravel()
    assert list(out[3:]) == [0.10] * 5
    assert len(set(out[3:])) == 1


def test_clip_keeps_values_below_upper_bound_graded(mod):
    tf = mod.make_form("clip(0, 0.10)")
    values = [0.0, 0.01, 0.04, 0.0999]
    out = tf.fit_transform(_col(values)).ravel()
    assert list(out) == values


def test_clip_equals_numpy_clip(mod):
    tf = mod.make_form("clip(0, 0.10)")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(50, 1))
    assert np.array_equal(tf.fit_transform(x), np.clip(x, 0, 0.10))


def test_binarise_form(mod):
    tf = mod.make_form("binarise(0.5)")
    out = tf.fit_transform(_col([0.0, 0.5, 0.51, 3.0])).ravel()
    assert list(out) == [0.0, 0.0, 1.0, 1.0]


def test_is_scikit_learn_transformer_with_one_column(mod):
    from sklearn.base import TransformerMixin
    tf = mod.make_form("clip(0, 0.10)")
    assert isinstance(tf, TransformerMixin)
    assert tf.fit_transform(_col([0.2, 0.3])).shape == (2, 1)


@pytest.mark.parametrize("spec", [
    "log1p()",                  # not in the toolbox
    "ratio(a, b)",              # two-column form: not a single-column stable form
    "clip(0)",                  # malformed
    "clip(a, b)",               # non-numeric bounds
    "clip(0.5, 0.1)",           # lower bound above upper bound
    "",
    "n_distinct_fonts",         # a feature name is not a form
])
def test_specs_outside_the_toolbox_are_refused(mod, spec):
    with pytest.raises(mod.NeutraliseError):
        mod.make_form(spec)


def test_neutralise_error_is_value_error(mod):
    assert issubclass(mod.NeutraliseError, ValueError)


def test_pickle_references_only_sklearn_and_numpy(mod):
    import pickletools
    tf = mod.make_form("clip(0, 0.10)").fit(_col([0.0, 1.0]))
    data = pickle.dumps(tf)
    assert b"fraud_detect" not in data
    modules = set()
    strings = []
    for op, arg, _ in pickletools.genops(data):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            strings.append(arg)
        elif op.name == "GLOBAL":
            modules.add(arg.split(" ")[0])
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            modules.add(strings[-2])
    assert modules, "expected at least one class or function reference in the pickle"
    assert all(m.split(".")[0] in {"sklearn", "numpy", "builtins", "copyreg", "collections"}
               for m in modules), modules


@pytest.mark.parametrize("spec", ["clip(0, 0.10)", "binarise(0.5)"])
def test_loads_in_clean_interpreter_without_retraining_package(mod, tmp_path, spec):
    tf = mod.make_form(spec).fit(_col([0.0, 0.2, 1.0]))
    path = tmp_path / "form.pkl"
    path.write_bytes(pickle.dumps(tf))
    code = (
        "import pickle, sys, numpy as np\n"
        f"tf = pickle.load(open({str(path)!r}, 'rb'))\n"
        "out = tf.transform(np.array([[0.05], [0.7]]))\n"
        "assert out.shape == (2, 1)\n"
        "assert not any(m.startswith('fraud_detect.ml.retraining') for m in sys.modules), "
        "[m for m in sys.modules if m.startswith('fraud_detect.ml.retraining')]\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_the_module_pulls_in_no_heavy_dependencies():
    code = (
        "import sys\n"
        "import fraud_detect.ml.retraining.neutralise\n"
        "assert 'torch' not in sys.modules and 'torch_geometric' not in sys.modules\n"
        "assert 'pdfplumber' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
