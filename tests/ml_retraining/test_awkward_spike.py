"""Task 2.1: failing tests for the Awkward_Set spike (module is built in task 2.2).

``run_spike(seed)`` measures, for each of five case types, the figures the Awkward_Set
needs, using only the public ``fraud_detect.corpus`` API. Result keys used here:

    seasonal_growth   : peak_trough_ratio, last3_first3_ratio
    instalments       : n_invoice_refs_on_two_receipts
    long_12_months    : n_months
    round_payroll_rent: round_share, min_running_balance
    legit_resave      : producer_is_export (1 or 0), moddate_gap_seconds

All values are plain numbers. The module is imported inside the tests so a missing
module fails these tests only and never breaks collection of the other test files.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import json
import numbers
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = "fraud_detect.ml.retraining.awkward_spike"
MODULE_FILE = REPO / "fraud_detect" / "ml" / "retraining" / "awkward_spike.py"
CASE_TYPES = ("seasonal_growth", "instalments", "long_12_months",
              "round_payroll_rent", "legit_resave")
FIGURES = {
    "seasonal_growth": ("peak_trough_ratio", "last3_first3_ratio"),
    "instalments": ("n_invoice_refs_on_two_receipts",),
    "long_12_months": ("n_months",),
    "round_payroll_rent": ("round_share", "min_running_balance"),
    "legit_resave": ("producer_is_export", "moddate_gap_seconds"),
}
# Public corpus API the spike may import from (no builder, finish, layouts, seeding).
ALLOWED_CORPUS_MODULES = {
    "fraud_detect.corpus",
    "fraud_detect.corpus.company",
    "fraud_detect.corpus.ledger",
    "fraud_detect.corpus.schema",
    "fraud_detect.corpus.tamper.catalog",
    "fraud_detect.corpus.tamper.pdf_ops",
}


def _spike():
    return importlib.import_module(MODULE)


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real)


def test_returns_five_case_types_with_plain_numbers():
    result = _spike().run_spike(20260701)
    assert set(result) == set(CASE_TYPES)
    for case, names in FIGURES.items():
        assert set(names) <= set(result[case]), case
        for name in names:
            assert _is_number(result[case][name]), (case, name)
    json.dumps(result)  # plain numbers: serialisable without custom encoders


def test_figures_are_sane():
    r = _spike().run_spike(20260701)
    sg = r["seasonal_growth"]
    assert sg["peak_trough_ratio"] >= 1.0
    assert sg["last3_first3_ratio"] > 0.0
    assert r["instalments"]["n_invoice_refs_on_two_receipts"] >= 1
    assert r["long_12_months"]["n_months"] == 12
    rp = r["round_payroll_rent"]
    assert 0.8 <= rp["round_share"] <= 1.0
    assert rp["min_running_balance"] >= 0.0  # opening balance raised after the rewrite
    lr = r["legit_resave"]
    assert lr["producer_is_export"] == 1
    assert 60 <= lr["moddate_gap_seconds"] <= 3 * 86400


@pytest.mark.parametrize("seed", [1, 20260701])
def test_deterministic_for_fixed_seed(seed):
    spike = _spike()
    assert spike.run_spike(seed) == spike.run_spike(seed)


def test_seed_changes_the_measurement():
    spike = _spike()
    assert spike.run_spike(1) != spike.run_spike(2)


def test_writes_nothing_to_disk(tmp_path, monkeypatch):
    spike = _spike()
    spike.run_spike(7)  # warm imports so only the measured call is guarded
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            raise AssertionError(f"run_spike tried to open {file!r} for writing")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("mkdir", "makedirs"):
        monkeypatch.setattr(os, name, lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("run_spike created a directory")))
    spike.run_spike(7)
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_imports_only_public_corpus_api():
    assert MODULE_FILE.exists(), "awkward_spike.py not implemented yet (task 2.2)"
    tree = ast.parse(MODULE_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        elif isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
            for m in mods:
                assert m not in {"torch", "torch_geometric"}, m
                if m.startswith("fraud_detect.corpus"):
                    assert m in ALLOWED_CORPUS_MODULES, m
            continue
        else:
            continue
        if mod.startswith("fraud_detect.corpus"):
            assert mod in ALLOWED_CORPUS_MODULES, f"internal corpus import: {mod}"
            for alias in node.names:
                assert not alias.name.startswith("_"), f"private name: {alias.name}"
        assert mod.split(".")[0] not in {"torch", "torch_geometric"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def test_spike_modifies_no_corpus_file():
    try:
        tracked = {}
        for line in _git("ls-tree", "-r", "HEAD", "fraud_detect/corpus").splitlines():
            meta, path = line.split("\t", 1)
            tracked[path] = meta.split()[2]
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert tracked, "no tracked corpus files found"
    _spike().run_spike(20260701)
    for path, blob in tracked.items():
        assert _git("hash-object", path).strip() == blob, f"{path} differs from git HEAD"
    assert _git("status", "--porcelain", "--", "fraud_detect/corpus").strip() == ""
