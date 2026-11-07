"""Task 1.2: failing tests for fraud_detect.ml.retraining.config (written before the module)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from fraud_detect.ml.retraining import config as cfg_mod
from fraud_detect.ml.retraining.config import AWKWARD_MASTER_SEED, RunConfig, Thresholds

DEFAULT_THRESHOLDS = {
    "leak_auc": 0.98,
    "forgery_auc_gap": 0.20,
    "style_share_points": 10,
    "style_min_docs": 30,
    "resaved_min_share": 15,
    "type_min_docs": 20,
    "subtle_max_relative_change": 5,
    "min_seeds": 5,
    "indicative_min_docs": 30,
    "skip_max_share": 2,
    "row_exclusion_max_share": 1,
    "implausible_auc": 0.99,
    "guard_condition_points": 5,
    "hard_negative_block_points": 10,
    "target_fa_medium": 5.0,
    "target_fa_high": 1.0,
}


def test_run_config_is_frozen():
    cfg = RunConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.held_out_layout = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.thresholds.leak_auc = 0.5  # type: ignore[misc]


def test_canonical_json_is_sorted_and_stable():
    text = RunConfig().canonical_json()
    assert text == RunConfig().canonical_json()
    parsed = json.loads(text)
    assert text == json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) or \
        text == json.dumps(parsed, sort_keys=True)
    assert list(parsed) == sorted(parsed)
    assert list(parsed["thresholds"]) == sorted(parsed["thresholds"])


def test_run_id_is_sha256_prefix_of_canonical_json():
    cfg = RunConfig()
    digest = hashlib.sha256(cfg.canonical_json().encode("utf-8")).hexdigest()
    assert digest.startswith(cfg.run_id)
    assert len(cfg.run_id) >= 8
    changed = dataclasses.replace(cfg, held_out_layout="generator_v1")
    assert changed.run_id != cfg.run_id


@pytest.mark.parametrize("name,value", sorted(DEFAULT_THRESHOLDS.items()))
def test_threshold_defaults(name, value):
    assert getattr(Thresholds(), name) == value
    assert getattr(RunConfig().thresholds, name) == value


def test_threshold_set_is_exactly_the_design_table():
    assert {f.name for f in dataclasses.fields(Thresholds)} == set(DEFAULT_THRESHOLDS)


def test_default_seeds_and_layout():
    cfg = RunConfig()
    assert tuple(cfg.seeds) == (20260801, 20260802, 20260803, 20260804, 20260805)
    assert cfg.held_out_seed == 20260899
    assert cfg.held_out_layout == "two_col_ddmmyy_cr"


def test_real_style_role_default_and_veto_only():
    assert RunConfig().real_style_role == "required"
    assert RunConfig(real_style_role="veto_only").real_style_role == "veto_only"


def test_node6a_baseline_path_default_and_override():
    assert Path(RunConfig().node6a_baseline_path) == Path("models/txn_fraud_clf.joblib")
    alt = RunConfig(node6a_baseline_path="models/txn_fraud_clf_pdf.joblib")
    assert Path(alt.node6a_baseline_path) == Path("models/txn_fraud_clf_pdf.joblib")


def test_real_style_and_forgery_paths_overridable(tmp_path):
    cfg = RunConfig(real_style_dir=str(tmp_path / "rs"), forgery_dir=str(tmp_path / "fg"))
    assert Path(cfg.real_style_dir) == tmp_path / "rs"
    assert Path(cfg.forgery_dir) == tmp_path / "fg"


def test_awkward_seed_differs_from_every_configured_seed():
    cfg = RunConfig()
    assert AWKWARD_MASTER_SEED == 20260701
    assert AWKWARD_MASTER_SEED not in set(cfg.seeds) | {cfg.held_out_seed}


def test_default_output_paths_are_under_private_folder():
    assert Path(cfg_mod.DEFAULT_OUTPUT_ROOT) == Path("data/private/ml_retraining")


def test_import_pulls_no_heavy_modules():
    import subprocess
    import sys
    code = ("import sys, fraud_detect.ml.retraining.config;"
            "bad=[m for m in ('torch','sklearn','pdfplumber') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
