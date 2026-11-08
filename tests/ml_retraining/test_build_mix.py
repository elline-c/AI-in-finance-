"""The build command passes the design tamper mix (found at full size: without it only ~6% of genuine
documents were re-saved, and the 15% re-saved gate failed in every Split). The mix has 17 types
(16 original + subtle_amount_edit), genuine_resaved 10 parts of 26, and the build uses Balanced_Styling."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import ALL_TAMPER_TYPES, BuildConfig
from fraud_detect.ml.retraining.runner import N_LAYOUTS, design_tamper_mix, plan_counts

# fixtures and constants of the CLI tests (the Env helper and its fixtures live there)
from .test_cli import HELD_OUT_SEED, MINI_HOLD_OUT_LAYOUT, TRAIN_SEED, env, shared_cache  # noqa: F401


def test_mix_has_all_types_and_resaved_is_10_of_26_parts():
    mix = dict(design_tamper_mix())
    assert set(mix) == set(ALL_TAMPER_TYPES) and len(mix) == 17
    assert mix["genuine_resaved"] == 10.0
    assert all(w == 1.0 for t, w in mix.items() if t != "genuine_resaved")
    assert sum(mix.values()) == 26.0
    assert mix["genuine_resaved"] / sum(mix.values()) == 10 / 26


def test_dealt_counts_match_the_plan_and_clear_the_resaved_gate(tmp_path):
    cfg = BuildConfig(out_dir=tmp_path / "c", master_seed=1, per_layout_genuine=80,
                      per_layout_tampered=80, tamper_mix=design_tamper_mix(),
                      command_line="pytest", style_version=2)
    types = CorpusBuilder(cfg)._tamper_types(80 * 6)
    plan = plan_counts(80, 80, 5)
    resaved = types.count("genuine_resaved")
    assert plan["genuine_resaved per seed"] == 184
    assert plan["other tamper types per seed"] == 18
    # quotas are rounded (largest remainder): the plan is the rounded-down count
    assert plan["genuine_resaved per seed"] <= resaved <= plan["genuine_resaved per seed"] + 1
    other = plan["other tamper types per seed"]
    assert all(other <= types.count(t) <= other + 1 for t in ALL_TAMPER_TYPES if t != "genuine_resaved")
    # genuine_resaved documents are genuine: share of all genuine documents
    assert resaved / (80 * 6 + resaved) >= 0.15


@settings(max_examples=100, deadline=None)
@given(per_genuine=st.integers(0, 500), per_tampered=st.integers(0, 500), n_seeds=st.integers(1, 12))
def test_plan_counts_follow_the_26_part_mix(per_genuine, per_tampered, n_seeds):
    """Feature: generator-styling-variety, Property 17: Plan counts follow the 26-part mix.

    **Validates: Requirements 12.2, 12.3**"""
    plan = plan_counts(per_genuine, per_tampered, n_seeds)
    tampered = per_tampered * N_LAYOUTS
    assert plan["documents per seed"] == N_LAYOUTS * (per_genuine + per_tampered)
    assert plan["corpora"] == n_seeds + 1
    assert plan["documents in total"] == plan["documents per seed"] * (n_seeds + 1)
    assert plan["genuine_resaved per seed"] == tampered * 10 // 26
    assert plan["other tamper types per seed"] == tampered // 26
    # the 17 types together never exceed the tampered slots
    assert plan["genuine_resaved per seed"] + 16 * plan["other tamper types per seed"] <= tampered


def test_build_passes_style_version_2_for_every_seed_and_the_held_out_seed(env):
    """cmd_build builds every corpus with Balanced_Styling. **Validates: Requirements 12.1**"""
    assert env.invoke("init", "--seeds", f"{TRAIN_SEED},{HELD_OUT_SEED}", "--held-out-seed", "20260899",
                      "--held-out-layout", MINI_HOLD_OUT_LAYOUT).code == 0
    calls = []
    res = env.invoke("build", "--per-layout-genuine", "2", "--per-layout-tampered", "2",
                     corpus_builder=lambda cfg: calls.append(cfg))
    assert res.code == 0, res.out
    assert len(calls) == 3
    assert 20260899 in {c.master_seed for c in calls}
    assert all(c.style_version == 2 for c in calls)
    assert all(set(dict(c.tamper_mix)) == set(ALL_TAMPER_TYPES) for c in calls)
