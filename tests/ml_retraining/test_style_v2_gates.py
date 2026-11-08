"""Gate simulation for Balanced_Styling (generator-styling-variety, task 8.1).

Builds style tables from ``make_style_v2`` (no PDFs rendered) and runs the ml-retraining
corpus gates on them. Master seeds are fixed. A failing table is a finding to report;
do not change a seed, weight or threshold to make it pass.

Synthetic data only.

**Validates: Requirements 2.6, 11.1, 11.2, 11.3, 11.5**
"""
from __future__ import annotations

import dataclasses
from collections import Counter

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import ALL_TAMPER_TYPES, BuildConfig
from fraud_detect.corpus.styling import make_style_v2
from fraud_detect.ml.retraining.corpus_gates import (
    capability_report,
    resaved_share_gate,
    styling_balance,
)

N_PER_CLASS = 480
N_BALANCE_TABLES = 300
N_CAPABILITY_TABLES = 20
MAX_FAILING_BALANCE_TABLES = 6
MIN_FAMILY_SHARE = 4.0          # percent of each class
MIN_DOCS = 30                   # styling_balance thresholds (Thresholds defaults)
SHARE_POINTS = 10
RESAVED_MIN_SHARE = 15
BASE_SEED = 20270000            # table i uses Master_Seed BASE_SEED + i
HOLD_OUT = "two_col_ddmmyy_cr"


def _style_row(master_seed: int, doc_id: str, tampered: bool) -> dict:
    s = make_style_v2(master_seed, doc_id)
    return {
        "is_tampered": tampered,
        "style_producer": s.producer,
        "style_producer_family": s.producer_family,
        "style_creator": s.creator,
        "style_fonts": tuple(sorted({s.font_regular, s.font_bold, "Helvetica"})),
        "style_bank_name": s.bank_name,
        "style_street": s.address_lines[0],
        "style_row_step_bucket": round(s.row_step, 1),
        "style_creation_hour": int(s.creation_time.split(":")[0]),
    }


def style_table(master_seed: int) -> list:
    rows = [_style_row(master_seed, f"g-{k:04d}", False) for k in range(N_PER_CLASS)]
    rows += [_style_row(master_seed, f"t-{k:04d}", True) for k in range(N_PER_CLASS)]
    return rows


def test_styling_balance_over_300_tables():
    failing = []
    for i in range(N_BALANCE_TABLES):
        seed = BASE_SEED + i
        found = styling_balance(style_table(seed), min_docs=MIN_DOCS, share_points=SHARE_POINTS)
        if found:
            failing.append((seed, [f"{f.column}={f.value}" for f in found]))
    assert len(failing) <= MAX_FAILING_BALANCE_TABLES, (
        f"{len(failing)} of {N_BALANCE_TABLES} tables fail styling_balance: {failing}")


def test_capability_report_empty_over_20_tables():
    bad = {}
    for i in range(N_CAPABILITY_TABLES):
        seed = BASE_SEED + 1000 + i
        report = capability_report(style_table(seed))
        if report:
            bad[seed] = report
    assert not bad, f"capability_report not empty: {bad}"


def test_each_producer_family_at_least_4_percent_of_each_class():
    bad = []
    for i in range(N_BALANCE_TABLES):
        seed = BASE_SEED + i
        rows = style_table(seed)
        for tampered in (False, True):
            fams = Counter(r["style_producer_family"] for r in rows if r["is_tampered"] == tampered)
            total = sum(fams.values())
            for fam in ("base", "export", "editor"):
                share = 100.0 * fams[fam] / total
                if share < MIN_FAMILY_SHARE:
                    bad.append((seed, "tampered" if tampered else "genuine", fam, round(share, 2)))
    assert not bad, f"Producer family below {MIN_FAMILY_SHARE}% of a class: {bad}"


def _mix_26() -> tuple:
    """16 original types and subtle_amount_edit: genuine_resaved 10 parts, others 1 (26 parts).
    Built here because runner.design_tamper_mix changes only in task 9.1."""
    mix = tuple((t, 10.0 if t == "genuine_resaved" else 1.0) for t in ALL_TAMPER_TYPES)
    assert len(ALL_TAMPER_TYPES) == 17 and sum(w for _, w in mix) == 26.0
    return mix


def test_resaved_share_gate_with_26_part_mix(tmp_path):
    seeds = [20260801, 20260802, 20260803, 20260804, 20260805]
    cfg0 = BuildConfig(out_dir=tmp_path, master_seed=seeds[0], per_layout_genuine=80,
                       per_layout_tampered=80, hold_out_layout=HOLD_OUT,
                       tamper_mix=_mix_26(), style_version=2)
    for seed in seeds:
        cfg = dataclasses.replace(cfg0, master_seed=seed)
        plans = CorpusBuilder(cfg).plan()
        rows = [{"split": p.split,
                 "is_tampered": p.tamper_type is not None and p.tamper_type != "genuine_resaved",
                 "genuine_resaved": p.tamper_type == "genuine_resaved"}
                for p in plans if p.split in ("train", "validation", "test")]
        assert {r["split"] for r in rows} == {"train", "validation", "test"}
        result = resaved_share_gate(rows, min_share=RESAVED_MIN_SHARE)
        assert result.passed, f"seed {seed}: {result.offenders}"
