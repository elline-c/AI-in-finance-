"""Task 6.1: failing tests for ``fraud_detect.ml.retraining.corpus_gates``.

Gate logic runs on synthetic in-memory tables (lists of dicts). Thresholds are passed as small
test values. Style columns carry the ``style_`` prefix used by the Node 6b feature table.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os

import pytest

from fraud_detect.ml.retraining import corpus_gates as cg
from fraud_detect.ml.retraining.gates import GateResult


# ----------------------------------------------------------------------------- helpers
def _doc(i, *, tampered=False, split="train", resaved=False, types=(), corpus="seed-1", **style):
    row = {"corpus_key": corpus, "doc_id": f"D{i:04d}", "split": split,
           "is_tampered": tampered, "genuine_resaved": resaved,
           "tamper_types": tuple(types)}
    row.update(style)
    return row


def _offender_text(result: GateResult) -> str:
    return " | ".join(result.offenders)


# ----------------------------------------------------------------------- styling balance
def _bank_rows(g_a, g_b, t_a, t_b):
    rows, i = [], 0
    for value, n, tampered in (("A", g_a, False), ("B", g_b, False), ("A", t_a, True), ("B", t_b, True)):
        for _ in range(n):
            rows.append(_doc(i, tampered=tampered, style_bank_name=value))
            i += 1
    return rows


def test_styling_balance_lists_exactly_the_failing_values():
    # A: genuine 80%, tampered 50% -> 30 points apart. B: 20% vs 50% -> 30 points apart.
    rows = _bank_rows(8, 2, 5, 5)
    failures = cg.styling_balance(rows, min_docs=5, share_points=10)
    assert {(f.column, f.value) for f in failures} == {("style_bank_name", "A"), ("style_bank_name", "B")}
    by_value = {f.value: f for f in failures}
    assert by_value["A"].genuine_share == pytest.approx(80.0)
    assert by_value["A"].tampered_share == pytest.approx(50.0)
    assert by_value["B"].genuine_share == pytest.approx(20.0)
    assert by_value["B"].tampered_share == pytest.approx(50.0)


def test_styling_balance_difference_exactly_at_limit_passes():
    rows = _bank_rows(6, 4, 5, 5)  # 60% vs 50%: exactly 10 points
    assert cg.styling_balance(rows, min_docs=5, share_points=10) == []


def test_styling_balance_ignores_values_below_min_docs():
    rows = _bank_rows(9, 1, 5, 5)  # B has 6 documents; A has 14
    rows_min_7 = cg.styling_balance(rows, min_docs=7, share_points=10)
    assert {f.value for f in rows_min_7} == {"A"}
    assert cg.styling_balance(rows, min_docs=20, share_points=10) == []


def test_styling_balance_gate_result_and_unchanged_corpus_files(tmp_path):
    corpus_file = tmp_path / "labels.csv"
    corpus_file.write_bytes(b"doc_id,split\nD0000,train\n")
    before = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    result = cg.styling_balance_gate(_bank_rows(8, 2, 5, 5), min_docs=5, share_points=10)
    assert isinstance(result, GateResult)
    assert result.passed is False and result.waivable is False
    text = _offender_text(result)
    assert "style_bank_name" in text and "A" in text and "80" in text and "50" in text
    assert hashlib.sha256(corpus_file.read_bytes()).hexdigest() == before


def test_styling_balance_gate_passes_on_balanced_table():
    result = cg.styling_balance_gate(_bank_rows(5, 5, 5, 5), min_docs=5, share_points=10)
    assert result.passed is True and result.offenders == ()


# -------------------------------------------------------------------------- resaved share
def _split_rows(split, n_plain, n_resaved, start=0):
    rows = [_doc(start + i, split=split) for i in range(n_plain)]
    rows += [_doc(start + n_plain + i, split=split, resaved=True) for i in range(n_resaved)]
    return rows


def test_resaved_share_names_split_and_share_when_below_minimum():
    rows = (_split_rows("train", 8, 2) + _split_rows("validation", 17, 3, 100)
            + _split_rows("test", 18, 2, 200))
    result = cg.resaved_share_gate(rows, min_share=15.0)
    assert result.passed is False and result.waivable is False
    text = _offender_text(result)
    assert "test" in text and "10.0" in text
    assert "train" not in text and "validation" not in text  # 20% and 15%: both ok


def test_resaved_share_exactly_at_minimum_passes():
    rows = _split_rows("train", 17, 3) + _split_rows("validation", 17, 3, 100) + _split_rows("test", 17, 3, 200)
    assert cg.resaved_share_gate(rows, min_share=15.0).passed is True


def test_resaved_share_ignores_tampered_documents_in_denominator():
    rows = _split_rows("train", 8, 2) + [_doc(900 + i, split="train", tampered=True) for i in range(50)]
    rows += _split_rows("validation", 8, 2, 100) + _split_rows("test", 8, 2, 200)
    assert cg.resaved_share_gate(rows, min_share=15.0).passed is True


# ---------------------------------------------------------------------- tamper-type minimum
def _typed(split, kind, n, start, corpus="seed-1"):
    return [_doc(start + i, split=split, tampered=True, types=(kind,), corpus=corpus) for i in range(n)]


def test_type_minimum_lists_types_below_minimum_with_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = _typed("train", "edited_amount", 5, 0) + _typed("train", "structuring", 2, 100)
    rows += _typed("validation", "rare_type", 1, 200)  # not present in train: not checked
    result = cg.tamper_type_minimum_gate(rows, min_docs=3)
    assert result.passed is False and result.waivable is False
    text = _offender_text(result)
    assert "structuring" in text and "2" in text
    assert "edited_amount" not in text and "rare_type" not in text
    assert list(tmp_path.iterdir()) == []  # no completion marker, nothing written


def test_type_minimum_pooled_by_default_across_corpora():
    rows = (_typed("train", "edited_amount", 2, 0, corpus="seed-1")
            + _typed("train", "edited_amount", 2, 100, corpus="seed-2"))
    assert cg.tamper_type_minimum_gate(rows, min_docs=4).passed is True
    assert cg.tamper_type_minimum_gate(rows, min_docs=4, pooled=True).passed is True


def test_type_minimum_per_corpus_when_pooling_is_off():
    rows = (_typed("train", "edited_amount", 2, 0, corpus="seed-1")
            + _typed("train", "edited_amount", 2, 100, corpus="seed-2"))
    result = cg.tamper_type_minimum_gate(rows, min_docs=4, pooled=False)
    assert result.passed is False
    assert "edited_amount" in _offender_text(result) and "seed-1" in _offender_text(result)


def test_type_minimum_counts_only_train_split():
    rows = _typed("test", "edited_amount", 50, 0) + _typed("train", "edited_amount", 1, 100)
    assert cg.tamper_type_minimum_gate(rows, min_docs=3).passed is False


def test_pooled_switch_is_on_by_default_in_run_config():
    from fraud_detect.ml.retraining.config import RunConfig
    cfg = RunConfig()
    assert cfg.pooled_type_minimum is True
    assert dataclasses.replace(cfg, pooled_type_minimum=False).pooled_type_minimum is False


# --------------------------------------------------------------------------------- coverage
def test_coverage_lists_type_split_pairs_with_zero_documents():
    rows = (_typed("train", "a", 2, 0) + _typed("validation", "a", 1, 10) + _typed("test", "a", 1, 20)
            + _typed("train", "b", 2, 30) + _typed("test", "b", 1, 40))
    gaps = cg.coverage_gaps(rows)
    assert set(gaps) == {("b", "validation")}
    result = cg.coverage_gate(rows)
    assert result.passed is False and result.waivable is False
    assert "b" in _offender_text(result) and "validation" in _offender_text(result)


def test_coverage_ignores_unseen_layout_split_and_passes_when_complete():
    rows = _typed("train", "a", 1, 0) + _typed("validation", "a", 1, 10) + _typed("test", "a", 1, 20)
    assert cg.coverage_gaps(rows) == []
    assert cg.coverage_gate(rows).passed is True


# ---------------------------------------------------------------------------------- subtle
def _aff(field, original, new, row_id="R1"):
    return {"row_id": row_id, "row_index": 1, "month": "2026-01",
            "field": field, "original": original, "new": new}


def test_classify_single_changed_amount_small_change():
    cats = cg.classify_subtle([_aff("amount", "100.00", "104.00")], max_relative_change=0.05)
    assert {"single_amount", "amount_le_5pct"} <= set(cats)
    assert "single_date" not in cats


def test_classify_amount_change_just_over_five_percent_is_not_subtle():
    cats = cg.classify_subtle([_aff("amount", "100.00", "105.01")], max_relative_change=0.05)
    assert "single_amount" in cats and "amount_le_5pct" not in cats


def test_classify_exactly_five_percent_counts_as_subtle():
    cats = cg.classify_subtle([_aff("amount", "200.00", "210.00")], max_relative_change=0.05)
    assert "amount_le_5pct" in cats


def test_classify_thousands_separator_and_negative_amounts():
    cats = cg.classify_subtle([_aff("amount", "-1,000.00", "-1,020.00")], max_relative_change=0.05)
    assert "amount_le_5pct" in cats


def test_classify_single_changed_date():
    cats = cg.classify_subtle([_aff("date", "2026-01-03", "2026-01-09")], max_relative_change=0.05)
    assert "single_date" in cats and "single_amount" not in cats and "amount_le_5pct" not in cats


def test_classify_two_changed_amounts_is_not_single():
    cats = cg.classify_subtle([_aff("amount", "100.00", "101.00", "R1"),
                               _aff("amount", "50.00", "51.00", "R2")], max_relative_change=0.05)
    assert "single_amount" not in cats and "amount_le_5pct" not in cats


def test_classify_unchanged_and_inserted_rows_do_not_count():
    assert cg.classify_subtle([_aff("amount", "100.00", "100.00")], max_relative_change=0.05) == frozenset()
    assert cg.classify_subtle([_aff("row", None, "10.00")], max_relative_change=0.05) == frozenset()
    assert cg.classify_subtle([], max_relative_change=0.05) == frozenset()


def test_classify_zero_original_amount_is_not_subtle():
    cats = cg.classify_subtle([_aff("amount", "0.00", "0.01")], max_relative_change=0.05)
    assert "amount_le_5pct" not in cats


def _tampered(i, split, affected):
    return {"doc_id": f"D{i}", "split": split, "is_tampered": True, "affected_rows": affected}


def test_subtle_gate_lists_categories_missing_in_each_tampered_split():
    docs = [
        _tampered(1, "train", [_aff("amount", "100.00", "103.00")]),
        _tampered(2, "train", [_aff("date", "2026-01-03", "2026-01-09")]),
        _tampered(3, "validation", [_aff("amount", "100.00", "180.00")]),
        {"doc_id": "G1", "split": "test", "is_tampered": False, "affected_rows": []},
    ]
    result = cg.subtle_gate(docs, max_relative_change=0.05)
    assert result.passed is False and result.waivable is True
    text = _offender_text(result)
    assert "train" not in text  # all three categories present in train
    assert "validation" in text and "single_date" in text and "amount_le_5pct" in text
    assert "test" not in text.replace("test_", "")  # split without tampered docs is not checked
    assert len(result.offenders) == 2  # validation: single_date, amount_le_5pct


def test_subtle_gate_passes_when_every_tampered_split_has_all_three():
    docs = []
    for k, split in enumerate(("train", "validation", "test")):
        docs += [_tampered(10 * k + 1, split, [_aff("amount", "100.00", "102.00")]),
                 _tampered(10 * k + 2, split, [_aff("date", "2026-01-03", "2026-01-09")])]
    assert cg.subtle_gate(docs, max_relative_change=0.05).passed is True


# ---------------------------------------------------------------------------- capability
FAMILY_OF = {"ReportLab PDF Library": "base", "Mac OS X Quartz PDFContext": "export",
             "PixelForge PDF Editor 5.2": "editor"}


def _style(i, tampered, **over):
    base = {"style_producer": "ReportLab PDF Library", "style_producer_family": "base",
            "style_creator": "c1", "style_fonts": ("Helvetica",), "style_bank_name": "B1",
            "style_street": "S1", "style_row_step_bucket": 12.1, "style_creation_hour": 9}
    base.update(over)
    return _doc(i, tampered=tampered, **base)


def _varied(n_each=4):
    """Both classes show 2 distinct values for every styling choice; producer families base only."""
    rows = []
    for i in range(n_each):
        for tampered in (False, True):
            v = i % 2
            rows.append(_style(10 * i + tampered, tampered,
                               style_creator=f"c{v}", style_fonts=(f"F{v}",), style_bank_name=f"B{v}",
                               style_street=f"S{v}", style_row_step_bucket=12.0 + 0.1 * v,
                               style_creation_hour=9 + v, style_producer=f"P{v}"))
    return rows


def test_capability_never_reports_choice_with_two_values_in_both_classes():
    assert cg.capability_report(_varied()) == []
    result = cg.capability_gate(_varied())
    assert result.passed is True and result.offenders == ()


def test_capability_reports_choice_when_one_class_is_constant():
    rows = [r for r in _varied()]
    for r in rows:
        if r["is_tampered"]:
            r["style_creator"] = "only-one"
    report = cg.capability_report(rows)
    assert any("creator" in item for item in report)
    assert not any("bank_name" in item for item in report)


def test_capability_reports_each_missing_choice_by_name():
    rows = _varied()
    for r in rows:
        r["style_fonts"] = ("Helvetica",)
        r["style_creator"] = "const"
    report = cg.capability_report(rows)
    joined = " ".join(report)
    assert "fonts" in joined and "creator" in joined
    assert "street" not in joined and "producer" not in joined


def test_capability_reports_class_exclusive_producer_family():
    rows = _varied()
    for r in rows:
        r["style_producer_family"] = "base"
    rows[1]["style_producer_family"] = "editor"   # tampered only
    report = cg.capability_report(rows)
    assert any("editor" in item for item in report)
    assert not any("base" in item for item in report)


def test_capability_family_present_in_both_classes_is_not_reported():
    rows = _varied()
    rows[0]["style_producer_family"] = "export"   # genuine
    rows[1]["style_producer_family"] = "export"   # tampered
    assert cg.capability_report(rows) == []


def test_capability_gate_is_waivable_and_stop_message_lists_exactly_the_report():
    rows = _varied()
    for r in rows:
        r["style_fonts"] = ("Helvetica",)
    report = cg.capability_report(rows)
    result = cg.capability_gate(rows)
    assert result.passed is False and result.waivable is True
    assert set(result.offenders) == set(report)
    message = cg.capability_stop_message(report)
    for item in report:
        assert item in message
    assert "street" not in message and "bank_name" not in message


# --------------------------------------------------------------------------- waivability
def test_only_capability_and_subtle_gates_are_waivable():
    waivable = {
        "styling": cg.styling_balance_gate(_bank_rows(8, 2, 5, 5), min_docs=5, share_points=10).waivable,
        "resaved": cg.resaved_share_gate(_split_rows("train", 9, 1), min_share=15.0).waivable,
        "types": cg.tamper_type_minimum_gate(_typed("train", "a", 1, 0), min_docs=3).waivable,
        "coverage": cg.coverage_gate(_typed("train", "a", 1, 0) + _typed("test", "b", 1, 9)).waivable,
        "subtle": cg.subtle_gate([_tampered(1, "train", [])], max_relative_change=0.05).waivable,
        "capability": cg.capability_gate([dict(r, style_fonts=("Helvetica",)) for r in _varied()]).waivable,
    }
    assert waivable == {"styling": False, "resaved": False, "types": False, "coverage": False,
                        "subtle": True, "capability": True}


def test_gates_leave_working_folder_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cg.styling_balance_gate(_bank_rows(8, 2, 5, 5), min_docs=5, share_points=10)
    cg.resaved_share_gate(_split_rows("train", 9, 1), min_share=15.0)
    cg.capability_gate(_varied())
    assert os.listdir(tmp_path) == []
