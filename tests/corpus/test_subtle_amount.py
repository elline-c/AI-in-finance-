"""Subtle amount-change tamper ``subtle_amount_edit`` (task 6.1, written first).

These tests fail until task 2.1 (schema constants), task 6.2 (``tamper/subtle.py``,
``TAMPER_INFO`` entry, ``catalog._IMPLEMENTATIONS``) and task 2.2 (builder accepts the
Extra_Tamper_Type in ``tamper_mix``) are done.

Pinned interface:
  * ``schema.EXTRA_TAMPER_TYPES == ("subtle_amount_edit",)``, ``schema.ALL_TAMPER_TYPES``
  * ``TAMPER_INFO["subtle_amount_edit"]``: group ``Document``, methods ``("overlay", "rerender")``
  * ``CATALOG["subtle_amount_edit"].plan(ledger, renderer, rng) -> TamperResult``
  * a row that cannot be changed by 1.0% to 4.9% raises ``ValueError`` naming the row id
  * ``CorpusBuilder`` records the type in ``labels.csv`` / Manifest ``label.tamper_types``
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest

from fraud_detect.corpus.seeding import rng
from tests.corpus.helpers import load_renderer, make_test_ledger

NAME = "subtle_amount_edit"


def _renderer():
    r = load_renderer("two_col_slash")
    if r is None:
        pytest.skip("renderer not implemented yet")
    return r


def test_schema_constants():
    from fraud_detect.corpus.schema import ALL_TAMPER_TYPES, EXTRA_TAMPER_TYPES, TAMPER_TYPES

    assert EXTRA_TAMPER_TYPES == (NAME,)
    assert ALL_TAMPER_TYPES == TAMPER_TYPES + EXTRA_TAMPER_TYPES
    assert len(TAMPER_TYPES) == 16 and NAME not in TAMPER_TYPES


def test_registered_in_tamper_info_and_catalog():
    from fraud_detect.corpus.tamper.catalog import CATALOG, TAMPER_INFO

    info = TAMPER_INFO[NAME]
    assert info.group == "Document"
    assert set(info.methods) == {"overlay", "rerender"}
    assert CATALOG[NAME].name == NAME
    assert CATALOG[NAME].group == "Document"
    assert set(CATALOG[NAME].methods) == {"overlay", "rerender"}


def test_tamper_info_coverage_accepts_all_tamper_types():
    from fraud_detect.corpus.schema import ALL_TAMPER_TYPES
    from fraud_detect.corpus.tamper.catalog import TAMPER_INFO

    assert set(TAMPER_INFO) == set(ALL_TAMPER_TYPES)


def test_plan_changes_one_row_by_one_to_five_percent():
    from fraud_detect.corpus.tamper.catalog import CATALOG

    for seed in range(8):
        ledger = make_test_ledger(6, 2000 + seed)
        result = CATALOG[NAME].plan(ledger, _renderer(), rng(seed, "tamper-test", NAME))
        assert result.label.tamper_types == (NAME,)
        assert result.label.method in {"overlay", "rerender"}
        assert len(result.label.affected_rows) == 1
        a = result.label.affected_rows[0]
        assert a["field"] == "amount"
        old, new = float(a["original"].replace(",", "")), float(a["new"].replace(",", ""))
        assert old != new
        assert 0.01 <= abs(new - old) / abs(old) <= 0.05


def test_too_small_ledger_raises_value_error_naming_the_row():
    from fraud_detect.corpus.tamper.catalog import CATALOG

    ledger = make_test_ledger(2, 7)
    rows = tuple(dataclasses.replace(r, amount=0.10) for r in ledger.rows)
    biggest = dataclasses.replace(rows[3], amount=0.20)   # the unique largest row
    rows = rows[:3] + (biggest,) + rows[4:]
    ledger = dataclasses.replace(ledger, rows=rows, printed_balances=None)
    with pytest.raises(ValueError) as exc:
        CATALOG[NAME].plan(ledger, _renderer(), rng(1, "tamper-test", NAME))
    assert biggest.row_id in str(exc.value)


def test_tiny_build_records_type_and_affected_row(tmp_path: Path):
    from fraud_detect.corpus.builder import CorpusBuilder
    from fraud_detect.corpus.schema import BuildConfig

    cfg = BuildConfig(
        out_dir=tmp_path, master_seed=20260803, per_layout_genuine=1, per_layout_tampered=2,
        months_min=6, months_max=6, hold_out_layout="two_col_ddmmyy_cr",
        tamper_mix=((NAME, 1.0),), command_line="pytest subtle")
    CorpusBuilder(cfg).build()

    seen = 0
    for p in sorted((tmp_path / "manifests").glob("*.json")):
        label = json.loads(p.read_text("utf-8"))["label"]
        if NAME in label["tamper_types"]:
            seen += 1
            assert label["is_tampered"] is True
            assert len(label["affected_rows"]) == 1
            assert label["affected_rows"][0]["field"] == "amount"
    assert seen > 0, "no document carries subtle_amount_edit"

    corpus = json.loads((tmp_path / "corpus_manifest.json").read_text("utf-8"))
    per_type = corpus["counts"]["per_tamper_type"]
    assert any(NAME in v for v in per_type.values()) or NAME in per_type
