"""Tests for the temporary company registration scope (Requirements 13.2 to 13.7).

Written before ``fraud_detect/corpus/scope.py`` exists: they fail with ImportError until then.

Assumed record shape (the scope accepts a list of these, or the path of a
``companies.json`` holding the same list)::

    {"uen": "...", "name": "...",
     "baseline": {"avg_monthly_credit": float, "avg_monthly_debit": float}}
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from fraud_detect import reference_data
from fraud_detect import registry
from fraud_detect.nodes import financial_logic
from fraud_detect.corpus.scope import registered_companies

from tests.corpus.helpers import implemented_layouts, make_test_ledger, render_and_ingest

REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (REPO / "fraud_detect" / "reference_data.py", REPO / "fraud_detect" / "registry.py")

RECORDS = [
    {"uen": "209900001Z", "name": "KESTREL FREIGHT PTE. LTD.",
     "baseline": {"avg_monthly_credit": 123456.78, "avg_monthly_debit": 111111.11}},
    {"uen": "20990002Y", "name": "AMBERWAY MEDIA PTE. LTD.",
     "baseline": {"avg_monthly_credit": 50000.0, "avg_monthly_debit": 42000.5}},
]


def _snapshot():
    return copy.deepcopy(reference_data.ACRA_REGISTRY), copy.deepcopy(reference_data.CASHFLOW_BASELINE)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Boom(RuntimeError):
    pass


def test_registry_and_baseline_visible_inside_scope():
    with registered_companies(RECORDS):
        for rec in RECORDS:
            result = registry.lookup_entity(rec["uen"])
            assert result.state == registry.FOUND
            assert result.record.name == rec["name"]
            # Node 3 reads the same dictionary object by name
            assert financial_logic.CASHFLOW_BASELINE[rec["uen"]] == rec["baseline"]
            assert financial_logic.CASHFLOW_BASELINE.get(rec["uen"], {}) == rec["baseline"]


def test_companies_json_path_is_accepted(tmp_path):
    path = tmp_path / "companies.json"
    path.write_text(json.dumps(RECORDS), encoding="utf-8")
    with registered_companies(path):
        assert registry.lookup_entity(RECORDS[0]["uen"]).record.name == RECORDS[0]["name"]
        assert financial_logic.CASHFLOW_BASELINE[RECORDS[1]["uen"]] == RECORDS[1]["baseline"]
    assert RECORDS[0]["uen"] not in reference_data.ACRA_REGISTRY


def test_dictionaries_changed_in_place_not_replaced():
    acra_id, base_id = id(reference_data.ACRA_REGISTRY), id(reference_data.CASHFLOW_BASELINE)
    with registered_companies(RECORDS):
        assert id(reference_data.ACRA_REGISTRY) == acra_id
        assert id(reference_data.CASHFLOW_BASELINE) == base_id
        assert registry.ACRA_REGISTRY is reference_data.ACRA_REGISTRY
        assert financial_logic.CASHFLOW_BASELINE is reference_data.CASHFLOW_BASELINE
    assert id(reference_data.ACRA_REGISTRY) == acra_id
    assert id(reference_data.CASHFLOW_BASELINE) == base_id


def test_dictionaries_restored_after_scope():
    before = _snapshot()
    with registered_companies(RECORDS):
        assert reference_data.ACRA_REGISTRY != before[0]
    assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before
    assert registry.lookup_entity(RECORDS[0]["uen"]).state == registry.NOT_FOUND


def test_dictionaries_restored_after_error_inside_scope():
    before = _snapshot()
    with pytest.raises(_Boom):
        with registered_companies(RECORDS):
            assert RECORDS[0]["uen"] in reference_data.ACRA_REGISTRY
            raise _Boom("measurement failed")
    assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before


def test_empty_list_changes_nothing():
    before = _snapshot()
    with registered_companies([]):
        assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before
    assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before


def test_live_lookup_stays_off():
    env_before = os.environ.get(registry.LIVE_ENV)
    with registered_companies(RECORDS):
        assert registry.live_enabled() is False
        result = registry.lookup_entity("209900099X")      # not registered anywhere
        assert result.state == registry.NOT_FOUND
        assert result.live_checked is False
    assert os.environ.get(registry.LIVE_ENV) == env_before
    assert registry.live_enabled() is False


def test_source_files_unchanged_on_disk():
    before = {p: _digest(p) for p in SOURCE_FILES}
    with registered_companies(RECORDS):
        assert {p: _digest(p) for p in SOURCE_FILES} == before
    with pytest.raises(_Boom):
        with registered_companies(RECORDS):
            raise _Boom()
    assert {p: _digest(p) for p in SOURCE_FILES} == before


def test_existing_uen_with_different_name_is_refused_and_reported():
    existing_uen, existing_name = next(iter(reference_data.ACRA_REGISTRY.items()))
    clash = {"uen": existing_uen, "name": existing_name + " IMPOSTOR",
             "baseline": {"avg_monthly_credit": 1.0, "avg_monthly_debit": 1.0}}
    before = _snapshot()
    with pytest.raises(ValueError) as info:
        with registered_companies([RECORDS[0], clash]):
            pytest.fail("the scope must not be entered when a clash exists")
    assert existing_uen in str(info.value)
    # nothing is left behind, not even the non-clashing record
    assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before


def test_existing_uen_with_same_name_is_not_a_clash():
    existing_uen, existing_name = next(iter(reference_data.ACRA_REGISTRY.items()))
    same = {"uen": existing_uen, "name": existing_name,
            "baseline": reference_data.CASHFLOW_BASELINE.get(
                existing_uen, {"avg_monthly_credit": 1.0, "avg_monthly_debit": 1.0})}
    before = _snapshot()
    with registered_companies([same]):
        assert registry.lookup_entity(existing_uen).record.name == existing_name
    assert (reference_data.ACRA_REGISTRY, reference_data.CASHFLOW_BASELINE) == before


def test_generated_genuine_file_raises_no_uen_not_found_only_inside_scope():
    layouts = implemented_layouts()
    if not layouts:
        pytest.skip("no layout implemented yet")
    layout = layouts[0]
    from fraud_detect.corpus.layouts import get_renderer

    ledger = make_test_ledger(6, 7, getattr(get_renderer(layout), "max_rows_per_month", None))
    company = ledger.company
    rendered = render_and_ingest(layout, 6, 7, ledger=ledger)
    ing = rendered.ingestion
    assert ing.uen == company.uen

    record = {"uen": company.uen, "name": company.name,
              "baseline": {"avg_monthly_credit": 80000.0, "avg_monthly_debit": 70000.0}}

    def codes() -> set[str]:
        return {f.code for f in financial_logic.analyze(ing).findings}

    assert "FL_UEN_NOT_FOUND" in codes()
    with registered_companies([record]):
        inside = codes()
        assert "FL_UEN_NOT_FOUND" not in inside
        assert "FL_UEN_NAME_MISMATCH" not in inside
    assert "FL_UEN_NOT_FOUND" in codes()
