"""Registry lookup behind the UEN cross-check (local fixtures + opt-in live source).

No test here touches the network: a fake fetcher stands in for data.gov.sg. The one
live test runs only when RUN_LIVE_TESTS=1.
"""
from __future__ import annotations

import json
import os
from urllib.parse import parse_qs, urlparse

import pytest

from fraud_detect import registry
from fraud_detect.models import DocumentMetadata, IngestionResult
from fraud_detect.nodes import financial_logic
from fraud_detect.registry import (
    FOUND, NOT_FOUND, UNAVAILABLE, lookup_entity, names_match,
)

LIVE_UEN = "201201936C"
LIVE_NAME = "MILLENNIUM AUTOMATION & SYSTEMS PTE. LTD."
DS1, DS2 = "d_3f960c10fed6145404ca7b821f263b87", "d_b1d2b840ab9e993570c037b706b39bb8"


class FakeService:
    """Stands in for data.gov.sg. ``records`` maps dataset id -> {uen: record dict}."""

    def __init__(self, records=None, fail=None, meta_fail=False, child=(DS1, DS2)):
        self.records = records or {DS1: {LIVE_UEN: {
            "uen": LIVE_UEN, "entity_name": LIVE_NAME, "uen_status_desc": "Deregistered"}}}
        self.fail, self.meta_fail, self.child = fail, meta_fail, list(child)
        self.urls: list[str] = []

    def __call__(self, url: str) -> dict:
        self.urls.append(url)
        if "collections" in url:
            if self.meta_fail:
                raise OSError("metadata down")
            return {"code": 0, "data": {"collectionMetadata": {"childDatasets": self.child}}}
        if self.fail:
            raise self.fail
        q = parse_qs(urlparse(url).query)
        uen = json.loads(q["filters"][0])["uen"]
        rec = self.records.get(q["resource_id"][0], {}).get(uen)
        return {"success": True, "result": {"records": [rec] if rec else []}}

    @property
    def datastore_calls(self):
        return [u for u in self.urls if "datastore_search" in u]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(registry.LIVE_ENV, raising=False)
    registry.clear_cache()
    yield
    registry.clear_cache()


# ---- local source ------------------------------------------------------------
def test_local_registry_answers_without_any_network_call():
    svc = FakeService()
    r = lookup_entity("201812345A", live=True, fetch=svc)
    assert r.state == FOUND and r.record.source == "local"
    assert r.record.name == "Meridian Trading Pte Ltd"
    assert svc.urls == []                        # local entries win; nothing sent


def test_test_company_uens_are_registered_locally():
    for uen in ("209931874D", "209947062H"):
        assert lookup_entity(uen).state == FOUND


def test_unknown_uen_without_live_is_not_found_and_sends_nothing():
    svc = FakeService()
    r = lookup_entity(LIVE_UEN, live=False, fetch=svc)
    assert r.state == NOT_FOUND and not r.live_checked
    assert svc.urls == []


def test_live_is_off_by_default(monkeypatch):
    svc = FakeService()
    assert lookup_entity(LIVE_UEN, fetch=svc).state == NOT_FOUND
    assert svc.urls == []
    monkeypatch.setenv(registry.LIVE_ENV, "1")
    assert lookup_entity(LIVE_UEN, fetch=svc).state == FOUND


# ---- live source -------------------------------------------------------------
def test_live_found_returns_name_and_status():
    svc = FakeService()
    r = lookup_entity(LIVE_UEN.lower(), live=True, fetch=svc)
    assert r.state == FOUND and r.live_checked
    assert (r.record.uen, r.record.name, r.record.status, r.record.source) == (
        LIVE_UEN, LIVE_NAME, "Deregistered", "data.gov.sg")


def test_live_checks_every_child_dataset_until_a_hit():
    svc = FakeService(records={DS2: {"T09SS0255G": {
        "uen": "T09SS0255G", "entity_name": "eCommerce Association of Singapore",
        "uen_status_desc": "Registered"}}})
    r = lookup_entity("T09SS0255G", live=True, fetch=svc)
    assert r.state == FOUND and r.record.name.startswith("eCommerce")
    assert len(svc.datastore_calls) == 2


def test_live_not_found_after_all_datasets_answered():
    svc = FakeService()
    r = lookup_entity("209931874X", live=True, fetch=svc)
    assert r.state == NOT_FOUND and r.live_checked
    assert len(svc.datastore_calls) == 2


def test_only_the_uen_is_sent_and_it_is_url_encoded():
    svc = FakeService()
    lookup_entity(LIVE_UEN, live=True, fetch=svc)
    for url in svc.datastore_calls:
        q = parse_qs(urlparse(url).query)
        assert json.loads(q["filters"][0]) == {"uen": LIVE_UEN}
        assert set(q) == {"resource_id", "limit", "filters"}


@pytest.mark.parametrize("bad", ["", "  ", "12", "ABC DEF 123", "1" * 30, '9"}&x=1', "../etc"])
def test_malformed_uen_is_never_sent(bad):
    svc = FakeService()
    assert lookup_entity(bad, live=True, fetch=svc).state == NOT_FOUND
    assert svc.urls == []


def test_none_uen_does_not_raise():
    assert lookup_entity(None, live=True, fetch=FakeService()).state == NOT_FOUND


def test_collection_metadata_supplies_dataset_ids_and_falls_back():
    svc = FakeService(child=[DS1])
    lookup_entity("209931874X", live=True, fetch=svc)
    assert len(svc.datastore_calls) == 1
    registry.clear_cache()
    svc2 = FakeService(meta_fail=True)
    assert lookup_entity(LIVE_UEN, live=True, fetch=svc2).state == FOUND   # fallback ids


def test_junk_dataset_ids_in_metadata_are_ignored():
    svc = FakeService(child=["not-a-dataset", "d_xyz", 5])
    assert lookup_entity(LIVE_UEN, live=True, fetch=svc).state == FOUND
    assert all(DS1 in u or DS2 in u for u in svc.datastore_calls)


# ---- failure is "unavailable", never "not found" -------------------------------
@pytest.mark.parametrize("exc", [OSError("down"), TimeoutError(), ValueError("bad json")])
def test_failures_are_unavailable_with_type_only(exc):
    r = lookup_entity(LIVE_UEN, live=True, fetch=FakeService(fail=exc))
    assert r.state == UNAVAILABLE and r.record is None and r.live_checked
    assert r.detail == type(exc).__name__


def test_service_reporting_failure_is_unavailable():
    def fetch(url):
        return {"code": 0, "data": {"collectionMetadata": {"childDatasets": [DS1]}}} \
            if "collections" in url else {"success": False}
    assert lookup_entity(LIVE_UEN, live=True, fetch=fetch).state == UNAVAILABLE


def test_malformed_response_is_unavailable():
    def fetch(url):
        return {"code": 0, "data": {"collectionMetadata": {"childDatasets": [DS1]}}} \
            if "collections" in url else {"success": True, "result": {}}
    assert lookup_entity(LIVE_UEN, live=True, fetch=fetch).state == UNAVAILABLE


# ---- cache ---------------------------------------------------------------------
def test_found_and_not_found_results_are_cached():
    svc = FakeService()
    lookup_entity(LIVE_UEN, live=True, fetch=svc)
    n = len(svc.urls)
    assert lookup_entity(LIVE_UEN, live=True, fetch=svc).state == FOUND
    lookup_entity("209931874X", live=True, fetch=svc)
    m = len(svc.urls)
    lookup_entity("209931874X", live=True, fetch=svc)
    assert len(svc.urls) == m and m > n


def test_unavailable_is_retried_after_a_short_wait():
    t = [1000.0]
    clock = lambda: t[0]
    bad = FakeService(fail=OSError("down"))
    assert lookup_entity(LIVE_UEN, live=True, fetch=bad, clock=clock).state == UNAVAILABLE
    calls = len(bad.urls)
    lookup_entity(LIVE_UEN, live=True, fetch=bad, clock=clock)
    assert len(bad.urls) == calls                       # inside the short window: cached
    t[0] += registry.UNAVAILABLE_TTL_S + 1
    good = FakeService()
    assert lookup_entity(LIVE_UEN, live=True, fetch=good, clock=clock).state == FOUND


def test_cache_expires():
    t = [0.0]
    clock = lambda: t[0]
    svc = FakeService()
    lookup_entity(LIVE_UEN, live=True, fetch=svc, clock=clock)
    n = len(svc.urls)
    t[0] += registry.FOUND_TTL_S + 1
    lookup_entity(LIVE_UEN, live=True, fetch=svc, clock=clock)
    assert len(svc.urls) > n


# ---- name comparison -----------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("HALBERD LANE JOINERY PTE. LTD.", "Halberd Lane Joinery Pte Ltd"),
    ("A & B TRADING PTE. LTD.", "A AND B TRADING PTE LTD"),
    ("  Acme   Pte.  Ltd ", "ACME PTE LTD"),
])
def test_names_match_ignores_case_punctuation_and_spacing(a, b):
    assert names_match(a, b)


@pytest.mark.parametrize("a,b", [
    ("Acme Pte Ltd", "Acme Holdings Pte Ltd"), ("Acme Pte Ltd", ""), (None, "Acme"), ("A", None),
])
def test_different_names_do_not_match(a, b):
    assert not names_match(a, b)


# ---- Node 3 cross-check ---------------------------------------------------------
def _ing(uen, holder):
    return IngestionResult(source_path="x.pdf", page_count=1, raw_text="",
                           metadata=DocumentMetadata(), uen=uen, account_holder=holder)


def _codes(ing):
    return {f.code: f for f in financial_logic._entity_crosscheck(ing)}


def test_registered_company_with_punctuation_variant_name_is_clean():
    assert _codes(_ing("201812345A", "MERIDIAN TRADING PTE. LTD.")) == {}


def test_wrong_holder_for_registered_uen_is_a_name_mismatch():
    got = _codes(_ing("201812345A", "SOMEONE ELSE PTE. LTD."))
    assert set(got) == {"FL_UEN_NAME_MISMATCH"} and got["FL_UEN_NAME_MISMATCH"].weight == 22


def test_unknown_uen_offline_is_not_found():
    got = _codes(_ing("209999999Z", "X PTE LTD"))
    assert set(got) == {"FL_UEN_NOT_FOUND"} and got["FL_UEN_NOT_FOUND"].weight == 10


def test_missing_uen_is_left_to_node_1():
    assert _codes(_ing(None, "X")) == {}


def test_node3_uses_the_live_source_when_enabled(monkeypatch):
    svc = FakeService()
    monkeypatch.setattr(registry, "_http_get_json", svc)
    monkeypatch.setenv(registry.LIVE_ENV, "1")
    assert _codes(_ing(LIVE_UEN, LIVE_NAME)) == {}                      # live hit, name matches
    assert set(_codes(_ing(LIVE_UEN, "OTHER PTE. LTD."))) == {"FL_UEN_NAME_MISMATCH"}
    nf = _codes(_ing("209999999Z", "X PTE LTD"))
    assert set(nf) == {"FL_UEN_NOT_FOUND"} and "data.gov.sg" in nf["FL_UEN_NOT_FOUND"].evidence


def test_live_outage_is_a_low_data_quality_note_not_a_not_found(monkeypatch):
    monkeypatch.setattr(registry, "_http_get_json", FakeService(fail=OSError("down")))
    monkeypatch.setenv(registry.LIVE_ENV, "1")
    got = _codes(_ing("209999999Z", "X PTE LTD"))
    assert set(got) == {"DQ_REGISTRY_UNAVAILABLE"}
    f = got["DQ_REGISTRY_UNAVAILABLE"]
    assert f.severity.value == "LOW" and f.weight == 2 and "OSError" in f.evidence


# ---- optional live test ----------------------------------------------------------
@pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1")
def test_real_data_gov_sg_lookup():
    r = lookup_entity(LIVE_UEN, live=True)
    assert r.state == FOUND and r.record.name == LIVE_NAME
    assert lookup_entity("209931874X", live=True).state == NOT_FOUND


# ---- web apps turn the live source on while the server runs -----------------------
def test_unset_variable_follows_the_process_default(monkeypatch):
    monkeypatch.delenv(registry.LIVE_ENV, raising=False)
    assert registry.live_enabled() is False
    registry.set_live_default(True)
    assert registry.live_enabled() is True
    monkeypatch.setenv(registry.LIVE_ENV, "0")          # explicit setting wins over the default
    assert registry.live_enabled() is False
    monkeypatch.setenv(registry.LIVE_ENV, "")           # empty counts as unset
    assert registry.live_enabled() is True


def test_importing_the_apps_leaves_live_lookup_off(monkeypatch):
    monkeypatch.delenv(registry.LIVE_ENV, raising=False)
    from fraud_detect import customer_app, webapp  # noqa: F401  (import only)
    assert registry.live_enabled() is False


@pytest.mark.parametrize("module", ["webapp", "customer_app"])
def test_screening_apps_switch_live_on_only_while_running(monkeypatch, module):
    import importlib
    from fastapi.testclient import TestClient

    app = importlib.import_module(f"fraud_detect.{module}").app
    monkeypatch.delenv(registry.LIVE_ENV, raising=False)
    assert registry.live_enabled() is False
    with TestClient(app):                                # runs startup/shutdown
        assert registry.live_enabled() is True
        monkeypatch.setenv(registry.LIVE_ENV, "0")       # explicit off still wins
        assert registry.live_enabled() is False
        monkeypatch.delenv(registry.LIVE_ENV)
    assert registry.live_enabled() is False


def test_banker_app_does_not_use_the_registry(monkeypatch):
    from fraud_detect import banker_app
    assert banker_app.app.router.lifespan_context is not registry.live_lookup_lifespan
