"""Test-suite wide guard: the live company-registry lookup is OFF in tests.

The web apps switch it on while a server runs. Tests must stay offline and repeatable,
so every test starts with it explicitly off, whatever the developer's shell has set.
Tests of the live source use a fake service (see tests/test_registry.py).
"""
import pytest

from fraud_detect import registry


@pytest.fixture(autouse=True)
def _live_registry_off(monkeypatch):
    monkeypatch.setenv(registry.LIVE_ENV, "0")
    registry.set_live_default(False)
    registry.clear_cache()
    yield
    registry.set_live_default(False)
    registry.clear_cache()
