"""Fixtures for the corpus generator tests.

The suite-wide guard in tests/conftest.py (live registry lookup off) applies here too.
Everything is written to pytest temporary folders; nothing goes under the repository.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def corpus_tmp_factory(tmp_path_factory):
    """Return a function that makes a fresh, empty temporary corpus folder per call."""
    def make(name: str = "corpus") -> Path:
        return tmp_path_factory.mktemp(name)
    return make


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    """A fresh empty folder for one test."""
    out = tmp_path / "corpus"
    out.mkdir()
    return out
