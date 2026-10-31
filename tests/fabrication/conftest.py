"""Shared paths and skip markers for the fabrication-rule tests.

Labelled files live in ~/Downloads and are never copied into the repo.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DOWNLOADS = Path.home() / "Downloads"
STATEMENT_E = DOWNLOADS / "statement_E.pdf"        # tampered
STATEMENT_F = DOWNLOADS / "statement_F.pdf"        # genuine
ANSWER_KEY = DOWNLOADS / "ANSWER_KEY_6M_v2.json"
FORGERY_DATASET = DOWNLOADS / "statement_forgery_dataset"

requires_labelled_statements = pytest.mark.skipif(
    not (STATEMENT_E.is_file() and STATEMENT_F.is_file() and ANSWER_KEY.is_file()),
    reason="statement_E.pdf, statement_F.pdf or ANSWER_KEY_6M_v2.json not found in ~/Downloads",
)
requires_dataset = pytest.mark.skipif(
    not FORGERY_DATASET.is_dir(),
    reason="statement_forgery_dataset not found in ~/Downloads",
)
