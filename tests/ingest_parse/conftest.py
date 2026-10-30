"""Shared paths and skip markers for the layout-parser tests.

The external files are synthetic and are never copied into the repo. Tests
that need them are skipped when the files are absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

DOWNLOADS = Path.home() / "Downloads"
DATASET_DIR = DOWNLOADS / "statement_forgery_dataset"
DATASET_LABELS = DATASET_DIR / "labels.csv"
STATEMENT_E = DOWNLOADS / "statement_E.pdf"
STATEMENT_F = DOWNLOADS / "statement_F.pdf"
ANSWER_KEY = DOWNLOADS / "ANSWER_KEY_6M_v2.json"

requires_dataset = pytest.mark.skipif(
    not (DATASET_DIR.is_dir() and DATASET_LABELS.is_file()),
    reason=f"forgery dataset not found at {DATASET_DIR}",
)

requires_labelled_statements = pytest.mark.skipif(
    not (STATEMENT_E.is_file() and STATEMENT_F.is_file() and ANSWER_KEY.is_file()),
    reason="statement_E.pdf / statement_F.pdf / ANSWER_KEY_6M_v2.json not found in ~/Downloads",
)
