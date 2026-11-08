"""Fixtures for the ml-retraining tests.

Everything is written to pytest temporary folders. No real statement, answer key,
Real_Style file or Forgery_Dataset file is ever read here (Requirement 17.9). Tests that
need the real files are skipped when ``RETRAIN_REAL_STYLE_DIR`` / ``RETRAIN_FORGERY_DIR``
do not point at them (Requirement 17.10).
"""
from __future__ import annotations

import dataclasses
import itertools
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pytest

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import BuildConfig

MINI_SEEDS = (20260801, 20260802)
MINI_HOLD_OUT_LAYOUT = "two_col_ddmmyy_cr"
MINI_TAMPER_MIX = (
    ("genuine_resaved", 1.0),
    ("metadata_inconsistency", 1.0),
    ("font_mismatch", 1.0),
    ("edited_amount", 1.0),
    ("fabricated_receipts", 1.0),
    ("structuring", 1.0),
)
REAL_STYLE_ENV = "RETRAIN_REAL_STYLE_DIR"
FORGERY_ENV = "RETRAIN_FORGERY_DIR"
NODE4_MODEL_NAMES = ("statement_forgery_clf.joblib", "txn_fraud_clf.joblib")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_without_real_style: skip when RETRAIN_REAL_STYLE_DIR is not set to a folder")


def pytest_collection_modifyitems(config, items):
    folder = os.environ.get(REAL_STYLE_ENV, "")
    if folder and Path(folder).is_dir():
        return
    skip = pytest.mark.skip(reason=f"{REAL_STYLE_ENV} not set or not a folder")
    for item in items:
        if "skip_without_real_style" in item.keywords:
            item.add_marker(skip)


@dataclasses.dataclass(frozen=True)
class MiniCorpus:
    root: Path
    seeds: tuple
    paths: dict          # master_seed -> corpus folder
    hold_out_layout: str
    summaries: dict      # master_seed -> CorpusSummary

    def path(self, seed=None) -> Path:
        return self.paths[self.seeds[0] if seed is None else seed]


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory) -> MiniCorpus:
    """6 layouts x (2 genuine + 2 tampered), 6 months, two Master_Seeds."""
    root = tmp_path_factory.mktemp("mini_corpus")
    paths, summaries = {}, {}
    for seed in MINI_SEEDS:
        out = root / f"seed_{seed}"
        cfg = BuildConfig(
            out_dir=out, master_seed=seed, per_layout_genuine=2, per_layout_tampered=2,
            months_min=6, months_max=6, hold_out_layout=MINI_HOLD_OUT_LAYOUT,
            tamper_mix=MINI_TAMPER_MIX, command_line="pytest mini_corpus")
        summaries[seed] = CorpusBuilder(cfg).build()
        paths[seed] = out
    return MiniCorpus(root=root, seeds=MINI_SEEDS, paths=paths,
                      hold_out_layout=MINI_HOLD_OUT_LAYOUT, summaries=summaries)


@pytest.fixture
def fake_current_models(tmp_path) -> Path:
    """A models/ folder with small fake model files (bytes only, not loadable)."""
    folder = tmp_path / "models"
    folder.mkdir()
    for i, name in enumerate(NODE4_MODEL_NAMES):
        (folder / name).write_bytes(f"fake current model {i}\n".encode() * 8)
    return folder


def _tiny_pdf(text: str, producer: str) -> bytes:
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    c.setProducer(producer)
    c.setCreator("fake-real-style")
    c.drawString(72, 770, text)
    c.save()
    return buf.getvalue()


@pytest.fixture
def fake_real_style_dir(tmp_path) -> Path:
    """Tiny synthetic PDFs standing in for the Real_Style files (never the real ones)."""
    folder = tmp_path / "fake_real_style"
    folder.mkdir()
    for name, producer in (("fake_style_1.pdf", "FakeWriter 1"), ("fake_style_2.pdf", "FakeWriter 2")):
        (folder / name).write_bytes(_tiny_pdf(f"SYNTHETIC {name}", producer))
    return folder


@pytest.fixture
def forgery_dir_or_skip() -> Path:
    folder = os.environ.get(FORGERY_ENV, "")
    if not folder or not Path(folder).is_dir():
        pytest.skip(f"{FORGERY_ENV} not set or not a folder")
    return Path(folder)


@pytest.fixture
def is_ignored() -> Callable:
    """Injectable write-guard predicate replacing ``git check-ignore``.

    Returns ``ignored`` (default True; tests can flip it) and records every path in ``calls``.
    """
    class Predicate:
        def __init__(self):
            self.calls = []
            self.ignored = True

        def __call__(self, path) -> bool:
            self.calls.append(Path(path))
            return self.ignored

    return Predicate()


@pytest.fixture
def clock() -> Callable:
    """Injectable deterministic clock: each call advances one second from a fixed start."""
    start = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
    ticks = itertools.count()

    def now() -> datetime:
        return start + timedelta(seconds=next(ticks))

    return now
