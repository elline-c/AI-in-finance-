"""Smoke tests for the task 1.1 scaffolding: CLI stubs, Mini_Corpus, protected hashes."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fraud_detect.ml.retraining import cli

REPO = Path(__file__).resolve().parents[2]


def test_import_is_light():
    code = ("import sys, fraud_detect.ml.retraining, fraud_detect.ml.retraining.cli as c;"
            "bad=[m for m in ('torch','sklearn','pdfplumber') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


def test_exit_codes():
    assert (cli.EXIT_OK, cli.EXIT_FAILURE, cli.EXIT_INTEGRITY, cli.EXIT_WAITING,
            cli.EXIT_IMPLAUSIBLE) == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("command", list(cli.COMMANDS))
def test_help_exits_zero(command, capsys):
    assert cli.main([command, "--help"]) == 0
    assert command in capsys.readouterr().out


def test_ten_commands():
    assert set(cli.COMMANDS) == {"init", "build", "check", "audit", "leak-test", "train",
                                 "compare", "switch", "restore", "decide"}


def test_no_command_and_unknown_command_are_bad_arguments():
    assert cli.main([]) == 2
    assert cli.main(["nope"]) == 2


def test_mini_corpus_shape(mini_corpus):
    assert len(mini_corpus.seeds) == 2
    for seed in mini_corpus.seeds:
        with open(mini_corpus.path(seed) / "labels.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 6 * 4
        assert not mini_corpus.summaries[seed].failed
    assert "test_unseen_layout" in {r["split"] for r in rows}


def test_fake_fixtures(fake_current_models, fake_real_style_dir, is_ignored, clock):
    assert len(list(fake_current_models.iterdir())) == 2
    pdfs = sorted(fake_real_style_dir.glob("*.pdf"))
    assert len(pdfs) == 2 and all(p.read_bytes().startswith(b"%PDF") for p in pdfs)
    assert is_ignored(Path("x")) is True and is_ignored.calls == [Path("x")]
    assert clock() < clock()


@pytest.mark.skip_without_real_style
def test_marker_runs_only_with_real_style_dir():
    pass


def test_protected_hashes_match_sources():
    data = json.loads((Path(__file__).parent / "protected_hashes.json").read_text())
    assert len(data["sha256"]) >= 7
    for rel, digest in data["sha256"].items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == digest, rel
    pinned = data["node4_pinned"]
    assert pinned["module_constants"] == {"AUTO_CLEAR_MAX_RISK": 25, "MIN_CONFIDENCE_FOR_AUTO": 60}
