"""Tests for the command ``python -m fraud_detect.corpus.build`` (task 19.1).

They fail with ``ImportError`` until task 19.3 creates the module.

Interface assumed: ``main(argv: list[str] | None = None) -> int`` with exit codes
0 (ok), 1 (some documents failed) and 2 (refused: bad settings, non-empty folder
without ``--force``, output inside a protected folder). Refusals print a plain-language
reason on stderr or stdout and write nothing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fraud_detect.corpus.build import main

REPO = Path(__file__).resolve().parents[2]
TINY = ["--per-layout-genuine", "1", "--per-layout-tampered", "0",
        "--months-min", "6", "--months-max", "6", "--seed", "7"]


def _labels(out: Path) -> list[dict]:
    with open(out / "labels.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_help_mentions_default_out_and_all_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for opt in ("--out", "--per-layout-genuine", "--per-layout-tampered", "--months-min",
                "--months-max", "--seed", "--hold-out-layout", "--tamper-mix", "--force"):
        assert opt in text
    assert "data/corpus" in text


def test_tiny_build_exits_zero_and_writes_corpus(tmp_path):
    out = tmp_path / "c"
    assert main(["--out", str(out), *TINY]) == 0
    assert len(_labels(out)) == 6
    assert (out / "README.txt").is_file()


def test_tamper_mix_option(tmp_path):
    out = tmp_path / "c"
    args = ["--out", str(out), "--per-layout-genuine", "0", "--per-layout-tampered", "1",
            "--months-min", "6", "--months-max", "6", "--seed", "7",
            "--tamper-mix", "metadata_inconsistency=1"]
    assert main(args) == 0
    assert {r["tamper_types"] for r in _labels(out)} == {"metadata_inconsistency"}


def test_hold_out_layout_option(tmp_path):
    out = tmp_path / "c"
    assert main(["--out", str(out), *TINY, "--hold-out-layout", "two_col_slash"]) == 0
    for r in _labels(out):
        assert (r["split"] == "test_unseen_layout") == (r["layout"] == "two_col_slash")


def test_non_empty_folder_refused_then_accepted_with_force(tmp_path, capsys):
    out = tmp_path / "c"
    out.mkdir()
    (out / "stray.txt").write_text("x", encoding="utf-8")
    assert main(["--out", str(out), *TINY]) == 2
    captured = capsys.readouterr()
    assert "--force" in captured.out + captured.err
    assert not (out / "labels.csv").exists()
    assert main(["--out", str(out), *TINY, "--force"]) == 0
    assert (out / "labels.csv").is_file()


@pytest.mark.parametrize("extra", [
    ["--months-min", "0"],
    ["--months-max", "25"],
    ["--months-min", "9", "--months-max", "7"],
    ["--hold-out-layout", "no_such_layout"],
])
def test_bad_settings_exit_two_and_write_nothing(tmp_path, extra):
    out = tmp_path / "c"
    base = ["--out", str(out), "--per-layout-genuine", "1", "--per-layout-tampered", "0"]
    assert main(base + extra) == 2
    assert not out.exists()


@pytest.mark.parametrize("inside", ["fraud_detect", "tests", ".kiro"])
def test_output_inside_protected_folder_exits_two(inside):
    target = REPO / inside / "_corpus_should_not_exist"
    assert main(["--out", str(target), *TINY, "--force"]) == 2
    assert not target.exists()


def test_failed_document_gives_exit_one(tmp_path, monkeypatch):
    import fraud_detect.corpus.builder as builder_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(builder_mod, "simulate_ledger", boom)
    assert main(["--out", str(tmp_path / "c"), *TINY]) == 1
