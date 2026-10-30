"""Repository hygiene: external evaluation files must never be committed.

Requirement 14.1. Skips when git is unavailable or the folder is not a checkout.
"""
from __future__ import annotations

import posixpath
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_NAMES = {
    "statement_e.pdf",
    "statement_f.pdf",
    "answer_key_6m_v2.json",
    "labels.csv",
}
FORBIDDEN_PATH_PART = "statement_forgery_dataset"

# Explicit allow-list of tracked PDFs (repo-relative, posix). Currently none.
ALLOWED_TRACKED_PDFS: set[str] = set()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    probe = _git("rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not a git checkout")
    out = _git("ls-files")
    if out.returncode != 0:
        pytest.skip("git ls-files failed")
    return [line for line in out.stdout.splitlines() if line]


def test_no_external_evaluation_files_tracked(tracked_files):
    bad = [
        p
        for p in tracked_files
        if posixpath.basename(p).lower() in FORBIDDEN_NAMES
        or FORBIDDEN_PATH_PART in p.lower()
    ]
    assert not bad, f"External evaluation files are tracked by git: {bad}"


def test_only_allow_listed_pdfs_tracked(tracked_files):
    pdfs = {p for p in tracked_files if p.lower().endswith(".pdf")}
    unexpected = sorted(pdfs - ALLOWED_TRACKED_PDFS)
    assert not unexpected, f"Unexpected tracked PDFs (renamed dataset copies?): {unexpected}"


def test_private_data_folder_is_ignored(tracked_files):
    res = _git("check-ignore", "-q", "data/private/x")
    assert res.returncode == 0, "data/private/ is not ignored by .gitignore"
