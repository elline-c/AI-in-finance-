"""Hygiene tests for the corpus generator (clean-room, no tracked corpus files, imports).

Requirements 12.6, 17.1, 17.4, 17.7, 18.3, 18.4, 18.6.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import fraud_detect.corpus as corpus_pkg
from fraud_detect.corpus import schema

REPO = Path(__file__).resolve().parents[2]
PKG_DIR = Path(corpus_pkg.__file__).resolve().parent

FORBIDDEN_STRINGS = (
    "Downloads",
    "statement_forgery_dataset",
    "statement_E.pdf",
    "statement_F.pdf",
    "ANSWER_KEY",
    "generate_documents",
)
THIRD_PARTY_ALLOWED = {"reportlab", "pypdf", "pdfplumber"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _package_sources() -> list[Path]:
    return sorted(p for p in PKG_DIR.rglob("*.py"))


def test_corpus_output_folder_is_git_ignored():
    res = _git("check-ignore", "-q", "data/corpus/x.pdf")
    assert res.returncode == 0, "data/corpus/ must be ignored by git"


def test_package_source_has_no_forbidden_strings():
    offenders = []
    for path in _package_sources():
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_STRINGS:
            if needle in text:
                offenders.append((path.name, needle))
    assert offenders == []


def test_no_corpus_files_tracked_by_git():
    res = _git("ls-files")
    assert res.returncode == 0
    bad = []
    for name in res.stdout.splitlines():
        low = name.lower()
        if (low.endswith(".pdf") or low.endswith("labels.csv")
                or low.startswith("data/corpus/") or low.endswith("rows.csv")
                or low.endswith("corpus_manifest.json")):
            bad.append(name)
    assert bad == []


def test_package_imports_only_allowed_modules():
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the package
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root in stdlib or root in THIRD_PARTY_ALLOWED or root == "fraud_detect":
                    continue
                bad.append((path.name, root))
    assert bad == []


def test_package_does_not_import_hypothesis():
    # hypothesis is allowed in tests only
    for path in _package_sources():
        assert "hypothesis" not in path.read_text(encoding="utf-8")


def test_tests_do_not_need_downloads_folder():
    # Tests under tests/corpus must not reference the user's Downloads folder.
    # (This file names the forbidden strings on purpose and is skipped.)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        assert "Downloads" not in text and "~/" not in text, path.name


def test_schema_constants():
    assert schema.GENERATOR_VERSION == "1.0.0"
    assert len(schema.LAYOUTS) == 6 and len(set(schema.LAYOUTS)) == 6
    assert len(schema.TAMPER_TYPES) == 16 and len(set(schema.TAMPER_TYPES)) == 16
    assert schema.LABELS_COLUMNS[0] == "doc_id" and schema.LABELS_COLUMNS[-1] == "generator_version"
    assert len(schema.LABELS_COLUMNS) == 25
    assert schema.ROWS_COLUMNS[-1] == "fraud_type" and len(schema.ROWS_COLUMNS) == 11


def test_schema_layouts_match_node1_registry():
    from fraud_detect.nodes.ingest_parse.layouts import REGISTRY
    names = set(REGISTRY.keys()) if hasattr(REGISTRY, "keys") else {p.name for p in REGISTRY}
    assert set(schema.LAYOUTS) == names
