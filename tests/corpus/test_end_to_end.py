"""End-to-end check on a mini corpus (task 23.1).

Builds a small corpus into a temporary folder with the real build command, runs the
real verification report (Nodes 1 to 4, in memory) on it, and checks that the pieces
hold together: outputs are consistent, nothing is written outside the temporary
folder, the fictional companies are not left registered, PDFs are marked synthetic and
the same seed gives the same labels.

Node 2, 3 and 4 figures are printed, not asserted. Only Node 1 facts on genuine
documents are asserted.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import os
from pathlib import Path

import pytest

from fraud_detect import reference_data, registry
from fraud_detect.corpus import verify
from fraud_detect.corpus.build import main as build_main
from fraud_detect.corpus.schema import LAYOUTS

REPO = Path(__file__).resolve().parents[2]
SEED = "20260601"
MIX = "edited_amount=1,metadata_inconsistency=1,fabricated_receipts=1,structuring=1"
HOLD_OUT = "two_col_slash"
BUILD_ARGS = ["--per-layout-genuine", "1", "--per-layout-tampered", "1",
              "--months-min", "6", "--months-max", "6", "--seed", SEED,
              "--hold-out-layout", HOLD_OUT, "--tamper-mix", MIX]

_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".hypothesis", ".pytest_cache",
              "node_modules", ".mypy_cache", ".ruff_cache"}


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file under root (caches skipped)."""
    snap: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return snap


def _registry_state() -> dict:
    return {
        "acra": copy.deepcopy(reference_data.ACRA_REGISTRY),
        "baseline": copy.deepcopy(reference_data.CASHFLOW_BASELINE),
        "live": registry._live_default,
    }


def _labels(out: Path) -> list[dict]:
    with open(out / "labels.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Build once, verify once, and remember what changed around them."""
    work = tmp_path_factory.mktemp("e2e")
    out = work / "corpus"
    repo_before = _snapshot(REPO)
    registry_before = _registry_state()

    build_code = build_main(["--out", str(out), *BUILD_ARGS])
    registry_after_build = _registry_state()

    report = io.StringIO()
    json_path = work / "report.json"
    md_path = work / "report.md"
    verify_code = verify.main(
        ["--corpus", str(out), "--json", str(json_path), "--markdown", str(md_path)],
        out=report)
    registry_after_verify = _registry_state()
    repo_after = _snapshot(REPO)
    print("\n" + report.getvalue())  # Node 2-4 figures: shown, not asserted

    return {
        "work": work, "out": out, "build_code": build_code, "verify_code": verify_code,
        "report": report.getvalue(), "json_path": json_path, "md_path": md_path,
        "repo_before": repo_before, "repo_after": repo_after,
        "registry_before": registry_before,
        "registry_after_build": registry_after_build,
        "registry_after_verify": registry_after_verify,
    }


def test_build_and_report_exit_zero(world):
    assert world["build_code"] == 0
    assert world["verify_code"] == 0
    assert "elapsed" in world["report"]
    assert not world["report"].startswith("skipped")


def test_outputs_exist_and_counts_are_consistent(world):
    out = world["out"]
    for name in ("labels.csv", "rows.csv", "companies.json", "corpus_manifest.json",
                 "README.txt"):
        assert (out / name).is_file(), name
    labels = _labels(out)
    assert len(labels) == 2 * len(LAYOUTS)
    assert {r["layout"] for r in labels} == set(LAYOUTS)
    for layout in LAYOUTS:
        group = [r for r in labels if r["layout"] == layout]
        assert sum(r["is_tampered"] == "0" for r in group) >= 1
        assert len(group) == 2
    assert len(list((out / "pdfs").glob("*.pdf"))) == len(labels)
    assert len(list((out / "manifests").glob("*.json"))) == len(labels)

    corpus_manifest = json.loads((out / "corpus_manifest.json").read_text("utf-8"))
    assert "SYNTHETIC DATA" in (out / "README.txt").read_text("utf-8")
    assert corpus_manifest.get("synthetic", True) is True
    assert str(corpus_manifest.get("seed", SEED)) == SEED

    companies = json.loads((out / "companies.json").read_text("utf-8"))
    companies = companies.get("companies", companies) if isinstance(companies, dict) else companies
    assert {c["company_id"] for c in companies} == {r["company_id"] for r in labels}


def test_labels_agree_with_manifests_and_rows_csv(world):
    out = world["out"]
    labels = _labels(out)
    with open(out / "rows.csv", newline="", encoding="utf-8") as fh:
        rows_csv = list(csv.DictReader(fh))
    total_rows = 0
    for lab in labels:
        m = json.loads((out / "manifests" / f"{lab['doc_id']}.json").read_text("utf-8"))
        assert m["doc_id"] == lab["doc_id"]
        assert m["layout"] == lab["layout"]
        assert m["split"] == lab["split"]
        assert m["synthetic"] is True
        assert (out / lab["file"]).is_file()
        assert m["label"]["is_tampered"] == (lab["is_tampered"] == "1")
        assert ";".join(m["label"]["tamper_types"]) == lab["tamper_types"]
        assert m["label"]["arithmetic_consistent"] == (lab["arithmetic_consistent"] == "1")
        assert len(m["rows"]) == int(lab["n_rows"])
        assert [r["row_index"] for r in m["rows"]] == list(range(1, len(m["rows"]) + 1))
        fraud_ids = {r["row_id"] for r in m["rows"] if r["is_fraud_row"]}
        assert int(lab["n_affected_rows"]) == len(m["label"]["affected_rows"])
        if lab["is_tampered"] == "0":
            assert not fraud_ids
        total_rows += len(m["rows"])
        # holder / split rule for the held-out layout
        assert (lab["split"] == "test_unseen_layout") == (lab["layout"] == HOLD_OUT)
    assert len(rows_csv) == total_rows
    assert any(r["is_tampered"] == "1" for r in labels), "mini corpus has no tampered document"


def test_nothing_written_outside_tmp_path(world):
    work = world["work"].resolve()
    assert REPO not in work.parents and work != REPO
    assert world["repo_before"] == world["repo_after"], (
        "files under the repo (including data/) changed during build + verify: "
        + ", ".join(sorted(set(world["repo_before"]) ^ set(world["repo_after"]))
                    or [k for k in world["repo_before"]
                        if world["repo_before"][k] != world["repo_after"].get(k)]))
    # default git-ignored location untouched by this test
    assert not (REPO / "data" / "corpus" / "labels.csv").exists() or \
        ("data/corpus/labels.csv" in world["repo_before"])
    # report files were only written where we asked
    assert world["json_path"].is_file() and world["md_path"].is_file()
    json.loads(world["json_path"].read_text("utf-8"))


def test_fictional_companies_not_left_registered(world):
    assert world["registry_after_build"] == world["registry_before"]
    assert world["registry_after_verify"] == world["registry_before"]
    companies = json.loads((world["out"] / "companies.json").read_text("utf-8"))
    companies = companies.get("companies", companies) if isinstance(companies, dict) else companies
    for c in companies:
        assert c["uen"] not in reference_data.ACRA_REGISTRY
        assert c["uen"] not in reference_data.CASHFLOW_BASELINE


def test_pdfs_carry_the_synthetic_marking(world):
    import pdfplumber
    from pypdf import PdfReader

    for lab in _labels(world["out"]):
        path = world["out"] / lab["file"]
        meta = PdfReader(str(path)).metadata or {}
        assert "synthetic" in f"{meta.get('/Title', '')} {meta.get('/Subject', '')}".lower(), \
            f"{lab['doc_id']}: metadata does not say synthetic"
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").lower()
                assert "synthetic" in text or "specimen" in text, \
                    f"{lab['doc_id']}: page {page.page_number} has no synthetic footer"


def test_node1_reads_genuine_documents_completely(world):
    records = verify.load_corpus(world["out"])
    genuine = [r for r in records if not r.is_tampered and not r.genuine_resaved]
    assert {r.layout for r in genuine} == set(LAYOUTS)
    results = verify.measure_node1(genuine)
    assert len(results) == len(genuine)
    for r in results:
        assert r.error is None, f"{r.doc_id}: {r.error}"
        assert r.rows_read == r.rows_expected, f"{r.doc_id}: rows {r.rows_read}/{r.rows_expected}"
        assert r.balance_ok, f"{r.doc_id}: replayed balance differs from printed closing"
        assert r.header_ok, f"{r.doc_id}: header holder/UEN/period not read back"
        assert r.profile == r.layout, f"{r.doc_id}: profile {r.profile} != layout {r.layout}"


def test_same_seed_gives_identical_labels(world, tmp_path):
    again = tmp_path / "again"
    assert build_main(["--out", str(again), *BUILD_ARGS]) == 0
    first = world["out"]
    for name in ("labels.csv", "rows.csv", "companies.json"):
        assert (first / name).read_bytes() == (again / name).read_bytes(), name
    for lab in _labels(first):
        name = f"manifests/{lab['doc_id']}.json"
        assert (first / name).read_bytes() == (again / name).read_bytes(), name
