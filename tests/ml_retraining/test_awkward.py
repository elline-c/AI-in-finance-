"""Task 17.1: failing tests for the Awkward_Set (module is built in task 17.2).

Pinned interface of ``fraud_detect.ml.retraining.awkward`` (everything synthetic):

    AWKWARD_VERSION: str                    # "awkward-v1"
    AWKWARD_MASTER_SEED = 20260701          # same value as config.AWKWARD_MASTER_SEED
    CASE_TYPES = ("seasonal_growth", "legit_resave", "instalments",
                  "long_12_months", "round_payroll_rent")
    TAMPER_ROTATION = ("edited_amount", "fabricated_receipts", "metadata_inconsistency")
    make_company                            # module-level name (spy-able), from corpus.company
    class AwkwardBuildError(Exception):     # attrs: case_type, n_genuine, n_tampered;
                                            # str() names the type and both counts
    @dataclass(frozen=True) class AwkwardDoc:
        doc_id, case_type, company_id ("AW-<case>-<k>"), uen, layout, is_tampered,
        tamper_type (None if genuine), pdf (bytes), ledger (ledger of the rendered doc),
        company (CompanyProfile)
    @dataclass(frozen=True) class AwkwardSet:
        version, seed, docs (tuple[AwkwardDoc], case-major order)
        genuine() / tampered() / by_case(case) -> tuple[AwkwardDoc]
    build_awkward_set(seed=AWKWARD_MASTER_SEED, *, uen_checker=None, knobs=None,
                      cases=None, max_attempts=12) -> AwkwardSet
        knobs: {case: SimulatorKnobs} override; cases: subset of CASE_TYPES;
        a case whose figure cannot be met in max_attempts raises AwkwardBuildError.
    render_document(plan, *, master_seed, company=None, knobs=None) -> (pdf_bytes, ledger)
        same bytes as CorpusBuilder._build_document for a standard document.
    @dataclass(frozen=True) class AwkwardOutcome:
        ok, exit_code, message, awkward_set, proceed_to_switch_rule
    prepare_awkward_set(*, uen_checker=None, **builder_kwargs) -> AwkwardOutcome
        never raises AwkwardBuildError; on failure ok False, exit_code 1,
        proceed_to_switch_rule False, message names the case type and counts.
    read_review_record(report: Mapping, version=AWKWARD_VERSION) -> dict | None
        report["awkward_review"] = {"reviewer", "date", "version"}
    awkward_results_status(report, version=AWKWARD_VERSION) -> "reviewed" | "unreviewed"

Seasonal cases are chosen by measurement (seeds/phases, existing knobs only) and only
cases meeting both figures are kept. The module is imported inside tests/fixtures so a
missing module fails these tests only.
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import importlib
import io
import os
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "fraud_detect" / "ml" / "retraining"
MODULE = "fraud_detect.ml.retraining.awkward"
MODULE_FILE = PKG / "awkward.py"
CASES = ("seasonal_growth", "legit_resave", "instalments", "long_12_months",
         "round_payroll_rent")
ROTATION = ("edited_amount", "fabricated_receipts", "metadata_inconsistency")
TRAINING_MODULES = ("features_a", "features_b", "modelops", "cutoffs", "leak", "neutralise",
                    "audit", "size_guard", "style_profile", "corpus_gates", "corpora",
                    "group_checks", "sources")


def _aw():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def built():
    aw = _aw()
    return aw.build_awkward_set()


@pytest.fixture(scope="module")
def built_again():
    aw = _aw()
    return aw.build_awkward_set()


# ------------------------------------------------------------------ constants
def test_version_and_seed_are_recorded():
    aw = _aw()
    from fraud_detect.ml.retraining import config
    assert isinstance(aw.AWKWARD_VERSION, str) and aw.AWKWARD_VERSION
    assert aw.AWKWARD_MASTER_SEED == 20260701 == config.AWKWARD_MASTER_SEED
    assert tuple(aw.CASE_TYPES) == CASES
    assert tuple(aw.TAMPER_ROTATION) == ROTATION


def test_set_records_version_and_seed(built):
    aw = _aw()
    assert built.version == aw.AWKWARD_VERSION
    assert built.seed == aw.AWKWARD_MASTER_SEED


# ------------------------------------------------------------------ structure
def test_counts_per_case_and_in_total(built):
    assert len(built.genuine()) == 15 and len(built.tampered()) == 15
    for case in CASES:
        docs = built.by_case(case)
        assert sum(not d.is_tampered for d in docs) >= 3, case
        assert sum(d.is_tampered for d in docs) >= 3, case
    assert {d.case_type for d in built.docs} == set(CASES)


def test_ids_follow_the_pattern_and_are_unique(built):
    assert len({d.doc_id for d in built.docs}) == len(built.docs)
    for d in built.docs:
        assert re.fullmatch(rf"AW-{d.case_type}-\d+", d.company_id), d.company_id
        assert d.doc_id.startswith(d.company_id)
        assert d.pdf[:5] == b"%PDF-"


def test_every_tampered_doc_pairs_with_a_genuine_doc(built):
    for case in CASES:
        genuine = {d.company_id: d for d in built.by_case(case) if not d.is_tampered}
        for t in (d for d in built.by_case(case) if d.is_tampered):
            g = genuine[t.company_id]
            assert (t.company.uen, t.layout) == (g.company.uen, g.layout)
            assert t.uen == g.uen
            assert t.company == g.company
            assert t.pdf != g.pdf


def test_tamper_types_rotate_one_per_tampered_doc(built):
    tampered = built.tampered()
    for i, d in enumerate(tampered):
        assert d.tamper_type == ROTATION[i % 3], (i, d.tamper_type)
    assert all(d.tamper_type is None for d in built.genuine())


# --------------------------------------------------------------------- figures
def _credits(ledger):
    out = [0.0] * len(ledger.months)
    for r in ledger.rows:
        if r.amount > 0:
            out[r.month_index] += r.amount
    return out


def test_seasonal_growth_figures_hold_on_every_genuine_ledger(built):
    for d in built.by_case("seasonal_growth"):
        if d.is_tampered:
            continue
        c = _credits(d.ledger)
        assert max(c) >= 2.0 * min(c), d.doc_id
        assert sum(c[-3:]) >= 1.10 * sum(c[:3]), d.doc_id


def test_legit_resave_has_export_producer_and_later_moddate(built):
    from pypdf import PdfReader
    from fraud_detect.corpus.tamper.pdf_ops import EXPORT_PRODUCERS
    for d in built.by_case("legit_resave"):
        if d.is_tampered:
            continue
        meta = PdfReader(io.BytesIO(d.pdf)).metadata
        assert meta.producer in EXPORT_PRODUCERS, d.doc_id
        assert meta.modification_date > meta.creation_date, d.doc_id


def test_instalment_docs_have_an_invoice_ref_on_two_receipts(built):
    for d in built.by_case("instalments"):
        if d.is_tampered:
            continue
        counts = {}
        for r in d.ledger.rows:
            if r.kind == "receipt" and r.invoice_ref:
                counts[r.invoice_ref] = counts.get(r.invoice_ref, 0) + 1
        assert any(n >= 2 for n in counts.values()), d.doc_id


def test_long_docs_cover_exactly_12_consecutive_months(built):
    for d in built.by_case("long_12_months"):
        months = d.ledger.months
        assert len(months) == 12, d.doc_id
        idx = [m.year * 12 + m.month for m in months]
        assert idx == list(range(idx[0], idx[0] + 12)), d.doc_id


def test_round_payroll_and_rent_share_is_at_least_80_percent(built):
    for d in built.by_case("round_payroll_rent"):
        if d.is_tampered:
            continue
        rows = [r for r in d.ledger.rows if r.kind in ("payroll", "rent")]
        assert rows, d.doc_id
        whole = sum(round(abs(r.amount) * 100) % 10000 == 0 for r in rows)
        assert whole / len(rows) >= 0.80, d.doc_id


# ------------------------------------------------------------------ determinism
def _fingerprint(s):
    return [(d.doc_id, d.company_id, d.uen, d.layout, d.tamper_type,
             hashlib.sha256(d.pdf).hexdigest()) for d in s.docs]


def test_two_builds_with_the_same_seed_are_identical(built, built_again):
    assert _fingerprint(built) == _fingerprint(built_again)


# ------------------------------------------------------------------------ UENs
def test_uens_are_drawn_through_make_company_with_the_checker(monkeypatch):
    aw = _aw()
    from fraud_detect.corpus.company import make_company as real
    from fraud_detect.corpus.uen_check import UenChecker
    case = ("round_payroll_rent",)
    free = aw.build_awkward_set(cases=case)
    taken = {d.uen for d in free.docs}
    checker = UenChecker(taken)
    calls = []

    def spy(seed, company_id, is_taken=None):
        calls.append(is_taken)
        return real(seed, company_id, is_taken=is_taken)

    monkeypatch.setattr(aw, "make_company", spy)
    redrawn = aw.build_awkward_set(cases=case, uen_checker=checker)
    assert calls and all(c == checker.is_taken for c in calls)
    assert not ({d.uen for d in redrawn.docs} & taken)
    assert checker.redraws >= 1
    # only the UEN moves: layouts and company ids are unchanged
    assert [(d.company_id, d.layout) for d in redrawn.docs] == \
           [(d.company_id, d.layout) for d in free.docs]


# ------------------------------------------------------------ in-memory building
def test_build_creates_no_file_or_folder(tmp_path, monkeypatch):
    aw = _aw()
    aw.build_awkward_set(cases=("round_payroll_rent",))  # warm imports
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            raise AssertionError(f"build opened {file!r} for writing")
        return real_open(file, mode, *a, **k)

    def boom(*a, **k):
        raise AssertionError("build created a temporary file or folder")

    monkeypatch.setattr(builtins, "open", guarded_open)
    for owner, name in ((os, "mkdir"), (os, "makedirs"), (tempfile, "mkdtemp"),
                        (tempfile, "mkstemp"), (tempfile, "NamedTemporaryFile"),
                        (tempfile, "TemporaryDirectory")):
        monkeypatch.setattr(owner, name, boom)
    aw.build_awkward_set(cases=("round_payroll_rent",))
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_source_uses_memory_only_and_no_pdf_is_committed():
    assert MODULE_FILE.exists(), "awkward.py not implemented yet (task 17.2)"
    text = MODULE_FILE.read_text()
    for banned in ("tempfile", "write_bytes", "write_text", "mkdir"):
        assert banned not in text, banned
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
    assert not [p for p in out.splitlines() if "awkward" in p.lower() and p.endswith(".pdf")]


def test_builder_reproduces_corpus_builder_for_a_standard_document(tmp_path):
    aw = _aw()
    from fraud_detect.corpus.builder import CorpusBuilder
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.schema import BuildConfig
    cfg = BuildConfig(out_dir=tmp_path / "c", master_seed=20260899, per_layout_genuine=1,
                      per_layout_tampered=0, months_min=6, months_max=6,
                      command_line="pytest awkward")
    builder = CorpusBuilder(cfg)
    plan = next(p for p in builder.plan() if p.tamper_type is None)
    company = make_company(builder._company_seed(plan), plan.company_id)
    builder._build_document(plan, tmp_path / "c", company)
    expected = (tmp_path / "c" / "pdfs" / f"{plan.doc_id}.pdf").read_bytes()
    pdf, ledger = aw.render_document(plan, master_seed=cfg.master_seed, company=company)
    assert pdf == expected
    assert len(ledger.months) == 6


# ------------------------------------------------------------------ build failure
def test_unreachable_figure_raises_awkward_build_error_naming_type_and_counts():
    aw = _aw()
    from fraud_detect.corpus.schema import SimulatorKnobs
    flat = SimulatorKnobs(seasonality_strength=0.0, growth_range=(-0.01, -0.01))
    with pytest.raises(aw.AwkwardBuildError) as exc:
        aw.build_awkward_set(cases=("seasonal_growth",), knobs={"seasonal_growth": flat},
                             max_attempts=3)
    err = exc.value
    assert err.case_type == "seasonal_growth"
    assert err.n_genuine < 3 or err.n_tampered < 3
    text = str(err)
    assert "seasonal_growth" in text and str(err.n_genuine) in text and str(err.n_tampered) in text


def test_pipeline_stops_before_switch_rule_and_leaves_current_model_unchanged(tmp_path):
    aw = _aw()
    from fraud_detect.corpus.schema import SimulatorKnobs
    model = tmp_path / "current.joblib"
    model.write_bytes(b"current model bytes")
    before = hashlib.sha256(model.read_bytes()).hexdigest()
    flat = SimulatorKnobs(seasonality_strength=0.0, growth_range=(-0.01, -0.01))
    out = aw.prepare_awkward_set(cases=("seasonal_growth",),
                                 knobs={"seasonal_growth": flat}, max_attempts=3)
    assert out.ok is False and out.exit_code == 1
    assert out.proceed_to_switch_rule is False and out.awkward_set is None
    assert "seasonal_growth" in out.message
    assert hashlib.sha256(model.read_bytes()).hexdigest() == before


def test_prepare_succeeds_on_a_buildable_set():
    aw = _aw()
    out = aw.prepare_awkward_set(cases=("round_payroll_rent",))
    assert out.ok and out.exit_code == 0 and out.proceed_to_switch_rule
    assert out.awkward_set is not None


# ----------------------------------------------------------------- review record
def test_review_record_is_read_from_the_report():
    aw = _aw()
    record = {"reviewer": "A. Owner", "date": "2026-07-02", "version": aw.AWKWARD_VERSION}
    report = {"awkward_review": record}
    assert aw.read_review_record(report) == record
    assert aw.awkward_results_status(report) == "reviewed"


def test_results_are_unreviewed_without_a_record_for_the_current_version():
    aw = _aw()
    assert aw.read_review_record({}) is None
    assert aw.awkward_results_status({}) == "unreviewed"
    old = {"awkward_review": {"reviewer": "A. Owner", "date": "2026-07-02",
                              "version": "awkward-v0"}}
    assert aw.read_review_record(old) is None
    assert aw.awkward_results_status(old) == "unreviewed"


# ------------------------------------------------------------------ source scans
def _imports_awkward(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[-1] == "awkward":
                return True
            if any(a.name == "awkward" for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name.split(".")[-1] == "awkward" for a in node.names):
                return True
    return False


def test_training_modules_do_not_import_the_awkward_set():
    checked = 0
    for name in TRAINING_MODULES:
        path = PKG / f"{name}.py"
        if path.exists():
            checked += 1
            assert not _imports_awkward(ast.parse(path.read_text())), name
    assert checked >= 5
    assert MODULE_FILE.exists(), "awkward.py not implemented yet (task 17.2)"
