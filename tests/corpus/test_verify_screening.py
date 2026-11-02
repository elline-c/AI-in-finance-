"""Nodes 2, 3 and 4 part of the verification report (task 22.3).

Written first; fails until task 22.4 extends ``fraud_detect/corpus/verify.py``.
Covers Requirements 15.1 to 15.9 (and 14.8 for the injected ``screen``).

The corpus is tiny and built by hand (no builder import) in the output layout
(``pdfs/``, ``manifests/``, ``labels.csv``, ``rows.csv``, ``companies.json``):
two layouts (generator_v1, two_col_slash), nine documents each, 2 months long:

    -0001, -0002   genuine
    -0003          genuine_resaved   (not tampered, hard negative)
    -0004          edited_amount, method overlay
    -0005          edited_amount, method rerender
    -0006          metadata_inconsistency (file)
    -0007          structuring
    -0008          shell_counterparty
    -0009          uen_mismatch (content)

Most tests inject a fake ``screen`` (fast, exact figures). One test runs the real
Nodes 1 to 4 on the whole tiny corpus.

verify API chosen here (extends the Node 1 API of ``test_verify_node1.py``)
---------------------------------------------------------------------------
``DocRecord`` gains two attributes (from ``labels.csv`` / the manifest):
    ``split`` (str) and ``tamper_type`` (str: first entry of the label's
    ``tamper_types``; "" for plain genuine documents; "genuine_resaved" for the
    hard negatives, whose ``is_tampered`` is still False).

``screen(path_str) -> report``
    Runs Nodes 2, 3 and 4 (the default also runs Node 1 first, in memory, no case
    store) on one PDF. ``path_str`` is the same ``str`` subclass given to
    ``ingest`` (it also exposes ``.manifest``, ``.doc_id`` ...). The report only
    needs ``.routing`` (``Routing`` or its string value) and ``.findings`` (objects
    with ``.agent`` such as ``"Node3:FinancialLogic"``, ``.code``, ``.severity``).
    A real ``RiskReport`` fits.

``measure_nodes_2_to_4(records, screen=<default screen>) -> list[ScreenResult]``
    One result per record, in record order. It does NOT enter the company scope
    (``main`` does, once). An exception from ``screen`` does not stop the run:
    that result has ``error`` = exception type name and ``routing`` = "".
    ``ScreenResult`` attributes: ``doc_id``, ``layout``, ``split``,
    ``is_tampered``, ``genuine_resaved``, ``tamper_type``, ``tamper_method``,
    ``hard_cases`` (manifest ``genuine_hard_cases``), ``routing`` (str),
    ``auto_cleared`` (bool), ``finding_codes`` (sorted unique codes, all severities),
    ``material`` (bool: some finding is MEDIUM or HIGH), ``nodes_fired`` (sorted
    unique ``Node<n>`` prefixes of the MEDIUM/HIGH findings' ``agent``),
    ``expected_codes`` (codes in the label's ``should_be_seen_by``: entries
    ``"Node3:FL_X"`` whose part after the colon looks like ``ABC_DEF``; descriptive
    entries such as ``"Node3:period checks"`` are ignored), ``expected_code_seen``
    (None when ``expected_codes`` is empty, else True if any of them is in
    ``finding_codes``), ``expected_nodes`` (sorted unique ``Node<n>`` prefixes of
    ``should_be_seen_by``), ``expected_node_fired`` (None when ``expected_nodes`` is
    empty, else True if any of them is in ``nodes_fired``), ``error``.

``summarise_genuine(results) -> dict``   (the caller picks the group)
    keys ``documents``, ``auto_clear_share`` (None if empty), ``code_counts``
    (code -> number of documents carrying it), ``screening_errors``,
    ``hard_cases`` = ``{"documents", "auto_clear_share"}`` over the documents with
    a non-empty ``hard_cases``. Documents with an error are counted in
    ``documents`` but left out of every share.

``summarise_screening(results) -> dict``   (one group of any kind)
    ``documents``, ``genuine_documents`` (not tampered, not re-saved),
    ``genuine_auto_clear_share``, ``tampered_documents``, ``material_share``,
    ``not_auto_cleared_share`` (both over the tampered documents; None if none),
    ``routing_counts`` (``{"AUTO_CLEAR": n, "HUMAN_REVIEW": n}``, always both keys),
    ``screening_errors``.

``group_screening(results, by) -> dict``    ``by`` is "layout" or "split";
    value -> ``summarise_screening`` of that group, keys sorted.

``detection_by_tamper_type(results) -> list[dict]``
    Tampered documents only (``is_tampered``), one row per (tamper_type, method),
    sorted by that pair. Keys ``tamper_type``, ``method``, ``documents``,
    ``material_share``, ``not_auto_cleared_share``, ``codes`` (sorted union),
    ``expected_codes``, ``expected_code_share`` (None if the type has none),
    ``expected_nodes``, ``expected_node_share`` (None if none), ``nodes_fired``
    (node -> number of documents), ``screening_errors``. Shares leave out error
    documents.

``blind_spots(table, genuine_not_auto_cleared_share) -> list[dict]``
    The rows of ``table`` (same order, same dicts) whose ``material_share`` == 0 or
    whose ``not_auto_cleared_share`` <= the genuine share. A None genuine share
    counts as 0.0. Rows with a None share are never listed by that criterion.

``follow_up_suggestions(blind) -> list[str]``
    Text only; at least one line naming each blind-spot tamper type.

``main(...)`` (extends the Node 1 behaviour; ``--sample`` also limits screening)
    * enters ``verify.registered_companies(<corpus>/companies.json)`` exactly once,
      around all screening (the name is looked up in the module at call time);
    * calls ``screen`` once per measured document, whatever the rates are;
    * exit code stays 0 whatever the detection and false-alarm rates are;
    * prints, besides the Node 1 lines:
        ``genuine: documents=N auto_clear=P% codes=CODE:n,CODE:n``
        ``genuine_resaved: documents=N auto_clear=P% codes=...``
        ``decisions: AUTO_CLEAR=a HUMAN_REVIEW=b``
        ``tamper <type> <method>: documents=N material=P% not_auto_cleared=P%
        codes=A,B expected_code=P%|n/a expected_node=P%|n/a``
        ``blind spot: <type> <method>: material=P% not_auto_cleared=P%``
        ``screening layout <name>: documents=N genuine_auto_clear=P% material=P%
        not_auto_cleared=P%`` and the same for ``screening split <name>:``
        ``screening errors: N (<doc_id>: <ExceptionType>, ...)`` when N > 0
        ``follow-up: <text>`` lines (suggestions only, nothing is implemented)
      percentages use one decimal and "n/a" for None.
    * ``--json`` adds ``payload["screening"]`` with ``genuine``, ``genuine_resaved``,
      ``tamper_types`` (the table), ``blind_spots``, ``layouts``, ``splits``,
      ``decisions`` and ``documents`` (``asdict`` of every ``ScreenResult``);
      ``--markdown`` adds a section whose heading contains "Blind spots".
"""
from __future__ import annotations

import dataclasses
import io
import json
import re
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from fraud_detect.corpus.verify import (  # noqa: E402  (new names: ImportError until 22.4)
    blind_spots,
    detection_by_tamper_type,
    follow_up_suggestions,
    group_screening,
    load_corpus,
    main,
    measure_nodes_2_to_4,
    summarise_genuine,
    summarise_screening,
)
from fraud_detect.corpus import verify
from fraud_detect.corpus.labels import (
    build_manifest, write_labels_csv, write_manifest, write_rows_csv)
from fraud_detect.corpus.schema import DocumentPlan, TamperLabel
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.models import Finding, Routing, Severity
from tests.corpus.helpers import DEFAULT_START, DEFAULT_STYLE, load_renderer, make_test_ledger

LAYOUTS_USED = ("generator_v1", "two_col_slash")
MASTER = 4343
REPO = Path(__file__).resolve().parents[2]
N2, N3 = "Node2:VisualForensics", "Node3:FinancialLogic"

# (doc number, tamper type or None, wanted method or None)
DOC_KINDS = (
    (1, None, None), (2, None, None), (3, "genuine_resaved", None),
    (4, "edited_amount", "overlay"), (5, "edited_amount", "rerender"),
    (6, "metadata_inconsistency", None), (7, "structuring", None),
    (8, "shell_counterparty", None), (9, "uen_mismatch", None),
)


# --------------------------------------------------------------------------
# a tiny corpus built by hand
# --------------------------------------------------------------------------
def _split_of(layout: str, n: int) -> str:
    if n == 1:
        return "train"
    if n == 2:
        return "validation"
    return "train" if layout == "generator_v1" else "test_unseen_layout"


def _build_doc(layout: str, n: int, seed0: int, tamper, method):
    renderer = load_renderer(layout)
    cap = getattr(renderer, "max_rows_per_month", None)
    for k in range(80):
        seed = seed0 + k
        ledger = make_test_ledger(2, seed, cap)
        if tamper is None:
            edited, label, edits, file_ops = ledger, TamperLabel(), (), ()
        else:
            try:
                res = CATALOG[tamper].plan(ledger, renderer, rng(seed, "verify-scr", layout, tamper))
            except Exception:
                continue
            if method and res.label.method != method:
                continue
            edited, label, edits, file_ops = res.ledger, res.label, res.edits, res.file_ops
        base = renderer.render(edited, DEFAULT_STYLE, seed)
        pdf = apply_file_ops(base.pdf, file_ops, base.row_map, seed)
        rendered = dataclasses.replace(base, pdf=pdf)
        plan = DocumentPlan(
            doc_id=f"{layout}-{n:04d}", index=n, layout=layout, company_id="C0001",
            company_key="C0001", split=_split_of(layout, n), start=DEFAULT_START,
            n_months=2, doc_seed=seed, tamper_type=tamper)
        manifest = build_manifest(plan, edited, rendered, label, master_seed=MASTER,
                                  edits=edits, file_ops=file_ops)
        credit = sum(r.amount for r in ledger.rows if r.amount > 0) / 2
        debit = -sum(r.amount for r in ledger.rows if r.amount < 0) / 2
        company = {"company_id": ledger.company.company_id, "uen": ledger.company.uen,
                   "name": ledger.company.name, "registered_name": ledger.company.name,
                   "baseline": {"avg_monthly_credit": round(credit, 2),
                                "avg_monthly_debit": round(debit, 2)}}
        return plan, pdf, manifest, company
    raise AssertionError(f"no {tamper}/{method} variant found for {layout}")


def _write_corpus(root: Path) -> None:
    manifests, companies = [], []
    for li, layout in enumerate(LAYOUTS_USED):
        for n, tamper, method in DOC_KINDS:
            plan, pdf, manifest, company = _build_doc(layout, n, 1000 * (li + 1) + 50 * n,
                                                      tamper, method)
            (root / "pdfs").mkdir(parents=True, exist_ok=True)
            (root / "pdfs" / f"{plan.doc_id}.pdf").write_bytes(pdf)
            write_manifest(root / "manifests" / f"{plan.doc_id}.json", manifest)
            manifests.append(manifest)
            companies.append(company)
    write_labels_csv(root / "labels.csv", manifests)
    write_rows_csv(root / "rows.csv", manifests)
    (root / "companies.json").write_text(json.dumps(companies, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    for layout in LAYOUTS_USED:
        if load_renderer(layout) is None:
            pytest.skip(f"renderer {layout} not implemented yet")
    root = tmp_path_factory.mktemp("verify_screening_corpus")
    _write_corpus(root)
    return root


@pytest.fixture
def corpus(tiny_corpus, tmp_path) -> Path:
    dst = tmp_path / "corpus"
    shutil.copytree(tiny_corpus, dst)
    return dst


def _tree(root: Path):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------
# the fake screen
# --------------------------------------------------------------------------
def _f(agent: str, code: str, severity: Severity) -> Finding:
    return Finding(agent=agent, code=code, title=code, severity=severity, evidence=code)


def _policy(record):
    """(routing, findings) the fake pipeline returns for a record.

    genuine -0001 clean; -0002 a false alarm (generator_v1: MEDIUM, slash: LOW only)
    re-saved: generator_v1 flagged (MEDIUM), slash clean
    edited_amount overlay: LOW finding only, still sent to review
    edited_amount rerender: generator_v1 FL_BALANCE_BREAK HIGH, slash FL_CLOSING_MISMATCH MEDIUM
    metadata_inconsistency: Node2 HIGH + MEDIUM;  structuring: Node3 HIGH
    shell_counterparty: nothing seen;  uen_mismatch: seen in generator_v1 only
    """
    gv1 = record.layout == "generator_v1"
    t = record.tamper_type
    H, A = Routing.HUMAN_REVIEW, Routing.AUTO_CLEAR
    if t == "":
        if record.doc_id.endswith("0001"):
            return A, []
        return (H, [_f(N3, "FL_SUDDEN_DEPOSIT", Severity.MEDIUM)] if gv1
                else [_f(N3, "FL_LOW_NOTE", Severity.LOW)])
    if t == "genuine_resaved":
        return (H, [_f(N2, "VF_MODIFIED_AFTER_CREATION", Severity.MEDIUM)]) if gv1 else (A, [])
    if t == "edited_amount" and record.tamper_method == "overlay":
        return H, [_f(N2, "VF_LOW_NOTE", Severity.LOW)]
    if t == "edited_amount":
        return (H, [_f(N3, "FL_BALANCE_BREAK", Severity.HIGH)]) if gv1 else (
            H, [_f(N3, "FL_CLOSING_MISMATCH", Severity.MEDIUM)])
    if t == "metadata_inconsistency":
        return H, [_f(N2, "VF_EDIT_SOFTWARE", Severity.HIGH),
                   _f(N2, "VF_MODIFIED_AFTER_CREATION", Severity.MEDIUM)]
    if t == "structuring":
        return H, [_f(N3, "FL_STRUCTURING", Severity.HIGH)]
    if t == "uen_mismatch" and gv1:
        return H, [_f(N3, "FL_UEN_NAME_MISMATCH", Severity.HIGH)]
    return A, []                      # shell_counterparty, uen_mismatch on two_col_slash


def _screen_for(records, *, raise_for=(), log=None):
    by_path = {str(r.path): r for r in records}

    def screen(p):
        rec = by_path[str(p)]
        if log is not None:
            log.append("screen")
        if rec.doc_id in raise_for:
            raise ValueError("boom")
        routing, findings = _policy(rec)
        return SimpleNamespace(routing=routing, findings=findings)

    return screen


def _perfect_ingest_for(records):
    from tests.corpus.test_verify_node1 import _fake_ingestion
    by_path = {str(r.path): r for r in records}
    return lambda p: _fake_ingestion(by_path[str(p)])


@pytest.fixture
def results(corpus):
    records = load_corpus(corpus)
    for r in records:                       # a known set of hard cases: only genuine -0002
        r.manifest["genuine_hard_cases"] = (
            ["instalment"] if (r.tamper_type == "" and r.doc_id.endswith("0002")
                               and r.layout == "generator_v1") else [])
    return measure_nodes_2_to_4(records, screen=_screen_for(records))


def _by_id(results):
    return {x.doc_id: x for x in results}


def _run_main(corpus, *extra, screen=None, records=None, out=None):
    records = records if records is not None else load_corpus(corpus)
    out = out if out is not None else io.StringIO()
    code = main(["--corpus", str(corpus), *extra], out=out,
                ingest=_perfect_ingest_for(records), screen=screen or _screen_for(records))
    return code, out.getvalue()


def _line(text: str, prefix: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith(prefix))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def test_records_carry_split_and_tamper_type(corpus):
    by = {r.doc_id: r for r in load_corpus(corpus)}
    assert len(by) == 18
    assert by["generator_v1-0001"].tamper_type == "" and by["generator_v1-0001"].split == "train"
    assert by["generator_v1-0002"].split == "validation"
    assert by["two_col_slash-0003"].tamper_type == "genuine_resaved"
    assert by["two_col_slash-0003"].is_tampered is False
    assert by["two_col_slash-0003"].genuine_resaved is True
    assert by["two_col_slash-0007"].tamper_type == "structuring"
    assert by["two_col_slash-0007"].split == "test_unseen_layout"
    assert by["generator_v1-0004"].tamper_method == "overlay"
    assert by["generator_v1-0005"].tamper_method == "rerender"


# --------------------------------------------------------------------------
# measure_nodes_2_to_4
# --------------------------------------------------------------------------
def test_one_screen_result_per_record_in_order_with_fields(corpus):
    records = load_corpus(corpus)
    log = []
    res = measure_nodes_2_to_4(records, screen=_screen_for(records, log=log))
    assert [x.doc_id for x in res] == [r.doc_id for r in records]
    assert log == ["screen"] * 18
    x = _by_id(res)

    clean = x["generator_v1-0001"]
    assert (clean.routing, clean.auto_cleared, clean.material) == ("AUTO_CLEAR", True, False)
    assert clean.finding_codes == [] and clean.nodes_fired == [] and clean.error is None
    assert clean.layout == "generator_v1" and clean.split == "train"
    assert clean.expected_codes == [] and clean.expected_code_seen is None
    assert clean.expected_nodes == [] and clean.expected_node_fired is None

    meta = x["generator_v1-0006"]
    assert meta.routing == "HUMAN_REVIEW" and meta.auto_cleared is False and meta.material
    assert meta.finding_codes == ["VF_EDIT_SOFTWARE", "VF_MODIFIED_AFTER_CREATION"]
    assert meta.nodes_fired == ["Node2"]
    assert meta.tamper_type == "metadata_inconsistency" and meta.tamper_method == "file"
    assert meta.is_tampered is True and meta.genuine_resaved is False
    assert "VF_EDIT_SOFTWARE" in meta.expected_codes and meta.expected_code_seen is True
    assert meta.expected_nodes == ["Node2"] and meta.expected_node_fired is True

    low = x["generator_v1-0004"]              # LOW only: a finding code, but nothing fired
    assert low.finding_codes == ["VF_LOW_NOTE"] and low.material is False
    assert low.nodes_fired == [] and low.auto_cleared is False
    assert low.expected_code_seen is False and low.expected_node_fired is False

    shell = x["two_col_slash-0008"]           # expected Node3, nothing seen
    assert shell.expected_nodes == ["Node3"] and shell.expected_node_fired is False
    assert shell.expected_codes == ["FL_UNVERIFIED_COUNTERPARTY"]

    resaved = x["generator_v1-0003"]
    assert resaved.genuine_resaved is True and resaved.is_tampered is False
    assert resaved.expected_codes == [] and resaved.expected_code_seen is None


def test_descriptive_expected_entries_are_not_codes(corpus):
    records = [r for r in load_corpus(corpus) if r.tamper_type == "edited_amount"]
    for r in records:
        r.manifest["label"]["should_be_seen_by"] = ["Node3:period checks", "Node2:FL_SOME_CODE"]
    res = measure_nodes_2_to_4(records, screen=_screen_for(records))
    for x in res:
        assert x.expected_codes == ["FL_SOME_CODE"]
        assert x.expected_nodes == ["Node2", "Node3"]


def test_string_routing_is_accepted(corpus):
    records = load_corpus(corpus)[:2]
    res = measure_nodes_2_to_4(
        records, screen=lambda p: SimpleNamespace(routing="HUMAN_REVIEW", findings=[]))
    assert [x.routing for x in res] == ["HUMAN_REVIEW", "HUMAN_REVIEW"]
    assert not any(x.auto_cleared for x in res)


def test_screen_exception_is_recorded_and_the_run_continues(corpus):
    records = load_corpus(corpus)
    log = []
    res = measure_nodes_2_to_4(
        records, screen=_screen_for(records, raise_for={"generator_v1-0007"}, log=log))
    assert len(res) == 18 and log == ["screen"] * 18
    bad = _by_id(res)["generator_v1-0007"]
    assert bad.error == "ValueError" and bad.routing == "" and bad.auto_cleared is False
    assert bad.finding_codes == [] and bad.material is False
    assert [x.doc_id for x in res if x.error] == ["generator_v1-0007"]


# --------------------------------------------------------------------------
# false alarms on genuine documents (15.2)
# --------------------------------------------------------------------------
def _genuine(res):
    return [x for x in res if not x.is_tampered and not x.genuine_resaved]


def test_genuine_figures_exclude_resaved_documents(results):
    g = summarise_genuine(_genuine(results))
    assert g["documents"] == 4
    assert g["auto_clear_share"] == pytest.approx(0.5)
    assert g["code_counts"] == {"FL_LOW_NOTE": 1, "FL_SUDDEN_DEPOSIT": 1}
    assert g["screening_errors"] == 0


def test_resaved_figures_are_reported_separately(results):
    r = summarise_genuine([x for x in results if x.genuine_resaved])
    assert r["documents"] == 2
    assert r["auto_clear_share"] == pytest.approx(0.5)
    assert r["code_counts"] == {"VF_MODIFIED_AFTER_CREATION": 1}


def test_genuine_hard_cases_are_traced(results):
    g = summarise_genuine(_genuine(results))
    assert g["hard_cases"]["documents"] == 1
    assert g["hard_cases"]["auto_clear_share"] == pytest.approx(0.0)


def test_code_counts_count_documents_not_occurrences_and_errors_leave_shares(corpus):
    records = [r for r in load_corpus(corpus) if r.tamper_type == ""]
    twice = SimpleNamespace(routing=Routing.HUMAN_REVIEW, findings=[
        _f(N3, "FL_X", Severity.LOW), _f(N2, "FL_X", Severity.LOW)])
    calls = iter([twice, twice, "raise", SimpleNamespace(routing=Routing.AUTO_CLEAR, findings=[])])

    def screen(p):
        v = next(calls)
        if v == "raise":
            raise KeyError("x")
        return v

    g = summarise_genuine(measure_nodes_2_to_4(records, screen=screen))
    assert g["documents"] == 4 and g["screening_errors"] == 1
    assert g["code_counts"] == {"FL_X": 2}
    assert g["auto_clear_share"] == pytest.approx(1 / 3)       # 1 of the 3 screened


def test_empty_group_has_no_shares():
    g = summarise_genuine([])
    assert g["documents"] == 0 and g["auto_clear_share"] is None and g["code_counts"] == {}


# --------------------------------------------------------------------------
# detection table (15.3, 15.4)
# --------------------------------------------------------------------------
def _table(res):
    return detection_by_tamper_type(res)


def _row(table, t, method):
    return next(r for r in table if (r["tamper_type"], r["method"]) == (t, method))


def test_table_has_one_row_per_type_and_method_for_tampered_documents_only(results):
    table = _table(results)
    keys = [(r["tamper_type"], r["method"]) for r in table]
    assert keys == [("edited_amount", "overlay"), ("edited_amount", "rerender"),
                    ("metadata_inconsistency", "file"), ("shell_counterparty", "rerender"),
                    ("structuring", "rerender"), ("uen_mismatch", "content")]
    assert all(r["documents"] == 2 for r in table)      # genuine and re-saved are not rows


def test_table_shares_codes_and_expected_figures(results):
    table = _table(results)
    ov = _row(table, "edited_amount", "overlay")
    assert (ov["material_share"], ov["not_auto_cleared_share"]) == (0.0, 1.0)
    assert ov["codes"] == ["VF_LOW_NOTE"] and ov["expected_code_share"] == 0.0
    assert ov["expected_node_share"] == 0.0 and ov["nodes_fired"] == {}

    rr = _row(table, "edited_amount", "rerender")
    assert (rr["material_share"], rr["not_auto_cleared_share"]) == (1.0, 1.0)
    assert rr["codes"] == ["FL_BALANCE_BREAK", "FL_CLOSING_MISMATCH"]
    assert rr["expected_codes"] == ["FL_BALANCE_BREAK", "VF_FONT_MISMATCH"]
    assert rr["expected_code_share"] == pytest.approx(0.5)
    assert rr["expected_node_share"] == 1.0 and rr["nodes_fired"] == {"Node3": 2}

    md = _row(table, "metadata_inconsistency", "file")
    assert md["codes"] == ["VF_EDIT_SOFTWARE", "VF_MODIFIED_AFTER_CREATION"]
    assert md["expected_code_share"] == 1.0 and md["nodes_fired"] == {"Node2": 2}

    sh = _row(table, "shell_counterparty", "rerender")
    assert (sh["material_share"], sh["not_auto_cleared_share"]) == (0.0, 0.0)
    assert sh["codes"] == [] and sh["expected_code_share"] == 0.0
    assert sh["expected_node_share"] == 0.0

    ue = _row(table, "uen_mismatch", "content")
    assert ue["material_share"] == pytest.approx(0.5)
    assert ue["not_auto_cleared_share"] == pytest.approx(0.5)
    assert ue["expected_code_share"] == pytest.approx(0.5)
    assert ue["expected_node_share"] == pytest.approx(0.5)
    st = _row(table, "structuring", "rerender")
    assert st["material_share"] == st["expected_code_share"] == 1.0


def test_expected_share_is_none_when_the_type_has_no_descriptive_code(corpus):
    records = [r for r in load_corpus(corpus) if r.tamper_type == "structuring"]
    for r in records:
        r.manifest["label"]["should_be_seen_by"] = []
    table = _table(measure_nodes_2_to_4(records, screen=_screen_for(records)))
    row = _row(table, "structuring", "rerender")
    assert row["expected_codes"] == [] and row["expected_code_share"] is None
    assert row["expected_node_share"] is None


def test_table_shares_leave_out_screening_errors(corpus):
    records = load_corpus(corpus)
    res = measure_nodes_2_to_4(
        records, screen=_screen_for(records, raise_for={"generator_v1-0007"}))
    st = _row(_table(res), "structuring", "rerender")
    assert st["documents"] == 2 and st["screening_errors"] == 1
    assert st["material_share"] == 1.0 and st["not_auto_cleared_share"] == 1.0


# --------------------------------------------------------------------------
# blind spots (15.5)
# --------------------------------------------------------------------------
def _fake_row(name, material, not_auto):
    return {"tamper_type": name, "method": "m", "documents": 4,
            "material_share": material, "not_auto_cleared_share": not_auto}


def test_blind_spot_rule_is_exactly_zero_material_or_not_above_genuine():
    rows = [_fake_row("a_zero_material", 0.0, 1.0),       # listed: 0% material
            _fake_row("b_equal_genuine", 0.5, 0.25),      # listed: equals genuine
            _fake_row("c_below_genuine", 0.25, 0.0),      # listed: below genuine
            _fake_row("d_just_above", 0.5, 0.2501),       # not listed
            _fake_row("e_all_seen", 1.0, 1.0)]            # not listed
    listed = blind_spots(rows, 0.25)
    assert [r["tamper_type"] for r in listed] == [
        "a_zero_material", "b_equal_genuine", "c_below_genuine"]
    assert listed[0] is rows[0]


def test_blind_spot_with_no_genuine_documents_uses_zero():
    rows = [_fake_row("x", 0.5, 0.0), _fake_row("y", 0.5, 0.1)]
    assert [r["tamper_type"] for r in blind_spots(rows, None)] == ["x"]


def test_blind_spots_of_the_tiny_corpus(results):
    genuine = summarise_screening(_genuine(results))
    assert genuine["genuine_documents"] == 4
    listed = blind_spots(_table(results), 1 - genuine["genuine_auto_clear_share"])
    assert [(r["tamper_type"], r["method"]) for r in listed] == [
        ("edited_amount", "overlay"), ("shell_counterparty", "rerender"),
        ("uen_mismatch", "content")]


# --------------------------------------------------------------------------
# per layout, per split, decisions (15.8)
# --------------------------------------------------------------------------
def test_decision_breakdown_counts_every_routing(results):
    s = summarise_screening(results)
    assert s["documents"] == 18 and s["screening_errors"] == 0
    assert s["routing_counts"] == {"AUTO_CLEAR": 6, "HUMAN_REVIEW": 12}
    assert summarise_screening([])["routing_counts"] == {"AUTO_CLEAR": 0, "HUMAN_REVIEW": 0}


def test_figures_per_layout(results):
    g = group_screening(results, "layout")
    assert list(g) == ["generator_v1", "two_col_slash"]
    a, b = g["generator_v1"], g["two_col_slash"]
    assert (a["documents"], a["genuine_documents"], a["tampered_documents"]) == (9, 2, 6)
    assert a["genuine_auto_clear_share"] == pytest.approx(0.5)
    assert a["material_share"] == pytest.approx(4 / 6)
    assert a["not_auto_cleared_share"] == pytest.approx(5 / 6)
    assert a["routing_counts"] == {"AUTO_CLEAR": 2, "HUMAN_REVIEW": 7}
    assert b["material_share"] == pytest.approx(3 / 6)
    assert b["not_auto_cleared_share"] == pytest.approx(4 / 6)
    assert b["routing_counts"] == {"AUTO_CLEAR": 4, "HUMAN_REVIEW": 5}


def test_figures_per_split_show_the_held_out_layout(results):
    g = group_screening(results, "split")
    assert list(g) == ["test_unseen_layout", "train", "validation"]
    held = g["test_unseen_layout"]
    assert held["documents"] == 7 and held["genuine_documents"] == 0
    assert held["genuine_auto_clear_share"] is None
    assert held["material_share"] == pytest.approx(3 / 6)
    assert held["not_auto_cleared_share"] == pytest.approx(4 / 6)
    val = g["validation"]
    assert val["documents"] == 2 and val["genuine_auto_clear_share"] == pytest.approx(0.0)
    assert val["tampered_documents"] == 0 and val["material_share"] is None
    assert g["train"]["documents"] == 9


# --------------------------------------------------------------------------
# follow-up suggestions (15.7)
# --------------------------------------------------------------------------
def test_follow_up_suggestions_name_each_blind_spot_type(results):
    genuine = summarise_screening(_genuine(results))
    listed = blind_spots(_table(results), 1 - genuine["genuine_auto_clear_share"])
    lines = follow_up_suggestions(listed)
    assert lines and all(isinstance(s, str) and s for s in lines)
    for r in listed:
        assert any(r["tamper_type"] in s for s in lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def test_main_prints_genuine_resaved_decisions_and_tamper_lines(corpus):
    code, text = _run_main(corpus)
    assert code == 0

    g = _line(text, "genuine:")
    for token in ("documents=4", "auto_clear=50.0%", "FL_LOW_NOTE:1", "FL_SUDDEN_DEPOSIT:1"):
        assert token in g
    r = _line(text, "genuine_resaved:")
    for token in ("documents=2", "auto_clear=50.0%", "VF_MODIFIED_AFTER_CREATION:1"):
        assert token in r
    assert "AUTO_CLEAR=6" in _line(text, "decisions:") and "HUMAN_REVIEW=12" in _line(
        text, "decisions:")

    rr = _line(text, "tamper edited_amount rerender:")
    for token in ("documents=2", "material=100.0%", "not_auto_cleared=100.0%",
                  "FL_BALANCE_BREAK", "FL_CLOSING_MISMATCH", "expected_code=50.0%",
                  "expected_node=100.0%"):
        assert token in rr
    assert "material=0.0%" in _line(text, "tamper shell_counterparty rerender:")
    assert len([ln for ln in text.splitlines() if ln.startswith("tamper ")]) == 6
    assert not any(ln.startswith("tamper genuine") for ln in text.splitlines())


def test_main_prints_na_for_missing_expected_code(corpus):
    # main() reloads the corpus from disk, so edit the manifests in the corpus copy
    for r in load_corpus(corpus):
        if r.tamper_type == "structuring":
            path = corpus / "manifests" / f"{r.doc_id}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["label"]["should_be_seen_by"] = []
            path.write_text(json.dumps(data), encoding="utf-8")
    code, text = _run_main(corpus)
    line = _line(text, "tamper structuring rerender:")
    assert "expected_code=n/a" in line and "expected_node=n/a" in line


def test_main_prints_blind_spots_and_follow_ups(corpus):
    _, text = _run_main(corpus)
    blind = [ln for ln in text.splitlines() if ln.startswith("blind spot:")]
    assert [ln.split(":")[1].strip() for ln in blind] == [
        "edited_amount overlay", "shell_counterparty rerender", "uen_mismatch content"]
    assert "material=0.0%" in blind[0] and "not_auto_cleared=50.0%" in blind[2]
    follow = [ln for ln in text.splitlines() if ln.startswith("follow-up:")]
    assert follow
    for name in ("edited_amount", "shell_counterparty", "uen_mismatch"):
        assert any(name in ln for ln in follow)


def test_main_prints_layout_and_split_figures(corpus):
    _, text = _run_main(corpus)
    a = _line(text, "screening layout generator_v1:")
    for token in ("documents=9", "genuine_auto_clear=50.0%", "material=66.7%",
                  "not_auto_cleared=83.3%"):
        assert token in a
    assert "documents=9" in _line(text, "screening layout two_col_slash:")
    held = _line(text, "screening split test_unseen_layout:")
    for token in ("documents=7", "genuine_auto_clear=n/a", "material=50.0%"):
        assert token in held
    assert "screening split train:" in text and "screening split validation:" in text


def test_main_runs_screen_once_per_document_inside_one_scope(corpus, monkeypatch):
    records = load_corpus(corpus)
    log, seen = [], []

    @contextmanager
    def fake_scope(companies):
        seen.append(companies)
        log.append("enter")
        try:
            yield
        finally:
            log.append("exit")

    monkeypatch.setattr(verify, "registered_companies", fake_scope)
    code, _ = _run_main(corpus, screen=_screen_for(records, log=log), records=records)
    assert code == 0
    assert log == ["enter"] + ["screen"] * 18 + ["exit"]
    assert len(seen) == 1 and Path(seen[0]) == corpus / "companies.json"


def test_main_scope_is_left_when_a_screen_raises(corpus, monkeypatch):
    records = load_corpus(corpus)
    log = []

    @contextmanager
    def fake_scope(companies):
        log.append("enter")
        try:
            yield
        finally:
            log.append("exit")

    monkeypatch.setattr(verify, "registered_companies", fake_scope)
    code, _ = _run_main(corpus, screen=_screen_for(records, raise_for={"generator_v1-0002"},
                                                   log=log), records=records)
    assert code == 0 and log.count("enter") == 1 and log[-1] == "exit"


def test_main_counts_screening_errors_names_the_exception_and_continues(corpus):
    records = load_corpus(corpus)
    log = []
    code, text = _run_main(
        corpus, records=records,
        screen=_screen_for(records, raise_for={"generator_v1-0007"}, log=log))
    assert code == 0 and len(log) == 18
    line = _line(text, "screening errors:")
    assert "1" in line and "generator_v1-0007" in line and "ValueError" in line
    assert "documents=4" in _line(text, "genuine:")       # the rest is still reported


def test_main_exit_code_is_zero_whatever_the_rates(corpus):
    records = load_corpus(corpus)
    none_seen = lambda p: SimpleNamespace(routing=Routing.AUTO_CLEAR, findings=[])
    all_seen = lambda p: SimpleNamespace(routing=Routing.HUMAN_REVIEW, findings=[
        _f(N3, "FL_BALANCE_BREAK", Severity.HIGH)])
    for screen in (none_seen, all_seen):
        code, text = _run_main(corpus, screen=screen, records=records)
        assert code == 0
        assert re.search(r"^elapsed\b", text, re.M)
    _, text = _run_main(corpus, screen=none_seen, records=records)
    assert len([ln for ln in text.splitlines() if ln.startswith("blind spot:")]) == 6
    assert "auto_clear=100.0%" in _line(text, "genuine:")
    _, text = _run_main(corpus, screen=all_seen, records=records)
    assert "auto_clear=0.0%" in _line(text, "genuine:")


def test_main_sample_limits_screening_too(corpus):
    records = load_corpus(corpus)
    seen = []
    base = _screen_for(records)

    def screen(p):
        seen.append(Path(p).name)
        return base(p)

    _, text = _run_main(corpus, "--sample", "1", screen=screen, records=records)
    assert seen == ["generator_v1-0001.pdf", "two_col_slash-0001.pdf"]
    assert "documents=2" in _line(text, "genuine:")


def test_main_missing_corpus_does_not_screen(tmp_path):
    out = io.StringIO()
    code = main(["--corpus", str(tmp_path / "nope")], out=out,
                ingest=lambda p: pytest.fail("ingest must not run"),
                screen=lambda p: pytest.fail("screen must not run"))
    assert code == 0 and out.getvalue().startswith("skipped")


def test_main_writes_nothing_without_json_or_markdown(corpus, tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    data_before = _tree(REPO / "data") if (REPO / "data").is_dir() else {}
    before = _tree(corpus)
    code, _ = _run_main(corpus)
    assert code == 0 and _tree(corpus) == before and list(work.rglob("*")) == []
    assert (_tree(REPO / "data") if (REPO / "data").is_dir() else {}) == data_before


def test_main_json_and_markdown_include_the_screening_part(corpus, tmp_path):
    js, md = tmp_path / "out" / "r.json", tmp_path / "out" / "r.md"
    js.parent.mkdir()
    code, _ = _run_main(corpus, "--json", str(js), "--markdown", str(md))
    assert code == 0 and sorted(p.name for p in js.parent.iterdir()) == ["r.json", "r.md"]
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "node1" in data
    s = data["screening"]
    assert set(s) >= {"genuine", "genuine_resaved", "tamper_types", "blind_spots", "layouts",
                      "splits", "decisions", "documents"}
    assert s["genuine"]["documents"] == 4 and s["genuine_resaved"]["documents"] == 2
    assert len(s["tamper_types"]) == 6 and len(s["blind_spots"]) == 3
    assert len(s["documents"]) == 18 and s["decisions"]["HUMAN_REVIEW"] == 12
    assert set(s["layouts"]) == set(LAYOUTS_USED)
    assert "blind spots" in md.read_text(encoding="utf-8").lower()


# --------------------------------------------------------------------------
# one tiny real run: Nodes 1 to 4 on the hand-built corpus
# --------------------------------------------------------------------------
def test_real_screening_of_the_tiny_corpus_uses_the_company_scope(corpus, tmp_path):
    from fraud_detect import reference_data
    acra_before = dict(reference_data.ACRA_REGISTRY)
    base_before = dict(reference_data.CASHFLOW_BASELINE)
    js = tmp_path / "real.json"
    out = io.StringIO()

    assert main(["--corpus", str(corpus), "--json", str(js)], out=out) == 0

    assert reference_data.ACRA_REGISTRY == acra_before
    assert reference_data.CASHFLOW_BASELINE == base_before
    text = out.getvalue()
    assert re.search(r"^genuine:", text, re.M) and re.search(r"^decisions:", text, re.M)
    assert re.search(r"^elapsed\b", text, re.M)

    docs = json.loads(js.read_text(encoding="utf-8"))["screening"]["documents"]
    assert len(docs) == 18
    assert [d["doc_id"] for d in docs if d["error"]] == []
    codes = Counter(c for d in docs for c in d["finding_codes"])
    assert codes["FL_UEN_NOT_FOUND"] == 0            # the companies were registered
    assert all(d["routing"] in ("AUTO_CLEAR", "HUMAN_REVIEW") for d in docs)
    meta = [d for d in docs if d["tamper_type"] == "metadata_inconsistency"]
    assert len(meta) == 2 and all("Node2" in d["nodes_fired"] for d in meta)
