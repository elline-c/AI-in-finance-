"""Node 1 part of the verification report (task 22.1).

Written first; fails with ImportError until task 22.2 adds
``fraud_detect/corpus/verify.py``. Covers Requirements 14.1 to 14.9.

The corpus used here is tiny and built by hand (no builder import) in the output
layout: ``pdfs/<doc_id>.pdf``, ``manifests/<doc_id>.json``, ``labels.csv``,
``rows.csv``. Two layouts (generator_v1, two_col_slash), each with two genuine
documents and one tampered document (``edited_amount`` by overlay).

verify API chosen here (names from the design, details fixed by these tests)
---------------------------------------------------------------------------
``load_corpus(corpus_dir) -> list[DocRecord]``
    Reads ``labels.csv`` plus ``manifests/<doc_id>.json``. Sorted by
    ``(layout, doc_id)``. ``DocRecord`` attributes used here: ``doc_id``,
    ``layout``, ``path`` (Path of the PDF), ``is_tampered`` (bool),
    ``genuine_resaved`` (bool), ``tamper_method`` (str, "" if genuine),
    ``manifest`` (the manifest dict).
``sample_records(records, n) -> list[DocRecord]``
    At most ``n`` records per layout, the first ones by ``doc_id`` (fixed order).
``measure_node1(records, ingest=<Node 1 ingest>) -> list[Node1Result]``
    ``ingest(path_str)`` returns an ``IngestionResult``; one result per record, in
    record order. ``Node1Result`` attributes: ``doc_id``, ``layout``,
    ``is_tampered``, ``genuine_resaved``, ``rows_expected`` (manifest rows),
    ``rows_read`` (manifest rows matched, in order, by date and signed amount
    within 0.005 against Node 1's transactions), ``profile`` (``layout_profile``),
    ``balance_ok`` (Node 1 opening + sum of its amounts == Node 1 closing),
    ``header_ok`` (holder == registered name, UEN, first and last period date
    equal the manifest), ``finding_codes`` (sorted codes of
    ``data_quality_findings``), ``dropped_overlay_rows`` (overlay documents only:
    per date, Node 1 returned fewer transactions than the manifest has rows,
    capped by the number of affected rows on that date; 0 otherwise),
    ``error`` (exception type name or None; an ingest exception does not stop
    the run, the document counts as 0 rows read).
``summarise_layout(results) -> dict`` (results of ONE layout) with keys
    ``profile``, ``documents``, ``rows_share``, ``rows_share_genuine``,
    ``rows_share_tampered`` (None when that group is empty), ``balance_share``,
    ``header_share``, ``finding_codes`` (sorted union), ``dropped_overlay_rows``.
    Shares are fractions in 0..1. "Genuine" means not ``is_tampered`` (re-saved
    genuine documents included).
``main(argv=None, *, corpus_dir=None, out=None, ingest=..., screen=...) -> int``
    Options ``--corpus DIR``, ``--sample N``, ``--json FILE``, ``--markdown FILE``.
    Prints one ``layout <name>: ...`` line per layout, a line
    ``overlay rows dropped by Node 1: <n>`` and a line starting ``elapsed``.
    A missing corpus folder prints a line starting ``skipped`` and returns 0.
    Writes nothing unless ``--json`` / ``--markdown`` is given. The Node 1 part
    never calls ``screen`` (Nodes 2 to 4 are measured by a later task).
"""
from __future__ import annotations

import dataclasses
import io
import json
import re
import shutil
from datetime import date
from pathlib import Path

import pytest

from fraud_detect.corpus.verify import (  # noqa: E402  (ImportError until 22.2)
    load_corpus,
    main,
    measure_node1,
    sample_records,
    summarise_layout,
)
from fraud_detect.corpus.labels import (
    build_manifest, write_labels_csv, write_manifest, write_rows_csv)
from fraud_detect.corpus.schema import DocumentPlan, TamperLabel
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.models import DocumentMetadata, IngestionResult, Transaction
from tests.corpus.helpers import DEFAULT_START, DEFAULT_STYLE, load_renderer, make_test_ledger

LAYOUTS_USED = ("generator_v1", "two_col_slash")
MASTER = 4242
REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# a tiny corpus built by hand
# --------------------------------------------------------------------------
def _plan(layout: str, n: int, seed: int, tamper: str | None) -> DocumentPlan:
    return DocumentPlan(
        doc_id=f"{layout}-{n:04d}", index=n, layout=layout, company_id="C0001",
        company_key="C0001", split="train", start=DEFAULT_START, n_months=2,
        doc_seed=seed, tamper_type=tamper)


def _genuine(layout: str, n: int, seed: int):
    renderer = load_renderer(layout)
    ledger = make_test_ledger(2, seed, getattr(renderer, "max_rows_per_month", None))
    rendered = renderer.render(ledger, DEFAULT_STYLE, seed)
    return _plan(layout, n, seed, None), ledger, rendered, TamperLabel(), ()


def _overlaid(layout: str, n: int, seed0: int):
    """An ``edited_amount`` document made by overlay (scan seeds for that variant)."""
    renderer = load_renderer(layout)
    for k in range(60):
        seed = seed0 + k
        ledger = make_test_ledger(2, seed, getattr(renderer, "max_rows_per_month", None))
        result = CATALOG["edited_amount"].plan(ledger, renderer, rng(seed, "verify-n1", layout))
        if result.label.method == "overlay":
            base = renderer.render(result.ledger, DEFAULT_STYLE, seed)
            pdf = apply_file_ops(base.pdf, result.file_ops, base.row_map, seed)
            rendered = dataclasses.replace(base, pdf=pdf)
            return (_plan(layout, n, seed, "edited_amount"), result.ledger, rendered,
                    result.label, result.file_ops)
    raise AssertionError("no overlay variant found")


def _write_corpus(root: Path) -> None:
    docs = []
    for i, layout in enumerate(LAYOUTS_USED):
        docs.append(_genuine(layout, 1, 100 + i))
        docs.append(_genuine(layout, 2, 200 + i))
        docs.append(_overlaid(layout, 3, 300 + i))
    manifests = []
    for plan, ledger, rendered, label, file_ops in docs:
        (root / "pdfs").mkdir(parents=True, exist_ok=True)
        (root / "pdfs" / f"{plan.doc_id}.pdf").write_bytes(rendered.pdf)
        m = build_manifest(plan, ledger, rendered, label, master_seed=MASTER, file_ops=file_ops)
        write_manifest(root / "manifests" / f"{plan.doc_id}.json", m)
        manifests.append(m)
    write_labels_csv(root / "labels.csv", manifests)
    write_rows_csv(root / "rows.csv", manifests)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    for layout in LAYOUTS_USED:
        if load_renderer(layout) is None:
            pytest.skip(f"renderer {layout} not implemented yet")
    root = tmp_path_factory.mktemp("verify_n1_corpus")
    _write_corpus(root)
    return root


@pytest.fixture
def corpus(tiny_corpus, tmp_path) -> Path:
    """A private copy per test, so a test can never disturb another."""
    dst = tmp_path / "corpus"
    shutil.copytree(tiny_corpus, dst)
    return dst


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------
# injected ingest helpers
# --------------------------------------------------------------------------
def _fake_ingestion(record, *, drop: set[str] = frozenset(), profile: str | None = None,
                    holder: str | None = None, codes: tuple[str, ...] = ()) -> IngestionResult:
    """What Node 1 would return for the manifest, optionally missing some rows."""
    m = record.manifest
    rows = m["rows"]
    kept = [r for r in rows if r["row_id"] not in drop]
    first_month = date.fromisoformat(m["months"][0] + "-01")
    ly, lm = (int(x) for x in m["months"][-1].split("-"))
    nxt = date(ly + (lm == 12), lm % 12 + 1, 1)
    last_day = date.fromordinal(nxt.toordinal() - 1)
    from fraud_detect.models import Finding, Severity
    return IngestionResult(
        source_path=str(record.path), page_count=1, raw_text="", metadata=DocumentMetadata(),
        account_holder=holder if holder is not None else m["company"]["registered_name"],
        uen=m["company"]["uen"], statement_period_start=first_month,
        statement_period_end=last_day,
        opening_balance=round(rows[0]["balance"] - rows[0]["amount"], 2),
        closing_balance=rows[-1]["balance"],
        layout_profile=profile or m["layout"],
        transactions=[Transaction(index=i, date=date.fromisoformat(r["date"]),
                                  amount=r["amount"], balance=r["balance"])
                      for i, r in enumerate(kept)],
        data_quality_findings=[Finding(agent="node1", code=c, title=c, severity=Severity.LOW,
                                       evidence=c) for c in codes])


def _perfect(record):
    return _fake_ingestion(record)


def _records_by_id(records):
    return {r.doc_id: r for r in records}


# --------------------------------------------------------------------------
# load_corpus
# --------------------------------------------------------------------------
def test_load_corpus_reads_every_document_in_layout_then_id_order(corpus):
    records = load_corpus(corpus)
    assert [(r.layout, r.doc_id) for r in records] == sorted(
        (r.layout, r.doc_id) for r in records)
    assert len(records) == 6
    r = _records_by_id(records)["generator_v1-0003"]
    assert r.is_tampered is True and r.tamper_method == "overlay"
    assert r.path == corpus / "pdfs" / "generator_v1-0003.pdf" and r.path.is_file()
    assert r.manifest["rows"]
    g = _records_by_id(records)["two_col_slash-0001"]
    assert g.is_tampered is False and g.tamper_method == "" and g.genuine_resaved is False


# --------------------------------------------------------------------------
# measure_node1 and summarise_layout with injected ingest (in memory)
# --------------------------------------------------------------------------
def test_perfect_reader_gives_full_shares_per_layout(corpus):
    records = load_corpus(corpus)
    results = measure_node1(records, ingest=lambda p: _perfect(
        next(r for r in records if str(r.path) == str(p))))
    assert [x.doc_id for x in results] == [r.doc_id for r in records]
    for layout in LAYOUTS_USED:
        s = summarise_layout([x for x in results if x.layout == layout])
        assert s["documents"] == 3
        assert s["profile"] == layout
        assert s["rows_share"] == s["balance_share"] == s["header_share"] == 1.0
        assert s["rows_share_genuine"] == s["rows_share_tampered"] == 1.0
        assert s["dropped_overlay_rows"] == 0
        assert s["finding_codes"] == []


def test_dropped_overlay_row_lowers_only_the_tampered_share_and_is_counted(corpus):
    records = load_corpus(corpus)
    by_path = {str(r.path): r for r in records}
    victim = _records_by_id(records)["generator_v1-0003"]
    affected = victim.manifest["label"]["affected_rows"][0]["row_id"]

    def ingest(p):
        rec = by_path[str(p)]
        return _fake_ingestion(rec, drop={affected} if rec is victim else set())

    results = measure_node1(records, ingest=ingest)
    one = [x for x in results if x.doc_id == "generator_v1-0003"][0]
    assert one.rows_expected == len(victim.manifest["rows"])
    assert one.rows_read == one.rows_expected - 1
    assert one.dropped_overlay_rows == 1
    assert sum(x.dropped_overlay_rows for x in results if x is not one) == 0

    s = summarise_layout([x for x in results if x.layout == "generator_v1"])
    assert s["rows_share_genuine"] == 1.0
    assert s["rows_share_tampered"] == pytest.approx((one.rows_expected - 1) / one.rows_expected)
    expected = sum(x.rows_expected for x in results if x.layout == "generator_v1")
    assert s["rows_share"] == pytest.approx((expected - 1) / expected)
    assert s["dropped_overlay_rows"] == 1
    assert one.balance_ok is False      # replay of the kept rows no longer reaches the closing


def test_header_mismatch_and_profile_and_finding_codes_are_reported(corpus):
    records = load_corpus(corpus)
    by_path = {str(r.path): r for r in records}

    def ingest(p):
        rec = by_path[str(p)]
        if rec.doc_id == "two_col_slash-0001":
            return _fake_ingestion(rec, holder="SOMEONE ELSE PTE. LTD.", profile="two_col_dmon",
                                   codes=("DQ_TEST_A",))
        return _fake_ingestion(rec, codes=("DQ_TEST_B",) if rec.doc_id.endswith("0002") else ())

    results = measure_node1(records, ingest=ingest)
    s = summarise_layout([x for x in results if x.layout == "two_col_slash"])
    assert s["header_share"] == pytest.approx(2 / 3)
    assert s["balance_share"] == 1.0 and s["rows_share"] == 1.0
    assert s["finding_codes"] == ["DQ_TEST_A", "DQ_TEST_B"]
    assert "two_col_dmon" in s["profile"] and "two_col_slash" in s["profile"]


def test_ingest_exception_is_recorded_and_the_run_continues(corpus):
    records = load_corpus(corpus)
    by_path = {str(r.path): r for r in records}

    def ingest(p):
        rec = by_path[str(p)]
        if rec.doc_id == "generator_v1-0001":
            raise RuntimeError("boom")
        return _perfect(rec)

    results = measure_node1(records, ingest=ingest)
    bad = [x for x in results if x.doc_id == "generator_v1-0001"][0]
    assert bad.error == "RuntimeError" and bad.rows_read == 0
    assert all(x.error is None for x in results if x is not bad)


def test_summarise_layout_share_is_none_for_an_empty_group(corpus):
    records = [r for r in load_corpus(corpus) if not r.is_tampered and r.layout == "generator_v1"]
    results = measure_node1(records, ingest=_perfect)
    s = summarise_layout(results)
    assert s["rows_share_tampered"] is None and s["rows_share_genuine"] == 1.0


# --------------------------------------------------------------------------
# --sample
# --------------------------------------------------------------------------
def test_sample_records_keeps_the_first_n_per_layout_by_doc_id(corpus):
    records = load_corpus(corpus)
    got = sample_records(list(reversed(records)), 2)
    assert [(r.layout, r.doc_id) for r in got] == [
        ("generator_v1", "generator_v1-0001"), ("generator_v1", "generator_v1-0002"),
        ("two_col_slash", "two_col_slash-0001"), ("two_col_slash", "two_col_slash-0002")]
    assert len(sample_records(records, 99)) == 6


def test_main_sample_measures_at_most_n_documents_per_layout(corpus):
    seen = []
    records = {str(r.path): r for r in load_corpus(corpus)}

    def ingest(p):
        seen.append(Path(p).name)
        return _perfect(records[str(p)])

    out = io.StringIO()
    assert main(["--corpus", str(corpus), "--sample", "1"], out=out, ingest=ingest) == 0
    assert seen == ["generator_v1-0001.pdf", "two_col_slash-0001.pdf"]
    assert re.search(r"layout generator_v1:.*documents=1\b", out.getvalue())


# --------------------------------------------------------------------------
# main: output, exit codes, no writes
# --------------------------------------------------------------------------
def test_main_prints_a_line_per_layout_the_drop_count_and_elapsed(corpus):
    records = {str(r.path): r for r in load_corpus(corpus)}
    victim = records[str(corpus / "pdfs" / "generator_v1-0003.pdf")]
    affected = victim.manifest["label"]["affected_rows"][0]["row_id"]

    def ingest(p):
        rec = records[str(p)]
        return _fake_ingestion(rec, drop={affected} if rec is victim else set())

    out = io.StringIO()
    assert main(["--corpus", str(corpus)], out=out, ingest=ingest) == 0
    text = out.getvalue()
    for layout in LAYOUTS_USED:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"layout {layout}:"))
        for token in ("documents=3", f"profile={layout}", "rows=", "genuine_rows=100.0%",
                      "tampered_rows=", "balance=", "header=", "findings="):
            assert token in line
    assert "overlay rows dropped by Node 1: 1" in text
    assert re.search(r"^elapsed\b", text, re.M)


def test_main_uses_the_corpus_dir_keyword_and_injected_functions(corpus):
    records = {str(r.path): r for r in load_corpus(corpus)}
    calls = {"ingest": 0, "screen": 0}

    def ingest(p):
        calls["ingest"] += 1
        return _perfect(records[str(p)])

    def screen(*a, **k):      # a harmless screen: the Nodes 2 to 4 part now calls it
        calls["screen"] += 1
        return type("R", (), {"routing": "AUTO_CLEAR", "findings": []})()

    out = io.StringIO()
    assert main([], corpus_dir=corpus, out=out, ingest=ingest, screen=screen) == 0
    assert calls == {"ingest": 6, "screen": 6}


def test_main_missing_corpus_prints_skipped_and_returns_zero(tmp_path):
    out = io.StringIO()
    code = main(["--corpus", str(tmp_path / "does-not-exist")], out=out,
                ingest=lambda p: pytest.fail("ingest must not run"))
    assert code == 0
    assert out.getvalue().startswith("skipped")


def test_main_writes_nothing_without_json_or_markdown(corpus, tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    data_before = _tree(REPO / "data") if (REPO / "data").is_dir() else {}
    before = _tree(corpus)
    records = {str(r.path): r for r in load_corpus(corpus)}
    assert main(["--corpus", str(corpus)], out=io.StringIO(),
                ingest=lambda p: _perfect(records[str(p)])) == 0
    assert _tree(corpus) == before
    assert list(work.rglob("*")) == []
    assert (_tree(REPO / "data") if (REPO / "data").is_dir() else {}) == data_before
    assert not (work / "data" / "cases.json").exists()


def test_main_writes_json_and_markdown_only_when_named(corpus, tmp_path):
    records = {str(r.path): r for r in load_corpus(corpus)}
    js, md = tmp_path / "out" / "r.json", tmp_path / "out" / "r.md"
    js.parent.mkdir()
    assert main(["--corpus", str(corpus), "--json", str(js), "--markdown", str(md)],
                out=io.StringIO(), ingest=lambda p: _perfect(records[str(p)])) == 0
    assert sorted(p.name for p in js.parent.iterdir()) == ["r.json", "r.md"]
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "generator_v1" in json.dumps(data) and "two_col_slash" in json.dumps(data)
    assert "generator_v1" in md.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# real Node 1 on the tiny corpus
# --------------------------------------------------------------------------
def test_real_node1_reads_genuine_documents_completely_and_overlay_drops_are_bounded(corpus):
    records = load_corpus(corpus)
    results = measure_node1(records)
    assert len(results) == 6
    for layout in LAYOUTS_USED:
        group = [x for x in results if x.layout == layout]
        s = summarise_layout(group)
        assert s["profile"] == layout
        assert s["rows_share_genuine"] == 1.0
        assert s["balance_share"] >= 2 / 3 and s["header_share"] >= 2 / 3
        for x in group:
            assert x.error is None
            if not x.is_tampered:
                assert x.rows_read == x.rows_expected and x.balance_ok and x.header_ok
                assert x.finding_codes == [] or all(isinstance(c, str) for c in x.finding_codes)
                assert x.dropped_overlay_rows == 0
            else:
                assert 0 <= x.dropped_overlay_rows <= 1
                assert x.dropped_overlay_rows <= x.rows_expected - x.rows_read


def test_main_with_real_node1_exits_zero_and_prints_elapsed(corpus):
    out = io.StringIO()
    assert main(["--corpus", str(corpus)], out=out) == 0
    assert re.search(r"^elapsed\b", out.getvalue(), re.M)
