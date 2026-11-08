"""Extra checks for ``parse_pages`` / ``ingest()`` added with task 9.2.

* generator_v1 keeps rows whose date format was switched to yyyy-mm-dd (the
  generator's "mixed date formats" tamper), as the old fixed regex did.
* A secondary date style never claims a page on its own.
* A blank page gives DQ_NO_TEXT but no DQ_LAYOUT_UNRECOGNISED.
* A section without printed closing/totals gets them from its parsed rows in
  the SectionSummary only (the self-check still sees no printed totals).
* ingest() stamps each segment with the file's basename.
"""
from __future__ import annotations

import os
from dataclasses import replace

from fraud_detect.nodes import ingestion
from fraud_detect.nodes.ingest_parse.parser import parse_pages
from fraud_detect.nodes.ingest_parse.types import PageContent
from fraud_detect.synthetic import generate_clean, generate_tampered

from .fixtures import generator_v1_fixture, signed_amount_iso_fixture


def _iso(ddmmyyyy: str) -> str:
    d, m, y = ddmmyyyy.split("/")
    return f"{y}-{m}-{d}"


def test_generator_v1_keeps_rows_with_switched_date_format():
    fx = generator_v1_fixture()
    page = fx.pages[0]
    target = fx.sections[0].txns[2].date.strftime("%d/%m/%Y")
    switched = PageContent(
        page=page.page,
        lines=[replace(ln, text=ln.text.replace(target, _iso(target))) for ln in page.lines],
        words=[replace(w, text=_iso(target)) if w.text == target else w for w in page.words],
    )
    res = parse_pages([switched])
    assert res.layout_profile == "generator_v1"
    assert len(res.transactions) == fx.n_txns
    assert res.transactions[2].date_raw == _iso(target)
    assert res.transactions[2].date == fx.sections[0].txns[2].date


def test_iso_only_layout_is_not_claimed_by_generator_v1():
    assert parse_pages(signed_amount_iso_fixture().pages).layout_profile == "signed_amount_iso"


def test_blank_page_is_no_text_not_layout_problem():
    codes = [f.code for f in parse_pages([PageContent(page=1)]).findings]
    assert "DQ_NO_TEXT" in codes and "DQ_NO_TXNS" in codes
    assert "DQ_LAYOUT_UNRECOGNISED" not in codes


def test_unprinted_summary_comes_from_rows_without_self_check_finding():
    fx = generator_v1_fixture()  # prints no closing and no totals
    res = parse_pages(fx.pages)
    sec = fx.sections[0]
    assert res.sections[0].closing == sec.closing
    assert res.sections[0].total_deposits == sec.total_deposits
    assert "DQ_PARSE_INCONSISTENT" not in [f.code for f in res.findings]


def test_ingest_fills_new_fields_and_segment_source(tmp_path):
    path = generate_clean(str(tmp_path)).path
    ing = ingestion.ingest(path)
    assert ing.layout_profile == "generator_v1"
    assert ing.parse_coverage == 1.0
    assert len(ing.sections) == len(ing.segments) == 1
    assert {s.source for s in ing.segments} == {os.path.basename(path)}
    assert all(t.direction_method == "printed_sign" for t in ing.transactions)


def test_ingest_generator_tampered_reads_mixed_date_rows(tmp_path):
    ing = ingestion.ingest(generate_tampered(str(tmp_path)).path)
    raws = [t.date_raw for t in ing.transactions]
    assert len(raws) == 28  # same count as the old fixed-regex parser
    assert any("-" in r for r in raws) and any("/" in r for r in raws)
    assert "DQ_LAYOUT_UNRECOGNISED" not in [f.code for f in ing.data_quality_findings]
