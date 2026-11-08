"""Node 2 fixes: currency notation judged on transaction amounts only, and a
'modified after it was created' metadata check that ignores plain re-saves.

Hand-built IngestionResults: no PDFs needed. Dataset/labelled-file checks skip when
the external files are missing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fraud_detect.models import DocumentMetadata, IngestionResult, PageText
from fraud_detect.nodes import visual_forensics as vf
from fraud_detect.nodes.ingestion import ingest

HEADER = "Vellichor Bank\nBUSINESS CURRENT ACCOUNT Currency SGD\nOpening Balance Total Deposits\nS$1,000.00 S$500.00\n"


def _ing(rows: str, header: str = HEADER, **meta) -> IngestionResult:
    text = header + rows
    return IngestionResult(
        source_path="x.pdf", page_count=1, raw_text=text,
        page_texts=[PageText(page=1, text=text)],
        metadata=DocumentMetadata(**meta),
    )


def _codes(ing) -> set[str]:
    return {f.code for f in vf.analyze(ing).findings}


# ---- currency notation: transaction amounts only ---------------------------------------
def test_header_currency_label_is_not_counted():
    rows = "1 Apr GIRO RENT S$5,365.00 S$34,514.66\n4 Apr FAST FROM X INV3263 S$6,341.56 S$40,856.22\n"
    assert "VF_CURRENCY_MIX" not in _codes(_ing(rows))


def test_summary_box_is_not_counted():
    header = "Currency SGD\nOpening Balance SGD 1,000.00 Closing Balance SGD 2,000.00\n"
    rows = "1 Apr GIRO RENT S$5,365.00 S$34,514.66\n"
    assert "VF_CURRENCY_MIX" not in _codes(_ing(rows, header))


def test_mixed_notation_in_rows_is_flagged():
    rows = "01/04/2026 GIRO RENT SGD-5,365.00 SGD34,514.66\n04/04/2026 FAST FROM X S$6,341.56 S$40,856.22\n"
    assert "VF_CURRENCY_MIX" in _codes(_ing(rows))


def test_usd_row_among_sgd_rows_is_flagged():
    rows = "2026-04-01 GIRO RENT SGD 5,365.00 SGD 34,514.66\n2026-04-04 TT FROM X USD 6,341.56 SGD 40,856.22\n"
    assert "VF_CURRENCY_MIX" in _codes(_ing(rows))


def test_signs_before_the_marker_are_read():
    rows = "2026-04-01 A -SGD 13.47 SGD 100.00\n2026-04-02 B +SGD 11.08 SGD 111.08\n"
    assert "VF_CURRENCY_MIX" not in _codes(_ing(rows))


def test_amounts_without_any_marker_add_no_style():
    rows = "01-04-26 A 1,200.00 5,000.00 CR\n02-04-26 B S$13.47 5,013.47 CR\n"
    assert "VF_CURRENCY_MIX" not in _codes(_ing(rows))


def test_lines_that_are_not_transaction_rows_are_ignored():
    rows = "Note: fees are charged in USD 10.00 and SGD 5.00\n1 Apr GIRO RENT S$5,365.00 S$34,514.66\n"
    assert "VF_CURRENCY_MIX" not in _codes(_ing(rows))


def test_falls_back_to_raw_text_without_page_texts():
    text = HEADER + "01/04/2026 A SGD 1.00 SGD 2.00\n02/04/2026 B S$3.00 S$5.00\n"
    ing = IngestionResult(source_path="x.pdf", page_count=1, raw_text=text, metadata=DocumentMetadata())
    assert "VF_CURRENCY_MIX" in _codes(ing)


def test_generator_planted_format_inconsistency_is_still_caught(tmp_path):
    from fraud_detect.synthetic import generate_tampered
    gen = generate_tampered(str(tmp_path), seed=3, issues=["FORMAT_INCONSISTENCY"])
    assert "VF_CURRENCY_MIX" in _codes(ingest(gen.path))


def test_generator_clean_statement_has_no_currency_finding(tmp_path):
    from fraud_detect.synthetic import generate_clean
    assert "VF_CURRENCY_MIX" not in _codes(ingest(generate_clean(str(tmp_path)).path))


# ---- metadata: editing software --------------------------------------------------------
@pytest.mark.parametrize("producer", [
    "Foxit PDF Editor 12", "PDF-XChange Editor", "Sejda PDF", "QuickPDF Editor 7.4", "pdfescape online",
])
def test_known_pdf_editors_raise_edit_software(producer):
    got = _codes(_ing("", producer=producer))
    assert "VF_EDIT_SOFTWARE" in got


@pytest.mark.parametrize("producer", [
    "Adobe PDF Library 15.0", "Corvane StatementEngine 4.2", "ReportLab PDF Library", "Vellichor Paperless 1.12",
])
def test_ordinary_bank_producers_do_not(producer):
    assert "VF_EDIT_SOFTWARE" not in _codes(_ing("", producer=producer))


# ---- metadata: modified after created ---------------------------------------------------
CREATED, LATER = "D:20261003023600+08'00'", "D:20261005133600+08'00'"


def test_later_moddate_from_a_non_print_tool_is_flagged_medium_14():
    ing = _ing("", producer="OpenPage Writer 3.8", creator="OpenPage Writer 3.8",
               creation_date=CREATED, mod_date=LATER)
    (f,) = [f for f in vf.analyze(ing).findings if f.code == "VF_MODIFIED_AFTER_CREATION"]
    assert f.severity.value == "MEDIUM" and f.weight == 14


@pytest.mark.parametrize("producer", [
    "BrowserPrint PDF 118", "PrintToPDF Driver 10.0", "SaveAsPDF Mobile 3.2",
    "OS Quartz-style PDF Export", "Skia/PDF m120", "Microsoft: Print To PDF",
])
def test_plain_resave_tools_are_not_flagged(producer):
    ing = _ing("", producer=producer, creation_date=CREATED, mod_date=LATER)
    assert "VF_MODIFIED_AFTER_CREATION" not in _codes(ing)


def test_equal_dates_are_not_flagged():
    ing = _ing("", producer="Vellichor Paperless 1.12", creation_date=CREATED, mod_date=CREATED)
    assert "VF_MODIFIED_AFTER_CREATION" not in _codes(ing)


def test_a_gap_of_a_minute_or_less_is_not_flagged_but_longer_is():
    base = "D:20261003023600"
    assert "VF_MODIFIED_AFTER_CREATION" not in _codes(
        _ing("", producer="Some Engine", creation_date=base, mod_date="D:20261003023700"))
    assert "VF_MODIFIED_AFTER_CREATION" in _codes(
        _ing("", producer="Some Engine", creation_date=base, mod_date="D:20261003023701"))


def test_same_day_edit_is_caught_because_the_time_is_read():
    ing = _ing("", producer="Some Engine", creation_date="D:20261003080000", mod_date="D:20261003173000")
    assert "VF_MODIFIED_AFTER_CREATION" in _codes(ing)


@pytest.mark.parametrize("created,modified", [(None, LATER), (CREATED, None), (None, None),
                                              ("garbage", "garbage"), (LATER, CREATED)])
def test_missing_unreadable_or_reversed_dates_are_not_flagged(created, modified):
    ing = _ing("", producer="Some Engine", creation_date=created, mod_date=modified)
    assert "VF_MODIFIED_AFTER_CREATION" not in _codes(ing)


def test_named_editor_is_not_double_counted():
    ing = _ing("", producer="Foxit PDF Editor", creation_date=CREATED, mod_date=LATER)
    got = _codes(ing)
    assert "VF_EDIT_SOFTWARE" in got and "VF_MODIFIED_AFTER_CREATION" not in got


def test_unknown_producer_is_treated_as_not_a_resave_tool():
    ing = _ing("", producer=None, creator=None, creation_date=CREATED, mod_date=LATER)
    assert "VF_MODIFIED_AFTER_CREATION" in _codes(ing)


# ---- labelled files and dataset (skip when missing) --------------------------------------
DOWNLOADS = Path.home() / "Downloads"
STATEMENT_E, STATEMENT_F = DOWNLOADS / "statement_E.pdf", DOWNLOADS / "statement_F.pdf"
DATASET = DOWNLOADS / "statement_forgery_dataset"


@pytest.mark.skipif(not (STATEMENT_E.is_file() and STATEMENT_F.is_file()), reason="statement_E/F not found")
def test_statement_e_is_flagged_and_statement_f_is_clean():
    e, f = _codes(ingest(str(STATEMENT_E))), _codes(ingest(str(STATEMENT_F)))
    assert "VF_MODIFIED_AFTER_CREATION" in e and "VF_CURRENCY_MIX" not in e
    assert f == set()


@pytest.mark.skipif(not (DATASET / "labels.csv").is_file(), reason="forgery dataset not found")
def test_dataset_resaved_genuine_documents_raise_no_node2_metadata_or_currency_finding():
    rows = list(csv.DictReader(open(DATASET / "labels.csv")))
    resaved = [r for r in rows if r["is_tampered"] == "0" and r.get("genuine_resaved") == "1"]
    plain = [r for r in rows if r["is_tampered"] == "0" and r.get("genuine_resaved") != "1"][:60]
    assert len(resaved) == 64
    for r in resaved + plain:
        got = _codes(ingest(str(DATASET / r["file"])))
        assert not got & {"VF_CURRENCY_MIX", "VF_MODIFIED_AFTER_CREATION", "VF_EDIT_SOFTWARE"}, r["file"]


@pytest.mark.skipif(not (DATASET / "labels.csv").is_file(), reason="forgery dataset not found")
def test_dataset_metadata_tampered_documents_are_flagged():
    rows = [r for r in csv.DictReader(open(DATASET / "labels.csv"))
            if r.get("tamper_type") == "metadata_inconsistency"][:60]
    assert rows
    for r in rows:
        got = _codes(ingest(str(DATASET / r["file"])))
        assert got & {"VF_MODIFIED_AFTER_CREATION", "VF_EDIT_SOFTWARE"}, r["file"]
