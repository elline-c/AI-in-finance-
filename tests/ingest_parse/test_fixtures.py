"""Sanity checks that the in-code fixtures are self-consistent."""
from __future__ import annotations

import pytest

from tests.ingest_parse import fixtures as fx
from tests.ingest_parse.conftest import (
    ANSWER_KEY, DATASET_DIR, STATEMENT_E, STATEMENT_F,
    requires_dataset, requires_labelled_statements,
)


@pytest.mark.parametrize("name", fx.PROFILE_NAMES)
def test_fixture_rows_replay_to_closing(name):
    f = fx.ALL_FIXTURES[name]()
    assert f.n_txns == len(fx.SAMPLE_TXNS)
    for sec in f.sections:
        assert round(sec.opening + sum(t.amount for t in sec.txns), 2) == sec.closing
    for pg in f.pages:
        assert pg.words and pg.lines
        for w in pg.words:
            assert w.x1 > w.x0 and w.top > 0


def test_continuation_page_has_no_period_line():
    f = fx.continuation_fixture()
    assert len(f.pages) == 2
    assert "Statement Period" in f.pages[0].text
    assert "Statement Period" not in f.pages[1].text
    assert "BROUGHT FORWARD" not in f.pages[1].text


def test_six_page_file_says_page_1_of_1_everywhere():
    f = fx.six_page_page1of1_fixture()
    assert len(f.pages) == 6 and len(f.sections) == 6
    assert all("Page 1 of 1" in p.text for p in f.pages)
    assert f.sections[1].opening == f.sections[0].closing


def test_right_edges_align_with_headers():
    f = fx.two_col_dmon_fixture()
    hdr = {w.text: w for w in f.pages[0].words if abs(w.top - 183) < 1}
    amt = [w for w in f.pages[0].words if w.text.startswith("S$") and abs(w.top - 212.4) < 1]
    assert amt and amt[0].x1 in (hdr["out"].x1, hdr["in"].x1)


def test_external_path_constants():
    assert DATASET_DIR.name == "statement_forgery_dataset"
    assert STATEMENT_E.name == "statement_E.pdf" and STATEMENT_F.name == "statement_F.pdf"
    assert ANSWER_KEY.name == "ANSWER_KEY_6M_v2.json"


@requires_dataset
def test_requires_dataset_marker_runs_when_present():
    assert DATASET_DIR.is_dir()


@requires_labelled_statements
def test_requires_labelled_marker_runs_when_present():
    assert STATEMENT_E.is_file()
