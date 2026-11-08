"""Task 12.3: property tests for genuine labels and synthetic marking.

Feature: synthetic-corpus-generator, Property 9: Labels tell the truth (genuine part)
Feature: synthetic-corpus-generator, Property 21: Documents are marked synthetic and use only built-in fonts

Each example renders a PDF, so the example budget is small.
"""
from __future__ import annotations

import csv

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_detect.corpus.labels import (
    build_manifest,
    text_layer_sha256,
    write_labels_csv,
    write_rows_csv,
)
from fraud_detect.corpus.schema import LAYOUTS, DocumentPlan, TamperLabel
from tests.corpus.helpers import (
    DEFAULT_START,
    fonts_of,
    is_builtin_font,
    load_renderer,
    metadata_of,
    render_and_ingest,
)

SETTINGS = settings(
    max_examples=4, deadline=None, derandomize=True,
    suppress_health_check=list(HealthCheck),
)
seeds = st.integers(min_value=0, max_value=10_000)
months = st.integers(min_value=6, max_value=12)

pytestmark = pytest.mark.parametrize("layout", LAYOUTS)


def _doc(layout, seed, n_months):
    renderer = load_renderer(layout)
    if renderer is None:
        pytest.skip(f"layout {layout!r} is not implemented yet")
    r = render_and_ingest(renderer, n_months, seed, ingest=False)
    plan = DocumentPlan(
        doc_id=f"{layout}-0001", index=1, layout=layout,
        company_id=r.ledger.company.company_id, company_key=r.ledger.company.company_id,
        split="train", start=DEFAULT_START, n_months=n_months, doc_seed=seed,
    )
    m = build_manifest(plan, r.ledger, r.result, TamperLabel(), master_seed=1)
    return r, m


@SETTINGS
@given(seed=seeds, n_months=months)
def test_property_9_genuine_labels_tell_the_truth(layout, seed, n_months, tmp_path_factory):
    """**Validates: Requirements 4.3, 4.4, 4.5, 9.6, 9.7**"""
    r, m = _doc(layout, seed, n_months)

    # every printed row is listed, in order, none is a fraud row
    assert [x["row_id"] for x in m["rows"]] == [p.row_id for p in r.result.printed]
    assert len(m["rows"]) == len(r.ledger.rows)
    assert not any(x["is_fraud_row"] for x in m["rows"])
    assert all(x["fraud_type"] == "" for x in m["rows"])

    lab = m["label"]
    assert lab["is_tampered"] is False
    assert not lab["tamper_types"]
    assert lab["affected_rows"] == [] and lab["edits"] == [] and lab["file_ops"] == []

    # the recorded text-layer hash equals the hash recomputed from the PDF
    assert m["text_layer_sha256"] == text_layer_sha256(r.result.pdf)

    # labels.csv / rows.csv agree with the manifest: fraud rows == n_affected_rows == 0
    d = tmp_path_factory.mktemp("lab")
    write_labels_csv(d / "labels.csv", [m])
    write_rows_csv(d / "rows.csv", [m])
    with (d / "labels.csv").open(newline="") as fh:
        line = next(csv.DictReader(fh))
    with (d / "rows.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert line["is_tampered"] == "0" and line["tamper_types"] == ""
    assert len(rows) == int(line["n_rows"]) == len(m["rows"])
    assert sum(x["is_fraud_row"] == "1" for x in rows) == int(line["n_affected_rows"]) == 0


@SETTINGS
@given(seed=seeds, n_months=months)
def test_property_21_marked_synthetic_and_builtin_fonts(layout, seed, n_months):
    """**Validates: Requirements 12.8, 18.5**"""
    r, m = _doc(layout, seed, n_months)
    assert m["synthetic"] is True
    meta = metadata_of(r.result.pdf)
    title, subject = meta.get("/Title", ""), meta.get("/Subject", "")
    assert "synthetic" in title.lower(), meta
    assert "synthetic" in subject.lower(), meta
    fonts = fonts_of(r.result.pdf)
    assert fonts
    bad = sorted(f for f in fonts if not is_builtin_font(f))
    assert bad == [], bad
