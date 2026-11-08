"""Task 12.1: manifests and label files for genuine documents (fail until task 12.2).

Interface assumed for ``fraud_detect/corpus/labels.py`` (task 12.2 implements it):

* ``text_layer_sha256(pdf_bytes) -> str``  (SHA-256 hex of the words and positions)
* ``build_manifest(plan, ledger, rendered, label, *, master_seed) -> dict``
  (``rendered`` is the ``RenderResult``; ``label`` a ``TamperLabel``)
* ``write_manifest(path, manifest)``  (JSON, sorted keys, fixed formatting)
* ``write_labels_csv(path, manifests)`` / ``write_rows_csv(path, manifests)``
  (header = ``LABELS_COLUMNS`` / ``ROWS_COLUMNS``; ``file`` = ``pdfs/<doc_id>.pdf``)
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date

import pytest

from fraud_detect.corpus.labels import (  # noqa: F401  (task 12.2)
    build_manifest,
    text_layer_sha256,
    write_labels_csv,
    write_manifest,
    write_rows_csv,
)
from fraud_detect.corpus.schema import (
    GENERATOR_VERSION,
    LABELS_COLUMNS,
    ROWS_COLUMNS,
    DocumentPlan,
    SimulatorKnobs,
    TamperLabel,
)
from tests.corpus.helpers import (
    DEFAULT_START,
    DEFAULT_STYLE,
    load_renderer,
    make_test_ledger,
    render_and_ingest,
    words_of,
)

LAYOUT = "generator_v1"
SEED = 4321
MASTER = 777


def _plan(ledger, layout=LAYOUT, seed=SEED):
    return DocumentPlan(
        doc_id=f"{layout}-0001", index=1, layout=layout,
        company_id=ledger.company.company_id, company_key=ledger.company.company_id,
        split="train", start=DEFAULT_START, n_months=len(ledger.months), doc_seed=seed,
    )


@pytest.fixture(scope="module")
def doc():
    """Genuine document: ledger, rendered result, Node 1 result, manifest."""
    if load_renderer(LAYOUT) is None:
        pytest.skip("generator_v1 renderer not available")
    r = render_and_ingest(LAYOUT, 6, SEED)
    manifest = build_manifest(_plan(r.ledger), r.ledger, r.result, TamperLabel(), master_seed=MASTER)
    return r, manifest


def test_lists_every_printed_row_with_required_fields(doc):
    r, m = doc
    rows = m["rows"]
    assert [x["row_id"] for x in rows] == [p.row_id for p in r.result.printed]
    assert len(rows) == len(r.ledger.rows)
    by_id = {row.row_id: row for row in r.ledger.rows}
    printed = {p.row_id: p for p in r.result.printed}
    for x in rows:
        led = by_id[x["row_id"]]
        assert {"row_id", "row_index", "page", "month", "date", "description",
                "amount", "balance", "is_fraud_row"} <= set(x)
        assert x["page"] == printed[x["row_id"]].page
        assert x["month"] == led.date.strftime("%Y-%m")
        assert x["date"] == led.date.isoformat()
        assert x["description"] == led.description
        assert x["amount"] == pytest.approx(led.amount)
        assert x["balance"] == pytest.approx(printed[x["row_id"]].balance)
        assert x["is_fraud_row"] is False


def test_row_index_is_one_based_and_matches_node1_order(doc):
    r, m = doc
    assert [x["row_index"] for x in m["rows"]] == list(range(1, len(m["rows"]) + 1))
    txns = r.ingestion.transactions
    assert len(txns) == len(m["rows"])
    for x, t in zip(m["rows"], txns):
        assert t.date == date.fromisoformat(x["date"])
        assert t.amount == pytest.approx(x["amount"], abs=0.005)
        assert t.balance == pytest.approx(x["balance"], abs=0.005)


def test_genuine_label_has_no_tamper(doc):
    _, m = doc
    lab = m["label"]
    assert lab["is_tampered"] is False
    assert not lab["tamper_types"]
    assert lab["arithmetic_consistent"] is True
    assert lab["affected_rows"] == []
    assert not any(x["is_fraud_row"] for x in m["rows"])


def test_text_layer_hash_matches_pdf(doc):
    r, m = doc
    h = text_layer_sha256(r.result.pdf)
    assert m["text_layer_sha256"] == h
    assert len(h) == 64 and int(h, 16) >= 0
    assert text_layer_sha256(r.result.pdf) == h               # stable
    other = render_and_ingest(LAYOUT, 6, SEED + 1, ingest=False)
    assert text_layer_sha256(other.result.pdf) != h            # sensitive to content
    assert h != hashlib.sha256(r.result.pdf).hexdigest()       # a text-layer hash, not a file hash
    assert words_of(r.result.pdf)                              # sanity


def test_manifest_marks_synthetic_and_version(doc):
    r, m = doc
    assert m["synthetic"] is True
    assert m["generator_version"] == GENERATOR_VERSION
    assert m["doc_id"] == f"{LAYOUT}-0001"
    assert m["layout"] == LAYOUT
    assert m["split"] == "train"
    assert m["seeds"] == {"master": MASTER, "document": SEED}
    assert m["months"] == [d.strftime("%Y-%m") for d in r.ledger.months]
    assert m["company"]["uen"] == r.ledger.company.uen


def test_genuine_hard_cases_lists_instalments():
    if load_renderer(LAYOUT) is None:
        pytest.skip("generator_v1 renderer not available")
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger

    ledger = simulate_ledger(make_company(SEED, "C0001"), DEFAULT_START, 6, SEED,
                             SimulatorKnobs(instalment_rate=1.0), None)
    expected = sorted({r.hard_case for r in ledger.rows if r.hard_case})
    assert expected, "test ledger should contain instalments"
    rendered = load_renderer(LAYOUT).render(ledger, DEFAULT_STYLE, SEED)
    m = build_manifest(_plan(ledger), ledger, rendered, TamperLabel(), master_seed=MASTER)
    assert sorted(set(m["genuine_hard_cases"])) == expected
    assert any(h.startswith("instalment") for h in m["genuine_hard_cases"])


def test_manifest_json_is_sorted_and_reproducible(doc, tmp_path):
    r, m = doc
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_manifest(p1, m)
    write_manifest(p2, build_manifest(_plan(r.ledger), r.ledger, r.result, TamperLabel(),
                                      master_seed=MASTER))
    assert p1.read_bytes() == p2.read_bytes()
    text = p1.read_text()
    loaded = json.loads(text)
    assert loaded == json.loads(json.dumps(m))
    assert list(loaded) == sorted(loaded)                      # sorted keys


def test_labels_csv_header_and_genuine_line(doc, tmp_path):
    r, m = doc
    path = tmp_path / "labels.csv"
    write_labels_csv(path, [m])
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == LABELS_COLUMNS
        lines = list(reader)
    assert len(lines) == 1
    L = lines[0]
    assert L["doc_id"] == m["doc_id"]
    assert L["file"] == f"pdfs/{m['doc_id']}.pdf"
    assert L["split"] == "train" and L["layout"] == LAYOUT
    assert L["doc_seed"] == str(SEED)
    assert L["n_months"] == "6"
    assert L["first_month"] == m["months"][0] and L["last_month"] == m["months"][-1]
    assert L["n_rows"] == str(len(m["rows"]))
    assert L["n_pages"] == str(r.result.n_pages)
    assert L["is_tampered"] == "0"
    assert L["tamper_types"] == "" and L["tamper_method"] == ""
    assert L["arithmetic_consistent"] == "1"
    assert L["genuine_resaved"] == "0"
    assert L["n_affected_rows"] == "0"
    assert L["generator_version"] == GENERATOR_VERSION


def test_rows_csv_header_and_lines(doc, tmp_path):
    _, m = doc
    path = tmp_path / "rows.csv"
    write_rows_csv(path, [m])
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == ROWS_COLUMNS
        lines = list(reader)
    assert len(lines) == len(m["rows"])
    for L, x in zip(lines, m["rows"]):
        assert L["doc_id"] == m["doc_id"]
        assert L["row_index"] == str(x["row_index"])
        assert L["row_id"] == x["row_id"]
        assert L["page"] == str(x["page"])
        assert L["month"] == x["month"] and L["date"] == x["date"]
        assert float(L["amount"]) == pytest.approx(x["amount"])
        assert float(L["balance"]) == pytest.approx(x["balance"])
        assert L["is_fraud_row"] == "0"
        assert L["fraud_type"] == ""
