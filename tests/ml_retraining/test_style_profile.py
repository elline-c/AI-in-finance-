"""Task 6.1: failing tests for ``fraud_detect.ml.retraining.style_profile``.

Style values are read back from the PDF itself (design finding F6); no manifest style value
is used. Only Mini_Corpus PDFs from the session fixture are read.
"""
from __future__ import annotations

import io
import os

import pytest

from fraud_detect.corpus.finish import BANK_NAMES, FOOTER_TEXT, PRODUCERS
from fraud_detect.corpus.tamper.pdf_ops import EDITOR_PRODUCERS, EXPORT_PRODUCERS
from fraud_detect.ml.retraining.style_profile import style_profile

FIELDS = ("producer", "producer_family", "creator", "fonts", "bank_name", "street",
          "second_address_line", "row_step_bucket", "creation_hour", "page_size", "footer_text")


def _get(profile, name):
    return profile[name] if isinstance(profile, dict) else getattr(profile, name)


def _pdfs(mini_corpus):
    folder = mini_corpus.path() / "pdfs"
    return sorted(folder.glob("*.pdf"))


def test_profile_has_every_field(mini_corpus):
    pdf = _pdfs(mini_corpus)[0].read_bytes()
    profile = style_profile(pdf)
    for name in FIELDS:
        _get(profile, name)  # raises when a field is missing


def test_profile_values_come_from_known_catalogues(mini_corpus):
    for path in _pdfs(mini_corpus):
        p = style_profile(path.read_bytes())
        family = _get(p, "producer_family")
        producer = _get(p, "producer")
        assert family in {"base", "export", "editor"}
        expected = ("base" if producer in PRODUCERS else
                    "export" if producer in EXPORT_PRODUCERS else
                    "editor" if producer in EDITOR_PRODUCERS else None)
        assert expected == family, (path.name, producer)
        assert _get(p, "bank_name") in BANK_NAMES
        # Generator-drawn street is not printed in the PDF, so it is not recoverable.
        assert _get(p, "street") is None or isinstance(_get(p, "street"), str)
        assert _get(p, "footer_text") == FOOTER_TEXT
        assert isinstance(_get(p, "creator"), str) and _get(p, "creator")


def test_fonts_sorted_and_non_empty(mini_corpus):
    for path in _pdfs(mini_corpus)[:6]:
        fonts = _get(style_profile(path.read_bytes()), "fonts")
        assert len(fonts) >= 1
        assert list(fonts) == sorted(fonts)


def test_row_step_bucket_is_a_tenth_of_a_point(mini_corpus):
    for path in _pdfs(mini_corpus)[:6]:
        step = _get(style_profile(path.read_bytes()), "row_step_bucket")
        assert 12.0 <= step <= 12.8 + 1e-9
        assert abs(step * 10 - round(step * 10)) < 1e-6


def test_creation_hour_and_page_size(mini_corpus):
    for path in _pdfs(mini_corpus)[:6]:
        p = style_profile(path.read_bytes())
        assert 0 <= _get(p, "creation_hour") <= 23
        width, height = _get(p, "page_size")
        assert (round(width), round(height)) == (595, 842)  # A4


def test_second_address_line_is_a_string_or_empty(mini_corpus):
    values = {_get(style_profile(p.read_bytes()), "second_address_line") for p in _pdfs(mini_corpus)}
    assert all(isinstance(v, str) or v is None for v in values)


def test_works_from_bytesio_and_bytes_alike(mini_corpus):
    raw = _pdfs(mini_corpus)[0].read_bytes()
    a = style_profile(raw)
    b = style_profile(io.BytesIO(raw))
    assert {n: _get(a, n) for n in FIELDS} == {n: _get(b, n) for n in FIELDS}


def test_profile_is_deterministic(mini_corpus):
    raw = _pdfs(mini_corpus)[0].read_bytes()
    assert style_profile(raw) == style_profile(raw)


def test_creates_no_file(mini_corpus, tmp_path, monkeypatch):
    raw = _pdfs(mini_corpus)[0].read_bytes()
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    corpus_before = sorted(p.name for p in mini_corpus.path().rglob("*"))
    style_profile(raw)
    assert set(os.listdir(tmp_path)) == before
    assert sorted(p.name for p in mini_corpus.path().rglob("*")) == corpus_before


def test_profile_for_resaved_genuine_uses_export_family(mini_corpus):
    import csv
    with open(mini_corpus.path() / "labels.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["genuine_resaved"] in ("1", "True", "true")]
    if not rows:
        pytest.skip("Mini_Corpus has no genuine_resaved document")
    raw = (mini_corpus.path() / "pdfs" / f"{rows[0]['doc_id']}.pdf").read_bytes()
    assert _get(style_profile(raw), "producer_family") == "export"


def test_import_does_not_pull_pdf_libraries_at_module_level():
    import subprocess
    import sys
    code = ("import sys, fraud_detect.ml.retraining.style_profile;"
            "sys.exit(1 if 'pdfplumber' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
