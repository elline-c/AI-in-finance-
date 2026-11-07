"""_apply_metadata: the producer param is optional."""
import io
from datetime import datetime, timedelta

from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas

from fraud_detect.corpus.schema import FileOp
from fraud_detect.corpus.tamper.pdf_ops import _apply_metadata

CREATED = datetime(2026, 3, 5, 10, 30, 0)


def _make_pdf() -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, invariant=1, pageCompression=0)
    c.setProducer("Fictional Base Producer 1")
    c.drawString(72, 700, "hello")
    c.showPage()
    c.save()
    # First pass stamps a known creation date via the op itself.
    op = FileOp(op="set_metadata", params=(("created", CREATED.isoformat()),
                                           ("modified_delta_s", "0")))
    return _apply_metadata(buf.getvalue(), op)


def _meta(pdf: bytes):
    return PdfReader(io.BytesIO(pdf)).metadata


def test_missing_producer_keeps_producer_and_sets_moddate():
    base = _make_pdf()
    assert _meta(base).producer == "Fictional Base Producer 1"
    op = FileOp(op="resave", params=(("modified_delta_s", "3600"),))
    out = _apply_metadata(base, op)
    meta = _meta(out)
    assert meta.producer == "Fictional Base Producer 1"
    assert meta.creation_date.replace(tzinfo=None) == CREATED
    assert meta.modification_date.replace(tzinfo=None) == CREATED + timedelta(seconds=3600)


def test_producer_param_still_replaces_producer():
    base = _make_pdf()
    op = FileOp(op="set_metadata", params=(("modified_delta_s", "60"),
                                           ("producer", "Fictional Editor 1")))
    meta = _meta(_apply_metadata(base, op))
    assert meta.producer == "Fictional Editor 1"
    assert meta.modification_date.replace(tzinfo=None) == CREATED + timedelta(seconds=60)
