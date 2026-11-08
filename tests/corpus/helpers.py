"""Shared helpers for the corpus generator tests (import: ``from tests.corpus.helpers import ...``).

Everything that touches not-yet-written modules (company, ledger, layouts registry)
imports lazily inside the function, so this module imports cleanly at any time.

Conventions
-----------
* ``CellBox.page`` and ``PrintedRow.page`` are 1-based (as in pdfplumber's ``page_number``).
* ``compare_rows`` compares *sequences* (date, signed amount, balance, order). It never
  compares ``Transaction.index`` to a corpus index.
* A "renderer" is any object with ``render(ledger, style, seed) -> RenderResult`` and
  ``max_rows_per_month``; tests can pass their own instead of a layout name.
"""
from __future__ import annotations

import io
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fraud_detect.corpus.schema import LAYOUTS, SimulatorKnobs, StyleKnobs

# A fixed style so tests do not depend on the (later) style generator.
DEFAULT_STYLE = StyleKnobs(
    bank_name="NORTHERN TEST BANK",
    address_lines=("1 Specimen Road", "Singapore 000000"),
    producer="ReportLab PDF Library",
    creation_time="09:30:00",
    row_step=12.4,
    second_address_line=False,
)
DEFAULT_START = date(2026, 1, 1)

# reportlab built-in (standard 14) font families allowed by Requirement 18.5.
_ALLOWED_FONT = re.compile(
    r"^(Helvetica(-(Bold|Oblique|BoldOblique))?|Times-(Roman|Bold|Italic|BoldItalic)"
    r"|Courier(-(Bold|Oblique|BoldOblique))?|Symbol|ZapfDingbats)$"
)


# --------------------------------------------------------------------------
# renderer lookup
# --------------------------------------------------------------------------
def load_renderer(layout: str) -> Optional[Any]:
    """Return the renderer for ``layout`` or ``None`` if it is not implemented yet.

    Only a missing registry entry / unimplemented layout counts as "not yet".
    A missing *base* module is not hidden here: the test module imports those itself.
    """
    try:
        from fraud_detect.corpus.layouts import get_renderer
    except ImportError:
        return None
    try:
        return get_renderer(layout)
    except (KeyError, NotImplementedError, ImportError, AttributeError):
        return None


def implemented_layouts() -> list[str]:
    return [name for name in LAYOUTS if load_renderer(name) is not None]


# --------------------------------------------------------------------------
# ledger + render + ingest
# --------------------------------------------------------------------------
def make_test_ledger(n_months: int, seed: int, max_rows_per_month: Optional[int] = None,
                     start: date = DEFAULT_START, company_id: str = "C0001"):
    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger

    company = make_company(seed, company_id)
    return simulate_ledger(company, start, n_months, seed, SimulatorKnobs(), max_rows_per_month)


@dataclass
class Rendered:
    ledger: Any
    result: Any                      # RenderResult
    pdf_path: Optional[Path] = None
    ingestion: Any = None            # IngestionResult (None when ingest=False)
    _tmp: Any = field(default=None, repr=False)


def render_ledger(renderer: Any, ledger: Any, seed: int, style: StyleKnobs = DEFAULT_STYLE):
    return renderer.render(ledger, style, seed)


def render_and_ingest(layout: Any, n_months: int, seed: int, *, ledger: Any = None,
                      style: StyleKnobs = DEFAULT_STYLE, ingest: bool = True,
                      out_dir: Optional[Path] = None) -> Rendered:
    """Simulate a ledger, render it, write a temporary PDF and run Node 1 on it.

    ``layout`` is a layout name (looked up via the registry) or a renderer object.
    Pass ``ledger`` to render a prepared ledger instead of simulating one.
    """
    renderer = load_renderer(layout) if isinstance(layout, str) else layout
    if renderer is None:
        raise LookupError(f"layout {layout!r} is not implemented yet")
    if ledger is None:
        ledger = make_test_ledger(n_months, seed, getattr(renderer, "max_rows_per_month", None))
    result = render_ledger(renderer, ledger, seed, style)

    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="corpus_test_")
        out_dir = Path(tmp.name)
    pdf_path = Path(out_dir) / f"{getattr(renderer, 'name', 'doc')}-{seed}.pdf"
    pdf_path.write_bytes(result.pdf)

    ingestion = None
    if ingest:
        from fraud_detect.nodes.ingestion import ingest as node1_ingest
        ingestion = node1_ingest(str(pdf_path))
    return Rendered(ledger=ledger, result=result, pdf_path=pdf_path, ingestion=ingestion, _tmp=tmp)


# --------------------------------------------------------------------------
# comparing ledger rows with Node 1 output
# --------------------------------------------------------------------------
def replayed_balances(ledger: Any) -> list[float]:
    bal, out = ledger.opening_balance, []
    for r in ledger.rows:
        bal = round(bal + r.amount, 2)
        out.append(bal)
    return out


def compare_rows(ledger_rows: Sequence[Any], node1_transactions: Sequence[Any],
                 balances: Optional[Sequence[float]] = None, tol: float = 0.005) -> list[str]:
    """Compare corpus rows with Node 1 transactions; return a list of problems (empty = equal).

    Checks, position by position: date, signed amount, balance (when ``balances`` is
    given) and the row count (hence order). ``Transaction.index`` is ignored on purpose.
    """
    problems: list[str] = []
    if len(ledger_rows) != len(node1_transactions):
        problems.append(f"row count: ledger {len(ledger_rows)} != node1 {len(node1_transactions)}")
    for i, (row, txn) in enumerate(zip(ledger_rows, node1_transactions)):
        if txn.date != row.date:
            problems.append(f"#{i} date: ledger {row.date} != node1 {txn.date}")
        if txn.amount is None or abs(txn.amount - row.amount) > tol:
            problems.append(f"#{i} amount: ledger {row.amount} != node1 {txn.amount}")
        if balances is not None:
            if txn.balance is None or abs(txn.balance - balances[i]) > tol:
                problems.append(f"#{i} balance: ledger {balances[i]} != node1 {txn.balance}")
    return problems


# --------------------------------------------------------------------------
# reading the PDF
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PdfWord:
    text: str
    x0: float
    x1: float
    top: float
    page: int           # 1-based


def words_of(pdf_bytes: bytes) -> list[PdfWord]:
    import pdfplumber

    out: list[PdfWord] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=False):
                out.append(PdfWord(w["text"], round(w["x0"], 2), round(w["x1"], 2),
                                   round(w["top"], 2), pno))
    return out


def page_texts_of(pdf_bytes: bytes) -> list[str]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def fonts_of(pdf_bytes: bytes) -> set[str]:
    """Font names used by characters (subset prefix ``ABCDEF+`` removed)."""
    import pdfplumber

    names: set[str] = set()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for ch in page.chars:
                names.add(re.sub(r"^[A-Z]{6}\+", "", ch["fontname"]))
    return names


def is_builtin_font(name: str) -> bool:
    return bool(_ALLOWED_FONT.match(name))


def metadata_of(pdf_bytes: bytes) -> dict[str, str]:
    from pypdf import PdfReader

    meta = PdfReader(io.BytesIO(pdf_bytes)).metadata or {}
    return {str(k): str(v) for k, v in meta.items()}


def text_at(words: Iterable[PdfWord], page: int, top: float, x0: float, x1: float,
            top_tol: float = 3.0, x_tol: float = 2.0) -> str:
    """Join the words on ``page`` lying in the given box (left to right)."""
    hit = [w for w in words
           if w.page == page and abs(w.top - top) <= top_tol
           and w.x0 >= x0 - x_tol and w.x1 <= x1 + x_tol]
    return " ".join(w.text for w in sorted(hit, key=lambda w: w.x0))
