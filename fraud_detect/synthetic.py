"""Synthetic bank-statement generator (Phase 2 + Phase 3 planted issues).

Produces PDF bank statements together with a ground-truth manifest describing
exactly which issues (if any) were planted. The manifest lets the test suite
assert that the detection agents catch what was deliberately introduced.

Planted-issue types (Phase 3 test areas):
    - ALTERED_BALANCE        : a running balance edited so arithmetic breaks
    - ALTERED_NUMBER         : a transaction amount overwritten (font mismatch)
    - FORMAT_INCONSISTENCY   : mixed date / currency / number formatting
    - METADATA_MISMATCH      : editing-software producer tag (Photoshop) + date skew
    - PERIOD_GAP             : an unexplained gap in the statement period
    - SUSPICIOUS_TXN         : sudden massive deposit off the cashflow baseline
    - STRUCTURING            : repeated deposits just under the reporting threshold
    - UEN_MISMATCH           : account holder name != ACRA record for the UEN
"""
from __future__ import annotations

import calendar
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from .reference_data import ACRA_REGISTRY, KNOWN_VENDORS, STRUCTURING_THRESHOLD


@dataclass
class PlantedIssue:
    kind: str
    detail: str


@dataclass
class GeneratedStatement:
    path: str
    manifest: list[PlantedIssue] = field(default_factory=list)
    is_clean: bool = True


@dataclass
class MonthGroundTruth:
    """Ground truth for one monthly section inside a combined multi-month PDF.

    ``year`` / ``month`` identify the calendar month. ``period_start`` /
    ``period_end`` are the printed "Statement Period:" dates for that section
    (first-of-month through last-of-month). ``page_start`` / ``page_end`` are
    the 1-based page numbers the section occupies in the final document, so a
    test can assert the ingestion segmentation matches exactly.
    """
    year: int
    month: int
    period_start: date
    period_end: date
    page_start: int
    page_end: int


@dataclass
class CombinedStatement:
    """A single combined PDF stapling several monthly sections together.

    ``segments`` lists the ground-truth per-month page ranges in the order the
    sections physically appear in the document (which, for a jumbled layout, is
    NOT chronological). ``month_count`` is the expected number of ingestion
    segments.
    """
    path: str
    segments: list[MonthGroundTruth] = field(default_factory=list)
    jumbled: bool = False

    @property
    def month_count(self) -> int:
        return len(self.segments)


@dataclass
class MonthlyFile:
    """One standalone single-month PDF plus the month its content declares.

    ``year`` / ``month`` are the calendar month the printed Statement Period
    dates identify. Note this identity lives in the file *content*, not the
    filename, so the detection must not depend on filename or submission order.
    """
    path: str
    year: int
    month: int
    period_start: date
    period_end: date


@dataclass
class MonthlySet:
    """A SET of standalone monthly PDFs submitted together as one application.

    ``files`` is the (possibly shuffled) list of generated files; ``kind``
    records what was deliberately engineered ("complete", "missing",
    "duplicate", "jumbled").
    """
    files: list[MonthlyFile] = field(default_factory=list)
    kind: str = "complete"

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]


# --------------------------------------------------------------------------- #
# Internal transaction model for generation
# --------------------------------------------------------------------------- #
@dataclass
class _Txn:
    dt: date
    desc: str
    amount: float          # signed
    balance: float
    date_fmt: str = "%d/%m/%Y"
    amount_font: str = "Helvetica"
    balance_font: str = "Helvetica"


_CREDIT_DESCS = [
    "SP Services", "DBS Merchant Payout", "Sheng Siong Distribution",
    "Ninja Van SG", "Customer Payment INV", "Sales Settlement",
]
_DEBIT_DESCS = [
    "M1 Limited", "CPF Board", "IRAS GST", "Comfort Transport",
    "Office Rental", "Supplier Payment", "Payroll Run",
]


def _last_day_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _build_transactions(rng: random.Random, start: date, opening: float,
                        n: int = 24) -> list[_Txn]:
    txns: list[_Txn] = []
    bal = opening
    dt = start
    for _ in range(n):
        dt = dt + timedelta(days=rng.randint(1, 6))
        if rng.random() < 0.5:
            amt = round(rng.uniform(500, 9000), 2)
            desc = rng.choice(_CREDIT_DESCS)
        else:
            amt = -round(rng.uniform(300, 7000), 2)
            desc = rng.choice(_DEBIT_DESCS)
        bal = round(bal + amt, 2)
        txns.append(_Txn(dt=dt, desc=desc, amount=amt, balance=bal))
    return txns


def _build_month_txns(rng: random.Random, year: int, month: int, opening: float,
                      n: int = 6) -> list[_Txn]:
    """A small set of transactions whose dates all fall inside one calendar
    month. Kept short so each monthly section fits comfortably on one page."""
    last_day = _last_day_of_month(year, month).day
    txns: list[_Txn] = []
    bal = opening
    for i in range(n):
        # spread days across the month deterministically
        day = min(last_day, 2 + (i * max(1, last_day // (n + 1))))
        dt = date(year, month, day)
        if rng.random() < 0.5:
            amt = round(rng.uniform(500, 9000), 2)
            desc = rng.choice(_CREDIT_DESCS)
        else:
            amt = -round(rng.uniform(300, 7000), 2)
            desc = rng.choice(_DEBIT_DESCS)
        bal = round(bal + amt, 2)
        txns.append(_Txn(dt=dt, desc=desc, amount=amt, balance=bal))
    return txns


# --------------------------------------------------------------------------- #
# PDF rendering
# --------------------------------------------------------------------------- #
def _render_pdf(path: Path, *, holder: str, uen: str, account_no: str,
                period_start: date, period_end: date, currency: str,
                opening: float, txns: list[_Txn], producer: str | None,
                creation_override: str | None, mixed_currency: bool) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    if producer:
        c.setProducer(producer)
    c.setAuthor(holder)
    c.setTitle(f"Bank Statement {account_no}")

    def header(page_no: int) -> float:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(20 * mm, height - 25 * mm, "OCBC-STYLE BANK  |  Corporate Statement")
        c.setFont("Helvetica", 9)
        y = height - 33 * mm
        for line in [
            f"Account Holder: {holder}",
            f"UEN: {uen}",
            f"Account No: {account_no}",
            f"Statement Period: {period_start:%d/%m/%Y} - {period_end:%d/%m/%Y}",
            f"Currency: {currency}",
            f"Opening Balance: {opening:,.2f}",
            f"Page {page_no}",
        ]:
            c.drawString(20 * mm, y, line)
            y -= 5 * mm
        # column headers
        y -= 3 * mm
        c.setFont("Helvetica-Bold", 9)
        c.drawString(20 * mm, y, "Date")
        c.drawString(55 * mm, y, "Description")
        c.drawString(120 * mm, y, "Amount")
        c.drawString(160 * mm, y, "Balance")
        return y - 6 * mm

    page_no = 1
    y = header(page_no)
    c.setFont("Helvetica", 9)
    for t in txns:
        if y < 25 * mm:
            c.showPage()
            page_no += 1
            y = header(page_no)
            c.setFont("Helvetica", 9)

        cur = currency
        if mixed_currency and t.amount > 5000:
            cur = "S$"  # inconsistent currency symbol on some rows

        c.setFont("Helvetica", 9)
        c.drawString(20 * mm, y, t.dt.strftime(t.date_fmt))
        c.drawString(55 * mm, y, t.desc)

        # Amount (possibly a different font to simulate overwrite)
        c.setFont(t.amount_font, 9)
        sign = "" if t.amount < 0 else "+"
        c.drawRightString(150 * mm, y, f"{sign}{cur}{t.amount:,.2f}")

        # Balance (possibly a different font to simulate overwrite)
        c.setFont(t.balance_font, 9)
        c.drawRightString(195 * mm, y, f"{cur}{t.balance:,.2f}")

        y -= 5.2 * mm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(20 * mm, y - 4 * mm,
                 f"Closing Balance: {currency}{txns[-1].balance:,.2f}")
    c.showPage()
    c.save()

    if creation_override:
        # reportlab overwrites CreationDate, so patch it in afterwards with pypdf
        # to simulate a document whose creation date predates the statement period.
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(path))
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)
        meta = {k: v for k, v in (reader.metadata or {}).items()}
        meta["/CreationDate"] = creation_override
        writer.add_metadata(meta)
        with open(path, "wb") as fh:
            writer.write(fh)


# --------------------------------------------------------------------------- #
# Multi-month / multi-file rendering (segmentation + coverage test fixtures)
# --------------------------------------------------------------------------- #
def _render_month_section(c: "canvas.Canvas", *, holder: str, uen: str,
                          account_no: str, currency: str, opening: float,
                          period_start: date, period_end: date,
                          txns: list[_Txn], page_no: int) -> None:
    """Render ONE monthly section onto the current canvas page.

    Mirrors ``_render_pdf``'s header()/row layout so the ingestion regexes parse
    it identically. The caller is responsible for ``c.showPage()`` so that each
    section starts on a fresh page (giving the combined document one period
    header per page). Kept short enough to fit on a single page.
    """
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, height - 25 * mm, "OCBC-STYLE BANK  |  Corporate Statement")
    c.setFont("Helvetica", 9)
    y = height - 33 * mm
    for line in [
        f"Account Holder: {holder}",
        f"UEN: {uen}",
        f"Account No: {account_no}",
        f"Statement Period: {period_start:%d/%m/%Y} - {period_end:%d/%m/%Y}",
        f"Currency: {currency}",
        f"Opening Balance: {opening:,.2f}",
        f"Page {page_no}",
    ]:
        c.drawString(20 * mm, y, line)
        y -= 5 * mm
    y -= 3 * mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "Date")
    c.drawString(55 * mm, y, "Description")
    c.drawString(120 * mm, y, "Amount")
    c.drawString(160 * mm, y, "Balance")
    y -= 6 * mm

    c.setFont("Helvetica", 9)
    for t in txns:
        c.setFont("Helvetica", 9)
        c.drawString(20 * mm, y, t.dt.strftime(t.date_fmt))
        c.drawString(55 * mm, y, t.desc)
        sign = "" if t.amount < 0 else "+"
        c.drawRightString(150 * mm, y, f"{sign}{currency}{t.amount:,.2f}")
        c.drawRightString(195 * mm, y, f"{currency}{t.balance:,.2f}")
        y -= 5.2 * mm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(20 * mm, y - 4 * mm,
                 f"Closing Balance: {currency}{txns[-1].balance:,.2f}")


# --------------------------------------------------------------------------- #
# Public generators
# --------------------------------------------------------------------------- #
def generate_combined_multi_month(
    out_dir: str, *, year: int = 2024, months: list[int] | None = None,
    jumbled: bool = False, seed: int = 101, uen: str = "201812345A",
    account_no: str = "0123456789", currency: str = "SGD",
    opening: float = 40000.0,
) -> CombinedStatement:
    """Generate ONE combined PDF with several monthly sections, each starting on
    a fresh page with its own "Statement Period:" header.

    With ``jumbled=False`` the sections are laid out in chronological order
    (Jan..Jun). With ``jumbled=True`` the physical page order is non-chronological
    (a later page covers an earlier month) to exercise FL_MONTH_OUT_OF_ORDER;
    each section's printed period dates still identify its true month.

    Returns ground truth: the expected segment count and the per-section
    ``(page_start, page_end)`` and period dates, in physical document order.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    holder = ACRA_REGISTRY[uen]
    months = months or [1, 2, 3, 4, 5, 6]

    # Physical layout order of the calendar months in the document.
    layout = list(months)
    if jumbled and len(layout) >= 2:
        # Deterministic non-chronological order: swap so an earlier month lands
        # on a later page (classic inserted/reordered-pages tampering shape).
        layout = sorted(months)
        layout[0], layout[-1] = layout[-1], layout[0]

    suffix = "jumbled" if jumbled else "chrono"
    path = out / f"combined_{year}_{suffix}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setAuthor(holder)
    c.setTitle(f"Combined Bank Statement {account_no}")

    segments: list[MonthGroundTruth] = []
    for page_no, m in enumerate(layout, start=1):
        p_start = date(year, m, 1)
        p_end = _last_day_of_month(year, m)
        txns = _build_month_txns(rng, year, m, opening, n=6)
        _render_month_section(
            c, holder=holder, uen=uen, account_no=account_no, currency=currency,
            opening=opening, period_start=p_start, period_end=p_end,
            txns=txns, page_no=page_no,
        )
        c.showPage()  # each section occupies exactly one page -> fresh header
        segments.append(MonthGroundTruth(
            year=year, month=m, period_start=p_start, period_end=p_end,
            page_start=page_no, page_end=page_no,
        ))
    c.save()

    return CombinedStatement(path=str(path), segments=segments, jumbled=jumbled)


def generate_monthly_set(
    out_dir: str, *, year: int = 2024, months: list[int] | None = None,
    kind: str = "complete", shuffle: bool = False, seed: int = 202,
    uen: str = "201812345A", account_no: str = "0123456789",
    currency: str = "SGD", opening: float = 40000.0,
) -> MonthlySet:
    """Generate a SET of standalone single-month PDFs, one file per month entry.

    ``months`` is the list of calendar months to produce for ``year``. The list
    is used verbatim, so callers craft the scenario by choosing it:
        * complete  : [1, 2, 3]         -> contiguous months
        * missing   : [1, 2, 4]         -> March omitted (interior gap)
        * duplicate : [1, 2, 2, 3]      -> February produced twice
    ``kind`` is recorded on the result for readability; it does not change what
    is produced (the ``months`` list is authoritative).

    Each file's month identity is carried ENTIRELY by its printed "Statement
    Period:" dates, not by its filename. When ``shuffle=True`` the returned file
    list is deterministically shuffled so callers can prove that submission
    order does not change month identification.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    holder = ACRA_REGISTRY[uen]
    months = months if months is not None else [1, 2, 3]

    files: list[MonthlyFile] = []
    for occurrence, m in enumerate(months):
        p_start = date(year, m, 1)
        p_end = _last_day_of_month(year, m)
        txns = _build_month_txns(rng, year, m, opening, n=6)
        # Filename intentionally uses a neutral running index (NOT the month) so
        # the detection cannot shortcut month identity from the filename.
        path = out / f"stmt_{year}_{kind}_{occurrence:02d}.pdf"
        _render_pdf(
            path, holder=holder, uen=uen, account_no=account_no,
            period_start=p_start, period_end=p_end, currency=currency,
            opening=opening, txns=txns, producer="ReportLab PDF Library",
            creation_override=None, mixed_currency=False,
        )
        files.append(MonthlyFile(
            path=str(path), year=year, month=m,
            period_start=p_start, period_end=p_end,
        ))

    if shuffle and len(files) > 1:
        order = random.Random(seed + 1)
        order.shuffle(files)

    return MonthlySet(files=files, kind=kind)


def generate_clean(out_dir: str, seed: int = 1, uen: str = "201812345A") -> GeneratedStatement:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    holder = ACRA_REGISTRY[uen]
    start = date(2025, 3, 1)
    opening = 42000.0
    txns = _build_transactions(rng, start, opening, n=22)
    end = txns[-1].dt

    path = out / f"clean_{uen}.pdf"
    _render_pdf(path, holder=holder, uen=uen, account_no="0123456789",
                period_start=start, period_end=end, currency="SGD",
                opening=opening, txns=txns, producer="ReportLab PDF Library",
                creation_override=None, mixed_currency=False)
    return GeneratedStatement(path=str(path), manifest=[], is_clean=True)


def generate_tampered(out_dir: str, seed: int = 7, uen: str = "200923456B",
                      issues: list[str] | None = None) -> GeneratedStatement:
    """Generate a statement with a chosen set of planted issues."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    issues = issues or [
        "ALTERED_BALANCE", "ALTERED_NUMBER", "FORMAT_INCONSISTENCY",
        "METADATA_MISMATCH", "PERIOD_GAP", "SUSPICIOUS_TXN",
        "STRUCTURING", "UEN_MISMATCH",
    ]
    manifest: list[PlantedIssue] = []

    holder = ACRA_REGISTRY[uen]
    start = date(2025, 1, 1)
    opening = 55000.0
    txns = _build_transactions(rng, start, opening, n=24)

    producer = "ReportLab PDF Library"
    creation_override = None
    mixed_currency = False

    if "UEN_MISMATCH" in issues:
        holder = "Meridian Trading Pte Ltd"  # wrong entity for this UEN
        manifest.append(PlantedIssue("UEN_MISMATCH",
                                     f"Holder '{holder}' != ACRA record for {uen}"))

    if "FORMAT_INCONSISTENCY" in issues:
        # a few rows use a different date format + mixed currency symbol
        for t in txns[3:6]:
            t.date_fmt = "%Y-%m-%d"
        mixed_currency = True
        manifest.append(PlantedIssue("FORMAT_INCONSISTENCY",
                                     "Mixed date formats (dd/mm/yyyy vs yyyy-mm-dd) + mixed currency symbol"))

    altered_number_row: _Txn | None = None
    if "ALTERED_NUMBER" in issues:
        t = txns[8]
        t.amount = round(t.amount + 5000, 2)   # inflate an amount
        t.amount_font = "Times-Bold"           # simulate an overwrite (font mismatch)
        altered_number_row = t
        manifest.append(PlantedIssue("ALTERED_NUMBER",
                                     f"Amount on txn #{t_index(txns, t)} inflated & re-typed in different font"))

    if "ALTERED_BALANCE" in issues:
        t = txns[14]
        t.balance = round(t.balance + 12000, 2)  # break running balance
        t.balance_font = "Courier-Bold"
        manifest.append(PlantedIssue("ALTERED_BALANCE",
                                     f"Running balance on txn #{t_index(txns, t)} overwritten (+12,000)"))

    if "SUSPICIOUS_TXN" in issues:
        t = txns[18]
        t.amount = 480000.0                      # sudden massive deposit
        t.desc = "INWARD TT UNKNOWN LLC"
        manifest.append(PlantedIssue("SUSPICIOUS_TXN",
                                     "Sudden 480,000 inward transfer from unverified counterparty"))

    if "STRUCTURING" in issues:
        # inject several deposits just below the threshold on consecutive days
        base_dt = txns[10].dt
        for i in range(4):
            amt = STRUCTURING_THRESHOLD - 200 - i  # 9,800-ish
            txns.insert(11 + i, _Txn(dt=base_dt + timedelta(days=i),
                                     desc="CASH DEPOSIT", amount=amt, balance=0.0))
        manifest.append(PlantedIssue("STRUCTURING",
                                     "4 cash deposits just below reporting threshold on consecutive days"))

    if "PERIOD_GAP" in issues:
        # push the tail transactions ~40 days later to create an unexplained gap
        gap_start = len(txns) - 3
        for t in txns[gap_start:]:
            t.dt = t.dt + timedelta(days=40)
        manifest.append(PlantedIssue("PERIOD_GAP",
                                     "~40-day unexplained gap before the final transactions"))

    # Recompute balances so ONLY the deliberately altered rows are wrong.
    _recompute_balances(txns, opening,
                         skip_balance_override="ALTERED_BALANCE" in issues)

    if altered_number_row is not None:
        # The fraudster inflated the AMOUNT but the printed running balance still
        # reflects the original amount -> row-level reconciliation must fail.
        altered_number_row.balance = round(altered_number_row.balance - 5000, 2)

    if "METADATA_MISMATCH" in issues:
        producer = "Adobe Photoshop 25.0 (Windows)"     # editing-software trace
        creation_override = "D:20240102030405"          # creation date before statement period
        manifest.append(PlantedIssue("METADATA_MISMATCH",
                                     "Producer=Adobe Photoshop; creation date predates statement period"))

    end = max(t.dt for t in txns)
    path = out / f"tampered_{uen}.pdf"
    _render_pdf(path, holder=holder, uen=uen, account_no="0987654321",
                period_start=start, period_end=end, currency="SGD",
                opening=opening, txns=txns, producer=producer,
                creation_override=creation_override, mixed_currency=mixed_currency)
    return GeneratedStatement(path=str(path), manifest=manifest, is_clean=False)


def t_index(txns: list[_Txn], t: _Txn) -> int:
    return txns.index(t)


def _recompute_balances(txns: list[_Txn], opening: float,
                        skip_balance_override: bool) -> None:
    """Recompute running balances. If skip_balance_override, preserve any row
    whose balance_font indicates a deliberate overwrite (so the math breaks)."""
    bal = opening
    for t in txns:
        bal = round(bal + t.amount, 2)
        overwritten = t.balance_font != "Helvetica"
        if skip_balance_override and overwritten:
            continue  # keep the tampered balance
        t.balance = bal


def generate_medium(out_dir: str, seed: int = 11, uen: str = "201534567C") -> GeneratedStatement:
    """A borderline statement: only mild formatting inconsistencies, no hard
    tampering or math breaks -> should land in the MEDIUM band and route to a human."""
    return generate_tampered(out_dir, seed=seed, uen=uen,
                             issues=["FORMAT_INCONSISTENCY"])


def generate_unreadable(out_dir: str, seed: int = 5) -> GeneratedStatement:
    """An (almost) empty / unreadable PDF standing in for a poor-quality scan
    where OCR fails -> should trigger data-quality findings and human review."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "unreadable_scan.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setProducer("ScanApp")
    # draw nothing parseable (simulates a blank/low-quality scan)
    c.showPage()
    c.save()
    return GeneratedStatement(path=str(path),
                              manifest=[PlantedIssue("DATA_QUALITY", "Unreadable scan / OCR failure")],
                              is_clean=False)


if __name__ == "__main__":  # pragma: no cover
    import json
    d = "samples"
    clean = generate_clean(d)
    tampered = generate_tampered(d)
    print(json.dumps({
        "clean": {"path": clean.path, "issues": [i.kind for i in clean.manifest]},
        "tampered": {"path": tampered.path, "issues": [i.kind for i in tampered.manifest]},
    }, indent=2))
