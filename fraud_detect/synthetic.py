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
# Public generators
# --------------------------------------------------------------------------- #
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
