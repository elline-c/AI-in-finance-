"""Shared constants and plain frozen data classes of the corpus generator.

No logic lives here beyond trivial helpers. Later tasks import these names.

Design choices where the design document was silent or loose (kept simple):

* Money is a plain ``float`` rounded to 2 decimals by the code that creates it
  (as in the design's ``LedgerRow.amount``); signed: credit +, debit -.
* Dates are ``datetime.date``; ``Ledger.months`` holds the first day of each month.
* Sequences are tuples (frozen dataclasses must stay hashable/immutable);
  mappings inside frozen classes are ordinary ``dict`` fields, never mutated after
  creation (``RowMap.cells`` follows the design's ``dict`` type).
* ``Party`` = ``name`` + ``weight`` (relative size; customers and suppliers share
  the class). ``CompanyProfile.first_invoice_number`` was added because the
  pseudocode uses it; ``base_turnover`` was added for the same reason.
* ``EditOp.original`` / ``.new`` are strings (the printed form), as in the design.
* ``FileOp`` = ``op`` ("overlay" | "set_font" | "set_metadata" | "resave"),
  ``row_id``, ``field``, ``new`` (text to draw, if any), ``font``, ``background``
  (overlay box yes/no) and ``params`` (sorted tuple of (key, value) string pairs
  for metadata values such as producer or dates).
* ``TamperLabel`` mirrors the manifest ``label`` block: ``tamper_types`` tuple,
  ``method`` ("" for genuine), the six 0/1-style flags (``overlay_bg_matched`` is
  ``None`` when there is no overlay), ``affected_rows`` as tuple of dicts,
  ``affected_months`` tuple of "YYYY-MM", ``should_be_seen_by`` tuple and ``notes``.
  ``n_affected_rows`` is a derived property (len of ``affected_rows``).
* ``CellBox`` is as in the design (page, top, x0, x1, text). ``RowMap.cells`` maps
  row_id -> {"date","description","amount","balance"} -> CellBox;
  ``pages_per_month`` as in the design.
* ``PrintedRow``: row_id, page, date_text, description_text, amount_text,
  balance_text, balance (float printed value).
* ``SectionSummary``: month ("YYYY-MM"), opening, total_withdrawals,
  total_deposits, closing (positive totals), as in the manifest example.
* ``RenderResult`` adds ``n_pages`` and ``spilled_months`` (months continued on a
  second page) so the manifest can report ``spill_not_in_real_samples``.
* ``StyleKnobs``: bank_name, address_lines, producer, creation_time (HH:MM:SS),
  row_step, second_address_line.
* ``SimulatorKnobs``: growth_range, seasonality_strength, instalment_rate (0.04),
  invoice_disorder (0.0), rows_per_month_range (20, 60), plus ``max_rows_per_month``
  is NOT a knob (passed separately, as in the design signature).
* ``BuildConfig``: as in the design; ``tamper_mix`` is a tuple of (type, weight)
  pairs or None (a frozen class should not hold a mutable mapping);
  ``force`` and ``command_line`` added for the build command and README.
* ``DocumentPlan``: doc_id, index, layout, company_id, company_key, split,
  start (date), n_months, tamper_type (None = genuine), variant ("" if none),
  doc_seed, plus ``style`` / ``knobs`` left out: the builder derives them from
  the seed when it runs (plans stay cheap and comparable).
* The tamper catalog is a flat tuple of the 16 type names. Groups, methods and
  "should be seen by" texts belong to ``tamper/catalog.py`` later.
* Column lists follow requirements 9.2 and 9.3 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

GENERATOR_VERSION = "1.0.0"

LAYOUTS: tuple[str, ...] = (
    "generator_v1",
    "two_col_dmon",
    "two_col_slash",
    "two_col_dual_date",
    "signed_amount_iso",
    "two_col_ddmmyy_cr",
)

TAMPER_TYPES: tuple[str, ...] = (
    "edited_amount",
    "altered_date",
    "broken_balance",
    "font_mismatch",
    "metadata_inconsistency",
    "genuine_resaved",
    "fabricated_receipts",
    "cross_month_copy",
    "turnover_inflation",
    "structuring",
    "sudden_large_deposit",
    "shell_counterparty",
    "rapid_pass_through",
    "dormant_reactivation",
    "period_gap",
    "uen_mismatch",
)

TAMPER_METHODS: tuple[str, ...] = ("overlay", "rerender", "content", "file")

SPLITS: tuple[str, ...] = ("train", "validation", "test", "test_unseen_layout")

LABELS_COLUMNS: tuple[str, ...] = (
    "doc_id", "file", "split", "layout", "company_id", "doc_seed", "n_months",
    "first_month", "last_month", "n_pages", "n_rows", "is_tampered",
    "tamper_types", "tamper_method", "arithmetic_consistent", "genuine_resaved",
    "metadata_altered", "font_inconsistent", "date_inconsistent",
    "overlay_bg_matched", "hidden_text_layer", "n_affected_rows",
    "spill_not_in_real_samples", "genuine_hard_cases", "generator_version",
)

ROWS_COLUMNS: tuple[str, ...] = (
    "doc_id", "row_index", "row_id", "page", "month", "date", "amount",
    "balance", "kind", "is_fraud_row", "fraud_type",
)


@dataclass(frozen=True)
class Party:
    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class CompanyProfile:
    company_id: str
    name: str
    uen: str
    address_lines: tuple[str, ...]
    account_number: str
    archetype: str                      # "services" | "trading" | "fnb" | "logistics"
    customers: tuple[Party, ...]
    suppliers: tuple[Party, ...]
    landlord: str
    payroll_base: float
    seed: int
    first_invoice_number: int = 1000
    base_turnover: float = 0.0


@dataclass(frozen=True)
class LedgerRow:
    row_id: str
    month_index: int
    date: date
    description: str
    amount: float
    kind: str
    customer: str | None = None
    invoice_ref: str | None = None
    hard_case: str | None = None
    fraud: bool = False


@dataclass(frozen=True)
class Ledger:
    company: CompanyProfile
    opening_balance: float
    months: tuple[date, ...]
    rows: tuple[LedgerRow, ...]
    printed_balances: tuple[float, ...] | None = None
    printed_holder: str | None = None
    trims: tuple[tuple[int, int], ...] = ()    # (month_index, n_dropped) from a row cap


@dataclass(frozen=True)
class EditOp:
    op: str
    row_id: str | None = None
    field: str | None = None
    original: str | None = None
    new: str | None = None


@dataclass(frozen=True)
class FileOp:
    op: str
    row_id: str | None = None
    field: str | None = None
    new: str | None = None
    font: str | None = None
    background: bool | None = None
    params: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TamperLabel:
    is_tampered: bool = False
    tamper_types: tuple[str, ...] = ()
    method: str = ""
    arithmetic_consistent: bool = True
    metadata_altered: bool = False
    font_inconsistent: bool = False
    date_inconsistent: bool = False
    overlay_bg_matched: bool | None = None
    hidden_text_layer: bool = False
    genuine_resaved: bool = False
    affected_rows: tuple[dict, ...] = ()
    affected_months: tuple[str, ...] = ()
    should_be_seen_by: tuple[str, ...] = ()
    notes: str = ""

    @property
    def n_affected_rows(self) -> int:
        return len(self.affected_rows)


@dataclass(frozen=True)
class TamperResult:
    ledger: Ledger
    edits: tuple[EditOp, ...]
    file_ops: tuple[FileOp, ...]
    label: TamperLabel


@dataclass(frozen=True)
class CellBox:
    page: int
    top: float
    x0: float
    x1: float
    text: str


@dataclass(frozen=True)
class RowMap:
    cells: dict
    pages_per_month: tuple[int, ...]


@dataclass(frozen=True)
class PrintedRow:
    row_id: str
    page: int
    date_text: str
    description_text: str
    amount_text: str
    balance_text: str
    balance: float


@dataclass(frozen=True)
class SectionSummary:
    month: str
    opening: float
    total_withdrawals: float
    total_deposits: float
    closing: float


@dataclass(frozen=True)
class RenderResult:
    pdf: bytes
    row_map: RowMap
    printed: tuple[PrintedRow, ...]
    section_summaries: tuple[SectionSummary, ...]
    n_pages: int = 0
    spilled_months: tuple[str, ...] = ()


@dataclass(frozen=True)
class StyleKnobs:
    bank_name: str = ""
    address_lines: tuple[str, ...] = ()
    producer: str = ""
    creation_time: str = "09:00:00"
    row_step: float = 0.0
    second_address_line: bool = False


@dataclass(frozen=True)
class SimulatorKnobs:
    growth_range: tuple[float, float] = (-0.01, 0.03)
    seasonality_strength: float = 0.15
    instalment_rate: float = 0.04
    invoice_disorder: float = 0.0
    rows_per_month_range: tuple[int, int] = (20, 60)


@dataclass(frozen=True)
class BuildConfig:
    out_dir: Path
    master_seed: int
    per_layout_genuine: int
    per_layout_tampered: int
    months_min: int | None = None
    months_max: int | None = None
    hold_out_layout: str | None = None
    tamper_mix: tuple[tuple[str, float], ...] | None = None
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    force: bool = False
    command_line: str = ""
    uen_snapshot: Path | None = None
    check_live_uen: bool = False


@dataclass(frozen=True)
class DocumentPlan:
    doc_id: str
    index: int
    layout: str
    company_id: str
    company_key: str
    split: str
    start: date
    n_months: int
    doc_seed: int
    tamper_type: str | None = None
    variant: str = ""
