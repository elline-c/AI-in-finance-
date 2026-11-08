"""Task 17.2: the Awkward_Set (definition ``awkward-v1``) and its builder.

A small hand-built set of hard cases made with the public ``fraud_detect.corpus``
API and one fixed master seed. Everything is synthetic and lives in memory only:
documents are plain ``bytes`` and nothing is written to disk.

Five case types, three companies each, one genuine and one tampered document per
company (15 genuine and 15 tampered in total):

* ``seasonal_growth``     existing knobs (seasonality 0.8, growth 0.04); candidate
                          ledgers (different seeds, hence different phases) are
                          measured and only those with peak >= 2 x trough and last
                          3 months >= 1.10 x first 3 months are kept.
* ``legit_resave``        genuine file re-saved by an export Producer, later ModDate.
* ``instalments``         ``instalment_rate`` 0.5; an invoice reference on two receipts.
* ``long_12_months``      exactly 12 consecutive months.
* ``round_payroll_rent``  payroll and rent rewritten to multiples of 100 with
                          ``dataclasses.replace`` (opening balance raised if the
                          replay would go below zero). No generator code changes.

Training modules never import this module (evaluation only).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Callable, Mapping, Optional

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.company import make_company as _make_company_unchecked
from fraud_detect.corpus.finish import make_style
from fraud_detect.corpus.layouts import get_renderer
from fraud_detect.corpus.ledger import simulate_ledger
from fraud_detect.corpus.schema import (
    LAYOUTS,
    CompanyProfile,
    DocumentPlan,
    Ledger,
    SimulatorKnobs,
    TamperLabel,
    TamperResult,
)
from fraud_detect.corpus.seeding import rng as make_rng
from fraud_detect.corpus.seeding import stable_seed
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.ml.retraining.config import AWKWARD_MASTER_SEED as _CONFIG_SEED

__all__ = [
    "AWKWARD_VERSION", "AWKWARD_MASTER_SEED", "CASE_TYPES", "TAMPER_ROTATION",
    "AwkwardBuildError", "AwkwardDoc", "AwkwardSet", "AwkwardOutcome",
    "build_awkward_set", "render_document", "prepare_awkward_set",
    "read_review_record", "awkward_results_status",
]

AWKWARD_VERSION = "awkward-v1"
AWKWARD_MASTER_SEED = _CONFIG_SEED          # 20260701, recorded in the report
CASE_TYPES = ("seasonal_growth", "legit_resave", "instalments", "long_12_months",
              "round_payroll_rent")
TAMPER_ROTATION = ("edited_amount", "fabricated_receipts", "metadata_inconsistency")

COMPANIES_PER_CASE = 3
MIN_PER_CLASS = 3
PEAK_TROUGH_MIN = 2.0
LAST3_FIRST3_MIN = 1.10
ROUND_SHARE_MIN = 0.80
_N_MONTHS = {"long_12_months": 12, "seasonal_growth": 12}
_DEFAULT_MONTHS = 6
_DEFAULT_KNOBS = {
    "seasonal_growth": SimulatorKnobs(seasonality_strength=0.8, growth_range=(0.04, 0.04)),
    "instalments": SimulatorKnobs(instalment_rate=0.5),
}


class AwkwardBuildError(Exception):
    """A case type could not reach its figure; names the type and both counts."""

    def __init__(self, case_type: str, n_genuine: int, n_tampered: int) -> None:
        self.case_type = case_type
        self.n_genuine = n_genuine
        self.n_tampered = n_tampered
        super().__init__(f"Awkward_Set case type {case_type!r} cannot be built: "
                         f"{n_genuine} genuine and {n_tampered} tampered documents "
                         f"(at least {MIN_PER_CLASS} of each are needed)")


@dataclass(frozen=True)
class AwkwardDoc:
    doc_id: str
    case_type: str
    company_id: str
    uen: str
    layout: str
    is_tampered: bool
    tamper_type: Optional[str]
    pdf: bytes
    ledger: Ledger
    company: CompanyProfile


@dataclass(frozen=True)
class AwkwardSet:
    version: str
    seed: int
    docs: tuple

    def genuine(self) -> tuple:
        return tuple(d for d in self.docs if not d.is_tampered)

    def tampered(self) -> tuple:
        return tuple(d for d in self.docs if d.is_tampered)

    def by_case(self, case: str) -> tuple:
        return tuple(d for d in self.docs if d.case_type == case)


# --------------------------------------------------------------------- rendering
def _company_seed(master_seed: int, plan: DocumentPlan) -> int:
    return stable_seed(master_seed, "company", plan.company_key)


def _render(plan: DocumentPlan, *, master_seed: int, company: Optional[CompanyProfile],
            knobs: Optional[SimulatorKnobs],
            rewrite: Optional[Callable[[Ledger], Ledger]] = None):
    """Same steps and random streams as ``CorpusBuilder._build_document``, in memory."""
    if company is None:
        company = make_company(_company_seed(master_seed, plan), plan.company_id)
    renderer = get_renderer(plan.layout)
    ledger = simulate_ledger(company, plan.start, plan.n_months, plan.doc_seed,
                             knobs or SimulatorKnobs(), renderer.max_rows_per_month)
    if rewrite is not None:
        ledger = rewrite(ledger)
    if plan.tamper_type is None:
        result = TamperResult(ledger=ledger, edits=(), file_ops=(), label=TamperLabel())
    else:
        tamper_rng = make_rng(master_seed, "tamper", plan.doc_id)
        result = CATALOG[plan.tamper_type].plan(ledger, renderer, tamper_rng)
    style = make_style(master_seed, plan.doc_id)
    rendered = renderer.render(result.ledger, style, plan.doc_seed)
    pdf = apply_file_ops(rendered.pdf, result.file_ops, rendered.row_map, plan.doc_seed)
    return pdf, result.ledger


def render_document(plan: DocumentPlan, *, master_seed: int,
                    company: Optional[CompanyProfile] = None,
                    knobs: Optional[SimulatorKnobs] = None):
    """Render one document in memory: ``(pdf_bytes, ledger)``."""
    return _render(plan, master_seed=master_seed, company=company, knobs=knobs)


# ------------------------------------------------------------------ figures/recipes
def _credits(ledger: Ledger) -> list:
    out = [0.0] * len(ledger.months)
    for row in ledger.rows:
        if row.amount > 0:
            out[row.month_index] += row.amount
    return out


def _seasonal_ok(ledger: Ledger) -> bool:
    c = _credits(ledger)
    if len(c) < 6 or min(c) <= 0:
        return False
    return max(c) >= PEAK_TROUGH_MIN * min(c) and sum(c[-3:]) >= LAST3_FIRST3_MIN * sum(c[:3])


def _instalments_ok(ledger: Ledger) -> bool:
    counts: dict = {}
    for row in ledger.rows:
        if row.kind == "receipt" and row.invoice_ref:
            counts[row.invoice_ref] = counts.get(row.invoice_ref, 0) + 1
    return any(n >= 2 for n in counts.values())


def _round_to_100(amount: float) -> float:
    sign = -1.0 if amount < 0 else 1.0
    return sign * float(max(100, int(round(abs(amount) / 100.0)) * 100))


def _round_rewrite(ledger: Ledger) -> Ledger:
    rows = tuple(dataclasses.replace(r, amount=_round_to_100(r.amount))
                 if r.kind in ("payroll", "rent") else r for r in ledger.rows)
    # opening balance stays as it was unless the replay dips below 0
    bal = ledger.opening_balance
    low = bal
    for r in rows:
        bal = round(bal + r.amount, 2)
        low = min(low, bal)
    opening = ledger.opening_balance if low >= 0 else round(ledger.opening_balance - low, 2)
    return dataclasses.replace(ledger, rows=rows, opening_balance=opening)


def _round_ok(ledger: Ledger) -> bool:
    rows = [r for r in ledger.rows if r.kind in ("payroll", "rent")]
    if not rows:
        return False
    whole = sum(round(abs(r.amount) * 100) % 10000 == 0 for r in rows)
    return whole / len(rows) >= ROUND_SHARE_MIN


def _long_ok(ledger: Ledger) -> bool:
    idx = [m.year * 12 + m.month for m in ledger.months]
    return len(idx) == 12 and idx == list(range(idx[0], idx[0] + 12))


_FIGURE = {
    "seasonal_growth": _seasonal_ok,
    "instalments": _instalments_ok,
    "round_payroll_rent": _round_ok,
    "long_12_months": _long_ok,
    "legit_resave": lambda ledger: True,
}


def _rewrite_for(case: str) -> Optional[Callable[[Ledger], Ledger]]:
    return _round_rewrite if case == "round_payroll_rent" else None


# ------------------------------------------------------------------------ builder
def _plan_for(seed: int, case: str, k: int, attempt: int, kind: str,
              tamper_type: Optional[str]) -> DocumentPlan:
    company_id = f"AW-{case}-{k}"
    shape = make_rng(seed, "awkward-shape", case, k)
    layout = LAYOUTS[shape.randrange(len(LAYOUTS))]
    r = make_rng(seed, "awkward-start", case, k, attempt)
    idx = 2025 * 12 + r.randrange(12)
    return DocumentPlan(
        doc_id=f"{company_id}-{kind}", index=k, layout=layout, company_id=company_id,
        company_key=company_id, split="awkward", start=date(idx // 12, idx % 12 + 1, 1),
        n_months=_N_MONTHS.get(case, _DEFAULT_MONTHS),
        doc_seed=stable_seed(seed, "awkward-doc", case, k, attempt), tamper_type=tamper_type)


def _draw_company(seed: int, case: str, k: int, uen_checker) -> CompanyProfile:
    company_id = f"AW-{case}-{k}"
    cseed = stable_seed(seed, "company", company_id)
    if uen_checker is None:
        return make_company(cseed, company_id)
    company = make_company(cseed, company_id, is_taken=uen_checker.is_taken)
    # make_company hands the redraw to ``is_taken`` silently; count it for the manifest
    if company.uen != _make_company_unchecked(cseed, company_id).uen:
        uen_checker.redraws += 1
    return company


def build_awkward_set(seed: int = AWKWARD_MASTER_SEED, *, uen_checker=None,
                      knobs: Optional[Mapping] = None, cases=None,
                      max_attempts: int = 12) -> AwkwardSet:
    """Build the Awkward_Set in memory. Deterministic for a fixed ``seed``."""
    case_list = tuple(cases) if cases is not None else CASE_TYPES
    unknown = [c for c in case_list if c not in CASE_TYPES]
    if unknown:
        raise ValueError(f"unknown Awkward_Set case type(s): {', '.join(unknown)}")
    knob_map = dict(_DEFAULT_KNOBS)
    knob_map.update(knobs or {})
    docs: list = []
    tampered_index = 0
    for case in case_list:
        case_knobs = knob_map.get(case)
        rewrite = _rewrite_for(case)
        figure = _FIGURE[case]
        genuine_type = "genuine_resaved" if case == "legit_resave" else None
        accepted: list = []          # (k, attempt, company, genuine_pdf, genuine_ledger)
        attempt = 0
        # candidate ledgers are tried in a fixed order; the first ones meeting the
        # figure fill company slots 1, 2, 3 (a different attempt gives another phase)
        while len(accepted) < COMPANIES_PER_CASE and attempt < max_attempts:
            k = len(accepted) + 1
            plan = _plan_for(seed, case, k, attempt, "g", genuine_type)
            company = _draw_company(seed, case, k, uen_checker)
            pdf, ledger = _render(plan, master_seed=seed, company=company,
                                  knobs=case_knobs, rewrite=rewrite)
            if figure(ledger):
                accepted.append((k, attempt, company, plan, pdf, ledger))
            attempt += 1
        if len(accepted) < MIN_PER_CLASS:
            raise AwkwardBuildError(case, len(accepted), len(accepted))
        for k, att, company, gplan, gpdf, gledger in accepted:
            docs.append(AwkwardDoc(
                doc_id=gplan.doc_id, case_type=case, company_id=gplan.company_id,
                uen=company.uen, layout=gplan.layout, is_tampered=False, tamper_type=None,
                pdf=gpdf, ledger=gledger, company=company))
        for k, att, company, gplan, gpdf, gledger in accepted:
            ttype = TAMPER_ROTATION[tampered_index % len(TAMPER_ROTATION)]
            tampered_index += 1
            tplan = dataclasses.replace(
                gplan, doc_id=f"{gplan.company_id}-t", tamper_type=ttype)
            tpdf, tledger = _render(tplan, master_seed=seed, company=company,
                                    knobs=case_knobs, rewrite=rewrite)
            docs.append(AwkwardDoc(
                doc_id=tplan.doc_id, case_type=case, company_id=tplan.company_id,
                uen=company.uen, layout=tplan.layout, is_tampered=True, tamper_type=ttype,
                pdf=tpdf, ledger=tledger, company=company))
    return AwkwardSet(version=AWKWARD_VERSION, seed=seed, docs=tuple(docs))


# ---------------------------------------------------------------- pipeline outcome
@dataclass(frozen=True)
class AwkwardOutcome:
    ok: bool
    exit_code: int
    message: str
    awkward_set: Optional[AwkwardSet]
    proceed_to_switch_rule: bool


def prepare_awkward_set(*, uen_checker=None, **builder_kwargs) -> AwkwardOutcome:
    """Build the set for a run. On failure the run stops before the Switch_Rule."""
    try:
        built = build_awkward_set(uen_checker=uen_checker, **builder_kwargs)
    except AwkwardBuildError as exc:
        return AwkwardOutcome(ok=False, exit_code=1, message=str(exc), awkward_set=None,
                              proceed_to_switch_rule=False)
    return AwkwardOutcome(
        ok=True, exit_code=0,
        message=(f"{built.version}: {len(built.genuine())} genuine and "
                 f"{len(built.tampered())} tampered documents (seed {built.seed})"),
        awkward_set=built, proceed_to_switch_rule=True)


# ------------------------------------------------------------------ review record
def read_review_record(report: Mapping, version: str = AWKWARD_VERSION) -> Optional[dict]:
    """The owner's review record for ``version`` from the report, else ``None``."""
    record = report.get("awkward_review") if isinstance(report, Mapping) else None
    if not isinstance(record, Mapping) or record.get("version") != version:
        return None
    if not str(record.get("reviewer") or "").strip() or not str(record.get("date") or "").strip():
        return None
    return dict(record)


def awkward_results_status(report: Mapping, version: str = AWKWARD_VERSION) -> str:
    return "reviewed" if read_review_record(report, version) is not None else "unreviewed"
