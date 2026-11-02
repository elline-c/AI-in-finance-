"""Corpus builder: plans documents, builds them one by one and writes the output tree.

Output tree under ``BuildConfig.out_dir``::

    pdfs/<doc_id>.pdf   manifests/<doc_id>.json   labels.csv   rows.csv
    companies.json      corpus_manifest.json      README.txt

Design notes
------------
* ``plan()`` is cheap and deterministic. The plan of a document depends only on the
  master seed, its layout and its position inside its own group (genuine ``g`` or
  tampered ``t``), never on other documents; the only exception is the tamper type of
  tampered documents, which is dealt out over the whole tampered set so the requested
  mix is met exactly (largest-remainder quotas).
* Every document has its own fictional company (design D7.6), so one company is in
  exactly one split. ``company_id`` is derived from layout, group and position.
* A document that raises is recorded under ``failed`` (doc id, exception type,
  message); the other documents are still written.
* Nothing is written until the settings and the output folder were validated. The
  builder refuses folders inside the repository's ``fraud_detect/``, ``tests/`` and
  ``.kiro/`` trees, and non-empty folders unless ``force`` is set (existing files are
  never deleted).
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.labels import (
    build_manifest,
    write_labels_csv,
    write_manifest,
    write_rows_csv,
)
from fraud_detect.corpus.layouts import get_renderer
from fraud_detect.corpus.ledger import simulate_ledger
from fraud_detect.corpus.finish import make_style
from fraud_detect.corpus.schema import (
    GENERATOR_VERSION,
    LAYOUTS,
    SPLITS,
    TAMPER_TYPES,
    BuildConfig,
    DocumentPlan,
    CompanyProfile,
    Ledger,
    SimulatorKnobs,
    TamperLabel,
    TamperResult,
)
from fraud_detect.corpus.seeding import rng as make_rng
from fraud_detect.corpus.seeding import stable_seed
from fraud_detect.corpus.splits import assign_split, find_gaps, validate_fractions
from fraud_detect.corpus.uen_check import UenChecker
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops

__all__ = ["BuildConfig", "CorpusBuilder", "CorpusSummary"]

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTECTED = ("fraud_detect", "tests", ".kiro")
MONTHS_LOW, MONTHS_HIGH = 1, 24          # accepted range of explicit month settings
DEFAULT_MONTHS = (6, 12)                  # drawn range when not given
SYNTHETIC_NOTICE = ("SYNTHETIC DATA: every file in this folder is fictional and was generated "
                    "by software. No real company, person or bank statement is included.")


@dataclasses.dataclass
class CorpusSummary:
    out_dir: Path
    planned: int
    built: int
    failed: list
    counts: dict
    warnings: list


NOT_CHECKED_WARNING = ("UENs were not checked against the real registry: they were compared only "
                       "with the in-repo registry and with each other (use --uen-snapshot or "
                       "--check-live-uen to check them against real company numbers).")


def _uen_line(info: dict) -> str:
    checked = "yes" if info["real_registry_checked"] else "no"
    text = (f"UEN check: source={info['source']}; UENs checked={info['uens_checked']}; "
            f"redraws={info['redraws']}; real registry checked={checked}")
    if info["snapshot_file"]:
        text += f"; snapshot={info['snapshot_file']} ({info['snapshot_uens']} UENs)"
    return text


class CorpusBuilder:
    def __init__(self, cfg: BuildConfig, uen_fetch: Optional[Callable[[str], dict]] = None) -> None:
        # ``uen_fetch`` is a fake JSON fetcher for tests; None means the registry's own.
        self.cfg = cfg
        self._uen_fetch = uen_fetch
        self._validate()

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        cfg = self.cfg
        lo, hi = self._month_range()
        if not (MONTHS_LOW <= lo <= MONTHS_HIGH and MONTHS_LOW <= hi <= MONTHS_HIGH):
            raise ValueError(f"months must be between {MONTHS_LOW} and {MONTHS_HIGH}, got {lo}..{hi}")
        if lo > hi:
            raise ValueError(f"months_min {lo} is larger than months_max {hi}")
        validate_fractions(cfg.split_fractions)
        if cfg.hold_out_layout is not None and cfg.hold_out_layout not in LAYOUTS:
            raise ValueError(f"unknown hold-out layout {cfg.hold_out_layout!r}; "
                             f"allowed: {', '.join(LAYOUTS)}")
        if cfg.per_layout_genuine < 0 or cfg.per_layout_tampered < 0:
            raise ValueError("document counts must not be negative")
        if cfg.tamper_mix is not None:
            for name, weight in cfg.tamper_mix:
                if name not in TAMPER_TYPES:
                    raise ValueError(f"unknown tamper type {name!r}")
                if weight < 0:
                    raise ValueError(f"tamper weight for {name!r} is negative")
            if sum(w for _, w in cfg.tamper_mix) <= 0:
                raise ValueError("tamper mix has no positive weight")
        out = Path(cfg.out_dir).expanduser().resolve()
        for name in _PROTECTED:
            root = (REPO_ROOT / name).resolve()
            if out == root or root in out.parents:
                raise ValueError(f"refusing to write the corpus inside {name}/ ({out})")

    def _month_range(self) -> tuple[int, int]:
        lo, hi = self.cfg.months_min, self.cfg.months_max
        if lo is None and hi is None:
            return DEFAULT_MONTHS
        return (lo if lo is not None else min(DEFAULT_MONTHS[0], hi),
                hi if hi is not None else max(DEFAULT_MONTHS[1], lo))

    def _check_out_dir(self) -> Path:
        out = Path(self.cfg.out_dir)
        if out.exists() and not out.is_dir():
            raise ValueError(f"output path {out} is not a folder")
        if out.exists() and any(out.iterdir()) and not self.cfg.force:
            raise ValueError(f"output folder {out} is not empty; use force to build into it anyway")
        return out

    # ------------------------------------------------------------------------ plan
    def _tamper_types(self, count: int) -> list[str]:
        """Deal ``count`` tamper types by the mix (largest-remainder quotas), shuffled."""
        if count == 0:
            return []
        mix = self.cfg.tamper_mix
        items = [(n, float(w)) for n, w in mix if w > 0] if mix else [(t, 1.0) for t in TAMPER_TYPES]
        total = sum(w for _, w in items)
        quotas = [count * w / total for _, w in items]
        base = [int(q) for q in quotas]
        order = sorted(range(len(items)), key=lambda i: (-(quotas[i] - base[i]), i))
        for i in order[: count - sum(base)]:
            base[i] += 1
        types = [n for (n, _), k in zip(items, base) for _ in range(k)]
        make_rng(self.cfg.master_seed, "tamper-order").shuffle(types)
        return types

    def plan(self) -> list[DocumentPlan]:
        cfg = self.cfg
        lo, hi = self._month_range()
        n_tampered = cfg.per_layout_tampered * len(LAYOUTS)
        types = iter(self._tamper_types(n_tampered))
        plans: list[DocumentPlan] = []
        for li, layout in enumerate(LAYOUTS, start=1):
            for kind, count in (("g", cfg.per_layout_genuine), ("t", cfg.per_layout_tampered)):
                for k in range(1, count + 1):
                    doc_id = f"{layout}-{kind}{k:04d}"
                    company_id = f"C{li}{kind.upper()}{k:04d}"
                    r = make_rng(cfg.master_seed, "plan", doc_id)
                    n_months = r.randint(lo, hi)
                    start = date(2025, 1, 1)
                    idx = start.year * 12 + r.randrange(12)
                    start = date(idx // 12, idx % 12 + 1, 1)
                    plans.append(DocumentPlan(
                        doc_id=doc_id, index=k, layout=layout, company_id=company_id,
                        company_key=doc_id,
                        split=assign_split(cfg.master_seed, company_id, layout,
                                           cfg.hold_out_layout, cfg.split_fractions),
                        start=start, n_months=n_months,
                        doc_seed=stable_seed(cfg.master_seed, "doc", layout, kind, k),
                        tamper_type=next(types) if kind == "t" else None))
        return plans

    # ----------------------------------------------------------------------- build
    def _company_seed(self, plan: DocumentPlan) -> int:
        return stable_seed(self.cfg.master_seed, "company", plan.company_key)

    def _choose_companies(self, plans: list[DocumentPlan], checker: UenChecker
                          ) -> dict[str, CompanyProfile]:
        """Make every company in plan order so each UEN is free (Requirement 20).

        Plan order is fixed by the seed, so which of two clashing companies keeps its
        first UEN does not depend on build speed or on failures.
        """
        claimed: set[str] = set()

        def taken(uen: str) -> bool:
            clash = checker.is_taken(uen) or uen in claimed
            if clash:
                checker.redraws += 1
            return clash

        profiles: dict[str, CompanyProfile] = {}
        for plan in plans:
            profile = make_company(self._company_seed(plan), plan.company_id, is_taken=taken)
            claimed.add(profile.uen)
            profiles[plan.doc_id] = profile
        return profiles

    def _build_document(self, plan: DocumentPlan, out: Path,
                        company: Optional[CompanyProfile] = None) -> tuple[dict, dict]:
        cfg = self.cfg
        if company is None:
            company = make_company(self._company_seed(plan), plan.company_id)
        renderer = get_renderer(plan.layout)
        ledger: Ledger = simulate_ledger(company, plan.start, plan.n_months, plan.doc_seed,
                                         SimulatorKnobs(), renderer.max_rows_per_month)
        n = len(ledger.months)
        record = {
            "company_id": company.company_id, "uen": company.uen,
            "name": company.name, "registered_name": company.name,
            "baseline": {
                "avg_monthly_credit": round(sum(r.amount for r in ledger.rows if r.amount > 0) / n, 2),
                "avg_monthly_debit": round(-sum(r.amount for r in ledger.rows if r.amount < 0) / n, 2),
            },
        }
        if plan.tamper_type is None:
            result = TamperResult(ledger=ledger, edits=(), file_ops=(), label=TamperLabel())
        else:
            tamper_rng = make_rng(cfg.master_seed, "tamper", plan.doc_id)
            result = CATALOG[plan.tamper_type].plan(ledger, renderer, tamper_rng)
        style = make_style(cfg.master_seed, plan.doc_id)
        rendered = renderer.render(result.ledger, style, plan.doc_seed)
        pdf = apply_file_ops(rendered.pdf, result.file_ops, rendered.row_map, plan.doc_seed)
        rendered = dataclasses.replace(rendered, pdf=pdf)
        manifest = build_manifest(plan, result.ledger, rendered, result.label,
                                  master_seed=cfg.master_seed, edits=result.edits,
                                  file_ops=result.file_ops)
        (out / "pdfs").mkdir(parents=True, exist_ok=True)
        (out / "pdfs" / f"{plan.doc_id}.pdf").write_bytes(pdf)
        write_manifest(out / "manifests" / f"{plan.doc_id}.json", manifest)
        return manifest, record

    def build(self, progress: Optional[Callable[[int, int, DocumentPlan], None]] = None
              ) -> CorpusSummary:
        cfg = self.cfg
        plans = self.plan()
        out = self._check_out_dir()
        # UEN check first: a bad snapshot or a live lookup that cannot answer raises
        # UenCheckError here, before the output folder is created.
        checker = UenChecker.from_options(cfg.uen_snapshot, live=cfg.check_live_uen,
                                          fetch=self._uen_fetch)
        profiles = self._choose_companies(plans, checker)
        uen_info = checker.describe()
        out.mkdir(parents=True, exist_ok=True)
        (out / "pdfs").mkdir(exist_ok=True)
        (out / "manifests").mkdir(exist_ok=True)

        manifests: list[dict] = []
        companies: list[dict] = []
        failed: list[dict] = []
        for i, plan in enumerate(plans, start=1):
            try:
                manifest, record = self._build_document(plan, out, profiles[plan.doc_id])
            except Exception as exc:  # recorded, the rest of the corpus is still built
                failed.append({"doc_id": plan.doc_id, "layout": plan.layout,
                               "tamper_type": plan.tamper_type,
                               "error_type": type(exc).__name__, "message": str(exc)})
            else:
                manifests.append(manifest)
                companies.append(record)
            if progress is not None:
                progress(i, len(plans), plan)

        counts = self._counts(manifests)
        warnings = self._warnings(counts, failed)
        if not uen_info["real_registry_checked"]:
            warnings.append(NOT_CHECKED_WARNING)
        write_labels_csv(out / "labels.csv", manifests)
        write_rows_csv(out / "rows.csv", manifests)
        self._write_json(out / "companies.json", companies)
        corpus_manifest = {
            "synthetic_data_notice": SYNTHETIC_NOTICE,
            "generator_version": GENERATOR_VERSION,
            "master_seed": cfg.master_seed,
            "command_line": cfg.command_line,
            "hold_out_layout": cfg.hold_out_layout,
            "split_fractions": list(cfg.split_fractions),
            "tamper_mix": [list(p) for p in cfg.tamper_mix] if cfg.tamper_mix else None,
            "uen_check": uen_info,
            "counts": counts,
            "failed": failed,
            "warnings": warnings,
        }
        self._write_json(out / "corpus_manifest.json", corpus_manifest)
        (out / "README.txt").write_text(self._readme(counts, failed, warnings, uen_info), encoding="utf-8")
        return CorpusSummary(out_dir=out, planned=len(plans), built=len(manifests),
                             failed=failed, counts=counts, warnings=warnings)

    # --------------------------------------------------------------------- outputs
    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    @staticmethod
    def _counts(manifests: list[dict]) -> dict:
        per_layout = Counter(m["layout"] for m in manifests)
        per_split = Counter(m["split"] for m in manifests)
        per_type: dict[str, Counter] = {s: Counter() for s in per_split}
        for m in manifests:
            for t in m["label"]["tamper_types"]:
                per_type[m["split"]][t] += 1
        return {
            "documents": len(manifests),
            "per_layout": {k: per_layout[k] for k in LAYOUTS if per_layout[k]},
            "per_split": {k: per_split[k] for k in SPLITS if per_split[k]},
            "per_tamper_type": {s: {t: per_type[s][t] for t in TAMPER_TYPES}
                                for s in SPLITS if s in per_type},
        }

    def _warnings(self, counts: dict, failed: list) -> list[str]:
        warnings = []
        if self.cfg.hold_out_layout is None:
            warnings.append("No layout is held out: there is no test_unseen_layout split "
                            "(use hold_out_layout to measure unseen layouts).")
        if failed:
            warnings.append(f"{len(failed)} document(s) failed and are missing from the corpus.")
        if self.cfg.per_layout_tampered:
            for split, missing in find_gaps(TAMPER_TYPES, counts["per_tamper_type"]).items():
                if missing:
                    warnings.append(f"split {split} has no document of tamper type: "
                                    + ", ".join(missing))
        return warnings

    def _readme(self, counts: dict, failed: list, warnings: list, uen_info: dict) -> str:
        cfg = self.cfg
        lines = [
            "SYNTHETIC DATA",
            "==============",
            SYNTHETIC_NOTICE,
            "",
            f"Generator version : {GENERATOR_VERSION}",
            f"Master seed       : {cfg.master_seed}",
            f"Command line      : {cfg.command_line}",
            f"Held-out layout   : {cfg.hold_out_layout or 'none'}",
            "",
            "Folder contents",
            "  pdfs/                 the statements (one PDF per document)",
            "  manifests/            the full truth for each document (JSON)",
            "  labels.csv            one line per document",
            "  rows.csv              one line per transaction row, with is_fraud_row",
            "  companies.json        fictional companies: UEN, registered name, baseline",
            "  corpus_manifest.json  settings, counts, failures",
            "",
            f"Documents: {counts['documents']}",
            "Per layout: " + ", ".join(f"{k}={v}" for k, v in counts["per_layout"].items()),
            "Per split : " + ", ".join(f"{k}={v}" for k, v in counts["per_split"].items()),
            "",
            _uen_line(uen_info),
            "",
            "The same seed and settings rebuild the same labels and text layers.",
            "Do not commit this folder; do not train on real statements mixed with it unlabelled.",
        ]
        if failed:
            lines += ["", "Failed documents:"] + [
                f"  {f['doc_id']}: {f['error_type']}: {f['message']}" for f in failed]
        if warnings:
            lines += ["", "Warnings:"] + [f"  {w}" for w in warnings]
        return "\n".join(lines) + "\n"
