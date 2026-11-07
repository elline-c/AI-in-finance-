"""Comparison_Report: model, Markdown / console / JSON renderers and writers (Task 22.2).

The header (banner, Switch_Rule text and hash, cutoff rule, awkward-set seed and version,
Real_Style limitation) is written before evaluation starts; the full report extends it.
All writes go through ``safe_io.atomic_write``. Named to avoid a clash with
``fraud_detect/ml/report.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

from fraud_detect.ml.retraining.safe_io import atomic_write
from fraud_detect.ml.retraining.vault import RuleHashMismatch

SYNTHETIC_WORDS = "synthetic data"
CIRCULARITY_STATEMENT = ("Every result is judged on the generator's output, so it is circular "
                         "and does not measure real-world accuracy.")
ROW_MARKER = "synthetic data, not real-world accuracy"
FORGERY_NOTE = "the Current model may have been fitted on similar documents"
NO_SWITCH_TEXT = "no switch is possible in this mode"
MIN_SEEDS = 5

REPORT_MD = "comparison_report.md"
REPORT_JSON = "comparison_report.json"
RUN_JSON = "run.json"


@dataclass(frozen=True)
class ReportHeader:
    switch_rule_text: str
    switch_rule_sha256: str
    cutoff_rule: Mapping[str, float]
    awkward_seed: int
    awkward_version: str
    real_style_role: str


@dataclass(frozen=True)
class MetricStat:
    name: str
    mean: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    std: Optional[float]
    n_pos: int
    n_neg: int
    computable: bool = True
    indicative_only: bool = False


@dataclass(frozen=True)
class ResultRow:
    node: str
    source: str
    split: str
    model: str
    metrics: Tuple[MetricStat, ...]
    layout: Optional[str] = None
    file: Optional[str] = None
    held_out: bool = True


@dataclass(frozen=True)
class AuditRow:
    feature: str
    decision: str
    description: str
    leak_auc: Optional[float]
    leak_note: str = ""


@dataclass(frozen=True)
class ComparisonReport:
    header: ReportHeader
    corpus_seeds: Tuple[int, ...]
    held_out_layout: str
    split_counts: Mapping
    layout_counts: Mapping
    total_documents: int
    seeds_attempted: Mapping[int, str]
    skipped: Mapping
    real_style_files: Tuple[Mapping, ...]
    decisions: Tuple[Mapping, ...]
    post_test_changes: Tuple[Mapping, ...]
    code_unchanged: bool
    gnn_coverage_pct: float
    audit_rows: Tuple[AuditRow, ...]
    rows: Tuple[ResultRow, ...]
    switch_text: str = ""


# ---------------------------------------------------------------- helpers

def banner_lines() -> List[str]:
    return [
        "WARNING: all numbers below come from synthetic data, not real-world accuracy.",
        CIRCULARITY_STATEMENT,
    ]


def _failed_seeds(report: ComparisonReport) -> List[int]:
    return sorted(s for s, outcome in report.seeds_attempted.items()
                  if not str(outcome).startswith("completed"))


def _incomplete(report: ComparisonReport) -> bool:
    done = sum(1 for o in report.seeds_attempted.values() if str(o).startswith("completed"))
    return done < MIN_SEEDS


def _validate(report: ComparisonReport) -> None:
    if sum(report.split_counts.values()) != report.total_documents:
        raise ValueError("split counts do not sum to total_documents")
    if sum(report.layout_counts.values()) != report.total_documents:
        raise ValueError("layout counts do not sum to total_documents")


def _header_lines(header: ReportHeader) -> List[str]:
    cutoff = ", ".join(f"{k} {float(v):.1f}%" for k, v in header.cutoff_rule.items())
    lines = ["# Comparison report", ""]
    lines += banner_lines()
    lines += ["", "Switch_Rule:"]
    lines += header.switch_rule_text.splitlines()
    lines += [f"Switch_Rule sha256: {header.switch_rule_sha256}",
              f"Cutoff rule: target false-alarm rate {cutoff}",
              f"awkward seed: {header.awkward_seed}",
              f"awkward version: {header.awkward_version}"]
    if header.real_style_role == "required":
        lines.append(f"Real_Style limitation: only a few real-style files exist; "
                     f"{NO_SWITCH_TEXT}.")
    lines.append("")
    return lines


def render_header_markdown(header: ReportHeader) -> str:
    return "\n".join(_header_lines(header)) + "\n"


def _num(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _result_lines(row: ResultRow) -> List[str]:
    out = []
    for m in row.metrics:
        computable = m.computable and m.mean is not None
        flags = []
        if m.indicative_only:
            flags.append("indicative only")
        if not computable:
            flags.append("not computable")
        if not row.held_out:
            flags.append("no longer held out")
        if row.model == "Candidate_all11":
            flags.append("informational")
        if row.layout:
            flags.append(f"layout={row.layout}")
        if row.file:
            flags.append(f"file={row.file}")
        marker = []
        if row.source != "real_style":
            marker.append(ROW_MARKER)
        if row.source == "forgery_dataset":
            marker.append(FORGERY_NOTE)
        stats = ([_num(m.mean), _num(m.minimum), _num(m.maximum), _num(m.std)]
                 if computable else ["-"] * 4)
        cells = [row.node, row.source, row.split, row.model, m.name, *stats,
                 f"{m.n_pos} pos / {m.n_neg} neg", ", ".join(flags), "; ".join(marker)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _audit_line(a: AuditRow) -> str:
    leak = "not computable" if a.leak_auc is None else f"{a.leak_auc:.4f}"
    return f"| {a.feature} | {a.decision} | {a.description} | {leak} | {a.leak_note} |"


def _body_lines(report: ComparisonReport, heading: Callable[[str], str]) -> List[str]:
    lines: List[str] = []
    lines.append(heading("Results"))
    lines.append("| node | source | split | model | metric | mean | min | max | std | "
                 "class counts | flags | marker |")
    for row in report.rows:
        lines += _result_lines(row)
    if report.switch_text:
        lines += ["", report.switch_text]

    lines += ["", heading("Feature audit"),
              "| feature | decision | description | leak AUC | note |"]
    lines += [_audit_line(a) for a in report.audit_rows]

    lines += ["", heading("Corpus")]
    lines += [f"master seed: {s}" for s in report.corpus_seeds]
    lines.append(f"held-out layout: {report.held_out_layout}")
    lines += [f"split {k}: {v}" for k, v in report.split_counts.items()]
    lines += [f"layout {k}: {v}" for k, v in report.layout_counts.items()]
    lines.append(f"total documents: {report.total_documents}")
    for split, (count, pct) in report.skipped.items():
        lines.append(f"skipped {split}: {count} ({pct:.1f}%)")

    lines += ["", heading("Seeds")]
    lines += [f"seed {s}: {o}" for s, o in report.seeds_attempted.items()]
    failed = _failed_seeds(report)
    if _incomplete(report):
        lines.append("Report status: incomplete (fewer than "
                     f"{MIN_SEEDS} seeds completed)")
        lines.append("failed seeds: " + ", ".join(str(s) for s in failed))

    lines += ["", heading("Real_Style files")]
    lines += [f"real_style {f['path']} sha256 {f['sha256']}" for f in report.real_style_files]

    lines += ["", heading("Decisions and waivers")]
    for d in report.decisions:
        label = "waiver" if d.get("kind") == "waive-gate" else "decision"
        lines.append(f"{d.get('id')} [{label}] {d.get('kind')} {d.get('target')}: "
                     f"{d.get('decision')} - {d.get('reason')} "
                     f"({d.get('author')}, {d.get('date')})")

    lines += ["", heading("Post-test changes")]
    lines += [f"post-test change: {c.get('description')}" for c in report.post_test_changes]

    lines += ["", heading("Status"),
              "generator and detector code unchanged since run start: "
              + ("yes" if report.code_unchanged else "no"),
              f"GNN lookup coverage: {report.gnn_coverage_pct:.1f}%"]
    return lines


def render_markdown(report: ComparisonReport) -> str:
    _validate(report)
    body = _body_lines(report, lambda t: f"## {t}" if t == "Results" else f"### {t}")
    return render_header_markdown(report.header) + "\n".join(body) + "\n"


def render_console(report: ComparisonReport) -> str:
    _validate(report)
    body = _body_lines(report, lambda t: f"== {t} ==")
    return render_header_markdown(report.header) + "\n".join(body) + "\n"


def _header_dict(header: ReportHeader) -> dict:
    return {
        "synthetic_data": True,
        "banner": banner_lines(),
        "switch_rule_text": header.switch_rule_text,
        "switch_rule_sha256": header.switch_rule_sha256,
        "cutoff_rule": {k: float(v) for k, v in header.cutoff_rule.items()},
        "awkward_seed": header.awkward_seed,
        "awkward_version": header.awkward_version,
        "real_style_role": header.real_style_role,
    }


def render_json(report: ComparisonReport) -> str:
    _validate(report)
    data = _header_dict(report.header)
    data.update({
        "corpus_seeds": list(report.corpus_seeds),
        "held_out_layout": report.held_out_layout,
        "split_counts": dict(report.split_counts),
        "layout_counts": dict(report.layout_counts),
        "total_documents": report.total_documents,
        "seeds_attempted": {str(k): v for k, v in report.seeds_attempted.items()},
        "incomplete": _incomplete(report),
        "failed_seeds": _failed_seeds(report),
        "skipped": {k: {"count": c, "percent": p} for k, (c, p) in report.skipped.items()},
        "real_style_files": [dict(f) for f in report.real_style_files],
        "decisions": [dict(d) for d in report.decisions],
        "post_test_changes": [dict(c) for c in report.post_test_changes],
        "code_unchanged": report.code_unchanged,
        "gnn_coverage_pct": report.gnn_coverage_pct,
        "switch_text": report.switch_text,
        "audit": [{"feature": a.feature, "decision": a.decision,
                   "description": a.description, "leak_auc": a.leak_auc,
                   "leak_note": a.leak_note} for a in report.audit_rows],
        "rows": [{
            "node": r.node, "source": r.source, "split": r.split, "model": r.model,
            "layout": r.layout, "file": r.file, "held_out": r.held_out,
            "metrics": [{"name": m.name, "mean": m.mean, "min": m.minimum,
                         "max": m.maximum, "std": m.std, "n_pos": m.n_pos,
                         "n_neg": m.n_neg, "computable": m.computable,
                         "indicative_only": m.indicative_only} for m in r.metrics],
        } for r in report.rows],
    })
    return json.dumps(data, indent=2, sort_keys=True)


# ---------------------------------------------------------------- writers

def write_header(run_dir, header: ReportHeader, is_ignored: Optional[Callable] = None) -> None:
    run_dir = Path(run_dir)
    atomic_write(run_dir / REPORT_MD, render_header_markdown(header), is_ignored=is_ignored)
    atomic_write(run_dir / REPORT_JSON, json.dumps(_header_dict(header), indent=2, sort_keys=True),
                 is_ignored=is_ignored)


def _check_rule_hash(run_dir: Path, rule_hash: str) -> None:
    run_json = run_dir / RUN_JSON
    if not run_json.exists():
        return
    recorded = json.loads(run_json.read_text(encoding="utf-8")).get("switch_rule_sha256")
    if recorded is not None and recorded != rule_hash:
        raise RuleHashMismatch("report Switch_Rule hash differs from run.json")


def write_report(run_dir, report: ComparisonReport,
                 is_ignored: Optional[Callable] = None) -> None:
    run_dir = Path(run_dir)
    _check_rule_hash(run_dir, report.header.switch_rule_sha256)
    md, js = render_markdown(report), render_json(report)
    atomic_write(run_dir / REPORT_MD, md, is_ignored=is_ignored)
    atomic_write(run_dir / REPORT_JSON, js, is_ignored=is_ignored)
