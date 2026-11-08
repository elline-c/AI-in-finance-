"""Switch_Rule: a pure decision function for switching a node to its Candidate model.

The rule is fixed and hashed before any evaluation. ``decide`` only reads the
evidence passed in; it never touches files and never copies a model (that is
done by the separate model operations, on explicit user command).

All rates are percentages (0..100). Synthetic data only; not real-world accuracy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DEFAULT_DECISIVE: Tuple[str, ...] = ("heldout_layout", "heldout_seed", "forgery_ood", "real_style")
KEEP_6A_TEXT = "keep the GBM and the GNN lookup off in the live app"


@dataclass(frozen=True)
class SwitchRule:
    version: str = "switch-v1"
    decisive: Tuple[str, ...] = DEFAULT_DECISIVE
    real_style_role: str = "required"  # "required" or "veto_only"
    hard_negative_block_points: float = 10.0

    def text(self) -> str:
        return "\n".join([
            f"Switch_Rule {self.version}",
            f"decisive sources: {', '.join(self.decisive)}",
            f"real_style_role: {self.real_style_role}",
            "node 6b: AUC not lower than current by more than the candidate's across-seed "
            "standard deviation; catch rate at the medium cutoff not lower; "
            "false-alarm rate at the medium cutoff not higher",
            "node 6a: PR-AUC not lower; per-statement false-alarm rate not higher",
            "a missing or indicative-only source is not decisive and never passes",
            "any failed decisive source gives keep; no decisive source gives keep",
            f"awkward_set: never passes a node; blocks only when reviewed and the hard-negative "
            f"false-alarm rate rises by more than {self.hard_negative_block_points!r} points",
            "the release candidate must pass the same checks",
            "no baseline or a failed baseline gives keep",
        ]) + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceMetrics:
    missing: bool = False
    indicative_only: bool = False
    # Node 6b
    cand_auc: Optional[float] = None
    cur_auc: Optional[float] = None
    cand_auc_std: Optional[float] = None
    cand_catch_medium: Optional[float] = None
    cur_catch_medium: Optional[float] = None
    cand_fa_medium: Optional[float] = None
    cur_fa_medium: Optional[float] = None
    # Node 6a
    cand_pr_auc: Optional[float] = None
    cur_pr_auc: Optional[float] = None
    cand_fa_stmt: Optional[float] = None
    cur_fa_stmt: Optional[float] = None


@dataclass(frozen=True)
class Evidence:
    baseline_state: str = "ok"  # "ok", "no baseline", "baseline failed"
    metrics: Dict[str, SourceMetrics] = field(default_factory=dict)
    awkward_reviewed: bool = False
    hn_fa_cand: float = 0.0
    hn_fa_cur: float = 0.0
    release_passes_same_checks: bool = True


@dataclass(frozen=True)
class NodeDecision:
    action: str  # "switch" or "keep"
    reason: str
    status: Dict[str, str]
    recommendation: str
    node: str = ""
    lines: Tuple[str, ...] = ()

    def report_text(self) -> str:
        return "\n".join(self.lines)


def _none(*vals) -> bool:
    return any(v is None for v in vals)


def _source_status(node: str, m: Optional[SourceMetrics]) -> str:
    if m is None or m.missing or m.indicative_only:
        return "not decisive"
    if node == "6b":
        if _none(m.cand_auc, m.cur_auc, m.cand_catch_medium, m.cur_catch_medium,
                 m.cand_fa_medium, m.cur_fa_medium):
            return "not decisive"
        std = m.cand_auc_std if m.cand_auc_std is not None else 0.0
        ok = (m.cand_auc >= m.cur_auc - std
              and m.cand_catch_medium >= m.cur_catch_medium
              and m.cand_fa_medium <= m.cur_fa_medium)
    else:
        if _none(m.cand_pr_auc, m.cur_pr_auc, m.cand_fa_stmt, m.cur_fa_stmt):
            return "not decisive"
        ok = m.cand_pr_auc >= m.cur_pr_auc and m.cand_fa_stmt <= m.cur_fa_stmt
    return "pass" if ok else "fail"


def decide(rule: SwitchRule, node: str, evidence: Evidence) -> NodeDecision:
    if node not in ("6a", "6b"):
        raise ValueError(f"unknown node: {node!r}")

    status = {src: _source_status(node, evidence.metrics.get(src)) for src in rule.decisive}
    required = rule.real_style_role == "required"
    rs_undecided = "real_style" in status and status["real_style"] == "not decisive"
    available = [s for s, v in status.items() if v != "not decisive"]
    failed = [s for s, v in status.items() if v == "fail"]

    # Sources that must be decisive for a switch.
    must = [s for s in rule.decisive if s != "real_style" or required]
    undecided = [s for s in must if status[s] == "not decisive"]

    action, reason = "keep", ""
    if evidence.baseline_state != "ok":
        reason = evidence.baseline_state
    elif not available:
        reason = "no decisive source is available"
    elif failed:
        reason = "failed: " + ", ".join(failed)
    elif undecided:
        reason = "not decisive: " + ", ".join(undecided)
    elif (evidence.awkward_reviewed
          and (evidence.hn_fa_cand - evidence.hn_fa_cur) > rule.hard_negative_block_points):
        reason = (f"awkward_set blocks: hard-negative false-alarm rate rose by "
                  f"{evidence.hn_fa_cand - evidence.hn_fa_cur:.1f} points")
    elif not evidence.release_passes_same_checks:
        reason = "release candidate does not pass the same checks"
    else:
        action, reason = "switch", "all decisive sources pass"

    lines = []
    if required and rs_undecided:
        lines.append("Real_Style is not decisive (role: required), so the result is keep.")
    lines.append(f"Node {node}: {action} ({reason})")
    lines.extend(f"  {s}: {v}" for s, v in status.items())
    if not available:
        lines.append("No decisive source is available, so the result is keep.")
    if evidence.awkward_reviewed:
        lines.append(f"  awkward hard-negative gap: "
                     f"{evidence.hn_fa_cand - evidence.hn_fa_cur:.1f} points")
    else:
        lines.append("  awkward_set: not reviewed, no effect")

    if action == "switch":
        rec = f"switch node {node} to the Candidate model (run the switch command yourself)"
    else:
        rec = f"keep the current model for node {node}"
        if node == "6a":
            rec += "; " + KEEP_6A_TEXT
    lines.append(rec)

    return NodeDecision(action=action, reason=reason, status=status,
                        recommendation=rec, node=node, lines=tuple(lines))
