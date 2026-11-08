"""Failing tests for the Comparison_Report (task 22.1; implemented in task 22.2).

Pinned interface of ``fraud_detect.ml.retraining.comparison_report``
=====================================================================

Constants (str):
    SYNTHETIC_WORDS      = "synthetic data"
    CIRCULARITY_STATEMENT  contains "judged on the generator's output"
    ROW_MARKER           = "synthetic data, not real-world accuracy"
    FORGERY_NOTE         = "the Current model may have been fitted on similar documents"
    NO_SWITCH_TEXT       = "no switch is possible in this mode"
    MIN_SEEDS            = 5

Dataclasses (frozen, keyword construction):
    ReportHeader(switch_rule_text, switch_rule_sha256, cutoff_rule: Mapping[str, float],
                 awkward_seed: int, awkward_version: str, real_style_role: str)
        cutoff_rule maps "medium"/"high" to the target false-alarm rate in percent.
    MetricStat(name, mean, minimum, maximum, std, n_pos, n_neg,
               computable=True, indicative_only=False)
    ResultRow(node "6a"|"6b", source, split, model "Candidate"|"Current"|"Candidate_all11",
              metrics: tuple[MetricStat, ...], layout=None, file=None, held_out=True)
    AuditRow(feature, decision, description, leak_auc: Optional[float], leak_note="")
        leak_auc None means "not computable".
    ComparisonReport(header, corpus_seeds: tuple[int], held_out_layout, split_counts: Mapping,
                     layout_counts: Mapping, total_documents: int,
                     seeds_attempted: Mapping[int, str]  ("completed" or "failed: ..."),
                     skipped: Mapping[split, (count, percent)],
                     real_style_files: tuple[Mapping(path, sha256)],
                     decisions: tuple[Mapping]  (DecisionLog entries; kind "waive-gate" = waiver),
                     post_test_changes: tuple[Mapping(description, ...)],
                     code_unchanged: bool, gnn_coverage_pct: float,
                     audit_rows: tuple[AuditRow], rows: tuple[ResultRow],
                     switch_text: str = "")

Functions:
    banner_lines() -> list[str]            (at most 4 lines, no table)
    render_header_markdown(header) -> str
    render_markdown(report) -> str         (starts with render_header_markdown(report.header))
    render_console(report) -> str
    render_json(report) -> str
    write_header(run_dir, header, is_ignored=None) -> None
    write_report(run_dir, report, is_ignored=None) -> None   (writes through safe_io)
        both write run_dir/comparison_report.md and run_dir/comparison_report.json;
        write_report raises vault.RuleHashMismatch if run_dir/run.json records a different
        "switch_rule_sha256"; renderers raise ValueError if split_counts do not sum to
        total_documents.

Layout of the text (Markdown file and console are identical for the lines below):
    - The results section starts at a line "## Results" (Markdown) or "== Results ==" (console).
      Everything in the header (banner, Switch_Rule text and hash, cutoff rule, awkward seed and
      version, Real_Style limitation) comes before it, and the banner is in the first 10 lines.
    - One result line per (ResultRow, metric), same text in both outputs:
      "| {node} | {source} | {split} | {model} | {metric} | {mean:.4f} | {min:.4f} | {max:.4f} |
       {std:.4f} | {n_pos} pos / {n_neg} neg | {flags} | {marker} |", where flags may hold
      "indicative only", "not computable" (then no numbers), "no longer held out",
      "informational" (always for Candidate_all11); a layout adds "layout=<name>" and a file
      adds "file=<name>" to the line; marker = ROW_MARKER for every source except real_style;
      forgery_dataset lines also hold FORGERY_NOTE.
    - Audit lines: "| {feature} | {decision} | {description} | {leak_auc:.4f or not computable} | ...".
    - Counts: "split <name>: <n>", "layout <name>: <n>", "seed <n>: <outcome>",
      "master seed: <n>", "held-out layout: <name>", "skipped <split>: <count> (<pct:.1f>%)",
      "real_style <path> sha256 <hex>", "generator and detector code unchanged since run start:
      yes|no", "GNN lookup coverage: 0.0%", "incomplete" plus failed seeds when fewer than
      MIN_SEEDS seeds completed (the word is absent otherwise).
"""
from __future__ import annotations

import json
import re

import pytest

from fraud_detect.ml.retraining import comparison_report as cr
from fraud_detect.ml.retraining.safe_io import WriteGuardError
from fraud_detect.ml.retraining.switch_rule import SwitchRule
from fraud_detect.ml.retraining.vault import RuleHashMismatch

RULE = SwitchRule()
SEEDS = (101, 102, 103, 104, 105)
LAYOUTS = {"layout_a": 30, "layout_b": 20, "layout_hold": 10}
SPLITS = {"train": 30, "validation": 10, "test": 10, "test_unseen_layout": 10}
SIGMA = re.compile(r"\d+\.\d{4}")


def _stat(name="auc", mean=0.9, **kw):
    base = dict(name=name, mean=mean, minimum=mean - 0.05, maximum=mean + 0.05, std=0.0123,
                n_pos=40, n_neg=50)
    base.update(kw)
    return cr.MetricStat(**base)


def _header(role="required"):
    return cr.ReportHeader(
        switch_rule_text=RULE.text(), switch_rule_sha256=RULE.sha256(),
        cutoff_rule={"medium": 5.0, "high": 1.0}, awkward_seed=20260701,
        awkward_version="awkward-v1", real_style_role=role)


def _row(source, model="Candidate", node="6b", split="test", metrics=None, **kw):
    return cr.ResultRow(node=node, source=source, split=split, model=model,
                        metrics=tuple(metrics or (_stat(),)), **kw)


def _report(**over):
    seeds = {s: "completed" for s in SEEDS}
    rows = (
        _row("generator_corpus"),
        _row("generator_corpus", model="Current"),
        _row("generator_corpus", model="Candidate_all11"),
        _row("generator_corpus", split="test_unseen_layout", layout="layout_hold"),
        _row("generator_corpus", split="test_unseen_layout", layout="layout_b",
             metrics=(_stat(mean=0.7777),)),
        _row("forgery_dataset", split="test_in_layout"),
        _row("awkward_set", split="all"),
        _row("real_style", split="all", file="statement_E.pdf",
             metrics=(_stat(mean=0.8888, indicative_only=True, n_pos=1, n_neg=0),)),
        _row("generator_corpus", node="6a", metrics=(
            cr.MetricStat(name="pr_auc", mean=None, minimum=None, maximum=None, std=None,
                          n_pos=0, n_neg=17, computable=False),)),
        _row("generator_corpus", split="validation", held_out=False,
             metrics=(_stat(mean=0.5432),)),
    )
    kw = dict(
        header=_header(), corpus_seeds=(7001, 7002), held_out_layout="layout_hold",
        split_counts=SPLITS, layout_counts=LAYOUTS, total_documents=60,
        seeds_attempted=seeds, skipped={"train": (1, 3.3), "test": (0, 0.0)},
        real_style_files=({"path": "/x/statement_E.pdf", "sha256": "ab" * 32},
                          {"path": "/x/statement_F.pdf", "sha256": "cd" * 32}),
        decisions=({"id": "D-0001", "kind": "waive-gate", "target": "capability",
                    "decision": "waive", "reason": "owner accepts missing tamper type",
                    "author": "owner", "date": "2026-08-01"},
                   {"id": "D-0002", "kind": "leak-feature", "target": "FEAT_alpha",
                    "decision": "exclude", "reason": "suspected leak", "author": "owner",
                    "date": "2026-08-01"}),
        post_test_changes=({"description": "changed medium cutoff after test",
                            "kind": "cutoff"},),
        code_unchanged=True, gnn_coverage_pct=0.0,
        audit_rows=(cr.AuditRow("FEAT_alpha", "drop", "what alpha measures", 0.5123),
                    cr.AuditRow("FEAT_beta", "keep", "what beta measures", None)),
        rows=rows,
        switch_text="Node 6b: keep (not decisive)",
    )
    kw.update(over)
    return cr.ComparisonReport(**kw)


def _section(text, marker):
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() in marker)
    return lines[:idx], lines[idx:]


def _outputs(report):
    md, con = cr.render_markdown(report), cr.render_console(report)
    return {"md": (md, ("## Results",)), "console": (con, ("== Results ==",))}


def _result_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith(("| 6a |", "| 6b |"))]


# ---------------------------------------------------------------- banner and header

def test_banner_lines_are_short_and_carry_both_statements():
    lines = cr.banner_lines()
    joined = "\n".join(lines).lower()
    assert len(lines) <= 4
    assert cr.SYNTHETIC_WORDS in joined
    assert "judged on the generator's output" in joined
    assert not any(ln.startswith("|") for ln in lines)


@pytest.mark.parametrize("which", ["md", "console"])
def test_first_ten_lines_carry_banner_before_any_table(which):
    text, marker = _outputs(_report())[which]
    first = "\n".join(text.splitlines()[:10]).lower()
    assert "synthetic data" in first
    assert "judged on the generator's output" in first
    assert not any(ln.startswith("|") for ln in text.splitlines()[:10])


@pytest.mark.parametrize("which", ["md", "console"])
def test_header_content_comes_before_results(which):
    text, marker = _outputs(_report())[which]
    before, after = _section(text, marker)
    head = "\n".join(before)
    for line in RULE.text().splitlines():
        assert line in head
    assert RULE.sha256() in head
    assert "target false-alarm rate" in head
    assert "5.0%" in head and "1.0%" in head
    assert "20260701" in head and "awkward-v1" in head
    assert cr.NO_SWITCH_TEXT in head
    assert not any(ln.startswith("| 6") for ln in before)


def test_veto_only_mode_has_no_limitation_text():
    report = _report(header=_header(role="veto_only"))
    for text, marker in _outputs(report).values():
        assert cr.NO_SWITCH_TEXT not in text


def test_markdown_starts_with_header_markdown():
    report = _report()
    assert cr.render_markdown(report).startswith(cr.render_header_markdown(report.header))


# ---------------------------------------------------------------- result rows

@pytest.mark.parametrize("which", ["md", "console"])
def test_markers_on_result_rows(which):
    text, marker = _outputs(_report())[which]
    lines = _result_lines(text)
    assert lines
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        source = cells[1]
        if source != "real_style":
            assert cr.ROW_MARKER in ln, ln
        if source == "forgery_dataset":
            assert cr.FORGERY_NOTE in ln, ln
        else:
            assert cr.FORGERY_NOTE not in ln, ln


@pytest.mark.parametrize("which", ["md", "console"])
def test_flags_on_rows(which):
    text, marker = _outputs(_report())[which]
    lines = _result_lines(text)
    indicative = [ln for ln in lines if "0.8888" in ln]
    assert len(indicative) == 1 and "indicative only" in indicative[0]
    assert "file=statement_E.pdf" in indicative[0]
    assert sum("indicative only" in ln for ln in lines) == 1
    not_comp = [ln for ln in lines if "| 6a |" in ln]
    assert len(not_comp) == 1 and "not computable" in not_comp[0]
    assert "17 neg" in not_comp[0] and not SIGMA.search(not_comp[0])
    stale = [ln for ln in lines if "0.5432" in ln]
    assert len(stale) == 1 and "no longer held out" in stale[0]
    assert sum("no longer held out" in ln for ln in lines) == 1
    abl = [ln for ln in lines if "Candidate_all11" in ln]
    assert len(abl) == 1 and "informational" in abl[0]
    assert sum("informational" in ln for ln in lines) == 1


@pytest.mark.parametrize("which", ["md", "console"])
def test_seed_statistics_and_class_counts_per_row(which):
    text, marker = _outputs(_report())[which]
    line = next(ln for ln in _result_lines(text) if "| Current |" in ln)
    for value in ("0.9000", "0.8500", "0.9500", "0.0123"):
        assert value in line
    assert "40 pos / 50 neg" in line


@pytest.mark.parametrize("which", ["md", "console"])
def test_results_per_held_out_layout(which):
    text, marker = _outputs(_report())[which]
    lines = _result_lines(text)
    assert any("layout=layout_hold" in ln for ln in lines)
    assert any("layout=layout_b" in ln and "0.7777" in ln for ln in lines)


# ---------------------------------------------------------------- audit, counts, seeds

@pytest.mark.parametrize("which", ["md", "console"])
def test_audit_row_shows_leak_result_in_same_row(which):
    text, marker = _outputs(_report())[which]
    alpha = [ln for ln in text.splitlines() if ln.startswith("| FEAT_alpha |")]
    beta = [ln for ln in text.splitlines() if ln.startswith("| FEAT_beta |")]
    assert len(alpha) == 1 and "drop" in alpha[0] and "0.5123" in alpha[0]
    assert len(beta) == 1 and "keep" in beta[0] and "not computable" in beta[0]


@pytest.mark.parametrize("which", ["md", "console"])
def test_corpus_facts_and_counts(which):
    text, marker = _outputs(_report())[which]
    assert "master seed: 7001" in text and "master seed: 7002" in text
    assert "held-out layout: layout_hold" in text
    for name, n in SPLITS.items():
        assert f"split {name}: {n}" in text
    for name, n in LAYOUTS.items():
        assert f"layout {name}: {n}" in text
    assert "skipped train: 1 (3.3%)" in text
    assert "skipped test: 0 (0.0%)" in text
    for seed in SEEDS:
        assert f"seed {seed}: completed" in text


def test_counts_must_sum_to_total():
    bad = _report(total_documents=61)
    for fn in (cr.render_markdown, cr.render_console, cr.render_json):
        with pytest.raises(ValueError):
            fn(bad)


@pytest.mark.parametrize("which", ["md", "console"])
def test_complete_run_is_not_marked_incomplete(which):
    text, marker = _outputs(_report())[which]
    assert "incomplete" not in text.lower()


@pytest.mark.parametrize("which", ["md", "console"])
def test_fewer_than_five_completed_seeds_is_incomplete_with_failed_seeds(which):
    attempted = {101: "completed", 102: "completed", 103: "completed", 104: "completed",
                 105: "failed: build error"}
    text, marker = _outputs(_report(seeds_attempted=attempted))[which]
    assert "incomplete" in text.lower()
    assert "seed 105: failed: build error" in text
    assert "failed seeds: 105" in text
    data = json.loads(cr.render_json(_report(seeds_attempted=attempted)))
    assert data["incomplete"] is True and data["failed_seeds"] == [105]


@pytest.mark.parametrize("which", ["md", "console"])
def test_real_style_files_decisions_changes_and_status_lines(which):
    text, marker = _outputs(_report())[which]
    assert f"real_style /x/statement_E.pdf sha256 {'ab' * 32}" in text
    assert f"real_style /x/statement_F.pdf sha256 {'cd' * 32}" in text
    assert "D-0001" in text and "owner accepts missing tamper type" in text
    assert "D-0002" in text and "suspected leak" in text
    assert "waiver" in text.lower()
    assert "changed medium cutoff after test" in text
    assert "generator and detector code unchanged since run start: yes" in text
    assert "GNN lookup coverage: 0.0%" in text
    assert "Node 6b: keep (not decisive)" in text


def test_code_changed_says_no():
    text = cr.render_markdown(_report(code_unchanged=False))
    assert "generator and detector code unchanged since run start: no" in text


# ---------------------------------------------------------------- agreement and JSON

def test_console_and_markdown_result_lines_agree():
    report = _report()
    assert _result_lines(cr.render_console(report)) == _result_lines(cr.render_markdown(report))
    md_audit = [ln for ln in cr.render_markdown(report).splitlines() if ln.startswith("| FEAT_")]
    con_audit = [ln for ln in cr.render_console(report).splitlines() if ln.startswith("| FEAT_")]
    assert md_audit == con_audit


def test_json_agrees_with_text():
    report = _report()
    data = json.loads(cr.render_json(report))
    assert data["synthetic_data"] is True
    assert data["switch_rule_sha256"] == RULE.sha256()
    assert data["cutoff_rule"] == {"medium": 5.0, "high": 1.0}
    assert data["incomplete"] is False
    assert data["code_unchanged"] is True and data["gnn_coverage_pct"] == 0.0
    assert len(data["rows"]) == len(report.rows)
    text = cr.render_markdown(report)
    for row in data["rows"]:
        for metric in row["metrics"]:
            if metric["computable"]:
                assert f"{metric['mean']:.4f}" in text
            else:
                assert metric["mean"] is None
    assert {r["model"] for r in data["rows"]} >= {"Candidate", "Current", "Candidate_all11"}
    assert [a["feature"] for a in data["audit"]] == ["FEAT_alpha", "FEAT_beta"]
    assert data["audit"][0]["leak_auc"] == 0.5123 and data["audit"][1]["leak_auc"] is None


# ---------------------------------------------------------------- writing

def test_write_report_writes_both_files_through_the_guard(tmp_path, is_ignored):
    cr.write_report(tmp_path, _report(), is_ignored=is_ignored)
    md = (tmp_path / "comparison_report.md").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "comparison_report.json").read_text(encoding="utf-8"))
    assert md == cr.render_markdown(_report())
    assert data["switch_rule_sha256"] == RULE.sha256()
    assert {p.name for p in is_ignored.calls} >= {"comparison_report.md", "comparison_report.json"}


def test_write_report_refuses_unignored_path(tmp_path, is_ignored):
    is_ignored.ignored = False
    with pytest.raises(WriteGuardError):
        cr.write_report(tmp_path, _report(), is_ignored=is_ignored)
    assert not (tmp_path / "comparison_report.md").exists()
    assert not (tmp_path / "comparison_report.json").exists()


def test_header_is_written_first_and_results_extend_it(tmp_path, is_ignored):
    header = _header()
    cr.write_header(tmp_path, header, is_ignored=is_ignored)
    first = (tmp_path / "comparison_report.md").read_text(encoding="utf-8")
    assert first.startswith(cr.render_header_markdown(header))
    assert RULE.sha256() in first and "5.0%" in first
    assert not any(ln.startswith("| 6") for ln in first.splitlines())
    assert json.loads((tmp_path / "comparison_report.json").read_text())[
        "switch_rule_sha256"] == RULE.sha256()
    cr.write_report(tmp_path, _report(), is_ignored=is_ignored)
    final = (tmp_path / "comparison_report.md").read_text(encoding="utf-8")
    assert final.startswith(first)
    assert len(final) > len(first)


def test_header_write_refuses_unignored_path(tmp_path, is_ignored):
    is_ignored.ignored = False
    with pytest.raises(WriteGuardError):
        cr.write_header(tmp_path, _header(), is_ignored=is_ignored)
    assert not (tmp_path / "comparison_report.md").exists()


def test_report_hash_must_equal_run_json(tmp_path, is_ignored):
    (tmp_path / "run.json").write_text(json.dumps({"switch_rule_sha256": RULE.sha256()}))
    cr.write_report(tmp_path, _report(), is_ignored=is_ignored)
    (tmp_path / "run.json").write_text(json.dumps({"switch_rule_sha256": "0" * 64}))
    with pytest.raises(RuleHashMismatch):
        cr.write_report(tmp_path, _report(), is_ignored=is_ignored)
