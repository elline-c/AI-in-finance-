"""Property 38: Report banner and markers (task 22.3).

**Validates: Requirements 3.6, 13.4, 16.5, 16.9**
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import comparison_report as cr
from fraud_detect.ml.retraining.switch_rule import SwitchRule

RULE = SwitchRule()
SOURCES = ["generator_corpus", "forgery_dataset", "awkward_set", "real_style"]
MODELS = ["Candidate", "Current", "Candidate_all11"]
NAME = st.text(alphabet="abcdefghij0123456789_", min_size=1, max_size=8)
SPLITS = {"train": 30, "validation": 10, "test": 20}
LAYOUTS = {"layout_a": 40, "layout_b": 20}

_prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def _metric(draw):
    computable = draw(st.booleans())
    n_pos, n_neg = draw(st.integers(0, 99)), draw(st.integers(0, 99))
    common = dict(name=draw(NAME), n_pos=n_pos, n_neg=n_neg, computable=computable,
                  indicative_only=draw(st.booleans()))
    if computable:
        return cr.MetricStat(mean=draw(_prob), minimum=draw(_prob), maximum=draw(_prob),
                             std=draw(_prob), **common)
    return cr.MetricStat(mean=None, minimum=None, maximum=None, std=None, **common)


@st.composite
def _row(draw):
    return cr.ResultRow(
        node=draw(st.sampled_from(["6a", "6b"])), source=draw(st.sampled_from(SOURCES)),
        split=draw(NAME), model=draw(st.sampled_from(MODELS)),
        metrics=tuple(draw(st.lists(_metric(), min_size=1, max_size=3))),
        layout=draw(st.one_of(st.none(), NAME)), file=draw(st.one_of(st.none(), NAME)),
        held_out=draw(st.booleans()))


@st.composite
def _audit(draw):
    return cr.AuditRow(feature="FEAT_" + draw(NAME), decision=draw(st.sampled_from(["keep", "drop"])),
                       description=draw(NAME), leak_auc=draw(st.one_of(st.none(), _prob)))


@st.composite
def _report(draw):
    return cr.ComparisonReport(
        header=cr.ReportHeader(
            switch_rule_text=RULE.text(), switch_rule_sha256=RULE.sha256(),
            cutoff_rule={"medium": 5.0, "high": 1.0}, awkward_seed=1, awkward_version="v1",
            real_style_role=draw(st.sampled_from(["required", "veto_only"]))),
        corpus_seeds=(1,), held_out_layout="layout_b", split_counts=SPLITS,
        layout_counts=LAYOUTS, total_documents=60,
        seeds_attempted={s: "completed" for s in range(draw(st.integers(0, 6)))},
        skipped={}, real_style_files=(), decisions=(), post_test_changes=(),
        code_unchanged=True, gnn_coverage_pct=0.0,
        audit_rows=tuple(draw(st.lists(_audit(), max_size=4))),
        rows=tuple(draw(st.lists(_row(), max_size=6))))


@settings(max_examples=60, deadline=None)
@given(_report())
def test_property_38_banner_and_markers(report):
    outputs = [(cr.render_markdown(report), "## Results"),
               (cr.render_console(report), "== Results ==")]
    for text, results_marker in outputs:
        lines = text.splitlines()
        first = "\n".join(lines[:10]).lower()
        assert "synthetic data" in first
        assert "judged on the generator's output" in first
        # banner precedes every table line and the results section
        first_table = next((i for i, ln in enumerate(lines) if ln.startswith("|")), len(lines))
        banner_at = max(i for i, ln in enumerate(lines[:10])
                        if "synthetic data" in ln.lower() or "judged on the generator's output" in ln.lower())
        assert banner_at < first_table
        assert banner_at < lines.index(results_marker)

        # every (row, metric) has one line with the right markers
        result_lines = [ln for ln in lines if ln.startswith(("| 6a |", "| 6b |"))]
        expected = [r for r in report.rows for _ in r.metrics]
        assert len(result_lines) == len(expected)
        for ln, row in zip(result_lines, expected):
            assert (cr.ROW_MARKER in ln) == (row.source != "real_style"), ln
            assert (cr.FORGERY_NOTE in ln) == (row.source == "forgery_dataset"), ln

        # each audit row carries its leak result in the same row
        audit_lines = [ln for ln in lines if ln.startswith("| FEAT_")]
        assert len(audit_lines) == len(report.audit_rows)
        for ln, a in zip(audit_lines, report.audit_rows):
            assert a.feature in ln and a.decision in ln
            want = "not computable" if a.leak_auc is None else f"{a.leak_auc:.4f}"
            assert want in ln
