"""Task 21.1: failing tests for fraud_detect.ml.retraining.switch_rule (written before the module).

Pinned interface (see also the task 21.1 summary):

    SwitchRule(version="switch-v1", decisive=("heldout_layout","heldout_seed","forgery_ood","real_style"),
               real_style_role="required"|"veto_only", hard_negative_block_points=10.0)
        .text() -> str, .sha256() -> str (SHA-256 hex of text().encode("utf-8"))
    SourceMetrics(missing=False, indicative_only=False,
                  cand_auc, cur_auc, cand_auc_std, cand_catch_medium, cur_catch_medium,
                  cand_fa_medium, cur_fa_medium,                     # Node 6b
                  cand_pr_auc, cur_pr_auc, cand_fa_stmt, cur_fa_stmt)  # Node 6a  (all default None)
    Evidence(baseline_state="ok"|"no baseline"|"baseline failed", metrics={source_id: SourceMetrics},
             awkward_reviewed=False, hn_fa_cand=0.0, hn_fa_cur=0.0, release_passes_same_checks=True)
    decide(rule, node in {"6a","6b"}, evidence) -> NodeDecision(action "switch"|"keep", reason, status,
             recommendation, report_text() ) ; status maps decisive source -> "pass"|"fail"|"not decisive"
Rates are in percent (0..100); hard-negative gap is in percentage points.
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from fraud_detect.ml.retraining.switch_rule import (
    Evidence, SourceMetrics, SwitchRule, decide)

DECISIVE = ("heldout_layout", "heldout_seed", "forgery_ood", "real_style")
KEEP_6A_TEXT = "keep the GBM and the GNN lookup off in the live app"


def m6b(**over) -> SourceMetrics:
    base = dict(cand_auc=0.90, cur_auc=0.90, cand_auc_std=0.02,
                cand_catch_medium=60.0, cur_catch_medium=60.0,
                cand_fa_medium=5.0, cur_fa_medium=5.0)
    base.update(over)
    return SourceMetrics(**base)


def m6a(**over) -> SourceMetrics:
    base = dict(cand_pr_auc=0.50, cur_pr_auc=0.50, cand_fa_stmt=2.0, cur_fa_stmt=2.0)
    base.update(over)
    return SourceMetrics(**base)


def ev(node="6b", overrides=None, **kw) -> Evidence:
    make = m6b if node == "6b" else m6a
    metrics = {d: make() for d in DECISIVE}
    metrics.update(overrides or {})
    return Evidence(metrics=metrics, **kw)


# ---------- SwitchRule ----------

def test_defaults():
    r = SwitchRule()
    assert r.version == "switch-v1"
    assert tuple(r.decisive) == DECISIVE
    assert r.real_style_role == "required"
    assert r.hard_negative_block_points == 10.0


def test_rule_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SwitchRule().version = "x"


def test_text_and_hash_are_stable_and_match():
    a, b = SwitchRule(), SwitchRule()
    assert a.text() == a.text() == b.text()
    assert a.sha256() == b.sha256() == hashlib.sha256(a.text().encode("utf-8")).hexdigest()
    assert "switch-v1" in a.text()


def test_hash_changes_with_role_and_threshold():
    base = SwitchRule()
    assert SwitchRule(real_style_role="veto_only").sha256() != base.sha256()
    assert SwitchRule(hard_negative_block_points=5.0).sha256() != base.sha256()
    assert "veto_only" in SwitchRule(real_style_role="veto_only").text()


# ---------- Node 6b "at least as good" ----------

def test_6b_all_sources_pass_and_equal_values_pass_with_veto_only():
    d = decide(SwitchRule(real_style_role="veto_only"), "6b",
               ev("6b", {"real_style": m6b(indicative_only=True)}))
    assert d.action == "switch"
    assert d.status["heldout_layout"] == "pass"
    assert d.status["real_style"] == "not decisive"


def test_6b_auc_tolerance_is_candidate_std_and_exact():
    rule = SwitchRule(real_style_role="veto_only")
    # exactly at the boundary passes (0.5 - 0.25 is exact in binary)
    ok = decide(rule, "6b", ev("6b", {"heldout_layout": m6b(cand_auc=0.25, cur_auc=0.5, cand_auc_std=0.25)}))
    assert ok.status["heldout_layout"] == "pass"
    # a hair below fails: no rounding before comparison
    bad = decide(rule, "6b", ev("6b", {"heldout_layout": m6b(cand_auc=0.9 - 0.02 - 1e-12, cur_auc=0.9, cand_auc_std=0.02)}))
    assert bad.status["heldout_layout"] == "fail"
    assert bad.action == "keep"


@pytest.mark.parametrize("over", [
    dict(cand_catch_medium=59.999999),
    dict(cand_fa_medium=5.000001),
])
def test_6b_catch_and_false_alarm_must_not_be_worse(over):
    d = decide(SwitchRule(real_style_role="veto_only"), "6b", ev("6b", {"heldout_seed": m6b(**over)}))
    assert d.status["heldout_seed"] == "fail"
    assert d.action == "keep"


# ---------- Node 6a ----------

def test_6a_pass_equal_and_fail_cases():
    rule = SwitchRule(real_style_role="veto_only")
    d = decide(rule, "6a", ev("6a", {"real_style": m6a(missing=True)}))
    assert d.action == "switch"
    assert KEEP_6A_TEXT not in d.recommendation
    for over in (dict(cand_pr_auc=0.4999999), dict(cand_fa_stmt=2.0000001)):
        bad = decide(rule, "6a", ev("6a", {"forgery_ood": m6a(**over)}))
        assert bad.status["forgery_ood"] == "fail" and bad.action == "keep"


def test_6a_keep_adds_gbm_gnn_text_and_6b_keep_does_not():
    keep_a = decide(SwitchRule(), "6a", ev("6a", {"real_style": m6a(indicative_only=True)}))
    assert keep_a.action == "keep" and KEEP_6A_TEXT in keep_a.recommendation
    keep_b = decide(SwitchRule(), "6b", ev("6b", {"real_style": m6b(indicative_only=True)}))
    assert keep_b.action == "keep" and KEEP_6A_TEXT not in keep_b.recommendation


# ---------- not decisive / failures ----------

@pytest.mark.parametrize("flag", ["missing", "indicative_only"])
def test_missing_or_indicative_source_is_not_decisive_and_never_passes(flag):
    d = decide(SwitchRule(real_style_role="veto_only"), "6b",
               ev("6b", {"forgery_ood": m6b(**{flag: True})}))
    assert d.status["forgery_ood"] == "not decisive"
    assert d.action == "keep"            # a non-real_style source that is not decisive blocks a switch


def test_source_absent_from_evidence_is_not_decisive():
    e = ev("6b")
    metrics = dict(e.metrics)
    del metrics["heldout_seed"]
    d = decide(SwitchRule(real_style_role="veto_only"), "6b", dataclasses.replace(e, metrics=metrics))
    assert d.status["heldout_seed"] == "not decisive" and d.action == "keep"


def test_any_failed_source_gives_keep_even_when_others_are_much_better():
    d = decide(SwitchRule(real_style_role="veto_only"), "6b",
               ev("6b", {"heldout_layout": m6b(cand_auc=0.99, cand_catch_medium=99.0, cand_fa_medium=1.0),
                         "forgery_ood": m6b(cand_fa_medium=5.5)}))
    assert d.action == "keep" and d.status["forgery_ood"] == "fail"


def test_no_available_decisive_source_gives_keep_with_statement():
    metrics = {d: m6b(missing=True) for d in DECISIVE}
    d = decide(SwitchRule(), "6b", Evidence(metrics=metrics))
    assert d.action == "keep"
    assert "no decisive source" in d.report_text().lower()
    assert "no decisive source" in d.reason.lower()


# ---------- Real_Style role ----------

def test_required_role_makes_two_file_real_style_always_keep_and_report_says_so():
    d = decide(SwitchRule(), "6b", ev("6b", {"real_style": m6b(indicative_only=True)}))
    assert d.action == "keep"
    head = "\n".join(d.report_text().splitlines()[:3]).lower()
    assert "real_style" in head and "not decisive" in head and "keep" in head


@pytest.mark.parametrize("flag", ["missing", "indicative_only"])
def test_required_role_keeps_even_with_everything_else_perfect(flag):
    d = decide(SwitchRule(), "6b", ev("6b", {"real_style": m6b(**{flag: True})}))
    assert d.action == "keep"


def test_veto_only_real_style_can_only_block():
    rule = SwitchRule(real_style_role="veto_only")
    blocked = decide(rule, "6b", ev("6b", {"real_style": m6b(cand_fa_medium=9.0)}))
    assert blocked.action == "keep" and blocked.status["real_style"] == "fail"
    # real_style passing does not stand in for another source that is not decisive
    only_rs = {d: m6b(missing=True) for d in DECISIVE}
    only_rs["real_style"] = m6b()
    assert decide(rule, "6b", Evidence(metrics=only_rs)).action == "keep"


def test_configured_role_is_what_matters_not_global_state():
    e = ev("6b", {"real_style": m6b(indicative_only=True)})
    assert decide(SwitchRule(real_style_role="required"), "6b", e).action == "keep"
    assert decide(SwitchRule(real_style_role="veto_only"), "6b", e).action == "switch"


# ---------- release candidate, baseline ----------

def test_release_candidate_must_pass_same_checks():
    rule = SwitchRule(real_style_role="veto_only")
    d = decide(rule, "6b", ev("6b", {"real_style": m6b(missing=True)}, release_passes_same_checks=False))
    assert d.action == "keep"
    assert "release" in d.reason.lower()


@pytest.mark.parametrize("state", ["no baseline", "baseline failed"])
@pytest.mark.parametrize("node", ["6a", "6b"])
def test_no_baseline_and_baseline_failed_give_keep(state, node):
    d = decide(SwitchRule(real_style_role="veto_only"), node, ev(node, baseline_state=state))
    assert d.action == "keep" and d.reason == state


# ---------- monotonic ----------

def test_improving_candidate_metrics_never_turns_pass_into_fail():
    rule = SwitchRule(real_style_role="veto_only")
    base = {"real_style": m6b(missing=True)}
    before = decide(rule, "6b", ev("6b", base))
    better = {"real_style": m6b(missing=True),
              "heldout_layout": m6b(cand_auc=0.95, cand_catch_medium=70.0, cand_fa_medium=3.0, cand_auc_std=0.0),
              "forgery_ood": m6b(cand_auc=0.91, cand_catch_medium=61.0, cand_fa_medium=4.0)}
    after = decide(rule, "6b", ev("6b", better))
    assert before.action == "switch" and after.action == "switch"
    assert all(after.status[k] == "pass" for k in ("heldout_layout", "forgery_ood"))


# ---------- Awkward_Set ----------

def _passing(**kw):
    return ev("6b", {"real_style": m6b(missing=True)}, **kw)


def test_awkward_set_cannot_pass_a_node():
    metrics = {d: m6b(missing=True) for d in DECISIVE}
    metrics["awkward_set"] = m6b()            # perfect Awkward_Set numbers must be ignored
    d = decide(SwitchRule(real_style_role="veto_only"), "6b", Evidence(metrics=metrics))
    assert d.action == "keep"
    assert "awkward_set" not in d.status


def test_awkward_blocks_only_when_reviewed_and_gap_exceeds_ten_points():
    rule = SwitchRule(real_style_role="veto_only")
    blocked = decide(rule, "6b", _passing(awkward_reviewed=True, hn_fa_cand=30.1, hn_fa_cur=20.0))
    assert blocked.action == "keep" and "awkward" in blocked.reason.lower()
    assert "awkward" in blocked.report_text().lower()
    at_ten = decide(rule, "6b", _passing(awkward_reviewed=True, hn_fa_cand=30.0, hn_fa_cur=20.0))
    assert at_ten.action == "switch"
    unreviewed = decide(rule, "6b", _passing(awkward_reviewed=False, hn_fa_cand=90.0, hn_fa_cur=0.0))
    assert unreviewed.action == "switch"


def test_awkward_threshold_follows_rule_field():
    rule = SwitchRule(real_style_role="veto_only", hard_negative_block_points=2.0)
    d = decide(rule, "6b", _passing(awkward_reviewed=True, hn_fa_cand=22.5, hn_fa_cur=20.0))
    assert d.action == "keep"


# ---------- no automatic model copy ----------

def test_decide_is_pure_and_touches_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e = _passing()
    first = decide(SwitchRule(real_style_role="veto_only"), "6b", e)
    second = decide(SwitchRule(real_style_role="veto_only"), "6b", e)
    assert first == second
    assert list(tmp_path.iterdir()) == []


def test_switch_rule_module_has_no_copy_or_model_write_code():
    import inspect
    from fraud_detect.ml.retraining import switch_rule
    src = inspect.getsource(switch_rule)
    for banned in ("shutil", "copy2", "copyfile", "os.replace", "joblib", "torch", "sklearn"):
        assert banned not in src
