"""Task 21.4: property tests for the Switch_Rule (design Properties 34 and 35).

**Validates: Requirements 8.9, 14.2, 14.3, 14.4, 14.5, 14.6, 14.13, 14.14**
"""
from __future__ import annotations

import dataclasses

from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining.switch_rule import (
    Evidence, SourceMetrics, SwitchRule, decide)

DECISIVE = ("heldout_layout", "heldout_seed", "forgery_ood", "real_style")

pct = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
small = st.floats(min_value=0.0, max_value=0.2, allow_nan=False)
flag = st.booleans()


@st.composite
def metrics_6b(draw):
    return SourceMetrics(
        missing=draw(st.booleans()) if draw(st.integers(0, 4)) == 0 else False,
        indicative_only=draw(st.integers(0, 5)) == 0,
        cand_auc=draw(unit), cur_auc=draw(unit), cand_auc_std=draw(small),
        cand_catch_medium=draw(pct), cur_catch_medium=draw(pct),
        cand_fa_medium=draw(pct), cur_fa_medium=draw(pct))


@st.composite
def metrics_6a(draw):
    return SourceMetrics(
        missing=draw(st.integers(0, 5)) == 0,
        indicative_only=draw(st.integers(0, 5)) == 0,
        cand_pr_auc=draw(unit), cur_pr_auc=draw(unit),
        cand_fa_stmt=draw(pct), cur_fa_stmt=draw(pct))


def _passes(node, m):
    """Independent oracle for one source: 'pass' | 'fail' | 'not decisive'."""
    if m.missing or m.indicative_only:
        return "not decisive"
    if node == "6b":
        ok = (m.cand_auc >= m.cur_auc - m.cand_auc_std
              and m.cand_catch_medium >= m.cur_catch_medium
              and m.cand_fa_medium <= m.cur_fa_medium)
    else:
        ok = m.cand_pr_auc >= m.cur_pr_auc and m.cand_fa_stmt <= m.cur_fa_stmt
    return "pass" if ok else "fail"


def _expected_switch(rule, node, ev):
    st_ = {s: _passes(node, ev.metrics[s]) for s in rule.decisive}
    if ev.baseline_state != "ok":
        return False
    for s, v in st_.items():
        if v == "fail":
            return False
        if v == "not decisive" and not (s == "real_style" and rule.real_style_role == "veto_only"):
            return False
    if not any(v != "not decisive" for v in st_.values()):
        return False
    if ev.awkward_reviewed and ev.hn_fa_cand - ev.hn_fa_cur > rule.hard_negative_block_points:
        return False
    return ev.release_passes_same_checks


@st.composite
def scenarios(draw, node=None):
    node = node or draw(st.sampled_from(["6a", "6b"]))
    gen = metrics_6b() if node == "6b" else metrics_6a()
    metrics = {s: draw(gen) for s in DECISIVE}
    rule = SwitchRule(real_style_role=draw(st.sampled_from(["required", "veto_only"])))
    ev = Evidence(
        baseline_state=draw(st.sampled_from(["ok", "ok", "ok", "no baseline", "baseline failed"])),
        metrics=metrics,
        awkward_reviewed=draw(flag),
        hn_fa_cand=draw(pct), hn_fa_cur=draw(pct),
        release_passes_same_checks=draw(flag))
    return rule, node, ev


# ---------------- Property 34 ----------------

@settings(max_examples=300, deadline=None)
@given(scenarios())
def test_p34_switch_iff_oracle(sc):
    rule, node, ev = sc
    d = decide(rule, node, ev)
    assert d.action in ("switch", "keep")
    assert (d.action == "switch") == _expected_switch(rule, node, ev)


@settings(max_examples=200, deadline=None)
@given(scenarios())
def test_p34_status_per_source(sc):
    rule, node, ev = sc
    d = decide(rule, node, ev)
    for s in rule.decisive:
        assert d.status[s] == _passes(node, ev.metrics[s])
    if any(v == "fail" for v in d.status.values()):
        assert d.action == "keep"
    if all(v == "not decisive" for v in d.status.values()):
        assert d.action == "keep"
        assert "No decisive source is available" in d.report_text() or "no decisive" in d.reason


@settings(max_examples=100, deadline=None)
@given(st.sampled_from(["6a", "6b"]), flag)
def test_p34_equal_values_pass(node, veto):
    if node == "6b":
        m = SourceMetrics(cand_auc=0.7, cur_auc=0.7, cand_auc_std=0.0,
                          cand_catch_medium=50.0, cur_catch_medium=50.0,
                          cand_fa_medium=4.0, cur_fa_medium=4.0)
    else:
        m = SourceMetrics(cand_pr_auc=0.3, cur_pr_auc=0.3, cand_fa_stmt=1.0, cur_fa_stmt=1.0)
    rule = SwitchRule(real_style_role="veto_only" if veto else "required")
    ev = Evidence(metrics={s: m for s in DECISIVE})
    assert decide(rule, node, ev).action == "switch"


@settings(max_examples=200, deadline=None)
@given(scenarios(node="6b"), pct, small)
def test_p34_improving_candidate_6b_never_hurts(sc, delta, auc_delta):
    rule, node, ev = sc
    before = decide(rule, node, ev)
    better = {}
    for s, m in ev.metrics.items():
        better[s] = dataclasses.replace(
            m,
            cand_auc=m.cand_auc + auc_delta,
            cand_catch_medium=m.cand_catch_medium + delta,
            cand_fa_medium=max(0.0, m.cand_fa_medium - delta))
    after = decide(rule, node, dataclasses.replace(ev, metrics=better))
    for s in rule.decisive:
        if before.status[s] == "pass":
            assert after.status[s] == "pass"
    if before.action == "switch":
        assert after.action == "switch"


@settings(max_examples=200, deadline=None)
@given(scenarios(node="6a"), pct, unit)
def test_p34_improving_candidate_6a_never_hurts(sc, delta, pr_delta):
    rule, node, ev = sc
    before = decide(rule, node, ev)
    better = {s: dataclasses.replace(m, cand_pr_auc=m.cand_pr_auc + pr_delta,
                                     cand_fa_stmt=max(0.0, m.cand_fa_stmt - delta))
              for s, m in ev.metrics.items()}
    after = decide(rule, node, dataclasses.replace(ev, metrics=better))
    for s in rule.decisive:
        if before.status[s] == "pass":
            assert after.status[s] == "pass"
    if before.action == "switch":
        assert after.action == "switch"


@settings(max_examples=100, deadline=None)
@given(scenarios())
def test_p34_rule_hash_is_stable(sc):
    rule, _, _ = sc
    assert rule.sha256() == SwitchRule(real_style_role=rule.real_style_role).sha256()


# ---------------- Property 35 ----------------

@settings(max_examples=300, deadline=None)
@given(scenarios(), flag, pct, pct)
def test_p35_awkward_never_turns_keep_into_switch(sc, reviewed, cand, cur):
    rule, node, ev = sc
    base = decide(rule, node, dataclasses.replace(ev, awkward_reviewed=False))
    changed = decide(rule, node, dataclasses.replace(
        ev, awkward_reviewed=reviewed, hn_fa_cand=cand, hn_fa_cur=cur))
    if base.action == "keep":
        assert changed.action == "keep"


@settings(max_examples=300, deadline=None)
@given(scenarios(), pct, pct)
def test_p35_unreviewed_has_no_effect(sc, cand, cur):
    rule, node, ev = sc
    a = decide(rule, node, dataclasses.replace(ev, awkward_reviewed=False, hn_fa_cand=0.0, hn_fa_cur=0.0))
    b = decide(rule, node, dataclasses.replace(ev, awkward_reviewed=False, hn_fa_cand=cand, hn_fa_cur=cur))
    assert a.action == b.action
    assert a.reason == b.reason
    assert "not reviewed" in b.report_text()


@settings(max_examples=300, deadline=None)
@given(st.sampled_from(["6a", "6b"]), flag, st.floats(0.0, 100.0), st.floats(0.0, 100.0))
def test_p35_reviewed_blocks_when_gap_exceeds_ten_points(node, veto, cur, gap_extra):
    # All sources pass, so only the Awkward_Set can change the result.
    if node == "6b":
        m = SourceMetrics(cand_auc=0.9, cur_auc=0.8, cand_auc_std=0.01,
                          cand_catch_medium=70.0, cur_catch_medium=60.0,
                          cand_fa_medium=3.0, cur_fa_medium=5.0)
    else:
        m = SourceMetrics(cand_pr_auc=0.6, cur_pr_auc=0.5, cand_fa_stmt=1.0, cur_fa_stmt=2.0)
    rule = SwitchRule(real_style_role="veto_only" if veto else "required")
    ev = Evidence(metrics={s: m for s in DECISIVE}, awkward_reviewed=True,
                  hn_fa_cur=cur, hn_fa_cand=cur + 10.5 + gap_extra)
    d = decide(rule, node, ev)
    assert d.action == "keep"
    assert "awkward" in d.reason.lower()

    # Exactly 10 points (exact in binary floating point): no block.
    ok = dataclasses.replace(ev, hn_fa_cur=0.0, hn_fa_cand=10.0)
    assert decide(rule, node, ok).action == "switch"
    # Within 10 points: no block.
    ok2 = dataclasses.replace(ev, hn_fa_cand=cur + 9.0)
    assert decide(rule, node, ok2).action == "switch"
