"""Task 20.1: failing tests for the implausible-measurement monitor and the stop lock.

``fraud_detect/ml/retraining/monitor.py`` does not exist yet (task 20.2). These tests pin its interface:

    Measurement(name, value, source, split, master_seed, n_pos, n_neg, *, model="candidate",
                roc_auc=None, flag=None, label=None, doc_count=None, manifest_count=None)  # frozen dataclass
    fired_rules(m, rule_c_scope="all") -> list[str]       # subset of ["a", "b", "c", "d"], in that order
    RULE_C_SCOPES = ("all", "false_alarms_only")          # decision 4: "false_alarms_only" = flag 1, label 0
    Monitor(run_dir, is_ignored=None, clock=None, rule_c_scope="all")
        .observe(m) -> None                               # raises MeasurementStop on a firing
        .is_stopped -> bool                               # true while run_dir/STOP exists (survives a new instance)
        .guard(change) -> None                            # raises StopLockActive naming `change` while stopped
        .decide(decision, reason, author) -> dict         # "continue" | "abort"; DecisionLog entry, kind "investigation"
    MeasurementStop(Exception): .exit_code == 4, .record (dict, same fields as investigation.json)
    StopLockActive(Exception): raised by guard() and by observe() while stopped
    InvestigationError(Exception): decide() with no open investigation
    BLOCKED_CHANGES = ("train", "compare", "audit_new_version", "cutoff_change", "feature_change",
                       "generator_change", "detector_change")
    SeedLedger(run_dir, is_ignored=None, clock=None)
        .attempt(seed) / .record(seed, outcome, reason="") / .entries() / .substitute(failed_seed, new_seed)
        outcomes: "completed" | "failed" | "stopped" | "discarded"; substitute raises SeedSubstitutionRefused
    record_post_test_change(run_dir, *, kind, description, reason, trigger_measurement, trigger_value,
                            is_ignored=None, clock=None) -> dict      # appends to post_test_changes.json

Files in run_dir: ``investigation.json`` (rule, measurement, value, source, split, master_seed, n_pos,
n_neg, plus id and date), ``STOP``, ``seeds.json`` ({"seeds": [...]}), ``decisions.json``.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from fraud_detect.ml.retraining import monitor as monitor_mod
from fraud_detect.ml.retraining.cli import EXIT_IMPLAUSIBLE
from fraud_detect.ml.retraining.monitor import (
    BLOCKED_CHANGES,
    RULE_C_SCOPES,
    InvestigationError,
    Measurement,
    MeasurementStop,
    Monitor,
    SeedLedger,
    SeedSubstitutionRefused,
    StopLockActive,
    fired_rules,
    record_post_test_change,
)

UNSEEN = "test_unseen_layout"


def M(**over):
    base = dict(name="roc_auc", value=0.80, source="corpus", split="test", master_seed=20260801,
                n_pos=100, n_neg=100, model="candidate", roc_auc=0.80)
    base.update(over)
    return Measurement(**base)


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "runs" / "r1"


@pytest.fixture
def mon(run_dir, is_ignored, clock):
    return Monitor(run_dir, is_ignored=is_ignored, clock=clock)


# ---- rule (a) -------------------------------------------------------------------------------

@pytest.mark.parametrize("auc,model,split,expected", [
    (0.99, "candidate", UNSEEN, True),
    (1.0, "candidate", UNSEEN, True),
    (0.9899, "candidate", UNSEEN, False),
    (0.999, "candidate", "test", False),
    (0.999, "candidate", "validation", False),
    (0.999, "current", UNSEEN, False),
])
def test_rule_a(auc, model, split, expected):
    assert ("a" in fired_rules(M(roc_auc=auc, value=auc, model=model, split=split))) is expected


# ---- rule (b) -------------------------------------------------------------------------------

@pytest.mark.parametrize("auc,n_pos,n_neg,expected", [
    (0.49, 30, 30, True),
    (0.0, 500, 31, True),
    (0.5, 100, 100, False),
    (0.49, 29, 100, False),
    (0.49, 100, 29, False),
    (None, 100, 100, False),
])
def test_rule_b(auc, n_pos, n_neg, expected):
    m = M(roc_auc=auc, value=auc if auc is not None else 0.0, n_pos=n_pos, n_neg=n_neg)
    assert ("b" in fired_rules(m)) is expected


def test_rule_b_applies_to_any_source_and_model():
    m = M(roc_auc=0.3, source="forgery", model="current", split="all")
    assert fired_rules(m) == ["b"]


# ---- rule (c) -------------------------------------------------------------------------------

def _rs(flag, label, **over):
    base = dict(name="flag", value=float(flag), source="real_style", split="real_style",
                n_pos=1, n_neg=1, flag=flag, label=label, roc_auc=None)
    base.update(over)
    return M(**base)


@pytest.mark.parametrize("flag,label,expected", [(0, 1, True), (1, 0, True), (1, 1, False), (0, 0, False)])
def test_rule_c_opposite_side_of_cutoff(flag, label, expected):
    assert ("c" in fired_rules(_rs(flag, label))) is expected


def test_rule_c_only_for_real_style_candidate():
    assert "c" not in fired_rules(_rs(0, 1, source="corpus"))
    assert "c" not in fired_rules(_rs(0, 1, model="current"))


def test_rule_c_alternative_scope_false_alarms_only():
    assert "false_alarms_only" in RULE_C_SCOPES and "all" in RULE_C_SCOPES
    assert fired_rules(_rs(0, 1), rule_c_scope="false_alarms_only") == []   # a miss no longer fires
    assert fired_rules(_rs(1, 0), rule_c_scope="false_alarms_only") == ["c"]
    assert fired_rules(_rs(1, 1), rule_c_scope="false_alarms_only") == []


# ---- rule (d) -------------------------------------------------------------------------------

@pytest.mark.parametrize("count,manifest,expected", [
    (120, 120, False), (119, 120, True), (121, 120, True), (0, 120, True), (120, None, False)])
def test_rule_d_count_differs_by_one_or_more(count, manifest, expected):
    m = M(doc_count=count, manifest_count=manifest)
    assert ("d" in fired_rules(m)) is expected


def test_plausible_measurement_fires_nothing():
    assert fired_rules(M()) == []


def test_several_rules_listed_in_order():
    m = M(roc_auc=0.995, value=0.995, split=UNSEEN, doc_count=5, manifest_count=6)
    assert fired_rules(m) == ["a", "d"]


def test_measurement_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        M().roc_auc = 0.1


# ---- firing: record, lock, print, exit code -------------------------------------------------

def _fire(mon, **over):
    m = M(roc_auc=0.995, value=0.995, split=UNSEEN, source="corpus", master_seed=20260803,
          n_pos=40, n_neg=60, **over)
    with pytest.raises(MeasurementStop) as info:
        mon.observe(m)
    return m, info.value


def test_observe_accepts_a_plausible_measurement(mon, run_dir):
    mon.observe(M())
    assert not mon.is_stopped
    assert not (run_dir / "STOP").exists() and not (run_dir / "investigation.json").exists()


def test_firing_writes_investigation_json_and_stop(mon, run_dir):
    _, stop = _fire(mon)
    data = json.loads((run_dir / "investigation.json").read_text())
    assert data["rule"] == "a"
    assert data["measurement"] == "roc_auc" and data["value"] == 0.995
    assert data["source"] == "corpus" and data["split"] == UNSEEN
    assert data["master_seed"] == 20260803 and data["n_pos"] == 40 and data["n_neg"] == 60
    assert (run_dir / "STOP").exists() and mon.is_stopped
    assert stop.record["rule"] == "a" and stop.record["master_seed"] == 20260803


def test_firing_exits_with_code_4(mon):
    _, stop = _fire(mon)
    assert stop.exit_code == 4 == EXIT_IMPLAUSIBLE


def test_firing_prints_the_same_fields(mon, capsys):
    _fire(mon)
    out = capsys.readouterr().out
    for text in ("roc_auc", "0.995", "corpus", UNSEEN, "20260803", "40", "60"):
        assert text in out


def test_results_already_written_stay_unchanged(mon, run_dir):
    results = run_dir / "results" / "6b" / "test" / "20260801.json"
    results.parent.mkdir(parents=True)
    results.write_text('{"roc_auc": 0.8}')
    before = results.read_bytes()
    _fire(mon)
    assert results.read_bytes() == before
    assert sorted(p.name for p in results.parent.iterdir()) == ["20260801.json"]


def test_stop_survives_a_new_monitor_instance(mon, run_dir, is_ignored, clock):
    _fire(mon)
    assert Monitor(run_dir, is_ignored=is_ignored, clock=clock).is_stopped


def test_firing_goes_through_the_write_guard(run_dir, is_ignored, clock):
    is_ignored.ignored = False
    m = Monitor(run_dir, is_ignored=is_ignored, clock=clock)
    with pytest.raises(Exception):
        m.observe(M(roc_auc=0.995, value=0.995, split=UNSEEN))
    assert not (run_dir / "STOP").exists() and not (run_dir / "investigation.json").exists()


def test_no_further_measurement_accepted_while_stopped(mon):
    _fire(mon)
    with pytest.raises(StopLockActive):
        mon.observe(M())


# ---- blocked changes while stopped ----------------------------------------------------------

def test_guard_allows_every_change_when_not_stopped(mon):
    for change in BLOCKED_CHANGES:
        mon.guard(change)


def test_blocked_changes_cover_the_requirement():
    assert set(BLOCKED_CHANGES) >= {"train", "compare", "audit_new_version", "cutoff_change",
                                    "feature_change", "generator_change", "detector_change"}


@pytest.mark.parametrize("change", ["train", "compare", "audit_new_version", "cutoff_change",
                                    "feature_change", "generator_change", "detector_change"])
def test_guard_refuses_and_names_the_change_while_stopped(mon, change):
    _fire(mon)
    with pytest.raises(StopLockActive) as info:
        mon.guard(change)
    assert change in str(info.value)


def test_guard_rejects_unknown_change_name(mon):
    with pytest.raises(ValueError):
        mon.guard("not-a-change")


# ---- the decision ---------------------------------------------------------------------------

def test_decide_without_an_investigation_is_refused(mon):
    with pytest.raises(InvestigationError):
        mon.decide("continue", "nothing to continue", "alice")


def test_decide_rejects_unknown_decision(mon):
    _fire(mon)
    with pytest.raises(Exception):
        mon.decide("ignore", "x", "alice")
    assert mon.is_stopped


def test_continue_records_decision_removes_stop_and_run_resumes(mon, run_dir):
    _fire(mon)
    entry = mon.decide("continue", "checked, the layout is easy", "alice")
    assert entry["kind"] == "investigation" and entry["decision"] == "continue"
    assert entry["reason"] == "checked, the layout is easy" and entry["author"] == "alice"
    stored = json.loads((run_dir / "decisions.json").read_text())["decisions"]
    assert [e["id"] for e in stored] == [entry["id"]]
    assert not (run_dir / "STOP").exists() and not mon.is_stopped
    assert (run_dir / "investigation.json").exists()      # the record stays for the report
    mon.observe(M())                                      # accepted again
    mon.guard("train")                                    # no longer blocked


def test_abort_records_decision_and_keeps_the_lock(mon, run_dir):
    _fire(mon)
    entry = mon.decide("abort", "stop the study", "alice")
    assert entry["decision"] == "abort"
    assert (run_dir / "STOP").exists() and mon.is_stopped
    with pytest.raises(StopLockActive):
        mon.guard("compare")


def test_a_second_firing_after_continue_opens_a_new_investigation(mon, run_dir):
    _fire(mon)
    mon.decide("continue", "ok", "alice")
    with pytest.raises(MeasurementStop):
        mon.observe(M(roc_auc=0.2, value=0.2, n_pos=50, n_neg=50, master_seed=20260804))
    assert (run_dir / "STOP").exists()
    assert json.loads((run_dir / "investigation.json").read_text())["rule"] == "b"


def test_rule_c_scope_is_configurable_on_the_monitor(run_dir, is_ignored, clock):
    miss = _rs(0, 1, master_seed=20260801)
    Monitor(run_dir / "x", is_ignored=is_ignored, clock=clock, rule_c_scope="false_alarms_only").observe(miss)
    with pytest.raises(MeasurementStop):
        Monitor(run_dir / "y", is_ignored=is_ignored, clock=clock).observe(miss)


# ---- seed ledger ----------------------------------------------------------------------------

@pytest.fixture
def ledger(run_dir, is_ignored, clock):
    return SeedLedger(run_dir, is_ignored=is_ignored, clock=clock)


def test_ledger_lists_every_seed_attempted_with_outcome(ledger, run_dir):
    for seed in (20260801, 20260802, 20260803, 20260804):
        ledger.attempt(seed)
    ledger.record(20260801, "completed")
    ledger.record(20260802, "failed", "corpus build crashed")
    ledger.record(20260803, "stopped", "rule a")
    ledger.record(20260804, "discarded", "duplicate")
    entries = {e["seed"]: e for e in ledger.entries()}
    assert set(entries) == {20260801, 20260802, 20260803, 20260804}
    assert [entries[s]["outcome"] for s in sorted(entries)] == ["completed", "failed", "stopped", "discarded"]
    assert entries[20260802]["reason"] == "corpus build crashed"
    stored = json.loads((run_dir / "seeds.json").read_text())["seeds"]
    assert {e["seed"] for e in stored} == set(entries)


def test_ledger_rejects_unknown_outcome_and_unattempted_seed(ledger):
    ledger.attempt(1)
    with pytest.raises(ValueError):
        ledger.record(1, "great")
    with pytest.raises(ValueError):
        ledger.record(2, "completed")


def test_ledger_persists_across_instances(ledger, run_dir, is_ignored, clock):
    ledger.attempt(7)
    ledger.record(7, "failed", "boom")
    again = SeedLedger(run_dir, is_ignored=is_ignored, clock=clock)
    assert [(e["seed"], e["outcome"]) for e in again.entries()] == [(7, "failed")]


@pytest.mark.parametrize("outcome", ["failed", "stopped"])
def test_substitute_seed_for_a_failed_or_stopped_one_is_refused(ledger, outcome):
    ledger.attempt(20260802)
    ledger.record(20260802, outcome)
    with pytest.raises(SeedSubstitutionRefused):
        ledger.substitute(20260802, 20260810)
    assert 20260810 not in {e["seed"] for e in ledger.entries()}


def test_ledger_write_goes_through_the_write_guard(run_dir, is_ignored, clock):
    is_ignored.ignored = False
    with pytest.raises(Exception):
        SeedLedger(run_dir, is_ignored=is_ignored, clock=clock).attempt(1)
    assert not (run_dir / "seeds.json").exists()


# ---- post-test change record ----------------------------------------------------------------

def test_post_test_change_records_date_trigger_reason_and_description(run_dir, is_ignored, clock):
    entry = record_post_test_change(
        run_dir, kind="generator", description="add Creator randomisation",
        reason="owner decided after investigation", trigger_measurement="roc_auc (test_unseen_layout, seed 20260803)",
        trigger_value=0.995, is_ignored=is_ignored, clock=clock)
    for key in ("date", "trigger_measurement", "trigger_value", "reason", "description"):
        assert key in entry
    assert entry["trigger_value"] == 0.995 and entry["description"] == "add Creator randomisation"
    stored = json.loads((run_dir / "post_test_changes.json").read_text())["changes"]
    assert stored == [entry] or [e["description"] for e in stored] == ["add Creator randomisation"]


def test_post_test_changes_are_appended_not_replaced(run_dir, is_ignored, clock):
    for desc in ("first", "second"):
        record_post_test_change(run_dir, kind="feature", description=desc, reason="r",
                                trigger_measurement="m", trigger_value=0.1,
                                is_ignored=is_ignored, clock=clock)
    stored = json.loads((run_dir / "post_test_changes.json").read_text())["changes"]
    assert [e["description"] for e in stored] == ["first", "second"]


def test_post_test_change_write_goes_through_the_write_guard(run_dir, is_ignored, clock):
    is_ignored.ignored = False
    with pytest.raises(Exception):
        record_post_test_change(run_dir, kind="feature", description="d", reason="r",
                                trigger_measurement="m", trigger_value=1, is_ignored=is_ignored, clock=clock)
    assert not (run_dir / "post_test_changes.json").exists()


def test_module_imports_no_heavy_dependency():
    import sys
    src = open(monitor_mod.__file__, encoding="utf-8").read()
    assert "torch" not in src and "sklearn" not in src
    assert "fraud_detect.ml.retraining.monitor" in sys.modules
