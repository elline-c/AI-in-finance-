"""Task 1.2: failing tests for gates.py and decisions.py (written before the modules)."""
from __future__ import annotations

import dataclasses
import json

import pytest

from fraud_detect.ml.retraining.decisions import (
    DecisionError, DecisionLog, PostTestChangeLog)
from fraud_detect.ml.retraining.gates import Gate, GateResult, run_gates


def _gate(gate_id, offenders, waivable=False, requirement="1.2"):
    return Gate(gate_id=gate_id, requirement=requirement, waivable=waivable,
                check=lambda ctx, _o=tuple(offenders): list(_o))


@pytest.fixture
def log(tmp_path, is_ignored, clock):
    return DecisionLog(tmp_path / "decisions.json", is_ignored=is_ignored, clock=clock)


def test_gate_result_fields():
    names = [f.name for f in dataclasses.fields(GateResult)]
    assert names == ["gate_id", "requirement", "passed", "offenders", "waivable", "waived_by"]
    r = GateResult("g", "1.2", True, (), False, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.passed = False  # type: ignore[misc]


def test_run_gates_returns_pass_and_offenders(log):
    results = run_gates([_gate("ok", []), _gate("bad", ["C1", "C2"])], ctx=None, decisions=log)
    assert [r.gate_id for r in results] == ["ok", "bad"]
    assert results[0].passed and results[0].offenders == ()
    assert not results[1].passed and results[1].offenders == ("C1", "C2")
    assert results[1].requirement == "1.2" and results[1].waived_by is None


def test_ctx_is_passed_to_check(log):
    seen = []
    gate = Gate("g", "1.2", False, lambda ctx: seen.append(ctx) or [])
    run_gates([gate], ctx={"k": 1}, decisions=log)
    assert seen == [{"k": 1}]


def test_integrity_gate_cannot_be_waived(log):
    log.append("waive-gate", target="integrity", decision="waive", reason="r", author="a")
    (res,) = run_gates([_gate("integrity", ["x"], waivable=False)], ctx=None, decisions=log)
    assert not res.passed and res.waived_by is None and res.waivable is False


def test_waivable_gate_waived_only_by_matching_entry(log):
    gate = _gate("capability", ["creator"], waivable=True)
    (res,) = run_gates([gate], None, log)
    assert not res.passed and res.waivable and res.waived_by is None

    log.append("waive-gate", target="subtle", decision="waive", reason="r", author="a")
    (res,) = run_gates([gate], None, log)
    assert not res.passed and res.waived_by is None  # other target

    log.append("leak-feature", target="capability", decision="exclude", reason="r", author="a")
    (res,) = run_gates([gate], None, log)
    assert not res.passed and res.waived_by is None  # other kind

    entry = log.append("waive-gate", target="capability", decision="waive", reason="r", author="a")
    (res,) = run_gates([gate], None, log)
    assert res.passed and res.waived_by == entry["id"]
    assert res.offenders == ("creator",)  # offenders stay visible for the report


def test_decision_log_ids_and_fields(log):
    e1 = log.append("leak-feature", target="f1", decision="exclude", reason="r1", author="alice")
    e2 = log.append("investigation", target="inv", decision="continue", reason="r2", author="bob")
    assert (e1["id"], e2["id"]) == ("D-0001", "D-0002")
    assert set(e1) == {"id", "kind", "target", "decision", "reason", "author", "date"}
    assert e1["date"].startswith("2026-08-01")
    on_disk = json.loads(log.path.read_text())
    entries = on_disk["decisions"] if isinstance(on_disk, dict) else on_disk
    assert [e["id"] for e in entries] == ["D-0001", "D-0002"]


def test_decision_log_persists_across_instances(tmp_path, is_ignored, clock):
    path = tmp_path / "decisions.json"
    DecisionLog(path, is_ignored=is_ignored, clock=clock).append(
        "waive-gate", target="g", decision="waive", reason="r", author="a")
    again = DecisionLog(path, is_ignored=is_ignored, clock=clock)
    assert [e["id"] for e in again.entries()] == ["D-0001"]
    assert again.append("waive-gate", target="h", decision="waive", reason="r", author="a")["id"] == "D-0002"


def test_decision_log_is_append_only(log):
    for name in ("remove", "delete", "edit", "update", "pop", "clear", "replace", "__setitem__", "__delitem__"):
        assert not hasattr(log, name), name
    log.append("leak-feature", target="f", decision="accept", reason="r", author="a")
    entries = log.entries()
    assert isinstance(entries, tuple)
    with pytest.raises(TypeError):
        entries[0]["decision"] = "exclude"  # type: ignore[index]


def test_decision_log_rejects_unknown_kind_and_writes_nothing(log):
    with pytest.raises(DecisionError):
        log.append("bogus", target="f", decision="exclude", reason="r", author="a")
    assert log.entries() == ()
    assert not log.path.exists()


@pytest.mark.parametrize("kind,decision", [
    ("leak-feature", "waive"), ("investigation", "exclude"), ("waive-gate", "continue")])
def test_decision_log_rejects_decision_not_valid_for_kind(log, kind, decision):
    with pytest.raises(DecisionError):
        log.append(kind, target="t", decision=decision, reason="r", author="a")


@pytest.mark.parametrize("kind,decision", [
    ("leak-feature", "exclude"), ("leak-feature", "accept"),
    ("investigation", "continue"), ("investigation", "abort"),
    ("waive-gate", "waive"), ("awkward-review", "reviewed")])
def test_decision_log_accepts_known_kinds(log, kind, decision):
    assert log.append(kind, target="t", decision=decision, reason="r", author="a")["kind"] == kind


def test_decision_log_write_goes_through_the_guard(tmp_path, is_ignored, clock):
    from fraud_detect.ml.retraining.safe_io import WriteGuardError
    is_ignored.ignored = False
    log = DecisionLog(tmp_path / "decisions.json", is_ignored=is_ignored, clock=clock)
    with pytest.raises(WriteGuardError):
        log.append("waive-gate", target="g", decision="waive", reason="r", author="a")
    assert not (tmp_path / "decisions.json").exists()


def test_post_test_change_log_records_entries(tmp_path, is_ignored, clock):
    log = PostTestChangeLog(tmp_path / "post_test_changes.json", is_ignored=is_ignored, clock=clock)
    log.append("cutoff", description="moved medium cutoff", affected=["test", "heldout_seed"])
    log.append("feature", description="dropped f", affected=["test"], trigger="auc=0.995")
    entries = log.entries()
    assert [e["kind"] for e in entries] == ["cutoff", "feature"]
    assert entries[0]["affected"] == ["test", "heldout_seed"]
    assert entries[1]["trigger"] == "auc=0.995"
    assert set(entries[0]) >= {"date", "kind", "description", "trigger", "affected"}
    with pytest.raises(DecisionError):
        log.append("nonsense", description="x", affected=[])
    assert not any(hasattr(log, n) for n in ("remove", "edit", "update", "clear"))
