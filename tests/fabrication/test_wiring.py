"""Task 7.1: tests for ``analyze_fabrication`` and the Node 3 wiring.

Written before the code exists (task 7.2); expected to fail until then, with
``AttributeError`` for ``fabrication.analyze_fabrication`` and ``TypeError`` for
the ``fabrication_config`` argument of ``financial_logic.analyze``.

Pinned behaviour
----------------
* ``analyze_fabrication(ing, cfg=DEFAULT_CONFIG) -> list[Finding]`` reads the
  credit references once, then runs the rules in this EXECUTION order:
  Rule B (sequence), Rule A (duplicates, given B's break rows as
  ``likely_copies``), Rule C (turnover). Findings are REPORTED in that same
  order: B, then A, then C.
* Each rule runs in its own ``try/except Exception``. A rule that raises is
  replaced by one LOW ``DQ_RULE_SKIPPED`` finding (weight 2, ``field`` = the
  rule's code, evidence mentions "internal error" and the exception type name,
  never the exception message). The other rules still run. If Rule B raises,
  Rule A runs with no ``likely_copies``.
* ``analyze_fabrication`` must call the rule functions through the module
  namespace (``check_invoice_sequence(...)``, not a captured reference), so
  that ``monkeypatch.setattr(fabrication, "check_...", ...)`` takes effect.
* ``financial_logic.analyze(ing, fabrication_config=None)`` appends the
  fabrication findings AFTER all existing Node 3 findings. ``None`` means
  ``DEFAULT_CONFIG``. The agent name is unchanged.
"""
from __future__ import annotations

import pytest

from fraud_detect.models import Severity
from fraud_detect.nodes import financial_logic, fabrication
from fraud_detect.nodes.fabrication import DEFAULT_CONFIG, FabricationConfig

from .fixtures import (
    business_file,
    credit_rows,
    insert_copy,
    make_ingestion,
    move_references_out_of_order,
)
from datetime import date

AGENT = "Node3:FinancialLogic"
DUP = "FL_DUP_INVOICE_REF"
SEQ = "FL_INVOICE_SEQUENCE_BREAK"
TURN = "FL_TURNOVER_COST_MISMATCH"
SKIP = "DQ_RULE_SKIPPED"
FAB_CODES = {DUP, SEQ, TURN, SKIP}
SECRET = "SECRET-TOKEN-12345"


def _codes(findings):
    return [f.code for f in findings]


def _one_copy_file():
    """Fixture file plus one copy of an early credit, dated late (breaks the order)."""
    txns, _ = business_file()
    first = credit_rows(txns)[0]
    copied = insert_copy(txns, first.index, date(2026, 9, 20), 1234.56)
    return copied, make_ingestion(copied)


def _two_copy_file():
    txns, _ = business_file()
    credits = credit_rows(txns)
    out = insert_copy(txns, credits[0].index, date(2026, 9, 20), 1234.56)
    # indices shifted by the insert; look the second source up by description
    src2 = next(t for t in out if t.description == credits[1].description and t.date == credits[1].date)
    out = insert_copy(out, src2.index, date(2026, 9, 22), 2345.67)
    return out, make_ingestion(out)


def _boom(*_a, **_k):
    raise RuntimeError(SECRET)


# ---- wiring through financial_logic.analyze ------------------------------------------

def test_analyze_returns_one_dup_finding_for_one_inserted_copy():
    _, ing = _one_copy_file()
    report = financial_logic.analyze(ing)
    dups = [f for f in report.findings if f.code == DUP]
    assert len(dups) == 1
    assert dups[0].agent == AGENT
    assert dups[0].severity == Severity.MEDIUM
    assert report.agent == AGENT


def test_single_argument_callers_still_work_and_clean_file_has_no_fabrication_codes():
    _, ing = business_file()
    report = financial_logic.analyze(ing)
    assert report.agent == AGENT
    assert not FAB_CODES & set(_codes(report.findings))


def test_config_override_is_honoured():
    _, ing = _one_copy_file()
    report = financial_logic.analyze(ing, fabrication_config=FabricationConfig(dup_escalate_min_refs=1))
    (dup,) = [f for f in report.findings if f.code == DUP]
    assert dup.severity == Severity.HIGH
    assert dup.weight == 22


def test_fabrication_findings_come_after_existing_findings():
    # A balance break gives an existing-rule finding; the copies give fabrication findings.
    txns, _ = _two_copy_file()
    broken = [t.model_copy(update={"balance": (t.balance or 0) + 500}) if t.index == 3 else t for t in txns]
    ing = make_ingestion(broken)
    codes = _codes(financial_logic.analyze(ing).findings)
    fab_positions = [i for i, c in enumerate(codes) if c in FAB_CODES]
    other_positions = [i for i, c in enumerate(codes) if c not in FAB_CODES]
    assert fab_positions and other_positions
    assert max(other_positions) < min(fab_positions)


def test_analyze_fabrication_reports_in_order_b_then_a_then_c():
    _, ing = _two_copy_file()
    codes = _codes(fabrication.analyze_fabrication(ing))
    assert SEQ in codes and DUP in codes
    assert codes.index(SEQ) < codes.index(DUP)
    if TURN in codes:
        assert codes.index(DUP) < codes.index(TURN)


def test_rule_b_break_rows_reach_rule_a_as_likely_copies():
    txns, ing = _one_copy_file()
    copy_row = next(t.index for t in txns if t.date == date(2026, 9, 20) and t.amount == 1234.56)
    findings = fabrication.analyze_fabrication(ing)
    (dup,) = [f for f in findings if f.code == DUP]
    assert f"row {copy_row} breaks the invoice number order, so it is the likelier copy" in dup.evidence


def test_analyze_fabrication_matches_what_analyze_appends():
    _, ing = _two_copy_file()
    direct = fabrication.analyze_fabrication(ing)
    wired = [f for f in financial_logic.analyze(ing).findings if f.code in FAB_CODES]
    assert [f.model_dump() for f in wired] == [f.model_dump() for f in direct]


# ---- failure isolation ---------------------------------------------------------------

@pytest.mark.parametrize("attr,code", [
    ("check_turnover_vs_costs", TURN),
    ("check_duplicate_invoice_refs", DUP),
    ("check_invoice_sequence", SEQ),
])
def test_a_raising_rule_gives_one_skip_finding_and_others_survive(monkeypatch, attr, code):
    _, ing = _two_copy_file()
    baseline = fabrication.analyze_fabrication(ing)
    monkeypatch.setattr(fabrication, attr, _boom)

    report = financial_logic.analyze(ing)          # must not raise
    fab = [f for f in report.findings if f.code in FAB_CODES]
    skips = [f for f in fab if f.code == SKIP]
    assert len(skips) == 1
    s = skips[0]
    assert s.severity == Severity.LOW
    assert s.weight == 2
    assert s.field == code
    assert s.agent == AGENT
    low = s.evidence.lower()
    assert "internal error" in low
    assert "RuntimeError" in s.evidence
    assert SECRET not in s.evidence and SECRET not in s.title

    survivors = [f.model_dump() for f in fab if f.code != SKIP]
    expected = [f.model_dump() for f in baseline if f.code != code]
    if attr == "check_invoice_sequence":
        # Rule A loses the order hint when Rule B is gone.
        for f in fab:
            if f.code == DUP:
                assert "likelier" not in f.evidence.lower()
        assert {f["code"] for f in survivors} == {f["code"] for f in expected}
    else:
        assert survivors == expected


def test_rule_a_still_runs_when_rule_b_raises(monkeypatch):
    _, ing = _one_copy_file()
    monkeypatch.setattr(fabrication, "check_invoice_sequence", _boom)
    findings = fabrication.analyze_fabrication(ing)
    assert [f.code for f in findings].count(SKIP) == 1
    (dup,) = [f for f in findings if f.code == DUP]
    assert "likelier" not in dup.evidence.lower()


def test_rule_a_gets_no_hint_when_find_order_breaks_raises(monkeypatch):
    _, ing = _one_copy_file()
    monkeypatch.setattr(fabrication, "find_order_breaks", _boom)
    findings = fabrication.analyze_fabrication(ing)       # must not raise
    assert DUP in _codes(findings)


def test_no_exception_reaches_caller_when_all_rules_raise(monkeypatch):
    _, ing = _two_copy_file()
    for name in ("check_turnover_vs_costs", "check_duplicate_invoice_refs", "check_invoice_sequence"):
        monkeypatch.setattr(fabrication, name, _boom)
    report = financial_logic.analyze(ing)
    skips = [f for f in report.findings if f.code == SKIP]
    assert len(skips) == 3
    assert {f.field for f in skips} == {DUP, SEQ, TURN}
    assert all(f.severity == Severity.LOW and f.weight == 2 for f in skips)
    assert not any(SECRET in f.evidence for f in skips)


# ---- data gaps, determinism, purity --------------------------------------------------

def test_file_without_references_has_no_rule_a_or_b_finding():
    txns, _ = business_file()
    stripped = [t.model_copy(update={"description": t.description.replace("INV", "REF-")
                                     if t.description else t.description}) for t in txns]
    # "REF-3263" has no INV prefix, so no reference is read.
    findings = fabrication.analyze_fabrication(make_ingestion(stripped))
    assert not {DUP, SEQ} & set(_codes(findings))
    assert SKIP not in _codes(findings)


def test_rows_with_missing_fields_never_raise():
    txns, _ = _two_copy_file()
    damaged = list(txns)
    damaged[5] = damaged[5].model_copy(update={"description": None})
    damaged[6] = damaged[6].model_copy(update={"amount": None})
    damaged[7] = damaged[7].model_copy(update={"date": None, "date_raw": None})
    ing = make_ingestion(damaged)
    findings = fabrication.analyze_fabrication(ing)       # must not raise
    assert SKIP not in _codes(findings)
    financial_logic.analyze(ing)


def test_two_calls_give_identical_findings_and_input_is_unchanged():
    _, ing = _two_copy_file()
    before = ing.model_dump()
    first = fabrication.analyze_fabrication(ing)
    second = fabrication.analyze_fabrication(ing)
    assert [f.model_dump() for f in first] == [f.model_dump() for f in second]
    r1 = financial_logic.analyze(ing)
    r2 = financial_logic.analyze(ing)
    assert r1.model_dump() == r2.model_dump()
    assert ing.model_dump() == before


def test_empty_transactions_returns_empty_list():
    ing = make_ingestion([])
    assert fabrication.analyze_fabrication(ing) == []


def test_rotated_references_alone_do_not_raise_or_skip():
    txns, _ = business_file()
    rows = [t.index for t in credit_rows(txns)[:4]]
    ing = make_ingestion(move_references_out_of_order(txns, rows))
    findings = fabrication.analyze_fabrication(ing)
    assert SKIP not in _codes(findings)


def test_default_config_is_used_when_none_given():
    _, ing = _one_copy_file()
    a = fabrication.analyze_fabrication(ing)
    b = fabrication.analyze_fabrication(ing, DEFAULT_CONFIG)
    assert [f.model_dump() for f in a] == [f.model_dump() for f in b]


def test_synthetic_clean_sample_has_no_fabrication_codes(tmp_path):
    from fraud_detect.nodes.ingestion import ingest
    from fraud_detect.synthetic import generate_clean

    ing = ingest(generate_clean(str(tmp_path)).path)
    report = financial_logic.analyze(ing)
    assert not FAB_CODES & set(_codes(report.findings))
