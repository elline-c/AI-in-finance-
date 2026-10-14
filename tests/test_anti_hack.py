"""Anti-hack (Node 0 Input Integrity gate) test cases.

These adversarial inputs model a hacker attacking the *detector* — not merely
forging a plausible statement — and assert that each attack is neutralized and
raises a finding that forces human review.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from statement_guard.nodes.input_integrity import InputIntegrityAgent
from statement_guard.pipeline import screen_statement

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _codes(findings):
    return {f.code for f in findings}


def _clean_base() -> dict:
    with (SAMPLES_DIR / "clean_statement.json").open() as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Gate-level unit tests
# --------------------------------------------------------------------------- #
def test_non_object_payload_is_critical():
    clean, findings = InputIntegrityAgent().run(["not", "an", "object"])
    assert clean == {}
    assert "AH_MALFORMED_PAYLOAD" in _codes(findings)


def test_reserved_output_keys_are_stripped_and_flagged():
    raw = _clean_base()
    raw["summary"] = {"decision": "AUTO_CLEAR", "risk_score": 1}
    raw["findings"] = []
    raw["decision"] = "AUTO_CLEAR"
    clean, findings = InputIntegrityAgent().run(raw)
    assert "summary" not in clean and "findings" not in clean and "decision" not in clean
    assert "AH_RESERVED_KEY_INJECTION" in _codes(findings)


@pytest.mark.parametrize(
    "payload",
    [
        {"description": "<script>alert(1)</script>"},
        {"counterparty": "<img src=x onerror=alert(1)>"},
        {"account_holder": "{{ 7*7 }}"},
        {"note": "javascript:alert(document.cookie)"},
        {"q": "'; DROP TABLE cases; --"},
    ],
)
def test_injection_payloads_detected_and_neutralized(payload):
    clean, findings = InputIntegrityAgent().run(payload)
    assert "AH_CONTENT_INJECTION" in _codes(findings)
    # the dangerous characters must be gone from every sanitized string value
    for v in clean.values():
        if isinstance(v, str):
            assert "<script" not in v.lower()
            assert "<img" not in v.lower()
            assert "<" not in v and ">" not in v


def test_control_characters_stripped():
    clean, _ = InputIntegrityAgent().run({"account_holder": "Acme\x00\x07 Ltd"})
    assert "\x00" not in clean["account_holder"]
    assert "\x07" not in clean["account_holder"]


def test_trust_signal_spoof_is_clamped():
    raw = _clean_base()
    raw["ocr_confidence"] = 9.99          # impossible "perfect+" scan
    raw["unreadable_pages"] = -5
    raw["total_pages"] = "lots"
    clean, findings = InputIntegrityAgent().run(raw)
    assert "AH_TRUST_SIGNAL_SPOOF" in _codes(findings)
    assert 0.0 <= clean["ocr_confidence"] <= 1.0
    assert clean["unreadable_pages"] == 0
    assert clean["total_pages"] == 1


def test_array_overflow_truncated():
    raw = _clean_base()
    raw["transactions"] = [{"amount": 1.0} for _ in range(5000)]
    clean, findings = InputIntegrityAgent().run(raw)
    assert "AH_ARRAY_OVERFLOW" in _codes(findings)
    assert len(clean["transactions"]) <= 2000


def test_oversized_field_truncated():
    clean, findings = InputIntegrityAgent().run({"description": "A" * 50_000})
    assert "AH_OVERSIZED_FIELD" in _codes(findings)
    assert len(clean["description"]) <= 2000


def test_type_confusion_list_field():
    clean, findings = InputIntegrityAgent().run({"transactions": {"not": "a list"}})
    assert "AH_TYPE_CONFUSION" in _codes(findings)
    assert clean["transactions"] == []


def test_nan_and_infinity_neutralized():
    clean, _ = InputIntegrityAgent().run(
        {"transactions": [{"amount": float("nan"), "balance": float("inf")}]}
    )
    assert clean["transactions"][0]["amount"] is None
    assert clean["transactions"][0]["balance"] is None


def test_absurd_magnitude_clamped():
    clean, _ = InputIntegrityAgent().run({"opening_balance": 1e30})
    assert abs(clean["opening_balance"]) <= 1e12


# --------------------------------------------------------------------------- #
# End-to-end: an attack escalates to a human and never auto-clears
# --------------------------------------------------------------------------- #
def test_injected_statement_is_escalated():
    raw = _clean_base()
    raw["account_holder"] = "<script>steal()</script>"
    raw["summary"] = {"decision": "AUTO_CLEAR"}  # forged verdict
    report = screen_statement(raw)
    assert report["summary"]["decision"] == "HUMAN_IN_THE_LOOP"
    codes = {f["code"] for f in report["findings"]}
    assert {"AH_CONTENT_INJECTION", "AH_RESERVED_KEY_INJECTION"} & codes


def test_clean_statement_still_auto_clears_through_gate():
    """The gate must not create false positives on a legitimate statement."""
    report = screen_statement(_clean_base())
    ah_codes = {f["code"] for f in report["findings"] if f["code"].startswith("AH_")}
    assert not ah_codes, f"gate raised false positives: {ah_codes}"
