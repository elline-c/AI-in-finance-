"""Shared helpers for the reviewer-layer tests (not a test module)."""
from __future__ import annotations

import json
from typing import Optional

from fraud_detect.models import (
    AgentReport, DocumentMetadata, Finding, IngestionResult, RiskReport, Severity, TxnFingerprint,
)
from fraud_detect.nodes import risk_synthesis
from fraud_detect.reviewer import breakdown
from fraud_detect.reviewer.llm_client import LLMResult

HOLDER = "Meridian Trading Pte Ltd"
UEN = "201812345A"


def finding(code="FL_BALANCE_BREAK", sev=Severity.HIGH, weight=26.0, *,
            agent="Node3:FinancialLogic", page: Optional[int] = None, field: Optional[str] = None,
            evidence="The balance does not add up.", title=None) -> Finding:
    return Finding(agent=agent, code=code, title=title or f"Title of {code}", severity=sev,
                   weight=weight, evidence=evidence, page=page, field=field)


def make_report(findings, *, case_id: Optional[str] = None, uen: Optional[str] = UEN,
                holder: Optional[str] = HOLDER, fingerprints=(), dq: int = 0) -> RiskReport:
    """A real Node 4 report (with its explanation) for hand-picked findings."""
    ing = IngestionResult(
        source_path=f"{case_id or 'case'}.pdf", page_count=1, raw_text="",
        metadata=DocumentMetadata(), account_holder=holder, uen=uen,
        data_quality_findings=[
            Finding(agent="Node1:Ingestion", code="DQ_UNREADABLE_ROW", title="Unreadable row",
                    severity=Severity.LOW, weight=3.0, evidence="x") for _ in range(dq)])
    report = risk_synthesis.synthesize(ing, [AgentReport(agent="X", findings=list(findings))])
    if case_id:
        report.case_id = case_id
    report.transaction_fingerprints = [
        TxnFingerprint(counterparty=cp, amount_band="~10,000", direction="credit")
        for cp in fingerprints]
    report.explanation = breakdown.explain_scores(ing, report)
    return report


class PerfectClient:
    """A stand-in model that answers correctly from the prompt alone, like a good model would."""
    model = "perfect:1b"

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, messages, schema, **_kw) -> LLMResult:
        self.calls += 1
        data = json.loads(messages[-1]["content"].split("\n", 1)[1])
        s = data["scores"]
        route = s["routing"]
        findings = sorted(data["findings"], key=lambda f: (-f["weight"], f["id"]))
        drivers = []
        if findings and findings[0]["weight"] > 0:
            f = findings[0]
            drivers.append({"text": f"{f['title']} [{f['code']}] adds {f['weight']:g} points.",
                            "cites": [f["id"]]})
        answer = {
            "headline": {"text": f"Risk {s['risk']}/100, confidence {s['confidence']}/100; {route}.",
                         "cites": ["ROUTE"]},
            "risk_drivers": drivers,
            "confidence_reading": {"text": "It starts at 85; the listed adjustments raised or lowered it from there.",
                                   "cites": ["A1"]},
            "routing_explanation": {"text": data["route"]["statement"][:600], "cites": ["ROUTE"]},
            "suggested_checks": [],
            "similar_cases": [],
        }
        return LLMResult(content=json.dumps(answer), seconds=0.05, model=self.model)


class PrecedentAwareClient(PerfectClient):
    """Like PerfectClient, and writes a similar-case claim whenever past cases are shown."""
    model = "precedent:1b"

    def chat_json(self, messages, schema, **kw):
        res = super().chat_json(messages, schema, **kw)
        data = json.loads(res.content)
        shown = json.loads(messages[-1]["content"].split("\n", 1)[1])["past_cases"]
        if shown:
            first = shown[0]
            data["similar_cases"] = [{
                "text": f"A person {first['decision'].lower()} an earlier case with the same pattern.",
                "cites": [first["id"]]}]
        return type(res)(json.dumps(data), res.seconds, res.model)
