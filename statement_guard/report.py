"""Human-readable rendering of a screening report (auditable evidence view)."""

from __future__ import annotations

from typing import Any

_SEV_MARK = {
    "critical": "[!!]",
    "high": "[! ]",
    "medium": "[* ]",
    "low": "[. ]",
    "info": "[i ]",
}
_BAR = "=" * 68


def render(report: dict[str, Any]) -> str:
    c = report["case"]
    s = report["summary"]
    dq = report["data_quality"]
    out: list[str] = []

    out.append(_BAR)
    out.append(" STATEMENTGUARD — SCREENING REPORT")
    out.append(_BAR)
    out.append(f" Account holder : {c['account_holder']}")
    out.append(f" Account number : {c['account_number']}")
    out.append(f" Period         : {c['period']}  ({c['currency']})")
    out.append(f" Transactions   : {c['transactions']}")
    out.append("")

    decision = s["decision"]
    verdict = "AUTO-CLEARED" if decision == "AUTO_CLEAR" else "ROUTED TO HUMAN REVIEW"
    out.append(f" DECISION       : {verdict}")
    out.append(
        f" RISK           : {s['risk_score']}/100  ({s['risk_band']})"
    )
    out.append(
        f" CONFIDENCE     : {s['confidence_score']}/100  ({s['confidence_band']})"
    )
    out.append(f" OCR confidence : {dq['ocr_confidence']:.0%}"
               f"   data quality: {'POOR' if dq['is_poor'] else 'OK'}")
    out.append("")
    out.append(" Routing rationale:")
    for r in s["routing_reasons"]:
        out.append(f"   - {r}")
    out.append("")
    out.append(" Confidence rationale:")
    for r in s["confidence_reasons"]:
        out.append(f"   - {r}")
    out.append("")
    out.append(f" Recommendation : {s['recommendation']}")
    out.append("")

    findings = report["findings"]
    out.append(_BAR)
    out.append(f" EVIDENCE — {len(findings)} finding(s)")
    out.append(_BAR)
    if not findings:
        out.append(" (none — no tampering or anomaly signals detected)")
    for f in findings:
        mark = _SEV_MARK.get(f["severity"], "[? ]")
        out.append(
            f" {mark} {f['severity'].upper():<8} +{f['points']:<2}  "
            f"{f['title']}"
        )
        out.append(f"        node : {f['node']}  |  code: {f['code']}")
        out.append(f"        why  : {f['explanation']}")
        if f["evidence"]:
            out.append(f"        data : {f['evidence']}")
        out.append("")
    out.append(_BAR)
    return "\n".join(out)
