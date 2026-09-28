"""Simple JSON-backed case store + audit log.

Persists everything about a submitted statement: the risk report, the human
review decision, and a full audit trail. In production this would be a database
governed by PDPA + MAS TRM (access-controlled, encrypted, retained per policy);
here it is a JSON file so the prototype is self-contained and inspectable.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Case, ReviewDecision, ReviewRecord, RiskReport

_LOCK = threading.Lock()


class CaseStore:
    def __init__(self, path: str = "data/cases.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    # ---- low-level ----------------------------------------------------- #
    def _read(self) -> dict:
        with self.path.open() as fh:
            return json.load(fh)

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2, default=str)
        tmp.replace(self.path)

    # ---- public API ---------------------------------------------------- #
    def upsert_report(self, report: RiskReport) -> Case:
        with _LOCK:
            data = self._read()
            existing = data.get(report.case_id)
            if existing:
                case = Case.model_validate(existing)
                case.report = report
            else:
                case = Case(
                    case_id=report.case_id,
                    report=report,
                    review=ReviewRecord(case_id=report.case_id),
                )
            case.audit_log.append({
                "ts": datetime.utcnow().isoformat(),
                "event": "SCREENED",
                "detail": (f"risk={report.risk_score} ({report.risk_band.value}), "
                           f"confidence={report.confidence_score} "
                           f"({report.confidence_band.value}), routing={report.routing.value}"),
            })
            data[case.case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    def get(self, case_id: str) -> Optional[Case]:
        data = self._read()
        raw = data.get(case_id)
        return Case.model_validate(raw) if raw else None

    def list_cases(self) -> list[Case]:
        data = self._read()
        cases = [Case.model_validate(v) for v in data.values()]
        # Sort: pending human-review first, then by risk desc
        cases.sort(key=lambda c: (
            c.review.decision != ReviewDecision.PENDING,
            -c.report.risk_score,
        ))
        return cases

    def record_decision(self, case_id: str, decision: ReviewDecision,
                        reviewer: str, reason: str,
                        feedback_for_training: Optional[str] = None) -> Optional[Case]:
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)
            case.review = ReviewRecord(
                case_id=case_id,
                decision=decision,
                reviewer=reviewer,
                reason=reason,
                reviewed_at=datetime.utcnow(),
                feedback_for_training=feedback_for_training,
            )
            case.audit_log.append({
                "ts": datetime.utcnow().isoformat(),
                "event": f"HUMAN_{decision.value}",
                "detail": f"reviewer={reviewer}; reason={reason}",
            })
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case
