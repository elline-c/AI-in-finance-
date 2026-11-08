"""Simple JSON-backed case store + audit log.

Persists everything about a submitted statement: the risk report, the human
review decision, and a full audit trail. In production this would be a database
governed by PDPA + MAS TRM (access-controlled, encrypted, retained per policy);
here it is a JSON file so the prototype is self-contained and inspectable.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .audit import AuditChain, build_signer_from_env
from .audit.verify import VerificationStatus
from .history import TrailingHistory, counterparty_key, credit_sizes, monthly_credit_totals
from .registry import names_match
from .models import (
    ApplicationDetails,
    Case,
    CreditDecision,
    CreditMemo,
    CreditRecord,
    ReviewDecision,
    ReviewRecord,
    RiskReport,
    Routing,
    TxnFingerprint,
)

_LOCK = threading.Lock()

# Trailing-history settings (Node 3 gap learning).
_ENV_REQUIRE_CHAIN = "TRAILING_HISTORY_REQUIRE_CHAIN"   # "0" switches the chain check off
_MAX_TRAILING_STATEMENTS = 6                             # moving window: most recent N

# ITEM 4 - revert window. A human reviewer may undo an APPROVED / REJECTED
# decision, but ONLY within this many minutes of the decision. Kept as a single
# named constant so the server guard and the UI (which shows/hides the revert
# control) agree on exactly one threshold.
REVERT_WINDOW_MINUTES = 30


# --------------------------------------------------------------------------- #
# Cross-case memory: fingerprinting helpers
# --------------------------------------------------------------------------- #
# Entity suffixes stripped when normalizing a counterparty name, so that
# "Offshore Nominee A Pte Ltd" and "Offshore Nominee A" match.
_ENTITY_SUFFIXES = (
    "pte ltd", "pte. ltd.", "pte ltd.", "pte",
    "ltd", "ltd.", "llc", "llp", "inc", "inc.", "co", "co.",
    "limited", "corporation", "corp", "corp.",
)

# Amount bucket size (SGD). Amounts in the same bucket are treated as "similar",
# so 150,000 and 149,500 collapse to the same fingerprint.
_AMOUNT_BUCKET = 5_000.0


def normalize_counterparty(name: Optional[str]) -> str:
    """Lower-case, strip punctuation and common entity suffixes."""
    if not name:
        return ""
    s = name.lower().strip()
    for ch in (",", ".", "/", "\\", "-", "_"):
        s = s.replace(ch, " ")
    tokens = [t for t in s.split() if t]
    # Drop trailing entity-type tokens (pte, ltd, llc, ...).
    while tokens and " ".join(tokens[-2:]) in _ENTITY_SUFFIXES:
        tokens = tokens[:-2]
    while tokens and tokens[-1] in {s.replace(" ", "") for s in _ENTITY_SUFFIXES}:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _amount_band(amount: Optional[float]) -> str:
    """Bucket an amount to the *nearest* bucket centre.

    Rounding (rather than flooring) means two amounts that are close but sit
    either side of a bucket boundary — e.g. 149,500 and 150,000 — still collapse
    to the same band, so a lightly-tweaked repeat amount still matches.
    """
    if amount is None:
        return "na"
    centre = round(abs(amount) / _AMOUNT_BUCKET) * _AMOUNT_BUCKET
    return f"~{centre:,.0f}"


def fingerprint(counterparty: Optional[str], amount: Optional[float],
                pattern_tag: str = "") -> TxnFingerprint:
    """Build a comparable fingerprint for one transaction/entity."""
    direction = "na"
    if amount is not None:
        direction = "credit" if amount >= 0 else "debit"
    return TxnFingerprint(
        counterparty=normalize_counterparty(counterparty),
        amount_band=_amount_band(amount),
        direction=direction,
        pattern_tag=pattern_tag,
    )


# --------------------------------------------------------------------------- #
# ITEM 4: revert-window helper (shared by the server guard and the UI)
# --------------------------------------------------------------------------- #
def revert_window_remaining(
    case: Case, *, now: Optional[datetime] = None
) -> tuple[bool, int]:
    """Whether ``case`` can still be reverted, and the seconds left if so.

    Returns ``(allowed, seconds_remaining)``:
      * ``allowed`` is True only when the case holds a final decision
        (APPROVED / REJECTED), has a recorded ``reviewed_at``, and the
        REVERT_WINDOW_MINUTES window has not yet elapsed.
      * ``seconds_remaining`` is the whole seconds left in the window (0 when
        not allowed). The webapp/template use this to show or hide the revert
        control and, optionally, a countdown.

    ``now`` is injectable so both tests and the request handler can pass a fixed
    clock; it defaults to datetime.utcnow().
    """
    if case.review.decision not in (ReviewDecision.APPROVED,
                                     ReviewDecision.REJECTED):
        return (False, 0)
    reviewed_at = case.review.reviewed_at
    if reviewed_at is None:
        return (False, 0)
    now = now or datetime.utcnow()
    deadline = reviewed_at + timedelta(minutes=REVERT_WINDOW_MINUTES)
    remaining = (deadline - now).total_seconds()
    if remaining <= 0:
        return (False, 0)
    return (True, int(remaining))


class CaseStore:
    def __init__(self, path: str = "data/cases.json",
                 audit_path: str = "data/audit_chain.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

        # Signed, hash-chained, append-only audit trail (separate from the
        # mutable cases.json snapshot above). ``build_signer_from_env`` returns
        # None when no key is configured (UNSIGNED mode, a supported state) and
        # fails loud (propagating ValueError) if a key is configured but
        # unreadable -- a misconfigured key must surface, never silently
        # downgrade to UNSIGNED. The append calls are wired in task 8.2.
        signer = build_signer_from_env()
        self._audit = AuditChain(audit_path, signer=signer)

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
            # Compute ts + detail ONCE so the authoritative hash-chained event
            # and the cases.json audit_log mirror agree exactly (same ts, same
            # detail). The chain is the source of truth (Req 3.1/3.3); the
            # embedded audit_log is kept as a read-only mirror for UI
            # convenience under the start-fresh migration stance.
            ts = datetime.utcnow().isoformat()
            detail = (f"risk={report.risk_score} ({report.risk_band.value}), "
                      f"confidence={report.confidence_score} "
                      f"({report.confidence_band.value}), routing={report.routing.value}")
            case.audit_log.append({
                "ts": ts,
                "event": "SCREENED",
                "detail": detail,
            })
            self._audit.append(
                case_id=case.case_id, event="SCREENED", detail=detail, ts=ts,
            )
            data[case.case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    def get(self, case_id: str) -> Optional[Case]:
        data = self._read()
        raw = data.get(case_id)
        return Case.model_validate(raw) if raw else None

    def set_application_details(
        self, case_id: str, details: ApplicationDetails
    ) -> Optional[Case]:
        """Attach the applicant's typed form values to an existing case.

        Called by the customer portal right after screening, so the confirmation
        and status pages can reflect what the applicant actually entered rather
        than the name parsed from the uploaded statement. Leaves the analyst-side
        report/review untouched.
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)
            case.application = details
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    def recall_prior_fingerprints(
        self, exclude_case_id: Optional[str] = None,
        authorised_only: bool = True,
    ) -> list[dict]:
        """Return the transaction fingerprints of prior cases, each tagged with
        the case id and the human decision on that case.

        This is the "authorised historical evidence" the memory node (Node 5)
        reads from: it lets a new screening know whether a similar transaction
        has been seen (and verified) before.

        ``authorised_only`` (default True) restricts the memory to cases a human
        reviewer has actually decided on — REJECTED or APPROVED. Cases still
        PENDING review (an AI screening with no human sign-off) are excluded, so
        the system only learns from *verified reviewer outcomes* and never lets
        an unverified AI suspicion become a fact used against a later applicant.

        ``exclude_case_id`` skips the case currently being screened so it never
        matches itself.
        """
        data = self._read()
        memory: list[dict] = []
        for cid, raw in data.items():
            if exclude_case_id and cid == exclude_case_id:
                continue
            case = Case.model_validate(raw)
            decision = case.review.decision.value   # PENDING / APPROVED / REJECTED
            if authorised_only and decision == ReviewDecision.PENDING.value:
                # Not yet verified by a human -> not authorised evidence.
                continue
            routing = case.report.routing.value      # AUTO_CLEAR / HUMAN_REVIEW
            for fp in case.report.transaction_fingerprints:
                memory.append({
                    "case_id": cid,
                    "decision": decision,
                    "routing": routing,
                    "account_holder": case.report.account_holder,
                    "fingerprint": fp,
                })
        return memory

    # ---- trailing history (Node 3 gap learning) ------------------------ #
    def _chain_is_sound(self) -> bool:
        """True unless the audit chain is broken or carries an invalid signature.

        An unsigned-but-intact chain is accepted (signing is optional). The result
        is cached against the chain file's size and mtime so a screening does not
        re-walk the whole chain every time.
        """
        path = self._audit.path
        try:
            st = path.stat()
            key = (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            key = (0, 0)
        cached = getattr(self, "_chain_check", None)
        if cached and cached[0] == key:
            return cached[1]
        status = self._audit.verify().status
        sound = status in (VerificationStatus.INTACT_SIGNED,
                           VerificationStatus.INTACT_UNSIGNED)
        self._chain_check = (key, sound)
        return sound

    def _human_approved_in_chain(self, case_id: str) -> bool:
        """True when the LAST human decision the chain holds for this case is APPROVED."""
        human = [e.event for e in self._audit.events_for(case_id)
                 if e.event.startswith("HUMAN_")]
        return bool(human) and human[-1] == "HUMAN_APPROVED"

    def trailing_history(
        self, uen: Optional[str], holder: Optional[str],
        exclude_case_id: Optional[str] = None, *,
        require_chain: Optional[bool] = None,
        max_statements: int = _MAX_TRAILING_STATEMENTS,
    ) -> TrailingHistory:
        """Behaviour history of a company (gaps, credit sizes, monthly credit totals),
        from earlier cases we can trust.

        A past case counts only when ALL of these hold:
          * a human reviewer APPROVED it (an auto-clear is a machine's guess and is
            not used, so a planted statement cannot teach Node 3 a false rhythm);
          * its UEN equals ``uen`` and its account holder matches ``holder``;
          * (``require_chain``, on by default) the audit chain is sound and its last
            human event for the case is HUMAN_APPROVED, so an edit to cases.json
            alone cannot make a case trusted.

        Gaps are measured INSIDE each earlier statement and then pooled. Dates are
        never joined across statements, which would hide the silence between them.
        The ``max_statements`` most recent qualifying statements are used, so the
        baseline moves with the company.
        """
        if not uen or not holder:
            return TrailingHistory()
        if require_chain is None:
            require_chain = os.environ.get(_ENV_REQUIRE_CHAIN, "1") != "0"
        if require_chain and not self._chain_is_sound():
            return TrailingHistory()

        from .nodes.financial_logic import gaps_between_active_days  # avoid import cycle

        picked: list[Case] = []
        for case in self.list_cases():
            if exclude_case_id and case.case_id == exclude_case_id:
                continue
            rep = case.report
            if case.review.decision != ReviewDecision.APPROVED:
                continue
            if rep.uen != uen or not names_match(rep.account_holder, holder):
                continue
            if require_chain and not self._human_approved_in_chain(case.case_id):
                continue
            picked.append(case)
        picked.sort(key=lambda c: c.report.created_at, reverse=True)
        picked = picked[:max_statements]

        gaps: list[int] = []
        credits: list[float] = []
        monthly: list[float] = []
        names: set[str] = set()
        for case in picked:
            txns = case.report.transactions
            gaps += gaps_between_active_days([t.date for t in txns if t.date is not None])
            credits += credit_sizes(txns)
            names |= {k for k in (counterparty_key(t.description) for t in txns) if k}
            segs = case.report.statement_segments
            starts = [s.period_start for s in segs if s.period_start is not None]
            ends = [s.period_end for s in segs if s.period_end is not None]
            monthly += monthly_credit_totals(
                txns, min(starts) if starts else None, max(ends) if ends else None)
        return TrailingHistory(
            gap_days=tuple(gaps),
            credits=tuple(credits),
            monthly_credits=tuple(monthly),
            case_ids=tuple(c.case_id for c in picked),
            counterparties=tuple(sorted(names)),
        )

    def list_cases(self) -> list[Case]:
        data = self._read()
        cases = [Case.model_validate(v) for v in data.values()]
        # Sort: still-open cases (awaiting a human or awaiting the customer) first,
        # then finalised ones; within each group, highest risk first.
        _open = {ReviewDecision.PENDING, ReviewDecision.NEEDS_INFO}
        cases.sort(key=lambda c: (
            c.review.decision not in _open,
            -c.report.risk_score,
        ))
        return cases

    def request_info(self, case_id: str, reviewer: str,
                     message_to_customer: str,
                     internal_note: Optional[str] = None) -> Optional[Case]:
        """Reviewer asks the applicant for more information.

        Sets the case to NEEDS_INFO and stores a *customer-safe* message (no
        internal scores/evidence) that the customer portal relays. This is a
        non-terminal state — the applicant can re-submit and the case returns to
        review. The machine still never rejects on its own.
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)
            now = datetime.utcnow()
            case.review = ReviewRecord(
                case_id=case_id,
                decision=ReviewDecision.NEEDS_INFO,
                reviewer=reviewer,
                reason=internal_note,
                info_request=message_to_customer,
                info_requested_at=now,
                # preserve any prior training feedback
                feedback_for_training=case.review.feedback_for_training,
            )
            case.audit_log.append({
                "ts": now.isoformat(),
                "event": "INFO_REQUESTED",
                "detail": f"reviewer={reviewer}; message={message_to_customer}",
            })
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

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
            # Compute ts + detail ONCE so the authoritative hash-chained event
            # and the cases.json audit_log mirror agree exactly (Req 3.2/3.3).
            ts = datetime.utcnow().isoformat()
            event = f"HUMAN_{decision.value}"
            detail = f"reviewer={reviewer}; reason={reason}"
            case.audit_log.append({
                "ts": ts,
                "event": event,
                "detail": detail,
            })
            self._audit.append(
                case_id=case_id, event=event, detail=detail, ts=ts,
            )
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    # ---- ITEM 4: revert a final decision (30-minute window) ------------ #
    def revert_decision(self, case_id: str, reviewer: str, note: str, *,
                        now: Optional[datetime] = None) -> Optional[Case]:
        """Undo an APPROVED / REJECTED decision, re-opening the case to PENDING.

        This exists so a reviewer who mis-clicked (e.g. rejected an application
        that should have been approved) can correct the mistake, but ONLY within
        REVERT_WINDOW_MINUTES of the original decision. Outside that window the
        decision stands and this raises ValueError (the route maps that to an
        HTTP 4xx).

        The revert is modelled as a NEW append-only audit event HUMAN_REVERTED
        written to BOTH the embedded case.audit_log mirror AND the hash-chained
        self._audit. The prior HUMAN_APPROVED / HUMAN_REJECTED entries are left
        completely untouched -- the audit trail is tamper-evident and append-only,
        so history is never deleted or rewritten. The machine never auto-rejects;
        a revert only ever re-opens the case for a human to decide again.

        ``now`` is injectable so tests can exercise the 30-minute boundary
        deterministically; it defaults to datetime.utcnow().
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)

            prior_decision = case.review.decision
            if prior_decision not in (ReviewDecision.APPROVED,
                                      ReviewDecision.REJECTED):
                raise ValueError(
                    "Only an APPROVED or REJECTED case can be reverted; this "
                    f"case is {prior_decision.value}."
                )
            reviewed_at = case.review.reviewed_at
            if reviewed_at is None:
                raise ValueError(
                    "Cannot revert: the case has no recorded decision time."
                )

            now = now or datetime.utcnow()
            # Window has passed once >= REVERT_WINDOW_MINUTES have elapsed. Using
            # >= (not >) means a revert AT exactly the 30-minute mark is already
            # too late, and keeps this guard in lockstep with
            # revert_window_remaining() below (which returns remaining <= 0 as
            # "not allowed"), so the UI and the server never disagree.
            if now - reviewed_at >= timedelta(minutes=REVERT_WINDOW_MINUTES):
                raise ValueError(
                    f"The {REVERT_WINDOW_MINUTES}-minute revert window has "
                    "passed; this decision can no longer be undone."
                )

            ts = now.isoformat()
            detail = (f"reviewer={reviewer}; note={note}; "
                      f"reverted={prior_decision.value}")
            case.audit_log.append({
                "ts": ts,
                "event": "HUMAN_REVERTED",
                "detail": detail,
            })
            self._audit.append(
                case_id=case_id, event="HUMAN_REVERTED", detail=detail, ts=ts,
            )
            # Re-open to PENDING, preserving any training feedback already
            # captured. The prior decision's audit entries remain in the chain.
            case.review = ReviewRecord(
                case_id=case_id,
                decision=ReviewDecision.PENDING,
                feedback_for_training=case.review.feedback_for_training,
            )
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    # ---- ITEM 6: escalate a case to a senior officer ------------------- #
    def escalate_to_senior(self, case_id: str, reviewer: str,
                          senior_officer: str, note: str, *,
                          now: Optional[datetime] = None) -> Optional[Case]:
        """Route a complex case to a named senior officer with a required note.

        A junior reviewer who feels a case is outside their experience can hand
        it to a senior officer. The assignment, the junior reviewer's name and
        the mandatory internal note are stored on the ReviewRecord, and a NEW
        append-only ESCALATED_TO_SENIOR audit event is written to BOTH the
        embedded mirror and the hash-chain.

        If the case is still open (PENDING / NEEDS_INFO) the decision stays
        PENDING so the senior officer picks it up. If the case already holds a
        final decision, that decision is left intact (a senior may re-open/re-
        decide via the normal decision path); escalation never rewrites history
        and never itself decides the case.

        ``now`` is injectable for deterministic tests; defaults to utcnow().
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)

            now = now or datetime.utcnow()
            ts = now.isoformat()
            case.review.assigned_senior_officer = senior_officer
            case.review.escalation_note = note
            case.review.escalated_by = reviewer
            case.review.escalated_at = now

            detail = f"by={reviewer}; to={senior_officer}; note={note}"
            case.audit_log.append({
                "ts": ts,
                "event": "ESCALATED_TO_SENIOR",
                "detail": detail,
            })
            self._audit.append(
                case_id=case_id, event="ESCALATED_TO_SENIOR", detail=detail,
                ts=ts,
            )
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    @staticmethod
    def revert_window_remaining(
        case: Case, *, now: Optional[datetime] = None
    ) -> tuple[bool, int]:
        """Instance-friendly alias for the module-level revert_window_remaining.

        Lets callers that already hold a CaseStore ask ``store.revert_window_
        remaining(case)`` without importing the free function. Both share the
        single REVERT_WINDOW_MINUTES threshold so the UI and server never drift.
        """
        return revert_window_remaining(case, now=now)

    # ---- banker / loan-officer stage ----------------------------------- #
    @staticmethod
    def fraud_is_cleared(case: Case) -> bool:
        """True when a case is safe to put in front of a loan officer.

        A banker should only make a lending call once the document's authenticity
        is settled — i.e. the fraud analyst APPROVED (cleared) it, or it
        auto-cleared with no human concern. A REJECTED or still-open fraud case
        never reaches the banker.
        """
        decision = case.review.decision
        if decision == ReviewDecision.APPROVED:
            return True
        if (decision == ReviewDecision.PENDING
                and case.report.routing == Routing.AUTO_CLEAR):
            return True
        return False

    def list_credit_queue(self) -> list[Case]:
        """Cases whose fraud concern is cleared, for the banker's queue.

        Sorted: cases still awaiting a lending decision first, then by requested
        confidence/risk so the cleanest, highest-confidence files surface first.
        """
        data = self._read()
        cases = [Case.model_validate(v) for v in data.values()]
        cleared = [c for c in cases if self.fraud_is_cleared(c)]
        cleared.sort(key=lambda c: (
            c.credit.decision != CreditDecision.PENDING,   # pending first
            -c.report.confidence_score,                    # most certain first
            c.report.risk_score,                           # lowest risk first
        ))
        return cleared

    def record_credit_decision(
        self, case_id: str, decision: CreditDecision, officer: str,
        rationale: str, approved_amount: Optional[float] = None,
        conditions: Optional[str] = None,
    ) -> Optional[Case]:
        """Persist the loan officer's final lending decision.

        Guardrail: refuses to record a decision on a case whose fraud concern is
        not cleared — the banker stage must never override the fraud outcome.
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)
            if not self.fraud_is_cleared(case):
                raise ValueError(
                    "Fraud concern is not cleared; this case cannot receive a "
                    "lending decision until screening approves it."
                )
            now = datetime.utcnow()
            case.credit = CreditRecord(
                case_id=case_id,
                decision=decision,
                officer=officer,
                rationale=rationale,
                approved_amount=approved_amount,
                conditions=conditions,
                decided_at=now,
            )
            case.audit_log.append({
                "ts": now.isoformat(),
                "event": f"CREDIT_{decision.value}",
                "detail": (f"officer={officer}; rationale={rationale}"
                           + (f"; approved_amount={approved_amount}"
                              if approved_amount is not None else "")),
            })
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case

    def save_credit_memo(
        self, case_id: str, memo: CreditMemo, officer: Optional[str] = None,
    ) -> Optional[Case]:
        """Persist the officer-entered credit memo (financials for ratios).

        Mirrors record_credit_decision: acquire the lock, read + validate, GUARD
        with fraud_is_cleared (a non-cleared case cannot receive memo inputs any
        more than it can receive a lending decision — separation of duties), then
        stamp ``updated_by`` / ``updated_at`` on the memo, attach it to the
        case's credit record, and append ONE shared CREDIT_MEMO_SAVED event to
        BOTH the embedded audit_log mirror AND the hash-chained self._audit. The
        memo figures are officer-entered, never parsed from the statement.

        Returns None if the case does not exist; raises ValueError if the case's
        fraud concern is not cleared.
        """
        with _LOCK:
            data = self._read()
            raw = data.get(case_id)
            if not raw:
                return None
            case = Case.model_validate(raw)
            if not self.fraud_is_cleared(case):
                raise ValueError(
                    "Fraud concern is not cleared; this case cannot receive a "
                    "credit memo until screening approves it."
                )
            now = datetime.utcnow()
            memo.updated_by = officer
            memo.updated_at = now
            if case.credit is None:
                case.credit = CreditRecord(case_id=case_id)
            case.credit.memo = memo

            ts = now.isoformat()
            detail = (f"officer={officer}"
                      if officer else "officer=(unspecified)")
            case.audit_log.append({
                "ts": ts,
                "event": "CREDIT_MEMO_SAVED",
                "detail": detail,
            })
            self._audit.append(
                case_id=case_id, event="CREDIT_MEMO_SAVED", detail=detail, ts=ts,
            )
            data[case_id] = json.loads(case.model_dump_json())
            self._write(data)
            return case
