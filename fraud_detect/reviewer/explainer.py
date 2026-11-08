"""The optional LLM reviewer: explains a finished screening result, never changes it.

Runs after Node 4 (called from ``pipeline.run_pipeline``) and only when switched on
(``FRAUD_ENABLE_LLM_REVIEWER=1``), like the optional ML nodes.

For each case it

1. builds the citation sources (findings, score components, routing, retrieved knowledge,
   past decided cases),
2. builds the deterministic template narrative (always),
3. asks the local model for the same structure, with citation ids limited to this case,
4. validates the answer (faithfulness, consistency, injection),
5. keeps the model's text only if it passes; otherwise keeps the template and records why.

Whatever happens, the scores and the routing are untouched, and a failure of any kind
(server down, bad JSON, failed validation, an exception) leaves the case exactly as it
would have been without this layer plus a template narrative.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol

from ..models import ReviewerNarrative, ReviewerNotes, RiskReport
from . import baseline, context, prompt, validator
from .context import KnowledgeDoc, SimilarCase
from .llm_client import LLMError, LLMUnavailable, OllamaClient

_log = logging.getLogger(__name__)

FLAG_ENV = "FRAUD_ENABLE_LLM_REVIEWER"


class KnowledgeSource(Protocol):
    version: str

    def retrieve(self, report: RiskReport) -> list[KnowledgeDoc]: ...


CaseFinder = Callable[[RiskReport, object], list[SimilarCase]]


@dataclass
class Outcome:
    """Everything one run produced, for the notes and for the evaluation harness."""
    notes: ReviewerNotes
    ctx: context.ExplainContext
    baseline: ReviewerNarrative
    baseline_violations: list[validator.Violation] = field(default_factory=list)
    prompt: Optional[str] = None             # the user message sent to the model
    raw: Optional[str] = None                # the model's raw answer
    schema_ok: Optional[bool] = None         # None when no answer came back
    violations: list[validator.Violation] = field(default_factory=list)   # of the model's answer


class Explainer:
    def __init__(self, client: Optional[OllamaClient] = None, *,
                 knowledge: Optional[KnowledgeSource] = None,
                 case_finder: Optional[CaseFinder] = None) -> None:
        self.client = client
        self.knowledge = knowledge
        self.case_finder = case_finder

    # ------------------------------------------------------------------ #
    def _knowledge(self, report: RiskReport) -> list[KnowledgeDoc]:
        if self.knowledge is None:
            return []
        try:
            return list(self.knowledge.retrieve(report))
        except Exception:  # noqa: BLE001 - retrieval is an optional extra
            _log.warning("reviewer: knowledge retrieval failed", exc_info=True)
            return []

    def _cases(self, report: RiskReport, store) -> list[SimilarCase]:
        if self.case_finder is None or store is None:
            return []
        try:
            return list(self.case_finder(report, store))
        except Exception:  # noqa: BLE001
            _log.warning("reviewer: past-case retrieval failed", exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    def explain_detailed(self, report: RiskReport, store=None) -> Optional[Outcome]:
        """Run the full flow. ``None`` when the report has no score explanation to narrate,
        or when that explanation does not reconcile with the scores."""
        if report.explanation is None:
            return None
        if not report.explanation.reconciled:
            # Narrating a breakdown that does not add up to the scores would explain the wrong thing.
            _log.warning("reviewer: no notes for %s: score breakdown does not reconcile (%s)",
                         report.case_id, "; ".join(report.explanation.notes))
            return None
        docs, cases = self._knowledge(report), self._cases(report, store)
        ctx = context.build_context(report, knowledge=docs, cases=cases)
        template = baseline.build_baseline(ctx)
        template_violations = validator.validate(template, ctx)
        if template_violations:      # a bug in the template; say so, but still show it
            _log.error("reviewer: template narrative failed validation for %s: %s",
                       report.case_id, validator.summarise(template_violations))

        def notes(source: str, narrative: ReviewerNarrative, *, reason: Optional[str] = None,
                  seconds: Optional[float] = None, model: Optional[str] = None) -> ReviewerNotes:
            return ReviewerNotes(
                source=source, narrative=narrative, model=model,
                prompt_version=prompt.PROMPT_VERSION,
                knowledge_version=getattr(self.knowledge, "version", None),
                knowledge_ids=[d.doc_id for d in docs], case_ids=[c.case_id for c in cases],
                citations=baseline.citation_labels(ctx), fallback_reason=reason, seconds=seconds)

        # The prompt is built even with no model attached, so the evaluation can inspect
        # exactly what a model would be shown.
        messages = prompt.build_messages(ctx)
        out = Outcome(notes=notes("BASELINE", template, reason="LLM reviewer not configured"),
                      ctx=ctx, baseline=template, baseline_violations=template_violations,
                      prompt=messages[-1]["content"])
        if self.client is None:
            return out

        started = time.monotonic()
        try:
            result = self.client.chat_json(messages, prompt.response_schema(ctx))
        except LLMUnavailable as exc:
            return self._fallback(out, notes, f"local model unavailable ({exc})", started)
        except (LLMError, ValueError) as exc:
            return self._fallback(out, notes, f"model call failed ({exc})", started)
        except Exception as exc:  # noqa: BLE001 - the screening must never fail because of this
            return self._fallback(out, notes, f"model call failed ({type(exc).__name__})", started)

        out.raw = result.content
        try:
            narrative = validator.normalise_narrative(
                ReviewerNarrative.model_validate(json.loads(result.content)))
        except Exception:  # noqa: BLE001 - bad JSON or wrong shape
            out.schema_ok = False
            return self._fallback(out, notes, "model answer did not match the schema",
                                  started, model=result.model)
        out.schema_ok = True

        out.violations = validator.validate(narrative, ctx)
        if out.violations:
            return self._fallback(out, notes,
                                  "model answer failed validation: " + validator.summarise(out.violations),
                                  started, model=result.model)
        out.notes = notes("LLM", narrative, seconds=result.seconds, model=result.model)
        _log.info("reviewer: case=%s source=LLM model=%s prompt=%s knowledge=%s cases=%s secs=%.1f",
                  report.case_id, result.model, prompt.PROMPT_VERSION, out.notes.knowledge_ids,
                  out.notes.case_ids, result.seconds)
        return out

    def _fallback(self, out: Outcome, make_notes, reason: str, started: float, *,
                  model: Optional[str] = None) -> Outcome:
        out.notes = make_notes("BASELINE", out.baseline, reason=reason,
                               seconds=time.monotonic() - started, model=model)
        _log.warning("reviewer: case=%s source=BASELINE reason=%s",
                     out.ctx.report.case_id, reason)
        return out

    def explain(self, report: RiskReport, store=None) -> Optional[ReviewerNotes]:
        """Notes for ``report``; never raises."""
        try:
            outcome = self.explain_detailed(report, store)
        except Exception:  # noqa: BLE001
            _log.warning("reviewer: failed for %s", report.case_id, exc_info=True)
            return None
        return outcome.notes if outcome else None


def enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(FLAG_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def load_live_explainer(env: Optional[Mapping[str, str]] = None) -> Optional[Explainer]:
    """The explainer for the live apps, or ``None`` while the reviewer is switched off.

    Never raises: a bad configuration (for example a non-loopback model URL) leaves the
    reviewer off and logs why.
    """
    env = os.environ if env is None else env
    if not enabled(env):
        return None
    try:
        from .case_memory import find_similar_cases
        from .knowledge import KnowledgeBase
        return Explainer(OllamaClient(env=env), knowledge=KnowledgeBase.load(),
                         case_finder=find_similar_cases)
    except Exception:  # noqa: BLE001
        _log.warning("reviewer: could not start the LLM reviewer; it stays off", exc_info=True)
        return None
