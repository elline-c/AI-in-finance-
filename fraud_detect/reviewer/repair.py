"""Fix two citation slips in a model's answer before it is validated.

The held-out runs showed two slips that waste an otherwise correct answer:

* the routing explanation is right but cites a knowledge entry instead of the routing decision,
* a claim names "row 9" while citing only the finding for row 8.

Both are about *what the claim points at*, not about what it says, and the right citation is
unambiguous. This stage adds exactly those citations and nothing else:

* it never edits text,
* it only adds ``ROUTE`` to the routing explanation, and finding ids for a row that really
  exists in a listed finding,
* a row that no finding has is left alone, so an invented row is still rejected by the
  validator,
* every fix is returned as a readable line, stored with the notes, written to the audit log and
  shown on the page, and the evaluation reports how many answers were accepted only after one.
"""
from __future__ import annotations

from ..models import ReviewerNarrative
from .context import KIND_FINDING, ExplainContext
from .validator import MAX_CITES, ROW_RE, _visible


def repair_narrative(narrative: ReviewerNarrative, ctx: ExplainContext
                     ) -> tuple[ReviewerNarrative, list[str]]:
    """``narrative`` with the two citation slips fixed, and a line for each fix."""
    out = narrative.model_copy(deep=True)
    repairs: list[str] = []

    if "ROUTE" in ctx.sources and "ROUTE" not in out.routing_explanation.cites:
        out.routing_explanation.cites.insert(0, "ROUTE")
        repairs.append("routing_explanation: added the routing decision (ROUTE) as a citation")

    claims = [("headline", out.headline), *[(f"risk_drivers[{i}]", c) for i, c in enumerate(out.risk_drivers)],
              ("confidence_reading", out.confidence_reading), ("routing_explanation", out.routing_explanation),
              *[(f"suggested_checks[{i}]", c) for i, c in enumerate(out.suggested_checks)]]
    for where, claim in claims:
        text, _ = _visible(claim, ctx)
        for m in ROW_RE.finditer(text):
            row = int(m.group(1))
            cited_rows = set().union(*(ctx.sources[x].rows for x in claim.cites if x in ctx.sources))
            if row in cited_rows:
                continue
            owner = next((s for s in ctx.sources.values()
                          if s.kind == KIND_FINDING and row in s.rows and s.id not in claim.cites), None)
            if owner is not None and len(claim.cites) < MAX_CITES:
                claim.cites.append(owner.id)
                repairs.append(f"{where}: added {owner.id} as a citation for row {row}")
    return out, repairs
