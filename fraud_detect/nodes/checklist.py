"""Human-decision checklist + per-item / per-stage confidence (ITEM 2 backend).

The analyst console's Human-decision panel shows a CHECKLIST of the key data
points the four-agent screening looks for. For each item the officer sees:

    * a TICK or CROSS  -> was this data point actually detected in the statement
      (``present``), tied to a concrete RiskReport field and/or a specific
      DQ_*/FL_* reason code, never guessed;
    * whether it is COMPULSORY or a recommended-optional field;
    * a CONFIDENCE BAND (High / Medium / Low / n/a) with a plain-English
      ``rationale`` so the officer can decide whether to re-check it manually.

CONFIDENCE IS HONEST AND EXPLAINABLE (MAS FEAT)
-----------------------------------------------
We do NOT invent precise probabilities. Each band is derived ONLY from signals
already present in the RiskReport / its findings, using the SAME logic family as
``risk_synthesis._compute_confidence`` (data-quality problems erode certainty).

Per-item band rule (documented so an officer can defend it):

    * ABSENT (cross)                     -> ``n/a``
      Nothing was detected, so there is nothing to be confident about.
      rationale: "field not detected".

    * PRESENT (tick) AND the source parsed cleanly -- i.e. NONE of the
      readability data-quality codes are in report.findings
      (DQ_NO_TEXT, DQ_UNREADABLE_ROW, DQ_PARSE_INCONSISTENT,
      DQ_LAYOUT_UNRECOGNISED) AND no DQ_MISSING_* finding touches a field
      RELATED to this item                -> ``High``
      rationale names the clean digital parse.

    * PRESENT (tick) BUT a readability data-quality finding is present
      (any of DQ_NO_TEXT / DQ_UNREADABLE_ROW / DQ_PARSE_INCONSISTENT /
      DQ_LAYOUT_UNRECOGNISED), OR a DQ_MISSING_* finding touches a related
      field                                -> ``Low``
      rationale: detected but parsed from low-quality/unreadable input;
      verify manually. These are the items FEAT-004 visually flags.

    * otherwise                           -> ``Medium``.

Per-stage band rule (``stage_confidence``) mirrors the same family:

    * a stage whose inputs parsed cleanly and digitally (no readability DQ_*
      codes) -> ``High``;
    * a stage operating on unreadable / low-quality input (any readability DQ_*
      code present) -> ``Low``;
    * otherwise ``Medium``.

  Node 1 (Ingestion & data quality) is special: its confidence reflects the
  COUNT of its OWN data-quality findings directly -- no DQ findings is High,
  a readability DQ finding is Low, any other DQ finding (e.g. a lone
  DQ_MISSING_* / DQ_NO_TXNS) is Medium. This mirrors the ``-8 per DQ finding``
  erosion in risk_synthesis._compute_confidence.

Everything here is a PURE function of the RiskReport: no I/O, no randomness.
"""
from __future__ import annotations

from typing import NamedTuple

from ..models import Finding, RiskReport

# --------------------------------------------------------------------------- #
# Reason codes we key off (all already emitted elsewhere in the pipeline):
#   header.missing_field_findings -> DQ_MISSING_HOLDER / _UEN / _PERIOD / _OPENING
#   ingest_parse/parser.py + selfcheck.py -> readability DQ codes below + DQ_NO_TXNS
#   financial_logic.py -> FL_PERIOD_LENGTH / FL_BALANCE_BREAK / FL_CLOSING_MISMATCH
# --------------------------------------------------------------------------- #
# Codes that mean the *source itself* was hard to read, so ANY field pulled from
# it is less trustworthy. Their presence drops a present item to Low.
READABILITY_DQ_CODES = frozenset({
    "DQ_NO_TEXT",
    "DQ_UNREADABLE_ROW",
    "DQ_PARSE_INCONSISTENT",
    "DQ_LAYOUT_UNRECOGNISED",
})

# Confidence band labels (strings, NOT numbers -- bands only, honest/explainable).
BAND_HIGH = "High"
BAND_MEDIUM = "Medium"
BAND_LOW = "Low"
BAND_NA = "n/a"

# Side-note shown under the checklist. Single source of truth so FEAT-004's
# template imports exactly this wording.
CHECKLIST_SIDE_NOTE = (
    "Recommendation: we advise the review officer to request ALL information in "
    "this checklist from the applicant - both the compulsory fields and the "
    "recommended-optional ones - paying particular attention to any item marked "
    "with a cross (not detected) or a Low confidence band, which should be "
    "verified manually before signing off."
)


class ChecklistItem(NamedTuple):
    """One row of the Human-decision checklist.

    key             stable identifier (template / test friendly).
    label           human-readable field name.
    compulsory      True if the statement MUST carry this field.
    present         the tick (True) / cross (False) - was it detected.
    confidence_band one of High / Medium / Low / n/a (see module docstring).
    rationale       plain-English reason for the band (never empty).
    """
    key: str
    label: str
    compulsory: bool
    present: bool
    confidence_band: str
    rationale: str


def _codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


def _has_readability_issue(codes: set[str]) -> bool:
    return bool(codes & READABILITY_DQ_CODES)


def _item_band(present: bool, codes: set[str], related_missing: tuple[str, ...]) -> tuple[str, str]:
    """Derive (band, rationale) for a single checklist item.

    ``related_missing`` lists the DQ_MISSING_* codes whose presence should also
    pull this item down to Low (e.g. the period item is Low if DQ_MISSING_PERIOD
    fired even though start/end dates were partially recovered).
    """
    if not present:
        # Nothing detected -> nothing to be confident about.
        return BAND_NA, "field not detected"

    readability = _has_readability_issue(codes)
    related = any(code in codes for code in related_missing)
    if readability or related:
        return (
            BAND_LOW,
            "detected but parsed from low-quality/unreadable input; verify manually",
        )
    if not codes:
        # No data-quality findings at all -> clean digital parse.
        return BAND_HIGH, "detected from a clean digital parse with no data-quality findings"
    # Some non-readability data-quality findings exist elsewhere, but none touch
    # this field: solid but not pristine.
    return (
        BAND_MEDIUM,
        "detected; the statement has unrelated data-quality findings, so treat as moderate",
    )


def build_checklist(report: RiskReport) -> list[ChecklistItem]:
    """Build the Human-decision checklist for a screened case.

    Each item's ``present`` flag is tied to a concrete RiskReport field and/or a
    specific DQ_/FL_ reason code (never guessed); each ``confidence_band`` is
    derived per the module docstring rule.
    """
    codes = _codes(report.findings)

    period_dates_known = (
        report.statement_period_start is not None
        and report.statement_period_end is not None
    )
    txns_present = len(report.transactions) > 0 and "DQ_NO_TXNS" not in codes

    items: list[ChecklistItem] = []

    def add(key, label, compulsory, present, related_missing=()):
        band, rationale = _item_band(present, codes, related_missing)
        items.append(ChecklistItem(key, label, compulsory, present, band, rationale))

    # --- Compulsory header fields (header.missing_field_findings) ------------
    add(
        "account_holder", "Account holder name", True,
        bool(report.account_holder) and "DQ_MISSING_HOLDER" not in codes,
        related_missing=("DQ_MISSING_HOLDER",),
    )
    add(
        "uen", "Company UEN", True,
        bool(report.uen) and "DQ_MISSING_UEN" not in codes,
        related_missing=("DQ_MISSING_UEN",),
    )
    add(
        "statement_period", "Statement period", True,
        period_dates_known and "DQ_MISSING_PERIOD" not in codes,
        related_missing=("DQ_MISSING_PERIOD",),
    )
    add(
        "opening_balance", "Opening balance", True,
        report.opening_balance is not None and "DQ_MISSING_OPENING" not in codes,
        related_missing=("DQ_MISSING_OPENING",),
    )

    # --- Transactions (compulsory) ------------------------------------------
    add("transactions_present", "Transaction rows", True, txns_present)

    # --- Optional / recommended header fields -------------------------------
    add("account_number", "Account number", False, bool(report.account_number))
    add("closing_balance", "Closing balance", False, report.closing_balance is not None)
    add("declared_currency", "Declared currency", False, bool(report.declared_currency))

    # --- Recommended financial-logic coverage checks ------------------------
    # 6-12 month coverage: present when the dates are known AND Node 3 did not
    # raise FL_PERIOD_LENGTH.
    add(
        "period_covers_6_to_12_months", "Period covers 6-12 months", False,
        period_dates_known and "FL_PERIOD_LENGTH" not in codes,
    )
    # Running balances reconcile: present when neither FL_BALANCE_BREAK nor
    # FL_CLOSING_MISMATCH fired.
    add(
        "running_balances_reconcile", "Running balances reconcile", False,
        "FL_BALANCE_BREAK" not in codes and "FL_CLOSING_MISMATCH" not in codes,
    )

    return items


def stage_confidence(stage_title: str, findings: list[Finding],
                     report: RiskReport) -> tuple[str, str]:
    """Return (band, rationale) for one pipeline stage.

    ``findings`` are the findings grouped into this stage (same grouping the
    console uses via _STAGE_META); ``report`` carries the whole-case signal set.

    Rule family (mirrors risk_synthesis._compute_confidence; documented above):

      * Node 1 (Ingestion & data quality) reflects the count/kind of its OWN
        data-quality findings: none -> High, a readability DQ code -> Low,
        otherwise (e.g. a lone DQ_MISSING_* / DQ_NO_TXNS) -> Medium.
      * Every other stage: Low if the SOURCE was unreadable (a readability DQ
        code is present anywhere in the case), else High if this stage parsed
        cleanly with no findings of its own, else Medium.
    """
    case_codes = _codes(report.findings)
    stage_codes = _codes(findings)

    if stage_title.startswith("1 "):
        # Node 1 owns the data-quality findings; grade on its own output.
        if not stage_codes:
            return BAND_HIGH, "ingestion read every field cleanly with no data-quality findings"
        if _has_readability_issue(stage_codes):
            return (
                BAND_LOW,
                "ingestion hit unreadable/low-quality input (readability data-quality finding)",
            )
        return (
            BAND_MEDIUM,
            "ingestion raised data-quality findings (e.g. a missing mandatory field) "
            "but the source was readable",
        )

    # Downstream stages operate on Node 1's parse; if the SOURCE was unreadable,
    # everything built on it is less trustworthy.
    if _has_readability_issue(case_codes):
        return (
            BAND_LOW,
            "this stage operated on unreadable/low-quality input flagged during ingestion; "
            "verify manually",
        )
    if not stage_codes:
        return BAND_HIGH, "ran on a clean digital parse and raised no findings"
    return BAND_MEDIUM, "ran on a clean parse and raised findings worth a human look"


def stage_confidence_map(stages, report: RiskReport) -> dict[str, tuple[str, str]]:
    """Convenience accessor keyed by the _STAGE_META title.

    ``stages`` is the list produced by webapp._group_by_stage(...) -- a sequence
    of (title, blurb, [findings]) tuples. Returns {title: (band, rationale)} so
    FEAT-004 can show a confidence band next to each inlined stage.
    """
    out: dict[str, tuple[str, str]] = {}
    for title, _blurb, stage_findings in stages:
        out[title] = stage_confidence(title, list(stage_findings), report)
    return out
