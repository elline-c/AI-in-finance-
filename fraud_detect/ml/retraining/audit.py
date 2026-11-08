"""Feature_Audit: which features a retrained model may see (audit-v1).

The audit is a Python literal. Change it only together with a new ``version``
and a new pinned digest in the tests. Standard library only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

DECISIONS = ("keep", "neutralise", "drop")
SECTIONS = ("model_feature", "candidate_tell")
YES_NO = ("yes", "no")

CANDIDATE_TELL_NAMES = (
    "producer_string", "creator_string", "font_names", "font_count",
    "page_width_height", "page_count", "round_amount_share", "row_order",
    "row_spacing", "creation_time_stamp", "modification_time_stamp", "footer_wording",
)


class AuditError(Exception):
    """The audit is invalid; the message names the affected feature."""


@dataclass(frozen=True)
class AuditEntry:
    name: str
    section: str
    node: str
    measures: str
    label_independent: str
    real_file_could_differ: str
    decision: str
    neutralised_form: str | None = None
    decision_rule: str | None = None
    maps_to: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class FeatureAudit:
    version: str
    entries: tuple[AuditEntry, ...]

    def canonical_json(self) -> str:
        payload = {
            "version": self.version,
            "entries": [dataclasses.asdict(e) for e in self.entries],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _f(name, node, measures, li, rfd, decision, note="", form=None, rule=None):
    return AuditEntry(name=name, section="model_feature", node=node, measures=measures,
                      label_independent=li, real_file_could_differ=rfd, decision=decision,
                      neutralised_form=form, decision_rule=rule, note=note)


def _t(name, measures, maps_to, note):
    return AuditEntry(name=name, section="candidate_tell", node="both", measures=measures,
                      label_independent="yes", real_file_could_differ="yes", decision="drop",
                      maps_to=maps_to, note=note)


AUDIT_V1 = FeatureAudit(
    version="audit-v1",
    entries=(
        # ---- Node 6b model features (11)
        _f("n_pages", "6b", "Page count", "yes", "yes", "drop",
           "Set by months and layout. Still feeds the Size_Guard"),
        _f("n_text_objects", "6b", "Glyph count", "yes", "yes", "drop",
           "Scale; depends on row density and font size. Still feeds the Size_Guard"),
        _f("n_distinct_fonts", "6b", "Number of fonts used", "no", "yes", "drop",
           "Corpus genuine files always have two; Courier appears only in font_mismatch. "
           "No single-column stable form: real files have logo and symbol fonts"),
        _f("minor_font_share", "6b", "Glyph share of the rarest font", "no", "yes", "neutralise",
           "Shares of 10% or more collapse to one value; only a very rare foreign font stays graded",
           form="clip(0, 0.10)"),
        _f("n_content_streams", "6b", "Content streams in the file", "no", "yes", "drop",
           "The renderer writes one per page; only overlays add more. Real producers use many per page"),
        _f("has_editing_software", "6b", "Editing tool word in Producer or Creator", "yes", "yes", "keep",
           "Constant 0 in the corpus (F4). Kept only if the Forgery_Dataset has at least one forged and "
           "one genuine file that differ on it, otherwise dropped; the audit command computes this "
           "and falls back to drop when the dataset is absent",
           rule="forgery_gate"),
        _f("moddate_after_creation", "6b", "ModDate later than CreationDate", "no", "yes", "keep",
           "Boolean, already stable. Genuine re-saves set it too, so it is not a perfect label"),
        _f("n_money_tokens", "6b", "Count of money values in the text", "yes", "yes", "drop",
           "Scale. Still feeds the Size_Guard"),
        _f("balance_reconcile_breaks", "6b", "Rows whose balance does not follow", "no", "yes", "keep",
           "Main arithmetic signal; one edit gives one break whatever the length"),
        _f("balance_reconcile_rate", "6b", "Breaks divided by rows checked", "no", "yes", "drop",
           "One edit gives a smaller rate in a longer statement, so the model would learn the corpus length"),
        _f("n_currency_styles", "6b", "Distinct currency notations", "yes", "yes", "drop",
           "Fixed by the layout; it identifies the layout"),
        # ---- Node 6a PDF_Feature_Set (13)
        _f("amount_sgd", "6a", "Absolute amount", "no", "yes", "keep",
           "Tampers set inserted amounts (legitimate signal). Scale risk noted"),
        _f("balance_after_sgd", "6a", "Balance after the row", "yes", "yes", "drop",
           "The generator sets the opening balance from the lowest replayed balance plus 5% to 20% of turnover"),
        _f("day_of_week", "6a", "Weekday of the date", "no", "yes", "drop",
           "F5. No single-column stable form"),
        _f("log_amount", "6a", "log(1 + amount)", "no", "yes", "keep",
           "Monotone copy of amount_sgd; harmless for trees"),
        _f("n_txn_24h", "6a", "Rows in the last 24 hours (same-day, dates only)", "no", "yes", "keep",
           "Burst signal"),
        _f("n_txn_7d", "6a", "Rows in 7 days", "no", "yes", "keep", "Burst signal"),
        _f("n_cr_7d", "6a", "Credits in 7 days", "no", "yes", "keep", "Fan-in"),
        _f("cr_sum_7d", "6a", "Credit total in 7 days", "no", "yes", "keep", "Scale caveat as amount_sgd"),
        _f("dr_sum_7d", "6a", "Debit total in 7 days", "no", "yes", "keep", "Scale caveat"),
        _f("out_in_ratio_7d", "6a", "Debits over credits in 7 days", "no", "yes", "keep",
           "Scale-free pass-through signal"),
        _f("gap_hours", "6a", "Hours since the previous row", "no", "yes", "keep",
           "Watch: whole multiples of 24 (dates only); also carries the weekday habit indirectly"),
        _f("amt_vs_acct_median", "6a", "Amount over running median", "no", "yes", "keep", "Scale-free"),
        _f("direction", "6a", "Credit or debit", "no", "no", "keep",
           "Sign convention is the same everywhere"),
        # ---- candidate tells (12), never offered to a model
        _t("producer_string", "Producer string", ("has_editing_software",),
           "4 base values; editor names only on tampered, export names only on resaved genuine"),
        _t("creator_string", "Creator string", ("has_editing_software",),
           "Constant. Capability gate: missing"),
        _t("font_names", "Font names", ("n_distinct_fonts", "minor_font_share"),
           "Helvetica family; Courier only in font_mismatch. Capability gate: missing"),
        _t("font_count", "Font count", ("n_distinct_fonts",),
           "Two, or three with font_mismatch. Checked by the Leak_Test"),
        _t("page_width_height", "Page width and height", (), "A4 always. Styling profile"),
        _t("page_count", "Page count", ("n_pages",), "Months times layout. Checked by the Leak_Test"),
        _t("round_amount_share", "Round-amount share", (),
           "Amounts have cents, so round amounts are rare. Checked by the Awkward_Set"),
        _t("row_order", "Row order", ("balance_reconcile_breaks",),
           "Date order; altered_date can break it. Checked by the date_inconsistent label"),
        _t("row_spacing", "Row spacing", (),
           "12.0 to 12.8, label-independent. Checked by the styling balance"),
        _t("creation_time_stamp", "Creation time stamp", ("moddate_after_creation",),
           "Last month end plus 1 to 3 days; random time of day. Checked by the styling balance"),
        _t("modification_time_stamp", "Modification time stamp", ("moddate_after_creation",),
           "Equal to creation except genuine_resaved and metadata_inconsistency. "
           "Checked by the styling balance and Producer family report"),
        _t("footer_wording", "Footer wording", (), "One fixed string. Styling profile"),
    ),
)


def _expected_names():
    from fraud_detect.ml.statement_features import FEATURE_NAMES
    from fraud_detect.ml.transaction_features import PDF_FEATURE_COLUMNS
    return list(FEATURE_NAMES), list(PDF_FEATURE_COLUMNS)


def validate(audit: FeatureAudit) -> None:
    """Raise AuditError (naming the feature) on any defect; return None otherwise."""
    seen: set[tuple[str, str, str]] = set()
    for e in audit.entries:
        key = (e.section, e.node, e.name)
        if key in seen:
            raise AuditError(f"duplicate audit entry for feature {e.name}")
        seen.add(key)
        for field in ("name", "section", "node", "measures", "label_independent",
                      "real_file_could_differ", "decision", "note"):
            value = getattr(e, field)
            if value is None or (field == "measures" and not str(value).strip()):
                raise AuditError(f"feature {e.name}: field {field} is missing")
        if e.section not in SECTIONS:
            raise AuditError(f"feature {e.name}: unknown section {e.section!r}")
        if e.label_independent not in YES_NO:
            raise AuditError(f"feature {e.name}: label_independent must be yes or no")
        if e.real_file_could_differ not in YES_NO:
            raise AuditError(f"feature {e.name}: real_file_could_differ must be yes or no")
        if e.decision not in DECISIONS:
            raise AuditError(f"feature {e.name}: unknown decision {e.decision!r}")
        if e.decision == "neutralise" and not (e.neutralised_form and e.neutralised_form.strip()):
            raise AuditError(f"feature {e.name}: decision neutralise needs a recorded neutralised_form")

    names_6b, names_6a = _expected_names()
    have = {(e.node, e.name) for e in audit.entries if e.section == "model_feature"}
    for node, names in (("6b", names_6b), ("6a", names_6a)):
        for n in names:
            if (node, n) not in have:
                raise AuditError(f"feature {n}: no model_feature audit entry for node {node}")
    expected = {("6b", n) for n in names_6b} | {("6a", n) for n in names_6a}
    for node, n in sorted(have - expected):
        raise AuditError(f"feature {n}: audit entry for node {node} matches no model feature")

    tells = {e.name for e in audit.entries if e.section == "candidate_tell"}
    for n in CANDIDATE_TELL_NAMES:
        if n not in tells:
            raise AuditError(f"feature {n}: candidate tell entry is missing")
    for n in sorted(tells - set(CANDIDATE_TELL_NAMES)):
        raise AuditError(f"feature {n}: unknown candidate tell")


def model_input_columns(audit: FeatureAudit, node: str) -> list[str]:
    """Ordered model input names for ``node`` ("6a" or "6b"); dropped features excluded."""
    if node not in ("6a", "6b"):
        raise ValueError(f"node must be '6a' or '6b', got {node!r}")
    names_6b, names_6a = _expected_names()
    order = names_6b if node == "6b" else names_6a
    decisions = {e.name: e.decision for e in audit.entries
                 if e.section == "model_feature" and e.node == node}
    return [n for n in order if decisions.get(n, "drop") != "drop"]


def creator_only_rule(forged_values, genuine_values) -> str:
    """Requirement 3.8: keep only if both lists are non-empty and differ."""
    forged, genuine = list(forged_values), list(genuine_values)
    if not forged or not genuine:
        return "drop"
    return "keep" if set(forged) != set(genuine) else "drop"
