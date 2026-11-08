"""Shared Hypothesis generators for the ml-retraining tests.

Pure data only: nothing here reads a file or imports a heavy library.
"""
from __future__ import annotations

from hypothesis import strategies as st

LAYOUTS = ("generator_v1", "two_col_dmon", "two_col_slash", "two_col_dual_date",
           "signed_amount_iso", "two_col_ddmmyy_cr")
SPLITS = ("train", "validation", "test", "test_unseen_layout")
TAMPER_TYPES = ("edited_amount", "altered_date", "broken_balance", "font_mismatch",
                "metadata_inconsistency", "genuine_resaved", "fabricated_receipts",
                "cross_month_copy", "turnover_inflation", "structuring")
PRODUCER_FAMILIES = ("base", "export", "editor")
SOURCES = ("synthetic", "real_style", "forgery", "awkward")
DATASETS = ("train", "validation", "test", "test_unseen_layout", "awkward",
            "real_style", "forgery")
EVENT_KINDS = ("cutoff", "hyperparameter", "feature", "generator", "detector", "score")


# ------------------------------------------------------------------ label tables
@st.composite
def label_rows(draw, max_rows: int = 40, one_split_per_company: bool = True):
    """A label table: list of dicts with doc_id, layout, company_key, split, flags."""
    n = draw(st.integers(1, max_rows))
    company_split: dict = {}
    rows = []
    for i in range(n):
        seed = draw(st.sampled_from((20260801, 20260802, 20260803)))
        company = (seed, f"C{draw(st.integers(1, max(2, n)))}")
        split = draw(st.sampled_from(SPLITS))
        if one_split_per_company:
            split = company_split.setdefault(company, split)
        tampered = draw(st.booleans())
        types = draw(st.lists(st.sampled_from(TAMPER_TYPES), max_size=2, unique=True)) if tampered else []
        resaved = (not tampered) and draw(st.booleans())
        rows.append({
            "doc_id": f"D{i:04d}", "layout": draw(st.sampled_from(LAYOUTS)),
            "company_key": company, "split": split, "is_tampered": tampered,
            "genuine_resaved": resaved, "tamper_types": types,
        })
    return rows


company_keys = st.tuples(st.integers(20260801, 20260899),
                         st.text("ABCDEF0123456789", min_size=1, max_size=6))


# ---------------------------------------------------------- scores and labels
@st.composite
def score_label_arrays(draw, min_size: int = 2, max_size: int = 80, both_classes: bool = True):
    """(scores, labels) with ties and plateaus; scores on a coarse grid in [0, 1]."""
    n = draw(st.integers(min_size, max_size))
    grid = draw(st.sampled_from((3, 5, 11, 101)))
    scores = draw(st.lists(st.integers(0, grid - 1).map(lambda k: k / (grid - 1)),
                           min_size=n, max_size=n))
    labels = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))
    if both_classes and n >= 2:
        labels[0], labels[1] = 0, 1
    if draw(st.booleans()):  # a plateau: a run of equal scores
        k = draw(st.integers(1, n))
        value = draw(st.sampled_from(scores))
        scores[:k] = [value] * k
    return scores, labels


# ----------------------------------------------------------- style profiles
@st.composite
def style_profile_tables(draw, max_rows: int = 60):
    """Rows of style values with an is_tampered flag."""
    n = draw(st.integers(1, max_rows))
    return [{
        "is_tampered": draw(st.booleans()),
        "producer_family": draw(st.sampled_from(PRODUCER_FAMILIES)),
        "creator": draw(st.sampled_from(("", "Word", "Excel", "Acrobat"))),
        "bank_name": draw(st.sampled_from(("Bank A", "Bank B", "Bank C"))),
        "street": draw(st.sampled_from(("1 Main St", "2 High St"))),
        "row_step_bucket": draw(st.integers(80, 120)) / 10,
        "creation_hour": draw(st.integers(0, 23)),
    } for _ in range(n)]


# ----------------------------------------------------------- corrupted audits
AUDIT_CORRUPTIONS = ("missing_field", "unknown_decision", "duplicate", "missing_feature",
                     "neutralise_without_form")


@st.composite
def corrupted_audits(draw, feature_names=("f1", "f2", "f3", "f4")):
    """(entries, corruption, feature): corruption is None for a valid audit, else a defect name."""
    entries = [{"feature": f, "kind": "model_feature",
                "decision": draw(st.sampled_from(("keep", "drop"))),
                "reason": "r", "form": None} for f in feature_names]
    corruption = draw(st.one_of(st.none(), st.sampled_from(AUDIT_CORRUPTIONS)))
    victim = draw(st.integers(0, len(entries) - 1))
    name = entries[victim]["feature"]
    if corruption == "missing_field":
        del entries[victim]["reason"]
    elif corruption == "unknown_decision":
        entries[victim]["decision"] = "maybe"
    elif corruption == "duplicate":
        entries.append(dict(entries[victim]))
    elif corruption == "missing_feature":
        del entries[victim]
    elif corruption == "neutralise_without_form":
        entries[victim]["decision"] = "neutralise"
    return entries, corruption, (name if corruption else None)


# -------------------------------------------------------------- path spellings
@st.composite
def path_spellings(draw):
    """Different spellings of paths: relative, dotted, doubled slashes, trailing slash."""
    base = draw(st.sampled_from(("data/private/ml_retraining/run", "models/backup/x",
                                 "models/current.joblib", "tmp/out", "fraud_detect/x.py")))
    spelling = draw(st.sampled_from(("plain", "dot", "double", "updown", "trailing", "absolute")))
    if spelling == "dot":
        return "./" + base
    if spelling == "double":
        return base.replace("/", "//")
    if spelling == "updown":
        return "tmp/../" + base
    if spelling == "trailing":
        return base + "/"
    if spelling == "absolute":
        return "/Users/example/repo/" + base
    return base


# ---------------------------------------------------------------- model bytes
model_byte_strings = st.binary(min_size=1, max_size=512)


# --------------------------------------------------------------- metric tables
@st.composite
def metric_tables(draw, max_rows: int = 20):
    """Rows (source, dataset, metric, value, n_genuine, n_tampered); value may be None."""
    n = draw(st.integers(1, max_rows))
    return [{
        "source": draw(st.sampled_from(SOURCES)),
        "dataset": draw(st.sampled_from(DATASETS)),
        "metric": draw(st.sampled_from(("roc_auc", "pr_auc", "catch_rate", "false_alarm_rate"))),
        "value": draw(st.one_of(st.none(), st.floats(0, 1, allow_nan=False))),
        "n_genuine": draw(st.integers(0, 200)),
        "n_tampered": draw(st.integers(0, 200)),
    } for _ in range(n)]


# ------------------------------------------------------------ event sequences
@st.composite
def event_sequences(draw, max_events: int = 25):
    """Ordered events: (day, kind, value); day is non-decreasing."""
    n = draw(st.integers(0, max_events))
    days = sorted(draw(st.lists(st.integers(0, 365), min_size=n, max_size=n)))
    return [{"day": d, "kind": draw(st.sampled_from(EVENT_KINDS)),
             "value": draw(st.integers(0, 9))} for d in days]
