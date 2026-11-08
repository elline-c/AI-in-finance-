"""Leak_Test (Requirement 4): can a tiny model separate the classes from one audited feature?

In memory only: no file is read or written. Heavy imports (numpy, scikit-learn) stay inside functions
so that importing the package stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Per-layout AUC value when a layout has fewer than two documents of a class.
NOT_COMPUTABLE = "not computable"

_DEC_KIND = "leak-feature"


class LeakTestError(Exception):
    """A split is empty or has a single class (exit 2)."""

    exit_code = 2

    def __init__(self, split: str, missing_class: Optional[str] = None):
        self.split = split
        self.missing_class = missing_class
        if missing_class is None:
            msg = f"the {split} split is empty"
        else:
            msg = f"the {split} split has no {missing_class} documents"
        super().__init__(msg)


@dataclass(frozen=True)
class LeakRow:
    feature: str
    auc: float
    per_layout: dict
    worst_layout: Optional[str]
    mean_genuine: float
    mean_tampered: float
    audit_decision: str
    forgery_auc: Optional[float] = None


@dataclass(frozen=True)
class Suspect:
    name: str
    kind: str  # "feature" or "model"
    auc: float
    mean_genuine: Optional[float] = None
    mean_tampered: Optional[float] = None
    worst_layout: Optional[str] = None


@dataclass(frozen=True)
class ForgeryGap:
    feature: str
    generator_auc: float
    forgery_auc: float


@dataclass(frozen=True)
class LeakResult:
    rows: tuple
    simple: dict
    suspects: tuple
    decided: tuple
    forgery: str  # "compared" or "skipped"
    forgery_flagged: tuple
    notice: str
    blocked: bool
    exit_code: int


def _require_both_classes(df, split: str) -> None:
    if len(df) == 0:
        raise LeakTestError(split)
    classes = set(int(v) for v in df["y"].unique())
    if 0 not in classes:
        raise LeakTestError(split, "genuine")
    if 1 not in classes:
        raise LeakTestError(split, "tampered")


def _auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, score)), 4)


def _decided(decisions, name: str) -> bool:
    if decisions is None:
        return False
    return any(e.get("kind") == _DEC_KIND and e.get("target") == name
               and e.get("decision") in ("exclude", "accept") for e in decisions.entries())


def leak_test(train, valid, audit, thresholds, decisions, forgery=None, node="6b") -> LeakResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from .audit import model_input_columns

    _require_both_classes(train, "train")
    _require_both_classes(valid, "validation")

    entries = [e for e in audit.entries if e.section == "model_feature" and e.node == node]
    ytr = train["y"].astype(int).to_numpy()
    yva = valid["y"].astype(int).to_numpy()

    use_forgery = False
    notice = ""
    if forgery is None:
        notice = "Forgery_Dataset comparison skipped: no forgery dataset available."
    else:
        try:
            _require_both_classes(forgery, "forgery")
            use_forgery = True
        except LeakTestError:
            notice = "Forgery_Dataset comparison skipped: forgery dataset lacks one class."
    yfo = forgery["y"].astype(int).to_numpy() if use_forgery else None

    gap_limit = thresholds.forgery_auc_gap
    rows, flagged = [], []
    for e in entries:
        f = e.name
        tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(train[[f]].to_numpy(), ytr)
        score = tree.predict_proba(valid[[f]].to_numpy())[:, 1]
        auc = _auc(yva, score)

        per_layout, worst, worst_dist = {}, None, -1.0
        layouts = valid["layout"].to_numpy()
        for layout in sorted(valid["layout"].unique()):
            mask = layouts == layout
            ys = yva[mask]
            n1 = int(ys.sum())
            if n1 < 2 or (len(ys) - n1) < 2:
                per_layout[layout] = NOT_COMPUTABLE
                continue
            a = _auc(ys, score[mask])
            per_layout[layout] = a
            if abs(a - 0.5) > worst_dist:
                worst, worst_dist = layout, abs(a - 0.5)

        f_auc = None
        if use_forgery:
            f_auc = _auc(yfo, tree.predict_proba(forgery[[f]].to_numpy())[:, 1])
            if abs(auc - f_auc) > gap_limit:
                flagged.append(ForgeryGap(f, auc, f_auc))

        rows.append(LeakRow(
            feature=f, auc=auc, per_layout=per_layout, worst_layout=worst,
            mean_genuine=float(valid.loc[valid["y"] == 0, f].mean()),
            mean_tampered=float(valid.loc[valid["y"] == 1, f].mean()),
            audit_decision=e.decision, forgery_auc=f_auc))

    # Simple models on the audited (non-dropped) inputs.
    cols = [c for c in model_input_columns(audit, node) if c in train.columns]
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0)),
        "tree3": DecisionTreeClassifier(max_depth=3, random_state=0),
    }
    simple = {}
    for name, model in models.items():
        model.fit(train[cols].to_numpy(), ytr)
        simple[name] = _auc(yva, model.predict_proba(valid[cols].to_numpy())[:, 1])

    limit = thresholds.leak_auc
    found = [Suspect(r.feature, "feature", r.auc, r.mean_genuine, r.mean_tampered, r.worst_layout)
             for r in rows if r.auc >= limit]
    found += [Suspect(n, "model", a) for n, a in simple.items() if a >= limit]
    decided = tuple(s for s in found if _decided(decisions, s.name))
    suspects = tuple(s for s in found if not _decided(decisions, s.name))

    return LeakResult(
        rows=tuple(rows), simple=simple, suspects=suspects, decided=decided,
        forgery="compared" if use_forgery else "skipped", forgery_flagged=tuple(flagged),
        notice=notice, blocked=bool(suspects), exit_code=3 if suspects else 0)
