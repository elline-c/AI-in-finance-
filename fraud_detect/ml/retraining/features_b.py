"""Node 6b feature table (Task 9.2; Requirements 10.1-10.4, 10.8-10.10).

One row per labelled document: corpus metadata, the raw ``extract_features`` outputs and the
style profile read back from the same PDF. Unreadable documents are skipped, counted per Split
and reported. The raw extract values are cached as CSV under a key made of the master seed, the
``labels.csv`` hash and the extractor source hash. ``pdfplumber`` and ``pandas`` are imported
inside functions so that importing this module stays light.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import gates
from .corpora import load_corpus
from .safe_io import atomic_write
from .style_profile import StyleProfile

# FEATURE_NAMES (model inputs) followed by the two evidence-only values; kept literal so the
# module does not import statement_features (and so pdfplumber) at import time.
EXTRACT_COLUMNS = (
    "n_pages", "n_text_objects", "n_distinct_fonts", "minor_font_share", "n_content_streams",
    "has_editing_software", "moddate_after_creation", "n_money_tokens",
    "balance_reconcile_breaks", "balance_reconcile_rate", "n_currency_styles",
    "n_distinct_date_formats", "duplicate_amount_overlap",
)
STYLE_FIELDS = tuple(f.name for f in dataclasses.fields(StyleProfile))
_META_COLUMNS = ("corpus_key", "doc_id", "split", "layout", "company_key", "is_tampered",
                 "genuine_resaved", "tamper_types", "tamper_method", "n_months", "skipped")


class SkippedShareError(Exception):
    """A Split has more skipped documents than allowed; the run stops with exit code 2."""

    exit_code = gates.EXIT_INTEGRITY

    def __init__(self, offenders: dict, limit: float):
        self.offenders = dict(offenders)
        self.limit = limit
        detail = ", ".join(f"{s}: {p:.1f}%" for s, p in sorted(self.offenders.items()))
        super().__init__(f"skipped documents exceed {limit}% in: {detail}")


@dataclass
class FeatureTableB:
    frame: object
    skipped_ids: tuple
    skip_report: dict
    cache_path: Path
    from_cache: bool


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def extractor_sha8() -> str:
    spec = importlib.util.find_spec("fraud_detect.ml.statement_features")
    return _sha8(Path(spec.origin).read_bytes())


def labels_sha8(corpus_dir) -> str:
    return _sha8((Path(corpus_dir) / "labels.csv").read_bytes())


def cache_path_b(cache_root, master_seed, labels_sha8, extractor_sha8) -> Path:
    return Path(cache_root) / "features_b" / f"{master_seed}-{labels_sha8}-{extractor_sha8}.csv"


def _header(master_seed, lsha, esha) -> str:
    return f"# master_seed={master_seed} labels_sha8={lsha} extractor_sha8={esha}"


def _read_cache(path: Path, header: str) -> Optional[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or lines[0] != header:
        return None
    try:
        reader = csv.DictReader(lines[1:])
        if not reader.fieldnames or "doc_id" not in reader.fieldnames \
                or any(c not in reader.fieldnames for c in EXTRACT_COLUMNS):
            return None
        return {r["doc_id"]: {c: float(r[c]) for c in EXTRACT_COLUMNS} for r in reader}
    except (ValueError, csv.Error):
        return None


def _cache_text(header: str, values: dict) -> str:
    buf = io.StringIO()
    buf.write(header + "\n")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(("doc_id",) + EXTRACT_COLUMNS)
    for doc_id, feats in values.items():
        writer.writerow([doc_id] + [repr(float(feats[c])) for c in EXTRACT_COLUMNS])
    return buf.getvalue()


def build_feature_table(corpus_dir, *, cache_root, is_ignored=None, extract: Optional[Callable] = None,
                        skip_max_share=2) -> FeatureTableB:
    import pandas as pd
    from .style_profile import style_profile

    corpus_dir = Path(corpus_dir)
    corpus = load_corpus(corpus_dir)
    lsha, esha = labels_sha8(corpus_dir), extractor_sha8()
    path = cache_path_b(cache_root, corpus.master_seed, lsha, esha)
    header = _header(corpus.master_seed, lsha, esha)

    cached = _read_cache(path, header)
    from_cache = cached is not None
    if extract is None:
        from fraud_detect.ml import statement_features
        extract = statement_features.extract_features

    values: dict = {}
    for d in corpus.docs:
        if from_cache:
            feats = cached.get(d.doc_id)
        else:
            try:
                feats = extract(corpus_dir / "pdfs" / f"{d.doc_id}.pdf")
            except Exception:
                feats = None
        if feats is not None:
            values[d.doc_id] = feats

    skipped_ids = tuple(d.doc_id for d in corpus.docs if d.doc_id not in values)
    report: dict = {}
    for d in corpus.docs:
        rep = report.setdefault(d.split, {"skipped": 0, "total": 0, "percent": 0.0})
        rep["total"] += 1
        rep["skipped"] += d.doc_id not in values
    for rep in report.values():
        rep["percent"] = 100.0 * rep["skipped"] / rep["total"]
    offenders = {s: r["percent"] for s, r in report.items() if r["percent"] > skip_max_share}
    if offenders:
        raise SkippedShareError(offenders, skip_max_share)

    rows = []
    for d in corpus.docs:
        feats = values.get(d.doc_id)
        if feats is None:
            continue
        row = {
            "corpus_key": d.corpus_key, "doc_id": d.doc_id, "split": d.split, "layout": d.layout,
            "company_key": d.company_key, "is_tampered": d.is_tampered,
            "genuine_resaved": d.genuine_resaved, "tamper_types": ";".join(d.tamper_types),
            "tamper_method": d.tamper_method, "n_months": d.n_months, "skipped": False,
        }
        row.update({c: feats[c] for c in EXTRACT_COLUMNS})
        try:
            prof = style_profile(corpus_dir / "pdfs" / f"{d.doc_id}.pdf")
        except Exception:
            prof = None
        for f in STYLE_FIELDS:
            row[f"style_{f}"] = getattr(prof, f) if prof is not None else None
        rows.append(row)

    columns = list(_META_COLUMNS) + list(EXTRACT_COLUMNS) + [f"style_{f}" for f in STYLE_FIELDS]
    frame = pd.DataFrame(rows, columns=columns)

    if not from_cache:
        atomic_write(path, _cache_text(header, values), is_ignored=is_ignored)

    return FeatureTableB(frame=frame, skipped_ids=skipped_ids, skip_report=report,
                         cache_path=path, from_cache=from_cache)
