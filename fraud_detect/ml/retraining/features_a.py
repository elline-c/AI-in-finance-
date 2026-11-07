"""Node 6a (PDF feature) row table for the retraining pipeline (Task 9.3).

Adapts a generated corpus's ``rows.csv`` into the frame ``build_pdf_features`` expects,
using the same mapping and defaults as ``ml_transaction._rows_from_ingestion``. Only
PDF-visible feature columns are ever produced. Labels are never an input to a feature.
History features are computed per qualified ``account_id`` so they are local to one
company of one corpus. The result is cached as a CSV with a key header line.

No ``pdfplumber``, ``torch`` or ``sklearn`` import happens here.
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.retraining import gates
from fraud_detect.ml.retraining.corpora import load_corpus
from fraud_detect.ml.retraining.safe_io import atomic_write

KEY_COLS = ["corpus_key", "doc_id", "row_index", "account_id"]

# Same defaults as ml_transaction._rows_from_ingestion.
_CAT_DEFAULTS = ("channel", "counterparty_type", "counterparty_country", "merchant_category",
                 "segment", "age_band", "business_type", "kyc_risk_rating")
_NUM_DEFAULTS = ("declared_monthly_income_sgd", "declared_monthly_turnover_sgd")


class ForbiddenFeatureError(Exception):
    exit_code = gates.EXIT_INTEGRITY

    def __init__(self, offenders):
        self.offenders = sorted(set(offenders))
        super().__init__("non-PDF feature columns refused: " + ", ".join(self.offenders))


class RowExclusionError(Exception):
    exit_code = gates.EXIT_INTEGRITY

    def __init__(self, share, limit=None):
        self.share = share
        super().__init__(f"{share:.4f}% of rows excluded for a missing or non-binary label"
                         + (f" (limit {limit}%)" if limit is not None else ""))


@dataclass
class RowTableA:
    rows: pd.DataFrame
    features: pd.DataFrame
    excluded_count: int
    excluded_share: float
    cache_path: Path
    from_cache: bool


def extractor_sha8() -> str:
    return hashlib.sha256(Path(tf.__file__).read_bytes()).hexdigest()[:8]


def rows_sha8(corpus_dir) -> str:
    return hashlib.sha256((Path(corpus_dir) / "rows.csv").read_bytes()).hexdigest()[:8]


def cache_path_a(cache_root, master_seed, rows_sha8, extractor_sha8) -> Path:  # noqa: A002
    return Path(cache_root) / "features_a" / f"{master_seed}-{rows_sha8}-{extractor_sha8}.csv"


def account_id_for(company_key) -> str:
    seed, company_id = company_key
    return f"{seed}-{company_id}"


def rows_to_frame(corpus_dir):
    """Return ``(included rows in rows.csv order, number excluded)``."""
    corpus_dir = Path(corpus_dir)
    corpus = load_corpus(corpus_dir)
    by_doc = {d.doc_id: d for d in corpus.docs}
    records, excluded = [], 0
    with open(corpus_dir / "rows.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            label = (r.get("is_fraud_row") or "").strip()
            doc = by_doc.get(r.get("doc_id"))
            if label not in ("0", "1") or doc is None:
                excluded += 1
                continue
            amount = float(r["amount"])
            records.append({
                "corpus_key": corpus.key,
                "doc_id": r["doc_id"],
                "row_index": int(r["row_index"]),
                "account_id": account_id_for(doc.company_key),
                "timestamp": r["date"],
                "direction": "CR" if amount >= 0 else "DR",
                "amount_sgd": abs(amount),
                "balance_after_sgd": float(r["balance"]),
                "y": int(label),
                "split": doc.split,
            })
    columns = KEY_COLS + ["timestamp", "direction", "amount_sgd", "balance_after_sgd", "y", "split"]
    return pd.DataFrame(records, columns=columns), excluded


def build_pdf_table(rows: pd.DataFrame, build: Optional[Callable] = None) -> pd.DataFrame:
    """PDF features for ``rows`` in the same row order; computed per account."""
    build = build or tf.build_pdf_features
    rows = rows.reset_index(drop=True)
    feature_cols = list(tf.PDF_FEATURE_COLUMNS)
    parts = []
    for _, idx in rows.groupby("account_id", sort=False).indices.items():
        grp = rows.iloc[idx]
        df = pd.DataFrame({
            "account_id": grp["account_id"].to_numpy(),
            "timestamp": grp["timestamp"].to_numpy(),
            "direction": grp["direction"].to_numpy(),
            "amount_sgd": grp["amount_sgd"].to_numpy(),
            "balance_after_sgd": grp["balance_after_sgd"].to_numpy(),
        })
        for col in _CAT_DEFAULTS:
            df[col] = "NA"
        for col in _NUM_DEFAULTS:
            df[col] = 0.0
        df["open_date"] = None
        df[tf.TARGET] = grp["y"].astype(int).to_numpy()
        df[tf.SPLIT_COL] = grp["split"].to_numpy()
        feats = build(df)[feature_cols].reset_index(drop=True)
        feats.index = grp.index
        parts.append(feats)
    if parts:
        feats = pd.concat(parts).sort_index()
    else:
        feats = pd.DataFrame(columns=feature_cols)
    out = rows[KEY_COLS].copy()
    for col in feature_cols:
        out[col] = feats[col].to_numpy()
    out["y"] = rows["y"].astype(int).to_numpy()
    out["split"] = rows["split"].to_numpy()
    return out


def _header(master_seed, rows_hash, extractor_hash) -> str:
    return f"# master_seed={master_seed} rows_sha8={rows_hash} extractor_sha8={extractor_hash}"


def _read_cache(path: Path, header: str, rows: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline().rstrip("\r\n")
            if first != header:
                return None
            body = fh.read()
        table = pd.read_csv(io.StringIO(body), dtype={"corpus_key": str, "doc_id": str,
                                                      "account_id": str, "split": str})
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    expected = KEY_COLS + list(tf.PDF_FEATURE_COLUMNS) + ["y", "split"]
    if list(table.columns) != expected or len(table) != len(rows):
        return None
    if (list(table["doc_id"]) != list(rows["doc_id"])
            or list(table["row_index"]) != list(rows["row_index"])):
        return None
    return table


def load_row_table(corpus_dir, *, cache_root, columns=None, is_ignored=None, build=None,
                   row_exclusion_max_share=1) -> RowTableA:
    corpus_dir = Path(corpus_dir)
    allowed = set(tf.PDF_FEATURE_COLUMNS)
    if columns is not None:
        offenders = [c for c in columns if c not in allowed]
        if offenders:
            raise ForbiddenFeatureError(offenders)

    rows, excluded = rows_to_frame(corpus_dir)
    total = len(rows) + excluded
    share = 100.0 * excluded / total if total else 0.0
    if share > row_exclusion_max_share:
        raise RowExclusionError(share, row_exclusion_max_share)

    master_seed = load_corpus(corpus_dir).master_seed
    rhash, ehash = rows_sha8(corpus_dir), extractor_sha8()
    path = cache_path_a(cache_root, master_seed, rhash, ehash)
    header = _header(master_seed, rhash, ehash)

    table = _read_cache(path, header, rows) if path.is_file() else None
    from_cache = table is not None
    if table is None:
        table = build_pdf_table(rows, build=build)
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        atomic_write(path, header + "\n" + buf.getvalue(), is_ignored=is_ignored)

    if columns is not None:
        wanted = set(columns)
        drop = [c for c in tf.PDF_FEATURE_COLUMNS if c not in wanted]
        table = table.drop(columns=drop)
    return RowTableA(rows=rows, features=table, excluded_count=excluded,
                     excluded_share=share, cache_path=path, from_cache=from_cache)
