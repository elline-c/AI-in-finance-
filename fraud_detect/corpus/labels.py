"""Manifests and label files (per-document JSON, ``labels.csv``, ``rows.csv``).

A manifest is a plain ``dict`` made of JSON types only, so it can be written with
sorted keys and fixed formatting and compared byte for byte between builds.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from fraud_detect.corpus.schema import (
    GENERATOR_VERSION,
    LABELS_COLUMNS,
    ROWS_COLUMNS,
    DocumentPlan,
    EditOp,
    FileOp,
    Ledger,
    RenderResult,
    TamperLabel,
)

# Layouts whose real sample statements continue a month on a second page (design D3).
# A spill on any other layout is an extension and is flagged in the label.
REAL_SPILL_LAYOUTS: frozenset[str] = frozenset({"two_col_slash", "two_col_dual_date"})


def text_layer_sha256(pdf_bytes: bytes) -> str:
    """SHA-256 (hex) of the words and their rounded positions; metadata is ignored."""
    import pdfplumber

    h = hashlib.sha256()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=False):
                h.update(f"{pno}|{w['text']}|{w['x0']:.2f}|{w['x1']:.2f}|{w['top']:.2f}\n".encode("utf-8"))
    return h.hexdigest()


def _flags(label: TamperLabel) -> dict:
    return {
        "metadata_altered": label.metadata_altered,
        "font_inconsistent": label.font_inconsistent,
        "date_inconsistent": label.date_inconsistent,
        "overlay_bg_matched": label.overlay_bg_matched,
        "hidden_text_layer": label.hidden_text_layer,
        "genuine_resaved": label.genuine_resaved,
    }


def build_manifest(plan: DocumentPlan, ledger: Ledger, rendered: RenderResult,
                   label: TamperLabel, *, master_seed: int,
                   edits: Sequence[EditOp] = (), file_ops: Sequence[FileOp] = ()) -> dict:
    """Build the manifest dict for one document (genuine or tampered)."""
    led_by_id = {r.row_id: r for r in ledger.rows}
    affected = [dict(a) for a in label.affected_rows]
    fraud_ids = {a.get("row_id") for a in affected if a.get("row_id")}
    fraud_type = {a["row_id"]: (a.get("tamper_type") or (label.tamper_types[0] if label.tamper_types else ""))
                  for a in affected if a.get("row_id")}

    rows = []
    for i, p in enumerate(rendered.printed, start=1):
        led = led_by_id[p.row_id]
        is_fraud = bool(led.fraud) or p.row_id in fraud_ids
        rows.append({
            "row_id": p.row_id, "row_index": i, "page": p.page,
            "month": led.date.strftime("%Y-%m"), "date": led.date.isoformat(),
            "description": led.description, "amount": led.amount,
            "balance": p.balance, "kind": led.kind, "is_fraud_row": is_fraud,
            "fraud_type": fraud_type.get(p.row_id, label.tamper_types[0] if is_fraud and label.tamper_types else ""),
        })
    index_of = {r["row_id"]: r["row_index"] for r in rows}
    page_of = {r["row_id"]: r["page"] for r in rows}
    for a in affected:
        rid = a.get("row_id")
        if rid in index_of:
            a.setdefault("row_index", index_of[rid])
            a.setdefault("page", page_of[rid])

    spilled = tuple(rendered.spilled_months)
    c = ledger.company
    return {
        "doc_id": plan.doc_id,
        "generator_version": GENERATOR_VERSION,
        "synthetic": True,
        "layout": plan.layout,
        "split": plan.split,
        "company": {"company_id": c.company_id, "name": c.name, "uen": c.uen,
                    "registered_name": c.name},
        "seeds": {"master": master_seed, "document": plan.doc_seed},
        "months": [d.strftime("%Y-%m") for d in ledger.months],
        "n_pages": rendered.n_pages,
        "label": {
            "is_tampered": label.is_tampered,
            "tamper_types": list(label.tamper_types),
            "method": label.method,
            "arithmetic_consistent": label.arithmetic_consistent,
            "flags": _flags(label),
            "affected_rows": affected,
            "affected_months": list(label.affected_months),
            "edits": [asdict(e) for e in edits],
            "file_ops": [{**asdict(f), "params": [list(kv) for kv in f.params]} for f in file_ops],
            "should_be_seen_by": list(label.should_be_seen_by),
            "notes": label.notes,
        },
        "spill_not_in_real_samples": bool(spilled) and plan.layout not in REAL_SPILL_LAYOUTS,
        "spilled_months": list(spilled),
        "genuine_hard_cases": sorted({r.hard_case for r in ledger.rows if r.hard_case}),
        "rows": rows,
        "section_summaries": [asdict(s) for s in rendered.section_summaries],
        "text_layer_sha256": text_layer_sha256(rendered.pdf),
    }


def write_manifest(path: Path | str, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def _b(v: Any) -> str:
    return "" if v is None else ("1" if v else "0")


def _labels_line(m: dict) -> dict:
    lab, fl, rows = m["label"], m["label"]["flags"], m["rows"]
    months = m["months"]
    return {
        "doc_id": m["doc_id"], "file": f"pdfs/{m['doc_id']}.pdf", "split": m["split"],
        "layout": m["layout"], "company_id": m["company"]["company_id"],
        "doc_seed": m["seeds"]["document"], "n_months": len(months),
        "first_month": months[0] if months else "", "last_month": months[-1] if months else "",
        "n_pages": m["n_pages"], "n_rows": len(rows),
        "is_tampered": _b(lab["is_tampered"]),
        "tamper_types": ";".join(lab["tamper_types"]), "tamper_method": lab["method"],
        "arithmetic_consistent": _b(lab["arithmetic_consistent"]),
        "genuine_resaved": _b(fl["genuine_resaved"]),
        "metadata_altered": _b(fl["metadata_altered"]),
        "font_inconsistent": _b(fl["font_inconsistent"]),
        "date_inconsistent": _b(fl["date_inconsistent"]),
        "overlay_bg_matched": _b(fl["overlay_bg_matched"]),
        "hidden_text_layer": _b(fl["hidden_text_layer"]),
        "n_affected_rows": len(lab["affected_rows"]),
        "spill_not_in_real_samples": _b(m["spill_not_in_real_samples"]),
        "genuine_hard_cases": ";".join(m["genuine_hard_cases"]),
        "generator_version": m["generator_version"],
    }


def _write_csv(path: Path | str, columns: Sequence[str], lines: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(lines)


def write_labels_csv(path: Path | str, manifests: Iterable[dict]) -> None:
    _write_csv(path, LABELS_COLUMNS, (_labels_line(m) for m in manifests))


def write_rows_csv(path: Path | str, manifests: Iterable[dict]) -> None:
    def lines():
        for m in manifests:
            for r in m["rows"]:
                yield {
                    "doc_id": m["doc_id"], "row_index": r["row_index"], "row_id": r["row_id"],
                    "page": r["page"], "month": r["month"], "date": r["date"],
                    "amount": f"{r['amount']:.2f}", "balance": f"{r['balance']:.2f}",
                    "kind": r["kind"], "is_fraud_row": _b(r["is_fraud_row"]),
                    "fraud_type": r["fraud_type"],
                }
    _write_csv(path, ROWS_COLUMNS, lines())
