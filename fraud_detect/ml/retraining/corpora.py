"""Read-only loading of a generated corpus (Requirements 1.1, 1.9, 1.10).

Reads ``labels.csv``, ``companies.json`` and ``corpus_manifest.json``. The Split of a
document is the value in ``labels.csv``; it is validated, never recomputed. PDFs are not
opened here. Standard library only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

VALID_SPLITS = frozenset({"train", "validation", "test", "test_unseen_layout"})


class CorpusError(Exception):
    """A corpus cannot be loaded; the message names the corpus and the offending doc ids."""


@dataclass(frozen=True)
class DocRef:
    corpus_key: str
    master_seed: int
    doc_id: str
    split: str
    layout: str
    company_key: tuple  # (master_seed, company_id): the Group_Key
    uen: str
    is_tampered: bool
    genuine_resaved: bool
    tamper_types: tuple
    tamper_method: str
    n_months: int


@dataclass(frozen=True)
class LoadedCorpus:
    key: str
    master_seed: int
    hold_out_layout: str | None
    docs: tuple
    manifest_counts: dict
    uen_check: dict | None

    def split_counts(self) -> dict:
        return dict(Counter(d.split for d in self.docs))

    def layout_counts(self) -> dict:
        return dict(Counter(d.layout for d in self.docs))


def _read_json(path: Path, name: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise CorpusError(f"corpus {name!r}: required file {path.name} is missing") from None
    except (OSError, ValueError) as exc:
        raise CorpusError(f"corpus {name!r}: cannot read {path.name}: {exc}") from None


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def load_corpus(path) -> LoadedCorpus:
    path = Path(path)
    name = path.name
    labels_path = path / "labels.csv"
    if not labels_path.is_file():
        raise CorpusError(f"corpus {name!r}: labels.csv not found in {path}")
    try:
        with open(labels_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise CorpusError(f"corpus {name!r}: cannot read labels.csv: {exc}") from None

    if "split" not in fields:
        ids = ", ".join(r.get("doc_id", "") for r in rows)
        raise CorpusError(f"corpus {name!r}: labels.csv has no 'split' column; "
                          f"affected documents: {ids}")

    bad = [r.get("doc_id", "") for r in rows if (r.get("split") or "") not in VALID_SPLITS]
    if bad:
        raise CorpusError(f"corpus {name!r}: empty or unknown split value for documents: "
                          f"{', '.join(bad)}")

    manifest = _read_json(path / "corpus_manifest.json", name)
    companies = _read_json(path / "companies.json", name)
    try:
        master_seed = int(manifest["master_seed"])
    except (KeyError, TypeError, ValueError):
        raise CorpusError(f"corpus {name!r}: corpus_manifest.json has no valid master_seed") from None
    uens = {c["company_id"]: c.get("uen", "") for c in companies}

    key = f"seed-{master_seed}"
    docs = []
    for r in rows:
        cid = r["company_id"]
        docs.append(DocRef(
            corpus_key=key,
            master_seed=master_seed,
            doc_id=r["doc_id"],
            split=r["split"],
            layout=r["layout"],
            company_key=(master_seed, cid),
            uen=uens.get(cid, ""),
            is_tampered=_flag(r["is_tampered"]),
            genuine_resaved=_flag(r.get("genuine_resaved", "0")),
            tamper_types=tuple(t for t in (r.get("tamper_types") or "").split(";") if t),
            tamper_method=r.get("tamper_method") or "",
            n_months=int(r["n_months"]),
        ))
    return LoadedCorpus(
        key=key,
        master_seed=master_seed,
        hold_out_layout=manifest.get("hold_out_layout"),
        docs=tuple(docs),
        manifest_counts=manifest.get("counts", {}),
        uen_check=manifest.get("uen_check"),
    )
