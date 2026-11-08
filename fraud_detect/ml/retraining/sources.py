"""Read-only access to Real_Style, Forgery_Dataset and answer-key files, plus the
training-list guard (Task 14.2).

Source files are opened ``rb`` only, through ``safe_io.ReadOnlySource``, hashed, and
handed to feature code as ``io.BytesIO``. Nothing here writes, caches or copies a
source file. Training modules must not import this module (the guard is applied by
the orchestration layer).

Standard library only at module level.
"""
from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from fraud_detect.ml.retraining.safe_io import ReadOnlySource, SourceChanged  # noqa: F401

SOURCE_GENERATOR = "generator_corpus"
SOURCE_FORGERY = "forgery_dataset"
SOURCE_REAL_STYLE = "real_style"
SOURCE_AWKWARD = "awkward_set"

DATASETS: dict[str, str] = {
    "test": SOURCE_GENERATOR,
    "heldout_layout": SOURCE_GENERATOR,
    "heldout_seed": SOURCE_GENERATOR,
    "forgery_test": SOURCE_FORGERY,
    "forgery_ood": SOURCE_FORGERY,
    "real_style": SOURCE_REAL_STYLE,
    "awkward": SOURCE_AWKWARD,
}

FORGERY_SPLIT_DATASETS: dict[str, str] = {
    "test_in_layout": "forgery_test",
    "test_ood_layout": "forgery_ood",
}

REAL_STYLE_NAMES: tuple = ("statement_E.pdf", "statement_F.pdf")

_FORGERY_TRAIN = ("train",)
_FORGERY_VALIDATION = ("val", "validation")


class RealStyleInTraining(Exception):
    """A Real_Style path appears in a training or validation list."""

    def __init__(self, offenders: Iterable):
        self.offenders = tuple(Path(p) for p in offenders)
        super().__init__(
            "Real_Style file(s) found in a training or validation list: "
            + ", ".join(str(p) for p in self.offenders))


@dataclass(frozen=True)
class SourceRead:
    path: Path
    sha256: str
    size: int
    value: Any


def _len_of_buffer(buf: io.BytesIO) -> int:
    return len(buf.getvalue())


def open_source(path, consumer: Callable[[io.BytesIO], Any]) -> SourceRead:
    """Read ``path`` read-only, hash it and give ``consumer`` an in-memory buffer.

    Raises ``SourceChanged`` if the file's size or mtime changed while it was open.
    """
    p = Path(path)
    with ReadOnlySource(p) as src:
        data = src.read_bytes()
        digest = src.sha256
        value = consumer(io.BytesIO(data))
    return SourceRead(path=p, sha256=digest, size=len(data), value=value)


# ---- Real_Style -------------------------------------------------------------------------------

@dataclass(frozen=True)
class RealStyleResult:
    files: dict
    missing: tuple
    status: str


def load_real_style(folder, names=REAL_STYLE_NAMES, consumer=_len_of_buffer,
                    notify: Callable[[str], Any] = print) -> RealStyleResult:
    base = Path(folder)
    files: dict = {}
    missing: list = []
    for name in names:
        target = base / name
        try:
            if not target.is_file():
                raise FileNotFoundError(str(target))
            files[name] = open_source(target, consumer)
        except SourceChanged:
            raise
        except OSError:
            missing.append(name)
            notify(f"Real_Style file {name} is missing or unreadable ({target}); skipped, "
                   f"the Real_Style input is marked 'missing'")
    return RealStyleResult(files=files, missing=tuple(missing),
                           status="missing" if missing else "ok")


def real_style_report_entries(result: RealStyleResult) -> list:
    """Path and SHA-256 per file, never content."""
    return [{"path": str(r.path), "sha256": r.sha256} for r in result.files.values()]


# ---- Forgery_Dataset --------------------------------------------------------------------------

@dataclass(frozen=True)
class ForgeryDoc:
    path: Path
    split: str
    is_tampered: bool


def _flag(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "t", "y", "1.0"):
        return True
    if v in ("0", "false", "no", "f", "n", "0.0"):
        return False
    raise ValueError(f"not a boolean label: {value!r}")


@dataclass(frozen=True)
class ForgeryDataset:
    status: str
    docs: tuple = ()

    def dataset_docs(self, dataset_id: str) -> tuple:
        splits = {s for s, d in FORGERY_SPLIT_DATASETS.items() if d == dataset_id}
        return tuple(d for d in self.docs if d.split in splits)

    def dataset_status(self) -> dict:
        return {ds: ("ok" if self.status == "ok" and self.dataset_docs(ds) else "missing")
                for ds in FORGERY_SPLIT_DATASETS.values()}


def load_forgery(folder, notify: Callable[[str], Any] = print) -> ForgeryDataset:
    base = Path(folder)
    labels = base / "labels.csv"
    try:
        if not labels.is_file():
            raise FileNotFoundError(str(labels))
        with open(labels, "rb") as fh:
            text = fh.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = set(reader.fieldnames or [])
        if not {"file", "split", "is_tampered"} <= fields:
            raise ValueError("labels.csv lacks the columns file, split, is_tampered")
        docs = tuple(
            ForgeryDoc(path=base / r["file"], split=r["split"].strip(),
                       is_tampered=_flag(r["is_tampered"]))
            for r in reader)
    except (OSError, UnicodeDecodeError, ValueError, KeyError, csv.Error, AttributeError) as exc:
        notify(f"Forgery_Dataset not available ({base}): {exc}; "
               f"forgery_test and forgery_ood are marked 'missing'")
        return ForgeryDataset(status="missing", docs=())
    return ForgeryDataset(status="ok", docs=docs)


# ---- answer key -------------------------------------------------------------------------------

def read_answer_key(path) -> dict:
    """``{file stem: is_tampered}`` from a JSON object of per-file records."""
    with open(path, "rb") as fh:
        raw = json.loads(fh.read().decode("utf-8"))
    out: dict = {}
    for name, rec in raw.items():
        stem = name[:-4] if name.lower().endswith(".pdf") else name
        out[stem] = bool(rec["is_tampered"])
    return out


# ---- training-list guard ----------------------------------------------------------------------

def normalise_path(p) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(p)))))


def assert_not_in_training(paths: Iterable, *, protected: Iterable) -> None:
    blocked = {normalise_path(p) for p in protected}
    offenders = [normalise_path(p) for p in paths if normalise_path(p) in blocked]
    if offenders:
        raise RealStyleInTraining(offenders)


@dataclass(frozen=True)
class TrainingLists:
    train: tuple = field(default_factory=tuple)
    validation: tuple = field(default_factory=tuple)


def build_training_lists(corpus_paths: Iterable, *, forgery: Optional[ForgeryDataset] = None,
                         train_on_forgery: bool = False) -> TrainingLists:
    if train_on_forgery:
        if forgery is None or forgery.status != "ok":
            raise ValueError("train_on_forgery requested but the Forgery_Dataset is not available")
        train = tuple(normalise_path(d.path) for d in forgery.docs if d.split in _FORGERY_TRAIN)
        valid = tuple(normalise_path(d.path) for d in forgery.docs
                      if d.split in _FORGERY_VALIDATION)
        return TrainingLists(train=train, validation=valid)

    train, valid = [], []
    for corpus in corpus_paths:
        root = Path(corpus)
        with open(root / "labels.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rel = row.get("file") or f"pdfs/{row['doc_id']}.pdf"
                target = normalise_path(root / rel)
                if row["split"] == "train":
                    train.append(target)
                elif row["split"] == "validation":
                    valid.append(target)
    return TrainingLists(train=tuple(train), validation=tuple(valid))
