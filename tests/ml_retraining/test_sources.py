"""Task 14.1: failing tests for read-only sources and the training-list guard.

``fraud_detect/ml/retraining/sources.py`` does not exist yet (task 14.2). These tests pin its interface:

    SOURCE_GENERATOR = "generator_corpus"; SOURCE_FORGERY = "forgery_dataset"
    SOURCE_REAL_STYLE = "real_style";      SOURCE_AWKWARD = "awkward_set"
    DATASETS: dict[str, str]          # the seven dataset ids -> Source
    FORGERY_SPLIT_DATASETS = {"test_in_layout": "forgery_test", "test_ood_layout": "forgery_ood"}
    REAL_STYLE_NAMES = ("statement_E.pdf", "statement_F.pdf")

    open_source(path, consumer) -> SourceRead(path: Path, sha256: str, size: int, value: Any)
        # opens through safe_io.ReadOnlySource ("rb" only), hands io.BytesIO to consumer(buf),
        # raises safe_io.SourceChanged if os.stat (size, mtime) or content changed.
    load_real_style(folder, names=REAL_STYLE_NAMES, consumer=len_of_buffer, notify=print) -> RealStyleResult
        # .files: dict[name, SourceRead]; .missing: tuple[str]; .status: "ok" | "missing"
        # a missing file is skipped, notify(message naming it) is called, others still processed.
    real_style_report_entries(result) -> list[{"path": str, "sha256": str}]     # never content
    load_forgery(folder, notify=print) -> ForgeryDataset
        # reads <folder>/labels.csv (columns file, split, is_tampered); .status "ok" | "missing";
        # .docs: tuple[ForgeryDoc(path: Path, split: str, is_tampered: bool)];
        # .dataset_docs(dataset_id) -> tuple[ForgeryDoc]; .dataset_status() -> {id: "ok" | "missing"}
        # for "forgery_test" and "forgery_ood". Missing/unreadable folder or labels.csv -> notify + "missing".
    read_answer_key(path) -> dict[str, bool]   # JSON object {name: {"is_tampered": bool, ...}} keyed by file stem
    normalise_path(p) -> Path                  # expanduser + absolute + normpath (dot segments, //, trailing /)
    assert_not_in_training(paths, *, protected) -> None
        # raises RealStyleInTraining(offenders: tuple[Path]) iff normalise_path(p) is in normalised protected
    build_training_lists(corpus_paths, *, forgery=None, train_on_forgery=False) -> TrainingLists(train, validation)
        # tuples of absolute Paths. Generator-corpus mode: split "train" / "validation" files from labels.csv
        # (column "file", relative to the corpus folder), zero forgery documents. Forgery mode: only the
        # forgery "train" Split for train and "val" for validation.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from fraud_detect.ml.retraining import sources as src
from fraud_detect.ml.retraining.corpora import load_corpus
from fraud_detect.ml.retraining.safe_io import SourceChanged, sha256_file
from fraud_detect.ml.retraining.sources import (
    DATASETS,
    FORGERY_SPLIT_DATASETS,
    RealStyleInTraining,
    assert_not_in_training,
    build_training_lists,
    load_forgery,
    load_real_style,
    normalise_path,
    open_source,
    read_answer_key,
    real_style_report_entries,
)

FAKE_NAMES = ("fake_style_1.pdf", "fake_style_2.pdf")
PKG = Path(src.__file__).parent


def snapshot(folder: Path) -> dict:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, sha256_file(p))
            for p in sorted(folder.iterdir()) if p.is_file()}


class Notes(list):
    def __call__(self, message):
        self.append(str(message))


@pytest.fixture
def forgery_dir(tmp_path, fake_real_style_dir):
    folder = tmp_path / "fake_forgery"
    folder.mkdir()
    rows = [("f_train_1.pdf", "train", 0), ("f_train_2.pdf", "train", 1),
            ("f_val_1.pdf", "val", 0), ("f_in_1.pdf", "test_in_layout", 1),
            ("f_in_2.pdf", "test_in_layout", 0), ("f_ood_1.pdf", "test_ood_layout", 1)]
    lines = ["file,split,is_tampered"]
    for name, split, tam in rows:
        (folder / name).write_bytes(b"%PDF-1.4 fake " + name.encode())
        lines.append(f"{name},{split},{tam}")
    (folder / "labels.csv").write_text("\n".join(lines) + "\n")
    return folder


# ---- registry ---------------------------------------------------------------------------------

def test_seven_datasets_and_their_sources():
    assert DATASETS == {
        "test": "generator_corpus", "heldout_layout": "generator_corpus",
        "heldout_seed": "generator_corpus", "forgery_test": "forgery_dataset",
        "forgery_ood": "forgery_dataset", "real_style": "real_style", "awkward": "awkward_set"}
    assert FORGERY_SPLIT_DATASETS == {"test_in_layout": "forgery_test", "test_ood_layout": "forgery_ood"}


# ---- read-only access -------------------------------------------------------------------------

def test_open_source_reads_rb_only_hashes_and_gives_bytesio(fake_real_style_dir, monkeypatch):
    target = fake_real_style_dir / FAKE_NAMES[0]
    modes = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **kw):
        if Path(str(file)).parent == fake_real_style_dir:
            modes.append(mode)
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    seen = []
    result = open_source(target, lambda buf: seen.append(buf) or len(buf.getvalue()))
    monkeypatch.undo()

    assert modes and all(m == "rb" for m in modes)
    assert isinstance(seen[0], io.BytesIO)
    assert result.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert result.value == target.stat().st_size == result.size
    assert result.path == target


def test_open_source_leaves_folder_untouched(fake_real_style_dir):
    before = snapshot(fake_real_style_dir)
    for name in FAKE_NAMES:
        open_source(fake_real_style_dir / name, lambda buf: buf.read())
    assert snapshot(fake_real_style_dir) == before


def test_changed_source_fails_the_run(fake_real_style_dir):
    target = fake_real_style_dir / FAKE_NAMES[0]

    def tamper(buf):
        with open(target, "ab") as fh:     # the test itself simulates a concurrent writer
            fh.write(b"x")
        return None

    with pytest.raises(SourceChanged):
        open_source(target, tamper)


# ---- Real_Style -------------------------------------------------------------------------------

def test_load_real_style_reads_all_present_files(fake_real_style_dir):
    before = snapshot(fake_real_style_dir)
    notes = Notes()
    res = load_real_style(fake_real_style_dir, names=FAKE_NAMES,
                          consumer=lambda buf: len(buf.getvalue()), notify=notes)
    assert res.status == "ok" and res.missing == () and not notes
    assert set(res.files) == set(FAKE_NAMES)
    assert all(res.files[n].sha256 == sha256_file(fake_real_style_dir / n) for n in FAKE_NAMES)
    assert snapshot(fake_real_style_dir) == before


def test_missing_real_style_file_is_skipped_named_and_marked_missing(fake_real_style_dir):
    notes = Notes()
    res = load_real_style(fake_real_style_dir, names=(FAKE_NAMES[0], "statement_gone.pdf"),
                          consumer=lambda buf: 1, notify=notes)
    assert res.missing == ("statement_gone.pdf",)
    assert res.status == "missing"
    assert list(res.files) == [FAKE_NAMES[0]]          # the rest is still processed
    assert any("statement_gone.pdf" in m for m in notes)


def test_missing_real_style_folder_does_not_abort(tmp_path):
    notes = Notes()
    res = load_real_style(tmp_path / "nope", names=FAKE_NAMES, consumer=lambda b: 1, notify=notes)
    assert res.status == "missing" and set(res.missing) == set(FAKE_NAMES) and res.files == {}
    assert len(notes) >= 2 and not (tmp_path / "nope").exists()   # nothing created


def test_report_records_path_and_sha256_never_content(fake_real_style_dir):
    res = load_real_style(fake_real_style_dir, names=FAKE_NAMES,
                          consumer=lambda buf: buf.getvalue(), notify=Notes())
    entries = real_style_report_entries(res)
    assert [set(e) for e in entries] == [{"path", "sha256"}] * 2
    for e in entries:
        p = Path(e["path"])
        assert p.name in FAKE_NAMES and e["sha256"] == sha256_file(p)
        assert isinstance(e["path"], str) and len(e["sha256"]) == 64
    json.dumps(entries)                                      # plain data


# ---- Forgery_Dataset --------------------------------------------------------------------------

def test_forgery_splits_map_to_dataset_ids(forgery_dir):
    ds = load_forgery(forgery_dir, notify=Notes())
    assert ds.status == "ok"
    assert {d.path.name for d in ds.dataset_docs("forgery_test")} == {"f_in_1.pdf", "f_in_2.pdf"}
    assert {d.path.name for d in ds.dataset_docs("forgery_ood")} == {"f_ood_1.pdf"}
    assert ds.dataset_status() == {"forgery_test": "ok", "forgery_ood": "ok"}
    assert all(d.path.parent == forgery_dir for d in ds.docs)
    assert {d.path.name: d.is_tampered for d in ds.docs}["f_ood_1.pdf"] is True


def test_missing_forgery_dataset_prints_notice_and_marks_missing(tmp_path):
    notes = Notes()
    ds = load_forgery(tmp_path / "absent", notify=notes)
    assert ds.status == "missing" and ds.docs == ()
    assert ds.dataset_status() == {"forgery_test": "missing", "forgery_ood": "missing"}
    assert notes and not (tmp_path / "absent").exists()


def test_unreadable_forgery_labels_are_marked_missing(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "labels.csv").write_bytes(b"\xff\xfe\x00 not,a,label table\n\x00\x01")
    notes = Notes()
    ds = load_forgery(folder, notify=notes)
    assert ds.status == "missing" and notes
    assert ds.dataset_status() == {"forgery_test": "missing", "forgery_ood": "missing"}


def test_forgery_documents_are_read_only(forgery_dir):
    before = snapshot(forgery_dir)
    ds = load_forgery(forgery_dir, notify=Notes())
    for doc in ds.docs:
        r = open_source(doc.path, lambda buf: isinstance(buf, io.BytesIO))
        assert r.value is True
    assert snapshot(forgery_dir) == before


# ---- answer key -------------------------------------------------------------------------------

def test_answer_key_reader_returns_is_tampered_per_file(tmp_path):
    key = tmp_path / "key.json"
    key.write_text(json.dumps({
        "fake_style_1": {"is_tampered": True, "company": "X"},
        "fake_style_2.pdf": {"is_tampered": False, "company": "Y"}}))
    assert read_answer_key(key) == {"fake_style_1": True, "fake_style_2": False}


# ---- training-list guard ----------------------------------------------------------------------

@pytest.fixture
def protected(fake_real_style_dir):
    return [fake_real_style_dir / n for n in FAKE_NAMES]


def _spellings(path: Path, cwd_rel_base: Path):
    d, n = path.parent, path.name
    return [
        str(d / "." / n),
        str(d) + "//" + n,
        str(d / "sub" / ".." / n),
        str(path) + os.sep,
        os.path.relpath(path, cwd_rel_base),
        path,
    ]


def test_every_spelling_of_a_protected_path_is_detected(protected, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for target in protected:
        for spelling in _spellings(target, tmp_path):
            with pytest.raises(RealStyleInTraining) as err:
                assert_not_in_training(["unrelated/a.pdf", spelling], protected=protected)
            assert normalise_path(spelling) == normalise_path(target)
            assert str(target) in str(err.value)
            assert [normalise_path(p) for p in err.value.offenders] == [normalise_path(target)]


def test_other_paths_are_not_detected(protected, tmp_path):
    other = protected[0].parent / "fake_style_3.pdf"
    elsewhere = tmp_path / "elsewhere" / FAKE_NAMES[0]          # same name, other folder
    assert_not_in_training([other, elsewhere, str(protected[0]) + "x"], protected=protected)
    assert_not_in_training([], protected=protected)


def test_normalise_path_is_absolute_and_collapses_spelling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert normalise_path("a/./b//c/../d/") == tmp_path / "a" / "b" / "d"
    assert normalise_path(tmp_path / "x").is_absolute()


def test_detection_aborts_before_fit_and_leaves_models_and_reports_unchanged(
        protected, fake_current_models, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("previous report")
    before_models, before_report = snapshot(fake_current_models), snapshot(tmp_path)["report.md"]
    fitted = []

    def fit(train, valid):                                  # stands in for any trainer
        assert_not_in_training(list(train) + list(valid), protected=protected)
        fitted.append(1)

    with pytest.raises(RealStyleInTraining):
        fit(["ok.pdf"], [str(protected[1])])
    assert fitted == []
    assert snapshot(fake_current_models) == before_models
    assert snapshot(tmp_path)["report.md"] == before_report


def test_generator_corpus_lists_contain_no_forgery_or_real_style(mini_corpus, forgery_dir, protected):
    paths = [mini_corpus.path(s) for s in mini_corpus.seeds]
    lists = build_training_lists(paths, forgery=load_forgery(forgery_dir, notify=Notes()))
    assert lists.train and lists.validation
    for p in (*lists.train, *lists.validation):
        assert Path(p).is_absolute()
        assert forgery_dir not in Path(p).parents
    assert_not_in_training([*lists.train, *lists.validation], protected=protected)


def test_generator_lists_follow_the_split_column(mini_corpus):
    seed = mini_corpus.seeds[0]
    corpus = load_corpus(mini_corpus.path(seed))
    lists = build_training_lists([mini_corpus.path(seed)])
    names = {s: {Path(p).stem for p in lst} for s, lst in
             (("train", lists.train), ("validation", lists.validation))}
    for split in ("train", "validation"):
        assert names[split] == {d.doc_id for d in corpus.docs if d.split == split}
    assert not names["train"] & names["validation"]


def test_forgery_training_mode_uses_only_train_and_val_splits(mini_corpus, forgery_dir):
    ds = load_forgery(forgery_dir, notify=Notes())
    lists = build_training_lists([mini_corpus.path()], forgery=ds, train_on_forgery=True)
    assert {Path(p).name for p in lists.train} == {"f_train_1.pdf", "f_train_2.pdf"}
    assert {Path(p).name for p in lists.validation} == {"f_val_1.pdf"}
    held = {d.path.name for d in ds.dataset_docs("forgery_test") + ds.dataset_docs("forgery_ood")}
    assert not held & {Path(p).name for p in (*lists.train, *lists.validation)}


def test_training_modules_do_not_import_sources():
    offenders = []
    for path in sorted(PKG.glob("*.py")):
        if not (path.name.startswith(("train", "fit", "candidate", "features_"))):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
                if (node.module or "").split(".")[-1] == "sources" or (
                        node.level and "sources" in names):
                    offenders.append(path.name)
            elif isinstance(node, ast.Import):
                if any(a.name.split(".")[-1] == "sources" for a in node.names):
                    offenders.append(path.name)
    assert offenders == []


# ---- real files (skipped unless the env vars point at them) -----------------------------------

@pytest.mark.skip_without_real_style
def test_real_style_folder_unchanged_after_reading():
    folder = Path(os.environ["RETRAIN_REAL_STYLE_DIR"])
    before = snapshot(folder)
    res = load_real_style(folder, consumer=lambda buf: len(buf.getvalue()), notify=Notes())
    assert res.files
    assert snapshot(folder) == before
    assert {e["sha256"] for e in real_style_report_entries(res)} <= {v[2] for v in before.values()}


def test_forgery_folder_unchanged_after_reading(forgery_dir_or_skip):
    before = snapshot(forgery_dir_or_skip)
    ds = load_forgery(forgery_dir_or_skip, notify=Notes())
    for doc in ds.docs[:5]:
        open_source(doc.path, lambda buf: len(buf.getvalue()))
    assert snapshot(forgery_dir_or_skip) == before
