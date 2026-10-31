"""Failing tests (task 12.1) for the fabrication evaluation script (Req 12, 13, 16, 17).

``fraud_detect/nodes/fabrication_eval.py`` is written in task 12.2. Until then this
file fails at import with ModuleNotFoundError.

Testable API that task 12.2 must provide
----------------------------------------
``main(argv=None, *, dataset_dir=None, statements_dir=None, out=None,
screen=None, generator_screen=None) -> int``
    * ``argv``: command-line words; ``None`` means ``sys.argv[1:]``. Options:
      ``--sample N`` (positive int; genuine dataset documents used, in label
      order; default: all genuine), ``--dataset DIR``, ``--statements DIR``,
      ``--generator-count N`` (int >= 0, default 30; 0 skips the generator
      section). The keyword arguments, when not None, win over the options.
    * ``out``: text file-like for the report (default ``sys.stdout``).
    * ``screen(path: pathlib.Path) -> dict``: injectable stand-in for
      ingest + ``financial_logic.analyze`` + ``risk_synthesis.synthesize``. The
      default does exactly that, in memory, and never builds a CaseStore.
      Returned dict (every key optional; missing counts as empty/zero/False):
        ``finding_codes`` (list[str]), ``risk_score`` (int), ``risk_band`` (str),
        ``routing`` (str), ``has_invoice_ref`` (bool: some credit carries an
        invoice reference), ``metadata`` (dict with optional "Producer",
        "Creator", "CreationDate", "ModDate"; absent/None/"" = not present).
    * ``generator_screen(count: int) -> list[dict]``: injectable; returns one
      result dict (same shape as ``screen``) per generator statement. The
      default writes ``fraud_detect.synthetic.generate_clean`` statements to a
      temporary folder, screens them with the default ``screen`` and removes it.
    * Returns the exit code. Argument errors (argparse SystemExit) are caught and
      their code returned (2); nothing is raised. Missing inputs give 0.
    * The dataset's ``genuine_resaved`` column of labels.csv (not the screen
      result) marks re-saved documents.

Pure helpers (all in ``fraud_detect.nodes.fabrication_eval``)
    ``RULE_CODES`` = ("FL_DUP_INVOICE_REF", "FL_INVOICE_SEQUENCE_BREAK",
        "FL_TURNOVER_COST_MISMATCH", "DQ_RULE_SKIPPED")
    ``count_rule_firings(results, rule_codes=RULE_CODES) -> dict[str, int]``
        number of results whose finding_codes contain each code (once per
        result); every rule_code is a key, zero rows included, in order.
    ``invoice_ref_share(results) -> float``: fraction with has_invoice_ref
        (0.0 for an empty list).
    ``count_code(results, code) -> int``: results whose finding_codes hold code.
    ``format_firing_table(label, counts, total) -> list[str]``: one line per
        rule ``"<label> firing <CODE>: <n> of <total>"``, in counts order.
    ``resaved_metadata_counts(results) -> dict``: over the given results,
        ``{"documents": n, "Producer": {value: count}, "ModDate": n_present,
        "CreationDate": n_present}``.

Report lines the tests rely on
    dataset:    ``dataset firing <CODE>: <n> of <total>`` (every rule),
                ``dataset invoice reference share: <pct>%`` (one decimal),
                ``dataset VF_CURRENCY_MIX: <n>``,
                ``resaved documents: <n>``, ``resaved Producer <value>: <n>``,
                ``resaved ModDate present: <n>``, ``resaved CreationDate present: <n>``
    generator:  ``generator firing <CODE>: <n> of <total>`` (every rule)
    statements: ``metadata E Producer=<v> ...`` / ``metadata F ...`` (one line each,
                containing Producer, Creator, CreationDate, ModDate as ``Key=value``),
                ``E score=<int> band=<band> routing=<routing> codes=<a,b>`` / ``F ...``,
                ``E above 25 and above F: yes|no`` (a report, never an assertion).
    Missing dataset folder / labels.csv -> a line containing ``skipped`` (dataset
    section). Missing statement_E.pdf or statement_F.pdf (an answer key is not
    needed) -> a line containing ``skipped`` (statements section). Generator
    section skipped only when the count is 0.
    One line starting ``elapsed``; everything else is deterministic.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

import fraud_detect.nodes.fabrication_eval as fabrication_eval
from fraud_detect.nodes.fabrication_eval import (
    RULE_CODES, count_code, count_rule_firings, format_firing_table, invoice_ref_share,
    main, resaved_metadata_counts,
)
from tests.fabrication.conftest import FORGERY_DATASET, requires_dataset

REPO = Path(__file__).resolve().parents[2]
HEADER = "doc_id,file,is_tampered,genuine_resaved\n"
DUP, SEQ, TURN, SKIP = RULE_CODES


# ---------------------------------------------------------------------------
# In-code fixtures
# ---------------------------------------------------------------------------

def make_dataset(root: Path, genuine: int = 10, tampered: int = 2, resaved: int = 3):
    """labels.csv plus empty PDFs. First `resaved` genuine docs are re-saved."""
    (root / "pdfs").mkdir(parents=True)
    lines, stems = [], []
    for i in range(genuine + tampered):
        stem = f"DOC{i + 1:05d}"
        (root / "pdfs" / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
        is_t = 1 if i >= genuine else 0
        rs = 1 if (not is_t and i < resaved) else 0
        lines.append(f"{stem},pdfs/{stem}.pdf,{is_t},{rs}\n")
        if not is_t:
            stems.append(stem)
    (root / "labels.csv").write_text(HEADER + "".join(lines))
    return stems


def make_statements(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "statement_E.pdf").write_bytes(b"%PDF-1.4\n")
    (root / "statement_F.pdf").write_bytes(b"%PDF-1.4\n")


class FakeScreen:
    """Deterministic, records calls. Genuine docs 1,2 fire DUP; doc 2 also CURRENCY_MIX."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, path: Path) -> dict:
        self.calls.append(path.stem)
        if path.stem == "statement_E":
            return {"finding_codes": [DUP, SEQ, TURN, "VF_CURRENCY_MIX"], "risk_score": 55,
                    "risk_band": "HIGH", "routing": "MANUAL_REVIEW", "has_invoice_ref": True,
                    "metadata": {"Producer": "Acme PDF", "Creator": "Writer",
                                 "CreationDate": "D:20260101", "ModDate": "D:20260301"}}
        if path.stem == "statement_F":
            return {"finding_codes": ["VF_CURRENCY_MIX"], "risk_score": 26,
                    "risk_band": "LOW", "routing": "AUTO_CLEAR", "has_invoice_ref": True,
                    "metadata": {"Producer": "Acme PDF"}}
        n = int(path.stem[3:])
        codes = []
        if n in (1, 2):
            codes.append(DUP)
        if n == 2:
            codes.append("VF_CURRENCY_MIX")
        meta = {"Producer": "Exporter" if n % 2 else "Printer"}
        if n == 1:
            meta["ModDate"] = "D:20260301"
            meta["CreationDate"] = "D:20260101"
        if n == 3:
            meta["CreationDate"] = "D:20260101"
        return {"finding_codes": codes, "risk_score": 5, "risk_band": "LOW",
                "routing": "AUTO_CLEAR", "has_invoice_ref": n <= 4, "metadata": meta}


def fake_generator(count: int) -> list[dict]:
    return [{"finding_codes": [SEQ] if i == 0 else [], "risk_score": 3,
             "risk_band": "LOW", "routing": "AUTO_CLEAR"} for i in range(count)]


def run(argv=(), *, dataset_dir, statements_dir, screen=None, generator_screen=fake_generator):
    buf = io.StringIO()
    code = main(list(argv), dataset_dir=dataset_dir, statements_dir=statements_dir, out=buf,
                screen=screen if screen is not None else FakeScreen(),
                generator_screen=generator_screen)
    return code, buf.getvalue()


def strip_timing(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("elapsed"))


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
            for p in sorted(root.rglob("*"))}


@pytest.fixture
def ds(tmp_path):
    root = tmp_path / "ds"
    make_dataset(root)
    return root


@pytest.fixture
def stm(tmp_path):
    root = tmp_path / "stm"
    make_statements(root)
    return root


@pytest.fixture
def absent(tmp_path):
    return tmp_path / "absent"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_count_rule_firings_counts_documents_and_includes_zero_rows():
    results = [{"finding_codes": [DUP, DUP, "X"]}, {"finding_codes": [DUP, SEQ]},
               {"finding_codes": []}, {}]
    counts = count_rule_firings(results)
    assert list(counts) == list(RULE_CODES)
    assert counts == {DUP: 2, SEQ: 1, TURN: 0, SKIP: 0}


def test_count_rule_firings_on_empty_input_is_all_zero():
    assert count_rule_firings([]) == {c: 0 for c in RULE_CODES}


def test_invoice_ref_share_and_count_code():
    results = [{"has_invoice_ref": True, "finding_codes": ["VF_CURRENCY_MIX"]},
               {"has_invoice_ref": False}, {"has_invoice_ref": True},
               {"finding_codes": ["VF_CURRENCY_MIX", "VF_CURRENCY_MIX"]}]
    assert invoice_ref_share(results) == pytest.approx(0.5)
    assert invoice_ref_share([]) == 0.0
    assert count_code(results, "VF_CURRENCY_MIX") == 2


def test_format_firing_table_lines_in_order_with_zero_rows():
    lines = format_firing_table("dataset", {DUP: 2, SEQ: 0}, 10)
    assert lines == [f"dataset firing {DUP}: 2 of 10", f"dataset firing {SEQ}: 0 of 10"]


def test_resaved_metadata_counts():
    results = [{"metadata": {"Producer": "A", "ModDate": "D:1", "CreationDate": "D:0"}},
               {"metadata": {"Producer": "A", "ModDate": None, "CreationDate": ""}},
               {"metadata": {"Producer": "B", "CreationDate": "D:0"}}, {}]
    c = resaved_metadata_counts(results)
    assert c["documents"] == 4
    assert c["Producer"] == {"A": 2, "B": 1}
    assert c["ModDate"] == 1 and c["CreationDate"] == 2


# ---------------------------------------------------------------------------
# Dataset section
# ---------------------------------------------------------------------------

def test_dataset_counts_use_genuine_documents_only(ds, absent):
    fake = FakeScreen()
    code, text = run(["--generator-count", "0"], dataset_dir=ds, statements_dir=absent, screen=fake)
    assert code == 0
    assert len(fake.calls) == 10  # 2 tampered never screened
    assert f"dataset firing {DUP}: 2 of 10" in text
    assert "dataset VF_CURRENCY_MIX: 1" in text
    assert "dataset invoice reference share: 40.0%" in text


def test_false_alarm_table_includes_zero_rows(ds, absent):
    _, text = run(["--generator-count", "0"], dataset_dir=ds, statements_dir=absent)
    assert f"dataset firing {SEQ}: 0 of 10" in text
    assert f"dataset firing {TURN}: 0 of 10" in text
    assert f"dataset firing {SKIP}: 0 of 10" in text


def test_sample_option_limits_documents_in_label_order(ds, absent):
    fake = FakeScreen()
    _, text = run(["--sample", "3", "--generator-count", "0"], dataset_dir=ds,
                  statements_dir=absent, screen=fake)
    assert fake.calls == ["DOC00001", "DOC00002", "DOC00003"]
    assert f"dataset firing {DUP}: 2 of 3" in text


def test_resaved_metadata_section(ds, absent):
    _, text = run(["--generator-count", "0"], dataset_dir=ds, statements_dir=absent)
    assert "resaved documents: 3" in text
    assert "resaved Producer Exporter: 2" in text  # docs 1 and 3
    assert "resaved Producer Printer: 1" in text   # doc 2
    assert "resaved ModDate present: 1" in text
    assert "resaved CreationDate present: 2" in text


def test_dataset_options_on_the_command_line(ds, stm):
    buf = io.StringIO()
    code = main(["--dataset", str(ds), "--statements", str(stm), "--generator-count", "0",
                 "--sample", "2"], out=buf, screen=FakeScreen(), generator_screen=fake_generator)
    assert code == 0
    assert f"dataset firing {DUP}: 2 of 2" in buf.getvalue()
    assert "E score=55" in buf.getvalue()


# ---------------------------------------------------------------------------
# Generator section
# ---------------------------------------------------------------------------

def test_generator_counts_and_zero_rows(absent):
    _, text = run(["--generator-count", "4"], dataset_dir=absent, statements_dir=absent)
    assert f"generator firing {SEQ}: 1 of 4" in text
    assert f"generator firing {DUP}: 0 of 4" in text


def test_generator_count_zero_skips_section(absent):
    _, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=absent)
    assert "generator firing" not in text


# ---------------------------------------------------------------------------
# Statements E and F (Req 17)
# ---------------------------------------------------------------------------

def test_statements_metadata_scores_and_target_yes(stm, absent):
    code, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=stm)
    assert code == 0
    lines = text.splitlines()
    e_meta = next(l for l in lines if l.startswith("metadata E"))
    for token in ("Producer=Acme PDF", "Creator=Writer", "CreationDate=D:20260101",
                  "ModDate=D:20260301"):
        assert token in e_meta
    assert any(l.startswith("metadata F") for l in lines)
    e = next(l for l in lines if l.startswith("E score="))
    assert "score=55" in e and "band=HIGH" in e and "routing=MANUAL_REVIEW" in e
    assert f"codes={DUP},{SEQ},{TURN},VF_CURRENCY_MIX" in e
    assert "F score=26 band=LOW routing=AUTO_CLEAR codes=VF_CURRENCY_MIX" in text
    assert "E above 25 and above F: yes" in text


def test_target_line_prints_no_without_failing(stm, absent):
    class Low(FakeScreen):
        def __call__(self, path):
            r = super().__call__(path)
            if path.stem == "statement_E":
                r = dict(r, risk_score=26)
            return r

    code, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=stm,
                     screen=Low())
    assert code == 0  # a reported target, not an assertion
    assert "E above 25 and above F: no" in text


def test_missing_statements_skip_with_exit_0(ds, absent):
    code, text = run(["--generator-count", "0"], dataset_dir=ds, statements_dir=absent)
    assert code == 0
    assert "skipped" in text.lower()
    assert "E score=" not in text
    assert "dataset firing" in text  # other sections still run


def test_one_missing_statement_skips_section(tmp_path, absent):
    s = tmp_path / "half"
    s.mkdir()
    (s / "statement_E.pdf").write_bytes(b"%PDF-1.4\n")
    code, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=s)
    assert code == 0 and "skipped" in text.lower()
    assert "E score=" not in text


# ---------------------------------------------------------------------------
# Missing dataset
# ---------------------------------------------------------------------------

def test_missing_dataset_folder_skips_with_exit_0(stm, absent):
    fake = FakeScreen()
    code, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=stm, screen=fake)
    assert code == 0
    assert "skipped" in text.lower()
    assert "dataset firing" not in text
    assert "E score=55" in text  # statements section still runs


def test_missing_labels_file_skips(tmp_path, absent):
    (tmp_path / "ds").mkdir()
    code, text = run(["--generator-count", "0"], dataset_dir=tmp_path / "ds", statements_dir=absent)
    assert code == 0 and "skipped" in text.lower()


def test_everything_missing_still_exits_0(absent):
    code, text = run(["--generator-count", "0"], dataset_dir=absent, statements_dir=absent)
    assert code == 0 and "skipped" in text.lower()


# ---------------------------------------------------------------------------
# Determinism and timing
# ---------------------------------------------------------------------------

def test_deterministic_output_excluding_elapsed(ds, stm):
    outs = [run(["--generator-count", "3"], dataset_dir=ds, statements_dir=stm)[1] for _ in range(2)]
    assert strip_timing(outs[0]) == strip_timing(outs[1])
    assert strip_timing(outs[0]).strip() != ""


def test_one_elapsed_line(ds, stm):
    _, text = run([], dataset_dir=ds, statements_dir=stm)
    assert len([l for l in text.splitlines() if l.startswith("elapsed")]) == 1


# ---------------------------------------------------------------------------
# Argument errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--sample", "abc"], ["--sample"], ["--sample", "0"], ["--sample", "-4"],
    ["--generator-count", "abc"], ["--generator-count", "-1"], ["--bogus"],
])
def test_argument_errors_return_nonzero_without_raising(argv, ds, stm):
    fake = FakeScreen()
    code, _ = run(argv, dataset_dir=ds, statements_dir=stm, screen=fake)
    assert code == 2
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Read-only and store isolation (Req 16.4)
# ---------------------------------------------------------------------------

def _data_state():
    data = REPO / "data"
    listing = sorted(str(p.relative_to(data)) for p in data.rglob("*")) if data.is_dir() else []
    hashes = {n: hashlib.sha256((data / n).read_bytes()).hexdigest()
              for n in ("cases.json", "audit_chain.jsonl") if (data / n).is_file()}
    return listing, hashes


def test_script_writes_nothing_outside_its_output(ds, stm, tmp_path):
    before_tmp, before_data = tree(tmp_path), _data_state()
    code, _ = run(["--generator-count", "2"], dataset_dir=ds, statements_dir=stm)
    assert code == 0
    assert tree(tmp_path) == before_tmp
    after_listing, after_hashes = _data_state()
    assert (after_listing, after_hashes) == before_data


def test_default_pipeline_never_constructs_a_case_store(monkeypatch, tmp_path, absent):
    import fraud_detect.store as store

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("CaseStore must not be constructed")

    monkeypatch.setattr(store, "CaseStore", Boom)
    before_tmp, before_data = tree(tmp_path), _data_state()
    buf = io.StringIO()
    # Real default screen on one real generator statement; no dataset or statements.
    code = main(["--generator-count", "1"], dataset_dir=absent, statements_dir=absent, out=buf)
    assert code == 0
    assert f"generator firing {DUP}: 0 of 1" in buf.getvalue()
    assert tree(tmp_path) == before_tmp
    assert _data_state() == before_data


def test_module_does_not_reference_case_store():
    src = Path(fabrication_eval.__file__).read_text()
    assert "CaseStore" not in src


# ---------------------------------------------------------------------------
# Optional: real dataset
# ---------------------------------------------------------------------------

@requires_dataset
def test_real_dataset_small_sample_runs(absent):
    buf = io.StringIO()
    code = main(["--sample", "2", "--generator-count", "0"], dataset_dir=FORGERY_DATASET,
                statements_dir=absent, out=buf)
    assert code == 0
    assert f"dataset firing {DUP}:" in buf.getvalue()
