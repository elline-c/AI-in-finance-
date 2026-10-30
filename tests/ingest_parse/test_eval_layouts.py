"""Failing tests (task 12.1) for the evaluation script (Requirement 13).

The module ``fraud_detect/nodes/ingest_parse/eval_layouts.py`` is written in
task 12.2. Until then this file fails at import with ModuleNotFoundError.

Testable API that task 12.2 must provide
----------------------------------------
``main(argv=None, *, dataset_dir=None, statements_dir=None, out=None,
ingest=None) -> int``
    * ``argv``: command-line words (``--sample N``, ``--all-genuine``).
      ``None`` means ``sys.argv[1:]``.
    * ``dataset_dir``: folder holding ``labels.csv`` and the PDFs named in its
      ``file`` column (default ``~/Downloads/statement_forgery_dataset``).
    * ``statements_dir``: folder holding ``statement_E.pdf``,
      ``statement_F.pdf`` and ``ANSWER_KEY_6M_v2.json`` (default
      ``~/Downloads``).
    * ``out``: text file-like for the report (default ``sys.stdout``).
    * ``ingest``: optional callable ``ingest(path: pathlib.Path) -> dict``
      used instead of the real Node 1 + Node 3 pipeline, so unit tests need no
      PDFs. The default calls the real pipeline and returns the same shape.
    * Returns the exit code. Argument errors (argparse would raise SystemExit)
      are caught and their code returned (2); nothing is raised. Missing
      inputs give 0.

Shape of the dict returned by ``ingest`` (every key optional except where
noted; missing counts as empty/zero):
    ``profile`` (str), ``routing`` (str), ``finding_codes`` (list[str]),
    ``rows_expected`` (int), ``rows_read`` (int),
    ``balance_ok`` (bool: replayed balance equals printed closing),
    ``header_ok`` (bool: header fields match),
    ``monthly`` (dict "YYYY-MM" -> {"opening","total_deposits",
    "total_withdrawals","closing"} as floats; statement_E/F only),
    ``parsed_rows`` (list of {"month","date","amount"}; statement_E only).

``summarise_layout(results: list[dict]) -> dict``
    Pure. Returns ``{"profile", "documents", "rows_share", "balance_share",
    "header_share", "finding_codes"}``. ``profile`` is the first result's
    profile, ``documents`` the number of results, shares are fractions in
    [0, 1] (rows_share = sum rows_read / sum rows_expected, 1.0 if no rows
    expected), ``finding_codes`` a sorted list of unique codes.

Report text (what the tests rely on)
------------------------------------
* Dataset: one line per layout starting ``layout <n>:`` and containing
  ``profile=<name>``, ``documents=<count>``, ``rows=<pct>%``,
  ``balance=<pct>%``, ``header=<pct>%`` (percent with one decimal, e.g.
  ``100.0%``) and ``findings=<comma separated codes>``.
* ``--all-genuine`` adds a section whose title line contains
  ``false-alarm table``, with lines ``routing <OUTCOME>: <count>`` and
  ``finding <CODE>: <count>`` counted over all genuine documents.
* Statements: for each month of E (key ``monthly_as_shown``) and F (key
  ``monthly``) a line starting ``E <YYYY-MM>`` / ``F <YYYY-MM>`` ending in
  ``MATCH`` or ``DIFF`` (all four amounts within 0.01 of the key), and for E a
  line containing ``inserted rows found: <k> of <n>``.
* Any missing input prints a line containing ``skipped`` and does not change
  the exit code.
* Timing goes on a single line that starts with ``elapsed`` (longest file).
  Everything else is deterministic; tests strip lines starting ``elapsed``.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from fraud_detect.nodes.ingest_parse import eval_layouts
from fraud_detect.nodes.ingest_parse.eval_layouts import main, summarise_layout
from tests.ingest_parse.conftest import requires_dataset, DATASET_DIR


# ---------------------------------------------------------------------------
# Tiny in-code fixtures
# ---------------------------------------------------------------------------

HEADER = "doc_id,file,bank_layout,is_tampered\n"


def make_dataset(root: Path, genuine_per_layout: int = 25, tampered_per_layout: int = 3,
                 layouts=(1, 2)) -> dict:
    """Write labels.csv and empty PDFs. Returns {file stem: layout}."""
    (root / "pdfs").mkdir(parents=True)
    lines, stem_layout = [], {}
    n = 0
    for layout in layouts:
        for i in range(genuine_per_layout + tampered_per_layout):
            n += 1
            tampered = 1 if i >= genuine_per_layout else 0
            stem = f"DOC{n:05d}"
            (root / "pdfs" / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
            lines.append(f"{stem},pdfs/{stem}.pdf,{layout},{tampered}\n")
            stem_layout[stem] = layout
    (root / "labels.csv").write_text(HEADER + "".join(lines))
    return stem_layout


class FakeIngest:
    """Deterministic stand-in for the real pipeline; records every call."""

    def __init__(self, stem_layout: dict | None = None):
        self.stem_layout = stem_layout or {}
        self.calls: list[str] = []

    def __call__(self, path: Path) -> dict:
        self.calls.append(path.stem)
        if path.stem in ("statement_E", "statement_F"):
            return self._statement(path.stem)
        layout = self.stem_layout.get(path.stem, 1)
        return {
            "profile": f"profile_{layout}",
            "routing": "AUTO_CLEAR" if layout == 1 else "REVIEW",
            "finding_codes": ["FL_X"] if layout == 2 else [],
            "rows_expected": 10, "rows_read": 10,
            "balance_ok": True, "header_ok": True,
        }

    @staticmethod
    def _statement(stem: str) -> dict:
        month = {"opening": 100.0, "total_deposits": 50.0,
                 "total_withdrawals": 20.0, "closing": 130.0}
        res = {"profile": "profile_6", "routing": "REVIEW", "finding_codes": [],
               "rows_expected": 4, "rows_read": 4, "balance_ok": True,
               "header_ok": True, "monthly": {"2026-04": month}}
        if stem == "statement_E":
            res["parsed_rows"] = [
                {"month": "2026-04", "date": "2026-04-17", "amount": 8318.10},
                {"month": "2026-04", "date": "2026-04-20", "amount": 10.00},
            ]
        return res


def make_statements(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "statement_E.pdf").write_bytes(b"%PDF-1.4\n")
    (root / "statement_F.pdf").write_bytes(b"%PDF-1.4\n")
    month = {"month": "2026-04", "opening": "S$100.00", "total_deposits": "S$50.00",
             "total_withdrawals": "S$20.00", "closing": "S$130.00"}
    key = {
        "note": "fictional",
        "statement_E": {
            "monthly_as_shown": [month],
            "inserted_rows": [
                {"month": "2026-04", "date": "2026-04-17",
                 "fabricated_amount": "S$8,318.10"},
                {"month": "2026-04", "date": "2026-04-18",
                 "fabricated_amount": "S$999.00"},
            ],
        },
        "statement_F": {"monthly": [month]},
    }
    (root / "ANSWER_KEY_6M_v2.json").write_text(json.dumps(key))


def run(argv, *, dataset_dir, statements_dir, ingest):
    buf = io.StringIO()
    code = main(argv, dataset_dir=dataset_dir, statements_dir=statements_dir,
                out=buf, ingest=ingest)
    return code, buf.getvalue()


def strip_timing(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("elapsed"))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    stem_layout = make_dataset(root)
    return root, stem_layout


@pytest.fixture
def no_statements(tmp_path):
    d = tmp_path / "empty_statements"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# summarise_layout (pure)
# ---------------------------------------------------------------------------

def test_summarise_layout_shares_and_codes():
    results = [
        {"profile": "p", "rows_expected": 10, "rows_read": 10, "balance_ok": True,
         "header_ok": True, "finding_codes": ["B", "A"]},
        {"profile": "p", "rows_expected": 10, "rows_read": 5, "balance_ok": False,
         "header_ok": True, "finding_codes": ["A"]},
    ]
    s = summarise_layout(results)
    assert s["profile"] == "p"
    assert s["documents"] == 2
    assert s["rows_share"] == pytest.approx(0.75)
    assert s["balance_share"] == pytest.approx(0.5)
    assert s["header_share"] == pytest.approx(1.0)
    assert s["finding_codes"] == ["A", "B"]


# ---------------------------------------------------------------------------
# Req 13.4 sample size
# ---------------------------------------------------------------------------

def test_default_sample_is_20_genuine_per_layout(dataset, no_statements):
    root, stem_layout = dataset
    fake = FakeIngest(stem_layout)
    code, text = run([], dataset_dir=root, statements_dir=no_statements, ingest=fake)
    assert code == 0
    assert len(fake.calls) == 40  # 2 layouts x 20, tampered never used
    assert "documents=20" in text
    assert "documents=25" not in text


def test_sample_option_is_honoured(dataset, no_statements):
    root, stem_layout = dataset
    fake = FakeIngest(stem_layout)
    code, text = run(["--sample", "3"], dataset_dir=root,
                     statements_dir=no_statements, ingest=fake)
    assert code == 0
    assert len(fake.calls) == 6
    assert text.count("documents=3") == 2
    # the first genuine documents, in label order
    assert fake.calls[:3] == ["DOC00001", "DOC00002", "DOC00003"]


def test_only_genuine_documents_are_used(dataset, no_statements):
    root, stem_layout = dataset
    fake = FakeIngest(stem_layout)
    run(["--all-genuine"], dataset_dir=root, statements_dir=no_statements, ingest=fake)
    assert len(fake.calls) == 50  # all genuine, no tampered (6 of 56 excluded)


def test_per_layout_line_contents(dataset, no_statements):
    root, stem_layout = dataset
    _, text = run(["--sample", "2"], dataset_dir=root,
                  statements_dir=no_statements, ingest=FakeIngest(stem_layout))
    l1 = next(l for l in text.splitlines() if l.startswith("layout 1:"))
    l2 = next(l for l in text.splitlines() if l.startswith("layout 2:"))
    for token in ("profile=profile_1", "documents=2", "rows=100.0%",
                  "balance=100.0%", "header=100.0%"):
        assert token in l1
    assert "findings=FL_X" in l2


# ---------------------------------------------------------------------------
# Req 13.5 false-alarm table
# ---------------------------------------------------------------------------

def test_all_genuine_prints_false_alarm_table(dataset, no_statements):
    root, stem_layout = dataset
    _, text = run(["--all-genuine"], dataset_dir=root,
                  statements_dir=no_statements, ingest=FakeIngest(stem_layout))
    assert "false-alarm table" in text
    assert "routing AUTO_CLEAR: 25" in text
    assert "routing REVIEW: 25" in text
    assert "finding FL_X: 25" in text


def test_no_false_alarm_table_without_all_genuine(dataset, no_statements):
    root, stem_layout = dataset
    _, text = run([], dataset_dir=root, statements_dir=no_statements,
                  ingest=FakeIngest(stem_layout))
    assert "false-alarm table" not in text


# ---------------------------------------------------------------------------
# Req 13.3 statements E/F vs the answer key
# ---------------------------------------------------------------------------

def test_statement_comparison_and_inserted_rows(tmp_path):
    stm = tmp_path / "stm"
    make_statements(stm)
    code, text = run([], dataset_dir=tmp_path / "absent", statements_dir=stm,
                     ingest=FakeIngest())
    assert code == 0
    lines = text.splitlines()
    e = next(l for l in lines if l.startswith("E 2026-04"))
    f = next(l for l in lines if l.startswith("F 2026-04"))
    assert e.rstrip().endswith("MATCH") and f.rstrip().endswith("MATCH")
    assert "inserted rows found: 1 of 2" in text


def test_statement_month_mismatch_is_reported(tmp_path):
    stm = tmp_path / "stm"
    make_statements(stm)
    key_path = stm / "ANSWER_KEY_6M_v2.json"
    key = json.loads(key_path.read_text())
    key["statement_F"]["monthly"][0]["closing"] = "S$131.00"
    key_path.write_text(json.dumps(key))
    _, text = run([], dataset_dir=tmp_path / "absent", statements_dir=stm,
                  ingest=FakeIngest())
    f = next(l for l in text.splitlines() if l.startswith("F 2026-04"))
    assert f.rstrip().endswith("DIFF")


# ---------------------------------------------------------------------------
# Req 13.6 missing inputs
# ---------------------------------------------------------------------------

def test_missing_dataset_and_statements_skip_with_exit_0(tmp_path):
    fake = FakeIngest()
    code, text = run([], dataset_dir=tmp_path / "nope", statements_dir=tmp_path / "nada",
                     ingest=fake)
    assert code == 0
    assert "skipped" in text.lower()
    assert fake.calls == []


def test_missing_labels_file_skips(tmp_path):
    (tmp_path / "ds").mkdir()
    code, text = run([], dataset_dir=tmp_path / "ds", statements_dir=tmp_path / "x",
                     ingest=FakeIngest())
    assert code == 0 and "skipped" in text.lower()


def test_missing_answer_key_skips_statement_section(tmp_path):
    stm = tmp_path / "stm"
    make_statements(stm)
    (stm / "ANSWER_KEY_6M_v2.json").unlink()
    code, text = run([], dataset_dir=tmp_path / "absent", statements_dir=stm,
                     ingest=FakeIngest())
    assert code == 0
    assert "skipped" in text.lower()
    assert "E 2026-04" not in text


# ---------------------------------------------------------------------------
# Req 13.7 determinism and timing
# ---------------------------------------------------------------------------

def test_same_input_gives_identical_output_excluding_elapsed(dataset, tmp_path):
    root, stem_layout = dataset
    stm = tmp_path / "stm"
    make_statements(stm)
    outs = []
    for _ in range(2):
        _, text = run(["--all-genuine"], dataset_dir=root, statements_dir=stm,
                      ingest=FakeIngest(stem_layout))
        outs.append(text)
    assert strip_timing(outs[0]) == strip_timing(outs[1])
    assert strip_timing(outs[0]).strip() != ""


def test_reports_elapsed_time_on_its_own_line(dataset, no_statements):
    root, stem_layout = dataset
    _, text = run(["--sample", "1"], dataset_dir=root, statements_dir=no_statements,
                  ingest=FakeIngest(stem_layout))
    elapsed = [l for l in text.splitlines() if l.startswith("elapsed")]
    assert len(elapsed) == 1


# ---------------------------------------------------------------------------
# Argument errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--sample", "abc"], ["--sample"], ["--sample", "0"], ["--sample", "-4"],
    ["--bogus"],
])
def test_argument_errors_return_nonzero_without_raising(argv, dataset, no_statements):
    root, stem_layout = dataset
    fake = FakeIngest(stem_layout)
    code, _ = run(argv, dataset_dir=root, statements_dir=no_statements, ingest=fake)
    assert code != 0
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Optional: real dataset
# ---------------------------------------------------------------------------

@requires_dataset
def test_real_dataset_small_sample_runs(no_statements):
    buf = io.StringIO()
    code = main(["--sample", "1"], dataset_dir=DATASET_DIR,
                statements_dir=no_statements, out=buf)
    assert code == 0
    assert "layout 1:" in buf.getvalue()
