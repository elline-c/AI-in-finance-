"""verify handles the Extra_Tamper_Type ``subtle_amount_edit`` (task 8.2).

A tiny corpus with only this tamper in the mix is built and run through the real
``verify.main``. The type has no ``should_be_seen_by`` entry, so the expected-code
and expected-node columns must be ``None`` (the ``rapid_pass_through`` path).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from fraud_detect.corpus import verify
from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.schema import BuildConfig

NAME = "subtle_amount_edit"


def test_verify_reports_subtle_amount_edit(tmp_path: Path):
    corpus = tmp_path / "corpus"
    CorpusBuilder(BuildConfig(
        out_dir=corpus, master_seed=20260803, per_layout_genuine=1, per_layout_tampered=2,
        months_min=6, months_max=6, hold_out_layout="two_col_ddmmyy_cr",
        tamper_mix=((NAME, 1.0),), command_line="pytest verify subtle")).build()

    out = io.StringIO()
    report = tmp_path / "report.json"
    md = tmp_path / "report.md"
    assert verify.main(["--corpus", str(corpus), "--json", str(report),
                        "--markdown", str(md)], out=out) == 0

    data = json.loads(report.read_text("utf-8"))
    rows = [t for t in data["screening"]["tamper_types"] if t["tamper_type"] == NAME]
    assert rows, "no table row for subtle_amount_edit"
    row = rows[0]
    assert row["documents"] >= 1
    assert row["screening_errors"] == 0
    assert row["expected_codes"] == [] and row["expected_code_share"] is None
    assert row["expected_nodes"] == [] and row["expected_node_share"] is None
    assert row["method"] in {"overlay", "rerender"}
    assert f"tamper {NAME} " in out.getvalue()
    assert NAME in md.read_text("utf-8")
    assert not any(x["error"] for x in data["node1"]["documents"])
