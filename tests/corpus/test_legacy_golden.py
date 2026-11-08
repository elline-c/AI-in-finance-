"""Legacy_Styling must reproduce the recorded Golden_Hashes (Requirements 1.2, 8.5, 8.6).

Never regenerate legacy_golden.json to make this test pass; fix the generator change instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.corpus import golden_tool

GOLDEN = json.loads((Path(__file__).parent / "legacy_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", sorted(GOLDEN["cases"]))
def test_legacy_output_matches_golden_hashes(case, tmp_path):
    expected = GOLDEN["cases"][case]
    actual = golden_tool.collect(golden_tool.build_case(case, tmp_path / case))

    for name, digest in expected["files"].items():
        assert actual["files"].get(name) == digest, f"{case}: file {name} differs"
    assert set(actual["manifests"]) == set(expected["manifests"]), f"{case}: manifest set differs"
    for doc_id, digest in expected["manifests"].items():
        assert actual["manifests"][doc_id] == digest, f"{case}: manifest of document {doc_id} differs"
    assert set(actual["pdfs"]) == set(expected["pdfs"]), f"{case}: PDF set differs"
    for doc_id, rec in expected["pdfs"].items():
        got = actual["pdfs"][doc_id]
        assert got["text_layer_sha256"] == rec["text_layer_sha256"], \
            f"{case}: text layer of document {doc_id} differs"
        assert got["metadata"] == rec["metadata"], f"{case}: PDF metadata of document {doc_id} differs"
        assert got["fonts"] == rec["fonts"], f"{case}: PDF fonts of document {doc_id} differs"
