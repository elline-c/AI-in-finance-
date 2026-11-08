"""Golden_Hashes tool: records / collects hashes of Legacy_Styling corpus output.

Raw PDF bytes are not hashed (same notion as the determinism test in test_builder.py):
for each PDF the text-layer hash and the metadata + font names are recorded.

Usage (run on the UNCHANGED generator only):
    python -m tests.corpus.golden_tool --write tests/corpus/legacy_golden.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.labels import text_layer_sha256
from fraud_detect.corpus.schema import BuildConfig

HOLD_OUT_LAYOUT = "two_col_ddmmyy_cr"
MINI_TAMPER_MIX = (
    ("genuine_resaved", 1.0),
    ("metadata_inconsistency", 1.0),
    ("font_mismatch", 1.0),
    ("edited_amount", 1.0),
    ("fabricated_receipts", 1.0),
    ("structuring", 1.0),
)
# case name -> (master_seed, tamper_mix or None for the default mix)
CASES: dict = {
    "seed_20260801_default_mix": (20260801, None),
    "seed_20260802_six_type_mix": (20260802, MINI_TAMPER_MIX),
}
_META_KEYS = ("/Title", "/Subject", "/Author", "/Creator", "/Producer", "/CreationDate", "/ModDate")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _font_names(pdf_bytes: bytes) -> list:
    import io

    from pypdf import PdfReader
    names: set = set()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages:
        res = page.get("/Resources")
        res = res.get_object() if res is not None else {}
        fonts = res.get("/Font")
        if fonts is None:
            continue
        for ref in fonts.get_object().values():
            base = ref.get_object().get("/BaseFont")
            if base is not None:
                names.add(str(base))
    return sorted(names)


def pdf_record(pdf_bytes: bytes) -> dict:
    import io

    from pypdf import PdfReader
    meta = PdfReader(io.BytesIO(pdf_bytes)).metadata or {}
    canon = {k: (None if meta.get(k) is None else str(meta.get(k))) for k in _META_KEYS}
    return {"text_layer_sha256": text_layer_sha256(pdf_bytes),
            "metadata": canon, "fonts": _font_names(pdf_bytes)}


def collect(corpus_dir) -> dict:
    root = Path(corpus_dir)
    out: dict = {"files": {}, "manifests": {}, "pdfs": {}}
    for name in ("labels.csv", "rows.csv", "companies.json", "corpus_manifest.json"):
        out["files"][name] = _sha((root / name).read_bytes())
    for p in sorted((root / "manifests").glob("*.json")):
        out["manifests"][p.stem] = _sha(p.read_bytes())
    for p in sorted((root / "pdfs").glob("*.pdf")):
        out["pdfs"][p.stem] = pdf_record(p.read_bytes())
    return out


def build_case(name: str, out_dir) -> Path:
    seed, mix = CASES[name]
    kwargs = {} if mix is None else {"tamper_mix": mix}
    cfg = BuildConfig(
        out_dir=Path(out_dir), master_seed=seed, per_layout_genuine=2, per_layout_tampered=2,
        months_min=6, months_max=6, hold_out_layout=HOLD_OUT_LAYOUT,
        command_line="pytest mini_corpus", **kwargs)
    CorpusBuilder(cfg).build()
    return Path(out_dir)


def collect_all(work_dir) -> dict:
    return {name: collect(build_case(name, Path(work_dir) / name)) for name in CASES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", required=True, metavar="PATH")
    args = ap.parse_args(argv)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        cases = collect_all(tmp)
    doc = {"recorded_from": {"git_commit": commit, "date": date.today().isoformat()}, "cases": cases}
    Path(args.write).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
