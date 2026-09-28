"""Command-line runner for the StatementGuard prototype.

Usage:
    python3 -m statement_guard.cli <input.json> [<input2.json> ...]
    python3 -m statement_guard.cli --all
    python3 -m statement_guard.cli <input.json> --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import screen_statement
from .report import render

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _run_one(path: Path, json_out: Path | None) -> None:
    with path.open() as fh:
        raw = json.load(fh)
    report = screen_statement(raw)
    print(render(report))
    print(f"\n(source: {path.name})\n")
    if json_out:
        json_out.write_text(json.dumps(report, indent=2))
        print(f"Machine-readable report written to {json_out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Screen bank statements for tampering.")
    p.add_argument("inputs", nargs="*", help="Path(s) to extracted statement JSON")
    p.add_argument("--all", action="store_true", help="Run all bundled samples")
    p.add_argument("--json", type=Path, help="Also write machine-readable JSON report")
    args = p.parse_args(argv)

    paths: list[Path] = [Path(i) for i in args.inputs]
    if args.all:
        paths = sorted(SAMPLES_DIR.glob("*.json"))

    if not paths:
        p.error("provide at least one input JSON, or use --all")

    for path in paths:
        if not path.exists():
            print(f"! file not found: {path}", file=sys.stderr)
            continue
        _run_one(path, args.json if len(paths) == 1 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
