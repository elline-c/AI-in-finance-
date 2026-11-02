"""Command line: build a synthetic corpus.

    python -m fraud_detect.corpus.build --out data/corpus --per-layout-genuine 2 \\
        --per-layout-tampered 2 --months-min 6 --months-max 12 --seed 20260601

Exit codes: 0 all documents built, 1 some documents failed, 2 refused (bad settings,
non-empty folder without --force, folder inside a protected tree, or a UEN check that
could not be completed: bad --uen-snapshot file, or --check-live-uen with no answer). A refusal writes nothing.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from fraud_detect.corpus.builder import BuildConfig, CorpusBuilder
from fraud_detect.corpus.schema import LAYOUTS, TAMPER_TYPES
from fraud_detect.corpus.uen_check import UenCheckError

DEFAULT_OUT = "data/corpus"
DEFAULT_SEED = 20260601


def _parse_mix(text: str) -> tuple[tuple[str, float], ...]:
    pairs = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, weight = part.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected type=weight, got {part!r}")
        try:
            pairs.append((name.strip(), float(weight)))
        except ValueError:
            raise argparse.ArgumentTypeError(f"weight in {part!r} is not a number") from None
    if not pairs:
        raise argparse.ArgumentTypeError("tamper mix is empty")
    return tuple(pairs)


def _parse_fractions(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("fractions must be numbers") from None
    if len(values) != 3:
        raise argparse.ArgumentTypeError("give three fractions: train,validation,test")
    return values  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fraud_detect.corpus.build",
        description="Build a synthetic bank-statement corpus (fictional data only).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"output folder (default: {DEFAULT_OUT}, which is git-ignored)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"master seed (default: {DEFAULT_SEED})")
    p.add_argument("--per-layout-genuine", type=int, default=2,
                   help="genuine documents per layout (default: 2)")
    p.add_argument("--per-layout-tampered", type=int, default=2,
                   help="tampered documents per layout (default: 2)")
    p.add_argument("--months-min", type=int, default=None, help="fewest months per statement (1-24)")
    p.add_argument("--months-max", type=int, default=None, help="most months per statement (1-24)")
    p.add_argument("--hold-out-layout", default=None, metavar="LAYOUT",
                   help="layout kept for the test_unseen_layout split; one of: " + ", ".join(LAYOUTS))
    p.add_argument("--tamper-mix", type=_parse_mix, default=None, metavar="TYPE=WEIGHT,...",
                   help="tamper types by weight, e.g. metadata_inconsistency=1,font_swap=2 "
                        "(default: even). Types: " + ", ".join(TAMPER_TYPES))
    p.add_argument("--split-fractions", type=_parse_fractions, default=(0.70, 0.15, 0.15),
                   metavar="TRAIN,VAL,TEST", help="split fractions, summing to 1 (default: 0.70,0.15,0.15)")
    p.add_argument("--force", action="store_true",
                   help="build into a non-empty folder (existing files are never deleted)")
    p.add_argument("--uen-snapshot", default=None, metavar="PATH",
                   help="file of real UENs (text, CSV with a 'uen' column, or JSON list) that "
                        "invented UENs must not equal")
    p.add_argument("--check-live-uen", action="store_true",
                   help="also ask the live data.gov.sg lookup about each invented UEN "
                        "(off by default; needs the network)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(args_list)  # --help / usage errors exit via SystemExit
    cfg = BuildConfig(
        out_dir=Path(args.out), master_seed=args.seed,
        per_layout_genuine=args.per_layout_genuine, per_layout_tampered=args.per_layout_tampered,
        months_min=args.months_min, months_max=args.months_max,
        hold_out_layout=args.hold_out_layout, tamper_mix=args.tamper_mix,
        split_fractions=args.split_fractions, force=args.force,
        command_line="python -m fraud_detect.corpus.build " + shlex.join(args_list),
        uen_snapshot=Path(args.uen_snapshot) if args.uen_snapshot else None,
        check_live_uen=args.check_live_uen)
    try:
        builder = CorpusBuilder(cfg)
        builder._check_out_dir()  # refuse before anything is written
    except ValueError as exc:
        reason = str(exc).replace("use force to", "use --force to")
        print(f"Refused: {reason}", file=sys.stderr)
        return 2

    def progress(i: int, total: int, plan) -> None:
        print(f"[{i}/{total}] {plan.doc_id}", file=sys.stderr)

    try:
        summary = builder.build(progress)
    except UenCheckError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(f"Built {summary.built} of {summary.planned} documents in {summary.out_dir}")
    for w in summary.warnings:
        print(f"Warning: {w}")
    for f in summary.failed:
        print(f"Failed: {f['doc_id']}: {f['error_type']}: {f['message']}", file=sys.stderr)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
