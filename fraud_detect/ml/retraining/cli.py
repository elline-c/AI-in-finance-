"""Command line of the retraining pipeline.

Parsing, exit codes and the run lock live here. The work is done in ``runner`` (orchestration),
which is imported inside the command handlers so that ``--help`` stays fast and importing this
module imports no ``torch``, ``sklearn``, ``pandas`` or ``pdfplumber``.

Exit codes: 0 success, 1 other failure, 2 integrity gate / refused / bad arguments / missing run /
lock held / write guard, 3 waivable gate or Leak_Test suspect waiting for a decision,
4 implausible measurement.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional, Sequence

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTEGRITY = 2
EXIT_WAITING = 3
EXIT_IMPLAUSIBLE = 4

COMMANDS = {
    "init": "Create the run folder and configuration.",
    "build": "Build the synthetic training corpora.",
    "check": "Run the corpus integrity and composition gates.",
    "audit": "Write and validate the Feature_Audit.",
    "leak-test": "Run the Leak_Test on the training and validation Splits.",
    "train": "Train Candidate models (never touches models/).",
    "compare": "Compare Candidate and Current models and write the report.",
    "switch": "Switch to a Candidate model when the Switch_Rule allows it.",
    "restore": "Restore the backed-up models.",
    "decide": "Record an owner decision in the decision log.",
}

#: Commands that write into the run folder and therefore take ``run.lock``.
LOCKED = frozenset({"check", "leak-test", "train", "compare", "decide"})
#: ``decide`` kinds: (allowed decisions, default decision or None when it must be given).
DECIDE_KINDS = {
    "leak-feature": (("exclude", "accept"), None),
    "waive-gate": (("waive",), "waive"),
    "investigation": (("continue", "abort"), None),
    "awkward-review": (("reviewed",), "reviewed"),
}
DEFAULT_OUTPUT_ROOT = "data/private/ml_retraining"


class Stop(Exception):
    """A command ends early with an exit code and a message."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ------------------------------------------------------------------------------ parser
def _int_list(text: str):
    try:
        values = tuple(int(p) for p in text.split(",") if p.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of integers: {text!r}") from None
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is needed")
    return values


def _threshold(text: str):
    from .config import Thresholds
    name, sep, raw = text.partition("=")
    name = name.strip()
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    defaults = Thresholds()
    if not hasattr(defaults, name):
        raise argparse.ArgumentTypeError(f"unknown threshold {name!r}")
    try:
        return name, type(getattr(defaults, name))(raw.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad value for threshold {name}: {raw!r}") from None


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output-root", default=None, metavar="DIR",
                   help=f"output folder (default <repo>/{DEFAULT_OUTPUT_ROOT})")
    p.add_argument("--cache-dir", default=None, metavar="DIR",
                   help="feature cache folder (default <output-root>/cache)")
    return p


def _with_run() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, parents=[_common()])
    p.add_argument("--run-id", default=None, metavar="ID", help="run folder name under <output-root>/runs")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fraud_detect.ml.retraining",
        description="ML retraining pipeline (synthetic data only).",
        epilog="Exit codes: 0 success, 1 other failure, 2 integrity gate or bad arguments, "
               "3 waivable gate waiting for a decision, 4 implausible measurement.")
    sub = parser.add_subparsers(dest="command", metavar="command")
    common, with_run = _common(), _with_run()

    def add(name, parents):
        return sub.add_parser(name, help=COMMANDS[name], description=COMMANDS[name], parents=[parents])

    p = add("init", common)
    p.add_argument("--seeds", type=_int_list, default=None, help="training Master_Seeds, A,B,...")
    p.add_argument("--held-out-seed", type=int, default=None)
    p.add_argument("--held-out-layout", default=None)
    p.add_argument("--real-style-role", choices=("required", "veto_only"), default="required")
    p.add_argument("--rule-c-scope", choices=("all", "genuine_only"), default="all")
    p.add_argument("--target-fa-medium", type=float, default=None)
    p.add_argument("--target-fa-high", type=float, default=None)
    p.add_argument("--threshold", type=_threshold, action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--real-style-dir", default=None)
    p.add_argument("--forgery-dir", default=None)
    p.add_argument("--plumbing", action="store_true",
                   help="plumbing run: the coverage gate is reported but not enforced")

    p = add("build", with_run)
    p.add_argument("--per-layout-genuine", type=int, required=True)
    p.add_argument("--per-layout-tampered", type=int, required=True)
    p.add_argument("--months-min", type=int, default=None)
    p.add_argument("--months-max", type=int, default=None)

    p = add("check", with_run)
    p.add_argument("--plan", action="store_true", help="print the document counts and stop")
    p.add_argument("--per-layout-genuine", type=int, default=80)
    p.add_argument("--per-layout-tampered", type=int, default=80)
    p.add_argument("--seeds", type=_int_list, default=None)

    add("audit", with_run)
    add("leak-test", with_run)
    p = add("train", with_run)
    p.add_argument("--node", choices=("6a", "6b", "both"), default="both")
    add("compare", with_run)

    p = add("switch", with_run)
    p.add_argument("--node", choices=("6a", "6b"), required=True)
    p.add_argument("--size-guard", choices=("candidate", "off", "current"), default=None)
    p.add_argument("--override-reason", default=None)

    p = add("restore", with_run)
    p.add_argument("--node", choices=("6a", "6b"), required=True)

    p = add("decide", with_run)
    kinds = p.add_subparsers(dest="kind", metavar="KIND", required=True)
    for kind, (choices, default) in DECIDE_KINDS.items():
        k = kinds.add_parser(kind, help=f"record a {kind} decision", parents=[with_run])
        k.add_argument("--target", required=True)
        k.add_argument("--reason", required=True)
        k.add_argument("--author", required=True)
        k.add_argument("--decision", choices=choices, default=default, required=default is None)
    return parser


# ------------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None, *, is_ignored: Optional[Callable] = None,
         clock: Optional[Callable] = None, repo_root=None, models_dir=None, backup_dir=None,
         corpus_builder: Optional[Callable] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 0 for --help, 2 for bad arguments
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help()
        return EXIT_INTEGRITY
    try:
        from . import runner
        session = runner.Session.create(
            args, is_ignored=is_ignored, clock=clock, repo_root=repo_root, models_dir=models_dir,
            backup_dir=backup_dir, corpus_builder=corpus_builder)
        return _dispatch(args, session, runner)
    except Stop as stop:
        print(stop.message, file=sys.stderr)
        return stop.code
    except Exception as exc:  # mapped to documented exit codes; unknown ones are failures
        code = runner_exit_code(exc)
        if code is None:
            traceback.print_exc()
            print(f"{args.command}: failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        print(f"{args.command}: {exc}", file=sys.stderr)
        return code


def runner_exit_code(exc: Exception) -> Optional[int]:
    """Documented exit code for an exception, or ``None`` when it is not a documented stop."""
    from .corpora import CorpusError
    from .cutoffs import CutoffError
    from .decisions import DecisionError
    from .modelops import ModelOpsError, SizeGuardRequired
    from .monitor import MeasurementStop, StopLockActive
    from .safe_io import WriteGuardError
    from .vault import MissingClassError, VaultClosed

    if isinstance(exc, (WriteGuardError, CorpusError, CutoffError, DecisionError, VaultClosed,
                        MissingClassError, SizeGuardRequired)):
        return EXIT_INTEGRITY
    if isinstance(exc, (MeasurementStop, StopLockActive)):
        return EXIT_IMPLAUSIBLE
    if isinstance(exc, ModelOpsError):
        return EXIT_FAILURE
    code = getattr(exc, "exit_code", None)  # SkippedShareError, LeakTestError, TrainRefused, ...
    return code if isinstance(code, int) else None


@contextlib.contextmanager
def run_lock(session):
    """``<run>/run.lock`` exists while a command that writes into the run folder runs."""
    from .safe_io import ensure_ignored
    lock = session.run_dir / "run.lock"
    ensure_ignored(lock, is_ignored=session.guard)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise Stop(EXIT_INTEGRITY,
                   f"the run folder is locked ({lock}): another command is using it. "
                   f"If no command is running, remove the stale lock file.") from None
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(f"pid {os.getpid()}\n")
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()


def _dispatch(args, session, runner) -> int:
    name = args.command
    if name == "init":
        return runner.cmd_init(session, args)
    if name == "check" and args.plan:
        return runner.cmd_check_plan(args)
    session.load_run(args.run_id)
    handler = {
        "build": runner.cmd_build, "check": runner.cmd_check, "audit": runner.cmd_audit,
        "leak-test": runner.cmd_leak_test, "train": runner.cmd_train, "compare": runner.cmd_compare,
        "switch": runner.cmd_switch, "restore": runner.cmd_restore, "decide": runner.cmd_decide,
    }[name]
    if name in LOCKED:
        with run_lock(session):
            return handler(session, args)
    return handler(session, args)
