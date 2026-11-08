"""Task 23.1: failing tests for the retraining command line (``fraud_detect.ml.retraining.cli``).

Written before the implementation (Task 23.2); against the ten-command stub they fail.
Everything runs on the Mini_Corpus (copied into pytest temporary folders), fake Current model
files in a temporary ``models`` folder and a temporary backup folder. The real ``models/``
folder, ``data/`` folder, Real_Style files and Forgery_Dataset are never read or written
(Requirement 17.9); the Real_Style and Forgery folders passed to ``init`` do not exist, so
those Sources are "missing" and the run continues.

PINNED COMMAND-LINE INTERFACE (what 23.2 must implement)
=========================================================
Entry point
    main(argv, *, is_ignored=None, clock=None, repo_root=None, models_dir=None,
         backup_dir=None, corpus_builder=None) -> int
        is_ignored     write-guard predicate, passed to every write (default: git check-ignore)
        clock          callable returning an aware datetime (decision dates, report dates)
        repo_root      folder whose source files feed the code fingerprint (default: repo root)
        models_dir     folder holding the Current models (default ``models``); never written
                       except by ``switch`` and ``restore``
        backup_dir     backup folder for switch/restore (default ``<models_dir>/backup``)
        corpus_builder callable(BuildConfig) -> anything; default builds with CorpusBuilder
    Exit codes: 0 ok, 1 other failure (e.g. restore without a backup, backup/hash failure),
    2 integrity gate / refused / bad arguments / missing run / lock held / write guard,
    3 waivable gate or Leak_Test suspect waiting for a decision, 4 implausible measurement.
    ``<cmd> --help`` exits 0 and prints the command name. Output goes to stdout/stderr.

Options accepted by EVERY command (also ``check --plan`` and each ``decide KIND``)
    --output-root DIR   default ``<repo_root>/data/private/ml_retraining``
    --cache-dir DIR     default ``<output-root>/cache`` (feature caches)
Options accepted by every command except ``init``
    --run-id ID         folder is ``<output-root>/runs/<ID>``; a missing run folder or run.json
                        exits 2 with a message naming run.json (``check --plan`` needs no run)

Commands
    init    --seeds A,B,..  --held-out-seed N  --held-out-layout NAME
            --real-style-role {required,veto_only}   (open decision 1, default required)
            --rule-c-scope {all,genuine_only}        (open decision 4, default all)
            --target-fa-medium F  --target-fa-high F (open decision 7, defaults 5.0 / 1.0)
            --threshold NAME=VALUE (repeatable; any ``Thresholds`` field)
            --real-style-dir DIR  --forgery-dir DIR  --plumbing
        ``--plumbing`` marks a plumbing run: the non-waivable coverage gate (Requirement 6.2)
        is reported but not enforced, because a Mini_Corpus cannot cover 16 types x 3 Splits.
        Writes ``<run>/run.json`` (vault.write_run_json layout: config (canonical JSON string,
        including real_style_role, rule_c_scope, plumbing), switch_rule_sha256, cutoff_rule,
        code_fingerprint = vault.code_fingerprint(repo_root), plus switch_rule_text) and the
        report header ``<run>/comparison_report.md`` (banner in the first 10 lines, Switch_Rule
        text and hash, cutoff rule). Prints ``run_id: <id>`` and ``run.json sha256: <hex>``.
    build   --per-layout-genuine N  --per-layout-tampered N  [--months-min N --months-max N]
            Calls corpus_builder(BuildConfig) once per seed in the run config and once for the
            held-out seed, all with the run's hold_out_layout, out_dir
            ``<output-root>/corpora/seed-<n>`` and ``<output-root>/corpora/heldout-seed-<n>``.
    check   [--plan]  [--per-layout-genuine N --per-layout-tampered N --seeds A,B,..]
            ``--plan`` prints lines (no run needed; defaults 80/80 and 5 seeds + held-out):
              ``documents per seed: <n>``, ``corpora: <k>``, ``documents in total: <n>``,
              ``genuine_resaved per seed: <n>``, ``other tamper types per seed: <n>``
            (6 layouts x (genuine + tampered); resaved = tampered*6*10/26, each of the other 16
            types = tampered*6/26, rounded down). Without ``--plan``: loads the corpora from
            ``<output-root>/corpora``, runs the integrity gates, writes ``<run>/gates.json``,
            prints every failing gate with its offenders; exit 0 or 2. Waivable gates are
            printed but do not fail ``check``.
    audit   writes ``<output-root>/audit/audit-v1.json`` ({"version", "digest", "entries"}).
    leak-test  prints ``suspect: <name>`` per suspect not covered by a decision; exit 3 while
            any exists; writes ``<run>/leak.json``.
    train   [--node {6a,6b,both}]  Order: integrity gates (exit 2) -> capability/subtle gates
            not waived (exit 3, names each gate and its offenders) -> Leak_Test suspects without
            decision (exit 3) -> fit per seed + release candidate -> choose cutoffs on
            validation -> write ``<run>/frozen.json`` (vault.write_freeze layout; cutoffs are
            {"6a": {"medium","high"}, "6b": {...}} of the release candidates; candidate_sha256
            {"6a","6b"} = SHA-256 of ``<run>/release/6x.joblib``). Candidates:
            ``<run>/models/6b/seed-<n>.joblib``, ``<run>/models/6a/seed-<n>.joblib``,
            ``<run>/release/6b.joblib``, ``<run>/release/6a.joblib``. Prints the validation
            operating curve, the GNN line, ``frozen.json`` and the waivers; updates the report
            header file with the waiver list. On any refusal no model file and no frozen.json.
    compare exit 2 unless the vault opens (frozen.json valid for the current run.json). Writes
            ``<run>/results/<node>/<dataset>/<seed>.json`` once each and atomically (no absolute
            paths, no timestamps inside), ``comparison_report.md`` and ``.json``; prints the
            console report (banner in its first 10 lines). Re-running skips existing result
            files and writes only the missing ones.
    switch  --node {6a,6b}  [--size-guard {candidate,off,current}]  [--override-reason TEXT]
            Candidate = ``<run>/release/<node>.joblib``, report id = run id, expected hash =
            frozen.json candidate_sha256. Exit 2 when the Switch_Rule recommends keep and no
            --override-reason is given, or for node 6b without --size-guard; otherwise
            modelops.switch(...); any ModelOpsError (hash mismatch, backup failure) exits 1.
    restore --node {6a,6b}  modelops.restore(...); NoBackupError exits 1.
    decide  KIND --target T --reason R --author A [--decision D]
            KIND in leak-feature (--decision exclude|accept), waive-gate (decision "waive"),
            investigation (--decision continue|abort), awkward-review (decision "reviewed").
            Appends to ``<run>/decisions.json`` through DecisionLog (id, kind, target, decision,
            reason, author, date). Missing/invalid argument exits 2 and writes nothing.
Lock: while a command that writes into a run folder runs, ``<run>/run.lock`` exists; if it
    already exists the command exits 2 with a message containing ``lock`` and leaves it alone.
    The lock is removed when the command ends.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import List, NamedTuple, Optional

import pytest

from fraud_detect.ml.retraining import cli
from fraud_detect.ml.retraining.audit import AUDIT_V1
from fraud_detect.ml.retraining.comparison_report import CIRCULARITY_STATEMENT, NO_SWITCH_TEXT
from fraud_detect.ml.retraining.safe_io import sha256_file
from fraud_detect.ml.retraining.switch_rule import SwitchRule
from fraud_detect.ml.retraining.vault import code_fingerprint

from .conftest import MINI_HOLD_OUT_LAYOUT, MINI_SEEDS, NODE4_MODEL_NAMES

REPO = Path(__file__).resolve().parents[2]
TRAIN_SEED, HELD_OUT_SEED = MINI_SEEDS
COMMANDS = ("init", "build", "check", "audit", "leak-test", "train", "compare", "switch",
            "restore", "decide")
PLUMBING_THRESHOLDS = {
    "type_min_docs": 1, "style_min_docs": 100000, "resaved_min_share": 0, "min_seeds": 1,
    "skip_max_share": 100, "row_exclusion_max_share": 100, "leak_auc": 1.01,
    "implausible_auc": 2.0, "indicative_min_docs": 1,
}


# ------------------------------------------------------------------------------ helpers
class Result(NamedTuple):
    code: int
    out: str


class Recorder:
    """Write-guard predicate that records every path it is asked about."""

    def __init__(self, ignored: bool = True):
        self.ignored = ignored
        self.calls: List[Path] = []

    def __call__(self, path) -> bool:
        self.calls.append(Path(path))
        return self.ignored


def make_clock():
    start = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
    ticks = count()
    return lambda: start + timedelta(seconds=next(ticks))


class Env:
    def __init__(self, root: Path, cache: Path, guard: Optional[Recorder] = None):
        self.root = root
        self.output_root = root / "data" / "private" / "ml_retraining"
        self.cache = cache
        self.models = root / "models"
        self.backup = root / "backup"
        self.guard = guard or Recorder()
        self.clock = make_clock()
        self.run_id: Optional[str] = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / "runs" / self.run_id

    def invoke(self, *argv, run_id=True, **kw) -> Result:
        args = list(map(str, argv))
        args += ["--output-root", str(self.output_root), "--cache-dir", str(self.cache)]
        if run_id and self.run_id and args[0] != "init":
            args += ["--run-id", self.run_id]
        kw.setdefault("is_ignored", self.guard)
        kw.setdefault("clock", self.clock)
        kw.setdefault("repo_root", REPO)
        kw.setdefault("models_dir", self.models)
        kw.setdefault("backup_dir", self.backup)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = cli.main(args, **kw)
        return Result(code, buf.getvalue())

    def init(self, *extra, thresholds=PLUMBING_THRESHOLDS, plumbing=True) -> Result:
        args = ["init", "--seeds", str(TRAIN_SEED), "--held-out-seed", str(HELD_OUT_SEED),
                "--held-out-layout", MINI_HOLD_OUT_LAYOUT,
                "--real-style-dir", self.root / "absent_real_style",
                "--forgery-dir", self.root / "absent_forgery"]
        for key, value in thresholds.items():
            args += ["--threshold", f"{key}={value}"]
        if plumbing:
            args.append("--plumbing")
        res = self.invoke(*args, *extra)
        m = re.search(r"^run_id: (\w+)$", res.out, re.M)
        if m:
            self.run_id = m.group(1)
        return res

    def decide(self, kind, target, decision=None, reason="test reason", author="tester") -> Result:
        args = ["decide", kind, "--target", target, "--reason", reason, "--author", author]
        if decision:
            args += ["--decision", decision]
        return self.invoke(*args)


def _fake_models(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(NODE4_MODEL_NAMES):
        (folder / name).write_bytes(f"fake current model {i}\n".encode() * 8)


def _snapshot(folder: Path) -> dict:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _carve_validation(corpus: Path, doc_id: str = "signed_amount_iso-g0001") -> None:
    """The Mini validation Split has no genuine document; move one whole company into it."""
    labels = corpus / "labels.csv"
    with open(labels, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = list(reader.fieldnames), list(reader)
    for r in rows:
        if r["doc_id"] == doc_id:
            assert r["split"] == "train"
            r["split"] = "validation"
    with open(labels, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = corpus / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for value in manifest.get("counts", {}).values():
        if isinstance(value, dict) and isinstance(value.get("train"), int) and "validation" in value \
                and "per_tamper_type" not in value:
            value["train"] -= 1
            value["validation"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _copy_corpora(prepared: Path, env: Env) -> None:
    target = env.output_root / "corpora"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prepared, target)


def _git_status() -> set:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return set(out.splitlines())


def _suspects(out: str) -> List[str]:
    return re.findall(r"^suspect: (\S+)", out, re.M)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _results(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "results").rglob("*.json"))


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def prepared(mini_corpus, tmp_path_factory) -> Path:
    """Mini_Corpus folders in the layout ``build`` produces, with a usable validation Split."""
    root = tmp_path_factory.mktemp("cli_prepared")
    shutil.copytree(mini_corpus.path(TRAIN_SEED), root / f"seed-{TRAIN_SEED}")
    shutil.copytree(mini_corpus.path(HELD_OUT_SEED), root / f"heldout-seed-{HELD_OUT_SEED}")
    _carve_validation(root / f"seed-{TRAIN_SEED}")
    return root


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cli_cache")


@pytest.fixture
def env(tmp_path, shared_cache) -> Env:
    e = Env(tmp_path, shared_cache)
    _fake_models(e.models)
    return e


@pytest.fixture
def ready(env, prepared) -> Env:
    """An initialised plumbing run with the Mini corpora in place (no step run yet)."""
    _copy_corpora(prepared, env)
    assert env.init().code == 0
    return env


class Plumbing(NamedTuple):
    env: Env
    outputs: dict
    models_before: dict
    status_before: set
    status_after: set
    network_calls: list


def _run_plumbing(root: Path, prepared: Path, cache: Path) -> Plumbing:
    env = Env(root, cache)
    _fake_models(env.models)
    _copy_corpora(prepared, env)
    models_before = _snapshot(env.models)
    status_before = _git_status()
    network_calls: list = []

    def refuse(*a, **k):
        network_calls.append(a)
        raise OSError("network access is not allowed in tests")

    out = {}
    with pytest.MonkeyPatch.context() as mp:
        for name in ("connect", "connect_ex"):
            mp.setattr(socket.socket, name, refuse)
        mp.setattr(socket, "getaddrinfo", refuse)
        out["init"] = env.init()
        out["check"] = env.invoke("check")
        out["audit"] = env.invoke("audit")
        out["leak"] = env.invoke("leak-test")
        out["train_blocked"] = env.invoke("train")
        out["waive_capability"] = env.decide("waive-gate", "capability", reason="plumbing")
        out["waive_subtle"] = env.decide("waive-gate", "subtle", reason="plumbing")
        out["train"] = env.invoke("train")
        out["compare"] = env.invoke("compare")
    return Plumbing(env, out, models_before, status_before, _git_status(), network_calls)


@pytest.fixture(scope="module")
def plumbing(prepared, shared_cache, tmp_path_factory) -> Plumbing:
    return _run_plumbing(tmp_path_factory.mktemp("cli_plumbing_a"), prepared, shared_cache)


@pytest.fixture(scope="module")
def plumbing_again(prepared, tmp_path_factory) -> Plumbing:
    """Second identical run with its own empty cache (Requirement 17.14)."""
    return _run_plumbing(tmp_path_factory.mktemp("cli_plumbing_b"), prepared,
                         tmp_path_factory.mktemp("cli_cache_b"))


def _copy_run(plumbing: Plumbing, tmp_path: Path, cache: Path) -> Env:
    """A private copy of a finished plumbing run, with its own fake models and backup folder."""
    e = Env(tmp_path, cache)
    shutil.copytree(plumbing.env.output_root, e.output_root)
    _fake_models(e.models)
    e.run_id = plumbing.env.run_id
    return e


# ------------------------------------------------------------------------------ --help
@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_exits_0_on_help(command):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main([command, "--help"])
    assert code == 0
    assert command in buf.getvalue()


@pytest.mark.parametrize("kind", ("leak-feature", "waive-gate", "investigation", "awkward-review"))
def test_decide_kinds_have_help(kind):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        assert cli.main(["decide", kind, "--help"]) == 0
    assert "--target" in buf.getvalue()


# ------------------------------------------------------------------------------ bad input
def test_bad_arguments_and_missing_run_exit_2(env):
    assert env.invoke("init", "--seeds", "abc").code == 2
    assert env.invoke("init", "--real-style-role", "bogus").code == 2
    assert env.invoke("init", "--rule-c-scope", "bogus").code == 2
    assert env.invoke("init", "--threshold", "no_such_threshold=1").code == 2
    env.run_id = "doesnotexist0000"
    for command in ("check", "audit", "leak-test", "train", "compare"):
        res = env.invoke(command)
        assert res.code == 2, command
        assert "run.json" in res.out
    assert env.invoke("switch", "--node", "6a").code == 2
    assert not env.output_root.exists() or not any(p.is_file() for p in env.output_root.rglob("*"))


# ------------------------------------------------------------------------------ init
def test_init_writes_run_json_and_report_header_before_any_evaluation(env):
    res = env.init()
    assert res.code == 0, res.out
    assert env.run_id and env.run_dir.name == env.run_id
    run_json = env.run_dir / "run.json"
    run = _read_json(run_json)
    assert f"run.json sha256: {sha256_file(run_json)}" in res.out
    rule = SwitchRule()
    assert run["switch_rule_sha256"] == rule.sha256()
    assert run["switch_rule_text"] == rule.text()
    assert run["code_fingerprint"] == code_fingerprint(REPO)
    assert run["cutoff_rule"]["target_fa_medium"] == 5.0 and run["cutoff_rule"]["target_fa_high"] == 1.0
    config = json.loads(run["config"])
    assert config["seeds"] == [TRAIN_SEED] and config["held_out_seed"] == HELD_OUT_SEED
    assert config["real_style_role"] == "required" and config["rule_c_scope"] == "all"
    assert config["plumbing"] is True
    # header file exists; nothing has been evaluated yet
    header = (env.run_dir / "comparison_report.md").read_text(encoding="utf-8")
    first10 = "\n".join(header.splitlines()[:10])
    assert "synthetic data" in first10 and CIRCULARITY_STATEMENT in first10
    assert rule.sha256() in header and NO_SWITCH_TEXT in header
    assert not (env.run_dir / "results").exists() and not (env.run_dir / "frozen.json").exists()


def test_init_open_decisions_are_configurable(env):
    res = env.init("--real-style-role", "veto_only", "--rule-c-scope", "genuine_only",
                   "--target-fa-medium", "7.5", "--target-fa-high", "2.5")
    assert res.code == 0, res.out
    run = _read_json(env.run_dir / "run.json")
    config = json.loads(run["config"])
    assert config["real_style_role"] == "veto_only" and config["rule_c_scope"] == "genuine_only"
    assert run["switch_rule_sha256"] == SwitchRule(real_style_role="veto_only").sha256()
    assert run["cutoff_rule"]["target_fa_medium"] == 7.5 and run["cutoff_rule"]["target_fa_high"] == 2.5
    header = (env.run_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert NO_SWITCH_TEXT not in header


def test_init_aborts_when_the_output_path_is_not_ignored(tmp_path, shared_cache):
    e = Env(tmp_path, shared_cache, guard=Recorder(ignored=False))
    res = e.init()
    assert res.code == 2
    assert str(e.output_root) in res.out and "ignored" in res.out
    assert not e.output_root.exists() or not any(p.is_file() for p in e.output_root.rglob("*"))


# ------------------------------------------------------------------------------ build
def test_build_calls_corpus_builder_for_each_seed_plus_held_out(env):
    assert env.invoke("init", "--seeds", f"{TRAIN_SEED},{HELD_OUT_SEED}", "--held-out-seed", "20260899",
                      "--held-out-layout", MINI_HOLD_OUT_LAYOUT).code == 0
    calls = []
    res = env.invoke("build", "--per-layout-genuine", "2", "--per-layout-tampered", "2",
                     corpus_builder=lambda cfg: calls.append(cfg))
    assert res.code == 0, res.out
    assert sorted(c.master_seed for c in calls) == sorted([TRAIN_SEED, HELD_OUT_SEED, 20260899])
    assert {c.hold_out_layout for c in calls} == {MINI_HOLD_OUT_LAYOUT}
    assert {(c.per_layout_genuine, c.per_layout_tampered) for c in calls} == {(2, 2)}
    corpora = env.output_root / "corpora"
    names = {c.master_seed: c.out_dir for c in calls}
    assert names[20260899] == corpora / "heldout-seed-20260899"
    assert names[TRAIN_SEED] == corpora / f"seed-{TRAIN_SEED}"
    # every write the command asked permission for is inside the output root
    assert env.guard.calls and all(env.output_root in p.parents or p == env.output_root
                                   for p in env.guard.calls)


def test_build_refuses_a_path_that_is_not_ignored(env):
    assert env.init().code == 0
    calls = []
    env.guard.ignored = False
    res = env.invoke("build", "--per-layout-genuine", "2", "--per-layout-tampered", "2",
                     corpus_builder=lambda cfg: calls.append(cfg))
    assert res.code == 2 and not calls


# ------------------------------------------------------------------------------ check
def test_check_plan_prints_counts_from_the_sizing_formula(env):
    res = env.invoke("check", "--plan", run_id=False)
    assert res.code == 0, res.out
    assert "documents per seed: 960" in res.out          # 6 layouts x (80 + 80)
    assert "corpora: 6" in res.out                       # 5 seeds + the held-out seed
    assert "documents in total: 5760" in res.out
    assert "genuine_resaved per seed: 184" in res.out    # 480 tampered x 10 // 26
    assert "other tamper types per seed: 18" in res.out  # 480 // 26
    small = env.invoke("check", "--plan", "--per-layout-genuine", "10", "--per-layout-tampered", "10",
                       "--seeds", "1,2", run_id=False)
    assert small.code == 0 and "documents per seed: 120" in small.out
    assert "corpora: 3" in small.out and "documents in total: 360" in small.out


def test_check_passes_on_the_mini_corpus_and_writes_gates_json(ready):
    res = ready.invoke("check")
    assert res.code == 0, res.out
    assert (ready.run_dir / "gates.json").is_file()


def test_failing_gate_prints_offenders_and_leaves_no_model_file(ready):
    labels = ready.output_root / "corpora" / f"seed-{TRAIN_SEED}" / "labels.csv"
    with open(labels, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    for r in rows:
        if r["doc_id"] == "two_col_slash-g0001":
            r["split"] = "bogus"
    with open(labels, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for command in ("check", "train"):
        res = ready.invoke(command)
        assert res.code == 2, command
        assert "two_col_slash-g0001" in res.out and f"seed-{TRAIN_SEED}" in res.out
    assert not list(ready.run_dir.rglob("*.joblib")) and not (ready.run_dir / "frozen.json").exists()


def test_held_out_layout_breach_is_named_and_stops_before_training(ready):
    labels = ready.output_root / "corpora" / f"seed-{TRAIN_SEED}" / "labels.csv"
    with open(labels, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    for r in rows:
        if r["doc_id"] == "two_col_ddmmyy_cr-g0001":
            r["split"] = "train"
    with open(labels, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    res = ready.invoke("train")
    assert res.code == 2 and "two_col_ddmmyy_cr-g0001" in res.out
    assert not list(ready.run_dir.rglob("*.joblib"))


def test_manifest_without_uen_check_refuses_training(ready):
    path = ready.output_root / "corpora" / f"seed-{TRAIN_SEED}" / "corpus_manifest.json"
    manifest = _read_json(path)
    manifest.pop("uen_check")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    res = ready.invoke("train")
    assert res.code == 2 and "uen_check" in res.out
    assert not list(ready.run_dir.rglob("*.joblib")) and not (ready.run_dir / "frozen.json").exists()


# ------------------------------------------------------------------------------ audit
def test_audit_writes_audit_v1_json(ready):
    res = ready.invoke("audit")
    assert res.code == 0, res.out
    data = _read_json(ready.output_root / "audit" / "audit-v1.json")
    assert data["version"] == "audit-v1" and data["digest"] == AUDIT_V1.digest()
    assert len(data["entries"]) == len(AUDIT_V1.entries)


# ------------------------------------------------------------------------------ decide
def test_decide_appends_entries_and_never_edits(ready):
    first = ready.decide("leak-feature", "balance_reconcile_breaks", "accept", reason="first")
    assert first.code == 0, first.out
    second = ready.decide("waive-gate", "capability", reason="second")
    third = ready.decide("awkward-review", "awkward-v1", reason="third")
    assert second.code == 0 and third.code == 0
    entries = _read_json(ready.run_dir / "decisions.json")["decisions"]
    assert [e["id"] for e in entries] == ["D-0001", "D-0002", "D-0003"]
    assert entries[0] == {**entries[0], "kind": "leak-feature", "target": "balance_reconcile_breaks",
                          "decision": "accept", "reason": "first", "author": "tester"}
    assert entries[1]["decision"] == "waive" and entries[2]["decision"] == "reviewed"
    assert entries[0]["date"].startswith("2026-08-01T09:")


def test_decide_rejects_bad_input_and_writes_nothing(ready):
    assert ready.decide("leak-feature", "x", "waive").code == 2          # decision not valid for kind
    assert ready.decide("leak-feature", "x").code == 2                   # decision required
    assert ready.invoke("decide", "waive-gate", "--target", "capability", "--author", "a").code == 2
    assert ready.invoke("decide", "no-such-kind", "--target", "x").code == 2
    assert not (ready.run_dir / "decisions.json").exists()


# ------------------------------------------------------------------------------ leak-test / train gates
def test_leak_test_exits_3_on_a_suspect_until_a_decision_is_recorded(env, prepared):
    _copy_corpora(prepared, env)
    assert env.init(thresholds={**PLUMBING_THRESHOLDS, "leak_auc": 0.0}).code == 0
    first = env.invoke("leak-test")
    assert first.code == 3, first.out
    names = _suspects(first.out)
    assert len(names) >= 2
    assert (env.run_dir / "leak.json").is_file()
    # train is blocked by the same undecided suspects, and fits nothing
    blocked = env.invoke("train")
    assert blocked.code == 3 and names[0] in blocked.out
    assert not list(env.run_dir.rglob("*.joblib")) and not (env.run_dir / "frozen.json").exists()
    # one decision is not enough; excluding or accepting every suspect clears the stop
    assert env.decide("leak-feature", names[0], "exclude").code == 0
    assert env.invoke("leak-test").code == 3
    for name in names[1:]:
        assert env.decide("leak-feature", name, "accept").code == 0
    assert env.invoke("leak-test").code == 0


def test_train_refuses_with_the_named_gates_until_they_are_waived(ready):
    res = ready.invoke("train")
    assert res.code == 3, res.out
    assert "capability" in res.out and "subtle" in res.out
    assert "creator" in res.out                        # a styling choice the generator never varies
    assert not list(ready.run_dir.rglob("*.joblib")) and not (ready.run_dir / "frozen.json").exists()
    assert ready.decide("waive-gate", "capability").code == 0
    still = ready.invoke("train")                      # subtle is still open
    assert still.code == 3 and "subtle" in still.out
    assert not list(ready.run_dir.rglob("*.joblib"))


def test_compare_refuses_without_a_valid_frozen_json(ready):
    res = ready.invoke("compare")
    assert res.code == 2 and "frozen.json" in res.out
    assert not (ready.run_dir / "results").exists()


# ------------------------------------------------------------------------------ plumbing run
def test_plumbing_run_exit_codes(plumbing):
    o = plumbing.outputs
    assert o["init"].code == 0, o["init"].out
    for step in ("check", "audit", "leak"):
        assert o[step].code == 0, (step, o[step].out)
    assert o["train_blocked"].code == 3
    assert o["waive_capability"].code == 0 and o["waive_subtle"].code == 0
    assert o["train"].code == 0, o["train"].out
    assert o["compare"].code == 0, o["compare"].out


def test_train_fits_after_the_gates_and_writes_frozen_json(plumbing):
    run = plumbing.env.run_dir
    frozen = _read_json(run / "frozen.json")
    assert frozen["run_hash"] == sha256_file(run / "run.json")
    assert frozen["switch_rule_sha256"] == SwitchRule().sha256()
    for node in ("6a", "6b"):
        assert (run / "models" / node / f"seed-{TRAIN_SEED}.joblib").is_file()
        release = run / "release" / f"{node}.joblib"
        assert frozen["candidate_sha256"][node] == sha256_file(release)
        assert 0.0 < frozen["cutoffs"][node]["medium"] <= frozen["cutoffs"][node]["high"] <= 1.0
    out = plumbing.outputs["train"].out
    assert "cutoff" in out and "alert rate" in out           # validation operating curve
    assert "GNN: not retrained, corpus has no transfer graph" in out
    assert "frozen.json" in out


def test_report_prints_the_waivers(plumbing):
    text = (plumbing.env.run_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "waive-gate" in text and "capability" in text and "subtle" in text
    assert "D-0001" in text and "D-0002" in text
    assert "capability" in plumbing.outputs["train"].out


def test_compare_report_first_10_lines_carry_the_banner(plumbing):
    run = plumbing.env.run_dir
    md = (run / "comparison_report.md").read_text(encoding="utf-8")
    console = plumbing.outputs["compare"].out
    for text in (md, console):
        first10 = "\n".join(text.splitlines()[:10])
        assert "synthetic data" in first10 and CIRCULARITY_STATEMENT in first10
    assert SwitchRule().sha256() in md and SwitchRule().sha256() in (run / "comparison_report.json").read_text()
    assert "baseline failed" in md                        # the fake Current models cannot be loaded
    assert SwitchRule().sha256() == _read_json(run / "run.json")["switch_rule_sha256"]


def test_results_are_written_per_unit_without_paths(plumbing):
    files = _results(plumbing.env.run_dir)
    assert len(files) >= 2
    for f in files:
        assert str(plumbing.env.root) not in f.read_text(encoding="utf-8")


def test_models_folder_is_byte_identical_and_nothing_sensitive_is_untracked(plumbing):
    assert _snapshot(plumbing.env.models) == plumbing.models_before
    assert not (plumbing.env.models / "backup").exists() and not plumbing.env.backup.exists()
    new = plumbing.status_after - plumbing.status_before
    assert not new, new
    forbidden = (".pdf", "labels.csv", "rows.csv", ".joblib", ".npz", ".pkl")
    assert not [line for line in plumbing.status_after if line.rstrip().endswith(forbidden)]


def test_every_write_went_through_the_guard_inside_the_run_tree(plumbing):
    calls = plumbing.env.guard.calls
    assert calls
    assert all(plumbing.env.root in p.parents for p in calls)


def test_no_network_call_and_no_lock_left_behind(plumbing):
    assert plumbing.network_calls == []
    assert not (plumbing.env.run_dir / "run.lock").exists()


def test_two_runs_with_the_same_inputs_give_equal_cutoffs_and_metrics(plumbing, plumbing_again):
    assert plumbing_again.outputs["compare"].code == 0, plumbing_again.outputs["compare"].out
    a, b = plumbing.env.run_dir, plumbing_again.env.run_dir
    assert _read_json(a / "frozen.json")["cutoffs"] == _read_json(b / "frozen.json")["cutoffs"]
    rel_a = [p.relative_to(a) for p in _results(a)]
    assert rel_a == [p.relative_to(b) for p in _results(b)]
    for rel in rel_a:
        assert _read_json(a / rel) == _read_json(b / rel), rel


# ------------------------------------------------------------------------------ lock, resume, vault
def test_a_lock_file_refuses_a_second_run_on_the_same_run_folder(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    lock = e.run_dir / "run.lock"
    lock.write_text("pid 1\n", encoding="utf-8")
    before = _snapshot(e.run_dir)
    for command in ("compare", "train"):
        res = e.invoke(command)
        assert res.code == 2 and "lock" in res.out.lower(), command
    assert lock.read_text(encoding="utf-8") == "pid 1\n"
    after = _snapshot(e.run_dir)
    assert after == before


def test_an_interrupted_compare_resumes_at_the_first_unwritten_unit(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    files = _results(e.run_dir)
    assert len(files) >= 2
    victim, kept = files[-1], files[:-1]
    expected = victim.read_bytes()
    stats = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in kept}
    victim.unlink()
    res = e.invoke("compare")
    assert res.code == 0, res.out
    assert victim.read_bytes() == expected                       # the missing unit was redone
    for p, (data, mtime) in stats.items():                        # written once, never rewritten
        assert (p.read_bytes(), p.stat().st_mtime_ns) == (data, mtime)


def test_compare_refuses_when_run_json_changed_after_the_freeze(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    run_json = e.run_dir / "run.json"
    run_json.write_text(run_json.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    shutil.rmtree(e.run_dir / "results")
    res = e.invoke("compare")
    assert res.code == 2
    assert not (e.run_dir / "results").exists()


# ------------------------------------------------------------------------------ switch / restore
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_switch_refuses_a_keep_recommendation_without_override(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    before = _snapshot(e.models)
    res = e.invoke("switch", "--node", "6a")
    assert res.code == 2, res.out
    assert _snapshot(e.models) == before and not e.backup.exists()


def test_switch_and_restore_round_trip_with_override(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    live = e.models / "txn_fraud_clf.joblib"
    original = live.read_bytes()
    release = e.run_dir / "release" / "6a.joblib"
    assert e.invoke("restore", "--node", "6a").code == 1             # no backup yet
    assert live.read_bytes() == original
    res = e.invoke("switch", "--node", "6a", "--override-reason", "plumbing test")
    assert res.code == 0, res.out
    assert _sha(live) == _sha(release)
    index = _read_json(e.backup / "index.json")["6a"]
    assert index["report_id"] == e.run_id and index["override_reason"] == "plumbing test"
    assert index["backup_sha256"] == hashlib.sha256(original).hexdigest()
    assert e.invoke("restore", "--node", "6a").code == 0
    assert live.read_bytes() == original
    # the other node was never touched
    assert (e.models / "statement_forgery_clf.joblib").read_bytes() == plumbing.models_before[
        "statement_forgery_clf.joblib"]


def test_switch_node_6b_needs_an_explicit_size_guard_choice(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    live = e.models / "statement_forgery_clf.joblib"
    original = live.read_bytes()
    assert e.invoke("switch", "--node", "6b", "--override-reason", "t").code == 2
    assert live.read_bytes() == original
    res = e.invoke("switch", "--node", "6b", "--size-guard", "off", "--override-reason", "t")
    assert res.code == 0, res.out
    assert live.read_bytes() != original
    assert e.invoke("restore", "--node", "6b").code == 0
    assert live.read_bytes() == original


def test_switch_stops_when_the_candidate_hash_differs_from_the_report(plumbing, tmp_path, shared_cache):
    e = _copy_run(plumbing, tmp_path, shared_cache)
    (e.run_dir / "release" / "6a.joblib").write_bytes(b"not the frozen candidate")
    before = _snapshot(e.models)
    res = e.invoke("switch", "--node", "6a", "--override-reason", "t")
    assert res.code == 1, res.out
    assert _snapshot(e.models) == before
