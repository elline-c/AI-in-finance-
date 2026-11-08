"""Orchestration behind the retraining command line (Task 23.2).

Each ``cmd_*`` function runs one command in the order of the design's data-flow diagram and
returns the exit code; refusals raise ``cli.Stop``. Heavy dependencies (pandas, scikit-learn,
pdfplumber, joblib) are imported lazily or by modules that are only imported here, and this
module itself is only imported when a command runs, never for ``--help``.

Nothing here reads a real ``models/`` file except through ``baseline.load_current`` on the
injected ``models_dir`` (read only), and every write goes through the write guard.
"""
from __future__ import annotations

import dataclasses
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import corpus_gates, gates, safe_io
from .cli import DEFAULT_OUTPUT_ROOT, DECIDE_KINDS, EXIT_FAILURE, EXIT_INTEGRITY, EXIT_OK, EXIT_WAITING, Stop
from .config import AWKWARD_MASTER_SEED, RunConfig, Thresholds
from .corpora import LoadedCorpus, load_corpus
from .decisions import DecisionLog
from .gates import Gate, GateResult, run_gates
from .safe_io import atomic_write, ensure_ignored, sha256_file

N_LAYOUTS = 6
MIX_PARTS = 26               # 16 original types + subtle_amount_edit (10 + 16 x 1)
RESAVED_PER_26 = 10          # genuine_resaved share of the tampered mix: 10 of 26 parts
TAMPER_TYPES_PER_26 = 1      # each of the other 16 types: 1 of 26 parts
INTEGRITY_GATE_IDS = frozenset({
    "corpus_seed", "company_split", "layout_separation", "held_out_seed", "uen_manifest",
    "awkward_overlap", "styling_balance", "resaved_share", "tamper_type_minimum",
    "tamper_type_coverage"})
DATASET_ORDER = ("test", "heldout_layout", "heldout_seed", "forgery_test", "forgery_ood",
                 "real_style", "awkward")
DATASET_SPLIT = {"test": "test", "heldout_layout": "test_unseen_layout", "heldout_seed": "heldout_seed",
                 "forgery_test": "test_in_layout", "forgery_ood": "test_ood_layout",
                 "real_style": "real_style", "awkward": "awkward"}
ROW_DATASETS = ("test", "heldout_layout", "heldout_seed")   # datasets that carry rows for Node 6a
REAL_STYLE_LABELS = {"statement_E.pdf": True, "statement_F.pdf": False}


# ------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class CliRunConfig(RunConfig):
    """``RunConfig`` plus the two open-decision / mode options the command line adds."""

    rule_c_scope: str = "all"          # "all" or "genuine_only" (open decision 4)
    plumbing: bool = False             # plumbing run: coverage gate reported, not enforced


def config_from_json(text: str) -> CliRunConfig:
    data = json.loads(text)
    data["thresholds"] = Thresholds(**data["thresholds"])
    data["seeds"] = tuple(data["seeds"])
    return CliRunConfig(**data)


def _now_iso(clock) -> str:
    return clock().isoformat()


# ------------------------------------------------------------------------------ session
class Guard:
    """Write-guard predicate: the injected one (or ``git check-ignore``) for everything, except an
    explicit cache folder outside the repository and the output root, which git cannot track."""

    def __init__(self, predicate, output_root: Path, cache_dir: Path, repo_root: Path):
        self._predicate = predicate if predicate is not None else safe_io._git_check_ignore
        cache, out, repo = cache_dir.resolve(), output_root.resolve(), repo_root.resolve()
        self._cache = cache if not (_under(cache, repo) or _under(cache, out)) else None

    def __call__(self, path) -> bool:
        p = Path(path)
        if self._cache is not None and _under(p.resolve(), self._cache):
            return True
        return bool(self._predicate(p))


def _under(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


@dataclass
class Session:
    output_root: Path
    cache_dir: Path
    guard: Callable
    clock: Callable
    repo_root: Path
    models_dir: Path
    backup_dir: Path
    corpus_builder: Optional[Callable]
    run_dir: Optional[Path] = None
    run: Optional[dict] = None
    config: Optional[CliRunConfig] = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def create(cls, args, *, is_ignored, clock, repo_root, models_dir, backup_dir, corpus_builder):
        repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        out = Path(args.output_root) if args.output_root else repo / DEFAULT_OUTPUT_ROOT
        cache = Path(args.cache_dir) if args.cache_dir else out / "cache"
        models = Path(models_dir) if models_dir is not None else Path("models")
        backup = Path(backup_dir) if backup_dir is not None else models / "backup"
        return cls(output_root=out, cache_dir=cache, guard=Guard(is_ignored, out, cache, repo),
                   clock=clock or (lambda: datetime.now(timezone.utc)), repo_root=repo,
                   models_dir=models, backup_dir=backup, corpus_builder=corpus_builder)

    # -- run folder
    def load_run(self, run_id: Optional[str]) -> None:
        if not run_id:  # the only run folder is unambiguous
            runs = self.output_root / "runs"
            found = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
            if len(found) != 1:
                raise Stop(EXIT_INTEGRITY, "--run-id is required (it names the folder under "
                                           f"{runs} that holds run.json)")
            run_id = found[0]
        run_dir = self.output_root / "runs" / run_id
        run_json = run_dir / "run.json"
        if not run_json.is_file():
            raise Stop(EXIT_INTEGRITY, f"run not found: {run_json} does not exist. "
                                       f"Create a run with the init command first.")
        try:
            self.run = json.loads(run_json.read_text(encoding="utf-8"))
            self.config = config_from_json(self.run["config"])
        except (ValueError, KeyError, TypeError) as exc:
            raise Stop(EXIT_INTEGRITY, f"cannot read {run_json}: {exc}") from None
        self.run_dir = run_dir

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def corpora_root(self) -> Path:
        return self.output_root / "corpora"

    def decisions(self) -> DecisionLog:
        return DecisionLog(self.run_dir / "decisions.json", is_ignored=self.guard, clock=self.clock)

    def write(self, path, data) -> None:
        atomic_write(path, data, is_ignored=self.guard)

    def rule(self):
        from .switch_rule import SwitchRule
        rule = SwitchRule(real_style_role=self.config.real_style_role)
        if rule.sha256() != self.run["switch_rule_sha256"]:
            raise Stop(EXIT_INTEGRITY, "the Switch_Rule hash differs from the one recorded in run.json")
        return rule

    def cutoff_rule(self, level: str):
        from .cutoffs import CutoffRule
        return CutoffRule(target_fa_medium=self.config.target_fa_medium,
                          target_fa_high=self.config.target_fa_high, level=level)


# ------------------------------------------------------------------------------ init
def _plain(text: str) -> str:
    return text


def cmd_init(session: Session, args) -> int:
    from .comparison_report import ReportHeader, write_header
    from .awkward import AWKWARD_VERSION
    from .switch_rule import SwitchRule
    from .vault import code_fingerprint, write_run_json

    base = RunConfig()
    thresholds = dataclasses.replace(base.thresholds, **dict(args.threshold))
    fa_medium = args.target_fa_medium if args.target_fa_medium is not None else thresholds.target_fa_medium
    fa_high = args.target_fa_high if args.target_fa_high is not None else thresholds.target_fa_high
    thresholds = dataclasses.replace(thresholds, target_fa_medium=fa_medium, target_fa_high=fa_high)
    seeds = tuple(args.seeds) if args.seeds else base.seeds
    held_seed = args.held_out_seed if args.held_out_seed is not None else base.held_out_seed
    if held_seed in seeds or AWKWARD_MASTER_SEED in seeds or held_seed == AWKWARD_MASTER_SEED:
        raise Stop(EXIT_INTEGRITY, "seeds must be distinct: the held-out seed and the Awkward_Set "
                                   f"seed {AWKWARD_MASTER_SEED} may not be training seeds")
    if len(set(seeds)) != len(seeds):
        raise Stop(EXIT_INTEGRITY, "training seeds must be distinct")
    config = CliRunConfig(
        thresholds=thresholds, seeds=seeds, held_out_seed=held_seed,
        held_out_layout=args.held_out_layout or base.held_out_layout,
        real_style_role=args.real_style_role, target_fa_medium=fa_medium, target_fa_high=fa_high,
        real_style_dir=args.real_style_dir or str(session.output_root / "real_style"),
        forgery_dir=args.forgery_dir or str(session.output_root / "forgery_dataset"),
        output_root=str(session.output_root), rule_c_scope=args.rule_c_scope, plumbing=bool(args.plumbing))

    ensure_ignored(session.output_root, is_ignored=session.guard)
    run_dir = session.output_root / "runs" / config.run_id
    if (run_dir / "run.json").exists():
        raise Stop(EXIT_INTEGRITY, f"run {config.run_id} already exists ({run_dir / 'run.json'}); "
                                   f"run.json is never rewritten")
    rule = SwitchRule(real_style_role=config.real_style_role)
    cutoff_rule = {"target_fa_medium": fa_medium, "target_fa_high": fa_high}
    digest = write_run_json(
        run_dir, config=config, switch_rule_sha256=rule.sha256(), cutoff_rule=cutoff_rule,
        fingerprint=code_fingerprint(session.repo_root), is_ignored=session.guard,
        extra={"switch_rule_text": rule.text()})
    write_header(run_dir, ReportHeader(
        switch_rule_text=rule.text(), switch_rule_sha256=rule.sha256(), cutoff_rule=cutoff_rule,
        awkward_seed=AWKWARD_MASTER_SEED, awkward_version=AWKWARD_VERSION,
        real_style_role=config.real_style_role), is_ignored=session.guard)
    print(f"run_id: {config.run_id}")
    print(f"run.json sha256: {digest}")
    print(f"run folder: {run_dir}")
    if config.plumbing:
        print("plumbing run: the tamper-type coverage gate is reported but not enforced")
    if config.real_style_role == "required":
        print("note: Real_Style is required and cannot be decisive with so few files; "
              "no switch is possible in this mode")
    return EXIT_OK


# ------------------------------------------------------------------------------ build / plan
def design_tamper_mix() -> tuple:
    """The design's tamper mix: genuine_resaved 10 parts, each of the other 16 types 1 part (of 26).

    The 16 types are the original ones plus subtle_amount_edit (``ALL_TAMPER_TYPES``). Without the
    mix the generator deals the types equally (about 6% re-saved genuine documents), which fails the
    15% re-saved gate in every Split.
    """
    from fraud_detect.corpus.schema import ALL_TAMPER_TYPES
    return tuple((t, float(RESAVED_PER_26) if t == "genuine_resaved" else float(TAMPER_TYPES_PER_26))
                 for t in ALL_TAMPER_TYPES)


def plan_counts(per_genuine: int, per_tampered: int, n_seeds: int) -> dict:
    per_seed = N_LAYOUTS * (per_genuine + per_tampered)
    corpora = n_seeds + 1
    return {
        "documents per seed": per_seed,
        "corpora": corpora,
        "documents in total": per_seed * corpora,
        "genuine_resaved per seed": per_tampered * N_LAYOUTS * RESAVED_PER_26 // MIX_PARTS,
        "other tamper types per seed": per_tampered * N_LAYOUTS * TAMPER_TYPES_PER_26 // MIX_PARTS,
    }


def cmd_check_plan(args) -> int:
    n_seeds = len(args.seeds) if args.seeds else len(RunConfig().seeds)
    for key, value in plan_counts(args.per_layout_genuine, args.per_layout_tampered, n_seeds).items():
        print(f"{key}: {value}")
    return EXIT_OK


def cmd_build(session: Session, args) -> int:
    from fraud_detect.corpus.schema import BuildConfig
    builder = session.corpus_builder
    if builder is None:
        def builder(cfg):
            from fraud_detect.corpus.builder import CorpusBuilder
            return CorpusBuilder(cfg).build()
    cfg = session.config
    ensure_ignored(session.corpora_root, is_ignored=session.guard)
    targets = [(s, session.corpora_root / f"seed-{s}") for s in cfg.seeds]
    targets.append((cfg.held_out_seed, session.corpora_root / f"heldout-seed-{cfg.held_out_seed}"))
    for seed, out_dir in targets:
        ensure_ignored(out_dir, is_ignored=session.guard)
    for seed, out_dir in targets:
        print(f"building corpus for Master_Seed {seed} in {out_dir}")
        builder(BuildConfig(
            out_dir=out_dir, master_seed=seed, per_layout_genuine=args.per_layout_genuine,
            per_layout_tampered=args.per_layout_tampered, months_min=args.months_min,
            months_max=args.months_max, hold_out_layout=cfg.held_out_layout,
            tamper_mix=design_tamper_mix(), style_version=2,
            command_line=f"retraining build --seed {seed} --per-layout-genuine "
                         f"{args.per_layout_genuine} --per-layout-tampered {args.per_layout_tampered}"))
    return EXIT_OK


# ------------------------------------------------------------------------------ corpora and gates
@dataclass
class Loaded:
    corpus: LoadedCorpus
    path: Path
    held_out: bool


def load_corpora(session: Session):
    cfg = session.config
    training = [Loaded(load_corpus(session.corpora_root / f"seed-{s}"),
                       session.corpora_root / f"seed-{s}", False) for s in cfg.seeds]
    held_path = session.corpora_root / f"heldout-seed-{cfg.held_out_seed}"
    return training, Loaded(load_corpus(held_path), held_path, True)


def _gate_result(gate_id, requirement, offenders, waivable=False) -> GateResult:
    return GateResult(gate_id, requirement, not offenders, tuple(offenders), waivable, None)


def awkward_ids():
    from .awkward import CASE_TYPES, COMPANIES_PER_CASE
    companies = [f"AW-{c}-{k}" for c in CASE_TYPES for k in range(1, COMPANIES_PER_CASE + 1)]
    docs = [f"{c}-{kind}" for c in companies for kind in ("g", "t")]
    return docs, companies


def integrity_results(session: Session, training: List[Loaded], held: Loaded) -> List[GateResult]:
    from . import group_checks as gc
    cfg = session.config
    everything = training + [held]
    seed_offenders = []
    for loaded, expected in zip(everything, list(cfg.seeds) + [cfg.held_out_seed]):
        if loaded.corpus.master_seed != expected:
            seed_offenders.append(f"{loaded.path.name}: manifest master_seed is "
                                  f"{loaded.corpus.master_seed}, the run expects {expected}")
        if loaded.corpus.hold_out_layout != cfg.held_out_layout:
            seed_offenders.append(f"{loaded.path.name}: hold_out_layout is "
                                  f"{loaded.corpus.hold_out_layout!r}, the run expects {cfg.held_out_layout!r}")
    corpora = [x.corpus for x in everything]
    train_corpora = [x.corpus for x in training]
    doc_ids, company_ids = awkward_ids()
    results = [
        _gate_result("corpus_seed", "1.4", seed_offenders),
        gc.company_split_gate(corpora),
        gc.layout_gate(corpora, cfg.held_out_layout),
        gc.held_out_seed_gate(held.corpus, train_corpora),
    ]
    uen_offenders = []
    for corpus in corpora:
        uen_offenders.extend(gc.uen_manifest_gate(corpus).offenders)
    results.append(_gate_result("uen_manifest", "17.11, 17.12", uen_offenders))
    results.append(gc.awkward_overlap_gate(train_corpora, doc_ids=doc_ids, company_ids=company_ids,
                                           master_seed=AWKWARD_MASTER_SEED))
    return results


def _records(frame) -> List[dict]:
    out = []
    for rec in frame.to_dict("records"):
        out.append({k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()})
    return out


def _merge(gate_id, requirement, parts, waivable=False) -> GateResult:
    offenders = [f"{label}: {o}" if label else o for label, res in parts for o in res.offenders]
    return _gate_result(gate_id, requirement, offenders, waivable)


def _affected_rows(path: Path, doc_id: str) -> list:
    try:
        manifest = json.loads((path / "manifests" / f"{doc_id}.json").read_text(encoding="utf-8"))
        return list(manifest.get("label", {}).get("affected_rows", []))
    except (OSError, ValueError):
        return []


def build_tables_b(session: Session, loaded: List[Loaded]) -> dict:
    from .features_b import build_feature_table
    th = session.config.thresholds
    tables = {}
    for item in loaded:
        print(f"Node 6b features for {item.path.name} ...")
        tables[item.path.name] = build_feature_table(
            item.path, cache_root=session.cache_dir, is_ignored=session.guard,
            skip_max_share=th.skip_max_share)
    return tables


def corpus_gate_results(session: Session, training: List[Loaded], tables: dict) -> List[GateResult]:
    cfg, th = session.config, session.config.thresholds
    per_corpus = {x.path.name: _records(tables[x.path.name].frame) for x in training}
    pooled = [r for rows in per_corpus.values() for r in rows]
    style, resaved = [], []
    for name, rows in per_corpus.items():
        style.append((name, corpus_gates.styling_balance_gate(
            rows, min_docs=th.style_min_docs, share_points=th.style_share_points)))
        resaved.append((name, corpus_gates.resaved_share_gate(rows, min_share=th.resaved_min_share)))
    subtle_docs = []
    for x in training:
        for rec in per_corpus[x.path.name]:
            subtle_docs.append({
                "is_tampered": rec["is_tampered"], "split": rec["split"],
                "affected_rows": _affected_rows(x.path, rec["doc_id"]) if rec["is_tampered"] else []})
    results = [
        _merge("styling_balance", "5.2, 5.3", style),
        _merge("resaved_share", "5.5", resaved),
        corpus_gates.tamper_type_minimum_gate(pooled, min_docs=th.type_min_docs,
                                              pooled=cfg.pooled_type_minimum),
        corpus_gates.coverage_gate(pooled),
        corpus_gates.capability_gate(pooled),
        corpus_gates.subtle_gate(subtle_docs, max_relative_change=th.subtle_max_relative_change / 100.0),
    ]
    return results


def apply_waivers(session: Session, results: List[GateResult]) -> List[GateResult]:
    checks = [Gate(r.gate_id, r.requirement, r.waivable, (lambda ctx, o=r.offenders: o)) for r in results]
    out = run_gates(checks, None, session.decisions())
    if session.config.plumbing:  # reported, not enforced (Requirement 6.2 cannot hold on a Mini_Corpus)
        out = [dataclasses.replace(r, passed=True, waived_by="plumbing")
               if r.gate_id == "tamper_type_coverage" and not r.passed else r for r in out]
    return out


def print_gates(results: List[GateResult]) -> None:
    for r in results:
        if r.passed and not r.waived_by:
            continue
        kind = "waivable" if r.waivable else "integrity"
        status = f"waived by {r.waived_by}" if r.waived_by else "FAILED"
        print(f"gate {r.gate_id} (requirement {r.requirement}, {kind}): {status}")
        for o in r.offenders:
            print(f"  - {o}")
        if not r.passed and r.waivable:
            print(f"  to waive: decide waive-gate --target {r.gate_id} --reason TEXT --author NAME")


def write_gates(session: Session, results: List[GateResult]) -> None:
    session.write(session.run_dir / "gates.json", json.dumps(
        {"plumbing": session.config.plumbing, "gates": [dataclasses.asdict(r) for r in results]},
        indent=2, sort_keys=True))


def _integrity_failed(results) -> List[GateResult]:
    return [r for r in results if not r.passed and not r.waivable]


def run_all_gates(session: Session, *, stop_on_integrity: bool = True):
    """Load, run the integrity gates, build the Node 6b tables, run the corpus gates.

    Returns ``(training, held, tables, results)``; raises ``Stop`` (exit 2) when an integrity gate
    fails.
    """
    training, held = load_corpora(session)
    results = apply_waivers(session, integrity_results(session, training, held))
    failed = _integrity_failed(results)
    if failed:
        write_gates(session, results)
        print_gates(results)
        raise Stop(EXIT_INTEGRITY, "integrity gate failed: nothing was trained and no model was written")
    tables = build_tables_b(session, training)
    results += apply_waivers(session, corpus_gate_results(session, training, tables))
    write_gates(session, results)
    failed = _integrity_failed(results)
    if failed:
        print_gates(results)
        raise Stop(EXIT_INTEGRITY, "integrity gate failed: nothing was trained and no model was written")
    return training, held, tables, results


def cmd_check(session: Session, args) -> int:
    training, held, tables, results = run_all_gates(session)
    print_gates(results)
    open_waivable = [r.gate_id for r in results if not r.passed and r.waivable]
    print(f"check passed: {len(results)} gates run, gates.json written")
    if open_waivable:
        print("waivable gates not passed (they stop `train` until waived): " + ", ".join(open_waivable))
    return EXIT_OK


# ------------------------------------------------------------------------------ audit / leak
def cmd_audit(session: Session, args) -> int:
    from .audit import AUDIT_V1, validate
    validate(AUDIT_V1)
    data = {"version": AUDIT_V1.version, "digest": AUDIT_V1.digest(),
            "entries": [dataclasses.asdict(e) for e in AUDIT_V1.entries]}
    path = session.output_root / "audit" / "audit-v1.json"
    session.write(path, json.dumps(data, indent=2, sort_keys=True))
    print(f"Feature_Audit {AUDIT_V1.version} valid; digest {AUDIT_V1.digest()}; written to {path}")
    return EXIT_OK


def _pooled_6b(tables: dict, training: List[Loaded]):
    import pandas as pd
    frame = pd.concat([tables[x.path.name].frame for x in training], ignore_index=True)
    frame = frame.assign(y=frame["is_tampered"].astype(int))
    return frame


def run_leak(session: Session, training: List[Loaded], tables: dict):
    from .audit import AUDIT_V1
    from .leak import leak_test
    frame = _pooled_6b(tables, training)
    train, valid = frame[frame["split"] == "train"], frame[frame["split"] == "validation"]
    result = leak_test(train, valid, AUDIT_V1, session.config.thresholds, session.decisions(), forgery=None)
    return result


def print_leak(result) -> None:
    if result.notice:
        print(result.notice)
    for name, auc in result.simple.items():
        print(f"simple model {name}: validation AUC {auc:.4f}")
    for s in result.suspects:
        print(f"suspect: {s.name} (AUC {s.auc:.4f}, {s.kind}); record a decision with: "
              f"decide leak-feature --target {s.name} --decision exclude|accept --reason TEXT --author NAME")
    for s in result.decided:
        print(f"decided: {s.name} (AUC {s.auc:.4f})")


def cmd_leak_test(session: Session, args) -> int:
    training, held = load_corpora(session)
    tables = build_tables_b(session, training)
    result = run_leak(session, training, tables)
    session.write(session.run_dir / "leak.json", json.dumps(
        json.loads(json.dumps(dataclasses.asdict(result), default=str)), indent=2, sort_keys=True))
    print_leak(result)
    if result.blocked:
        print("Leak_Test: suspects wait for a decision")
        return EXIT_WAITING
    print("Leak_Test: no undecided suspect")
    return EXIT_OK


# ------------------------------------------------------------------------------ decide
def cmd_decide(session: Session, args) -> int:
    allowed, _ = DECIDE_KINDS[args.kind]
    decision = args.decision
    if decision not in allowed:
        raise Stop(EXIT_INTEGRITY, f"decision {decision!r} is not valid for {args.kind}")
    if args.kind == "waive-gate" and args.target in INTEGRITY_GATE_IDS:
        raise Stop(EXIT_INTEGRITY, f"{args.target} is an integrity gate and can never be waived")
    if args.kind == "investigation":
        from .monitor import Monitor
        monitor = Monitor(session.run_dir, is_ignored=session.guard, clock=session.clock)
        record_path = session.run_dir / "investigation.json"
        if not record_path.is_file():
            raise Stop(EXIT_INTEGRITY, "no investigation is open in this run")
        if json.loads(record_path.read_text(encoding="utf-8"))["id"] != args.target:
            raise Stop(EXIT_INTEGRITY, f"--target must be the open investigation id")
        entry = monitor.decide(decision, args.reason, args.author)
    else:
        entry = session.decisions().append(args.kind, args.target, decision, args.reason, args.author)
    print(f"{entry['id']}: {entry['kind']} {entry['target']} -> {entry['decision']} "
          f"({entry['author']}, {entry['date']})")
    return EXIT_OK


# ------------------------------------------------------------------------------ train
def _curve_lines(label: str, curve: list) -> List[str]:
    lines = [f"{label} (validation operating curve)"]
    for r in curve:
        lines.append(f"{r['name']} cutoff {r['cutoff']:.3f}: alert rate {r['alert_rate']:.1f}%, "
                     f"false-alarm rate {r['false_alarm_rate']:.1f}%, catch rate {r['catch_rate']:.1f}%")
    return lines


def _train_6a(session: Session, training: List[Loaded], gate_results, ledger_seeds) -> dict:
    import pandas as pd
    from .audit import AUDIT_V1
    from .features_a import load_row_table
    from .train_a import train_candidate

    cfg, th = session.config, session.config.thresholds
    rule = session.cutoff_rule("statement_max_row")
    feats, stmts = {}, {}
    for x in training:
        print(f"Node 6a rows for {x.path.name} ...")
        table = load_row_table(x.path, cache_root=session.cache_dir, is_ignored=session.guard,
                               row_exclusion_max_share=th.row_exclusion_max_share)
        feats[x.corpus.master_seed] = table.features
        stmts[x.corpus.master_seed] = pd.DataFrame(
            [{"corpus_key": d.corpus_key, "doc_id": d.doc_id, "is_tampered": d.is_tampered}
             for d in x.corpus.docs], columns=["corpus_key", "doc_id", "is_tampered"])
    common = dict(audit=AUDIT_V1, rule=rule, held_out_layout=cfg.held_out_layout, is_ignored=session.guard)
    out = {}
    for seed in sorted(feats):
        f = feats[seed]
        res = train_candidate(f, f[f["split"] == "validation"], stmts[seed],
                              out_path=session.run_dir / "models" / "6a" / f"seed-{seed}.joblib",
                              master_seed=seed, **common)
        out[seed] = res
        for line in _curve_lines(f"Node 6a seed {seed}", res.bundle["validation_curve"]):
            print(line)
        n_zero = res.bundle["zero_fraud_statements"]["count"]
        print(f"Node 6a seed {seed}: {n_zero} tampered statement(s) with no fraud row left out of detection")
    pooled = pd.concat([feats[s] for s in sorted(feats)], ignore_index=True)
    pooled_stmts = pd.concat([stmts[s] for s in sorted(stmts)], ignore_index=True)
    release = train_candidate(pooled, pooled[pooled["split"] == "validation"], pooled_stmts,
                              out_path=session.run_dir / "release" / "6a.joblib", master_seed=None, **common)
    for line in _curve_lines("Node 6a release", release.bundle["validation_curve"]):
        print(line)
    out["release"] = release
    return out


def _train_6b(session: Session, training: List[Loaded], tables: dict, gate_results, leak) -> dict:
    from .audit import AUDIT_V1
    from .train_b import train_run
    from .vault import check_classes

    seed_tables = {}
    for x in training:
        frame = tables[x.path.name].frame
        y = frame["is_tampered"].astype(int)
        try:
            check_classes({"train": list(y[frame["split"] == "train"]),
                           "validation": list(y[frame["split"] == "validation"])})
        except Exception as exc:
            raise Stop(EXIT_INTEGRITY, f"{x.path.name}: {exc}") from None
        seed_tables[x.corpus.master_seed] = frame
    outputs = train_run(seed_tables, held_out_layout=session.config.held_out_layout,
                        out_dir=session.run_dir, audit=AUDIT_V1,
                        rule=session.cutoff_rule("document"), gate_results=gate_results,
                        leak=leak, is_ignored=session.guard)
    bad = [str(r.path) for r in list(outputs.per_seed.values()) + [outputs.release] if r.failed]
    if bad:
        raise Stop(EXIT_FAILURE, "candidate(s) failed to load or score: " + ", ".join(bad))
    return outputs


def _waiver_lines(session: Session) -> List[str]:
    return [f"{e['id']} [waiver] waive-gate {e['target']}: {e['reason']} ({e['author']}, {e['date']})"
            for e in session.decisions().entries() if e["kind"] == "waive-gate"]


def _update_header(session: Session) -> None:
    from .comparison_report import ReportHeader, render_header_markdown
    header = header_from_run(session)
    text = render_header_markdown(header)
    waivers = _waiver_lines(session)
    if waivers:
        text += "\nWaivers (waive-gate decisions):\n" + "\n".join(waivers) + "\n"
    session.write(session.run_dir / "comparison_report.md", text)


def header_from_run(session: Session):
    from .awkward import AWKWARD_VERSION
    from .comparison_report import ReportHeader
    run = session.run
    return ReportHeader(
        switch_rule_text=run["switch_rule_text"], switch_rule_sha256=run["switch_rule_sha256"],
        cutoff_rule=run["cutoff_rule"], awkward_seed=AWKWARD_MASTER_SEED,
        awkward_version=AWKWARD_VERSION, real_style_role=session.config.real_style_role)


def cmd_train(session: Session, args) -> int:
    from .monitor import SeedLedger
    from .vault import write_freeze

    if (session.run_dir / "frozen.json").exists():
        raise Stop(EXIT_INTEGRITY, "frozen.json already exists: the cutoffs and candidates of this run "
                                   "are frozen; start a new run to train again")
    training, held, tables, results = run_all_gates(session)
    open_waivable = [r for r in results if not r.passed and r.waivable]
    leak = run_leak(session, training, tables)
    if open_waivable or leak.blocked:
        print_gates(open_waivable)
        print_leak(leak)
        print("Training refused: the items above wait for a decision (exit 3). "
              "Nothing was fitted and no model file was written.")
        return EXIT_WAITING
    print_gates(results)
    print_leak(leak)

    ledger = SeedLedger(session.run_dir, is_ignored=session.guard, clock=session.clock)
    seeds = list(session.config.seeds)
    for s in seeds:
        ledger.attempt(s)
    nodes = ("6b", "6a") if args.node == "both" else (args.node,)
    trained = {}
    try:
        if "6b" in nodes:
            trained["6b"] = _train_6b(session, training, tables, results, leak)
        if "6a" in nodes:
            trained["6a"] = _train_6a(session, training, results, seeds)
    except BaseException as exc:
        for s in seeds:
            with_reason = f"{type(exc).__name__}: {exc}"
            try:
                ledger.record(s, "failed", with_reason)
            except Exception:
                pass
        raise
    for s in seeds:
        ledger.record(s, "completed")

    cutoffs, hashes = {}, {}
    for node in nodes:
        release_path = session.run_dir / "release" / f"{node}.joblib"
        bundle = (trained[node].release.bundle if node == "6b" else trained[node]["release"].bundle)
        cutoffs[node] = {"medium": float(bundle["thresholds"]["medium"]),
                         "high": float(bundle["thresholds"]["high"])}
        hashes[node] = sha256_file(release_path)
    write_freeze(session.run_dir, cutoffs=cutoffs, candidate_sha256=hashes,
                 switch_rule_sha256=session.run["switch_rule_sha256"], is_ignored=session.guard)
    _update_header(session)
    for node in nodes:
        print(f"Node {node} release cutoffs: medium {cutoffs[node]['medium']:.3f}, "
              f"high {cutoffs[node]['high']:.3f}; candidate sha256 {hashes[node]}")
    print(f"frozen.json written: {session.run_dir / 'frozen.json'}")
    waivers = _waiver_lines(session)
    print("waivers: " + ("none" if not waivers else ""))
    for line in waivers:
        print("  " + line)
    return EXIT_OK


# ------------------------------------------------------------------------------ compare
@dataclass
class Bundle:
    loaded: Loaded
    table_b: object
    rows_a: Optional[object]


def _bundle(session: Session, item: Loaded, want_rows: bool) -> Bundle:
    from .features_a import load_row_table
    from .features_b import build_feature_table
    th = session.config.thresholds
    table_b = build_feature_table(item.path, cache_root=session.cache_dir, is_ignored=session.guard,
                                  skip_max_share=th.skip_max_share)
    rows_a = None
    if want_rows:
        rows_a = load_row_table(item.path, cache_root=session.cache_dir, is_ignored=session.guard,
                                row_exclusion_max_share=th.row_exclusion_max_share)
    return Bundle(item, table_b, rows_a)


def make_docs(bundle: Bundle, split: Optional[str] = None) -> list:
    from .evaluation import EvalDoc
    from .features_b import EXTRACT_COLUMNS
    refs = {d.doc_id: d for d in bundle.loaded.corpus.docs}
    rows_by, empty = {}, None
    if bundle.rows_a is not None:
        all_rows = bundle.rows_a.rows
        empty = all_rows.iloc[0:0]
        for doc_id, grp in all_rows.groupby("doc_id", sort=False):
            rows_by[doc_id] = grp.reset_index(drop=True)
    docs = []
    for rec in bundle.table_b.frame.to_dict("records"):
        if split is not None and rec["split"] != split:
            continue
        ref = refs[rec["doc_id"]]
        docs.append(EvalDoc(
            doc_id=f"{ref.corpus_key}:{ref.doc_id}", is_tampered=bool(ref.is_tampered),
            features={c: float(rec[c]) for c in EXTRACT_COLUMNS},
            rows=rows_by.get(ref.doc_id, empty) if bundle.rows_a is not None else None,
            layout=ref.layout, n_months=ref.n_months, tamper_types=tuple(ref.tamper_types),
            tamper_method=ref.tamper_method, genuine_resaved=bool(ref.genuine_resaved)))
    return docs


def _byte_docs(items, extract) -> list:
    """items: (doc_id, path, is_tampered, file) read-only; unreadable documents are skipped."""
    from .evaluation import EvalDoc
    from .sources import open_source
    docs = []
    for doc_id, path, tampered, file in items:
        try:
            feats = open_source(path, extract).value
        except Exception as exc:
            print(f"skipped {doc_id}: {type(exc).__name__}")
            continue
        if feats is not None:
            docs.append(EvalDoc(doc_id=doc_id, is_tampered=tampered, features=feats, file=file))
    return docs


def _source_loaders(session: Session) -> dict:
    """Loaders for the Forgery_Dataset and Real_Style datasets (Node 6b only; read-only bytes)."""
    from fraud_detect.ml import statement_features as sf
    from . import sources
    from .evaluation import EvalDataset

    cfg = session.config
    loaders = {}
    forgery = sources.load_forgery(cfg.forgery_dir)
    for ds, status in forgery.dataset_status().items():
        if status == "ok":
            def load(ds=ds):
                items = [(d.path.name, d.path, d.is_tampered, None) for d in forgery.dataset_docs(ds)]
                return EvalDataset(ds, sources.SOURCE_FORGERY,
                                   tuple(_byte_docs(items, lambda buf: sf.extract_features(buf))))
            loaders[ds] = load
    real = sources.load_real_style(cfg.real_style_dir)
    if real.files:
        def load_real():
            items = [(name, r.path, REAL_STYLE_LABELS.get(name, False), name) for name, r in real.files.items()]
            return EvalDataset("real_style", sources.SOURCE_REAL_STYLE,
                               tuple(_byte_docs(items, lambda buf: sf.extract_features(buf))))
        loaders["real_style"] = load_real
    return loaders, real


def _awkward_loader():
    from fraud_detect.ml import statement_features as sf
    from . import sources
    from .awkward import prepare_awkward_set
    from .evaluation import EvalDataset, EvalDoc

    def load():
        outcome = prepare_awkward_set()
        if not outcome.ok:
            raise RuntimeError(outcome.message)
        docs = []
        for d in outcome.awkward_set.docs:
            feats = sf.extract_features(io.BytesIO(d.pdf))
            if feats is not None:
                docs.append(EvalDoc(doc_id=d.doc_id, is_tampered=d.is_tampered, features=feats,
                                    layout=d.layout, tamper_types=(d.tamper_type,) if d.tamper_type else (),
                                    case_type=d.case_type))
        return EvalDataset("awkward", sources.SOURCE_AWKWARD, tuple(docs))
    return load


class _UnitVault:
    """The opened vault with the cutoffs of one candidate (per-seed candidates use their own)."""

    def __init__(self, vault, cutoffs=None):
        self._vault, self._cutoffs = vault, cutoffs

    def load(self, dataset_id):
        return self._vault.load(dataset_id)

    def cutoffs(self, node):
        if self._cutoffs is None:
            return self._vault.cutoffs(node)
        from .vault import FrozenCutoffs
        return FrozenCutoffs(medium=float(self._cutoffs["medium"]), high=float(self._cutoffs["high"]))

    def record_score(self, key, values):
        self._vault.record_score(key, values)


def _matched(session: Session, node: str, current, val_docs):
    """Cutoffs for the Current model from the same rule on validation scores, guard off (D3)."""
    import numpy as np
    from . import baseline
    from .cutoffs import choose_cutoffs
    if current.status != baseline.STATUS_OK:
        return current, None
    try:
        if node == "6b":
            sc = baseline.score_b(current, [d.features for d in val_docs], dataset="validation", guard=False)
            if sc.status != baseline.STATUS_OK:
                raise RuntimeError(sc.detail)
            probs, genuine = sc.probs, [not d.is_tampered for d in val_docs]
            rule = session.cutoff_rule("document")
        else:
            import pandas as pd
            usable = [d for d in val_docs if not (d.is_tampered and not bool((d.rows["y"] == 1).any()))]
            sc = baseline.score_a(current, pd.concat([d.rows for d in usable], ignore_index=True),
                                  dataset="validation")
            if sc.status != baseline.STATUS_OK:
                raise RuntimeError(sc.detail)
            probs, pos = [], 0
            for d in usable:
                n = len(d.rows)
                probs.append(float(np.max(sc.probs[pos:pos + n])) if n else 0.0)
                pos += n
            genuine = [not d.is_tampered for d in usable]
            rule = session.cutoff_rule("statement_max_row")
        return current, choose_cutoffs(probs, genuine, rule)
    except Exception as exc:
        print(f"Current model, Node {node}: matched cutoffs could not be chosen "
              f"({type(exc).__name__}: {exc}); reported as baseline failed")
        return dataclasses.replace(current, status=baseline.UNLOADABLE), None


def collect_results(run_dir: Path) -> dict:
    out: dict = {}
    root = run_dir / "results"
    for p in sorted(root.rglob("*.json")) if root.is_dir() else []:
        node, dataset, name = p.relative_to(root).parts
        out.setdefault(node, {}).setdefault(dataset, {})[name[:-5]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def cmd_compare(session: Session, args) -> int:
    import joblib
    from . import baseline, evaluation
    from .evaluation import EvalDataset
    from .modelops import MODEL_FILES
    from .monitor import Monitor
    from .vault import HeldOutVault, VaultClosed, unchanged_since_run_start

    from .comparison_report import banner_lines
    for line in banner_lines():  # the console output opens with the banner (Requirement 17.x)
        print(line)
    cfg, th = session.config, session.config.thresholds
    run_dir = session.run_dir
    monitor = Monitor(run_dir, is_ignored=session.guard, clock=session.clock,
                      rule_c_scope="all" if cfg.rule_c_scope == "all" else "false_alarms_only",
                      implausible_auc=th.implausible_auc)
    if monitor.is_stopped:
        raise Stop(4, "the STOP lock is in place: an implausible measurement is waiting for a decision "
                      "(decide investigation --decision continue|abort)")

    # The vault opens only on a valid freeze record for the current run.json.
    training, held = load_corpora(session)
    shell = HeldOutVault(run_dir, {}, is_ignored=session.guard, clock=session.clock)
    try:
        frozen_nodes = tuple(n for n in ("6a", "6b") if n in shell.open().freeze.cutoffs)
    except VaultClosed as exc:
        raise Stop(EXIT_INTEGRITY, f"vault is closed: {exc}. Run train to write a valid frozen.json "
                                   f"for the current run.json.") from None

    want_rows = "6a" in frozen_nodes
    bundles = [_bundle(session, x, want_rows) for x in training]
    held_bundle = _bundle(session, held, want_rows)

    datasets: Dict[str, EvalDataset] = {}

    def memo(ds_id, build):
        def load():
            if ds_id not in datasets:
                datasets[ds_id] = build()
            return datasets[ds_id]
        return load

    loaders = {
        "test": memo("test", lambda: EvalDataset("test", "generator_corpus", tuple(
            d for b in bundles for d in make_docs(b, "test")))),
        "heldout_layout": memo("heldout_layout", lambda: EvalDataset("heldout_layout", "generator_corpus", tuple(
            d for b in bundles for d in make_docs(b, "test_unseen_layout")))),
        "heldout_seed": memo("heldout_seed", lambda: EvalDataset("heldout_seed", "generator_corpus",
                                                                  tuple(make_docs(held_bundle)))),
    }
    present = [ds for ds in ROW_DATASETS if len(loaders[ds]().docs) > 0]
    loaders = {ds: loaders[ds] for ds in present}
    source_loaders, real = _source_loaders(session)
    loaders.update({k: memo(k, v) for k, v in source_loaders.items()})
    if not cfg.plumbing:
        loaders["awkward"] = memo("awkward", _awkward_loader())

    vault = HeldOutVault(run_dir, loaders, is_ignored=session.guard, clock=session.clock).open()
    if not unchanged_since_run_start(run_dir, session.repo_root):
        print("note: generator and detector code changed since the run started")

    val_docs = [d for b in bundles for d in make_docs(b, "validation")]
    failures: List[tuple] = []
    seeds = list(cfg.seeds)
    for node in frozen_nodes:
        current = baseline.load_current(node, session.models_dir / MODEL_FILES[node])
        current, matched = _matched(session, node, current, val_docs)
        candidates = []
        for s in seeds:
            path = run_dir / "models" / node / f"seed-{s}.joblib"
            if path.is_file():
                candidates.append((s, joblib.load(path), _UnitVault(vault, None)))
        candidates.append(("release", joblib.load(run_dir / "release" / f"{node}.joblib"), vault))
        for ds in DATASET_ORDER:
            if ds not in loaders or (node == "6a" and ds not in ROW_DATASETS):
                continue
            for label, bundle, uv in candidates:
                if label != "release":
                    uv = _UnitVault(vault, bundle["thresholds"])
                try:
                    out = evaluation.run_unit(
                        uv, node, ds, label, candidate=bundle, current=current, matched=matched,
                        results_root=run_dir / "results", monitor=monitor, is_ignored=session.guard)
                    print(f"{'kept' if out.skipped else 'wrote'} result: Node {node} {ds} {label}")
                except Exception as exc:
                    if runner_stop(exc):
                        raise
                    failures.append((node, ds, label, f"{type(exc).__name__}: {exc}"))
                    print(f"unit failed: Node {node} {ds} {label}: {type(exc).__name__}: {exc}")

    report = build_report(session, training, held, bundles, held_bundle, real, failures, seeds)
    from .comparison_report import render_console, write_report
    write_report(run_dir, report, is_ignored=session.guard)
    print(render_console(report))
    return EXIT_FAILURE if failures else EXIT_OK


def runner_stop(exc: Exception) -> bool:
    """True for exceptions that must end the command and not be recorded as one failed unit."""
    from .monitor import MeasurementStop, StopLockActive
    from .safe_io import WriteGuardError
    from .vault import VaultError
    return isinstance(exc, (MeasurementStop, StopLockActive, WriteGuardError, VaultError, Stop))


# ------------------------------------------------------------------------------ evidence
def _mean(values):
    return None if not values or any(v is None for v in values) else sum(values) / len(values)


def _pct(rate):
    return None if rate is None or rate.get("value") is None else 100.0 * rate["value"]


def source_metrics(node: str, results: list):
    from .metrics import summarise_seeds
    from .switch_rule import SourceMetrics
    if not results:
        return SourceMetrics(missing=True)
    cands = [r["candidate"]["matched"] for r in results]
    curs = [(r.get("current") or {}).get("matched") for r in results]
    indicative = any(b["flag"] != "ok" for b in cands)
    if any(c is None for c in curs):
        return SourceMetrics(indicative_only=indicative)
    if node == "6b":
        aucs = [b["roc_auc"] for b in cands]
        std = summarise_seeds(aucs).std if aucs and all(a is not None for a in aucs) else None
        return SourceMetrics(
            indicative_only=indicative, cand_auc=_mean(aucs), cur_auc=_mean([b["roc_auc"] for b in curs]),
            cand_auc_std=std,
            cand_catch_medium=_mean([_pct(b["catch_rate"]) for b in cands]),
            cur_catch_medium=_mean([_pct(b["catch_rate"]) for b in curs]),
            cand_fa_medium=_mean([_pct(b["false_alarm_rate"]) for b in cands]),
            cur_fa_medium=_mean([_pct(b["false_alarm_rate"]) for b in curs]))
    return SourceMetrics(
        indicative_only=indicative, cand_pr_auc=_mean([b["pr_auc"] for b in cands]),
        cur_pr_auc=_mean([b["pr_auc"] for b in curs]),
        cand_fa_stmt=_mean([_pct(b["false_alarm_rate"]) for b in cands]),
        cur_fa_stmt=_mean([_pct(b["false_alarm_rate"]) for b in curs]))


def node_decision(rule, node: str, node_results: dict, awkward_reviewed: bool):
    from .switch_rule import Evidence, decide
    states = {r["current_state"] for ds in node_results.values() for r in ds.values()}
    if not states or states == {"ok"}:
        baseline_state = "ok"
    elif "baseline failed" in states:
        baseline_state = "baseline failed"
    else:
        baseline_state = "no baseline"

    def pick(source, release):
        return [r for lab, r in node_results.get(source, {}).items() if (lab == "release") == release]

    seed_metrics = {s: source_metrics(node, pick(s, False)) for s in rule.decisive}
    release_metrics = {s: source_metrics(node, pick(s, True)) for s in rule.decisive}
    hn_c = hn_u = 0.0
    awk = pick("awkward", False)
    if awk:
        hn_c = _mean([_pct(r["candidate"]["matched"]["hard_negative_fa"]) for r in awk]) or 0.0
        cur = [(r.get("current") or {}).get("matched") for r in awk]
        hn_u = (_mean([_pct(b["hard_negative_fa"]) for b in cur]) if all(cur) else None) or 0.0
    base = Evidence(baseline_state=baseline_state, metrics=seed_metrics, awkward_reviewed=awkward_reviewed,
                    hn_fa_cand=hn_c, hn_fa_cur=hn_u)
    has_release = any(node_results.get(s, {}).get("release") for s in rule.decisive)
    release_ok = has_release and decide(
        rule, node, dataclasses.replace(base, metrics=release_metrics)).action == "switch"
    return decide(rule, node, dataclasses.replace(base, release_passes_same_checks=release_ok))


def switch_decisions(session: Session) -> dict:
    rule = session.rule()
    results = collect_results(session.run_dir)
    reviewed = any(e["kind"] == "awkward-review" for e in session.decisions().entries())
    return {node: node_decision(rule, node, results.get(node, {}), reviewed) for node in ("6a", "6b")
            if node in results}


# ------------------------------------------------------------------------------ report
def _metric_stats(name, values, n_pos, n_neg, indicative):
    from .comparison_report import MetricStat
    from .metrics import summarise_seeds
    if not values or any(v is None for v in values):
        return MetricStat(name, None, None, None, None, n_pos, n_neg, False, indicative)
    s = summarise_seeds(values)
    return MetricStat(name, s.mean, s.min, s.max, s.std, n_pos, n_neg, True, indicative)


def report_rows(results: dict) -> tuple:
    from .comparison_report import ResultRow
    rows = []
    for node in sorted(results):
        for ds in DATASET_ORDER:
            by_label = results[node].get(ds)
            if not by_label:
                continue
            groups = {"seeds": [r for lab, r in by_label.items() if lab != "release"],
                      "release": [r for lab, r in by_label.items() if lab == "release"]}
            for group, items in groups.items():
                if not items:
                    continue
                for model, key in (("Candidate" if group == "seeds" else "Candidate (release)", "candidate"),
                                   ("Current" if group == "seeds" else "Current (release)", "current")):
                    bases = [(r.get(key) or {}).get("matched") for r in items]
                    if any(b is None for b in bases):
                        continue
                    first = bases[0]
                    indicative = any(b["flag"] != "ok" for b in bases)
                    metrics = (
                        _metric_stats("roc_auc", [b["roc_auc"] for b in bases], first["n_pos"], first["n_neg"], indicative),
                        _metric_stats("pr_auc", [b["pr_auc"] for b in bases], first["n_pos"], first["n_neg"], indicative),
                        _metric_stats("catch_rate", [_pct(b["catch_rate"]) for b in bases], first["n_pos"], first["n_neg"], indicative),
                        _metric_stats("false_alarm_rate", [_pct(b["false_alarm_rate"]) for b in bases], first["n_pos"], first["n_neg"], indicative),
                    )
                    rows.append(ResultRow(node=node, source=items[0]["source"], split=DATASET_SPLIT[ds],
                                          model=model, metrics=metrics))
    return tuple(rows)


def build_report(session, training, held, bundles, held_bundle, real, failures, seeds):
    from collections import Counter
    from .audit import AUDIT_V1
    from .comparison_report import AuditRow, ComparisonReport
    from .gnn_eval import lookup_coverage
    from .sources import real_style_report_entries
    from .vault import unchanged_since_run_start

    cfg = session.config
    run_dir = session.run_dir
    results = collect_results(run_dir)
    rule = session.rule()
    reviewed = any(e["kind"] == "awkward-review" for e in session.decisions().entries())
    texts = [node_decision(rule, node, results[node], reviewed).report_text() for node in sorted(results)]

    corpora = [x.corpus for x in training] + [held.corpus]
    split_counts, layout_counts = Counter(), Counter()
    for c in corpora:
        split_counts.update(c.split_counts())
        layout_counts.update(c.layout_counts())
    skipped: Dict[str, list] = {}
    for b in bundles + [held_bundle]:
        for split, rep in b.table_b.skip_report.items():
            tot = skipped.setdefault(split, [0, 0])
            tot[0] += rep["skipped"]
            tot[1] += rep["total"]
    skipped = {s: (n, 100.0 * n / t if t else 0.0) for s, (n, t) in skipped.items() if n}

    failed_seeds = {}
    for node, ds, label, why in failures:
        if label != "release":
            failed_seeds.setdefault(label, why)
    seeds_attempted = {s: ("completed" if s not in failed_seeds else f"failed: {failed_seeds[s]}")
                       for s in seeds}

    leak_auc = {}
    leak_path = run_dir / "leak.json"
    if leak_path.is_file():
        leak_auc = {r["feature"]: r["auc"] for r in json.loads(leak_path.read_text(encoding="utf-8"))["rows"]}
    audit_rows = tuple(AuditRow(e.name, e.decision, e.measures, leak_auc.get(e.name), e.note)
                       for e in AUDIT_V1.entries if e.section == "model_feature" and e.node == "6b")

    uens = {d.uen for d in held.corpus.docs if d.uen}
    gnn_path = session.models_dir / "gnn_scores.json"
    store = None
    try:
        from fraud_detect.ml.gnn.gnn_lookup import load_scores
        store = load_scores(str(gnn_path))
    except Exception:
        store = None
    coverage = lookup_coverage(uens, store)

    changes_path = run_dir / "post_test_changes.json"
    changes = (json.loads(changes_path.read_text(encoding="utf-8")).get("changes", [])
               if changes_path.is_file() else [])
    return ComparisonReport(
        header=header_from_run(session), corpus_seeds=tuple(cfg.seeds), held_out_layout=cfg.held_out_layout,
        split_counts=dict(split_counts), layout_counts=dict(layout_counts),
        total_documents=sum(split_counts.values()), seeds_attempted=seeds_attempted, skipped=skipped,
        real_style_files=tuple(real_style_report_entries(real)),
        decisions=tuple(dict(e) for e in session.decisions().entries()), post_test_changes=tuple(changes),
        code_unchanged=unchanged_since_run_start(run_dir, session.repo_root),
        gnn_coverage_pct=coverage.share or 0.0, audit_rows=audit_rows, rows=report_rows(results),
        switch_text="\n".join(texts))


# ------------------------------------------------------------------------------ switch / restore
def cmd_switch(session: Session, args) -> int:
    from . import modelops
    node = args.node
    if node == "6b" and args.size_guard is None:
        raise Stop(EXIT_INTEGRITY, "Node 6b needs an explicit --size-guard {candidate,off,current}")
    frozen_path = session.run_dir / "frozen.json"
    if not frozen_path.is_file():
        raise Stop(EXIT_INTEGRITY, f"{frozen_path} not found: train and compare this run first")
    expected = json.loads(frozen_path.read_text(encoding="utf-8"))["candidate_sha256"].get(node)
    if expected is None:
        raise Stop(EXIT_INTEGRITY, f"Node {node} was not trained in this run")
    decisions = switch_decisions(session)
    if node not in decisions:
        raise Stop(EXIT_INTEGRITY, "no comparison results for this node: run compare first")
    decision = decisions[node]
    print(decision.report_text())
    if decision.action != "switch" and not args.override_reason:
        raise Stop(EXIT_INTEGRITY, f"the Switch_Rule recommends keep for Node {node}; "
                                   f"give --override-reason TEXT to switch anyway")
    record = modelops.switch(
        node, session.run_id, size_guard=args.size_guard, override_reason=args.override_reason,
        models_dir=session.models_dir, backup_dir=session.backup_dir,
        candidate_path=session.run_dir / "release" / f"{node}.joblib", report_candidate_sha256=expected,
        is_ignored=session.guard)
    print(f"switched Node {node}: backup {record.backup_path} (sha256 {record.backup_sha256}); "
          f"installed sha256 {record.installed_sha256}")
    return EXIT_OK


def cmd_restore(session: Session, args) -> int:
    from . import modelops
    record = modelops.restore(args.node, models_dir=session.models_dir, backup_dir=session.backup_dir,
                              is_ignored=session.guard)
    print(f"restored Node {args.node} from backup (sha256 {record.restored_sha256})")
    return EXIT_OK
