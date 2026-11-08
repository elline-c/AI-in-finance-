"""Implausible-measurement monitor, stop lock and seed ledger (Task 20.2). Standard library only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .decisions import DecisionLog
from .safe_io import atomic_write, ensure_ignored

EXIT_IMPLAUSIBLE = 4
UNSEEN_LAYOUT_SPLIT = "test_unseen_layout"
RULE_A_AUC = 0.99
RULE_B_AUC = 0.5
RULE_B_MIN_DOCS = 30
RULE_C_SCOPES = ("all", "false_alarms_only")
BLOCKED_CHANGES = ("train", "compare", "audit_new_version", "cutoff_change", "feature_change",
                   "generator_change", "detector_change")
SEED_OUTCOMES = ("completed", "failed", "stopped", "discarded")


class MeasurementStop(Exception):
    """An implausible measurement fired a rule; the run is locked."""

    exit_code = EXIT_IMPLAUSIBLE

    def __init__(self, record: dict):
        self.record = dict(record)
        super().__init__(
            f"implausible measurement (rule {record['rule']}): {record['measurement']}="
            f"{record['value']} source={record['source']} split={record['split']} "
            f"seed={record['master_seed']}")


class StopLockActive(Exception):
    """Raised while the STOP lock is in place."""


class InvestigationError(Exception):
    """decide() called with no open investigation."""


class SeedSubstitutionRefused(Exception):
    """A failed or stopped seed may not be swapped for another one."""


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    source: str
    split: str
    master_seed: int
    n_pos: int
    n_neg: int
    model: str = "candidate"
    roc_auc: Optional[float] = None
    flag: Optional[int] = None
    label: Optional[int] = None
    doc_count: Optional[int] = None
    manifest_count: Optional[int] = None


def fired_rules(m: Measurement, rule_c_scope: str = "all",
                implausible_auc: float = RULE_A_AUC) -> List[str]:
    if rule_c_scope not in RULE_C_SCOPES:
        raise ValueError(f"unknown rule_c_scope: {rule_c_scope!r}")
    fired: List[str] = []
    auc = m.roc_auc
    if (m.model == "candidate" and m.split == UNSEEN_LAYOUT_SPLIT
            and auc is not None and auc >= implausible_auc):
        fired.append("a")
    if (auc is not None and auc < RULE_B_AUC
            and m.n_pos >= RULE_B_MIN_DOCS and m.n_neg >= RULE_B_MIN_DOCS):
        fired.append("b")
    if (m.source == "real_style" and m.model == "candidate"
            and m.flag is not None and m.label is not None):
        if rule_c_scope == "all":
            if int(m.flag) != int(m.label):
                fired.append("c")
        elif int(m.flag) == 1 and int(m.label) == 0:
            fired.append("c")
    if (m.doc_count is not None and m.manifest_count is not None
            and m.doc_count != m.manifest_count):
        fired.append("d")
    return fired


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class Monitor:
    def __init__(self, run_dir, is_ignored: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None, rule_c_scope: str = "all",
                 implausible_auc: float = RULE_A_AUC):
        if rule_c_scope not in RULE_C_SCOPES:
            raise ValueError(f"unknown rule_c_scope: {rule_c_scope!r}")
        self.implausible_auc = implausible_auc
        self.run_dir = Path(run_dir)
        self._is_ignored = is_ignored
        self._clock = clock or _default_clock
        self.rule_c_scope = rule_c_scope

    @property
    def _stop_path(self) -> Path:
        return self.run_dir / "STOP"

    @property
    def _investigation_path(self) -> Path:
        return self.run_dir / "investigation.json"

    @property
    def is_stopped(self) -> bool:
        return self._stop_path.exists()

    def _log(self) -> DecisionLog:
        return DecisionLog(self.run_dir / "decisions.json", is_ignored=self._is_ignored,
                           clock=self._clock)

    def guard(self, change: str) -> None:
        if change not in BLOCKED_CHANGES:
            raise ValueError(f"unknown change: {change!r}")
        if self.is_stopped:
            raise StopLockActive(
                f"change '{change}' is blocked while the STOP lock is in place; "
                "record a 'continue' decision to lift it")

    def observe(self, m: Measurement) -> None:
        if self.is_stopped:
            raise StopLockActive("a measurement was refused: the STOP lock is in place")
        rules = fired_rules(m, self.rule_c_scope, self.implausible_auc)
        if not rules:
            return
        previous = _read_json(self._investigation_path, {})
        prev_n = int(str(previous.get("id", "I-0000"))[2:]) if previous else 0
        record = {
            "id": f"I-{prev_n + 1:04d}",
            "date": self._clock().isoformat(),
            "rule": rules[0],
            "rules": rules,
            "measurement": m.name,
            "value": m.value,
            "source": m.source,
            "split": m.split,
            "master_seed": m.master_seed,
            "n_pos": m.n_pos,
            "n_neg": m.n_neg,
        }
        atomic_write(self._investigation_path, json.dumps(record, indent=2, sort_keys=True),
                     is_ignored=self._is_ignored)
        atomic_write(self._stop_path, record["id"], is_ignored=self._is_ignored)
        print(f"IMPLAUSIBLE MEASUREMENT (rule {record['rule']}): {m.name} = {m.value}; "
              f"source={m.source}; split={m.split}; master_seed={m.master_seed}; "
              f"n_pos={m.n_pos}; n_neg={m.n_neg}. Run stopped (exit {EXIT_IMPLAUSIBLE}).")
        raise MeasurementStop(record)

    def decide(self, decision: str, reason: str, author: str) -> dict:
        record = _read_json(self._investigation_path, None)
        if record is None:
            raise InvestigationError("no investigation to decide on")
        log = self._log()
        if any(e["kind"] == "investigation" and e["target"] == record["id"]
               for e in log.entries()):
            raise InvestigationError(f"investigation {record['id']} already has a decision")
        entry = log.append("investigation", record["id"], decision, reason, author)
        if decision == "continue" and self._stop_path.exists():
            self._stop_path.unlink()
        return entry


class SeedLedger:
    def __init__(self, run_dir, is_ignored: Optional[Callable] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.path = Path(run_dir) / "seeds.json"
        self._is_ignored = is_ignored
        self._clock = clock or _default_clock

    def entries(self) -> List[dict]:
        return list(_read_json(self.path, {"seeds": []})["seeds"])

    def _save(self, entries: List[dict]) -> None:
        atomic_write(self.path, json.dumps({"seeds": entries}, indent=2, sort_keys=True),
                     is_ignored=self._is_ignored)

    def attempt(self, seed: int) -> None:
        entries = self.entries()
        if any(e["seed"] == seed for e in entries):
            return
        entries.append({"seed": seed, "outcome": "attempted", "reason": "",
                        "date": self._clock().isoformat()})
        self._save(entries)

    def record(self, seed: int, outcome: str, reason: str = "") -> None:
        if outcome not in SEED_OUTCOMES:
            raise ValueError(f"unknown outcome: {outcome!r}")
        entries = self.entries()
        for e in entries:
            if e["seed"] == seed:
                e["outcome"] = outcome
                e["reason"] = reason
                break
        else:
            raise ValueError(f"seed {seed} was never attempted")
        self._save(entries)

    def substitute(self, failed_seed: int, new_seed: int) -> None:
        entry: Optional[Dict] = next((e for e in self.entries() if e["seed"] == failed_seed), None)
        if entry is None:
            raise ValueError(f"seed {failed_seed} was never attempted")
        if entry["outcome"] in ("failed", "stopped"):
            raise SeedSubstitutionRefused(
                f"seed {failed_seed} {entry['outcome']}: it may not be replaced by {new_seed}")
        self.attempt(new_seed)


def record_post_test_change(run_dir, *, kind: str, description: str, reason: str,
                            trigger_measurement: str, trigger_value,
                            is_ignored: Optional[Callable] = None,
                            clock: Optional[Callable[[], datetime]] = None) -> dict:
    path = Path(run_dir) / "post_test_changes.json"
    ensure_ignored(path, is_ignored=is_ignored)
    entry = {
        "date": (clock or _default_clock)().isoformat(),
        "kind": kind,
        "description": description,
        "reason": reason,
        "trigger_measurement": trigger_measurement,
        "trigger_value": trigger_value,
    }
    changes = list(_read_json(path, {"changes": []})["changes"]) + [entry]
    atomic_write(path, json.dumps({"changes": changes}, indent=2, sort_keys=True),
                 is_ignored=is_ignored)
    return dict(entry)
