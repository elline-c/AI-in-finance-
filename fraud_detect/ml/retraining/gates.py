"""Gate framework (Task 1.3). Standard library only."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INTEGRITY = 2
EXIT_WAITING = 3
EXIT_IMPLAUSIBLE = 4


@dataclasses.dataclass(frozen=True)
class Gate:
    """A named check. ``check(ctx)`` returns the list of offenders (empty = pass)."""

    gate_id: str
    requirement: str
    waivable: bool
    check: Callable[[Any], Sequence[str]]


@dataclasses.dataclass(frozen=True)
class GateResult:
    gate_id: str
    requirement: str
    passed: bool
    offenders: Tuple[str, ...]
    waivable: bool
    waived_by: Optional[str]


def run_gates(gates: Iterable[Gate], ctx: Any, decisions) -> List[GateResult]:
    """Run every gate; a waivable failing gate passes only with a matching waive-gate entry.

    Integrity gates (``waivable=False``) can never be waived. Offenders stay
    visible on waived results so the report can still list them.
    """
    entries = decisions.entries() if decisions is not None else ()
    results: List[GateResult] = []
    for gate in gates:
        offenders = tuple(str(o) for o in gate.check(ctx))
        passed = not offenders
        waived_by: Optional[str] = None
        if not passed and gate.waivable:
            for entry in entries:
                if (entry["kind"] == "waive-gate" and entry["target"] == gate.gate_id
                        and entry["decision"] == "waive"):
                    waived_by = entry["id"]
                    passed = True
                    break
        results.append(GateResult(gate.gate_id, gate.requirement, passed, offenders,
                                  gate.waivable, waived_by))
    return results
