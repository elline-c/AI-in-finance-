"""Temporary registration of the fictional companies (Requirements 13.2 to 13.7).

``registered_companies`` adds the companies to ``reference_data.ACRA_REGISTRY`` and
``CASHFLOW_BASELINE`` (changed in place, because Node 3 and the registry import the
dictionaries by name), keeps the live registry lookup off, and restores everything on
exit, also when the body raises. Source files are never written.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from fraud_detect import reference_data
from fraud_detect import registry


def _load(companies: Any) -> list[dict]:
    if isinstance(companies, (str, Path)):
        companies = json.loads(Path(companies).read_text(encoding="utf-8"))
    if isinstance(companies, dict):
        companies = companies.get("companies", [])
    return list(companies)


def _field(rec: Any, key: str, default: Any = None) -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def _normalise(rec: Any) -> tuple[str, str, dict | None]:
    uen = _field(rec, "uen")
    name = _field(rec, "name")
    if name is None:
        name = _field(rec, "registered_name")
    if not uen or name is None:
        raise ValueError(f"company record needs a uen and a name: {rec!r}")
    baseline = _field(rec, "baseline")
    return str(uen), str(name), (dict(baseline) if baseline is not None else None)


@contextlib.contextmanager
def registered_companies(companies: "Path | str | Iterable[Any]") -> Iterator[None]:
    """Add the fictional companies to ACRA_REGISTRY and CASHFLOW_BASELINE, then restore."""
    entries = [_normalise(r) for r in _load(companies)]
    acra = reference_data.ACRA_REGISTRY
    base = reference_data.CASHFLOW_BASELINE

    # Validate everything before touching anything.
    clashes = sorted({u for u, n, _ in entries if u in acra and acra[u] != n})
    if clashes:
        raise ValueError(
            "UEN already registered under a different name: " + ", ".join(clashes))

    missing = object()
    saved_acra = {u: acra.get(u, missing) for u, _, _ in entries}
    saved_base = {u: base.get(u, missing) for u, _, _ in entries}
    previous_live = registry._live_default
    try:
        registry.set_live_default(False)
        for uen, name, baseline in entries:
            acra[uen] = name
            if baseline is not None:
                base[uen] = baseline
        yield
    finally:
        for saved, target in ((saved_acra, acra), (saved_base, base)):
            for uen, old in saved.items():
                if old is missing:
                    target.pop(uen, None)
                else:
                    target[uen] = old
        registry.set_live_default(previous_live)
