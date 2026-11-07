"""The tamper catalog: what each tamper type is, and the registry of implemented ones.

``TAMPER_INFO`` has descriptive metadata for all 16 types of ``schema.TAMPER_TYPES``
(group, possible methods, "should in principle be seen by" - descriptive, not a promise).
``CATALOG`` holds only the types that have a ``plan`` implemented so far.

How to register a tamper (from any module)::

    from fraud_detect.corpus.tamper.catalog import register

    @register("font_mismatch")
    def plan_font_mismatch(ledger, renderer, rng) -> TamperResult:
        ...

and make sure the module is imported at the bottom of this file (see
``_IMPLEMENTATIONS``), so ``CATALOG`` is complete whenever it is imported.
``CATALOG[name]`` is a ``Tamperer``: ``.name``, ``.group``, ``.methods``,
``.should_be_seen_by`` and ``.plan(ledger, renderer, rng) -> TamperResult``.
"""
from __future__ import annotations

import importlib

# The registry itself lives in ``_registry`` (no implementation imports), so the
# implementation modules can import ``register`` without a cycle in any import order.
from fraud_detect.corpus.tamper._registry import (  # noqa: F401
    CATALOG, PlanFn, TAMPER_INFO, TamperInfo, Tamperer, info_for, register)

# Modules that call ``register``. Later tasks append their module here
# (e.g. "pdf_ops", "content_ops", "behaviour_ops").
_IMPLEMENTATIONS: tuple[str, ...] = ("ledger_ops", "pdf_ops", "content", "behaviour", "subtle")

for _mod in _IMPLEMENTATIONS:
    importlib.import_module(f"fraud_detect.corpus.tamper.{_mod}")
