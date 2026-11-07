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
from dataclasses import dataclass
from typing import Callable

from fraud_detect.corpus.schema import ALL_TAMPER_TYPES, TamperResult


@dataclass(frozen=True)
class TamperInfo:
    name: str
    group: str                       # "Document" | "Hard negative" | "Content" | "Behaviour"
    methods: tuple[str, ...]         # methods the type may use (schema.TAMPER_METHODS)
    arithmetic: str                  # "yes" | "no" | "random"
    should_be_seen_by: tuple[str, ...]
    description: str


def _i(name, group, methods, arithmetic, seen, description) -> tuple[str, TamperInfo]:
    return name, TamperInfo(name, group, tuple(methods), arithmetic, tuple(seen), description)


TAMPER_INFO: dict[str, TamperInfo] = dict([
    _i("edited_amount", "Document", ("overlay", "rerender"), "random",
       ("Node3:FL_BALANCE_BREAK", "Node2:VF_FONT_MISMATCH"),
       "One amount changed; balances recomputed or left as printed (random, recorded)."),
    _i("altered_date", "Document", ("overlay", "rerender"), "yes",
       ("Node3:period checks", "Node2:VF_MIXED_DATE_FORMATS"),
       "One date changed; date_inconsistent when order or month is broken."),
    _i("broken_balance", "Document", ("overlay", "rerender"), "no",
       ("Node3:FL_BALANCE_BREAK",),
       "One printed balance changed, amounts untouched."),
    _i("font_mismatch", "Document", ("overlay", "rerender"), "random",
       ("Node2:VF_FONT_MISMATCH",),
       "The edited amount is drawn in Courier."),
    _i("metadata_inconsistency", "Document", ("file",), "yes",
       ("Node2:VF_EDIT_SOFTWARE", "Node2:VF_MODIFIED_AFTER_CREATION",
        "Node2:VF_CREATION_BEFORE_PERIOD"),
       "Editor-style Producer and later ModDate (variant: creation before period)."),
    _i("genuine_resaved", "Hard negative", ("file",), "yes", (),
       "Print/export Producer and later ModDate; is_tampered is false."),
    _i("fabricated_receipts", "Content", ("rerender",), "yes",
       ("Node3:FL_DUP_INVOICE_REF", "Node3:FL_TURNOVER_COST_MISMATCH"),
       "6 to 12 extra credits copying an existing receipt's customer and invoice number."),
    _i("cross_month_copy", "Content", ("rerender",), "yes",
       ("Node3:FL_DUP_INVOICE_REF",),
       "Receipt copy landing in a different month."),
    _i("turnover_inflation", "Content", ("rerender",), "yes",
       ("Node3:FL_TURNOVER_COST_MISMATCH",),
       "Extra credits with new invoice numbers in 2 or more months."),
    _i("structuring", "Behaviour", ("rerender",), "yes",
       ("Node3:FL_STRUCTURING",),
       "3 to 5 credits between 9,500 and 9,999.99 within 14 days (slow variant: over 30 days)."),
    _i("sudden_large_deposit", "Behaviour", ("rerender",), "yes",
       ("Node3:FL_SUDDEN_DEPOSIT",),
       "One credit above 3 times the average monthly credit (moderate variant: 1.5 to 2.5 times)."),
    _i("shell_counterparty", "Behaviour", ("rerender",), "yes",
       ("Node3:FL_UNVERIFIED_COUNTERPARTY",),
       "One credit of 50,000 or more from a name such as UNKNOWN TRADING LLC."),
    _i("rapid_pass_through", "Behaviour", ("rerender",), "yes", (),
       "Money in, almost the same out within 1 to 3 days; expected blind spot."),
    _i("dormant_reactivation", "Behaviour", ("rerender",), "yes", (),
       "Months of only bank charges, then large activity; expected blind spot."),
    _i("period_gap", "Behaviour", ("rerender",), "yes",
       ("Node3:FL_MONTH_MISSING", "Node3:FL_PERIOD_GAP"),
       "One middle month section removed."),
    _i("uen_mismatch", "Behaviour", ("content",), "yes",
       ("Node3:FL_UEN_NAME_MISMATCH",),
       "Printed holder name differs from the registered name."),
    _i("subtle_amount_edit", "Document", ("overlay", "rerender"), "random", (),
       "One amount changed by 1.0% to 4.9%; balances recomputed or left as printed."),
])

assert set(TAMPER_INFO) == set(ALL_TAMPER_TYPES), "TAMPER_INFO must cover every ALL_TAMPER_TYPES name"

PlanFn = Callable[..., TamperResult]


@dataclass(frozen=True)
class Tamperer:
    """A registered tamper type: metadata plus its ``plan`` function."""

    info: TamperInfo
    _plan: PlanFn

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def group(self) -> str:
        return self.info.group

    @property
    def methods(self) -> tuple[str, ...]:
        return self.info.methods

    @property
    def should_be_seen_by(self) -> tuple[str, ...]:
        return self.info.should_be_seen_by

    def plan(self, ledger, renderer, rng) -> TamperResult:
        return self._plan(ledger, renderer, rng)


CATALOG: dict[str, Tamperer] = {}


def register(name: str) -> Callable[[PlanFn], PlanFn]:
    """Decorator: register ``plan(ledger, renderer, rng) -> TamperResult`` under ``name``."""
    if name not in TAMPER_INFO:
        raise KeyError(f"unknown tamper type {name!r}")

    def deco(fn: PlanFn) -> PlanFn:
        CATALOG[name] = Tamperer(TAMPER_INFO[name], fn)
        return fn

    return deco


def info_for(name: str) -> TamperInfo:
    return TAMPER_INFO[name]
