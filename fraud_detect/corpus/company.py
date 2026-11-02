"""Fictional company profiles for the corpus generator (Requirement 1.2).

Everything here is invented: the word lists are made-up words, the UEN-style id
is random (it follows the shape ``\\d{8,9}[A-Z]`` only), and no real company,
person or address is used. ``make_company`` depends only on its two arguments
(private ``random.Random`` from ``stable_seed``; no global state).

Naming conventions, because Node 3 reads descriptions as plain text:

* Customer and supplier names never contain ``UNKNOWN``, ``LLC``, ``HOLDINGS`` or
  ``TT`` anywhere, and never contain the whole words ``RENT``, ``CPF``,
  ``PAYROLL``, ``SALARY``, ``LOAN``, ``CAPITAL``, ``TRANSFER``, ``INTEREST``
  (or ``INV``, which would confuse invoice-reference reading).
  ``_clean`` enforces this on every generated name, so a future edit to the word
  lists cannot silently break it.
"""
from __future__ import annotations

import re
from random import Random
from typing import Callable, Optional

from .schema import CompanyProfile, Party
from .seeding import rng
from .uen_check import UenCheckError

_BANNED_SUBSTRINGS = ("UNKNOWN", "LLC", "HOLDINGS", "TT", "INV")
_BANNED_WORDS = ("RENT", "CPF", "PAYROLL", "SALARY", "LOAN", "CAPITAL", "TRANSFER", "INTEREST")

_FIRST = (
    "NORTHBROOK", "SILVERLINE", "KESTREL", "AMBERWAY", "BLUEFERN", "CEDARVALE",
    "DRIFTMARK", "EMBERLY", "FALCONRY", "GLENMORE", "HARBOURLY", "IRONWOOD",
    "JASPERLY", "KINGFERN", "LANTERN", "MERIDIAN", "NOVAREX", "OAKRIDGE",
    "PINEWELL", "QUILLON", "RIVERMEAD", "STONEBAY", "TIDEWAY", "UPLANDER",
    "VELMORA", "WILLOWAY", "YARROW", "ZENFIELD", "ASHCOMBE", "BRIGHTON",
    "COPPERLY", "DAWNMERE", "ELMSTEAD", "FERNDALE", "GRANWELL", "HOLLYBY",
    "ISLAMORE", "JUNIPERA", "KELPIE", "LARKSPUR", "MAPLEFORD", "NIMBUS",
    "ORCHARDLY", "PEBBLEY", "QUARTZEL", "RAVENSBY", "SUNDIAL", "THISTLEY",
)
_SECOND = (
    "FACILITIES", "SUPPLIES", "LOGISTICS", "CATERING", "DESIGN", "TRADING",
    "ENGINEERING", "SERVICES", "SOLUTIONS", "FOODS", "FREIGHT", "WORKS",
    "MEDIA", "CONSULTING", "SYSTEMS", "INDUSTRIES", "MARKETING", "DISTRIBUTION",
    "KITCHENS", "PACKAGING", "TECHNIQUES", "STUDIOS", "ENTERPRISES", "RESOURCES",
)
_UTILITIES = (
    "BRIGHTGRID POWER", "CLEARSTREAM WATER", "LUMENET TELECOM", "AEROLINK BROADBAND",
    "SUNVALE GAS", "METROFIBRE COMMS",
)
_LANDLORD_A = ("ORCHID", "MARBLEHILL", "CASCADE", "TANGLEWOOD", "HAVENCREST", "SKYBRIDGE")
_LANDLORD_B = ("PROPERTIES", "ESTATES", "REALTY", "BUILDINGS", "PLACE")
_STREETS = (
    "Aldermoor Road", "Beacon Street", "Corbel Avenue", "Dunmere Lane", "Eastcliff Drive",
    "Fenwick Way", "Garnet Crescent", "Harrow Close", "Isle Walk", "Jetty Road",
)
_ARCHETYPES = {
    # archetype: (base monthly turnover range, payroll share of turnover range)
    "services": ((60_000.0, 220_000.0), (0.28, 0.40)),
    "trading": ((90_000.0, 320_000.0), (0.12, 0.22)),
    "fnb": ((40_000.0, 150_000.0), (0.25, 0.35)),
    "logistics": ((70_000.0, 260_000.0), (0.20, 0.30)),
}


def _clean(name: str) -> bool:
    up = name.upper()
    if any(s in up for s in _BANNED_SUBSTRINGS):
        return False
    return not any(re.search(rf"\b{w}\b", up) for w in _BANNED_WORDS)


def _unique_names(r: Random, count: int, suffix: bool) -> list[str]:
    """Pick ``count`` distinct clean names from the word lists."""
    pairs = [(a, b) for a in _FIRST for b in _SECOND if _clean(f"{a} {b}")]
    r.shuffle(pairs)
    tail = " PTE. LTD." if suffix else ""
    return [f"{a} {b}{tail}" for a, b in pairs[:count]]


def _parties(r: Random, count: int) -> tuple[Party, ...]:
    names = _unique_names(r, count, suffix=True)
    # skewed sizes: a few big accounts, many small ones
    return tuple(Party(name=n, weight=round(r.paretovariate(1.6), 3)) for n in names)


def _uen(r: Random) -> str:
    digits = "".join(str(r.randrange(10)) for _ in range(r.choice((8, 9))))
    return digits + r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")


MAX_UEN_REDRAWS = 1000


def _free_uen(seed: int, company_id: str, first: str,
              is_taken: Optional[Callable[[str], bool]]) -> str:
    """Return ``first`` unless ``is_taken`` rejects it, then redraw (Requirement 20.2).

    Redraws come from their own child stream, so the rest of the profile (drawn from the
    main stream after the UEN) is not shifted by a clash.
    """
    if is_taken is None or not is_taken(first):
        return first
    stream = rng(seed, "company", company_id, "uen-redraw")
    for _ in range(MAX_UEN_REDRAWS):
        candidate = _uen(stream)
        if not is_taken(candidate):
            return candidate
    raise UenCheckError(f"no free UEN found for company {company_id} after "
                        f"{MAX_UEN_REDRAWS} tries")


def make_company(seed: int, company_id: str,
                 is_taken: Optional[Callable[[str], bool]] = None) -> CompanyProfile:
    """Build one fictional company; equal arguments give an equal profile.

    ``is_taken(uen)`` is optional. With ``None`` the result is exactly what it was before
    the UEN check existed. If it says the first UEN is taken, only the UEN changes.
    """
    r = rng(seed, "company", company_id)
    archetype = r.choice(sorted(_ARCHETYPES))
    (t_lo, t_hi), (p_lo, p_hi) = _ARCHETYPES[archetype]

    while True:
        name = f"{r.choice(_FIRST)} {r.choice(_SECOND)} PTE. LTD."
        if _clean(name.replace(" PTE. LTD.", "")):
            break

    customers = _parties(r, r.randint(5, 25))
    suppliers = _parties(r, r.randint(3, 8))
    # keep customers and suppliers distinct from each other and from the company
    taken = {c.name for c in customers} | {name}
    suppliers = tuple(s for s in suppliers if s.name not in taken) or (
        Party(name="HALCYON PACKAGING PTE. LTD.", weight=1.0),
    )

    base_turnover = round(r.uniform(t_lo, t_hi), 2)
    payroll_base = round(base_turnover * r.uniform(p_lo, p_hi), 2)
    block = r.randint(1, 99)
    postal = f"{r.randint(10, 82):02d}{r.randint(0, 9999):04d}"
    address = (f"{block} {r.choice(_STREETS)}", f"SINGAPORE {postal}")
    account = f"{r.randint(100, 999)}-{r.randint(100000, 999999)}-{r.randint(0, 99):02d}"

    uen = _free_uen(seed, company_id, _uen(r), is_taken)

    return CompanyProfile(
        company_id=company_id,
        name=name,
        uen=uen,
        address_lines=address,
        account_number=account,
        archetype=archetype,
        customers=customers,
        suppliers=suppliers,
        landlord=f"{r.choice(_LANDLORD_A)} {r.choice(_LANDLORD_B)} PTE. LTD.",
        payroll_base=payroll_base,
        seed=seed,
        first_invoice_number=r.randint(100, 4000),
        base_turnover=base_turnover,
    )


def utility_names(company: CompanyProfile) -> tuple[str, ...]:
    """Invented utility/telco names; a stable subset per company (used by ledger.py)."""
    r = rng(company.seed, "utilities", company.company_id)
    k = r.randint(2, 3)
    return tuple(r.sample(_UTILITIES, k))
