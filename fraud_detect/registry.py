"""Company registry lookup used by the UEN cross-check in Node 3.

One function, ``lookup_entity(uen)``, answers "is this UEN a registered entity, and
under what name?". It reads from two sources, in this order:

1. **Local registry** (``reference_data.ACRA_REGISTRY``): always on, used for the
   fictional test companies and as an offline fallback. Local entries win.
2. **Live registry** (data.gov.sg "Registered Entities with Unique Entity Number",
   collection 1, published by ACRA). Off by default for library use, tests and
   ``demo.py``; the web apps turn it on while the server runs (``live_lookup_lifespan``).
   ``ACRA_LIVE_LOOKUP=1`` / ``=0`` overrides either way.

Why it is not simply always on: it makes a network call per UEN, and tests and the
audit trail are easier to reason about when the default is offline and deterministic.

Three outcomes are kept apart on purpose:

* ``found``        the UEN is registered (record has the registered name)
* ``not_found``    every enabled source answered and none knows this UEN
* ``unavailable``  the live source was enabled but could not answer (network error,
                   timeout, bad response). This is NOT evidence about the company, so
                   callers must not treat it as "not found".

Privacy: only the UEN, which is a public identifier, is sent to the live source. No
statement content, name or amount leaves the machine.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .reference_data import ACRA_REGISTRY

_log = logging.getLogger(__name__)

LIVE_ENV = "ACRA_LIVE_LOOKUP"

COLLECTION_ID = 1
COLLECTION_URL = (
    "https://api-production.data.gov.sg/v2/public/api/collections/{cid}/metadata"
)
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"
# Used when the collection metadata cannot be read: the two datasets of collection 1
# (ACRA-issued UENs, and UENs from the other issuing agencies).
FALLBACK_DATASETS = (
    "d_3f960c10fed6145404ca7b821f263b87",
    "d_b1d2b840ab9e993570c037b706b39bb8",
)

TIMEOUT_S = 5.0
FOUND_TTL_S = 24 * 3600        # a registration changes rarely
NOT_FOUND_TTL_S = 3600
UNAVAILABLE_TTL_S = 60         # retry soon, but do not hammer a failing service
_MAX_BYTES = 1_000_000

FOUND = "found"
NOT_FOUND = "not_found"
UNAVAILABLE = "unavailable"

_UEN_RE = re.compile(r"[0-9A-Z]{9,10}")
_DATASET_RE = re.compile(r"d_[0-9a-f]{16,64}")

JsonFetcher = Callable[[str], dict]


@dataclass(frozen=True)
class RegistryRecord:
    uen: str
    name: str
    status: Optional[str] = None      # e.g. "Registered" / "Deregistered" (live source only)
    source: str = "local"             # "local" | "data.gov.sg"


@dataclass(frozen=True)
class RegistryResult:
    state: str                        # FOUND | NOT_FOUND | UNAVAILABLE
    record: Optional[RegistryRecord] = None
    live_checked: bool = False        # True when the live source was consulted (or tried)
    detail: str = ""                  # short reason when unavailable (exception type only)


# --------------------------------------------------------------------------- #
# Name comparison
# --------------------------------------------------------------------------- #
def _norm_name(text: str) -> str:
    t = text.casefold().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)       # punctuation: "PTE. LTD." == "Pte Ltd"
    return " ".join(t.split())


def names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Case-, punctuation- and spacing-insensitive company-name comparison."""
    if not a or not b:
        return False
    return _norm_name(a) == _norm_name(b)


# --------------------------------------------------------------------------- #
# Live source
# --------------------------------------------------------------------------- #
_live_default = False     # what "not set" means; the web apps switch it on at startup


def set_live_default(value: bool) -> None:
    """Set what ``ACRA_LIVE_LOOKUP`` means when the variable is not set at all."""
    global _live_default
    _live_default = bool(value)


def live_enabled() -> bool:
    """Explicit ``ACRA_LIVE_LOOKUP`` (1/true/yes/on or 0/false/...) always wins; if it is
    not set, the process-wide default applies (off, unless a web app turned it on)."""
    raw = os.environ.get(LIVE_ENV)
    if raw is not None and raw.strip():
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return _live_default


@contextlib.asynccontextmanager
async def live_lookup_lifespan(app):
    """FastAPI lifespan for the screening web apps: live lookup on while the server runs.

    It only runs when the server really starts (or a test uses ``with TestClient(app)``),
    so merely importing an app, as the test suite does, leaves live lookup off. Setting
    ``ACRA_LIVE_LOOKUP=0`` turns it off for the server as well.
    """
    previous = _live_default
    set_live_default(True)
    try:
        yield
    finally:
        set_live_default(previous)


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context. Prefers the certifi CA bundle: some Python builds (notably
    the python.org macOS installer) ship without root certificates, which would make
    every live lookup fail. Certificate checking is never switched off."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the platform default
        return ssl.create_default_context()


def _http_get_json(url: str) -> dict:
    if not url.startswith("https://"):
        raise ValueError("only https URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "fraud-detect-prototype/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S,  # noqa: S310 - https only
                                context=_ssl_context()) as resp:
        body = resp.read(_MAX_BYTES + 1)
    if len(body) > _MAX_BYTES:
        raise ValueError("response too large")
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("unexpected JSON shape")
    return data


_lock = threading.Lock()
_cache: dict[str, tuple[float, RegistryResult]] = {}
_dataset_cache: dict[str, object] = {}


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _dataset_cache.clear()


def _dataset_ids(fetch: JsonFetcher) -> tuple[str, ...]:
    """Child datasets of collection 1, read from its metadata; fallback to known ids."""
    with _lock:
        cached = _dataset_cache.get("ids")
    if cached:
        return cached  # type: ignore[return-value]
    ids: tuple[str, ...] = FALLBACK_DATASETS
    try:
        meta = fetch(COLLECTION_URL.format(cid=COLLECTION_ID))
        child = meta["data"]["collectionMetadata"]["childDatasets"]
        valid = tuple(d for d in child if isinstance(d, str) and _DATASET_RE.fullmatch(d))
        if valid:
            ids = valid
    except Exception as exc:  # noqa: BLE001 - metadata is optional, fallback is fine
        _log.info("registry: collection metadata unavailable (%s); using fallback ids",
                  type(exc).__name__)
    with _lock:
        _dataset_cache["ids"] = ids
    return ids


def _query_live(uen: str, fetch: JsonFetcher) -> Optional[RegistryRecord]:
    for dataset_id in _dataset_ids(fetch):
        query = urllib.parse.urlencode({
            "resource_id": dataset_id,
            "limit": 1,
            "filters": json.dumps({"uen": uen}),
        })
        data = fetch(f"{DATASTORE_URL}?{query}")
        if not data.get("success"):
            raise ValueError("registry service reported failure")
        records = data["result"]["records"]
        if records:
            rec = records[0]
            return RegistryRecord(
                uen=str(rec.get("uen") or uen),
                name=str(rec["entity_name"]),
                status=rec.get("uen_status_desc"),
                source="data.gov.sg",
            )
    return None


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #
def lookup_entity(
    uen: Optional[str],
    *,
    live: Optional[bool] = None,
    fetch: Optional[JsonFetcher] = None,
    clock: Callable[[], float] = time.monotonic,
) -> RegistryResult:
    """Look a UEN up in the local registry, then (if enabled) the live registry.

    ``live=None`` follows the ``ACRA_LIVE_LOOKUP`` environment variable. ``fetch`` and
    ``clock`` exist so tests never touch the network. Never raises.
    """
    key = (uen or "").strip().upper()
    if key in ACRA_REGISTRY:
        return RegistryResult(FOUND, RegistryRecord(key, ACRA_REGISTRY[key], None, "local"))
    # also allow a differently-cased key in the local fixture
    for k, v in ACRA_REGISTRY.items():
        if k.upper() == key and key:
            return RegistryResult(FOUND, RegistryRecord(k, v, None, "local"))

    use_live = live_enabled() if live is None else live
    if not use_live or not _UEN_RE.fullmatch(key):
        return RegistryResult(NOT_FOUND)     # nothing else to ask (or malformed: never sent)

    now = clock()
    with _lock:
        hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]

    fetcher = fetch or _http_get_json
    try:
        rec = _query_live(key, fetcher)
    except Exception as exc:  # noqa: BLE001 - any failure means "could not answer"
        detail = type(exc).__name__
        _log.warning("registry: live lookup unavailable (%s)", detail)
        result = RegistryResult(UNAVAILABLE, None, True, detail)
        ttl = UNAVAILABLE_TTL_S
    else:
        if rec is not None:
            result, ttl = RegistryResult(FOUND, rec, True), FOUND_TTL_S
        else:
            result, ttl = RegistryResult(NOT_FOUND, None, True), NOT_FOUND_TTL_S
    with _lock:
        _cache[key] = (now + ttl, result)
    return result
