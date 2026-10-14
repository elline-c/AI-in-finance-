"""Node 0: Input Integrity & Anti-Hack Gate.

This runs *before* ingestion. Its job is NOT to detect financial tampering
(that is what nodes 2 and 3 do) but to defend the pipeline itself against an
adversary who controls the raw input and is trying to *attack the detector*
rather than merely forge a plausible statement.

Threat model (what a hacker tries to do to a rule-based screening pipeline):

  1. Score / routing manipulation — spoof the trust signals that drive the
     AUTO_CLEAR decision (`ocr_confidence`, `unreadable_pages`, `total_pages`),
     or smuggle in reserved keys the pipeline trusts (e.g. a pre-baked
     ``summary`` / ``decision`` / ``findings`` block hoping it is echoed back).

  2. Content injection — stuff free-text fields (description, counterparty,
     metadata, account holder) with HTML/JS (stored XSS in the review UI),
     template/report-injection payloads, or control characters that corrupt
     logs and the audit report.

  3. Resource abuse / DoS — enormous transaction arrays, pathologically long
     strings, absurd numeric magnitudes, or deeply nested structures designed
     to exhaust memory/CPU during parsing and scoring.

  4. Type confusion — values of the wrong type (dict where a number is
     expected, NaN/Infinity floats) intended to crash a node or slip past a
     coercion step.

Design principle (consistent with the rest of StatementGuard): the gate never
*rejects* on its own. It **sanitizes** what it safely can, **neutralizes**
dangerous content, and **emits findings** — so an attack attempt itself raises
the risk score and forces the human-in-the-loop review. A CRITICAL
anti-tamper finding guarantees escalation via the existing routing rules.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from ..contracts import Finding, Node, Severity

# --------------------------------------------------------------------------- #
# Hard bounds. These are deliberately generous for real statements but far
# below what a resource-exhaustion payload needs. Tune per deployment.
# --------------------------------------------------------------------------- #
MAX_TRANSACTIONS = 2_000           # a year of daily activity is ~a few hundred
MAX_SUPPORTING_DOCS = 500
MAX_STRING_LEN = 2_000             # longest legitimate free-text field
MAX_NESTING_DEPTH = 8              # statement JSON is shallow; deep == abuse
MAX_ABS_AMOUNT = 1e12              # SGD 1 trillion ceiling on any single value
MAX_TOTAL_KEYS = 50_000            # guard against key-count explosions

# Fields whose values must never be trusted from input: they are computed by
# the pipeline. If present in the raw payload, that is an injection attempt.
RESERVED_OUTPUT_KEYS = frozenset(
    {
        "summary",
        "decision",
        "risk_score",
        "risk_band",
        "confidence_score",
        "confidence_band",
        "findings",
        "routing_reasons",
        "recommendation",
        "data_quality",  # computed by ingestion, not supplied
    }
)

# Trust signals an attacker would spoof to force an AUTO_CLEAR.
_TRUST_SIGNAL_KEYS = ("ocr_confidence", "unreadable_pages", "total_pages")

# Content-injection signatures. Kept conservative to avoid false positives on
# legitimate statement text (which is almost never HTML/script/SQL).
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("html_script", re.compile(r"<\s*script\b", re.IGNORECASE)),
    ("html_tag", re.compile(r"<\s*(img|svg|iframe|object|embed|a|style)\b", re.IGNORECASE)),
    ("event_handler", re.compile(r"\bon(error|load|click|mouseover)\s*=", re.IGNORECASE)),
    ("js_uri", re.compile(r"javascript\s*:", re.IGNORECASE)),
    ("template_inject", re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}")),
    ("sql_meta", re.compile(r"(;\s*drop\s+table|union\s+select|--\s|/\*)", re.IGNORECASE)),
    # Classic "ignore your instructions" style payload — harmless here (no LLM)
    # but a strong signal of a probing attacker, worth flagging.
    ("prompt_inject", re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE)),
]

# Control characters (except tab/newline/carriage-return) have no place in
# extracted statement text and are used for log/terminal injection.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputIntegrityAgent:
    """Pre-ingestion defensive gate. Returns a *sanitized* copy of the raw
    payload plus the anti-tamper findings it raised."""

    node = Node.INGESTION  # reuses the ingestion node bucket for reporting

    def run(self, raw: Any) -> tuple[dict[str, Any], list[Finding]]:
        findings: list[Finding] = []

        # --- 0. top-level type check ------------------------------------- #
        if not isinstance(raw, dict):
            findings.append(
                Finding(
                    self.node,
                    "AH_MALFORMED_PAYLOAD",
                    "Input is not a JSON object",
                    Severity.CRITICAL,
                    "The raw payload is not a key/value document as the contract "
                    f"requires (got {type(raw).__name__}). Treated as empty and "
                    "escalated; a well-formed statement is always an object.",
                    {"received_type": type(raw).__name__},
                )
            )
            return {}, findings

        # --- 1. structural abuse: nesting depth & key explosion ---------- #
        depth = _max_depth(raw)
        if depth > MAX_NESTING_DEPTH:
            findings.append(
                Finding(
                    self.node,
                    "AH_EXCESSIVE_NESTING",
                    "Excessively nested input structure",
                    Severity.HIGH,
                    f"Input nests {depth} levels deep (limit {MAX_NESTING_DEPTH}). "
                    "A real statement is shallow; deep nesting is a parser "
                    "resource-exhaustion vector.",
                    {"depth": depth, "limit": MAX_NESTING_DEPTH},
                )
            )

        total_keys = _count_keys(raw)
        if total_keys > MAX_TOTAL_KEYS:
            findings.append(
                Finding(
                    self.node,
                    "AH_KEY_EXPLOSION",
                    "Abnormal number of keys in payload",
                    Severity.HIGH,
                    f"Payload contains {total_keys} keys (limit {MAX_TOTAL_KEYS}), "
                    "consistent with a resource-exhaustion attack rather than a "
                    "statement.",
                    {"total_keys": total_keys, "limit": MAX_TOTAL_KEYS},
                )
            )

        # --- 2. reserved output keys smuggled into the input ------------- #
        smuggled = sorted(k for k in raw if k in RESERVED_OUTPUT_KEYS)
        if smuggled:
            findings.append(
                Finding(
                    self.node,
                    "AH_RESERVED_KEY_INJECTION",
                    "Pipeline-reserved output keys present in input",
                    Severity.CRITICAL,
                    "The input supplied keys that only the pipeline is allowed to "
                    "produce (" + ", ".join(smuggled) + "). This is an attempt to "
                    "forge the verdict; the values were stripped before processing.",
                    {"smuggled_keys": smuggled},
                )
            )

        # Build the sanitized copy, dropping reserved keys entirely.
        clean: dict[str, Any] = {
            k: v for k, v in raw.items() if k not in RESERVED_OUTPUT_KEYS
        }

        # --- 3. trust-signal spoofing ------------------------------------ #
        self._check_trust_signals(clean, findings)

        # --- 4. array size caps (DoS) ------------------------------------ #
        clean["transactions"] = self._cap_list(
            clean.get("transactions"), "transactions", MAX_TRANSACTIONS, findings
        )
        clean["supporting_docs"] = self._cap_list(
            clean.get("supporting_docs"), "supporting_docs", MAX_SUPPORTING_DOCS,
            findings,
        )

        # --- 5. deep string sanitization + injection detection ----------- #
        inj_hits: dict[str, int] = {}
        oversized = {"count": 0}
        clean = _sanitize_strings(clean, inj_hits, oversized)

        if oversized["count"]:
            findings.append(
                Finding(
                    self.node,
                    "AH_OVERSIZED_FIELD",
                    "Oversized text field(s) truncated",
                    Severity.MEDIUM,
                    f"{oversized['count']} field(s) exceeded the {MAX_STRING_LEN}-"
                    "char limit and were truncated. Over-long fields are used to "
                    "exhaust memory or hide payloads.",
                    {"truncated_fields": oversized["count"], "limit": MAX_STRING_LEN},
                )
            )

        if inj_hits:
            total = sum(inj_hits.values())
            findings.append(
                Finding(
                    self.node,
                    "AH_CONTENT_INJECTION",
                    "Injection payload detected in text fields",
                    Severity.CRITICAL,
                    "Suspicious markup/script/injection signatures were found in "
                    "free-text fields and were neutralized (HTML-escaped, control "
                    "characters stripped). Categories: "
                    + ", ".join(f"{k}×{v}" for k, v in sorted(inj_hits.items()))
                    + ". Legitimate bank statement text does not contain these.",
                    {"categories": inj_hits, "total_matches": total},
                )
            )

        return clean, findings

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _check_trust_signals(
        self, clean: dict[str, Any], findings: list[Finding]
    ) -> None:
        """Clamp and sanity-check the signals that drive AUTO_CLEAR."""
        oc = clean.get("ocr_confidence", None)
        if oc is not None:
            n = _safe_float(oc)
            if n is None or n < 0.0 or n > 1.0:
                findings.append(
                    Finding(
                        self.node,
                        "AH_TRUST_SIGNAL_SPOOF",
                        "Implausible ocr_confidence value",
                        Severity.HIGH,
                        f"ocr_confidence={oc!r} is out of the valid 0..1 range or "
                        "non-numeric. Clamped to a conservative value so a forged "
                        "'perfect scan' signal cannot force an auto-clear.",
                        {"field": "ocr_confidence", "received": oc},
                    )
                )
                # Conservative: treat an impossible confidence as low.
                clean["ocr_confidence"] = 0.0 if (n is None or n < 0) else 1.0

        for key in ("unreadable_pages", "total_pages"):
            if key in clean:
                n = _safe_float(clean.get(key))
                if n is None or n < 0 or n != int(n) or n > 100_000:
                    findings.append(
                        Finding(
                            self.node,
                            "AH_TRUST_SIGNAL_SPOOF",
                            f"Implausible {key} value",
                            Severity.HIGH,
                            f"{key}={clean.get(key)!r} is not a plausible page "
                            "count. Reset to a safe default; page-count signals "
                            "feed the data-quality / routing decision.",
                            {"field": key, "received": clean.get(key)},
                        )
                    )
                    clean[key] = 0 if key == "unreadable_pages" else 1

    def _cap_list(
        self,
        value: Any,
        name: str,
        limit: int,
        findings: list[Finding],
    ) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            findings.append(
                Finding(
                    self.node,
                    "AH_TYPE_CONFUSION",
                    f"'{name}' is not a list",
                    Severity.HIGH,
                    f"Expected a list for '{name}' but received "
                    f"{type(value).__name__}; replaced with an empty list to "
                    "prevent a type-confusion crash downstream.",
                    {"field": name, "received_type": type(value).__name__},
                )
            )
            return []
        if len(value) > limit:
            findings.append(
                Finding(
                    self.node,
                    "AH_ARRAY_OVERFLOW",
                    f"'{name}' array exceeds safe size",
                    Severity.HIGH,
                    f"'{name}' had {len(value)} entries (limit {limit}); truncated. "
                    "Oversized arrays are a denial-of-service vector against the "
                    "reconciliation and scoring loops.",
                    {"field": name, "received": len(value), "limit": limit},
                )
            )
            return value[:limit]
        return value


# --------------------------------------------------------------------------- #
# module-level pure helpers
# --------------------------------------------------------------------------- #
def _safe_float(value: Any) -> float | None:
    """Float coercion that rejects NaN/Infinity (type-confusion vectors)."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(n) or math.isinf(n):
        return None
    return n


def _max_depth(obj: Any, _depth: int = 1) -> int:
    if _depth > MAX_NESTING_DEPTH + 2:  # stop early; we only need the verdict
        return _depth
    if isinstance(obj, dict):
        return max((_max_depth(v, _depth + 1) for v in obj.values()), default=_depth)
    if isinstance(obj, list):
        return max((_max_depth(v, _depth + 1) for v in obj), default=_depth)
    return _depth


def _count_keys(obj: Any) -> int:
    if isinstance(obj, dict):
        return len(obj) + sum(_count_keys(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_keys(v) for v in obj)
    return 0


def _scrub_string(s: str, inj_hits: dict[str, int], oversized: dict[str, int]) -> str:
    """Normalize, size-cap, detect injection, strip control chars, HTML-escape."""
    # Unicode-normalize to fold homoglyph/compatibility tricks.
    s = unicodedata.normalize("NFKC", s)

    if len(s) > MAX_STRING_LEN:
        s = s[:MAX_STRING_LEN]
        oversized["count"] += 1

    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(s):
            inj_hits[name] = inj_hits.get(name, 0) + 1

    # Strip control characters (log/terminal injection).
    s = _CONTROL_CHARS_RE.sub("", s)

    # HTML-escape so neutralized payloads render as inert text everywhere
    # (web template, JSON report, terminal). Jinja autoescaping is a second
    # layer; this makes the stored value itself safe.
    s = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
    return s


def _sanitize_strings(
    obj: Any, inj_hits: dict[str, int], oversized: dict[str, int]
) -> Any:
    """Recursively sanitize every string, and clamp absurd numeric magnitudes."""
    if isinstance(obj, str):
        return _scrub_string(obj, inj_hits, oversized)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # neutralize NaN/Infinity
        if abs(obj) > MAX_ABS_AMOUNT:
            return math.copysign(MAX_ABS_AMOUNT, obj)
        return obj
    if isinstance(obj, int):
        if abs(obj) > MAX_ABS_AMOUNT:
            return int(math.copysign(MAX_ABS_AMOUNT, obj))
        return obj
    if isinstance(obj, dict):
        return {
            _scrub_string(str(k), inj_hits, oversized) if isinstance(k, str) else k:
            _sanitize_strings(v, inj_hits, oversized)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_strings(v, inj_hits, oversized) for v in obj]
    return obj
