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
]

# Instruction-injection ("prompt injection") signatures. In this rule-based
# system there is no LLM to hijack, so these strings are INERT DATA — they can
# never become instructions. But their presence is a strong, deliberate attack
# signal: a company statement never contains text telling the reviewer to lower
# the risk or suppress findings. We detect these separately, capture the exact
# location + offending text as evidence, and raise a dedicated CRITICAL finding.
_PROMPT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instructions", re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|the\s+following)\s+instructions",
        re.IGNORECASE)),
    ("disregard", re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|your|the)\b", re.IGNORECASE)),
    ("force_low_risk", re.compile(
        r"mark\s+(this|the)?.{0,30}?(as\s+)?(low[\s-]*risk|safe|clean|approved?|not\s+fraud)",
        re.IGNORECASE)),
    ("suppress_findings", re.compile(
        r"(do\s*not|don'?t|never)\s+(report|flag|record|raise|log|escalate)\b",
        re.IGNORECASE)),
    ("force_approval", re.compile(
        r"(auto[\s-]*clear|approve\s+(this|the)\s+(loan|application|statement))",
        re.IGNORECASE)),
    ("role_hijack", re.compile(
        r"\b(you\s+are\s+now|act\s+as|system\s*:|as\s+an?\s+ai)\b", re.IGNORECASE)),
    ("instruction_marker", re.compile(
        r"###\s*(instruction|system|prompt)|\[/?(system|inst)\]", re.IGNORECASE)),
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
        ctx = _ScrubContext()
        clean = _sanitize_strings(clean, ctx)

        if ctx.oversized:
            findings.append(
                Finding(
                    self.node,
                    "AH_OVERSIZED_FIELD",
                    "Oversized text field(s) truncated",
                    Severity.MEDIUM,
                    f"{ctx.oversized} field(s) exceeded the {MAX_STRING_LEN}-"
                    "char limit and were truncated. Over-long fields are used to "
                    "exhaust memory or hide payloads.",
                    {"truncated_fields": ctx.oversized, "limit": MAX_STRING_LEN},
                )
            )

        if ctx.inj_hits:
            total = sum(ctx.inj_hits.values())
            findings.append(
                Finding(
                    self.node,
                    "AH_CONTENT_INJECTION",
                    "Injection payload detected in text fields",
                    Severity.CRITICAL,
                    "Suspicious markup/script/injection signatures were found in "
                    "free-text fields and were neutralized (HTML-escaped, control "
                    "characters stripped). Categories: "
                    + ", ".join(f"{k}×{v}" for k, v in sorted(ctx.inj_hits.items()))
                    + ". Legitimate bank statement text does not contain these.",
                    {"categories": ctx.inj_hits, "total_matches": total},
                )
            )

        # --- 6. embedded instruction / prompt injection ------------------ #
        # The crux of the "hidden instructions in the document" attack: text
        # that tries to command the system to lower the risk or hide findings.
        # It is INERT DATA here (no LLM), so it cannot change any agent's
        # instructions or permissions — but we record WHERE and WHAT as
        # evidence and raise a CRITICAL finding so it escalates to a human.
        if ctx.prompt_hits:
            cats = sorted({h["category"] for h in ctx.prompt_hits})
            findings.append(
                Finding(
                    self.node,
                    "AH_EMBEDDED_INSTRUCTION",
                    "Embedded instruction / prompt-injection in document text",
                    Severity.CRITICAL,
                    "The document contains text attempting to instruct the "
                    "screening system (e.g. to mark the application low-risk or "
                    "suppress findings). This content is treated as untrusted "
                    "DATA and cannot alter any agent's instructions, permissions, "
                    "or the risk score. It is recorded here as evidence of a "
                    "deliberate tampering attempt and forces human review. "
                    "Categories: " + ", ".join(cats) + ".",
                    {
                        "attempts": len(ctx.prompt_hits),
                        "categories": cats,
                        # location + extracted text for each hit (step 2):
                        "evidence": ctx.prompt_hits[:20],
                    },
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


class _ScrubContext:
    """Threads sanitization state (injection tallies, oversized counter, and
    located prompt-injection evidence) through the recursive walk."""

    __slots__ = ("inj_hits", "oversized", "prompt_hits")

    def __init__(self) -> None:
        self.inj_hits: dict[str, int] = {}
        self.oversized: int = 0
        # Each entry: {location, category, extracted_text} — step 2 of the
        # anti-hack flow: record WHERE and WHAT, as auditable evidence.
        self.prompt_hits: list[dict[str, str]] = []


def _snippet(s: str, match: re.Match[str], span: int = 60) -> str:
    """Extract the offending text around a match for the evidence record."""
    start = max(0, match.start() - span)
    end = min(len(s), match.end() + span)
    frag = s[start:end].strip()
    if start > 0:
        frag = "…" + frag
    if end < len(s):
        frag = frag + "…"
    return frag


def _scrub_string(s: str, ctx: _ScrubContext, location: str) -> str:
    """Normalize, size-cap, detect injection, strip control chars, HTML-escape.

    ``location`` is the dotted field path (e.g. ``transactions[2].description``)
    so prompt-injection evidence can point to exactly where the payload was.
    """
    # Unicode-normalize to fold homoglyph/compatibility tricks.
    s = unicodedata.normalize("NFKC", s)

    if len(s) > MAX_STRING_LEN:
        s = s[:MAX_STRING_LEN]
        ctx.oversized += 1

    # Markup/script/SQL injection (neutralized by escaping below).
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(s):
            ctx.inj_hits[name] = ctx.inj_hits.get(name, 0) + 1

    # Instruction/prompt injection — record location + extracted text as
    # evidence. This text is INERT DATA; recording it never executes it.
    for name, pattern in _PROMPT_INJECTION_PATTERNS:
        m = pattern.search(s)
        if m:
            ctx.prompt_hits.append({
                "location": location,
                "category": name,
                "extracted_text": _snippet(s, m),
            })

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


def _sanitize_strings(obj: Any, ctx: _ScrubContext, location: str = "") -> Any:
    """Recursively sanitize every string, and clamp absurd numeric magnitudes.

    Tracks the dotted ``location`` of each value for evidence purposes.
    """
    if isinstance(obj, str):
        return _scrub_string(obj, ctx, location or "<root>")
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
        out = {}
        for k, v in obj.items():
            child = f"{location}.{k}" if location else str(k)
            key = _scrub_string(str(k), ctx, child) if isinstance(k, str) else k
            out[key] = _sanitize_strings(v, ctx, child)
        return out
    if isinstance(obj, list):
        return [
            _sanitize_strings(v, ctx, f"{location}[{i}]")
            for i, v in enumerate(obj)
        ]
    return obj
