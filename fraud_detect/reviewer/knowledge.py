"""Curated knowledge for the reviewer (retrieval-augmented generation, stage one).

Two small, versioned sources shipped with the code:

* ``knowledge_data/finding_codes.json``  one entry per finding code: meaning, why it matters, how
  to confirm it, common false alarms. Looked up by exact code, so it cannot retrieve the
  wrong entry.
* ``knowledge_data/policy.md``          short policy passages. Chosen by the routing cause plus a
  small TF-IDF match against the case's findings. Numbers are filled in from the code
  constants when loaded, so the text cannot drift from the thresholds in use.

Pure Python, no embedding model and no network, so retrieval is deterministic and cheap.
``version`` is a hash of both files and is stored with every set of notes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models import RiskReport
from ..nodes import risk_synthesis as node4
from ..reference_data import STRUCTURING_THRESHOLD
from . import breakdown
from .context import KnowledgeDoc

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge_data"
MAX_DOCS = 6
MAX_CODE_DOCS = 3
MAX_POLICY_DOCS = 3

# Confidence rules as Node 4 applies them (pinned in tests/ml_retraining/protected_hashes.json;
# tests/test_reviewer_knowledge.py compares these with the pinned numbers).
CONFIDENCE_RULES = {
    "CONF_DQ_STEP": 8, "CONF_DQ_MAX": 40, "CONF_NO_TXNS": 25, "CONF_CORROBORATION": 10,
    "CONF_SINGLE_DIMENSION": 12, "CONF_THIN_EVIDENCE": 10,
}

_ROUTING_POLICY = {
    breakdown.MATERIAL_FINDING: "POL-ROUTING",
    breakdown.LOW_CONFIDENCE: "POL-CONFIDENCE",
    breakdown.RISK_THRESHOLD: "POL-RISK-SCORE",
    breakdown.AUTO_CLEAR: "POL-ROUTING",
}
_STOPWORDS = frozenset(
    "the and for with that this from are was were has have had not but all any can may "
    "into than then they them their its out one two per via who how why what when where "
    "which while will would should could also only more most such each other about over "
    "under after before between because".split())
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
_SECTION_RE = re.compile(r"^##\s+([A-Z0-9-]+)\s*\|\s*(.+?)\s*$", re.MULTILINE)


def constants() -> dict[str, object]:
    """Values substituted into policy.md."""
    return {
        "MIN_CONFIDENCE_FOR_AUTO": node4.MIN_CONFIDENCE_FOR_AUTO,
        "AUTO_CLEAR_MAX_RISK": node4.AUTO_CLEAR_MAX_RISK,
        "CORROBORATION_BONUS": breakdown.CORROBORATION_BONUS,
        "MEDIUM_FLOOR": breakdown.MEDIUM_FLOOR,
        "CONFIDENCE_BASE": breakdown.CONFIDENCE_BASE,
        "STRUCTURING_THRESHOLD": f"{STRUCTURING_THRESHOLD:,.0f}",
        **CONFIDENCE_RULES,
    }


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def parse_policy(text: str, values: dict[str, object]) -> list[KnowledgeDoc]:
    """Split policy.md into passages and fill in the constants."""
    body = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    marks = list(_SECTION_RE.finditer(body))
    docs = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        passage = " ".join(body[m.end():end].split()).format_map(values)
        docs.append(KnowledgeDoc(doc_id=m.group(1), title=m.group(2), text=passage))
    return docs


def code_entry_text(entry: dict) -> str:
    """The text of one catalogue entry, in the fixed order the baseline reads it."""
    parts = [f"Meaning: {entry['meaning']}", f"Why it matters: {entry['why']}",
             f"How to confirm: {entry['confirm']}"]
    if entry.get("false_alarms"):
        parts.append(f"Common false alarms: {entry['false_alarms']}")
    return " ".join(parts)


@dataclass
class _Indexed:
    doc: KnowledgeDoc
    vec: dict[str, float]
    norm: float


class KnowledgeBase:
    def __init__(self, codes: dict[str, dict], policy: list[KnowledgeDoc], version: str) -> None:
        self.codes, self.policy, self.version = codes, policy, version
        df: Counter[str] = Counter()
        toks = {d.doc_id: Counter(_tokens(d.title + " " + d.text)) for d in policy}
        for c in toks.values():
            df.update(c.keys())
        n = max(len(policy), 1)
        self._idf = {t: math.log((n + 1) / (k + 1)) + 1.0 for t, k in df.items()}
        self._index = []
        for d in policy:
            vec = {t: (1 + math.log(c)) * self._idf[t] for t, c in toks[d.doc_id].items()}
            self._index.append(_Indexed(d, vec, math.sqrt(sum(v * v for v in vec.values())) or 1.0))

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "KnowledgeBase":
        d = Path(directory) if directory else KNOWLEDGE_DIR
        codes_raw = (d / "finding_codes.json").read_bytes()
        policy_raw = (d / "policy.md").read_bytes()
        version = "kb-" + hashlib.sha256(codes_raw + policy_raw).hexdigest()[:12]
        return cls(json.loads(codes_raw), parse_policy(policy_raw.decode("utf-8"), constants()), version)

    # ------------------------------------------------------------------ #
    def code_doc(self, code: str) -> Optional[KnowledgeDoc]:
        entry = self.codes.get(code)
        if not entry:
            return None
        return KnowledgeDoc(doc_id=f"CODE:{code}", title=f"{code}: {entry['title']}",
                            text=code_entry_text(entry))

    def search_policy(self, query: str, k: int, exclude: set[str]) -> list[KnowledgeDoc]:
        q = Counter(_tokens(query))
        qvec = {t: (1 + math.log(c)) * self._idf[t] for t, c in q.items() if t in self._idf}
        qnorm = math.sqrt(sum(v * v for v in qvec.values()))
        if not qnorm:
            return []
        scored = []
        for ix in self._index:
            if ix.doc.doc_id in exclude:
                continue
            dot = sum(w * ix.vec.get(t, 0.0) for t, w in qvec.items())
            if dot > 0:
                scored.append((dot / (qnorm * ix.norm), ix.doc.doc_id, ix.doc))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [d for _, _, d in scored[:k]]

    def retrieve(self, report: RiskReport) -> list[KnowledgeDoc]:
        """Catalogue entries for the heaviest finding codes, then policy passages."""
        weight: dict[str, float] = {}
        for f in report.findings:
            weight[f.code] = weight.get(f.code, 0.0) + f.weight
        ranked = sorted(weight, key=lambda c: (-weight[c], c))

        docs: list[KnowledgeDoc] = []
        for code in ranked:
            if len(docs) >= MAX_CODE_DOCS:
                break
            d = self.code_doc(code)
            if d:
                docs.append(d)

        have = {d.doc_id for d in docs}
        ex = report.explanation
        wanted = _ROUTING_POLICY.get(ex.routing.primary) if ex else None
        by_id = {d.doc_id: d for d in self.policy}
        if wanted in by_id:
            docs.append(by_id[wanted])
            have.add(wanted)

        query = " ".join([f.code.lower().replace("_", " ") + " " + f.title for f in report.findings]
                         + [l.label for l in (ex.confidence_lines if ex else [])]
                         + [ex.routing.statement if ex else ""])
        room = min(MAX_POLICY_DOCS - (1 if wanted in by_id else 0), MAX_DOCS - len(docs))
        if room > 0:
            docs.extend(self.search_policy(query, room, have))
        return docs[:MAX_DOCS]
