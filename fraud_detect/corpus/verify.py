"""Verification report for the synthetic corpus (read-only).

    python -m fraud_detect.corpus.verify --corpus data/corpus

Part 1 (this section, Requirement 14) measures Node 1 per layout: share of rows
read (genuine and tampered separately), share of documents whose replayed
balance equals the printed closing balance, share with matching header, the
chosen profile, the finding codes, and the overlay rows Node 1 dropped.

Part 2 (Nodes 2 to 4, Requirement 15) runs every measured document through the
pipeline (Nodes 1 to 4, in memory, no case store, live registry lookup off) with the
fictional companies registered once around all screening. It reports false alarms on
genuine documents, detection per tamper type, blind spots, figures per layout and
split, the decision breakdown and follow-up suggestions (text only).

The corpus is only read. Files are written only when ``--json`` / ``--markdown``
name them. Nothing touches a case store, ``data/`` or the audit chain.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence, TextIO

from fraud_detect.corpus.scope import registered_companies

DEFAULT_CORPUS_DIR = Path("data") / "corpus"
AMOUNT_TOL = 0.005
BALANCE_TOL = 0.01


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class DocRecord:
    doc_id: str
    layout: str
    path: Path
    is_tampered: bool
    genuine_resaved: bool
    tamper_method: str
    manifest: dict = field(repr=False, default_factory=dict)
    split: str = ""
    tamper_type: str = ""


@dataclass
class Node1Result:
    doc_id: str
    layout: str
    is_tampered: bool
    genuine_resaved: bool
    rows_expected: int
    rows_read: int
    profile: str
    balance_ok: bool
    header_ok: bool
    finding_codes: list
    dropped_overlay_rows: int
    error: Optional[str] = None


def _flag(v) -> bool:
    return str(v).strip() in ("1", "true", "True")


def load_corpus(corpus_dir) -> list[DocRecord]:
    """Read ``labels.csv`` and the per-document manifests, sorted by (layout, doc_id)."""
    root = Path(corpus_dir)
    records: list[DocRecord] = []
    with (root / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for line in csv.DictReader(fh):
            doc_id = line["doc_id"]
            manifest = json.loads(
                (root / "manifests" / f"{doc_id}.json").read_text(encoding="utf-8"))
            types = [t for t in (line.get("tamper_types") or "").split(";") if t]
            if not types:
                types = list(manifest.get("label", {}).get("tamper_types") or [])
            resaved = _flag(line.get("genuine_resaved", ""))
            records.append(DocRecord(
                doc_id=doc_id, layout=line["layout"],
                path=root / (line.get("file") or f"pdfs/{doc_id}.pdf"),
                is_tampered=_flag(line["is_tampered"]),
                genuine_resaved=resaved,
                tamper_method=line.get("tamper_method", "") or "",
                manifest=manifest,
                split=line.get("split") or manifest.get("split", "") or "",
                tamper_type=types[0] if types else ("genuine_resaved" if resaved else "")))
    records.sort(key=lambda r: (r.layout, r.doc_id))
    return records


def sample_records(records: Sequence[DocRecord], n: int) -> list[DocRecord]:
    """At most ``n`` records per layout: the first ones by doc_id."""
    by_layout: dict[str, list[DocRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda r: (r.layout, r.doc_id)):
        if len(by_layout[r.layout]) < n:
            by_layout[r.layout].append(r)
    return [r for layout in sorted(by_layout) for r in by_layout[layout]]


# --------------------------------------------------------------------------- #
# Node 1 measurement
# --------------------------------------------------------------------------- #
def _default_ingest(path: str):
    from fraud_detect.nodes.ingestion import ingest
    return ingest(path)


def _count_matched_rows(rows: Sequence[dict], txns) -> int:
    """Manifest rows matched in order (date and signed amount) against Node 1 rows."""
    j = 0
    matched = 0
    for row in rows:
        d = date.fromisoformat(row["date"])
        for k in range(j, len(txns)):
            t = txns[k]
            if (t.date == d and t.amount is not None
                    and abs(t.amount - row["amount"]) <= AMOUNT_TOL + 1e-9):
                matched += 1
                j = k + 1
                break
    return matched


def _dropped_overlay_rows(manifest: dict, txns) -> int:
    rows = manifest.get("rows", [])
    date_of = {r["row_id"]: r["date"] for r in rows}
    affected_per_date = Counter(
        date_of[a["row_id"]] for a in manifest.get("label", {}).get("affected_rows", [])
        if a.get("row_id") in date_of)
    expected = Counter(r["date"] for r in rows)
    got = Counter(t.date.isoformat() for t in txns if t.date is not None)
    return sum(min(max(expected[d] - got[d], 0), cap) for d, cap in affected_per_date.items())


def _header_ok(manifest: dict, ing) -> bool:
    months = manifest.get("months") or []
    if not months:
        return False
    company = manifest.get("company", {})
    y0, m0 = (int(x) for x in months[0].split("-"))
    y1, m1 = (int(x) for x in months[-1].split("-"))
    last_day = date.fromordinal(date(y1 + (m1 == 12), m1 % 12 + 1, 1).toordinal() - 1)
    return (ing.account_holder == company.get("registered_name")
            and ing.uen == company.get("uen")
            and ing.statement_period_start == date(y0, m0, 1)
            and ing.statement_period_end == last_day)


class _RecordPath(str):
    """The path string given to ``ingest``; it also exposes the record's fields.

    Real ingest functions see a plain ``str``. Injected test doubles may read
    ``manifest`` / ``path`` / ``doc_id`` straight from the argument instead.
    """

    def __new__(cls, record: "DocRecord"):
        obj = super().__new__(cls, str(record.path))
        obj._record = record
        return obj

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._record, name)


def _measure_one(record: DocRecord, ingest: Callable) -> Node1Result:
    m = record.manifest
    rows = m.get("rows", [])
    base = dict(doc_id=record.doc_id, layout=record.layout, is_tampered=record.is_tampered,
                genuine_resaved=record.genuine_resaved, rows_expected=len(rows))
    overlay = record.is_tampered and record.tamper_method == "overlay"
    try:
        ing = ingest(_RecordPath(record))
    except Exception as exc:  # the run continues; this document counts as 0 rows read
        return Node1Result(**base, rows_read=0, profile="", balance_ok=False, header_ok=False,
                           finding_codes=[],
                           dropped_overlay_rows=_dropped_overlay_rows(m, []) if overlay else 0,
                           error=type(exc).__name__)
    txns = list(ing.transactions)
    if ing.opening_balance is None or ing.closing_balance is None:
        balance_ok = False
    else:
        total = ing.opening_balance + sum(t.amount or 0.0 for t in txns)
        balance_ok = abs(total - ing.closing_balance) <= BALANCE_TOL
    return Node1Result(
        **base, rows_read=_count_matched_rows(rows, txns), profile=ing.layout_profile or "",
        balance_ok=balance_ok, header_ok=_header_ok(m, ing),
        finding_codes=sorted({f.code for f in ing.data_quality_findings}),
        dropped_overlay_rows=_dropped_overlay_rows(m, txns) if overlay else 0)


def measure_node1(records: Sequence[DocRecord], ingest: Optional[Callable] = None
                  ) -> list[Node1Result]:
    """One ``Node1Result`` per record, in record order."""
    ingest = ingest if ingest is not None else _default_ingest
    return [_measure_one(r, ingest) for r in records]


def _share(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def summarise_layout(results: Sequence[Node1Result]) -> dict:
    """Aggregate the results of ONE layout. Shares are fractions in 0..1."""
    def rows_share(group):
        return _share(sum(x.rows_read for x in group), sum(x.rows_expected for x in group))

    n = len(results)
    genuine = [x for x in results if not x.is_tampered]
    tampered = [x for x in results if x.is_tampered]
    overall = rows_share(results)
    return {
        "profile": ",".join(sorted({x.profile for x in results if x.profile})),
        "documents": n,
        "rows_share": overall if overall is not None else 0.0,
        "rows_share_genuine": rows_share(genuine),
        "rows_share_tampered": rows_share(tampered),
        "balance_share": sum(x.balance_ok for x in results) / n if n else 0.0,
        "header_share": sum(x.header_ok for x in results) / n if n else 0.0,
        "finding_codes": sorted({c for x in results for c in x.finding_codes}),
        "dropped_overlay_rows": sum(x.dropped_overlay_rows for x in results),
    }


# --------------------------------------------------------------------------- #
# Nodes 2 to 4 measurement
# --------------------------------------------------------------------------- #
ROUTINGS = ("AUTO_CLEAR", "HUMAN_REVIEW")
_CODE_RE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_NODE_RE = re.compile(r"Node\d+")


@dataclass
class ScreenResult:
    doc_id: str
    layout: str
    split: str
    is_tampered: bool
    genuine_resaved: bool
    tamper_type: str
    tamper_method: str
    hard_cases: list
    routing: str
    auto_cleared: bool
    finding_codes: list
    material: bool
    nodes_fired: list
    expected_codes: list
    expected_code_seen: Optional[bool]
    expected_nodes: list
    expected_node_fired: Optional[bool]
    error: Optional[str] = None


def _default_screen(path: str):
    """Real Nodes 1 to 4 on one PDF, in memory: no case store, no ML models, no writes."""
    from fraud_detect.pipeline import run_pipeline
    return run_pipeline(str(path))


def _enum_text(v) -> str:
    return str(getattr(v, "value", v))


def _node_of(agent) -> Optional[str]:
    m = _NODE_RE.match(str(agent or ""))
    return m.group(0) if m else None


def _expected(entries: Sequence[str]) -> tuple[list, list]:
    codes, nodes = set(), set()
    for entry in entries or []:
        head, sep, tail = str(entry).partition(":")
        node = _node_of(head.strip())
        if node:
            nodes.add(node)
        if sep and _CODE_RE.fullmatch(tail.strip()):
            codes.add(tail.strip())
    return sorted(codes), sorted(nodes)


def _screen_one(record: DocRecord, screen: Callable) -> ScreenResult:
    label = record.manifest.get("label", {})
    exp_codes, exp_nodes = _expected(label.get("should_be_seen_by") or [])
    base = dict(
        doc_id=record.doc_id, layout=record.layout, split=record.split,
        is_tampered=record.is_tampered, genuine_resaved=record.genuine_resaved,
        tamper_type=record.tamper_type, tamper_method=record.tamper_method,
        hard_cases=list(record.manifest.get("genuine_hard_cases") or []),
        expected_codes=exp_codes, expected_nodes=exp_nodes)
    try:
        report = screen(_RecordPath(record))
        routing = _enum_text(report.routing)
        findings = list(report.findings)
    except Exception as exc:  # the run continues; this document is left out of every share
        return ScreenResult(**base, routing="", auto_cleared=False, finding_codes=[],
                            material=False, nodes_fired=[], expected_code_seen=None,
                            expected_node_fired=None, error=type(exc).__name__)
    codes = sorted({f.code for f in findings})
    strong = [f for f in findings if _enum_text(f.severity) in ("MEDIUM", "HIGH")]
    fired = sorted({n for n in (_node_of(f.agent) for f in strong) if n})
    return ScreenResult(
        **base, routing=routing, auto_cleared=routing == "AUTO_CLEAR", finding_codes=codes,
        material=bool(strong), nodes_fired=fired,
        expected_code_seen=(any(c in codes for c in exp_codes) if exp_codes else None),
        expected_node_fired=(any(n in fired for n in exp_nodes) if exp_nodes else None))


def measure_nodes_2_to_4(records: Sequence[DocRecord], screen: Optional[Callable] = None
                         ) -> list[ScreenResult]:
    """One ``ScreenResult`` per record, in record order. Does not enter the company scope."""
    screen = screen if screen is not None else _default_screen
    return [_screen_one(r, screen) for r in records]


def _ok(results: Sequence[ScreenResult]) -> list[ScreenResult]:
    return [x for x in results if not x.error]


def _frac(items: Sequence, pred: Callable) -> Optional[float]:
    return _share(sum(1 for x in items if pred(x)), len(items))


def summarise_genuine(results: Sequence[ScreenResult]) -> dict:
    ok = _ok(results)
    hard = [x for x in results if x.hard_cases]
    return {
        "documents": len(results),
        "auto_clear_share": _frac(ok, lambda x: x.auto_cleared),
        "code_counts": dict(sorted(Counter(c for x in ok for c in x.finding_codes).items())),
        "screening_errors": len(results) - len(ok),
        "hard_cases": {"documents": len(hard),
                       "auto_clear_share": _frac(_ok(hard), lambda x: x.auto_cleared)},
    }


def summarise_screening(results: Sequence[ScreenResult]) -> dict:
    all_genuine = [x for x in results if not x.is_tampered and not x.genuine_resaved]
    tampered = [x for x in results if x.is_tampered]
    tampered_ok = _ok(tampered)
    counts = Counter(x.routing for x in results)
    return {
        "documents": len(results),
        "genuine_documents": len(all_genuine),
        "genuine_auto_clear_share": _frac(_ok(all_genuine), lambda x: x.auto_cleared),
        "tampered_documents": len(tampered),
        "material_share": _frac(tampered_ok, lambda x: x.material),
        "not_auto_cleared_share": _frac(tampered_ok, lambda x: not x.auto_cleared),
        "routing_counts": {r: counts.get(r, 0) for r in ROUTINGS},
        "screening_errors": sum(1 for x in results if x.error),
    }


def group_screening(results: Sequence[ScreenResult], by: str) -> dict:
    groups: dict[str, list[ScreenResult]] = defaultdict(list)
    for x in results:
        groups[getattr(x, by)].append(x)
    return {k: summarise_screening(v) for k, v in sorted(groups.items())}


def detection_by_tamper_type(results: Sequence[ScreenResult]) -> list[dict]:
    groups: dict[tuple, list[ScreenResult]] = defaultdict(list)
    for x in results:
        if x.is_tampered:
            groups[(x.tamper_type, x.tamper_method)].append(x)
    table = []
    for (ttype, method), group in sorted(groups.items()):
        ok = _ok(group)
        with_codes = [x for x in ok if x.expected_code_seen is not None]
        with_nodes = [x for x in ok if x.expected_node_fired is not None]
        table.append({
            "tamper_type": ttype, "method": method, "documents": len(group),
            "material_share": _frac(ok, lambda x: x.material),
            "not_auto_cleared_share": _frac(ok, lambda x: not x.auto_cleared),
            "codes": sorted({c for x in ok for c in x.finding_codes}),
            "expected_codes": sorted({c for x in group for c in x.expected_codes}),
            "expected_code_share": _frac(with_codes, lambda x: x.expected_code_seen),
            "expected_nodes": sorted({n for x in group for n in x.expected_nodes}),
            "expected_node_share": _frac(with_nodes, lambda x: x.expected_node_fired),
            "nodes_fired": dict(sorted(Counter(n for x in ok for n in x.nodes_fired).items())),
            "screening_errors": len(group) - len(ok),
        })
    return table


def blind_spots(table: Sequence[dict], genuine_not_auto_cleared_share: Optional[float]
                ) -> list[dict]:
    g = genuine_not_auto_cleared_share if genuine_not_auto_cleared_share is not None else 0.0
    listed = []
    for row in table:
        material, not_auto = row.get("material_share"), row.get("not_auto_cleared_share")
        if (material is not None and material == 0) or (
                not_auto is not None and not_auto <= g):
            listed.append(row)
    return listed


def follow_up_suggestions(blind: Sequence[dict]) -> list[str]:
    """Text only. Nothing is implemented from these."""
    return [
        f"{r['tamper_type']} ({r['method']}): check which Node 2/3 rule should flag it "
        f"(expected: {', '.join(r.get('expected_codes') or []) or 'see label'}) and whether "
        f"the corpus variant is realistic; consider a new or tuned rule."
        for r in blind]


def _scope_for(root: Path):
    """The company scope for the corpus; without companies.json only the live lookup is off."""
    companies = root / "companies.json"
    if companies.is_file():
        return registered_companies(companies)
    return _live_off()


@contextlib.contextmanager
def _live_off():
    from fraud_detect import registry
    previous = registry._live_default
    registry.set_live_default(False)
    try:
        yield
    finally:
        registry.set_live_default(previous)


def _screening_report(screened: Sequence[ScreenResult]) -> dict:
    genuine = [x for x in screened if not x.is_tampered and not x.genuine_resaved]
    resaved = [x for x in screened if x.genuine_resaved]
    table = detection_by_tamper_type(screened)
    share = summarise_screening(genuine)["genuine_auto_clear_share"]
    blind = blind_spots(table, None if share is None else 1 - share)
    return {
        "genuine": summarise_genuine(genuine),
        "genuine_resaved": summarise_genuine(resaved),
        "tamper_types": table,
        "blind_spots": blind,
        "follow_ups": follow_up_suggestions(blind),
        "layouts": group_screening(screened, "layout"),
        "splits": group_screening(screened, "split"),
        "decisions": summarise_screening(screened)["routing_counts"],
    }


def _codes_text(counts: dict) -> str:
    return ",".join(f"{c}:{n}" for c, n in counts.items()) or "-"


def _print_screening(sc: dict, screened: Sequence[ScreenResult], out: TextIO) -> None:
    for name in ("genuine", "genuine_resaved"):
        g = sc[name]
        print(f"{name}: documents={g['documents']} auto_clear={_pct(g['auto_clear_share'])} "
              f"codes={_codes_text(g['code_counts'])}", file=out)
    print("decisions: " + " ".join(f"{k}={v}" for k, v in sc["decisions"].items()), file=out)
    for t in sc["tamper_types"]:
        print(f"tamper {t['tamper_type']} {t['method']}: documents={t['documents']} "
              f"material={_pct(t['material_share'])} "
              f"not_auto_cleared={_pct(t['not_auto_cleared_share'])} "
              f"codes={','.join(t['codes']) or '-'} "
              f"expected_code={_pct(t['expected_code_share'])} "
              f"expected_node={_pct(t['expected_node_share'])}", file=out)
    for b in sc["blind_spots"]:
        print(f"blind spot: {b['tamper_type']} {b['method']}: "
              f"material={_pct(b['material_share'])} "
              f"not_auto_cleared={_pct(b['not_auto_cleared_share'])}", file=out)
    for key in ("layouts", "splits"):
        kind = key[:-1]
        for k, v in sc[key].items():
            print(f"screening {kind} {k}: documents={v['documents']} "
                  f"genuine_auto_clear={_pct(v['genuine_auto_clear_share'])} "
                  f"material={_pct(v['material_share'])} "
                  f"not_auto_cleared={_pct(v['not_auto_cleared_share'])}", file=out)
    bad = [x for x in screened if x.error]
    if bad:
        print(f"screening errors: {len(bad)} ("
              + ", ".join(f"{x.doc_id}: {x.error}" for x in bad) + ")", file=out)
    for line in sc["follow_ups"]:
        print(f"follow-up: {line}", file=out)


def _screening_markdown(sc: dict) -> list[str]:
    g, r = sc["genuine"], sc["genuine_resaved"]
    lines = ["## Nodes 2 to 4", "",
             f"Genuine: {g['documents']} documents, auto-clear {_pct(g['auto_clear_share'])}, "
             f"codes {_codes_text(g['code_counts'])}", "",
             f"Genuine re-saved: {r['documents']} documents, auto-clear "
             f"{_pct(r['auto_clear_share'])}, codes {_codes_text(r['code_counts'])}", "",
             "| Tamper type | Method | Documents | Material | Not auto-cleared | Codes "
             "| Expected code | Expected node |",
             "|---|---|---|---|---|---|---|---|"]
    for t in sc["tamper_types"]:
        lines.append(
            f"| {t['tamper_type']} | {t['method']} | {t['documents']} "
            f"| {_pct(t['material_share'])} | {_pct(t['not_auto_cleared_share'])} "
            f"| {', '.join(t['codes']) or '-'} | {_pct(t['expected_code_share'])} "
            f"| {_pct(t['expected_node_share'])} |")
    lines += ["", "## Blind spots", ""]
    if sc["blind_spots"]:
        lines += [f"- {b['tamper_type']} {b['method']}: material {_pct(b['material_share'])}, "
                  f"not auto-cleared {_pct(b['not_auto_cleared_share'])}"
                  for b in sc["blind_spots"]]
    else:
        lines.append("None.")
    lines += ["", "## Follow-up suggestions", ""]
    lines += [f"- {s}" for s in sc["follow_ups"]] or ["None."]
    lines += ["", "## Decisions", "",
              ", ".join(f"{k}={v}" for k, v in sc["decisions"].items()), ""]
    for name, key in (("Layout", "layouts"), ("Split", "splits")):
        lines += [f"## Screening per {name.lower()}", "",
                  f"| {name} | Documents | Genuine auto-clear | Material | Not auto-cleared |",
                  "|---|---|---|---|---|"]
        for k, v in sc[key].items():
            lines.append(f"| {k} | {v['documents']} | {_pct(v['genuine_auto_clear_share'])} "
                         f"| {_pct(v['material_share'])} | {_pct(v['not_auto_cleared_share'])} |")
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _layout_line(layout: str, s: dict) -> str:
    return (f"layout {layout}: documents={s['documents']} profile={s['profile'] or '-'} "
            f"rows={_pct(s['rows_share'])} genuine_rows={_pct(s['rows_share_genuine'])} "
            f"tampered_rows={_pct(s['rows_share_tampered'])} balance={_pct(s['balance_share'])} "
            f"header={_pct(s['header_share'])} "
            f"findings={','.join(s['finding_codes']) or '-'}")


def _group_by_layout(results: Sequence[Node1Result]) -> dict[str, list[Node1Result]]:
    groups: dict[str, list[Node1Result]] = defaultdict(list)
    for x in results:
        groups[x.layout].append(x)
    return dict(sorted(groups.items()))


def _markdown(summaries: dict[str, dict], dropped: int, screening: Optional[dict] = None) -> str:
    lines = ["# Corpus verification report", "", "## Node 1", "",
             "| Layout | Documents | Profile | Rows read | Genuine rows | Tampered rows "
             "| Balance | Header | Finding codes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for layout, s in summaries.items():
        lines.append(
            f"| {layout} | {s['documents']} | {s['profile'] or '-'} | {_pct(s['rows_share'])} "
            f"| {_pct(s['rows_share_genuine'])} | {_pct(s['rows_share_tampered'])} "
            f"| {_pct(s['balance_share'])} | {_pct(s['header_share'])} "
            f"| {', '.join(s['finding_codes']) or '-'} |")
    lines += ["", f"Overlay rows dropped by Node 1: {dropped}", ""]
    if screening is not None:
        lines += _screening_markdown(screening)
    return "\n".join(lines)


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify", description=__doc__.splitlines()[0])
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="corpus folder")
    p.add_argument("--sample", type=_positive_int, default=None,
                   help="measure at most N documents per layout")
    p.add_argument("--json", default=None, help="write the report as JSON to this file")
    p.add_argument("--markdown", default=None, help="write the report as Markdown to this file")
    return p


def main(argv=None, *, corpus_dir=None, out: Optional[TextIO] = None,
         ingest: Optional[Callable] = None, screen: Optional[Callable] = None) -> int:
    """Run the report. ``screen`` replaces the real Nodes 1 to 4 run (tests inject it)."""
    out = out if out is not None else sys.stdout
    args = _build_parser().parse_args(argv)
    root = Path(corpus_dir) if corpus_dir is not None else Path(args.corpus)
    started = time.perf_counter()

    if not (root.is_dir() and (root / "labels.csv").is_file()):
        print(f"skipped: corpus folder not found or has no labels.csv: {root}", file=out)
        return 0

    records = load_corpus(root)
    if args.sample is not None:
        records = sample_records(records, args.sample)
    results = measure_node1(records, ingest=ingest)

    summaries = {layout: summarise_layout(g) for layout, g in _group_by_layout(results).items()}
    for layout, s in summaries.items():
        print(_layout_line(layout, s), file=out)
    dropped = sum(s["dropped_overlay_rows"] for s in summaries.values())
    print(f"overlay rows dropped by Node 1: {dropped}", file=out)
    errors = [x.doc_id for x in results if x.error]
    if errors:
        print(f"ingest errors: {len(errors)} ({', '.join(errors)})", file=out)

    with _scope_for(root):
        screened = measure_nodes_2_to_4(records, screen=screen)
    screening = _screening_report(screened)
    _print_screening(screening, screened, out)

    if args.json:
        payload = {"corpus": str(root), "node1": {
            "layouts": summaries, "overlay_rows_dropped": dropped,
            "documents": [asdict(x) for x in results]},
            "screening": {**screening, "documents": [asdict(x) for x in screened]}}
        _write_text(args.json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        _write_text(args.markdown, _markdown(summaries, dropped, screening))

    print(f"elapsed {time.perf_counter() - started:.1f}s", file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
