"""Property 20: the report's arithmetic is right (pure functions of verify.py).

**Validates: Requirements 14.2, 14.9, 15.2, 15.3, 15.5, 15.8**
"""
from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings, strategies as st

from fraud_detect.corpus.verify import (
    DocRecord, Node1Result, ScreenResult, blind_spots, detection_by_tamper_type,
    group_screening, sample_records, summarise_genuine, summarise_layout,
    summarise_screening)

SET = settings(max_examples=100, deadline=None)

LAYOUTS = ["A", "B", "C"]
SPLITS = ["train", "val", "test"]
TTYPES = ["", "genuine_resaved", "content", "behavioural"]
METHODS = ["", "overlay", "edit"]
CODES = ["C_ONE", "C_TWO", "C_THREE"]
NODES = ["Node2", "Node3", "Node4"]


@st.composite
def node1_results(draw):
    expected = draw(st.integers(0, 40))
    return Node1Result(
        doc_id=draw(st.text("abcdef", min_size=1, max_size=4)),
        layout="L", is_tampered=draw(st.booleans()), genuine_resaved=draw(st.booleans()),
        rows_expected=expected, rows_read=draw(st.integers(0, expected)),
        profile=draw(st.sampled_from(["", "p1", "p2"])),
        balance_ok=draw(st.booleans()), header_ok=draw(st.booleans()),
        finding_codes=sorted(draw(st.sets(st.sampled_from(CODES)))),
        dropped_overlay_rows=draw(st.integers(0, 5)))


@st.composite
def screen_results(draw):
    routing = draw(st.sampled_from(["AUTO_CLEAR", "HUMAN_REVIEW"]))
    error = draw(st.sampled_from([None, None, None, "Boom"]))
    codes = sorted(draw(st.sets(st.sampled_from(CODES))))
    tampered = draw(st.booleans())
    return ScreenResult(
        doc_id=draw(st.text("abcdef", min_size=1, max_size=4)),
        layout=draw(st.sampled_from(LAYOUTS)), split=draw(st.sampled_from(SPLITS)),
        is_tampered=tampered,
        genuine_resaved=False if tampered else draw(st.booleans()),
        tamper_type=draw(st.sampled_from(TTYPES)),
        tamper_method=draw(st.sampled_from(METHODS)),
        hard_cases=draw(st.sampled_from([[], [], ["h"]])),
        routing="" if error else routing,
        auto_cleared=False if error else routing == "AUTO_CLEAR",
        finding_codes=[] if error else codes,
        material=False if error else draw(st.booleans()),
        nodes_fired=[] if error else sorted(draw(st.sets(st.sampled_from(NODES)))),
        expected_codes=sorted(draw(st.sets(st.sampled_from(CODES)))),
        expected_code_seen=draw(st.sampled_from([None, True, False])),
        expected_nodes=sorted(draw(st.sets(st.sampled_from(NODES)))),
        expected_node_fired=draw(st.sampled_from([None, True, False])),
        error=error)


def _ratio(num, den):
    return num / den if den else None


@SET
@given(st.lists(node1_results(), max_size=15))
def test_summarise_layout_matches_direct_computation(results):
    s = summarise_layout(results)
    gen = [x for x in results if not x.is_tampered]
    tam = [x for x in results if x.is_tampered]

    def rs(g):
        return _ratio(sum(x.rows_read for x in g), sum(x.rows_expected for x in g))

    assert s["documents"] == len(results)
    assert s["rows_share_genuine"] == rs(gen)
    assert s["rows_share_tampered"] == rs(tam)
    assert s["rows_share"] == (rs(results) if rs(results) is not None else 0.0)
    n = len(results)
    assert s["balance_share"] == (sum(x.balance_ok for x in results) / n if n else 0.0)
    assert s["header_share"] == (sum(x.header_ok for x in results) / n if n else 0.0)
    assert s["dropped_overlay_rows"] == sum(x.dropped_overlay_rows for x in results)
    assert s["finding_codes"] == sorted({c for x in results for c in x.finding_codes})
    for key in ("rows_share", "rows_share_genuine", "rows_share_tampered",
                "balance_share", "header_share"):
        assert s[key] is None or 0.0 <= s[key] <= 1.0


@SET
@given(st.lists(screen_results(), max_size=20))
def test_summarise_genuine_matches_direct_computation(results):
    s = summarise_genuine(results)
    ok = [x for x in results if not x.error]
    assert s["documents"] == len(results)
    assert s["screening_errors"] == len(results) - len(ok)
    assert s["auto_clear_share"] == _ratio(sum(x.auto_cleared for x in ok), len(ok))
    assert sum(s["code_counts"].values()) == sum(len(x.finding_codes) for x in ok)
    for c, n in s["code_counts"].items():
        assert n == sum(c in x.finding_codes for x in ok)
    hard = [x for x in results if x.hard_cases]
    hard_ok = [x for x in hard if not x.error]
    assert s["hard_cases"]["documents"] == len(hard)
    assert s["hard_cases"]["auto_clear_share"] == _ratio(
        sum(x.auto_cleared for x in hard_ok), len(hard_ok))


@SET
@given(st.lists(screen_results(), max_size=20))
def test_summarise_screening_matches_direct_computation(results):
    s = summarise_screening(results)
    genuine = [x for x in results if not x.is_tampered and not x.genuine_resaved]
    tampered = [x for x in results if x.is_tampered]
    t_ok = [x for x in tampered if not x.error]
    g_ok = [x for x in genuine if not x.error]
    assert s["documents"] == len(results)
    assert s["genuine_documents"] == len(genuine)
    assert s["tampered_documents"] == len(tampered)
    assert s["genuine_auto_clear_share"] == _ratio(sum(x.auto_cleared for x in g_ok), len(g_ok))
    assert s["material_share"] == _ratio(sum(x.material for x in t_ok), len(t_ok))
    assert s["not_auto_cleared_share"] == _ratio(
        sum(not x.auto_cleared for x in t_ok), len(t_ok))
    assert s["screening_errors"] == sum(1 for x in results if x.error)
    assert s["routing_counts"]["AUTO_CLEAR"] == sum(x.routing == "AUTO_CLEAR" for x in results)
    assert s["routing_counts"]["HUMAN_REVIEW"] == sum(
        x.routing == "HUMAN_REVIEW" for x in results)


@SET
@given(st.lists(screen_results(), max_size=25), st.sampled_from(["layout", "split"]))
def test_group_screening_partitions_and_matches_per_group(results, by):
    grouped = group_screening(results, by)
    assert sum(g["documents"] for g in grouped.values()) == len(results)
    assert list(grouped) == sorted({getattr(x, by) for x in results})
    for key, g in grouped.items():
        assert g == summarise_screening([x for x in results if getattr(x, by) == key])


@SET
@given(st.lists(screen_results(), max_size=25))
def test_detection_by_tamper_type_matches_direct_computation(results):
    table = detection_by_tamper_type(results)
    tampered = [x for x in results if x.is_tampered]
    assert sum(r["documents"] for r in table) == len(tampered)
    keys = [(r["tamper_type"], r["method"]) for r in table]
    assert keys == sorted(set(keys))
    for r in table:
        group = [x for x in tampered
                 if (x.tamper_type, x.tamper_method) == (r["tamper_type"], r["method"])]
        ok = [x for x in group if not x.error]
        assert r["documents"] == len(group)
        assert r["screening_errors"] == len(group) - len(ok)
        assert r["material_share"] == _ratio(sum(x.material for x in ok), len(ok))
        assert r["not_auto_cleared_share"] == _ratio(
            sum(not x.auto_cleared for x in ok), len(ok))
        wc = [x for x in ok if x.expected_code_seen is not None]
        assert r["expected_code_share"] == _ratio(sum(x.expected_code_seen for x in wc), len(wc))
        wn = [x for x in ok if x.expected_node_fired is not None]
        assert r["expected_node_share"] == _ratio(sum(x.expected_node_fired for x in wn), len(wn))


share = st.one_of(st.none(), st.sampled_from([0.0, 0.25, 0.5, 1.0]))


@SET
@given(st.lists(st.fixed_dictionaries({
    "tamper_type": st.text("xyz", max_size=3), "method": st.text("mn", max_size=2),
    "material_share": share, "not_auto_cleared_share": share}), max_size=12),
    share)
def test_blind_spots_iff_rule(table, genuine_share):
    g = genuine_share if genuine_share is not None else 0.0
    listed = blind_spots(table, genuine_share)
    for row in table:
        m, n = row["material_share"], row["not_auto_cleared_share"]
        expected = (m is not None and m == 0) or (n is not None and n <= g)
        assert any(row is r for r in listed) == expected
    assert len(listed) == sum(1 for r in table if any(r is x for x in listed))


def _rec(layout, doc_id):
    return DocRecord(doc_id=doc_id, layout=layout, path=Path(doc_id), is_tampered=False,
                     genuine_resaved=False, tamper_method="")


@SET
@given(st.lists(st.tuples(st.sampled_from(LAYOUTS), st.integers(0, 999)),
                max_size=40, unique=True),
       st.integers(1, 6))
def test_sample_records_caps_per_layout(pairs, n):
    records = [_rec(layout, f"d{i:04d}") for layout, i in pairs]
    sampled = sample_records(records, n)
    for layout in LAYOUTS:
        have = [r for r in records if r.layout == layout]
        got = [r for r in sampled if r.layout == layout]
        assert len(got) == min(n, len(have))
        assert [r.doc_id for r in got] == sorted(r.doc_id for r in have)[:n]
