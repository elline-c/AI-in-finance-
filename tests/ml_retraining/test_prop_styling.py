"""Task 6.4: property tests for the styling-balance and capability gates.

Property 16: Styling balance.
Property 17: Capability gate.
"""
from __future__ import annotations

import hashlib

from hypothesis import given, settings, strategies as st

from fraud_detect.ml.retraining import corpus_gates as cg

SETTINGS = settings(max_examples=100, deadline=None)

BANKS = ("A", "B", "C")
CREATORS = ("x", "y")


# ------------------------------------------------------------------ Property 16
@st.composite
def balance_tables(draw):
    n = draw(st.integers(0, 60))
    rows = []
    for i in range(n):
        rows.append({
            "doc_id": f"D{i}",
            "is_tampered": draw(st.booleans()),
            "style_bank_name": draw(st.sampled_from(BANKS)),
            "style_creator": draw(st.one_of(st.none(), st.sampled_from(CREATORS))),
        })
    return rows


def _expected_failures(rows, min_docs, points):
    n_gen = sum(1 for r in rows if not r["is_tampered"])
    n_tam = len(rows) - n_gen
    expected = {}
    for col in ("style_bank_name", "style_creator"):
        values = {r[col] for r in rows if r[col] is not None}
        for v in values:
            g = sum(1 for r in rows if r[col] == v and not r["is_tampered"])
            t = sum(1 for r in rows if r[col] == v and r["is_tampered"])
            if g + t < min_docs:
                continue
            gs = 100.0 * g / n_gen if n_gen else 0.0
            ts = 100.0 * t / n_tam if n_tam else 0.0
            # exact integer comparison of |g/n_gen - t/n_tam| * 100 > points
            if n_gen and n_tam:
                differs = abs(g * n_tam - t * n_gen) * 100 > points * n_gen * n_tam
            else:
                differs = abs(gs - ts) > points
            if differs:
                expected[(col, v)] = (gs, ts)
    return expected


@SETTINGS
@given(rows=balance_tables(), min_docs=st.integers(1, 30), points=st.integers(0, 60))
def test_styling_balance_lists_failing_values_iff_rule(rows, min_docs, points):
    """**Validates: Requirements 5.2, 5.3**"""
    failures = cg.styling_balance(rows, min_docs=min_docs, share_points=points)
    expected = _expected_failures(rows, min_docs, points)
    assert {(f.column, f.value) for f in failures} == set(expected)
    assert len(failures) == len(expected)
    for f in failures:
        gs, ts = expected[(f.column, f.value)]
        assert abs(f.genuine_share - gs) < 1e-6
        assert abs(f.tampered_share - ts) < 1e-6


@SETTINGS
@given(rows=balance_tables(), min_docs=st.integers(1, 30), points=st.integers(0, 60))
def test_styling_balance_gate_matches_failures_and_leaves_files_unchanged(
        rows, min_docs, points, tmp_path_factory):
    """**Validates: Requirements 5.2, 5.3**"""
    folder = tmp_path_factory.mktemp("corpus")
    corpus_file = folder / "labels.csv"
    corpus_file.write_bytes(b"doc_id,split\nD0,train\n")
    before = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    failures = cg.styling_balance(rows, min_docs=min_docs, share_points=points)
    result = cg.styling_balance_gate(rows, min_docs=min_docs, share_points=points)
    assert result.passed == (not failures)
    assert len(result.offenders) == len(failures)
    for f, line in zip(failures, result.offenders):
        assert f.column in line and str(f.value) in line
        assert f"{f.genuine_share:.1f}%" in line and f"{f.tampered_share:.1f}%" in line
    assert hashlib.sha256(corpus_file.read_bytes()).hexdigest() == before


# ------------------------------------------------------------------ Property 17
FAMILIES = ("base", "export", "editor")


@st.composite
def capability_tables(draw):
    n = draw(st.integers(0, 30))
    rows = []
    for i in range(n):
        row = {"doc_id": f"D{i}", "is_tampered": draw(st.booleans())}
        for _, col in cg.CAPABILITY_CHOICES:
            if col == "style_fonts":
                row[col] = draw(st.one_of(st.none(), st.sampled_from([("F1",), ("F1", "F2")])))
            else:
                row[col] = draw(st.one_of(st.none(), st.sampled_from(("v1", "v2", "v3"))))
        row[cg.FAMILY_COLUMN] = draw(st.one_of(st.none(), st.sampled_from(FAMILIES)))
        rows.append(row)
    return rows


def _distinct(rows, col, tampered):
    return {r[col] for r in rows if r[col] is not None and r["is_tampered"] == tampered}


@SETTINGS
@given(rows=capability_tables())
def test_capability_report_names_exactly_the_missing_choices(rows):
    """**Validates: Requirements 5.4, 5.6**"""
    report = cg.capability_report(rows)

    expected_choices = set()
    for name, col in cg.CAPABILITY_CHOICES:
        g, t = _distinct(rows, col, False), _distinct(rows, col, True)
        if not g and not t:
            continue  # nothing known: not assessable
        if len(g) < 2 or len(t) < 2:
            expected_choices.add(name)
    reported_choices = {line.split(":")[0] for line in report if not line.startswith("family ")}
    assert reported_choices == expected_choices

    fam_g = _distinct(rows, cg.FAMILY_COLUMN, False)
    fam_t = _distinct(rows, cg.FAMILY_COLUMN, True)
    expected_fam = fam_g ^ fam_t
    reported_fam = {line.split(":")[0].split(" ", 1)[1] for line in report if line.startswith("family ")}
    assert reported_fam == expected_fam

    # No duplicates, and nothing else is reported.
    assert len(report) == len(set(report))


@SETTINGS
@given(rows=capability_tables())
def test_capability_stop_message_and_gate_list_exactly_the_report(rows):
    """**Validates: Requirements 5.4, 5.6**"""
    report = cg.capability_report(rows)
    message = cg.capability_stop_message(report)
    bullets = [ln.strip()[2:] for ln in message.splitlines() if ln.strip().startswith("- ")]
    assert bullets == list(report)
    result = cg.capability_gate(rows)
    assert result.passed == (not report)
    assert list(result.offenders) == list(report)
    assert result.waivable is True


@SETTINGS
@given(n=st.integers(2, 6), m=st.integers(2, 6))
def test_capability_never_reports_fully_varied_table(n, m):
    """**Validates: Requirements 5.4, 5.6**"""
    rows = []
    for tampered, count in ((False, n), (True, m)):
        for i in range(count):
            row = {"is_tampered": tampered, cg.FAMILY_COLUMN: FAMILIES[i % 2]}
            for _, col in cg.CAPABILITY_CHOICES:
                row[col] = f"v{i % 2}"
            rows.append(row)
    assert cg.capability_report(rows) == []
    assert cg.capability_gate(rows).passed is True
