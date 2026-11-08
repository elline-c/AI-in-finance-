"""The Node 3 evaluation script: its statistics, its ledger conversion and a tiny run."""
from __future__ import annotations

import io
import json

import pytest

from fraud_detect import reference_data
from fraud_detect.nodes import node3_eval as ev
from fraud_detect.nodes.logic_config import DEFAULT_LOGIC_CONFIG


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 6, 0.0, 0.390),            # no failures in 6 still allows up to ~39%
    (6, 6, 0.610, 1.0),
    (5, 10, 0.237, 0.763),
    (125, 200, 0.556, 0.689),
])
def test_wilson_interval_matches_known_values(k, n, lo, hi):
    a, b = ev.wilson_interval(k, n)
    assert a == pytest.approx(lo, abs=0.002) and b == pytest.approx(hi, abs=0.002)


def test_wilson_interval_stays_inside_zero_and_one_and_handles_no_trials():
    for k, n in ((0, 1), (1, 1), (0, 1000), (1000, 1000)):
        lo, hi = ev.wilson_interval(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    assert ev.wilson_interval(0, 0) == (0.0, 1.0)


def test_a_bigger_sample_gives_a_tighter_interval():
    small = ev.wilson_interval(5, 10)
    big = ev.wilson_interval(500, 1000)
    assert (big[1] - big[0]) < (small[1] - small[0])


def test_rate_is_formatted_with_counts_and_interval():
    assert ev.fmt_rate(1, 4).startswith("1/4 (25.0%) [")


# --------------------------------------------------------------------------- #
# Ledger conversion and scenarios
# --------------------------------------------------------------------------- #
def test_ledger_becomes_an_ingestion_result_that_replays_balances():
    ledger = ev.make_ledger(2000, 6)
    ing = ev.ledger_to_ingestion(ledger)
    assert len(ing.transactions) == len(ledger.rows)
    assert ing.uen == ledger.company.uen and ing.account_holder == ledger.company.name
    assert ing.statement_period_start == ledger.months[0].replace(day=1)
    bal = ledger.opening_balance
    for row, t in zip(ledger.rows, ing.transactions):
        bal = round(bal + row.amount, 2)
        assert t.balance == bal
    assert len(ing.segments) == 6                       # one section per month


def test_same_seed_gives_the_same_ledger():
    a, b = ev.make_ledger(2007, 6), ev.make_ledger(2007, 6)
    assert a.rows == b.rows


def test_short_silence_removes_a_twelve_day_stretch_and_only_that():
    ledger = ev.make_ledger(2001, 9)
    cut = ev.tampered_ledger("short_silence", ledger, 2001)
    removed = [r for r in ledger.rows if r not in cut.rows]
    assert removed and set(cut.rows) <= set(ledger.rows)
    assert (max(r.date for r in removed) - min(r.date for r in removed)).days < ev.SILENCE_DAYS
    # The stretch is contiguous: nothing remains between its first and last removed day.
    lo, hi = min(r.date for r in removed), max(r.date for r in removed)
    assert not [r for r in cut.rows if lo <= r.date <= hi]


def test_catalogue_tamper_inserts_rows():
    ledger = ev.make_ledger(2002, 9)
    assert len(ev.tampered_ledger("sudden_large_deposit", ledger, 2002).rows) == len(ledger.rows) + 1


def test_every_default_scenario_has_an_expected_rule():
    assert set(ev.DEFAULT_TAMPERS) == set(ev.EXPECTED)
    assert all(codes for codes in ev.EXPECTED.values())


# --------------------------------------------------------------------------- #
# A tiny run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny():
    before = dict(reference_data.CASHFLOW_BASELINE)
    data = ev.run(4, 9, 2000, ("sudden_large_deposit", "rapid_pass_through"), ev.PRESETS)
    assert reference_data.CASHFLOW_BASELINE == before        # the declared baselines are removed again
    return data


def test_counts_cover_every_document(tiny):
    for preset in ev.PRESETS:
        for scenario in ("genuine", "sudden_large_deposit", "rapid_pass_through"):
            assert tiny["results"][preset][scenario]["n"] == 4


def test_a_code_never_fires_on_more_documents_than_there_are(tiny):
    for preset in tiny["results"].values():
        for cell in preset.values():
            assert all(0 <= k <= cell["n"] for k in cell["codes"].values())


def test_learned_catches_what_the_fixed_settings_cannot(tiny):
    learned = tiny["results"]["learned"]["rapid_pass_through"]
    fixed = tiny["results"]["fixed"]["rapid_pass_through"]
    assert learned["codes"].get("FL_RAPID_PASS_THROUGH", 0) == 4
    assert fixed["codes"].get("FL_RAPID_PASS_THROUGH", 0) == 0          # switched off in 'fixed'


def test_the_run_is_repeatable():
    a = ev.run(2, 9, 2000, ("structuring",), {"learned": DEFAULT_LOGIC_CONFIG})
    b = ev.run(2, 9, 2000, ("structuring",), {"learned": DEFAULT_LOGIC_CONFIG})
    assert a == b


def test_the_declared_baseline_can_be_left_out():
    with_decl = ev.run(3, 9, 2000, ("sudden_large_deposit",), {"fixed": ev.PRESETS["fixed"]})
    without = ev.run(3, 9, 2000, ("sudden_large_deposit",), {"fixed": ev.PRESETS["fixed"]},
                     declared=False)
    got = lambda d: d["results"]["fixed"]["sudden_large_deposit"]["codes"].get("FL_SUDDEN_DEPOSIT", 0)
    assert got(without) == 0                  # 'fixed' has nothing to judge against without it
    assert got(with_decl) >= got(without)


def test_report_has_the_sections_a_reader_needs(tiny):
    text = ev.format_report(tiny)
    for needle in ("SYNTHETIC DATA", "== preset: learned ==", "== preset: fixed ==",
                   "Genuine statements (n=4)", "Planted tampers", "side by side",
                   "FL_RAPID_PASS_THROUGH", "95% Wilson", "[0.0%-"):
        assert needle in text, needle


def test_cli_writes_the_report_and_the_json(tmp_path):
    out_txt, out_json, buf = tmp_path / "r.txt", tmp_path / "r.json", io.StringIO()
    rc = ev.main(["--companies", "2", "--tampers", "structuring", "--presets", "learned",
                  "--out", str(out_txt), "--json", str(out_json)], out=buf)
    assert rc == 0
    assert out_txt.read_text() == buf.getvalue()
    assert json.loads(out_json.read_text())["settings"]["companies"] == 2


@pytest.mark.parametrize("args", [["--presets", "nope"], ["--tampers", "nope"],
                                  ["--companies", "0"], ["--months", "1"]])
def test_cli_rejects_bad_arguments(args):
    assert ev.main(args, out=io.StringIO()) == 2


# --------------------------------------------------------------------------- #
# Overrides and silence length
# --------------------------------------------------------------------------- #
def test_numeric_settings_can_be_overridden_for_every_preset():
    out = ev.apply_overrides(ev.PRESETS, ["gap_min_days=14", "spike_month_share=0.25"])
    assert all(c.gap_min_days == 14 and c.spike_month_share == 0.25 for c in out.values())
    assert ev.PRESETS["learned"].gap_min_days == 10               # the originals are untouched


@pytest.mark.parametrize("bad", ["nope=1", "gap_min_days", "gap_min_days=abc", "pass_exclude_words=x",
                                 "gap_min_days=99"])
def test_bad_overrides_are_rejected(bad):
    with pytest.raises(ValueError):
        ev.apply_overrides(ev.PRESETS, [bad])


def test_cli_rejects_a_bad_override():
    assert ev.main(["--set", "nope=1"], out=io.StringIO()) == 2


def test_silence_length_is_a_setting():
    ledger = ev.make_ledger(2003, 9)
    short = ev.tampered_ledger("short_silence", ledger, 2003, silence_days=8)
    long_ = ev.tampered_ledger("short_silence", ledger, 2003, silence_days=20)
    assert len(long_.rows) < len(short.rows) < len(ledger.rows)


def test_report_lists_changed_settings_and_silence_length():
    data = ev.run(2, 9, 2000, ("short_silence",),
                  ev.apply_overrides({"learned": ev.PRESETS["learned"]}, ["gap_min_days=14"]),
                  silence_days=15)
    text = ev.format_report(data)
    assert "gap_min_days=14" in text and "short_silence_days=15" in text


def test_a_higher_floor_means_fewer_false_gap_alarms():
    data_low = ev.run(30, 9, 2000, (), {"learned": ev.PRESETS["learned"]})
    high = ev.apply_overrides({"learned": ev.PRESETS["learned"]}, ["gap_min_days=14"])
    data_high = ev.run(30, 9, 2000, (), high)
    lo = data_low["results"]["learned"]["genuine"]["codes"].get("FL_PERIOD_GAP", 0)
    hi = data_high["results"]["learned"]["genuine"]["codes"].get("FL_PERIOD_GAP", 0)
    assert hi <= lo
