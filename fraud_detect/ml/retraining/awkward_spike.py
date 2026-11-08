"""Task 2.2: can the existing simulator knobs reach the Awkward_Set figures?

``run_spike(seed)`` measures five case types with the public ``fraud_detect.corpus``
API only (no generator change, nothing written). ``python -m
fraud_detect.ml.retraining.awkward_spike`` prints a knob grid and writes
``data/private/ml_retraining/spike/awkward_spike.json`` through ``safe_io``.
All data is synthetic.
"""
from __future__ import annotations

import dataclasses
import json
import random
import statistics
from datetime import date
from pathlib import Path

from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.ledger import simulate_ledger
from fraud_detect.corpus.schema import SimulatorKnobs
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import EXPORT_PRODUCERS

# Knob values used by run_spike (chosen from the grid below).
SEASONALITY = 0.8
GROWTH_RANGE = (0.04, 0.04)
INSTALMENT_RATE = 0.5
N_MONTHS = 12
START = date(2025, 1, 1)

GRID_SEASONALITY = (0.15, 0.5, 0.8, 1.2, 1.6)
GRID_GROWTH = ((-0.01, 0.03), (0.0, 0.0), (0.02, 0.02), (0.04, 0.04))
GRID_SEEDS = (20260701, 20260702, 20260703, 20260704, 20260705)

PEAK_TROUGH_MIN = 2.0
LAST3_FIRST3_MIN = 1.10
ROUND_SHARE_MIN = 0.80


def _ledger(seed: int, seasonality: float, growth: tuple[float, float],
            instalment_rate: float = INSTALMENT_RATE, n_months: int = N_MONTHS):
    company = make_company(seed, "AWK1")
    knobs = SimulatorKnobs(growth_range=growth, seasonality_strength=seasonality,
                           instalment_rate=instalment_rate)
    return simulate_ledger(company, START, n_months, seed, knobs, None)


def _seasonal(ledger) -> dict:
    credits = [0.0] * len(ledger.months)
    for row in ledger.rows:
        if row.amount > 0:
            credits[row.month_index] += row.amount
    return {
        "peak_trough_ratio": max(credits) / min(credits),
        "last3_first3_ratio": sum(credits[-3:]) / sum(credits[:3]),
    }


def _instalments(ledger) -> dict:
    counts: dict[str, int] = {}
    for row in ledger.rows:
        if row.kind == "receipt" and row.invoice_ref:
            counts[row.invoice_ref] = counts.get(row.invoice_ref, 0) + 1
    return {"n_invoice_refs_on_two_receipts": sum(1 for n in counts.values() if n == 2)}


def _round_to_100(amount: float) -> float:
    sign = -1.0 if amount < 0 else 1.0
    return sign * float(max(100, int(round(abs(amount) / 100.0)) * 100))


def _round_payroll_rent(ledger) -> dict:
    rows = tuple(dataclasses.replace(r, amount=_round_to_100(r.amount))
                 if r.kind in ("payroll", "rent") else r for r in ledger.rows)
    # dry replay with cents rounding at each step, as the generator does
    cents = round(ledger.opening_balance, 2)
    bal = cents
    low = bal
    for r in rows:
        bal = round(bal + r.amount, 2)
        low = min(low, bal)
    opening = ledger.opening_balance
    if low < 0:
        opening = round(opening - low, 2)
    new = dataclasses.replace(ledger, rows=rows, opening_balance=opening)
    bal = new.opening_balance
    min_bal = bal
    for r in new.rows:
        bal = round(bal + r.amount, 2)
        min_bal = min(min_bal, bal)
    targets = [r for r in new.rows if r.kind in ("payroll", "rent")]
    round_n = sum(1 for r in targets if round(abs(r.amount) * 100) % 10000 == 0)
    return {"round_share": round_n / len(targets), "min_running_balance": float(min_bal)}


def _legit_resave(ledger, seed: int) -> dict:
    result = CATALOG["genuine_resaved"].plan(ledger, None, random.Random(seed))
    params = dict(result.file_ops[0].params)
    return {
        "producer_is_export": 1 if params["producer"] in EXPORT_PRODUCERS else 0,
        "moddate_gap_seconds": int(params["modified_delta_s"]),
    }


def run_spike(seed: int) -> dict:
    """Measure the five case types for one seed. Plain numbers, deterministic, no I/O."""
    ledger = _ledger(seed, SEASONALITY, GROWTH_RANGE)
    return {
        "seasonal_growth": _seasonal(ledger),
        "instalments": _instalments(ledger),
        "long_12_months": {"n_months": len(ledger.months)},
        "round_payroll_rent": _round_payroll_rent(ledger),
        "legit_resave": _legit_resave(ledger, seed),
    }


def knob_grid(seeds=GRID_SEEDS) -> list[dict]:
    """Seasonal/growth figures over the knob grid, per setting across ``seeds``."""
    out = []
    for s in GRID_SEASONALITY:
        for g in GRID_GROWTH:
            peaks, growths, both = [], [], 0
            for seed in seeds:
                m = _seasonal(_ledger(seed, s, g))
                peaks.append(m["peak_trough_ratio"])
                growths.append(m["last3_first3_ratio"])
                both += (m["peak_trough_ratio"] >= PEAK_TROUGH_MIN
                         and m["last3_first3_ratio"] >= LAST3_FIRST3_MIN)
            out.append({
                "seasonality_strength": s, "growth_range": list(g), "n_seeds": len(seeds),
                "peak_trough_min": min(peaks), "peak_trough_median": statistics.median(peaks),
                "last3_first3_min": min(growths), "last3_first3_median": statistics.median(growths),
                "seeds_peak_ge_2x": sum(p >= PEAK_TROUGH_MIN for p in peaks),
                "seeds_last3_ge_1_10": sum(x >= LAST3_FIRST3_MIN for x in growths),
                "seeds_both": both,
            })
    return out


def round_share_by_seed(seeds=GRID_SEEDS) -> list[dict]:
    rows = []
    for seed in seeds:
        r = _round_payroll_rent(_ledger(seed, SEASONALITY, GROWTH_RANGE))
        rows.append({"seed": seed, **r})
    return rows


def _print_table(grid: list[dict], rounds: list[dict], sample: dict) -> None:
    n = grid[0]["n_seeds"]
    print(f"Knob grid over {n} seeds (needs peak/trough >= {PEAK_TROUGH_MIN}, "
          f"last3/first3 >= {LAST3_FIRST3_MIN})")
    print(f"{'season':>7} {'growth':>14} {'pk/tr min':>9} {'pk/tr med':>9} "
          f"{'l3/f3 min':>9} {'l3/f3 med':>9} {'peak ok':>7} {'grow ok':>7} {'both':>5}")
    for g in grid:
        print(f"{g['seasonality_strength']:>7} {str(tuple(g['growth_range'])):>14} "
              f"{g['peak_trough_min']:>9.2f} {g['peak_trough_median']:>9.2f} "
              f"{g['last3_first3_min']:>9.2f} {g['last3_first3_median']:>9.2f} "
              f"{g['seeds_peak_ge_2x']:>5}/{n} {g['seeds_last3_ge_1_10']:>5}/{n} "
              f"{g['seeds_both']:>3}/{n}")
    print("\nRound payroll and rent (rewrite to multiples of 100):")
    for r in rounds:
        print(f"  seed {r['seed']}: round share {r['round_share']:.2%}, "
              f"min running balance {r['min_running_balance']:.2f}")
    print(f"\nrun_spike sample (seed {GRID_SEEDS[0]}; season={SEASONALITY}, "
          f"growth={GROWTH_RANGE}, instalment_rate={INSTALMENT_RATE}):")
    print(json.dumps(sample, indent=2, sort_keys=True))


def main() -> int:
    from fraud_detect.ml.retraining.safe_io import atomic_write

    grid = knob_grid()
    rounds = round_share_by_seed()
    sample = run_spike(GRID_SEEDS[0])
    per_seed = {str(s): run_spike(s) for s in GRID_SEEDS}
    _print_table(grid, rounds, sample)
    repo = Path(__file__).resolve().parents[3]
    out = repo / "data" / "private" / "ml_retraining" / "spike" / "awkward_spike.json"
    payload = {
        "note": "Synthetic data only; spike of existing simulator knobs.",
        "chosen": {"seasonality_strength": SEASONALITY, "growth_range": list(GROWTH_RANGE),
                   "instalment_rate": INSTALMENT_RATE, "n_months": N_MONTHS},
        "grid": grid, "round_payroll_rent": rounds, "run_spike": per_seed,
    }
    atomic_write(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
