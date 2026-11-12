"""Calibrate (or reject) month-by-month scoring for the Node 6b statement model.

Node 6b was trained on single-month statements and abstains on longer ones. It can
instead score each calendar month and keep the highest score, but the maximum of
several scores needs its own cutoffs. This tool scores every multi-month document
of a labelled corpus that way and reports, for a range of cutoffs, the false-alarm
rate on genuine documents and the catch rate on tampered ones. It separates the
tamper types a file-forensics model can see (balance breaks, fonts, metadata,
edited amounts) from behavioural fraud patterns it cannot.

Cutoffs are chosen on genuine documents of ``--splits`` only: ``medium`` is the lowest
cutoff whose false-alarm rate is at most ``--target-fa`` (``high`` uses half of it).
With ``--check-splits`` those fixed cutoffs are then applied, unchanged, to other
splits (for example a layout the model never trained on) as an independent check.

The cutoffs pass when, at ``medium``, the model catches at least ``--min-catch`` of the
visible tamper types on the calibration splits AND on the check splits, with false
alarms on the check splits at most twice ``--target-fa``. They are written into the
bundle as ``multi_month_thresholds`` only with ``--write`` and only if they pass;
otherwise the bundle is left unchanged and the node keeps abstaining on long
statements.

Synthetic data: the numbers describe this corpus, not real-world accuracy.

    python -m fraud_detect.ml.calibrate_statement_months \
        --corpus data/corpus_large --model models/statement_forgery_clf.joblib
"""
from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import statement_features as sf

# Tamper types that change the file in ways the Node 6b features measure.
VISIBLE_TYPES = {"broken_balance", "font_mismatch", "metadata_inconsistency", "edited_amount"}


def _cutoff_for(genuine: np.ndarray, target: float) -> float | None:
    """Lowest cutoff from the genuine scores whose false-alarm rate is <= ``target``."""
    for c in np.unique(np.concatenate([genuine, [1.0 + 1e-9]])):
        if (genuine >= c).mean() <= target:
            return float(c)
    return None


def _parse_splits(text: str) -> set[str]:
    return {s.strip() for s in text.split(",") if s.strip()}


def score_corpus(corpus: Path, rows: list[dict], model) -> dict:
    """Max-month score of every multi-month document in ``rows``.

    Returns arrays ``scores``, ``tampered``, ``visible`` and ``kinds`` plus
    ``skipped`` (documents with no scorable month)."""
    from ..nodes import ml_statement as ms

    scores, labels, kinds, skipped = [], [], [], 0
    for r in rows:
        if int(r.get("n_months") or 1) < 2:
            continue
        res = ms.assess_by_month(sf.month_segments(str(corpus / r["file"])), model)
        if res["abstain"]:
            skipped += 1
            continue
        scores.append(res["prob"])
        labels.append(int(r["is_tampered"]))
        kinds.append(r.get("tamper_types") or "")
    y = np.array(labels, dtype=int)
    return {"scores": np.array(scores, dtype=float), "tampered": y == 1,
            "visible": np.array([k in VISIBLE_TYPES for k in kinds], dtype=bool),
            "kinds": kinds, "skipped": skipped}


def _summary(name: str, sc: dict) -> None:
    s, t, v = sc["scores"], sc["tampered"], sc["visible"]
    print(f"[{name}] {len(s)} multi-month documents scored ({sc['skipped']} had no scorable "
          f"month): {int((~t).sum())} genuine, {int((t & v).sum())} visible tampers, "
          f"{int((t & ~v).sum())} behavioural tampers")


def _curve(sc: dict) -> None:
    s, t, v = sc["scores"], sc["tampered"], sc["visible"]
    gen, tv, to = s[~t], s[t & v], s[t & ~v]
    print(f"{'cutoff':>8} {'false alarms':>13} {'visible caught':>15} {'behavioural caught':>19}")
    for c in (0.5, 0.7, 0.9, 0.99, 0.999, 0.9999):
        def pct(a):
            return f"{100 * (a >= c).mean():.1f}%" if len(a) else "n/a"
        print(f"{c:>8} {pct(gen):>13} {pct(tv):>15} {pct(to):>19}")


def _at(sc: dict, cut: float) -> tuple[float, float]:
    """(false-alarm rate, visible catch rate) at ``cut``."""
    s, t, v = sc["scores"], sc["tampered"], sc["visible"]
    gen, tv = s[~t], s[t & v]
    fa = float((gen >= cut).mean()) if len(gen) else 0.0
    catch = float((tv >= cut).mean()) if len(tv) else 0.0
    return fa, catch


def _by_type(sc: dict, cut: float) -> None:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for k, p, t in zip(sc["kinds"], sc["scores"], sc["tampered"]):
        key = k or "genuine"
        agg[key][0] += 1
        agg[key][1] += int(p >= cut)
    for k, (n, f) in sorted(agg.items()):
        print(f"    {k:<24} flagged {f:>3} / {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="data/corpus_large")
    ap.add_argument("--model", default="models/statement_forgery_clf.joblib")
    ap.add_argument("--target-fa", type=float, default=0.05)
    ap.add_argument("--min-catch", type=float, default=0.5)
    ap.add_argument("--write", action="store_true", help="store the cutoffs in the bundle if they pass")
    ap.add_argument("--splits", default="",
                    help="comma-separated corpus splits to choose cutoffs on (default: all). A "
                         "model trained on part of the corpus must use the other splits only.")
    ap.add_argument("--check-splits", default="",
                    help="comma-separated splits on which the chosen cutoffs are checked unchanged")
    args = ap.parse_args(argv)
    splits, check = _parse_splits(args.splits), _parse_splits(args.check_splits)
    if splits & check:
        print("--splits and --check-splits must not overlap.")
        return 1

    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(args.model)
    if not isinstance(bundle, dict):
        print("Model file is a bare estimator with no size guard; nothing to calibrate.")
        return 1
    scoring = {k: v for k, v in bundle.items() if k != "multi_month_thresholds"}

    corpus = Path(args.corpus)
    all_rows = list(csv.DictReader(open(corpus / "labels.csv", newline="")))
    # Split names repeat across corpora, so the check applies only to the corpus the
    # model was actually trained on; a separately built corpus may be used whole.
    trained_on = set(bundle.get("corpus_train_splits") or ())
    same_corpus = bool(bundle.get("corpus")) and Path(bundle["corpus"]).resolve() == corpus.resolve()
    if trained_on and same_corpus and (not splits or (splits | check) & trained_on):
        print(f"Refusing: this model was trained on corpus split(s) {sorted(trained_on)}; "
              f"pass --splits / --check-splits with held-out splits only.")
        return 1

    rows = [r for r in all_rows if not splits or r.get("split") in splits]
    cal = score_corpus(corpus, rows, scoring)
    _summary("calibration " + (",".join(sorted(splits)) or "all"), cal)
    if not (~cal["tampered"]).any():
        print("No genuine documents to calibrate on.")
        return 1
    _curve(cal)

    medium = _cutoff_for(cal["scores"][~cal["tampered"]], args.target_fa)
    high = _cutoff_for(cal["scores"][~cal["tampered"]], args.target_fa / 2)
    fa, catch = _at(cal, medium) if medium is not None else (1.0, 0.0)
    print(f"\nCutoffs for <= {args.target_fa:.0%} false alarms: medium={medium}, high={high}")
    print(f"  calibration: false alarms {fa:.1%}, visible tampers caught {catch:.0%} "
          f"(needs {args.min_catch:.0%})")
    passed = medium is not None and high is not None and catch >= args.min_catch
    if medium is not None:
        _by_type(cal, medium)

    chk_info = None
    if check and medium is not None:
        chk = score_corpus(corpus, [r for r in all_rows if r.get("split") in check], scoring)
        print()
        _summary("check " + ",".join(sorted(check)), chk)
        cfa, ccatch = _at(chk, medium)
        print(f"  check at the same medium cutoff: false alarms {cfa:.1%} (max "
              f"{2 * args.target_fa:.0%}), visible tampers caught {ccatch:.0%} "
              f"(needs {args.min_catch:.0%})")
        _by_type(chk, medium)
        passed = passed and cfa <= 2 * args.target_fa and ccatch >= args.min_catch
        chk_info = {"splits": sorted(check), "false_alarms": cfa, "visible_caught": ccatch}

    if not passed:
        print("\nREJECTED: month-by-month scoring does not separate genuine from tampered "
              "documents well enough for this model. Bundle left unchanged; Node 6b keeps "
              "abstaining on long statements.")
        return 2
    if args.write:
        bundle["multi_month_thresholds"] = {
            "medium": medium, "high": high, "target_fa": args.target_fa, "corpus": str(corpus),
            "splits": sorted(splits) or "all", "check": chk_info,
            "n_genuine": int((~cal["tampered"]).sum()),
            "n_visible": int((cal["tampered"] & cal["visible"]).sum()),
            "calibration_false_alarms": fa, "calibration_visible_caught": catch,
        }
        joblib.dump(bundle, args.model)
        print(f"\nPASSED. Wrote multi_month_thresholds to {args.model}.")
    else:
        print("\nPASSED. Re-run with --write to store these cutoffs in the bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
