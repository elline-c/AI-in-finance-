"""Month-by-month ledger simulator (Requirements 1.1, 1.3 to 1.12).

The ledger is the truth the renderers draw from. Everything is invented and
depends only on the arguments (private ``random.Random`` streams from
``stable_seed``; no global state).

Choices worth knowing about (the tests pin most of them):

* Receipts are ``<RAIL> <CUSTOMER> INV<n>`` with RAIL in FAST FROM, PAYNOW FROM,
  GIRO. Invoice numbers are handed out in receipt order, so they never decrease
  with date. An instalment invoice is paid in two parts that repeat the same
  (customer, reference); to keep the numbering monotone the two parts are always
  consecutive receipts: either the same day, or the last receipt of one month and
  the first receipt of the next (one such cross-month pair per month boundary).
* ``invoice_disorder`` swaps the numbers of that share of the (non-instalment)
  receipts, using its own random stream.
* The row cap is applied after the full month has been drawn, so the cap never
  changes any random draw: card payments go first (smallest first), then
  suppliers, then plain receipts, then utilities. Fixed costs are never dropped.
* Genuine credits avoid Node 3's shell words (names come from ``company.py``),
  and no receipt falls in the 9,500 to 9,999.99 structuring band.
"""
from __future__ import annotations

import calendar
import math
from datetime import date, timedelta

from .company import utility_names
from .schema import CompanyProfile, Ledger, LedgerRow, SimulatorKnobs
from .seeding import rng

_RAILS = ("FAST FROM", "PAYNOW FROM", "GIRO")
_RAIL_WEIGHTS = (0.55, 0.30, 0.15)
_MERCHANTS = (
    "STATIONERY MART", "PRINTWORKS", "COURIER HUB", "FUEL STOP", "OFFICE PANTRY",
    "SOFTWARE SUBSCRIPTION", "HARDWARE DEPOT", "PARKING SERVICES", "CLEANING SUPPLIES",
)
_CPF_RATE = 0.17
# drop order when a cap forces rows out of a month
_DROP_ORDER = ("card", "supplier", "plain_receipt", "utility")


def _add_months(first: date, n: int) -> date:
    idx = first.year * 12 + (first.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def _month_end(first: date) -> date:
    return first.replace(day=calendar.monthrange(first.year, first.month)[1])


def _weekday(d: date, first: date, last: date, forward: bool = True) -> date:
    """Nearest weekday on or after (forward) / before ``d``, kept inside the month."""
    for step in ((1, -1) if forward else (-1, 1)):
        x = d
        while x.weekday() >= 5:
            x += timedelta(days=step)
        if first <= x <= last:
            return x
    return d  # unreachable: every month has weekdays near any day


def _weekdays_of(first: date) -> list[date]:
    return [first.replace(day=k) for k in range(1, _month_end(first).day + 1)
            if first.replace(day=k).weekday() < 5]


def _money(x: float, floor: float = 1.0) -> float:
    return round(max(floor, x), 2)


def replay(ledger: Ledger) -> list[float]:
    """Balance after every row, rounded to cents at each step."""
    out: list[float] = []
    bal = ledger.opening_balance
    for row in ledger.rows:
        bal = round(bal + row.amount, 2)
        out.append(bal)
    return out


def _split_parts(r, amount: float) -> tuple[float, float, str]:
    if r.random() < 0.4:
        half = _money(amount / 2)
        return half, half, "instalment_identical"
    p1 = _money(amount * r.uniform(0.4, 0.7))
    return p1, _money(amount - p1), "instalment"


def simulate_ledger(company: CompanyProfile, start: date, n_months: int, seed: int,
                    knobs: SimulatorKnobs, max_rows_per_month: int | None) -> Ledger:
    r = rng(seed, "ledger", company.company_id)
    first_month = start.replace(day=1)
    months = tuple(_add_months(first_month, i) for i in range(n_months))
    last_day = _month_end(months[-1])

    growth = r.uniform(*knobs.growth_range)
    phase = r.uniform(0.0, 12.0)
    utils = utility_names(company)
    util_base = {u: round(company.base_turnover * r.uniform(0.003, 0.010), 2) for u in utils}
    rent_base = round(company.base_turnover * r.uniform(0.05, 0.10), 2)
    base = company.base_turnover or 100_000.0

    # planned turnover per month: growth x seasonality x small noise
    planned = []
    for m in range(n_months):
        # half-amplitude: a full-strength swing makes ordinary genuine peaks look
        # like Node 3's "turnover jumps while fixed costs stay flat" pattern
        season = 1.0 + 0.5 * knobs.seasonality_strength * math.sin(2 * math.pi * (m + phase) / 12)
        planned.append(base * (1 + growth) ** m * season * r.uniform(0.96, 1.04))

    seq = 0
    others: list[dict] = []     # non-receipt rows
    receipts: list[dict] = []   # receipt rows (before numbering)

    def add_other(m, d, desc, amount, kind, slot=0):
        nonlocal seq
        seq += 1
        others.append(dict(m=m, date=d, desc=desc, amount=round(amount, 2), kind=kind,
                           slot=slot, pid=0, part=0, seq=seq))

    lo, hi = knobs.rows_per_month_range
    cust_weights = [p.weight for p in company.customers]
    sup_weights = [p.weight for p in company.suppliers]
    n_inst = 0
    for m, first in enumerate(months):
        last = _month_end(first)
        T = planned[m]
        wdays = _weekdays_of(first)
        quarter_end = first.month in (3, 6, 9, 12)
        n_target = r.randint(lo, hi)
        fixed_n = 5 + len(utils) + (1 if quarter_end else 0)
        v = max(6, n_target - fixed_n)
        k_rec = max(3, round(v * 0.5))
        k_sup = max(1, round(v * 0.2))
        k_card = max(1, v - k_rec - k_sup)

        # fixed costs ------------------------------------------------------
        payroll = _money(company.payroll_base * r.uniform(0.985, 1.015))
        add_other(m, _weekday(first.replace(day=min(last.day, r.randint(25, 28))), first, last, False),
                  "PAYROLL BATCH", -payroll, "payroll")
        add_other(m, _weekday(first.replace(day=14), first, last, True),
                  "CPF CONTRIBUTION", -round(payroll * _CPF_RATE, 2), "cpf")
        add_other(m, _weekday(first.replace(day=r.randint(1, 3)), first, last, True),
                  f"RENT {company.landlord}", -rent_base, "rent")
        for u in utils:
            add_other(m, _weekday(first.replace(day=r.randint(5, 25)), first, last, True),
                      f"BILL PAYMENT {u}", -_money(util_base[u] * r.uniform(0.9, 1.1)), "utility")
        if quarter_end:
            q_total = sum(planned[k] for k in range(n_months)
                          if months[k].year == first.year and (months[k].month - 1) // 3 == (first.month - 1) // 3)
            add_other(m, _weekday(first.replace(day=r.randint(20, 28)), first, last, False),
                      "GST PAYMENT", -_money(q_total * r.uniform(0.025, 0.045)), "gst")
        add_other(m, last, "SERVICE CHARGE", -round(r.uniform(8.0, 25.0), 2), "service_charge")
        add_other(m, last, "INTEREST CREDIT", _money(base * r.uniform(0.00002, 0.0001), 0.5), "interest")

        # suppliers and cards ---------------------------------------------
        sw = [r.uniform(0.5, 1.5) for _ in range(k_sup)]
        s_total = T * r.uniform(0.18, 0.28)
        for w in sw:
            sup = r.choices(company.suppliers, sup_weights)[0]
            add_other(m, r.choice(wdays), f"FAST TO {sup.name}", -_money(s_total * w / sum(sw)), "supplier")
        cw = [r.uniform(0.3, 1.7) for _ in range(k_card)]
        c_total = T * r.uniform(0.04, 0.08)
        for w in cw:
            add_other(m, r.choice(wdays), f"NETS/CARD PAYMENT {r.choice(_MERCHANTS)}",
                      -_money(c_total * w / sum(cw)), "card")

        # receipts ---------------------------------------------------------
        rw = [r.uniform(0.5, 1.5) for _ in range(k_rec)]
        for w in rw:
            cust = r.choices(company.customers, cust_weights)[0]
            amount = _money(T * w / sum(rw))
            if 9500.0 <= amount < 10000.0:
                amount = round(amount - 600.0, 2)
            seq += 1
            receipts.append(dict(m=m, date=r.choice(wdays), customer=cust.name,
                                 rail=r.choices(_RAILS, _RAIL_WEIGHTS)[0], amount=amount,
                                 slot=0, pid=0, part=0, seq=seq, hard=None,
                                 marked=r.random() < knobs.instalment_rate,
                                 cross_roll=r.random()))

    # instalments --------------------------------------------------------
    by_month: dict[int, list[dict]] = {}
    for rc in receipts:
        by_month.setdefault(rc["m"], []).append(rc)
    firsts = {m: min(x["date"] for x in rows) for m, rows in by_month.items()}
    lasts = {m: max(x["date"] for x in rows) for m, rows in by_month.items()}
    seconds: list[dict] = []
    for m in range(n_months):
        cross_used = False
        for rc in [x for x in by_month.get(m, []) if x["marked"]]:
            # Only cross-month pairs (one per month boundary): a same-month repeat
            # would be a Node 3 same-month duplicate, which is not what a genuine
            # instalment looks like. An invoice that cannot be split stays whole.
            if cross_used or m + 1 >= n_months or (m + 1) not in by_month:
                continue
            n_inst += 1
            p1, p2, tag = _split_parts(r, rc["amount"])
            cross_used = True
            second = dict(rc)
            rc.update(amount=p1, hard=tag, pid=n_inst, part=0, date=lasts[m], slot=2)
            second.update(amount=p2, hard=tag, pid=n_inst, part=1, marked=False,
                          m=m + 1, date=firsts[m + 1], slot=-1)
            seq += 1
            second["seq"] = seq
            seconds.append(second)
    receipts.extend(seconds)

    # invoice numbers in receipt order ------------------------------------
    receipts.sort(key=lambda x: (x["date"], x["slot"], x["pid"], x["part"], x["seq"]))
    counter = company.first_invoice_number
    nums: dict[int, int] = {}
    for rc in receipts:
        if rc["pid"] and rc["part"] == 1:
            rc["num"] = nums[rc["pid"]]
            continue
        counter += 1 + (1 if r.random() < 0.3 else 0)
        rc["num"] = counter
        if rc["pid"]:
            nums[rc["pid"]] = counter
    if knobs.invoice_disorder > 0:
        rd = rng(seed, "disorder", company.company_id)
        plain = [i for i, rc in enumerate(receipts) if not rc["pid"]]
        k = int(len(plain) * knobs.invoice_disorder) // 2 * 2
        picked = rd.sample(plain, min(k, len(plain) // 2 * 2))
        for a, b in zip(picked[0::2], picked[1::2]):
            receipts[a]["num"], receipts[b]["num"] = receipts[b]["num"], receipts[a]["num"]

    # assemble rows in print order ---------------------------------------
    items: list[tuple] = []
    for rc in receipts:
        items.append(((rc["date"], rc["slot"], rc["pid"], rc["part"], rc["seq"]), dict(
            m=rc["m"], date=rc["date"], desc=f"{rc['rail']} {rc['customer']} INV{rc['num']}",
            amount=rc["amount"], kind="receipt", customer=rc["customer"],
            ref=f"INV{rc['num']}", hard=rc["hard"])))
    for o in others:
        items.append(((o["date"], o["slot"], 0, 0, o["seq"]), dict(
            m=o["m"], date=o["date"], desc=o["desc"], amount=o["amount"], kind=o["kind"],
            customer=None, ref=None, hard=None)))
    items.sort(key=lambda t: t[0])
    rows = [LedgerRow(row_id=f"R{i:04d}", month_index=d["m"], date=d["date"], description=d["desc"],
                      amount=d["amount"], kind=d["kind"], customer=d["customer"],
                      invoice_ref=d["ref"], hard_case=d["hard"])
            for i, (_, d) in enumerate(items, start=1)]

    rows, trims = _apply_cap(rows, max_rows_per_month)

    # opening balance from a dry run --------------------------------------
    bal, low = 0.0, 0.0
    for row in rows:
        bal = round(bal + row.amount, 2)
        low = min(low, bal)
    opening = round(-low + base * r.uniform(0.05, 0.20), 2)

    return Ledger(company=company, opening_balance=opening, months=months,
                  rows=tuple(rows), trims=trims)


def _apply_cap(rows: list[LedgerRow], cap: int | None):
    if cap is None:
        return rows, ()
    drop: set[str] = set()
    trims: list[tuple[int, int]] = []
    for m in sorted({r.month_index for r in rows}):
        month_rows = [r for r in rows if r.month_index == m]
        excess = len(month_rows) - cap
        if excess <= 0:
            continue

        def group(row: LedgerRow) -> str | None:
            if row.kind == "receipt":
                return "plain_receipt" if row.hard_case is None else None
            return row.kind if row.kind in ("card", "supplier", "utility") else None

        dropped = 0
        for g in _DROP_ORDER:
            cands = sorted((r for r in month_rows if group(r) == g),
                           key=lambda r: (abs(r.amount), r.row_id))
            for row in cands:
                if dropped >= excess:
                    break
                drop.add(row.row_id)
                dropped += 1
        if dropped:
            trims.append((m, dropped))
    return [r for r in rows if r.row_id not in drop], tuple(trims)
