# Node 3 Fabrication Rules: Evaluation Results

These figures are indicative only. They come from two labelled files (`statement_E`, tampered, and `statement_F`, genuine). The Rule C jump ratio default (1.2) was informed by both files, so E and F separating under it is a sanity check, not proof. Nothing was tuned afterwards: all figures use `DEFAULT_CONFIG`.

## Labelled-file acceptance

Command: `env -u AUDIT_SIGNING_KEY_PATH -u FRAUD_ENABLE_TXN_MODEL .venv/bin/python -m pytest -q -p no:cacheprovider -W ignore tests/test_fabrication_ef.py -s` (8 passed). Decision point: the Rule C check on E passed with default settings, and every check on F passed.

### statement_E (tampered, 12 inserted rows)

New-rule codes reported: `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH`. No `DQ_RULE_SKIPPED`.

| Measure | Result |
|---|---|
| Rows named by `FL_DUP_INVOICE_REF` | 24 (12 of 12 fakes + 12 genuine twins, so half of the named rows are genuine) |
| Fakes missed | none |
| Fakes named by the likelier-copy hint | 9 of 12 (rows 23, 34, 60, 98, 110, 138, 174, 183, 203) |
| Genuine rows named by the hint | 0 |
| `FL_INVOICE_SEQUENCE_BREAK` rows | 10: 9 fakes, 1 genuine (row 168 is genuine) |

The hint stays silent for the other 3 fakes, which is by design: it names a row only when exactly one row of a group breaks the invoice order.

Rule C jump months for E (jump ratio 1.2, median turnover of the other months 78,205.51):

| Month | Turnover | Median | Ratio | Fixed costs |
|---|---|---|---|---|
| 2026-04 | 97,458.03 | 78,205.51 | 1.25x | 30,278.61 |
| 2026-06 | 101,420.15 | 78,205.51 | 1.30x | 30,328.54 |
| 2026-08 | 97,549.27 | 78,205.51 | 1.25x | 30,149.24 |

Monthly figures for E (median fixed costs 30,258.97; all six months within the stability band):

| Month | Turnover | Fixed costs | Ratio vs median of other months |
|---|---|---|---|
| 2026-04 | 97,458.03 | 30,278.61 | 1.25x |
| 2026-05 | 78,205.51 | 30,239.32 | 0.80x |
| 2026-06 | 101,420.15 | 30,328.54 | 1.30x |
| 2026-07 | 65,843.78 | 30,299.56 | 0.68x |
| 2026-08 | 97,549.27 | 30,149.24 | 1.25x |
| 2026-09 | 59,453.74 | 29,878.44 | 0.61x |

### statement_F (genuine)

Nothing fires on F: no `FL_DUP_INVOICE_REF` (0 rows named, 0 hints), no `FL_INVOICE_SEQUENCE_BREAK`, no `FL_TURNOVER_COST_MISMATCH`, no `DQ_RULE_SKIPPED`.

Monthly figures for F (median fixed costs 30,325.11). The highest ratio is 1.14x, below the 1.2 default, so no month is a jump month:

| Month | Turnover | Fixed costs | Ratio vs median of other months |
|---|---|---|---|
| 2026-04 | 77,840.40 | 30,160.28 | 0.84x |
| 2026-05 | 92,286.70 | 30,296.41 | 1.13x |
| 2026-06 | 49,889.65 | 30,553.05 | 0.54x |
| 2026-07 | 92,713.16 | 30,353.81 | 1.14x |
| 2026-08 | 81,378.62 | 29,812.18 | 0.88x |
| 2026-09 | 92,615.69 | 30,490.38 | 1.14x |

Margin note: F's highest ratio (1.14x) and E's lowest jump ratio (1.25x) bracket the 1.2 default narrowly. With two files this gap is thin evidence.

### Why these weights (first principles, not fitted)

Node 4 adds up finding weights, and any MEDIUM or HIGH finding already routes to a human. The weights therefore only set how much each signal moves the score, and are placed in the range of existing Node 3 rules (`FL_MONTH_*` 12, `FL_BALANCE_BREAK` 26).

- `FL_DUP_INVOICE_REF` 10 (MEDIUM, one repeated reference): the same customer and invoice reference twice can be an honest instalment, so it is a prompt to check, not strong proof. It is kept below the 12 of a month-total mismatch.
- `FL_DUP_INVOICE_REF` 22 (HIGH, 3 or more repeated references): one customer paying in parts is plausible, but many customers doing so in one file is a pattern. The weight sits just below `FL_BALANCE_BREAK` (26). It is one summary finding, not 12 findings of 22, which would clamp the score at 100 and hide other signals.
- `FL_INVOICE_SEQUENCE_BREAK` 5 (LOW): customers do pay late and out of order, so this only supports other findings and helps name the likelier copy.
- `FL_TURNOVER_COST_MISMATCH` 8 (MEDIUM): flat fixed costs with unexplained turnover jumps is suggestive, and the check depends on a ratio informed by two files. A false alarm costs a reviewer a look and cannot reject anyone.
- `DQ_RULE_SKIPPED` 2 (LOW): a skipped check is a data-quality note, not evidence of fraud.

None of these values was adjusted after seeing E or F.

## False-alarm report

Command: `.venv/bin/python -m fraud_detect.nodes.fabrication_eval` (defaults: full genuine dataset `~/Downloads/statement_forgery_dataset`, statements in `~/Downloads`, `--generator-count 30`). Exit code 0.

| Rule | Genuine dataset docs firing (of 1,000) | Generator statements firing (of 30) |
|---|---|---|
| `FL_DUP_INVOICE_REF` | 0 | 0 |
| `FL_INVOICE_SEQUENCE_BREAK` | 0 | 0 |
| `FL_TURNOVER_COST_MISMATCH` | 0 | 0 |
| `DQ_RULE_SKIPPED` | 0 | 0 |

- Share of dataset documents carrying an invoice reference: 0.0%.
- `VF_CURRENCY_MIX` count on the dataset: 200 of 1,000.
- Re-saved genuine documents: 64. Producer counts: BrowserPrint PDF 118 (18), OS Quartz-style PDF Export (13), PrintToPDF Driver 10.0 (18), SaveAsPDF Mobile 3.2 (15). ModDate present on 64, CreationDate present on 64.

Honest reading: zero firings on the dataset means NO SIGNAL, not "verified clean". These documents carry no invoice references (0.0%), so the duplicate-reference and sequence rules have nothing to read. The dataset is single-month personal accounts, so Rule C's requirement of at least 4 full months is never met and the turnover rule cannot run. The same applies to the generator statements. The only real test of these rules against genuine data is `statement_F`, which stays clean (see the acceptance section). The false-alarm rate on genuine business statements with invoice references is therefore still unmeasured.

## Risk-score effect

Before this spec: E risk 26, F risk 40, from the earlier measured pipeline outcome tests. After: the script's figures.

| File | Before (score) | After score | After band | After routing | After codes |
|---|---|---|---|---|---|
| E (tampered) | 26 | 55 | HIGH | HUMAN_REVIEW | `VF_CURRENCY_MIX`, `FL_UEN_NOT_FOUND`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_DUP_INVOICE_REF`, `FL_TURNOVER_COST_MISMATCH` |
| F (genuine) | 40 | 26 | MEDIUM | HUMAN_REVIEW | `VF_CURRENCY_MIX`, `FL_UEN_NOT_FOUND` |

E above 25 and above F: yes.

Synthetic clean sample (`generate_clean`, default seed, run through `run_pipeline`): risk 1, band LOW, routing AUTO_CLEAR. This figure is not in the eval script output; I ran it separately without a case store.

What moved each score:
- E, 26 to 55 (+29): `FL_DUP_INVOICE_REF` HIGH +22, `FL_INVOICE_SEQUENCE_BREAK` +5, `FL_TURNOVER_COST_MISMATCH` +8. These three weights sum to 35, but the observed rise is 29; I did not trace how Node 4 combines the weights, so I do not explain the difference here.
- F, 40 to 26: the earlier `FL_STRUCTURING` HIGH (20) is gone. The 3 near-threshold credits are on 28 Jul, 1 Aug and 28 Aug, so they are not within 14 days of each other. F still routes HUMAN_REVIEW because of `FL_UEN_NOT_FOUND` and `VF_CURRENCY_MIX`, both MEDIUM (confirmed by the codes in the script output).

Caveats: Node 4 weights are unchanged. The new rule weights were chosen from first principles, not fitted. E and F are the only business files, so this is indicative.

Data safety: `data/cases.json` was byte-identical before and after (md5 e66f9cb35a6d26f3eee050b8dadcee35). `data/audit_chain.jsonl` grew from 474 to 484 lines during the run.

## Appendix: script output

```
dataset firing FL_DUP_INVOICE_REF: 0 of 1000
dataset firing FL_INVOICE_SEQUENCE_BREAK: 0 of 1000
dataset firing FL_TURNOVER_COST_MISMATCH: 0 of 1000
dataset firing DQ_RULE_SKIPPED: 0 of 1000
dataset invoice reference share: 0.0%
dataset VF_CURRENCY_MIX: 200
resaved documents: 64
resaved Producer BrowserPrint PDF 118: 18
resaved Producer OS Quartz-style PDF Export: 13
resaved Producer PrintToPDF Driver 10.0: 18
resaved Producer SaveAsPDF Mobile 3.2: 15
resaved ModDate present: 64
resaved CreationDate present: 64
generator firing FL_DUP_INVOICE_REF: 0 of 30
generator firing FL_INVOICE_SEQUENCE_BREAK: 0 of 30
generator firing FL_TURNOVER_COST_MISMATCH: 0 of 30
generator firing DQ_RULE_SKIPPED: 0 of 30
metadata E Producer=OpenPage Writer 3.8 Creator=OpenPage Writer 3.8 CreationDate=D:20261003023600+08'00' ModDate=D:20261005133600+08'00'
metadata F Producer=Vellichor Paperless 1.12 Creator=VCB Statements CreationDate=D:20261001034500+08'00' ModDate=D:20261001034500+08'00'
E score=55 band=HIGH routing=HUMAN_REVIEW codes=VF_CURRENCY_MIX,FL_UEN_NOT_FOUND,FL_INVOICE_SEQUENCE_BREAK,FL_DUP_INVOICE_REF,FL_TURNOVER_COST_MISMATCH
F score=26 band=MEDIUM routing=HUMAN_REVIEW codes=VF_CURRENCY_MIX,FL_UEN_NOT_FOUND
E above 25 and above F: yes
elapsed (longest file): 0.34s
```
