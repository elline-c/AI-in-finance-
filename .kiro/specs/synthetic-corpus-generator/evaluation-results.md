# Evaluation results: synthetic corpus generator

**Read this first.** Every number below comes from fictional, computer-made bank statements. They are demonstrations. They are **not** real-world accuracy and must not be quoted as if they were. The samples are small (a handful of documents per fraud type), so one document moving changes a percentage by 25 to 100 points.

Date of the runs: 2 November 2026. Generator_Version: 1.0.0. Machine: the development laptop (macOS).

## 1. What was built

A small program that makes fake but realistic company bank statements as PDFs, in the six layouts that Node 1 (the statement reader) already knows. For each PDF it also writes the "answer": which rows are fake, what was changed, how, and in which layout. It can make clean ("genuine") statements and 16 kinds of tampered ones. Then a second command feeds all the PDFs through Nodes 1 to 4 and reports what they read and what they flagged.

The fake companies are registered only for the length of the measurement, in memory. Nothing in Nodes 1 to 4 and nothing in the reference data was changed. The PDFs are in `data/` which git ignores; none is committed.

## 2. How it was run

| | Smoke corpus | Large corpus |
|---|---|---|
| Folder | `data/corpus` | `data/corpus_large` |
| Seed | 20260601 | 20260602 |
| Documents | 24 (2 genuine + 2 tampered per layout) | 120 (10 genuine + 10 tampered per layout) |
| Genuine / tampered | 12 + 12 | 60 + 60 (4 of the "tampered" are genuine re-saved files, so 56 truly tampered) |
| Held-out layout | `two_col_ddmmyy_cr` | `two_col_ddmmyy_cr` |
| Splits | train 13, validation 5, test 2, test_unseen_layout 4 | train 69, validation 17, test 14, test_unseen_layout 20 |
| Measuring time (`verify`) | 22.2 s | 120.9 s |
| Build time | not recorded in the logs | not recorded in the logs |

Both used an even mix of all 16 tamper types. The smoke corpus has only one document per tamper type, which is too few to say anything, so the large corpus (3 to 4 per type) is the one used for most of the figures. Even that is thin.

**Time to render one 12-month statement** (measured afterwards, in memory, same laptop, one example per layout, about 400 to 450 rows and 12 to 15 pages): between 0.02 and 0.03 seconds per layout. The company simulation itself rounds to 0.00 s. Far below the 30 second ceiling in the requirements.

## 3. What Node 1 reads (large corpus, 20 documents per layout)

| Layout | Rows read | Genuine rows | Tampered rows | Printed balance matches | Header matches | Finding codes |
|---|---|---|---|---|---|---|
| generator_v1 | 100% | 100% | 100% | 100% | 95% | DQ_DIRECTION_UNCERTAIN |
| signed_amount_iso | 100% | 100% | 100% | 90% | 95% | none |
| two_col_ddmmyy_cr | 100% | 100% | 100% | 100% | 100% | none |
| two_col_dmon | 100% | 100% | 100% | 100% | 100% | none |
| two_col_dual_date | 100% | 100% | 100% | 100% | 95% | none |
| two_col_slash | 100% | 100% | 100% | 100% | 100% | none |

In the smoke corpus every column was 100% in every layout. Node 1 picked the right layout profile for every document in both corpora. Overlay rows dropped by Node 1: **0** in both.

### The 90% and 95% figures: what is going on

I looked at each document behind them in `data/corpus_large/report.json` and matched it with `labels.csv`. There are six documents, and **all six are deliberately tampered ones, none is genuine**. Node 1 read every genuine document perfectly (100% rows, balance and header on all 60).

| Document | Column that failed | Tamper on it | Why |
|---|---|---|---|
| generator_v1-t0002 | Header | uen_mismatch | The statement prints a company name that differs on purpose from the registered name, so "header matches the label" is false by design |
| signed_amount_iso-t0010 | Header | uen_mismatch | Same |
| two_col_dual_date-t0007 | Header | uen_mismatch | Same |
| signed_amount_iso-t0002 | Balance | edited_amount (rerender) | One amount was changed and the later balances were **not** recomputed (`arithmetic_consistent` = false). The replayed balance then cannot equal the printed one, by design |
| signed_amount_iso-t0004 | Balance | edited_amount (rerender) | Same |
| generator_v1-t0005 | none (shows DQ_DIRECTION_UNCERTAIN) | broken_balance (rerender) | One printed balance is wrong on purpose, so Node 1 could not be sure whether that row was a debit or credit and said so |

So the dips are the tampers working as designed, shown through a measuring column that was written for genuine documents. They are **not** reading mistakes. I did not change any Node or the generator. A small caveat for the next person: the Node 1 table mixes genuine and tampered documents in the Balance and Header columns, which makes the 90% and 95% look like a reading problem. Splitting those two columns into genuine and tampered would remove the confusion (suggestion only).

## 4. False alarms on genuine documents

| Corpus | Genuine documents | Auto-cleared | Findings seen |
|---|---|---|---|
| Smoke | 12 | 100% | FL_DUP_INVOICE_REF on 10 |
| Large | 60 | 93.3% (56 of 60) | FL_DUP_INVOICE_REF on 50, FL_TURNOVER_COST_MISMATCH on 4 |
| Large, genuine re-saved | 4 | 100% | FL_DUP_INVOICE_REF on 3 |

**FL_TURNOVER_COST_MISMATCH, 4 of 60 (6.7%).** This is the only real false alarm, and the only reason any genuine document went to human review. The four are signed_amount_iso-g0007, two_col_ddmmyy_cr-g0001, two_col_dual_date-g0001 and two_col_dual_date-g0006. All four also had the DUP note. I did not dig into the rule, so I cannot say why these four. I did **not** see evidence that it is the "TT" substring check (the report holds no row-level detail to show that), so I am not claiming it.

**FL_DUP_INVOICE_REF, 50 of 60 genuine documents.** The claim was that this is the intended low-level note that comes from instalments and repeated payments, and that it does not block auto-clear. Checking it against the data:

- Every one of the 60 genuine documents carries the genuine-hard-case tags `instalment` (46 of them also `instalment_identical`). The generator makes customers pay the same invoice in parts on purpose. That matches the note.
- Of the 50 documents with the DUP note, **46 were auto-cleared**. The 4 that were not cleared are exactly the 4 that also had FL_TURNOVER_COST_MISMATCH. So DUP alone never sent a genuine document to review: **confirmed**.
- The code in `fraud_detect/nodes/fabrication.py` gives the "reused across months, close enough to be an instalment" case severity LOW, and calls it "weak signal only". The report itself does not store severity, so this part comes from reading the code, not from the data.
- The 10 genuine documents without the note are the ones where the random draw gave no repeated reference.

One side effect: because the note shows up on most clean documents, it also shows up on most tampered ones (almost every row in the tamper table carries it). It tells you nothing about tampering.

## 5. Detection by tamper type (large corpus)

"Material" means Nodes 2 to 4 raised at least one MEDIUM or HIGH finding. "Expected code" is whether the specific rule we would naturally expect fired. That column is descriptive, not a target. "n/a" means no code was named for that type.

| Tamper type | Method | Docs | Material / not auto-cleared | Expected code seen | Codes seen (excluding the DUP note) |
|---|---|---|---|---|---|
| altered_date | overlay | 3 | 33.3% | 0% | FL_TURNOVER_COST_MISMATCH |
| altered_date | rerender | 1 | 100% | 0% | FL_TURNOVER_COST_MISMATCH |
| broken_balance | overlay | 1 | **0%** | 0% | none |
| broken_balance | rerender | 3 | 100% | 100% | FL_BALANCE_BREAK, DQ_DIRECTION_UNCERTAIN |
| cross_month_copy | content | 4 | 100% | 100% | FL_INVOICE_SEQUENCE_BREAK |
| dormant_reactivation | rerender | 3 | 100% | n/a | FL_PERIOD_GAP |
| edited_amount | rerender | 4 | 50% | 50% | FL_BALANCE_BREAK |
| fabricated_receipts | content | 4 | 75% | 75% | FL_INVOICE_SEQUENCE_BREAK |
| font_mismatch | overlay | 4 | 100% | 100% | VF_FONT_MISMATCH |
| metadata_inconsistency | file | 4 | 100% | 100% | VF_EDIT_SOFTWARE, VF_CREATION_BEFORE_PERIOD, FL_TURNOVER_COST_MISMATCH |
| period_gap | rerender | 3 | 100% | 100% | FL_MONTH_MISSING, FL_PERIOD_GAP |
| rapid_pass_through | rerender | 3 | **0%** | n/a | none |
| shell_counterparty | rerender | 4 | 75% | 75% | FL_UNVERIFIED_COUNTERPARTY |
| structuring | rerender | 4 | 75% | 75% | FL_STRUCTURING |
| sudden_large_deposit | rerender | 4 | 100% | 50% | FL_UNVERIFIED_COUNTERPARTY, FL_SUDDEN_DEPOSIT |
| turnover_inflation | content | 4 | 50% | 50% | FL_INVOICE_SEQUENCE_BREAK, FL_TURNOVER_COST_MISMATCH |
| uen_mismatch | content | 3 | 100% | 100% | FL_UEN_NAME_MISMATCH, FL_TURNOVER_COST_MISMATCH |

Overall, 43 of the 56 truly tampered documents (76.8%) drew a MEDIUM or HIGH finding. That is the sum of very small groups and says little about any single type. The 4 genuine re-saved files were all auto-cleared, as they should be.

Things worth noticing:

- **edited_amount (rerender), 2 of 4 caught.** The two caught are the ones where the later balances were left unchanged, so the arithmetic breaks. The two missed (two_col_dual_date-t0010, two_col_slash-t0002) had their balances recomputed so the sums still add up; with no font change in a re-rendered file there is nothing left to see. That is a realistic limit of what a statement alone can show.
- **fabricated_receipts, 75%.** The one miss (generator_v1-t0007) came out of screening with no findings at all, not even the DUP note.
- **metadata_inconsistency, 4 of 4** is a good result but the easy one: the file literally names an editing tool.
- **sudden_large_deposit** was caught every time, but only half the time by the rule named for it (see section 6).

## 6. Blind spots

The report's automatic rule lists a type as a blind spot when no MEDIUM/HIGH finding fires on it. On the large corpus that gave two. I add the others I noticed.

| Blind spot | What was measured | What it means, plainly |
|---|---|---|
| broken_balance as overlay (1 doc) | 0% material; FL_BALANCE_BREAK expected, not seen | The balance was patched by drawing over the PDF, and Node 1 did not drop the row (dropped count is 0). Yet the balance rule did not fire. With one document I cannot say whether it is a Node 1 or a Node 3 matter. The smoke corpus shows the same miss on its one overlay document |
| rapid_pass_through (3 docs) | 0% material; nothing at all fired | Money in and out again within days is not looked for by any rule. No node has a rule for it |
| altered_date as overlay (3 docs) | 33% material, and only by the unrelated TURNOVER code | The date rule VF_MIXED_DATE_FORMATS **never fired** in either corpus, for overlay or for re-rendered tampers. The one overlay and one re-render that got flagged did so for a different reason (side effect on turnover figures) |
| sudden_large_deposit | Flagged 4 of 4, but FL_SUDDEN_DEPOSIT fired on only 2 | The other two were caught by FL_UNVERIFIED_COUNTERPARTY instead (the deposit was from an unknown name). Either the sudden-deposit rule's threshold is higher than the generator's "moderate" variant, or the expected code in the label is too narrow; I did not check which. Smoke corpus: 0 of 1 |
| Nearby-month copies with different amounts | cross_month_copy was caught 4 of 4 | Not a blind spot in this sample, but the catch is by FL_INVOICE_SEQUENCE_BREAK, not by the duplicate rule. The smoke corpus missed its one cross_month_copy |
| Overlay rows dropped by Node 1 | **0** | This known risk (Node 1 might drop the pasted-over row) was **not observed** in these runs. It remains untested rather than cleared, because only a few overlay documents were made |

Smoke corpus extras: its blind-spot list had four entries (altered_date, broken_balance, cross_month_copy, edited_amount), all with one document each, so those are the luck of the draw and not a pattern.

## 7. Results by split (large corpus)

| Split | Documents | Genuine auto-clear | Material | Not auto-cleared |
|---|---|---|---|---|
| train | 69 | 94.1% | 76.5% | 76.5% |
| validation | 17 | 88.9% | 75.0% | 75.0% |
| test | 14 | 100% | 50.0% | 50.0% |
| test_unseen_layout (`two_col_ddmmyy_cr`) | 20 | 90.0% | 100% | 100% |

And per layout:

| Layout | Documents | Genuine auto-clear | Material |
|---|---|---|---|
| generator_v1 | 20 | 100% | 55.6% |
| signed_amount_iso | 20 | 90% | 88.9% |
| two_col_ddmmyy_cr | 20 | 90% | 100% |
| two_col_dmon | 20 | 100% | 90% |
| two_col_dual_date | 20 | 80% | 80% |
| two_col_slash | 20 | 100% | 50% |

Do not read a trend into these. The nodes are rule-based and were not trained on this data, so there is no "train versus test" gap to find; the split only keeps the data ready for a later model. Each split holds only a few tampered documents of each type, so the percentages jump around (the build log warns that several tamper types are missing from the validation, test and unseen-layout splits). The unseen layout scoring 100% "material" is on only 8 tampered documents (altered_date, broken_balance, 2 cross_month_copy, dormant_reactivation, font_mismatch, 2 sudden_large_deposit); weaker types such as edited_amount, fabricated_receipts or rapid_pass_through did not land in it, so it is not evidence that the unseen layout is easier or harder to screen.

## 8. Honest limits

- **Synthetic only.** The documents come from our own simulator, in our own six layouts. They are not real bank statements and not real fraud.
- **Small samples.** 3 or 4 documents per tamper type, 20 per layout, a single seed per corpus. One document is 25 to 33 points. Treat results as indicative.
- **Our own tamper designs.** Each tamper variant is our idea of how it looks. A real forger may do it differently. A rule that finds "our" structuring may miss a real one, and the reverse.
- **Possible circularity.** The generator and the nodes were written by the same team; a rule may look good because the generator makes things the way the rule expects. The held-out layout and the extra seed help a little but do not remove that.
- **Document level.** The score is "was the document flagged", not "was the right row pointed out". Row-level accuracy was not measured.
- **Findings are partly shared noise.** The DUP note appears on most documents, so "any finding" is not a clean signal. The report counts only MEDIUM/HIGH ("material") for that reason.
- **False-alarm cause not fully traced.** The 4 TURNOVER_COST_MISMATCH false alarms on genuine documents were not investigated down to the rule. The "TT substring" false-alarm risk was not seen in the figures, but the report could not show it either way.
- **Overlay drop risk not yet seen.** Node 1 dropped 0 overlay rows, so that blind spot is unobserved, not ruled out.
- **Bad numbers are reported as found.** Nothing was tuned to improve a number. No node or generator code was changed for this task.

## 9. What to do next (suggestions only)

1. Look into the 4 genuine documents that raised FL_TURNOVER_COST_MISMATCH (list in section 4) and decide whether the rule or the simulator's cost mix is at fault.
2. Look at why FL_BALANCE_BREAK stays silent on the overlay `broken_balance` document and why VF_MIXED_DATE_FORMATS never fires. Decide whether those are rule gaps or unrealistic corpus variants.
3. Decide whether "rapid pass-through" deserves a rule, or whether the corpus variant is too mild.
4. Fix the `sudden_large_deposit` "expected code" label or the deposit size so it matches the rule threshold.
5. Split the Node 1 Balance and Header columns into genuine and tampered so intentional tampering no longer looks like a reading dip.
6. Build more documents (for example 30 per type) with several seeds, to find out which numbers above are stable.
7. Optionally record the build time in `build.log` for next time.
8. Stage B, retraining Node 6a and 6b, belongs in its own separate spec ("ml-retraining"). This corpus is a starting point for it; the circularity and small-sample limits above would need to be handled there.

## 10. Reproduction commands

Run from `/Users/weiyang01/AI-in-finance-`, with the project venv (`.venv/bin/python`).

Smoke corpus (seed 20260601, 24 documents):

```
python -m fraud_detect.corpus.build --out data/corpus --per-layout-genuine 2 --per-layout-tampered 2 --seed 20260601 --hold-out-layout two_col_ddmmyy_cr --tamper-mix edited_amount=1,altered_date=1,broken_balance=1,font_mismatch=1,metadata_inconsistency=1,genuine_resaved=1,fabricated_receipts=1,cross_month_copy=1,turnover_inflation=1,structuring=1,sudden_large_deposit=1,shell_counterparty=1,rapid_pass_through=1,dormant_reactivation=1,period_gap=1,uen_mismatch=1
python -m fraud_detect.corpus.verify --corpus data/corpus
```

Large corpus (seed 20260602, 120 documents):

```
python -m fraud_detect.corpus.build --out data/corpus_large --force --per-layout-genuine 10 --per-layout-tampered 10 --seed 20260602 --hold-out-layout two_col_ddmmyy_cr --tamper-mix edited_amount=1,altered_date=1,broken_balance=1,font_mismatch=1,metadata_inconsistency=1,genuine_resaved=1,fabricated_receipts=1,cross_month_copy=1,turnover_inflation=1,structuring=1,sudden_large_deposit=1,shell_counterparty=1,rapid_pass_through=1,dormant_reactivation=1,period_gap=1,uen_mismatch=1
python -m fraud_detect.corpus.verify --corpus data/corpus_large
```

Same seed and settings rebuild the same labels and text layers. The reports used above are `report.md`, `report.json` and `verify.log` in each corpus folder; the folders themselves are git-ignored. To get the same tables again, add `--markdown <file>` and `--json <file>` to the `verify` command if you want files written (the command writes nothing otherwise).

## 11. Note on the UEN check

The runs reported above were made before the build-time UEN check existed, so they used the default behaviour: each invented UEN was compared only with the in-repo registry and with the other UENs of the corpus, and the real ACRA registry was **not** checked. A rebuild with the same seed gives the same corpus unless a UEN clashes with a snapshot or with live data, in which case only that company's UEN changes. The check stores no real company data: only the check source, the snapshot file name and hash, and counts (see `FRAUD_DETECT_README.md`, "UEN check"). The numbers in this file are not affected by the check.
