# Implementation Plan: Node 3 Fabrication Rules

## Overview

Add three fabrication rules and a safety wrapper to Node 3, plus a small separable calibration of `FL_STRUCTURING`, in small test-first steps. Each rule gets its failing tests first, then the code, then an optional property test. New code lives in `fraud_detect/nodes/fabrication.py` (rules), a small change in `fraud_detect/nodes/financial_logic.py` (wiring and structuring), and `fraud_detect/nodes/fabrication_eval.py` (read-only report script). New tests live in `tests/fabrication/` (in-code fixtures, always run) and in two flat files for the labelled-file acceptance and the structuring calibration.

Language: Python (matches the repo). Test command, run from `/Users/weiyang01/AI-in-finance-`:

```
env -u AUDIT_SIGNING_KEY_PATH -u FRAUD_ENABLE_TXN_MODEL /Users/weiyang01/AI-in-finance-/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore
```

Starting point: 571 passed and 1 xfail. Tests that need `~/Downloads/statement_E.pdf`, `statement_F.pdf`, `ANSWER_KEY_6M_v2.json` or `statement_forgery_dataset/` skip when those are missing. None of those files is ever committed.

Out of scope: Node 6a/6b retraining, generator extension to 6 to 12 month fraud patterns, any Node 1 or Node 2 change, UIs, Node 4 weights, OCR. Do not edit `fraud_detect/store.py` or the folder `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/`. No new dependencies.

Honest limits to keep in mind while working: two labelled files only; the Rule C jump ratio default (1.2) was informed by both files, so it separating E from F is a sanity check, not proof. If a labelled-file check fails with the default settings, report it and ask. Do not tune a threshold to make it pass.

## Tasks

- [x] 1. Foundations and safety net
  - [x] 1.1 Create test scaffolding and in-code fixtures
    - Create `tests/fabrication/__init__.py`, `conftest.py` and `fixtures.py`
    - `conftest.py` defines local path constants for `~/Downloads/statement_E.pdf`, `statement_F.pdf`, `ANSWER_KEY_6M_v2.json` and `statement_forgery_dataset`, with `skipif` markers
    - `fixtures.py` builds `Transaction` lists and a minimal `IngestionResult` with fictional values: a six-month business file with flat payroll, CPF and rent and one quarterly GST payment, credits that carry `INV` references in date order, helpers to insert a copied credit (same customer and reference, other date and amount), to move references out of order, and to scale one month's credits
    - _Requirements: 14.6, 14.7, 19.4_
  - [x] 1.2 Record the baseline
    - Run the full suite and record the counts (expected 571 passed, 1 xfail) in the task notes
    - Add a characterisation test in `tests/fabrication/test_baseline.py` that Node 3 on the built-in fixture business file returns no fabrication codes yet and that the generator clean sample returns no Node 3 findings
    - This test must keep passing after every later task
    - _Requirements: 15.1, 15.2_

- [x] 2. Settings and reference reading
  - [x] 2.1 Write failing tests for the config and the reference reader
    - In `tests/fabrication/test_config_reading.py`: `DEFAULT_CONFIG` values (jump ratio 1.2, escalate count 3, weights 10, 22, 5, 8), `ValueError` table (ratio of 1 or less, stability percentage outside 0 up to but not including 1, count below 1, negative weight), config is immutable
    - Reader table: `INV0042` equals `inv-042` equals `INV #042`; references need at least three digits, so `INV12` (two digits) has no reference; `Customer Payment INV` has no reference; rail prefixes removed (`FAST FROM CINDERPATH HOLDINGS INV3277` gives `CINDERPATH HOLDINGS` and `INV3277`); descriptions cut at 500 characters; `None`, empty and zero or negative amounts are skipped without error
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [x] 2.2 Implement `FabricationConfig`, `DEFAULT_CONFIG` and the reader
    - Create `fraud_detect/nodes/fabrication.py` with the frozen dataclass and `__post_init__` validation from the design, `CreditRef`, `extract_invoice_ref`, `normalise_counterparty`, `extract_credit_refs`
    - Patterns use group 1 as prefix and group 2 as number; the key is the upper-cased prefix plus the integer value of the number
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [x]* 2.3 Write property tests for the reader and the config
    - **Property 1: Invoice reference normalisation**
    - **Property 12: Invalid configuration is rejected**
    - **Validates: Requirements 1.3, 2.1, 2.2, 2.3**

- [x] 3. Rule B: invoice sequence break
  - [x] 3.1 Write failing tests for `find_order_breaks` and `check_invoice_sequence`
    - In `tests/fabrication/test_sequence.py`: in-order list is silent; two moved rows are reported and named; one moved row is silent; same-day rows in any order are silent; fewer than 6 dated references is silent; a row with no date is left out; one finding per file with LOW severity, weight 5, `field` such as `rows 17, 23`, `page` of the latest named row; running twice gives identical output
    - _Requirements: 5.1, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 3.2 Implement Rule B
    - In `fabrication.py` add `find_order_breaks` (sort by date then reference number, longest non-decreasing run, deterministic tie-break, returns the set of row numbers) and `check_invoice_sequence`
    - _Requirements: 5.1, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x]* 3.3 Write property test for Rule B
    - **Property 6: Sequence break detection**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

- [x] 4. Rule A: duplicate invoice reference
  - [x] 4.1 Write failing tests for `check_duplicate_invoice_refs`
    - In `tests/fabrication/test_dup.py`: unique references are silent; one repeated reference gives one MEDIUM finding with weight 10 and the sentence "may be an instalment payment; check the invoice"; four payments, or a span over 45 days, leave that sentence out; identical amounts say "identical amount, possible duplicate posting"; three repeated references give exactly one HIGH finding with weight 22 listing every group; same reference under two customers is silent; debits are ignored; evidence lists date, amount, page and row for every occurrence; `field` is `rows a, b` ascending and `page` is the later row's page; the evidence says it cannot tell which row is the copy; `evidence_max_groups` cuts the list and ends with "and N more"; a `likely_copies` row is named as "breaks the invoice number order, so it is the likelier copy" only when exactly one row of the group is in the set
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 7.1, 7.2, 7.3_
  - [x] 4.2 Implement Rule A
    - In `fabrication.py` add `check_duplicate_invoice_refs(txns, cfg, likely_copies)` using `extract_credit_refs`; group by (counterparty, reference); build the evidence as specified; apply the escalation rule
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 7.1, 7.2, 7.3_
  - [x]* 4.3 Write property tests for Rule A
    - **Property 2: Duplicate reference detection**
    - **Property 3: Unique references are quiet**
    - **Property 4: Severity follows the number of repeated references**
    - **Property 5: Evidence names every occurrence**
    - **Property 7: Likelier copy is named only when unambiguous**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 5.2, 5.3, 5.4, 7.1, 7.2, 9.4**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Rule C: turnover against fixed costs
  - [x] 6.1 Write failing tests for `check_turnover_vs_costs`
    - In `tests/fabrication/test_turnover.py`: flat fixed costs with two jump months is reported (MEDIUM, weight 8, one finding); one jump month is silent; fixed costs rising with credits is silent; a month whose fixed costs rise more than 10% above the median is not a jump month (the quarterly GST month, when `GST` is added to the keywords); partial first and last months are skipped; fewer than 4 full months is silent; fewer than two thirds of months stable is silent; excluded credits (`LOAN`, `CAPITAL`, `TRANSFER`, `INTEREST`) are not counted as turnover; whole-word keyword matching (`CURRENT` does not match `RENT`); a whole-file credit-to-debit ratio alone never triggers; evidence lists turnover, median, ratio, fixed costs and their median for each jump month
    - _Requirements: 5.1, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x] 6.2 Implement Rule C
    - In `fabrication.py` add `check_turnover_vs_costs(ing, cfg)` with full-month selection (uses the declared period when present), keyword matching with `re.escape` and word boundaries, medians with `statistics.median`
    - _Requirements: 5.1, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x]* 6.3 Write property test for Rule C
    - **Property 8: Turnover rule needs both conditions**
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6**

- [x] 7. Wire into Node 3 and isolate failures
  - [x] 7.1 Write failing tests for `analyze_fabrication` and the Node 3 wiring
    - In `tests/fabrication/test_wiring.py`: `financial_logic.analyze(ing)` on the fixture file with an inserted copy returns `FL_DUP_INVOICE_REF` with agent `Node3:FinancialLogic`; `fabrication_config=` override is honoured (for example `dup_escalate_min_refs=1` turns a single repeat HIGH); existing callers with one argument still work; rule B's break rows reach rule A as `likely_copies`; monkeypatching one rule to raise gives one LOW `DQ_RULE_SKIPPED` (weight 2, `field` = that rule's code) and the other rules' findings stay; no exception reaches the caller; a file with no references gives no rule A or B finding; calling twice gives identical output and the input is unchanged
    - _Requirements: 1.4, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2_
  - [x] 7.2 Implement `analyze_fabrication` and wire it in
    - In `fabrication.py` add `analyze_fabrication(ing, cfg=DEFAULT_CONFIG)`: extract once, run B, then A (with B's rows), then C, each inside its own `try/except Exception`, log the exception type with `logging`, append `DQ_RULE_SKIPPED` on error
    - In `financial_logic.py` add the optional `fabrication_config` argument and append the fabrication findings after the existing rules; change nothing else in that file in this task
    - _Requirements: 1.4, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5_
  - [x]* 7.3 Write property tests for isolation and determinism
    - **Property 9: Rules never crash and skips are visible**
    - **Property 10: Determinism and no mutation**
    - **Validates: Requirements 2.5, 6.6, 9.1, 9.2, 9.3, 9.5**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Acceptance on the labelled files
  - [x] 9.1 Write the labelled-file acceptance tests
    - In `tests/test_fabrication_ef.py` (local path constants, skip when any file is missing): ingest E and F, run `financial_logic.analyze` with `DEFAULT_CONFIG`
    - E: `FL_DUP_INVOICE_REF` is reported and its evidence names all 12 `inserted_rows` (matched by date and amount; the answer key amounts are strings like `S$8,318.10`); print how many fakes the likelier-copy hint names and how many named rows are genuine; `FL_TURNOVER_COST_MISMATCH` is reported; no `DQ_RULE_SKIPPED`
    - F: none of `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH`, `DQ_RULE_SKIPPED`
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_
  - [x] 9.2 Run the acceptance tests and record the figures
    - Run `tests/test_fabrication_ef.py -s`; write the measured figures (rows named 12 of 12, fakes named by the hint, genuine rows named, Rule C months and ratios for E and F) to `.kiro/specs/node3-fabrication-rules/evaluation-results.md` under the heading "Labelled-file acceptance"
    - If the Rule C check on E or any check on F fails with the default settings, stop, record the monthly figures, and ask the user. Do not change a threshold to make the test pass
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 17.3_

- [x] 10. Calibrate the structuring rule (separable group; can be dropped without affecting tasks 1 to 9)
  - [x] 10.1 Write failing tests for the calibrated rule
    - In `tests/test_structuring_calibration.py`: four deposits 9,797 to 9,800 on consecutive days fire; three on one day fire; three spread over 40 days are silent; the `statement_F` dates (28 Jul, 1 Aug, 28 Aug with 9,721.98, 9,926.42, 9,817.90) are silent; an undated near-threshold row counts as inside every span; evidence names the span and the rows; `test_detection.py::test_each_planted_issue_is_detected[STRUCTURING]` still passes
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - [x] 10.2 Implement the cluster condition
    - In `financial_logic.py` add `_STRUCTURING_CLUSTER_DAYS = 14` and change `_structuring` to fire only when three or more near-threshold credits fit inside that span; keep severity HIGH and weight 20
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  - [x] 10.3 Remove the strict xfail marker
    - In `tests/test_ingest_pipeline_outcomes.py` remove the `@pytest.mark.xfail(strict=True, ...)` on `test_statement_f_outcome` once F no longer produces a HIGH finding; confirm the test passes or skips cleanly
    - _Requirements: 11.7, 15.2_
  - [x]* 10.4 Write property test for the cluster condition
    - **Property 11: Structuring needs a cluster in time**
    - **Validates: Requirements 11.1, 11.2, 11.3**

- [x] 11. Regression sweep
  - [x] 11.1 Add the regression tests
    - In `tests/test_fabrication_regression.py`: for generator seeds 1 to 20, `generate_clean` and `generate_tampered` (all issue kinds) produce no `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH` or `DQ_RULE_SKIPPED`; the clean sample stays `AUTO_CLEAR` with a risk score of 5 or less
    - Run the whole suite and confirm all existing tests still pass
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 12. Evaluation script and report-only investigations
  - [x] 12.1 Write tests for the evaluation script
    - In `tests/fabrication/test_eval.py`: the pure counting and table-printing helpers on in-memory results; a missing dataset folder prints "skipped" and returns 0; the script writes nothing outside its printed output (no `data/cases.json`, no audit chain); it screens through `ingest` plus `financial_logic.analyze` and `risk_synthesis.synthesize` only, never through a `CaseStore`
    - _Requirements: 16.1, 16.2, 16.3, 16.4_
  - [x] 12.2 Implement `fraud_detect/nodes/fabrication_eval.py`
    - Command: `python -m fraud_detect.nodes.fabrication_eval [--sample N] [--dataset DIR] [--statements DIR] [--generator-count N]`
    - Prints, per rule, how many of the genuine dataset documents and generator statements fired; the share of dataset documents that carry an invoice reference; the number of dataset documents that trigger `VF_CURRENCY_MIX`; metadata of E and F; the Producer, ModDate and CreationDate counts for the 64 `genuine_resaved = 1` documents; and risk score, band, routing and finding codes for E and F with the line "E above 25 and above F: yes/no" (a reported target, not an assertion)
    - Read-only; follows the layout of `fraud_detect/nodes/ingest_parse/eval_layouts.py`
    - _Requirements: 12.1, 12.4, 13.1, 13.2, 16.1, 16.2, 16.3, 16.4, 16.5, 17.1, 17.2_
  - [x] 12.3 Run the script and record the false-alarm and risk-score tables
    - Run it over the full genuine set and 30 generator statements; write the tables and the before-and-after scores of E and F (today 26 and 40; expected about 55 and 26 or 40) to `.kiro/specs/node3-fabrication-rules/evaluation-results.md` under "False-alarm report" and "Risk-score effect", with the limits restated
    - Report numbers as found, including a non-zero false-alarm count if one appears
    - _Requirements: 16.1, 16.5, 17.1, 17.2, 17.3_
  - [x] 12.4 Write the investigation notes
    - Read `visual_forensics.py` and the PDF metadata of E and F read-only (pypdf) and write `.kiro/specs/node3-fabrication-rules/investigation.md` with I-1 (cause of the missing metadata finding; the recognised tool list; whether any ModDate check exists; the false-alarm risk from the 64 re-saved documents) and I-2 (the text matches behind `VF_CURRENCY_MIX`, the count over the genuine dataset, and a keep, narrow or drop recommendation for a later Node 2 spec)
    - Change no Node 2 behaviour
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4_

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Documentation
  - [x] 14.1 Update `FRAUD_DETECT_README.md`
    - Add a section "Node 3 fabrication rules" listing `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH` and `DQ_RULE_SKIPPED` with severity and weight
    - Explain each `FabricationConfig` field and give one worked tuning example, how to run `fabrication_eval`, and the structuring calibration if it was kept
    - State the honest limits: two labelled files only; the Rule C jump ratio was informed by both files; a brand-new invoice number evades Rule A; genuine instalments can look like duplicates; about half of the rows Rule A names are genuine; no finding means no signal, not verified; Node 4 weights were not changed; multi-row findings are not highlighted on a table row in the UI
    - Add the results from `evaluation-results.md` as a table labelled indicative
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm `git status` shows no external PDF or answer key, no change to `fraud_detect/store.py`, and no change under `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/`.
  - _Requirements: 19.1, 19.2, 19.3_

## Notes

- Tasks marked with `*` are optional property tests and can be skipped for a faster first version. Unit tests are not optional.
- Each rule follows test first: the x.1 task writes tests that fail until the x.2 task is done.
- Task group 10 (structuring) is independent of the three new rules. To drop it, skip tasks 10.1 to 10.4 and leave the `xfail` marker in place; Requirements 11 then stay open.
- Property tests use `hypothesis` with at least 100 examples and carry the tag `Feature: node3-fabrication-rules, Property N: ...`.
- External files are read-only, never committed, and every test that uses them skips when they are missing.
- Task 9.2 is a decision point: if the Rule C check on E fails with the default settings, the thresholds are discussed with the user, not tuned silently.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["2.3", "3.1"] },
    { "id": 4, "tasks": ["3.2"] },
    { "id": 5, "tasks": ["3.3", "4.1"] },
    { "id": 6, "tasks": ["4.2"] },
    { "id": 7, "tasks": ["4.3", "6.1"] },
    { "id": 8, "tasks": ["6.2"] },
    { "id": 9, "tasks": ["6.3", "7.1"] },
    { "id": 10, "tasks": ["7.2"] },
    { "id": 11, "tasks": ["7.3", "9.1", "10.1", "12.1"] },
    { "id": 12, "tasks": ["10.2", "12.2"] },
    { "id": 13, "tasks": ["9.2", "10.3", "10.4", "11.1", "12.4"] },
    { "id": 14, "tasks": ["12.3"] },
    { "id": 15, "tasks": ["14.1"] }
  ]
}
```
