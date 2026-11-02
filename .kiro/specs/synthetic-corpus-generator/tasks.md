# Implementation Plan: Synthetic Corpus Generator

## Overview

Build the corpus generator in small, test-first steps. Each group first writes tests that fail, then writes the code that makes them pass, then (optionally) adds a property test. New code lives in a new package `fraud_detect/corpus/`. New tests live in `tests/corpus/`. Nothing in Nodes 1 to 4, `fraud_detect/synthetic.py`, `fraud_detect/registry.py`, `fraud_detect/reference_data.py` or any existing test is changed. The corpus itself is written to the git-ignored folder `data/corpus/` and is never committed.

Language: Python (matches the repo). Test command, run from `/Users/weiyang01/AI-in-finance-`:

```
env -u AUDIT_SIGNING_KEY_PATH -u FRAUD_ENABLE_TXN_MODEL /Users/weiyang01/AI-in-finance-/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore
```

Starting point: 1155 passed and 1 skipped, on branch `feature/Machine-learning-optimization`. Run only `tests/corpus` while working and the whole suite at each checkpoint.

Out of scope (do not do): retraining Node 6a or 6b (that is the separate Stage B spec "ml-retraining"); any change to Nodes 1 to 4, the registry or the reference data; user interfaces; OCR or scanned documents; real-bank layouts. Findings such as "Node 1 drops overlay rows" are reported in the measurement, never fixed here.

Clean-room rule for every task: do not copy or import anything from `~/Downloads/statement_forgery_dataset` or its script `generate_documents.py` (unknown licence). Write the renderers from the layout facts in `tests/ingest_parse/fixtures.py` and `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/design.md`. Reading a few sample PDFs read-only with pdfplumber to double-check geometry is allowed; copy nothing. Never train on, commit, or reference `statement_E` / `statement_F` or the answer key in package code. Use only reportlab's built-in fonts. Invent all names.

Honest limits to keep in mind: results are demonstrations on synthetic data; do not tune the generator or any node to make a number look better; report what is measured, including bad numbers.

## Tasks

- [x] 1. Foundations and safety net
  - [x] 1.1 Create the package skeleton, shared data classes and test scaffolding
    - Create `fraud_detect/corpus/__init__.py` and `fraud_detect/corpus/schema.py` with `GENERATOR_VERSION = "1.0.0"`, `LAYOUTS`, the tamper-type catalog names, the label and row column lists, and plain frozen dataclasses (`CompanyProfile`, `Party`, `LedgerRow`, `Ledger`, `EditOp`, `FileOp`, `TamperLabel`, `TamperResult`, `CellBox`, `RowMap`, `PrintedRow`, `SectionSummary`, `RenderResult`, `StyleKnobs`, `SimulatorKnobs`, `BuildConfig`, `DocumentPlan`); no logic yet
    - Create `tests/corpus/__init__.py`, `tests/corpus/conftest.py` and `tests/corpus/test_hygiene.py`
    - Hygiene tests: `data/corpus/x.pdf` is ignored by git (`git check-ignore`); the package source contains none of the strings `Downloads`, `statement_forgery_dataset`, `statement_E.pdf`, `statement_F.pdf`, `ANSWER_KEY`, `generate_documents`; no PDF, `labels.csv` or corpus file is tracked by git; the package imports nothing outside reportlab, pypdf, pdfplumber, hypothesis (tests only) and the standard library; tests do not need `~/Downloads`
    - Run the whole suite once and write the counts (expected 1155 passed, 1 skipped) in the task notes
    - _Requirements: 12.6, 17.1, 17.4, 17.7, 18.3, 18.4, 18.6_
  - [x] 1.2 Write failing tests for stable seeding
    - In `tests/corpus/test_seeding.py`: the same parts give the same seed on every call; different parts give different seeds; the result does not change when the global `random` state is changed; a known fixed value is pinned so a change is noticed
    - _Requirements: 10.1, 10.6_
  - [x] 1.3 Implement `fraud_detect/corpus/seeding.py`
    - `stable_seed(master_seed, *parts) -> int` from SHA-256 of the master seed and parts; `rng(master_seed, *parts) -> random.Random`; never use `hash()` or global `random`
    - _Requirements: 10.1, 10.6_
  - [x]* 1.4 Write property test for stable seeding
    - **Property 8: Same seed, same corpus (seeding part)**
    - **Validates: Requirements 10.1, 10.6**

- [x] 2. Company and ledger simulator
  - [x] 2.1 Write failing tests for the simulator
    - In `tests/corpus/test_simulator.py`: month count and consecutive months for n = 6 and 12; company name ends with "PTE. LTD.", 5 to 25 customers, UEN-style id; receipt descriptions match a rail word + customer + `INV` with 3 or more digits and `fabrication.extract_credit_refs` reads them; invoice numbers never decrease with date at disorder 0 and the disorder knob can break that; payroll, CPF and rent within 10% of their medians; a GST row in each quarter whose last month is inside the statement; supplier, utility, card, service charge and interest rows present; every amount has two decimals; replayed balance never below zero; each month opens at the previous close; weekday dates for receipts, card, supplier and GIRO; instalment rate 0 gives no repeated reference, a positive rate gives some, each recorded as a hard case; a row cap is respected, card payments are dropped first and the trim is recorded
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12_
  - [x] 2.2 Implement `fraud_detect/corpus/company.py`
    - `make_company(seed, company_id)` with invented word lists for names, customers, suppliers, utilities and a landlord; fictional UEN-style id, address, account number, archetype, payroll base
    - _Requirements: 1.2_
  - [x] 2.3 Implement `fraud_detect/corpus/ledger.py`
    - `simulate_ledger(company, start, n_months, seed, knobs, max_rows_per_month)` and `replay(ledger)` following the design (growth, seasonality, receipts numbered in receipt-date order, instalments, fixed costs, quarterly GST, other rows, volume cap, opening balance chosen from a dry run); stable `row_id` values
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12_
  - [x]* 2.4 Write property tests for the simulator
    - **Property 2: Receipts carry readable, rising invoice references**
    - **Property 3: Fixed costs are flat and the balance behaves**
    - **Property 4: The ledger has the requested shape**
    - **Property 5: Instalments are recorded as genuine hard cases**
    - **Property 6: Row caps are respected and the trim is recorded**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.10, 1.11, 1.12, 12.2**

- [x] 3. Renderer foundation
  - [x] 3.1 Write shared test helpers and failing tests for the renderer base
    - `tests/corpus/helpers.py`: `render_and_ingest(layout, n_months, seed)` that simulates a ledger, renders it, writes a temporary PDF and runs `ingest`; `compare_rows(ledger_rows, node1_transactions)` that checks date, signed amount, balance and order; `words_of(pdf_bytes)` using pdfplumber
    - `tests/corpus/test_renderer_base.py`: the `RowMap` has an entry for every ledger row and the text at each mapped position equals the printed cell; every page carries the specimen footer; PDF Title and Subject say synthetic; fonts used are only reportlab built-in fonts; `ledger.printed_holder` is used for the holder when set; the same seed gives equal extracted words and metadata
    - _Requirements: 2.11, 2.12, 2.14, 10.3, 12.8, 18.5_
  - [x] 3.2 Implement the renderer base and the finishing step
    - `fraud_detect/corpus/layouts/base.py` (`LayoutRenderer` protocol, row map helpers, page capacity helpers) and `fraud_detect/corpus/finish.py` (reportlab fixed-output mode, metadata with fixed dates written with pypdf, Title and Subject that say synthetic, footer wording in the generator's own words, fictional bank-name list, style knobs)
    - _Requirements: 2.11, 2.12, 2.14, 10.3, 12.8, 18.5_

- [x] 4. Layout `generator_v1`
  - [x] 4.1 Write failing tests for the `generator_v1` renderer
    - In `tests/corpus/test_layout_generator_v1.py`: Node 1 round trip at 6 and 12 months (every row with the generated date, amount and balance; replayed balance equals printed closing; chosen profile `generator_v1`; holder, UEN and period equal; no MEDIUM/HIGH data-quality finding); the period line prints once per monthly section; opening-balance and closing-balance lines equal the replay; the footer does not disturb row reading; a long month continues on a second page
    - _Requirements: 2.2, 2.3, 2.5, 2.10, 2.11, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 4.2 Implement `fraud_detect/corpus/layouts/generator_v1.py`
    - Colon-label skeleton using a fictional bank title (not a real brand), one signed amount column, `dd/mm/yyyy`; geometry from the fixtures' `generator_v1_fixture`
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.10, 2.11, 2.12_

- [x] 5. Layout `two_col_slash` and the shared table engine
  - [x] 5.1 Write failing tests for `two_col_slash`
    - In `tests/corpus/test_layout_two_col_slash.py`: Node 1 round trip at 6 and 12 months; a month with more rows than fit continues on a second page that repeats the table header and has no period line and no brought-forward row; summary box equals replay; right edges of amount and balance cells within 2 points of the fixture columns (Withdrawals 400, Deposits 475, Balance 555); `dd/mm/yyyy` dates
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.10, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 5.2 Implement the table engine `layouts/dataset_family.py` and the `two_col_slash` spec
    - The engine is driven by a spec (date style, column positions, header words, amount style, balance suffix, second date column, page marker, money format). Implement all spec options named in the design so later layouts only add specs; register `two_col_slash` in `layouts/specs.py`; spill rule from the design (about 43 rows on the first page and 54 on a continuation page, from the fixtures' row step)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.10, 2.11, 2.12, 2.13_

- [x] 6. Layout `two_col_dmon`
  - [x] 6.1 Write failing tests for `two_col_dmon`
    - In `tests/corpus/test_layout_two_col_dmon.py`: Node 1 round trip at 6 and 12 months (including a statement that crosses a year end); every monthly page prints "Page 1 of 1"; rows print `d Mon` without a year and amounts with `S$`; header lines include `Account No.`, `Company UEN` and `Statement Period dd Mon yyyy to dd Mon yyyy`; no month spills (the simulator cap keeps each month on one page); right edges match the fixture columns (Money out 392, Money in 472, Balance 557)
    - _Requirements: 2.2, 2.3, 2.4, 2.6, 2.10, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 6.2 Add the `two_col_dmon` spec
    - Add the spec to `layouts/specs.py` and set its row cap; adjust the engine only if a test shows a gap
    - _Requirements: 2.1, 2.6, 1.12_

- [x] 7. Layout `two_col_dual_date`
  - [x] 7.1 Write failing tests for `two_col_dual_date`
    - In `tests/corpus/test_layout_two_col_dual_date.py`: Node 1 round trip at 6 and 12 months; every row prints a transaction date and a value date (`dd Mon yyyy`); continuation behaviour as for `two_col_slash`; right edges match the fixture columns (Debit 420, Credit 490, Balance 559)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.7, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 7.2 Add the `two_col_dual_date` spec
    - Add the spec to `layouts/specs.py`; value date is the transaction date plus 0 to 2 days from the document seed
    - _Requirements: 2.1, 2.7_

- [x] 8. Layout `signed_amount_iso`
  - [x] 8.1 Write failing tests for `signed_amount_iso`
    - In `tests/corpus/test_layout_signed_amount_iso.py`: Node 1 round trip at 6 and 12 months; one signed amount column with `+SGD` for credits and `-SGD` for debits; `yyyy-mm-dd` dates; right edges match the fixture columns (Amount 450, Balance 555)
    - _Requirements: 2.2, 2.3, 2.4, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 8.2 Add the `signed_amount_iso` spec
    - Add the spec to `layouts/specs.py`
    - _Requirements: 2.1, 2.8_

- [x] 9. Layout `two_col_ddmmyy_cr`
  - [x] 9.1 Write failing tests for `two_col_ddmmyy_cr`
    - In `tests/corpus/test_layout_two_col_ddmmyy_cr.py`: Node 1 round trip at 6 and 12 months; `dd-mm-yy` dates; `CR` after every balance (brought-forward and closing rows included) and not in the summary box; right edges match the fixture columns (Debit 395, Credit 470, balance 546)
    - _Requirements: 2.2, 2.3, 2.4, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 9.2 Add the `two_col_ddmmyy_cr` spec
    - Add the spec to `layouts/specs.py`
    - _Requirements: 2.1, 2.9_

- [x] 10. Renderer registry and the all-layout round trip
  - [x] 10.1 Write failing tests for the registry
    - In `tests/corpus/test_layout_registry.py`: `RENDERERS` has exactly the six names in `LAYOUTS`; each renderer's `name` equals its key; the layout names equal Node 1's `REGISTRY` profile names; a spilled month on a layout that does not spill in real samples is reported as `spill_not_in_real_samples`
    - _Requirements: 2.1, 2.13_
  - [x] 10.2 Implement `fraud_detect/corpus/layouts/__init__.py`
    - `RENDERERS` dictionary and `get_renderer(name)`; unknown names raise `ValueError` with the allowed names
    - _Requirements: 2.1_
  - [x]* 10.3 Write property tests for the round trip and the geometry
    - **Property 1: Node 1 reads every genuine generated statement completely**
    - **Property 7: Printed cells sit where the layout says, and the footer is there**
    - Use about 10 examples per layout (each example renders a PDF); months drawn from 6 to 12
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.10, 2.11, 2.12, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm the whole suite still shows 1155 passed (plus the new tests) and 1 skipped, and that Node 1 reads 100% of genuine rows in all six layouts at 6 and 12 months.

- [x] 12. Manifests and label files for genuine documents
  - [x] 12.1 Write failing tests for manifests and label files
    - In `tests/corpus/test_manifest.py`: a genuine document's Manifest lists every printed row with `row_id`, 1-based `row_index`, page, month, date, description, amount, balance and `is_fraud_row` false; `row_index` equals the order in which Node 1 reads the rows; no Tamper_Type and `is_tampered` false; `text_layer_sha256` equals the hash recomputed from the PDF; `labels.csv` header equals the required columns exactly; `rows.csv` header equals the required columns exactly; the Manifest carries `synthetic: true` and the Generator_Version; `genuine_hard_cases` lists instalments
    - _Requirements: 4.3, 4.4, 4.5, 9.1, 9.2, 9.3, 10.4_
  - [x] 12.2 Implement `fraud_detect/corpus/labels.py`
    - Build a Manifest from a ledger, a `RenderResult` and a label; write the manifest JSON (sorted keys, fixed formatting), the `labels.csv` line and the `rows.csv` lines; compute the text-layer hash from words and positions
    - _Requirements: 4.3, 4.4, 4.5, 9.1, 9.2, 9.3, 10.2, 10.4_
  - [x]* 12.3 Write property tests for genuine labels and synthetic marking
    - **Property 9: Labels tell the truth (genuine part)**
    - **Property 21: Documents are marked synthetic and use only built-in fonts**
    - **Validates: Requirements 4.3, 4.4, 4.5, 9.6, 9.7, 12.8, 18.5**

- [x] 13. Document tampers by re-rendering
  - [x] 13.1 Write failing tests for ledger-level document tampers
    - In `tests/corpus/test_tamper_document.py`: `edited_amount` (rerender) changes exactly one amount and records original and new value, with the recompute choice recorded as `arithmetic_consistent`; `altered_date` changes exactly one date and sets `date_inconsistent` only when order or month is broken; `broken_balance` changes exactly one printed balance and sets `arithmetic_consistent` false; applying the recorded edit list to the genuine ledger reproduces the edited ledger; the Fraud_Rows are exactly the edited rows; `arithmetic_consistent` is true if and only if printed balances equal the replay of the printed amounts
    - _Requirements: 4.2, 5.1, 5.2, 5.3, 5.8, 9.4_
  - [x] 13.2 Implement `tamper/ledger_ops.py` and `tamper/catalog.py`
    - Edit operations (`set_amount`, `set_date`, `set_balance`, `insert_row`, `delete_row`, `delete_month`), edit-log replay, balance recomputation including following months' openings, the catalog with the descriptive "should be seen by" lists; register the three document tampers
    - _Requirements: 4.2, 5.1, 5.2, 5.3, 5.8, 9.4, 9.5_
  - [x]* 13.3 Write property tests for single-edit tampers and the arithmetic flag
    - **Property 10: The arithmetic flag is honest**
    - **Property 11: Single-edit tampers change exactly one thing**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.8**

- [x] 14. Tampers on the finished PDF
  - [x] 14.1 Write failing tests for overlay, font, metadata and re-save
    - In `tests/corpus/test_tamper_pdf.py`: overlay leaves both the old and the new value extractable at the mapped position and nothing else on the page changes; `hidden_text_layer` is true; `overlay_bg_matched` follows the background box choice; the manifest says Node 1 might drop the row; rerender leaves only the new value; `font_mismatch` uses `Courier` on the changed cell only and `font_inconsistent` is true; `metadata_inconsistency` changes only metadata (same text-layer hash), Producer names an editing tool, ModDate later than CreationDate, and the "creation before period" variant sets the earlier creation date; `genuine_resaved` keeps the text-layer hash, has a print/export Producer and a later ModDate, `is_tampered` false and `genuine_resaved` true
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 14.2 Implement `tamper/pdf_ops.py` and register the file-level tampers
    - `apply_file_ops(pdf, file_ops, row_map, seed)`: overlay page with reportlab merged with pypdf at the row-map position, `set_font` with the built-in `Courier`, `set_metadata`, `resave`; register `font_mismatch`, `metadata_inconsistency`, `genuine_resaved` and the overlay variants of `edited_amount`, `altered_date` and `broken_balance` in the catalog
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x]* 14.3 Write property tests for overlay and file-level tampers
    - **Property 12: Overlay keeps both values; rerender keeps only the new one**
    - **Property 13: File-level tampers leave the pages alone**
    - Use about 10 examples per layout
    - **Validates: Requirements 5.5, 5.7, 6.2, 6.4, 6.5**

- [x] 15. Fabrication and content fraud
  - [x] 15.1 Write failing tests for content fraud
    - In `tests/corpus/test_tamper_content.py`: `fabricated_receipts` inserts 6 to 12 credits that copy customer and invoice reference of an existing receipt, with a different date and amount in the same calendar month; `cross_month_copy` copies into a different month; `turnover_inflation` adds credits with new references in at least 2 months and leaves payroll, CPF and rent rows equal; balances, summary boxes and following openings are recomputed and `arithmetic_consistent` is true; only inserted rows are Fraud_Rows; each copy names the genuine `row_id` it copies; Node 3's `check_duplicate_invoice_refs` finds the same-month copies (a sanity check, not a target)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 15.2 Implement `tamper/content.py` and register the three tampers
    - Insert rows by recorded operations, recompute balances, write the copied-row ids into the label
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x]* 15.3 Write property test for content fraud
    - **Property 14: Fabrication tampers copy real receipts the way they should**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.5, 7.6**

- [x] 16. Behavioural fraud typologies
  - [x] 16.1 Write failing tests for behavioural tampers
    - In `tests/corpus/test_tamper_behaviour.py`: `structuring` (3 to 5 credits from 9,500.00 to 9,999.99 within 14 days; slow variant over more than 30 days); `sudden_large_deposit` (above 3 times the average monthly credit; moderate variant 1.5 to 2.5 times); `shell_counterparty` (50,000.00 or more with a shell-like name); `rapid_pass_through` (2 to 4 pairs, debit 95% to 100% within 1 to 3 days); `dormant_reactivation` (first 3 to 5 months only charges and interest, later months unchanged); `period_gap` (one middle month removed and listed as an affected month, each section internally consistent); `uen_mismatch` (printed holder differs from the registered name, registered name kept in the companies record); every behavioural tamper keeps each monthly section consistent
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x] 16.2 Implement `tamper/behaviour.py` and register the seven tampers
    - Recorded operations, balance recomputation per section, variant selection from the document seed
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x]* 16.3 Write property test for behavioural tampers
    - **Property 15: Behavioural tampers have the right shape**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7**

- [x] 17. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm every tamper type in the catalog has a test, a recorded label and a "should be seen by" entry.

- [x] 18. Splits and leakage rules
  - [x] 18.1 Write failing tests for splits
    - In `tests/corpus/test_splits.py`: every document gets exactly one split from `train`, `validation`, `test`, `test_unseen_layout`; all documents of one company share a split; the split of a company depends only on master seed and company id; with a held-out layout a document is in `test_unseen_layout` if and only if its layout is the held-out one; fractions that do not sum to 1 within 0.001 or are negative are refused with a clear message; a tamper type missing from a non-empty split is listed as a gap
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8_
  - [x] 18.2 Implement `fraud_detect/corpus/splits.py`
    - `assign_split(...)`, fraction validation, the gap report
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8_
  - [x]* 18.3 Write property test for splits
    - **Property 16: Splits never leak**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**

- [x] 19. Corpus builder and build command
  - [x] 19.1 Write failing tests for the builder and the command
    - In `tests/corpus/test_builder.py`: the plan is deterministic and independent of build order; with `--per-layout-genuine 2 --per-layout-tampered 2` the plan has two genuine and two tampered per layout; tamper types follow `--tamper-mix` and are even when none is given; months are drawn from 6 to 12 when not given and invalid month settings are refused; a non-empty output folder is refused without `--force`; output inside `fraud_detect/`, `tests/` or `.kiro/` is refused; README and corpus manifest contain "SYNTHETIC DATA", the Generator_Version, seed, command line and counts, plus the counts per split, layout and tamper type; `companies.json` holds UEN, registered name and baselines equal to the genuine ledger's averages; a deliberately failing document is recorded under `failed`, the others are still written and the exit code is 1; building twice with the same settings gives byte-equal manifests, CSVs and companies file and equal PDF text and metadata; counts add up
    - _Requirements: 4.1, 9.1, 10.2, 10.3, 10.4, 11.6, 12.1, 12.2, 12.3, 12.4, 12.5, 12.7, 12.9, 12.10, 12.11, 13.1_
  - [x] 19.2 Implement `fraud_detect/corpus/builder.py`
    - `CorpusBuilder.plan()` and `.build()` following the design flow (simulate, tamper, render, file operations, finish, write outputs); document seeds from `stable_seed`; companies file; corpus manifest and README text; failure accounting
    - _Requirements: 4.1, 9.1, 10.2, 10.3, 10.4, 11.6, 12.2, 12.7, 12.9, 12.10, 12.11, 13.1_
  - [x] 19.3 Implement the command `fraud_detect/corpus/build.py`
    - `python -m fraud_detect.corpus.build` with `--out` (default `data/corpus`), `--per-layout-genuine`, `--per-layout-tampered`, `--months-min`, `--months-max`, `--seed`, `--hold-out-layout`, `--tamper-mix`, `--force`; validation and refusal messages in plain language; exit codes 0, 1 (failed documents) and 2 (refused)
    - _Requirements: 12.1, 12.3, 12.4, 12.5, 12.6, 12.9_
  - [x]* 19.4 Write property tests for the builder
    - **Property 8: Same seed, same corpus (whole build)**
    - **Property 17: Counts add up**
    - **Property 19: The companies file matches the ledger**
    - Use small corpora (two layouts, one or two documents each) to keep time down
    - **Validates: Requirements 10.2, 10.3, 10.5, 11.6, 12.10, 12.11, 13.1**

- [x] 20. Temporary registration of the fictional companies
  - [x] 20.1 Write failing tests for the registration scope
    - In `tests/corpus/test_scope.py`: inside the scope `lookup_entity` finds each UEN under its registered name and Node 3 finds the baseline; after the scope and after an error inside it the two dictionaries equal their earlier contents; live lookup stays off; the source files `reference_data.py` and `registry.py` are unchanged on disk; a UEN that already exists with a different name is refused and reported; a generated genuine file raises no `FL_UEN_NOT_FOUND` inside the scope and does raise it outside
    - _Requirements: 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_
  - [x] 20.2 Implement `fraud_detect/corpus/scope.py`
    - `registered_companies(...)` context manager that changes the dictionaries in place and restores them in a `finally` block; accepts the path of `companies.json` or a list of records
    - _Requirements: 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_
  - [x]* 20.3 Write property test for registration
    - **Property 18: Company registration is temporary**
    - **Validates: Requirements 13.2, 13.3, 13.4**

- [x] 21. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Build a tiny corpus by hand with the command into a temporary folder and open two of the PDFs to look at them (delete the folder afterwards).

- [x] 22. Verification report
  - [x] 22.1 Write failing tests for the Node 1 part of the report
    - In `tests/corpus/test_verify_node1.py`: the per-layout table (documents, share of rows read, share with replayed balance equal to printed closing, share with matching header, chosen profile, finding codes) computed from in-memory results; genuine and tampered rows-read shares are separate; the count of dropped overlay rows; a missing corpus folder prints a line starting with "skipped" and returns 0; an "elapsed" line is printed; nothing is written unless `--json` or `--markdown` is given; no `data/cases.json` or audit file appears; injected `ingest` and `screen` functions are used; `--sample N` measures at most N documents per layout in a fixed order
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9_
  - [x] 22.2 Implement the Node 1 part of `fraud_detect/corpus/verify.py`
    - `load_corpus`, `measure_node1`, `summarise_layout`, argument parsing, output helpers; modelled on `fraud_detect/nodes/ingest_parse/eval_layouts.py`
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9_
  - [x] 22.3 Write failing tests for the Node 2, 3 and 4 part of the report
    - In `tests/corpus/test_verify_nodes.py`: false-alarm figures for genuine documents (AUTO_CLEAR share, finding-code counts) and separately for re-saved genuine documents; detection table by tamper type and method (share with MEDIUM/HIGH finding, share not auto-cleared, codes seen); share with the descriptive expected code; the blind-spot rule (listed if and only if the MEDIUM/HIGH share is 0% or the not-auto-cleared share is no higher than the genuine share); per-split figures; follow-up suggestions printed; the exit code stays 0 whatever the rates are; an exception from a node is counted as "screening error" and the run continues; the scope is entered once around the whole run
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9_
  - [x] 22.4 Implement the Node 2, 3 and 4 part of `verify.py`
    - `measure_nodes_2_to_4`, `detection_by_tamper_type`, `blind_spots`, the follow-up suggestion text; the default `screen` runs `ingest`, `visual_forensics.analyze`, `financial_logic.analyze` and `risk_synthesis.synthesize` in memory (no case store) inside `registered_companies`
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9_
  - [x]* 22.5 Write property test for the report arithmetic
    - **Property 20: The report's arithmetic is right**
    - **Validates: Requirements 14.2, 14.9, 15.2, 15.3, 15.5, 15.8**

- [x] 23. End-to-end check on a mini corpus
  - [x] 23.1 Write the end-to-end test
    - In `tests/corpus/test_end_to_end.py`: build a mini corpus into a temporary folder (six layouts, 1 genuine and 1 tampered each, small month counts, one tamper type from each group), run the report with the default functions, and assert only the Node 1 facts on genuine documents (100% of rows, balances, headers, no MEDIUM/HIGH data-quality finding); print but do not assert the Node 2, 3 and 4 figures; assert that nothing was written outside the temporary folder
    - Keep the whole `tests/corpus` folder to about 90 seconds; share the mini corpus between tests with a module-scoped fixture
    - _Requirements: 3.7, 14.7, 15.6, 17.6, 18.6_

- [x] 24. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Record the time taken by `tests/corpus`.

- [x] 25. Measurement and results
  - [x] 25.1 Build the smoke corpus and run the report
    - Run `python -m fraud_detect.corpus.build --out data/corpus --per-layout-genuine 2 --per-layout-tampered 2 --seed 20260601` and then `python -m fraud_detect.corpus.verify --corpus data/corpus` (the folder is git-ignored)
    - Time the rendering of one 12-month statement of each layout
    - If the smoke corpus has too few documents to say anything about a tamper type, also build a larger one (for example 10 genuine and 10 tampered per layout, a different seed) and say so in the results
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.8, 17.5_
  - [x] 25.2 Write `.kiro/specs/synthetic-corpus-generator/evaluation-results.md`
    - Include the command, seed, counts, Generator_Version, date and elapsed time; the Node 1 table; the false-alarm table for genuine documents (including re-saved ones, and which false alarms trace to genuine hard cases or to the `TT` substring check); the detection table by tamper type and method; the blind-spot table (for example overlay rows dropped by Node 1, nearby-month copies with different amounts, rapid pass-through, dormant reactivation); the 12-month render time; the follow-up suggestions
    - State plainly that the figures are demonstrations on synthetic data, not real-world accuracy, and report bad numbers as they are. Do not change any node or the generator to improve a number; if a result looks wrong, stop and ask the user
    - _Requirements: 15.7, 16.1, 16.2, 16.3, 17.5, 19.1_

- [x] 26. Documentation
  - [x] 26.1 Add the section "Synthetic corpus generator" to `FRAUD_DETECT_README.md`
    - Plain language: what it is for, how to build and measure (the two commands), where the files go and that they are never committed, the 16 fraud types in short, the splits and the held-out layout, how the companies are registered for the measurement, the results table (labelled indicative) taken from the evaluation file
    - Honest limits: synthetic only; circularity risk and its mitigations (held-out layout, several seeds, randomised styling, no tuning of detectors to the generator, no training on `statement_E` / `statement_F`); only the six known layouts; simplifications (invoice numbers follow receipt order, simple company model, textbook fraud patterns); the "should be seen by" column is descriptive, not a target
    - Describe Stage B (retraining Node 6a and 6b) as a separate follow-up spec and list its prerequisites
    - Add a small test in `tests/corpus/test_hygiene.py` that the README contains the section heading
    - _Requirements: 16.4, 16.5, 16.6, 19.1, 19.2, 19.3, 19.4, 19.5_

- [x] 27. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run the whole suite: all 1155 earlier tests still pass, the one skipped test is still skipped, and the new tests pass.
  - Hygiene: `git status` shows no PDF, no `labels.csv`, no corpus file and no external dataset file; `data/` is still ignored; `git diff` shows no change to `fraud_detect/synthetic.py`, `fraud_detect/nodes/`, `fraud_detect/registry.py`, `fraud_detect/reference_data.py` or any earlier test; no new dependency was added; the package source contains none of the forbidden strings; delete any temporary folders.
  - Confirm `tests/corpus` ran in about 90 seconds or less and record the time.
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.6, 17.7, 18.3, 18.4_

- [ ] 28. Build-time UEN collision check
  - [ ] 28.1 Write failing tests for the UEN check
    - In `tests/corpus/test_uen_check.py` (see the design section "Build-time UEN collision check"): snapshot loader (text, CSV with a `uen` column in any case, JSON list of strings and of objects, blank and `#` lines, lower-case input, missing file, empty file, CSV without a `uen` column); `UenChecker.is_taken` with a tiny fake snapshot, with a UEN from `ACRA_REGISTRY`, and with a fake `fetch` that returns a found record, an empty result and a failure (failure must raise `UenCheckError`, never "free"); `describe()` content, with no real company name anywhere
    - `make_company`: `is_taken=None` gives exactly the old profile (pin it); a forced clash gives a different UEN and an otherwise equal profile; the redraw is the same on every call; an always-taken callback ends in `UenCheckError`
    - Builder: in-corpus uniqueness (force two companies onto the same first UEN); same seed and same snapshot give equal `companies.json`; a snapshot holding one company's first UEN changes only that company; default run prints and stores the "not checked against the real registry" warning; manifest and README show source, counts and redraws; a bad snapshot and a failing fake live fetch both give exit code 2 and an empty output folder
    - `build.py`: `--uen-snapshot` and `--check-live-uen` parse; defaults are `None` and `False`
    - No network: a fixture makes `urllib.request.urlopen` raise; a test fails if `uen_check.py` imports `urllib`, `socket` or `requests` directly; use `registry.clear_cache()` between tests
    - These tests must fail now for the right reason (missing module and options)
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 20.10, 20.11, 20.13_
  - [ ] 28.2 Implement `fraud_detect/corpus/uen_check.py` and wire it in
    - Create `uen_check.py` with `UenCheckError`, `load_snapshot` and `UenChecker` as in the design; use only the public `registry.lookup_entity` (`live=False` for the in-repo part, `live=True` plus the optional `fetch` for the live part; `UNAVAILABLE` raises)
    - `company.py`: add the optional `is_taken` argument to `make_company`; first draw unchanged; redraws from the child stream `rng(seed, "company", company_id, "uen-redraw")`; limit of 1000 tries
    - `schema.py`: add `uen_snapshot: Path | None = None` and `check_live_uen: bool = False` to `BuildConfig`
    - `builder.py`: build the checker from the config; pre-pass over the plans in plan order before the output folder is created; keep profiles by `doc_id`; add the `uen_check` object to `corpus_manifest.json`, a "UEN check" line to `README.txt`, and the warning when the real registry was not checked
    - `build.py`: add `--uen-snapshot PATH` and `--check-live-uen`; print `UenCheckError` messages and return exit code 2
    - Do not change `registry.py`, `reference_data.py`, Nodes 1 to 4 or any earlier test; add no dependency
    - Make the 28.1 tests pass, and check that the earlier determinism and end-to-end tests still pass unchanged
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 20.10, 20.11, 20.12_
  - [ ]* 28.3 Write property test for UEN determinism and no clash
    - **Property 22: Generated UENs never clash and are repeatable**
    - With `hypothesis` (at least 100 examples; pure functions, no PDFs): for any seed, any set of reserved UENs and any number of company ids, no company UEN is in the reserved set or repeats, and calling twice gives the same UENs; a company whose first draw is not reserved keeps its profile
    - **Validates: Requirements 20.1, 20.2, 20.3**
  - [ ] 28.4 Checkpoint - Ensure all tests pass
    - Ensure all tests pass, ask the user if questions arise.
    - Run the whole suite: all earlier tests still pass, the one skipped test is still skipped, and the new tests pass; `tests/corpus` still runs in about 90 seconds or less.
    - Hygiene: `git diff` shows no change to `fraud_detect/registry.py`, `fraud_detect/reference_data.py`, `fraud_detect/nodes/` or any earlier test; no network was used; no snapshot file or corpus file is tracked by git; delete any temporary folders.
    - _Requirements: 20.12, 20.13, 17.1, 17.3, 17.6, 17.7_
  - [ ] 28.5 Update `FRAUD_DETECT_README.md` and `evaluation-results.md`
    - In the README section "Synthetic corpus generator": explain in plain language the UEN check, the two options, that the default checks only the in-repo registry and the corpus itself and prints a warning, that live lookup is never on by default, that a failed snapshot or live check stops the build with exit code 2, and how to get a snapshot (download the ACRA UEN list from data.gov.sg yourself and keep it outside the repository)
    - In `evaluation-results.md`: add a short note that the existing run was made before the check existed and with the default option (real registry not checked), and that no real company data are stored
    - Add a check in `tests/corpus/test_hygiene.py` that the README mentions `--uen-snapshot` and `--check-live-uen`
    - _Requirements: 20.6, 20.7, 20.10, 20.11, 16.4_
## Notes

- Tasks marked with `*` are optional property tests and can be skipped for a faster first version. Unit and example tests are not optional.
- Each group is test first: the x.1 task writes tests that fail until the x.2 task is done.
- Property tests use `hypothesis`. Pure properties use at least 100 examples; properties that draw PDFs use about 10 to 20 examples per layout and say so. Tag format: `Feature: synthetic-corpus-generator, Property N: ...`.
- Layout tasks 6 to 9 add one entry each to `layouts/specs.py`; they are in separate waves because they share that file and the table engine.
- The corpus is generated into `data/corpus/` (ignored by git). Never commit a PDF, `labels.csv` or any corpus file.
- External files (`statement_E`, `statement_F`, the answer key, the forgery dataset) are never read by the package or the tests, never trained on, and never committed.
- If a measurement looks wrong, stop and ask the user; do not tune the generator or any node to change a number.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1"] },
    { "id": 2, "tasks": ["1.3", "3.2"] },
    { "id": 3, "tasks": ["1.4", "2.2", "4.1", "5.1", "6.1", "7.1", "8.1", "9.1", "10.1"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.4", "4.2", "5.2"] },
    { "id": 6, "tasks": ["6.2"] },
    { "id": 7, "tasks": ["7.2"] },
    { "id": 8, "tasks": ["8.2"] },
    { "id": 9, "tasks": ["9.2"] },
    { "id": 10, "tasks": ["10.2", "12.1", "13.1"] },
    { "id": 11, "tasks": ["10.3", "12.2", "13.2", "14.1"] },
    { "id": 12, "tasks": ["12.3", "13.3", "14.2", "15.1", "16.1"] },
    { "id": 13, "tasks": ["14.3", "15.2"] },
    { "id": 14, "tasks": ["15.3", "16.2"] },
    { "id": 15, "tasks": ["16.3", "18.1"] },
    { "id": 16, "tasks": ["18.2"] },
    { "id": 17, "tasks": ["18.3", "19.1"] },
    { "id": 18, "tasks": ["19.2"] },
    { "id": 19, "tasks": ["19.3", "20.1", "22.1"] },
    { "id": 20, "tasks": ["19.4", "20.2", "22.2"] },
    { "id": 21, "tasks": ["20.3", "22.3"] },
    { "id": 22, "tasks": ["22.4"] },
    { "id": 23, "tasks": ["22.5", "23.1"] },
    { "id": 24, "tasks": ["25.1"] },
    { "id": 25, "tasks": ["25.2"] },
    { "id": 26, "tasks": ["26.1"] },
    { "id": 27, "tasks": ["28.1"] },
    { "id": 28, "tasks": ["28.2"] },
    { "id": 29, "tasks": ["28.3", "28.5"] }
  ]
}
```
