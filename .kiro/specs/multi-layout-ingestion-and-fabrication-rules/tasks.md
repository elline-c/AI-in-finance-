# Implementation Plan: Multi-Layout Ingestion (Stage 1, Node 1 only)

## Overview

Build the layout-tolerant Node 1 parser with exactly six profiles, in small test-first steps. Each component gets its failing tests first, then the implementation, then an optional property test. Component code lives in `fraud_detect/nodes/ingest_parse/`; tests live in `tests/ingest_parse/` and use text and word-box fixtures built in code with fictional values. Tests that need `~/Downloads/statement_forgery_dataset/`, `statement_E.pdf`, `statement_F.pdf` or `ANSWER_KEY_6M_v2.json` skip cleanly when those paths are missing. None of those files is ever committed.

Language: Python (matches the repo). Test command: `/Users/weiyang01/AI-in-finance-/.venv/bin/python -m pytest -q` with `AUDIT_SIGNING_KEY_PATH` and `FRAUD_ENABLE_TXN_MODEL` unset.

Out of scope for these tasks: the Node 3 fabrication rules (**Deferred to a follow-up spec**), the `FL_PERIOD_LENGTH` fix (already done), ML retraining, OCR, UIs, Node 4 weights. No new dependencies. Do not edit `fraud_detect/store.py`.

Honest limit: with only synthetic layouts, passing these tasks proves the parser design, not that real Singapore bank statements will parse.

## Tasks

- [x] 1. Foundations and safety net
  - [x] 1.1 Create test scaffolding and in-code fixtures
    - Create `tests/ingest_parse/__init__.py`, `conftest.py` and `fixtures.py`
    - `conftest.py` defines path constants for the external files and `skipif` markers (`requires_dataset`, `requires_labelled_statements`)
    - `fixtures.py` builds `PageContent` text lines and word boxes (text, x0, x1, top) for each of the six layouts using fictional names and numbers, including a continuation page and a six-page "Page 1 of 1" file
    - Create the empty package `fraud_detect/nodes/ingest_parse/__init__.py`
    - _Requirements: 14.2, 14.3, 14.5_

  - [x] 1.2 Write the compatibility safety-net test first
    - In `tests/test_ingest_compat.py`, ingest the existing generator statement with the current Node 1 and store the expected header fields and transactions as a literal in the test
    - Assert `_ROW_RE`, `_PERIOD_RE`, `_parse_date`, `_parse_money` import from `fraud_detect.nodes.ingestion`
    - This test must pass now and keep passing after every later task
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.3 Add the optional model fields
    - In `fraud_detect/models.py` add `Transaction.value_date_raw`, `Transaction.direction_method`, `SectionSummary`, `IngestionResult.layout_profile`, `IngestionResult.parse_coverage`, `IngestionResult.sections`, all optional with defaults
    - Write a small test that old `IngestionResult` objects still build without the new fields
    - _Requirements: 1.4, 7.7, 8.4_

- [x] 2. Date reader
  - [x] 2.1 Write failing tests for `dates.py`
    - Table-driven tests for every style: `dd/mm/yyyy`, `yyyy-mm-dd`, `dd Mon yyyy`, `d Mon yyyy`, `dd-mm-yy`, `d Mon`
    - Cases: two-digit year to 20yy, no-year uses period year, year-end crossing, out-of-period gives no date and keeps raw text, period line with and without colon and with `-`, `to`, en dash, day-first reading
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x] 2.2 Implement `fraud_detect/nodes/ingest_parse/dates.py`
    - `ParsedDate`, `parse_full_date`, `parse_period`, `parse_day_month`
    - Make 2.1 pass
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [x]* 2.3 Write property test for the date reader
    - **Property 1: Date round trip**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.6**

- [x] 3. Money reader
  - [x] 3.1 Write failing tests for `money.py`
    - Cases: `S$5,365.00`, `SGD 28,249.88`, `-SGD 13.47`, `+SGD 11.08`, `(1,200.00)`, `3,900.18 CR`, `DR` suffix, plain number, invalid text returns no value, rounding to two decimals
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Implement `fraud_detect/nodes/ingest_parse/money.py`
    - `Money`, `parse_money`; keep `ingestion._parse_money` as a thin wrapper
    - Make 3.1 pass
    - _Requirements: 1.2, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x]* 3.3 Write property test for the money reader
    - **Property 2: Money round trip**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 4. Header reader
  - [x] 4.1 Write failing tests for `header.py`
    - Fixtures: the header block of each of the six layouts, including `NAME Account No. X`, `Company UEN`, `BUSINESS CURRENT ACCOUNT` line, summary row with `S$`, `SGD` and plain values
    - Cases: holder from the left half of the account line, holder label variants, UEN accepted only if it fits the pattern, no UEN gives none, summary values mapped by label, merged-digit summary read from word positions, unreadable summary left empty, ambiguous holder left empty, missing field gives the matching `DQ_MISSING_*`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_

  - [x] 4.2 Implement `fraud_detect/nodes/ingest_parse/header.py`
    - `SectionHeader`, `read_header` with same-line, left/right-pair and summary-box strategies
    - Make 4.1 pass
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9_

  - [x]* 4.3 Write property test for unreadable fields
    - **Property 7: Unreadable means flagged, not guessed**
    - **Validates: Requirements 2.8, 4.7, 4.8**

- [x] 5. Table reader
  - [x] 5.1 Write failing tests for `table.py`
    - Fixtures for: Withdrawals/Deposits, Debit/Credit, `Money out`/`Money in` (two-word headers), single `Amount`, dual date columns, wrapped description, brought-forward row, closing row, footer and `SPECIMEN` lines, repeated header on a continuation page, no header (line fallback)
    - Right-edge column assignment test using the measured right edges (debit 400, credit 475)
    - Each printed row is read exactly once
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 5.2 Implement `fraud_detect/nodes/ingest_parse/table.py`
    - `AmountCell`, `RawRow`, `TableLayout`, `find_table`, `read_rows`
    - Make 5.1 pass
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x]* 5.3 Write property test for non-transaction rows
    - **Property 5: Non-transaction rows are excluded**
    - **Validates: Requirements 5.5, 5.9**

- [x] 6. Direction resolver
  - [x] 6.1 Write failing tests for `direction.py`
    - Cases per method (`column`, `printed_sign`, `balance_change`), magnitude equals printed amount, conflicting evidence gives `DQ_DIRECTION_UNCERTAIN` and keeps the column or sign direction, balance-only with mismatch gives `DQ_DIRECTION_UNCERTAIN`, no evidence gives `DQ_UNREADABLE_ROW` and no transaction
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 6.2 Implement `fraud_detect/nodes/ingest_parse/direction.py`
    - `Resolved`, `resolve_direction`
    - Make 6.1 pass
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x]* 6.3 Write property tests for direction and printed amount
    - **Property 3: Printed amount is preserved**
    - **Property 4: Direction follows the evidence**
    - **Validates: Requirements 5.9, 6.1, 6.2, 6.3, 6.4**

- [x] 7. Layout profiles and selection
  - [x] 7.1 Write failing tests for `layouts.py`
    - Registry has exactly six profiles in the required order with the required layout mapping
    - Each fixture layout selects its own profile with Coverage of 1.0
    - Tie goes to the earlier profile; tie with different outputs gives `DQ_LAYOUT_UNRECOGNISED`; Coverage below 0.9 gives `DQ_LAYOUT_UNRECOGNISED` (MEDIUM, weight 8) and still returns the rows read
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 7.2 Implement `fraud_detect/nodes/ingest_parse/layouts.py`
    - `LayoutProfile`, `REGISTRY` with the six profiles, `choose_profile`, `ProfileChoice`
    - Make 7.1 pass
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 8. Self-check and sections
  - [x] 8.1 Write failing tests for `selfcheck.py`
    - Replay to the printed closing balance, compare debit and credit sums to printed totals
    - Rows consistent but a printed total off by more than 0.01 gives `DQ_PARSE_INCONSISTENT` (LOW, weight 3)
    - An unread printed total skips that comparison with no finding
    - Row-by-row arithmetic problems are not reported here
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 8.2 Implement `fraud_detect/nodes/ingest_parse/selfcheck.py`
    - `self_check`; make 8.1 pass
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 8.3 Write failing tests for `sections.py`
    - New section at each `Statement Period` line; page without one continues the previous section; `Page 1 of 1` on every page is ignored; six pages give six segments with distinct periods; file opening equals first section opening and file closing equals last section closing; one `SectionSummary` per section
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 8.4 Implement `fraud_detect/nodes/ingest_parse/sections.py`
    - `split_into_sections`, file-level balance rules; make 8.3 pass
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 9. Wire the parser into `ingest()`
  - [x] 9.1 Write failing end-to-end tests on fixture pages
    - Add `parse_pages(pages)` as the testable entry point; test it on the fixture of each of the six layouts (header fields, transactions, replayed balance equals printed closing, chosen profile name)
    - Six-page "Page 1 of 1" fixture run through Node 3 `analyze` emits none of `FL_MONTH_DUPLICATE`, `FL_MONTH_MISSING`, `FL_MONTH_OUT_OF_ORDER`
    - Pathological 10,000-character line finishes within 1 second; a page with no text gives `DQ_NO_TEXT`; an error on one page gives `DQ_LAYOUT_UNRECOGNISED` for that page only
    - _Requirements: 7.7, 8.7, 8.8, 10.2, 10.3, 17.1, 17.2, 17.3_

  - [x] 9.2 Replace the fixed regex path in `ingestion.py`
    - `ingest()` extracts text lines and word boxes per page with pdfplumber, calls `parse_pages`, and fills the new `IngestionResult` fields
    - Keep `_ROW_RE`, `_PERIOD_RE`, `_parse_date`, `_parse_money` defined; keep metadata and font extraction unchanged
    - Make 9.1 and the safety-net test 1.2 pass; run the whole suite
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 7.7, 8.5, 8.6_

  - [x]* 9.3 Write PDF-render property tests with reportlab
    - Render small tables (up to about 30 rows, modest `max_examples`) in each supported layout, then ingest the PDF
    - **Property 6: Layout independence**
    - **Property 8: Tampered amount stays visible**
    - **Property 9: Generated multi-page month files do not raise coverage false alarms**
    - **Validates: Requirements 7.3, 8.1, 8.2, 8.3, 8.7, 8.8, 10.1, 10.2, 12.2**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Acceptance tests on the external synthetic files (skip when absent)
  - [x] 11.1 Write the dataset acceptance test
    - `tests/test_ingest_dataset.py`, marked `requires_dataset`
    - For each of the five layouts take the Sample (first 20 genuine documents from `labels.csv`) and assert: at least 99 percent of rows read against `n_transactions`, replayed balance equals printed closing within 0.01, holder, account, period, opening and closing match the printed values, expected profile chosen
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 14.2_

  - [x] 11.2 Close gaps found by 11.1
    - For each failing layout add a word-box fixture to `tests/ingest_parse/fixtures.py` first, then adjust the profile or reader until 11.1 passes
    - Do not add code that targets a single document
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 11.3 Write and pass the Statement_E and Statement_F test
    - `tests/test_ingest_statement_ef.py`, marked `requires_labelled_statements`
    - Assert header fields and file opening against the Answer_Key; per-month opening, deposits, withdrawals and closing within 0.01 (E: `monthly_as_shown`, F: `monthly`); file closing equals the last section closing; all 12 `inserted_rows` found as parsed transactions by month, date and amount; no `DQ_NO_TXNS` for E
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 14.2_

  - [x] 11.4 Write and pass the pipeline outcome tests
    - Synthetic_Clean_Sample still `AUTO_CLEAR` with risk about 1 (runs in CI, no external file)
    - Statement_F: assert `AUTO_CLEAR` or no HIGH Finding, and print every other Finding that fires
    - Statement_E: print routing outcome, risk score and Finding codes (report, do not assert)
    - _Requirements: 11.7, 11.8, 12.1_

- [x] 12. Evaluation script
  - [x] 12.1 Write failing tests for `eval_layouts.py`
    - On tiny in-code fixtures: output is identical on repeat runs, `--sample N` default 20, missing folders print a skipped message and exit code 0
    - _Requirements: 13.4, 13.6, 13.7, 14.3_

  - [x] 12.2 Implement `fraud_detect/nodes/ingest_parse/eval_layouts.py`
    - Runs with `python -m fraud_detect.nodes.ingest_parse.eval_layouts`
    - Per-layout table: profile chosen, documents, share of rows read, share of balances matching, share of headers matching, Finding codes fired
    - E/F month-by-month comparison and inserted rows found; `--sample N`; `--all-genuine` prints the false-alarm table of routing outcomes and Finding codes; reports elapsed time for the longest file
    - _Requirements: 10.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 17.4_

  - [x] 12.3 Run the script and record the numbers
    - Run once with the default sample and once with `--all-genuine`, and keep the output for the README results table
    - _Requirements: 10.5, 13.2, 13.5_

- [x] 13. Documentation
  - [x] 13.1 Update `FRAUD_DETECT_README.md`
    - Add "How to add a layout profile" (build a word-box fixture, write a failing test, add the Profile, register it, run the Eval_Script)
    - List the six profiles and the layout each covers
    - State the synthetic-only limitation, that real-bank profiles come later when data is safe, and that the external evaluation files are not stored in the repository
    - Add the results table from 12.3, labelled synthetic-only
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 10.5_

  - [x] 13.2 Check repository hygiene
    - Add a test that fails if any dataset PDF, `statement_E.pdf`, `statement_F.pdf` or the Answer_Key is tracked by git (`git ls-files` check, skipped outside a git checkout)
    - _Requirements: 14.1_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; the component unit tests are not optional.
- Each component follows test first (x.1 tests, x.2 implementation), so a task that writes tests expects them to fail until the next task is done.
- Properties 10 to 15 of the design (period length, fabrication rules) have no tasks here: the period fix is done and the fabrication rules are Deferred to a follow-up spec. Property 16 is covered by tasks 1.2, 9.2 and 11.4.
- Property tests use `hypothesis` with at least 100 examples (fewer for PDF-render tests) and carry the tag `Feature: multi-layout-ingestion-and-fabrication-rules, Property N: ...`.
- External files are read-only, never committed, and tests using them skip when missing.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "3.2"] },
    { "id": 3, "tasks": ["2.3", "3.3", "4.1", "5.1", "8.1"] },
    { "id": 4, "tasks": ["4.2", "5.2", "8.2"] },
    { "id": 5, "tasks": ["4.3", "5.3", "6.1", "7.1", "8.3"] },
    { "id": 6, "tasks": ["6.2", "7.2", "8.4"] },
    { "id": 7, "tasks": ["6.3", "9.1"] },
    { "id": 8, "tasks": ["9.2"] },
    { "id": 9, "tasks": ["9.3", "11.1", "12.1"] },
    { "id": 10, "tasks": ["11.2", "12.2"] },
    { "id": 11, "tasks": ["11.3", "12.3"] },
    { "id": 12, "tasks": ["11.4", "13.1"] },
    { "id": 13, "tasks": ["13.2"] }
  ]
}
```
