# Requirements Document

## Introduction

This is Stage 1 of the multi-layout ingestion work. Stage 1 covers **Node 1 (Ingestion) only**: a layout-tolerant statement parser that reads exactly six layout profiles: our own generator, plus the five fictional bank layouts in the synthetic `statement_forgery_dataset` (one of which is also the layout of `statement_E.pdf` and `statement_F.pdf`).

The Node 3 fabrication rules in the design (`FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH`) are **Deferred to a follow-up spec**. The `FL_PERIOD_LENGTH` inclusive-end fix is **already done** (`period_months()` in `financial_logic.py`, branch `fix/period-length-inclusive`) and is not re-implemented here. Retraining ML, OCR, UIs and Node 4 weights are out of scope.

**Evidence limitation (stated honestly).** The team cannot collect real bank statements (data-leakage and PDPA concerns). All evidence comes from the synthetic dataset, `statement_E`/`statement_F` with their answer key, and our own generator. Success in this stage proves that the parser design works on these six layouts. It does **not** prove that real Singapore bank statements will parse. Real-bank profiles are added later, when data can be handled safely (for example on-premises, anonymised, or with consent).

**External files.** `~/Downloads/statement_forgery_dataset/`, `~/Downloads/statement_E.pdf`, `~/Downloads/statement_F.pdf` and `~/Downloads/ANSWER_KEY_6M_v2.json` have an unknown licence. They are never committed to the repository. Tests that need them skip cleanly when they are missing. Pure unit tests built on in-code word-box and text fixtures run without them.

**Layout facts verified by reading sample PDFs** (20 genuine documents per layout, 100 in total; see the corrections note in `design.md`):

- Every layout puts the holder on the same text line as `Account No.`, with the holder name on the left. No layout has an `Account Holder:` label.
- Every layout writes `Statement Period <start> to <end>` with no colon and `to` as the separator.
- Every layout has a four-label summary row (`Opening Balance`, `Total Withdrawals`, `Total Deposits`, `Closing Balance`) followed by one row of four values.
- Every layout has a `BALANCE BROUGHT FORWARD` row and a `CLOSING BALANCE` row (the closing row is on the last page).
- Layouts 1 and 2 continue onto a second page that has a repeated table header, no `Statement Period` line and no brought-forward row.
- Amounts are right-aligned under their column header.

## Glossary

- **Node_1**: The Ingestion stage, `fraud_detect/nodes/ingestion.py`, with its helpers in `fraud_detect/nodes/ingest_parse/`.
- **Parser**: The layout-tolerant statement reader inside Node_1.
- **Profile**: A small bundle of settings (date styles, period separators, amount style, second date column, balance suffix) describing one statement layout.
- **Profile_Registry**: The ordered list of the six Profiles. `generator_v1` is first.
- **Profile_Selector**: The part of the Parser that scores every Profile on every page and keeps the best.
- **Coverage**: For a page, the share of table-looking lines that became a valid row, from 0 to 1. The default minimum is 0.9.
- **Section**: One statement period within a file, together with the pages that continue it.
- **Continuation_Page**: A page that belongs to the previous Section because it has no `Statement Period` line.
- **Date_Reader**: The component that turns printed dates into calendar dates.
- **Money_Reader**: The component that turns printed amounts into numbers.
- **Header_Reader**: The component that reads holder, UEN, account number, currency, period, opening, closing and printed totals.
- **Table_Reader**: The component that finds the transaction table and reads its rows.
- **Direction_Resolver**: The component that decides whether an amount is a credit (positive) or a debit (negative).
- **Self_Check**: The component that replays parsed rows and compares them to the printed totals.
- **Eval_Script**: The reusable evaluation script `fraud_detect/nodes/ingest_parse/eval_layouts.py`, run with `python -m fraud_detect.nodes.ingest_parse.eval_layouts`.
- **Transaction_Row**: A dated row of the transaction table with an amount and a running balance. Brought-forward, closing-balance, total and footer rows are not Transaction_Rows.
- **Dataset**: The synthetic `statement_forgery_dataset`, five fictional bank layouts, 1,000 genuine and 1,000 tampered documents.
- **Dataset_Layout_N**: Layout N of the Dataset: 1 Corvane Trust, 2 Halcyon Ledger Bank, 3 Aldermoor Savings, 4 Kestrelmark Bank, 5 Vellichor Bank.
- **Sample**: The first 20 genuine documents of a layout, in `labels.csv` order, where `is_tampered` is 0.
- **Statement_E**: `statement_E.pdf`, a tampered six-month business statement with 12 inserted rows.
- **Statement_F**: `statement_F.pdf`, a genuine six-month business statement.
- **Answer_Key**: `ANSWER_KEY_6M_v2.json`, the ground truth for Statement_E and Statement_F.
- **Synthetic_Clean_Sample**: The existing statement made by our own generator that currently reaches `AUTO_CLEAR` with a risk score of about 1 in `tests/test_synthetic_clean.py`.
- **Pipeline**: The full screening flow in `fraud_detect/pipeline.py`.
- **Finding**: A reason-coded result with a severity and a weight, as used by Nodes 1 and 3.
- **README**: `FRAUD_DETECT_README.md`.

## Requirements

### Requirement 1: Backward compatibility and constraints

**User Story:** As a maintainer, I want the new parser to keep every existing behaviour, so that adding layouts does not break the pipeline or its tests.

#### Acceptance Criteria

1. THE Node_1 SHALL leave all 199 existing tests passing when the test suite runs with `AUDIT_SIGNING_KEY_PATH` and `FRAUD_ENABLE_TXN_MODEL` unset.
2. THE `ingestion` module SHALL keep `_ROW_RE`, `_PERIOD_RE`, `_parse_date` and `_parse_money` importable under those names.
3. WHEN a statement made by the existing generator is ingested, THE Node_1 SHALL return the same header fields and the same transactions as the previous Node_1.
4. THE Node_1 SHALL add its new fields to `IngestionResult` and `Transaction` as optional fields with defaults, with every existing field keeping its meaning.
5. THE Node_1 SHALL add no new runtime dependency to `requirements.txt`.
6. THE Node_1 SHALL perform no network call and no call to a language model or OCR service.

### Requirement 2: Date reading

**User Story:** As a reviewer, I want dates in every supported style read correctly or flagged, so that no wrong date is used silently.

#### Acceptance Criteria

1. THE Date_Reader SHALL read the styles `dd/mm/yyyy`, `yyyy-mm-dd`, `dd Mon yyyy`, `d Mon yyyy`, `dd-mm-yy` and `d Mon`.
2. WHEN a date has a two-digit year, THE Date_Reader SHALL read the year as 20yy.
3. WHEN a date has no year, THE Date_Reader SHALL take the year from the Section period.
4. IF a date without a year falls outside the Section period after the year is assigned, THEN THE Date_Reader SHALL return no date and keep the raw text.
5. WHEN a Section period crosses a year end and a date without a year has a month earlier than the period start month, THE Date_Reader SHALL assign the end year of the period.
6. THE Date_Reader SHALL read period lines with or without a colon after `Statement Period`, and with the separators `-`, `to` and the en dash.
7. THE Date_Reader SHALL read `dd/mm/yyyy` as day first.
8. IF a date cannot be read safely, THEN THE Node_1 SHALL store no date for that row and keep the printed text in `date_raw`.

### Requirement 3: Money reading

**User Story:** As a reviewer, I want amounts read exactly as printed in every supported style, so that sums and balances are trustworthy.

#### Acceptance Criteria

1. THE Money_Reader SHALL read the prefixes `S$`, `SGD`, `$` and no prefix.
2. THE Money_Reader SHALL read explicit signs `-` and `+`, including before a prefix (for example `-SGD 13.47` and `+SGD 11.08`).
3. THE Money_Reader SHALL read parentheses as a negative number.
4. WHEN a balance has the suffix `CR`, THE Money_Reader SHALL return the number with its sign unchanged and record the suffix.
5. WHEN a balance has the suffix `DR`, THE Money_Reader SHALL return the number as negative and record the suffix.
6. THE Money_Reader SHALL return values rounded to two decimals.
7. IF a text is not a valid amount, THEN THE Money_Reader SHALL return no value.

### Requirement 4: Header reading

**User Story:** As a reviewer, I want holder, account, period and balances read from each layout's header, so that the statement is identified and cross-checked.

#### Acceptance Criteria

1. WHEN a text line has the form `<name> Account No. <number>`, THE Header_Reader SHALL read the left part as the holder and the right part as the account number.
2. WHEN an `Account Holder` or `Account Name` label is present, THE Header_Reader SHALL read the holder from that label.
3. WHEN the label `UEN` or `Company UEN` is present, THE Header_Reader SHALL read a value that fits the Singapore UEN pattern and ignore a value that does not fit.
4. WHEN a Section has no UEN label, THE Header_Reader SHALL leave the UEN empty.
5. THE Header_Reader SHALL read the opening balance, total withdrawals, total deposits and closing balance from the summary row by matching each value to the label above it.
6. WHEN the summary values cannot be separated cleanly, THE Header_Reader SHALL read them from word positions, and IF they still cannot be separated, THEN THE Header_Reader SHALL leave them empty.
7. WHEN a holder cannot be chosen without guessing, THE Header_Reader SHALL leave the holder empty.
8. IF a header field is left empty, THEN THE Node_1 SHALL emit the matching existing `DQ_MISSING_*` Finding.
9. THE Header_Reader SHALL read `Currency`, `Statement Date` and the period from each Section.

### Requirement 5: Table reading

**User Story:** As a reviewer, I want every real transaction row read once and no summary or footer row counted, so that totals reconcile.

#### Acceptance Criteria

1. THE Table_Reader SHALL find the table header line that contains a date column, a description column and a balance column, and take the column positions from it.
2. THE Table_Reader SHALL accept the amount column names `Withdrawals`/`Deposits`, `Debit`/`Credit`, `Money out`/`Money in` and a single `Amount`.
3. THE Table_Reader SHALL assign an amount to a column by the right edge of the amount compared with the right edge of the column header.
4. WHEN a row has a second date, THE Table_Reader SHALL store it as the value date.
5. THE Table_Reader SHALL exclude `BALANCE BROUGHT FORWARD`, `CLOSING BALANCE`, total, page-counter and footer lines from the Transaction_Rows.
6. WHEN a line has no date and no amount, THE Table_Reader SHALL append its text to the description of the previous row.
7. WHEN the table header is repeated on a Continuation_Page, THE Table_Reader SHALL read the rows of that page as part of the same Section.
8. WHEN no table header is found on a page, THE Table_Reader SHALL read rows line by line using the date, description, amount and balance pattern.
9. THE Table_Reader SHALL read each printed Transaction_Row exactly once.

### Requirement 6: Direction resolution

**User Story:** As a reviewer, I want credits and debits assigned from the document's own evidence, so that an edited amount stays visible.

#### Acceptance Criteria

1. WHEN a layout has two amount columns and an amount lies under a known column, THE Direction_Resolver SHALL assign the direction from the column and record the method `column`.
2. WHEN an amount carries an explicit sign, THE Direction_Resolver SHALL assign the direction from the sign and record the method `printed_sign`.
3. WHEN neither a column nor a sign is available and a previous balance exists, THE Direction_Resolver SHALL assign the direction from the balance change and record the method `balance_change`.
4. THE Direction_Resolver SHALL return a signed amount whose magnitude equals the printed amount.
5. IF the direction from a column or sign disagrees with the change in running balance, THEN THE Direction_Resolver SHALL keep the column or sign direction and emit `DQ_DIRECTION_UNCERTAIN` for that row.
6. IF only the balance-change method is available and the printed amount differs from the balance change by more than 0.01, THEN THE Direction_Resolver SHALL keep the balance-change direction and emit `DQ_DIRECTION_UNCERTAIN` for that row.
7. IF no direction can be determined, THEN THE Node_1 SHALL emit `DQ_UNREADABLE_ROW` for that row and exclude it from the transactions.

### Requirement 7: Layout profiles and selection

**User Story:** As a maintainer, I want exactly six named profiles chosen automatically, so that each known layout parses and an unknown layout is reported.

#### Acceptance Criteria

1. THE Profile_Registry SHALL contain exactly six Profiles in this order: `generator_v1`, `two_col_dmon`, `two_col_slash`, `two_col_dual_date`, `signed_amount_iso`, `two_col_ddmmyy_cr`.
2. THE Profile_Registry SHALL map `two_col_dmon` to Dataset_Layout_5 and to the layout of Statement_E and Statement_F, `two_col_slash` to Dataset_Layout_1, `two_col_dual_date` to Dataset_Layout_2, `signed_amount_iso` to Dataset_Layout_3 and `two_col_ddmmyy_cr` to Dataset_Layout_4.
3. WHEN a file is ingested, THE Profile_Selector SHALL score every Profile on every page and choose the Profile with the highest Coverage.
4. IF two Profiles have the same Coverage, THEN THE Profile_Selector SHALL choose the earlier Profile in the Profile_Registry.
5. IF two Profiles tie and would produce different transactions, THEN THE Profile_Selector SHALL emit `DQ_LAYOUT_UNRECOGNISED` for that page instead of choosing one.
6. IF the best Coverage of a page is below 0.9, THEN THE Node_1 SHALL emit `DQ_LAYOUT_UNRECOGNISED` (MEDIUM, weight 8) naming the page and still return the rows that were read.
7. THE Node_1 SHALL record the chosen Profile name in `layout_profile` and the Coverage in `parse_coverage` on the `IngestionResult`.

### Requirement 8: Sections, multi-page files and file-level balances

**User Story:** As a reviewer, I want multi-page and multi-month files segmented correctly, so that month checks and closing balance checks work.

#### Acceptance Criteria

1. WHEN a page contains a `Statement Period` line, THE Node_1 SHALL start a new Section on that page.
2. WHEN a page contains no `Statement Period` line and the previous page belongs to a Section, THE Node_1 SHALL treat the page as a Continuation_Page of that Section.
3. THE Node_1 SHALL ignore the printed page counter, including `Page 1 of 1` repeated on every page, when it decides where a Section starts.
4. THE Node_1 SHALL record one `SectionSummary` per Section with its period, pages, opening, closing, total deposits and total withdrawals.
5. THE Node_1 SHALL set the file-level `opening_balance` to the opening of the first Section.
6. THE Node_1 SHALL set the file-level `closing_balance` to the closing of the last Section.
7. WHEN a file has six monthly pages each marked `Page 1 of 1`, THE Node_1 SHALL return six segments with six distinct periods.
8. WHEN the six monthly segments are analysed by Node 3, THE Node_3 SHALL emit none of `FL_MONTH_DUPLICATE`, `FL_MONTH_MISSING`, `FL_MONTH_OUT_OF_ORDER`.

### Requirement 9: Self-check

**User Story:** As a reviewer, I want our own reading mistakes separated from document fraud, so that a misread page is not reported as forgery.

#### Acceptance Criteria

1. WHEN a Section has a printed closing balance, THE Self_Check SHALL replay the parsed rows from the Section opening balance and compare the result to the printed closing balance.
2. WHEN a Section has printed totals, THE Self_Check SHALL compare the sum of debits and the sum of credits to the printed total withdrawals and total deposits.
3. IF every row balance reconciles row by row and a printed total disagrees with the parsed sum by more than 0.01, THEN THE Self_Check SHALL emit `DQ_PARSE_INCONSISTENT` (LOW, weight 3).
4. IF a printed total was not read, THEN THE Self_Check SHALL skip that comparison and emit no `DQ_PARSE_INCONSISTENT` for it.
5. THE Self_Check SHALL leave row-by-row arithmetic problems to the existing `FL_BALANCE_BREAK` and `FL_CLOSING_MISMATCH` rules.

### Requirement 10: Accuracy on the five Dataset layouts

**User Story:** As a project owner, I want measured reading accuracy on each of the five synthetic layouts, so that progress is shown by numbers.

#### Acceptance Criteria

1. WHEN the Sample of 20 genuine documents of a Dataset layout is ingested, THE Node_1 SHALL read at least 99 percent of the Transaction_Rows of that Sample, measured against `n_transactions` in `labels.csv`.
2. WHEN a Sample document is ingested, THE Node_1 SHALL return a running balance, replayed from the opening balance with the parsed amounts, that equals the printed closing balance within 0.01.
3. WHEN a Sample document is ingested, THE Node_1 SHALL return holder, account number, period start, period end, opening balance and closing balance equal to the values printed on the document.
4. WHEN a Sample document is ingested, THE Node_1 SHALL choose the Profile that the Profile_Registry maps to that Dataset layout.
5. WHERE the Dataset folder is present, THE Node_1 SHALL also have its false-alarm impact reported (not asserted) across all 1,000 genuine Dataset documents: count of documents by routing outcome and by Finding code.

### Requirement 11: Accuracy on Statement_E and Statement_F

**User Story:** As a project owner, I want the two labelled business statements read to the cent, so that later fabrication rules have correct data.

#### Acceptance Criteria

1. WHEN Statement_E and Statement_F are ingested, THE Node_1 SHALL return holder, UEN, account number, period 1 Apr 2026 to 30 Sep 2026 and file-level opening balance equal to the Answer_Key values.
2. WHEN Statement_E is ingested, THE Node_1 SHALL return, for each month, opening, total deposits, total withdrawals and closing equal to the `monthly_as_shown` values of the Answer_Key within 0.01.
3. WHEN Statement_F is ingested, THE Node_1 SHALL return, for each month, opening, total deposits, total withdrawals and closing equal to the `monthly` values of the Answer_Key within 0.01.
4. WHEN Statement_E is ingested, THE Node_1 SHALL return all 12 `inserted_rows` of the Answer_Key as parsed transactions, matched by month, date and amount.
5. WHEN Statement_E is ingested, THE Node_1 SHALL return transactions and SHALL NOT emit `DQ_NO_TXNS`.
6. WHEN Statement_E and Statement_F are ingested, THE Node_1 SHALL return the file-level closing balance of the last Section.
7. WHEN Statement_F is run through the Pipeline, THE Pipeline SHALL reach `AUTO_CLEAR` or SHALL produce no HIGH Finding, and the evaluation report SHALL list every other Finding that fires on Statement_F.
8. WHEN Statement_E is run through the Pipeline, THE evaluation report SHALL list the routing outcome, the risk score and every Finding code that fires.

### Requirement 12: Regression on the synthetic clean sample

**User Story:** As a maintainer, I want the known-good generator sample to stay clean, so that new layouts do not cause new false alarms.

#### Acceptance Criteria

1. WHEN the Synthetic_Clean_Sample is run through the Pipeline, THE Pipeline SHALL return `AUTO_CLEAR` with a risk score of about 1.
2. WHEN a tampered amount is introduced in a statement from the generator, THE Node_1 SHALL return the amount as printed and THE Node_3 SHALL still emit `FL_BALANCE_BREAK` for that row.

### Requirement 13: Evaluation script

**User Story:** As a project owner, I want one command that prints per-layout coverage, so that every new profile can be measured the same way.

#### Acceptance Criteria

1. THE Eval_Script SHALL run with the command `python -m fraud_detect.nodes.ingest_parse.eval_layouts`.
2. WHEN the Dataset folder is present, THE Eval_Script SHALL print, for each Dataset layout, the Profile chosen, the number of documents, the share of Transaction_Rows read, the share of documents whose replayed balance equals the printed closing balance, the share of documents whose header fields match, and the Finding codes that fired.
3. WHEN Statement_E, Statement_F and the Answer_Key are present, THE Eval_Script SHALL print the month-by-month comparison with the Answer_Key and the count of the 12 inserted rows found.
4. WHERE a command-line option `--sample N` is given, THE Eval_Script SHALL use the first N genuine documents per layout, with N defaulting to 20.
5. WHERE a command-line option `--all-genuine` is given, THE Eval_Script SHALL use all genuine documents and print the false-alarm table of Requirement 10.5.
6. IF an input folder or file is missing, THEN THE Eval_Script SHALL print a skipped message for it and exit with code 0.
7. THE Eval_Script SHALL print the same output for the same input files.

### Requirement 14: Test hygiene and external data

**User Story:** As a maintainer, I want tests to run anywhere, so that CI does not depend on files we are not allowed to commit.

#### Acceptance Criteria

1. THE repository SHALL contain no file from the Dataset, no copy of Statement_E, Statement_F or the Answer_Key.
2. WHEN a test needs a Dataset or labelled-file path that does not exist, THE test SHALL skip using `pytest.mark.skipif` on that path.
3. THE repository SHALL contain unit tests for the Date_Reader, Money_Reader, Header_Reader, Table_Reader, Direction_Resolver, Profile_Selector and Self_Check that use text and word-box fixtures built in code.
4. WHEN the external files are absent, THE test suite SHALL pass with no failures.
5. THE test fixtures SHALL describe fictional values only.

### Requirement 15: Documentation

**User Story:** As a future contributor, I want a short guide to adding a profile, so that new banks are added the same way.

#### Acceptance Criteria

1. THE README SHALL contain a section "How to add a layout profile" that lists the steps: build a word-box fixture, write a failing test, add the Profile, register it, run the Eval_Script.
2. THE README SHALL list the six Profiles and the layout each one covers.
3. THE README SHALL state that, with only synthetic layouts, success proves the parser design and does not prove that real Singapore bank statements will parse.
4. THE README SHALL state that real-bank profiles are added later when data is available safely.
5. THE README SHALL state that the external evaluation files are not stored in the repository.

### Requirement 16: Scope boundary

**User Story:** As a project owner, I want the stage boundary written down, so that deferred and finished work is not repeated or forgotten.

#### Acceptance Criteria

1. THE spec SHALL mark `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK` and `FL_TURNOVER_COST_MISMATCH` as "Deferred to a follow-up spec", and THE Node_3 SHALL gain no fabrication rule in this stage.
2. THE spec SHALL mark the `FL_PERIOD_LENGTH` inclusive-end fix as done, and THE tasks SHALL contain no task that re-implements `period_months()`.
3. THE spec SHALL treat retraining of ML models, OCR, web UIs and Node 4 weights as out of scope.
4. WHEN the design text for deferred rules is kept in `design.md`, THE design SHALL carry a visible "Deferred" note at the start of each deferred rule section.

### Requirement 17: Robustness

**User Story:** As a security reviewer, I want hostile or odd input handled safely, so that a crafted PDF cannot stall or crash screening.

#### Acceptance Criteria

1. WHEN a line of 10,000 characters or more is parsed, THE Parser SHALL finish within 1 second for that line.
2. IF a page contains no text, THEN THE Node_1 SHALL emit `DQ_NO_TEXT` and return without raising an error.
3. IF the Parser raises an unexpected error on one page, THEN THE Node_1 SHALL emit `DQ_LAYOUT_UNRECOGNISED` for that page and continue with the other pages.
4. WHEN a 12-page statement is ingested, THE Eval_Script SHALL report the elapsed time, with a target of 2 seconds on the development laptop.
