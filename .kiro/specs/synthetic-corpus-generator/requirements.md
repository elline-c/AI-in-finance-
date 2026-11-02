# Requirements Document

## Introduction

This feature adds a **synthetic corpus generator** to the prototype. It writes fictional but realistic business bank statements (6 to 12 months each) as PDF files in all six layouts that Node 1 can read, plants commonly detected fraud patterns in some of them, and records exactly what was planted. A separate read-only report then measures how the existing screening nodes behave on the corpus: whether Node 1 reads every row, how often genuine statements raise false alarms, which planted frauds are noticed, and which are not (the "blind spots").

The prototype will be shown to financial institutions. Real corporate statements cannot be obtained (data-leakage and PDPA concerns), so all data here is invented, reproducible from a seed, and clearly marked as synthetic. No external licensed data is used.

This is **Stage A**. Retraining Node 6a and Node 6b is a separate follow-up spec (Stage B, "ml-retraining") and is not part of this work. Nodes 1 to 4 are measured, never changed.

## Glossary

- **Corpus_Generator**: The new package `fraud_detect/corpus/` and its two commands (`build` and `verify`).
- **Simulator**: The part of Corpus_Generator that invents a fictional company and its month-by-month bank ledger.
- **Ledger**: The list of transactions the Simulator produces. Each row has a stable `row_id`. The Ledger is the truth from which every PDF is drawn.
- **Layout_Renderer**: The part of Corpus_Generator that draws a Ledger as a PDF in one named layout and returns a Row_Map.
- **Layout**: One of the six Node 1 profiles: `generator_v1`, `two_col_dmon`, `two_col_slash`, `two_col_dual_date`, `signed_amount_iso`, `two_col_ddmmyy_cr`.
- **Row_Map**: For each Ledger row, the page and the position of each printed cell in the PDF.
- **Tamperer**: The part of Corpus_Generator that applies recorded edits to a Ledger or to a finished PDF.
- **Tamper_Type**: One of the named fraud or tampering types in the catalog (`edited_amount`, `altered_date`, `broken_balance`, `font_mismatch`, `metadata_inconsistency`, `genuine_resaved`, `fabricated_receipts`, `cross_month_copy`, `turnover_inflation`, `structuring`, `sudden_large_deposit`, `shell_counterparty`, `rapid_pass_through`, `dormant_reactivation`, `period_gap`, `uen_mismatch`).
- **Tamper_Method**: How a tamper is applied: `overlay` (new value drawn over the old one), `rerender` (page redrawn from an edited Ledger), `content` (statement content changed), or `file` (only file data changed).
- **Genuine_Statement**: A generated PDF with no planted fraud. A Genuine_Statement that was only re-saved (`genuine_resaved`) is also genuine.
- **Fraud_Row**: A transaction row that a tamper added or whose value a tamper changed.
- **Manifest**: The JSON file describing one generated document (truth, label, rows, seeds).
- **Labels_File**: `labels.csv`, one line per document. **Rows_File**: `rows.csv`, one line per printed transaction row.
- **Corpus_Builder**: The part that plans and writes a whole corpus (`CorpusBuilder` and the `build` command).
- **Corpus_Folder**: The output folder, by default `data/corpus/`, which is ignored by git.
- **Split**: The group a document belongs to: `train`, `validation`, `test`, or `test_unseen_layout`.
- **Held_Out_Layout**: A Layout chosen with `--hold-out-layout`, whose documents all go to `test_unseen_layout`.
- **Company_Registration_Scope**: The context manager that adds the generated fictional companies to the in-memory reference dictionaries for the duration of a measurement and then restores them.
- **Verification_Report**: The read-only `verify` command that runs the corpus through Nodes 1 to 4 and prints measurements.
- **Node_1 / Node_2 / Node_3 / Node_4**: The existing ingestion, visual-forensics, financial-logic and risk-synthesis nodes.
- **Master_Seed**: The single integer that determines a whole corpus.
- **Document_Seed**: The seed of one document, derived from the Master_Seed.
- **Generator_Version**: The version string written into every Manifest.
- **Mini_Corpus**: The small in-code corpus built inside tests (no external files).
- **Blind_Spot**: A Tamper_Type, or a variant of one, that none of Nodes 1 to 4 flags in the measurement.
- **Fixture_Geometry**: The layout facts (column right edges, header words, footer lines, date styles) recorded in `tests/ingest_parse/fixtures.py` and in `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/design.md`.

## Requirements

### Requirement 1: Simulate a business-like fictional company

**User Story:** As a project owner, I want a fictional company with realistic monthly activity, so that generated genuine statements look like real business statements and can serve as a false-alarm test set.

#### Acceptance Criteria

1. WHEN the Simulator is given a start month and a number of months from 6 to 12, THE Simulator SHALL produce a Ledger covering exactly that number of consecutive calendar months.
2. THE Simulator SHALL give each fictional company a name in the form "<invented words> PTE. LTD.", a fictional UEN-style identifier, an address, an account number, 5 to 25 customers and a set of suppliers, all taken from invented word lists inside the code.
3. THE Simulator SHALL produce customer receipts whose descriptions contain a rail word (`FAST FROM`, `PAYNOW FROM` or `GIRO`), a customer name and an invoice reference of the form `INV` followed by at least 3 digits.
4. WHERE the invoice disorder setting is 0, THE Simulator SHALL assign invoice numbers so that the number never decreases as the receipt date increases.
5. THE Simulator SHALL produce a monthly PAYROLL batch, a monthly CPF CONTRIBUTION and a monthly RENT payment, each within 10% of its own median across the statement's months.
6. THE Simulator SHALL produce a GST PAYMENT in each calendar quarter of the statement, sized from turnover.
7. THE Simulator SHALL produce supplier payments (`FAST TO <supplier>`), utility payments, card payments, a monthly service charge and small interest credits, each with an amount that has two decimals.
8. WHERE the instalment rate setting is above 0, THE Simulator SHALL pay a share of invoices in two parts, with the second part sometimes in the next month and sometimes with an amount identical to the first part, and SHALL record each such invoice as a genuine hard case in the Manifest.
9. THE Simulator SHALL apply a per-company growth rate and a monthly seasonality pattern to turnover, both inside configured limits.
10. THE Simulator SHALL choose the opening balance so that the replayed balance never falls below zero, and SHALL set each month's opening balance equal to the previous month's closing balance.
11. THE Simulator SHALL place receipts, card payments, supplier payments and GIRO payments on weekdays.
12. WHERE a Layout limits the number of rows per month, THE Simulator SHALL keep each month within that limit by dropping small card payments first and SHALL record the trim in the Manifest.

### Requirement 2: Draw statements in all six layouts

**User Story:** As a project owner, I want the same kind of statement in each of the six layouts Node 1 can read, so that the screening can be measured across all layouts.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL provide one Layout_Renderer for each of the six Layouts.
2. WHEN a Layout_Renderer is given a Ledger, THE Layout_Renderer SHALL produce one multi-page PDF with one monthly section per Ledger month in date order.
3. THE Layout_Renderer SHALL print every Ledger row exactly once, in Ledger order, with the date style, amount style and balance style of its Layout.
4. THE Layout_Renderer SHALL place right-aligned amount and balance cells so that the right edge of each cell is within 2 points of the right edge recorded for that column in the Fixture_Geometry.
5. WHEN a monthly section has more rows than fit on a page, THE Layout_Renderer SHALL continue it on a following page that repeats the table header and prints no period line and no brought-forward row.
6. THE Layout_Renderer for `two_col_dmon` SHALL print "Page 1 of 1" on every monthly page, print rows as `d Mon` without a year, and print amounts with the `S$` marker.
7. THE Layout_Renderer for `two_col_dual_date` SHALL print a transaction date and a value date on every row.
8. THE Layout_Renderer for `signed_amount_iso` SHALL print one signed amount column using `+SGD` for credits and `-SGD` for debits.
9. THE Layout_Renderer for `two_col_ddmmyy_cr` SHALL print the suffix `CR` after every balance except in the summary box.
10. THE Layout_Renderer SHALL print in each monthly section a summary box (or, for `generator_v1`, the opening-balance line and closing-balance line) whose opening, total withdrawals, total deposits and closing equal the values replayed from the Ledger rows of that month.
11. THE Layout_Renderer SHALL print on every page a footer in the generator's own words stating that the document is a fictional specimen of synthetic data.
12. THE Layout_Renderer SHALL return a Row_Map that gives, for every Ledger row, the page and the position of its date, description, amount and balance cells.
13. WHERE the Layout_Renderer continues a month onto a second page for a Layout that does not do so in the real samples, THE Layout_Renderer SHALL record that fact in the Manifest as `spill_not_in_real_samples`.
14. THE Layout_Renderer SHALL vary the bank name, address wording, producer string, creation time of day and row spacing within configured limits, using the Document_Seed.

### Requirement 3: Node 1 reads every generated genuine row

**User Story:** As a project owner, I want proof that the screening's first step reads the generated statements completely, so that later measurements are about fraud and not about unreadable files.

#### Acceptance Criteria

1. WHEN a Genuine_Statement of 6 months in any Layout is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that Node_1 reads 100% of the generated transaction rows with the generated date, amount and balance.
2. WHEN a Genuine_Statement of 12 months in any Layout is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that Node_1 reads 100% of the generated transaction rows with the generated date, amount and balance.
3. WHEN a Genuine_Statement is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that the balance replayed from the rows read equals the printed closing balance.
4. WHEN a Genuine_Statement is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that the layout profile chosen by Node_1 equals the Layout that was drawn.
5. WHEN a Genuine_Statement is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that the holder name, UEN, and the first period start and last period end read by Node_1 equal the generated values.
6. WHEN a Genuine_Statement is ingested by Node_1, THE Corpus_Generator test suite SHALL assert that Node_1 raises no data-quality finding of severity MEDIUM or HIGH.
7. THE Corpus_Generator test suite SHALL run requirements 3.1 to 3.6 on a Mini_Corpus built in memory or in a temporary folder, without any external file.

### Requirement 4: Ground truth comes from the Ledger and recorded edits

**User Story:** As a project owner, I want the correct answer for every file to be known without reading the PDF back, so that measurements cannot grade themselves.

#### Acceptance Criteria

1. THE Corpus_Builder SHALL draw every PDF from a Ledger.
2. WHEN the Corpus_Builder creates a tampered document, THE Tamperer SHALL derive it from the Genuine Ledger by explicit edit operations and SHALL record each operation (operation name, `row_id`, field, original value, new value) in the Manifest.
3. THE Manifest SHALL list every printed transaction row with its `row_id`, its 1-based `row_index` in print order, page, month, date, description, amount, printed balance and `is_fraud_row`.
4. WHEN a document is a Genuine_Statement, THE Manifest SHALL contain no Tamper_Type, SHALL mark no row as a Fraud_Row, and SHALL set `is_tampered` to false.
5. THE Manifest SHALL contain a SHA-256 hash of the text layer of the PDF.

### Requirement 5: Document-level tamper types

**User Story:** As a project owner, I want the usual kinds of document tampering planted and labelled, so that Node 2 and Node 3 can be tested on them.

#### Acceptance Criteria

1. WHEN the Tamper_Type is `edited_amount`, THE Tamperer SHALL change exactly one transaction amount, by the `overlay` method or the `rerender` method, and SHALL record the original and new value.
2. WHEN the Tamper_Type is `altered_date`, THE Tamperer SHALL change exactly one transaction date, by the `overlay` method or the `rerender` method, and SHALL set `date_inconsistent` to true if the new date breaks date order or lies outside its month.
3. WHEN the Tamper_Type is `broken_balance`, THE Tamperer SHALL change exactly one printed balance without changing any printed amount and SHALL set `arithmetic_consistent` to false.
4. WHEN the Tamper_Type is `font_mismatch`, THE Tamperer SHALL draw the changed value in the built-in font `Courier` and SHALL set `font_inconsistent` to true.
5. WHEN the Tamper_Type is `metadata_inconsistency`, THE Tamperer SHALL set the PDF Producer to an editing-tool name, set the modification date later than the creation date, leave all page content unchanged, and set `metadata_altered` to true.
6. WHERE the metadata variant "creation before period" is chosen, THE Tamperer SHALL set the creation date earlier than the first period start.
7. WHEN the Tamper_Type is `genuine_resaved`, THE Tamperer SHALL set the Producer to a print or export style string, set the modification date later than the creation date, leave all page content unchanged, set `genuine_resaved` to true and set `is_tampered` to false.
8. THE Tamperer SHALL set `arithmetic_consistent` in the label to true if and only if every printed balance equals the balance replayed from the printed amounts of that document.

### Requirement 6: Overlay tampering keeps both values in the text layer

**User Story:** As a project owner, I want overlay tampering to behave like the real thing, so that the known weakness of Node 1 on overlaid rows is measured honestly.

#### Acceptance Criteria

1. WHEN the Tamper_Method is `overlay`, THE Tamperer SHALL create a one-page overlay with reportlab and merge it onto the original page with pypdf at the position given by the Row_Map.
2. WHEN the Tamper_Method is `overlay`, THE Tamperer SHALL leave the text of the original value and the text of the new value both extractable from the page and SHALL set `hidden_text_layer` to true.
3. WHERE the overlay includes a background box, THE Tamperer SHALL set `overlay_bg_matched` to true, and WHERE the overlay has no background box, THE Tamperer SHALL set it to false.
4. WHEN the Tamper_Method is `overlay`, THE Tamperer SHALL leave every text element outside the overlaid cell unchanged.
5. WHEN the Tamper_Method is `rerender`, THE Layout_Renderer SHALL draw the edited Ledger so that only the new value appears in the text layer.
6. THE Manifest of an overlay tamper SHALL state that Node_1 might drop the overlaid row and SHALL list the overlaid row as an affected row.

### Requirement 7: Fabrication and content fraud with consistent arithmetic

**User Story:** As a project owner, I want fabricated receipts and inflated turnover whose numbers still add up, so that Node 3's content rules are tested the way the real tampered example is built.

#### Acceptance Criteria

1. WHEN the Tamper_Type is `fabricated_receipts`, THE Tamperer SHALL insert 6 to 12 credit rows, each copying the customer and invoice reference of an existing genuine receipt, with a different date and a different amount, in the same calendar month as the copied receipt.
2. WHEN the Tamper_Type is `cross_month_copy`, THE Tamperer SHALL insert credit rows that copy the customer and invoice reference of an existing genuine receipt with a different amount, in a calendar month different from the copied receipt.
3. WHEN the Tamper_Type is `turnover_inflation`, THE Tamperer SHALL insert credit rows with new invoice references in at least 2 months, and SHALL leave the payroll, CPF and rent rows unchanged.
4. WHEN a content-fraud tamper inserts rows, THE Tamperer SHALL recompute every following printed balance, every affected summary box and every following month's opening balance, and SHALL set `arithmetic_consistent` to true.
5. WHEN a content-fraud tamper inserts rows, THE Tamperer SHALL set `is_fraud_row` to true for each inserted row and false for every other row.
6. THE Manifest of a content-fraud tamper SHALL record, for each inserted row, the `row_id` of the genuine receipt it copies, when it copies one.

### Requirement 8: Behavioural fraud typologies

**User Story:** As a project owner, I want the commonly detected behavioural fraud patterns planted and labelled, so that detection and blind spots can be reported by pattern.

#### Acceptance Criteria

1. WHEN the Tamper_Type is `structuring`, THE Tamperer SHALL insert 3 to 5 credits between 9,500.00 and 9,999.99 within a span of 14 days, and WHERE the slow variant is chosen, THE Tamperer SHALL spread them over more than 30 days.
2. WHEN the Tamper_Type is `sudden_large_deposit`, THE Tamperer SHALL insert one credit larger than 3 times the company's average monthly credit, and WHERE the moderate variant is chosen, THE Tamperer SHALL insert one credit between 1.5 and 2.5 times that average.
3. WHEN the Tamper_Type is `shell_counterparty`, THE Tamperer SHALL insert one credit of 50,000.00 or more whose description contains a shell-like name such as `UNKNOWN TRADING LLC`.
4. WHEN the Tamper_Type is `rapid_pass_through`, THE Tamperer SHALL insert 2 to 4 pairs of a credit followed within 1 to 3 days by a debit to a different party for 95% to 100% of the credit.
5. WHEN the Tamper_Type is `dormant_reactivation`, THE Tamperer SHALL remove all business activity except bank charges and interest in the first 3 to 5 months and SHALL keep the later months' activity.
6. WHEN the Tamper_Type is `period_gap`, THE Tamperer SHALL remove one middle monthly section and SHALL list the missing month in the Manifest as an affected month.
7. WHEN the Tamper_Type is `uen_mismatch`, THE Tamperer SHALL print a holder name that differs from the registered name of the company and SHALL keep the registered name in the companies file.
8. WHEN a behavioural tamper inserts, moves or removes rows, THE Tamperer SHALL recompute printed balances so that every monthly section is internally consistent, and SHALL set `arithmetic_consistent` to true.

### Requirement 9: Labels at document and row level

**User Story:** As a project owner, I want one clear label file and one row-level file, so that detection and later machine learning can use them directly.

#### Acceptance Criteria

1. THE Corpus_Builder SHALL write one Manifest per document into the Corpus_Folder.
2. THE Corpus_Builder SHALL write a Labels_File with the columns `doc_id`, `file`, `split`, `layout`, `company_id`, `doc_seed`, `n_months`, `first_month`, `last_month`, `n_pages`, `n_rows`, `is_tampered`, `tamper_types`, `tamper_method`, `arithmetic_consistent`, `genuine_resaved`, `metadata_altered`, `font_inconsistent`, `date_inconsistent`, `overlay_bg_matched`, `hidden_text_layer`, `n_affected_rows`, `spill_not_in_real_samples`, `genuine_hard_cases` and `generator_version`.
3. THE Corpus_Builder SHALL write a Rows_File with the columns `doc_id`, `row_index`, `row_id`, `page`, `month`, `date`, `amount`, `balance`, `kind`, `is_fraud_row` and `fraud_type`.
4. THE Manifest of a tampered document SHALL record the Layout, the months, the Tamper_Types, the Tamper_Method, the affected rows (row index, page, month, field, original value, new value where applicable) and the `arithmetic_consistent` flag.
5. THE Manifest of a tampered document SHALL record a descriptive list of the Nodes that should in principle see the tamper, and SHALL state that the list is descriptive and not a target.
6. FOR ALL documents, every `row_id` named in `affected_rows` SHALL exist in the `rows` of the same Manifest, and every month named in `affected_months` SHALL be a month of the statement or a recorded missing month.
7. FOR ALL documents, the count of rows with `is_fraud_row` true in the Rows_File SHALL equal `n_affected_rows` for that document in the Labels_File.

### Requirement 10: Reproducibility

**User Story:** As a project owner, I want the same settings to rebuild the same corpus, so that results can be repeated and checked by others.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL derive every random choice from a `random.Random` object seeded from the Master_Seed and a fixed list of identifying parts, using a SHA-256 based stable seed and never the global random state or Python's built-in `hash`.
2. WHEN the Corpus_Builder is run twice with the same configuration and Master_Seed, THE Corpus_Builder SHALL write Manifests, a Labels_File, a Rows_File and a companies file that are byte-for-byte equal.
3. WHEN the Corpus_Builder is run twice with the same configuration and Master_Seed, THE Corpus_Builder SHALL write PDFs whose extracted words, positions and metadata are equal.
4. THE Corpus_Builder SHALL write the Generator_Version into every Manifest and into the corpus manifest.
5. WHEN the Master_Seed differs, THE Corpus_Builder SHALL produce at least one differing company name among the documents of a corpus of 6 or more documents.
6. WHEN two documents are planned, THE Corpus_Builder SHALL derive each Document_Seed so that it does not depend on the order in which other documents are built.

### Requirement 11: Splits and leakage rules

**User Story:** As a project owner, I want clean training and testing groups, so that future results are not inflated by near-copies of the same company.

#### Acceptance Criteria

1. THE Corpus_Builder SHALL assign every document to exactly one Split, and SHALL write the Split in the Labels_File.
2. THE Corpus_Builder SHALL assign all documents of one simulated company to the same Split.
3. THE Corpus_Builder SHALL choose the Split of a company from a hash of the Master_Seed and the company id, and from the configured split fractions (default 70%, 15%, 15%), so that the assignment does not depend on build order.
4. WHERE a Held_Out_Layout is given, THE Corpus_Builder SHALL assign every document of that Layout to `test_unseen_layout` and SHALL assign no document of that Layout to `train`, `validation` or `test`.
5. WHERE a Held_Out_Layout is given, THE Corpus_Builder SHALL assign no document of any other Layout to `test_unseen_layout`.
6. THE Corpus_Builder SHALL write into the corpus manifest the count of documents per Split, per Layout and per Tamper_Type within each Split.
7. IF a Tamper_Type is absent from a Split that is not empty, THEN THE Corpus_Builder SHALL list that gap in the corpus manifest and print a warning.
8. IF the configured split fractions do not sum to 1 within 0.001 or any fraction is negative, THEN THE Corpus_Builder SHALL refuse to start and SHALL print the reason.

### Requirement 12: Build command, output hygiene and synthetic marking

**User Story:** As a project owner, I want a simple command that builds a corpus into an ignored folder and marks everything as synthetic, so that no fake data can be mistaken for real data or committed by accident.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL provide the command `python -m fraud_detect.corpus.build` with the options `--out`, `--per-layout-genuine`, `--per-layout-tampered`, `--months-min`, `--months-max`, `--seed`, `--hold-out-layout`, `--tamper-mix` and `--force`.
2. WHERE `--months-min` and `--months-max` are not given, THE Corpus_Builder SHALL choose the number of months of each document at random from 6 to 12 using the Document_Seed.
3. IF `--months-min` is below 1, or `--months-max` is above 24, or `--months-min` is above `--months-max`, THEN THE Corpus_Builder SHALL refuse to start and SHALL print the reason.
4. IF the output folder already contains files and `--force` is not given, THEN THE Corpus_Builder SHALL refuse to write and SHALL exit with a non-zero code.
5. IF the output folder lies inside `fraud_detect/`, `tests/` or `.kiro/`, THEN THE Corpus_Builder SHALL refuse to write.
6. THE Corpus_Builder SHALL write by default into `data/corpus/`, which `.gitignore` already ignores.
7. THE Corpus_Builder SHALL write a README file and a corpus manifest into the Corpus_Folder, each containing a "SYNTHETIC DATA" statement, the Generator_Version, the Master_Seed, the command line and the counts.
8. THE Layout_Renderer SHALL set the PDF Title and Subject so that they state the document is synthetic.
9. IF a document fails to build, THEN THE Corpus_Builder SHALL record its id and the exception type in the corpus manifest, continue with the other documents, and exit with a non-zero code at the end.
10. WHEN a corpus is built with `--per-layout-genuine 2 --per-layout-tampered 2`, THE Corpus_Builder SHALL produce 12 genuine documents and 12 tampered or hard-negative documents, two of each per Layout.
11. THE Corpus_Builder SHALL allocate tamper types to the tampered documents according to `--tamper-mix`, and WHERE it is not given, SHALL spread them evenly over all Tamper_Types.

### Requirement 13: Register the fictional companies without touching production data

**User Story:** As a project owner, I want the measurement to treat generated companies as registered, so that false alarms are not drowned out by "UEN not found".

#### Acceptance Criteria

1. THE Corpus_Builder SHALL write a companies file listing, for each generated company, its UEN, its registered name, and a baseline of average monthly credit and average monthly debit computed from its genuine Ledger.
2. WHILE a Company_Registration_Scope is active, THE Company_Registration_Scope SHALL make `registry.lookup_entity` find each listed UEN under its registered name, and SHALL make Node_3 find the listed baseline for that UEN.
3. WHEN a Company_Registration_Scope ends, THE Company_Registration_Scope SHALL restore `ACRA_REGISTRY` and `CASHFLOW_BASELINE` to exactly their earlier contents.
4. IF an error occurs inside a Company_Registration_Scope, THEN THE Company_Registration_Scope SHALL restore `ACRA_REGISTRY` and `CASHFLOW_BASELINE` before the error leaves the scope.
5. THE Company_Registration_Scope SHALL leave the source files `fraud_detect/reference_data.py` and `fraud_detect/registry.py` unchanged.
6. THE Company_Registration_Scope SHALL leave the live registry lookup switched off.
7. IF a listed UEN already exists in `ACRA_REGISTRY` with a different name, THEN THE Company_Registration_Scope SHALL refuse to register it and SHALL report the clash.

### Requirement 14: Verification report for Node 1 (asserted in tests, reported in the command)

**User Story:** As a project owner, I want to see how completely Node 1 reads each layout, so that I can trust later numbers.

#### Acceptance Criteria

1. THE Verification_Report SHALL be started with `python -m fraud_detect.corpus.verify --corpus DIR`.
2. THE Verification_Report SHALL print, for each Layout, the number of documents, the share of generated transaction rows that Node_1 read, the share of documents whose replayed balance equals the printed closing balance, the share of documents whose header fields match, the profile chosen by Node_1 and the Node_1 finding codes seen.
3. THE Verification_Report SHALL print the share of rows read separately for Genuine_Statements and for tampered documents, and SHALL list the number of overlay rows that Node_1 dropped.
4. WHERE the corpus folder is missing, THE Verification_Report SHALL print a message that starts with "skipped" and SHALL exit with code 0.
5. THE Verification_Report SHALL print an "elapsed" line.
6. THE Verification_Report SHALL write no file unless the option `--json` or `--markdown` names one.
7. THE Verification_Report SHALL screen documents in memory without a case store, so that `data/cases.json` and the audit chain are not written.
8. THE Verification_Report SHALL accept injected `ingest` and `screen` functions, so that its helper functions can be tested without real PDFs.
9. WHERE `--sample N` is given, THE Verification_Report SHALL measure at most N documents per Layout, chosen in a fixed order.

### Requirement 15: Measure Nodes 2, 3 and 4, with a blind-spot table

**User Story:** As a project owner, I want honest numbers on false alarms and detection, so that I can describe what the prototype does and does not catch.

#### Acceptance Criteria

1. WHILE a Company_Registration_Scope is active, THE Verification_Report SHALL run Node_2 and Node_3 and Node_4 on each document.
2. THE Verification_Report SHALL print, for Genuine_Statements, the share routed to AUTO_CLEAR, the count of each finding code, and, separately, the same figures for `genuine_resaved` documents.
3. THE Verification_Report SHALL print, for each Tamper_Type and Tamper_Method, the number of documents, the share in which at least one finding of severity MEDIUM or HIGH appeared, the share not routed to AUTO_CLEAR, and the finding codes seen.
4. THE Verification_Report SHALL print, for each Tamper_Type that has a descriptive expected code, the share of documents in which that code appeared.
5. THE Verification_Report SHALL print a blind-spot table listing each Tamper_Type, method and variant whose share of documents with a MEDIUM or HIGH finding is 0%, or whose share not routed to AUTO_CLEAR equals or is below the share for Genuine_Statements.
6. THE Verification_Report SHALL print the measured figures as reported values and SHALL NOT fail or change its exit code because of a detection rate or a false-alarm rate.
7. THE Verification_Report SHALL print the follow-up suggestions that follow from the measurement, without implementing any of them.
8. THE Verification_Report SHALL print the figures per Split, so that results on the Held_Out_Layout are visible.
9. IF Node_2, Node_3 or Node_4 raises an exception for a document, THEN THE Verification_Report SHALL count that document as "screening error", name the exception type, and continue.

### Requirement 16: Written results and documentation

**User Story:** As a project owner, I want the measured numbers written down in the project, so that I can quote them with their limits.

#### Acceptance Criteria

1. THE Corpus_Generator work SHALL include a file `.kiro/specs/synthetic-corpus-generator/evaluation-results.md` that contains the measured figures of one stated run (command, seed, counts, Generator_Version, date, elapsed time).
2. THE evaluation-results file SHALL contain the Node 1 table, the false-alarm table for Genuine_Statements, the detection table by Tamper_Type and Tamper_Method, the blind-spot table, the time to render a 12-month statement, and the follow-up suggestions.
3. THE evaluation-results file SHALL state that the figures are demonstrations on synthetic data and not real-world accuracy.
4. THE project README (`FRAUD_DETECT_README.md`) SHALL contain a section "Synthetic corpus generator" that explains in plain language what the generator is for, how to build and measure a corpus, where the files go, the fraud types, the splits, and the honest limits.
5. THE README section SHALL state that the corpus is never committed, that the generator is, and that `statement_E` and `statement_F` are never used for training.
6. THE README section SHALL describe Stage B (ML retraining) as a separate follow-up and SHALL list its prerequisites.

### Requirement 17: Backward compatibility, dependencies and performance

**User Story:** As a project owner, I want the new feature to leave everything else working, so that the prototype stays stable.

#### Acceptance Criteria

1. THE Corpus_Generator work SHALL leave all 1155 existing passing tests passing and the one existing skipped test skipped.
2. THE Corpus_Generator work SHALL leave the public functions of `fraud_detect/synthetic.py` and their behaviour unchanged.
3. THE Corpus_Generator work SHALL leave the source files of Nodes 1 to 4, the registry and the reference data unchanged.
4. THE Corpus_Generator work SHALL add no new runtime dependency.
5. WHEN a 12-month Genuine_Statement is rendered on the development laptop, THE Layout_Renderer SHALL finish in a time that the evaluation-results file records, and the test suite SHALL not assert a time limit tighter than 30 seconds for that case.
6. THE new test folder `tests/corpus/` SHALL run to completion in no more than about 90 seconds on the development laptop, which the final task records.
7. THE Corpus_Generator work SHALL leave `git status` free of any PDF, `labels.csv`, corpus file or external dataset file.

### Requirement 18: Clean-room and no external data

**User Story:** As a project owner, I want a clear guarantee that no unlicensed material is copied, so that the generator can be kept and shown safely.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL contain no code, text, name list or file copied from the external forgery dataset or its script `generate_documents.py`.
2. THE Corpus_Generator SHALL write each Layout_Renderer from the Fixture_Geometry and the layout notes already in the repository.
3. THE Corpus_Generator SHALL contain no reference to the paths `~/Downloads`, `statement_forgery_dataset`, `statement_E.pdf`, `statement_F.pdf` or `ANSWER_KEY` in its package source.
4. THE Corpus_Generator tests SHALL contain a test that scans the package source for the strings in 18.3 and fails if any is found.
5. THE Corpus_Generator SHALL use only built-in reportlab fonts, so that no font file is added.
6. THE Corpus_Generator tests SHALL contain no dependence on any file outside the repository.

### Requirement 19: Honest limits are stated

**User Story:** As a project owner, I want the limits written next to the results, so that the prototype is not oversold.

#### Acceptance Criteria

1. THE README section and the evaluation-results file SHALL state that the data is synthetic and that results are demonstrations and not real-world accuracy.
2. THE README section SHALL state the circularity risk (a model can learn the generator's habits) and the mitigations: a Held_Out_Layout, multiple seeds, randomised styling, no tuning of detectors to the generator, and no training on `statement_E` or `statement_F`.
3. THE README section SHALL state that only the six known layouts are covered and that real-bank layouts are not.
4. THE README section SHALL state the simplifications: invoice numbers follow receipt order, the company model is simple, and the fraud patterns are textbook patterns.
5. THE README section SHALL state that the "should be seen by" column is descriptive and not a target.

### Requirement 20: Build-time UEN collision check

**User Story:** As a project owner, I want every invented company number to be checked against real company numbers while the corpus is built, so that no generated statement can be mistaken for a statement of a real Singapore company.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL ensure that the UEN of every fictional company equals no UEN in the real Singapore company registry published by ACRA on data.gov.sg, no UEN in `fraud_detect.reference_data.ACRA_REGISTRY`, and no other UEN in the same corpus.
2. WHEN a drawn UEN equals a UEN named in 20.1, THE Simulator SHALL draw a new UEN from the same seeded stream of the company and SHALL keep every other part of the company profile equal to what it would have been without the clash.
3. FOR ALL equal Master_Seed values and equal registry snapshots, THE Corpus_Builder SHALL write the same UEN for every company, including every redraw (same seed and same snapshot give the same corpus).
4. WHERE the option `--uen-snapshot PATH` is given, THE Corpus_Builder SHALL read the UENs of the file at PATH (a text file with one UEN per line, a CSV file with a column named `uen`, or a JSON list of UENs) and SHALL treat each of them as a real UEN.
5. WHERE the option `--check-live-uen` is given, THE Corpus_Builder SHALL ask the live data.gov.sg lookup in `fraud_detect/registry.py` about each drawn UEN.
6. WHEN neither `--uen-snapshot` nor `--check-live-uen` is given, THE Corpus_Builder SHALL check each UEN only against `ACRA_REGISTRY` and the other UENs of the corpus, SHALL make no network request, and SHALL print a warning that collisions with the real registry were not checked.
7. THE Corpus_Builder SHALL leave the live lookup switched off unless `--check-live-uen` is given.
8. IF the snapshot file named by `--uen-snapshot` is missing, unreadable, empty or in an unknown format, THEN THE Corpus_Builder SHALL print the reason, exit with code 2 and write no file.
9. IF the live lookup cannot answer for a UEN while `--check-live-uen` is given, THEN THE Corpus_Builder SHALL print the reason, exit with code 2 and write no file.
10. THE Corpus_Builder SHALL record in `corpus_manifest.json` and in `README.txt` the check source used (in-repo registry only, snapshot, live, or snapshot and live), the number of UENs checked, the number of redraws and a statement of whether the real registry was checked.
11. THE Corpus_Builder SHALL store no name, financial figure or other detail of any real company in the Corpus_Folder; only the fact of the check, its source, the file name and hash of the snapshot, and counts.
12. THE Corpus_Generator work SHALL leave `fraud_detect/registry.py`, `fraud_detect/reference_data.py` and Nodes 1 to 4 unchanged and SHALL use only the public function `lookup_entity` of the registry.
13. THE Corpus_Generator tests SHALL make no network request.
