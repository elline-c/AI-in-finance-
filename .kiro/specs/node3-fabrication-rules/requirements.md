# Requirements Document

## Introduction

Some forgeries keep every number consistent, so the arithmetic checks in Node 3 cannot see them. `statement_E.pdf` is one: twelve fake incoming receipts were inserted and every balance was recalculated. The giveaways are in the content. Each fake receipt repeats the customer and invoice number of a real receipt, the invoice numbers fall out of date order, and money in rises while payroll, CPF and rent stay flat.

This spec adds three rules to Node 3 to catch that pattern (`FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH`), a safety wrapper so a faulty rule never stops the pipeline (`DQ_RULE_SKIPPED`), and a small, separable calibration of the existing `FL_STRUCTURING` rule, which currently gives the genuine `statement_F` a false HIGH finding. It also asks for two report-only investigations and an evaluation with honest limits. The design is in `design.md`; it is adapted from the deferred Component 8 of `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/design.md`, whose own requirements and tasks (Stage 1, Node 1) are finished and are not edited.

**Measured starting point.** After Node 1 was upgraded, `statement_F` (genuine) routes `HUMAN_REVIEW` with risk 40 and `statement_E` (tampered) routes `HUMAN_REVIEW` with risk 26, so the tampered file scores lower than the genuine one. No metadata finding fires on `statement_E`.

**Evidence limitation (stated honestly).** Only two labelled files exist. Thresholds are set from first principles, except the Rule C jump ratio, which was informed by both files (see the decision record in `design.md`). Results are indicative, not proof. A forger who invents a brand-new invoice number evades `FL_DUP_INVOICE_REF`. Genuine instalment payments can look like duplicates. Passing these requirements does not show that real Singapore bank statements behave the same way.

**External files.** `~/Downloads/statement_E.pdf`, `statement_F.pdf`, `ANSWER_KEY_6M_v2.json` and `~/Downloads/statement_forgery_dataset/` are never committed. Tests that need them skip when they are missing. Tests with in-code transaction fixtures always run.

**Out of scope.** Retraining Node 6a and 6b, extending the generator to 6 to 12 month statements with fraud patterns, any Node 1 change, UIs, Node 4 weights and thresholds, OCR, and `fraud_detect/store.py`.

## Glossary

- **Node_3**: The Financial Logic stage, `fraud_detect/nodes/financial_logic.py`, whose findings carry the agent label `Node3:FinancialLogic`.
- **Fabrication_Module**: The new module `fraud_detect/nodes/fabrication.py`, called by Node_3.
- **Fabrication_Config**: The immutable settings object `FabricationConfig` that holds every threshold of the Fabrication_Module, with `DEFAULT_CONFIG` as its default instance.
- **Dup_Rule**: The rule that reports `FL_DUP_INVOICE_REF`.
- **Sequence_Rule**: The rule that reports `FL_INVOICE_SEQUENCE_BREAK`.
- **Turnover_Rule**: The rule that reports `FL_TURNOVER_COST_MISMATCH`.
- **Structuring_Rule**: The existing Node_3 check that reports `FL_STRUCTURING`.
- **Credit**: A transaction whose amount is greater than zero.
- **Invoice_Reference**: A prefix and number read from a Credit description by a pattern in the Fabrication_Config, normalised so that leading zeros are removed (`INV0042` and `inv-042` are the same Invoice_Reference). A reference needs at least three digits.
- **Counterparty**: The Credit description with the Invoice_Reference and any leading payment rail removed, upper-cased, with repeated spaces collapsed.
- **Repeated_Reference_Group**: Two or more Credits that share one Counterparty and one Invoice_Reference.
- **Order_Break_Row**: A Credit with an Invoice_Reference that lies outside the longest run of non-decreasing reference numbers when the dated Credits are sorted by date and then by reference number.
- **Full_Month**: A calendar month that the declared statement period covers from its first day to its last day; when no period is declared, any calendar month that holds a dated transaction.
- **Turnover**: The sum of Credits in a Full_Month, leaving out Credits whose description holds a word listed in `turnover_exclude_keywords`.
- **Fixed_Costs**: The sum of the absolute values of debits in a Full_Month whose description holds a word listed in `fixed_cost_keywords`.
- **Jump_Month**: A Full_Month whose Turnover is more than `credit_jump_ratio` times the median Turnover of the other Full_Months and whose Fixed_Costs are within `fixed_cost_stability_pct` of the median Fixed_Costs.
- **Near_Threshold_Credit**: A Credit from 9,500 up to but not including 10,000.
- **Risk_Synthesis**: Node 4, `fraud_detect/nodes/risk_synthesis.py`, which sums finding weights into the risk score and routes the case.
- **Labelled_Files**: `statement_E.pdf` (tampered), `statement_F.pdf` (genuine) and `ANSWER_KEY_6M_v2.json`.
- **Forgery_Dataset**: The 2,000 PDFs in `~/Downloads/statement_forgery_dataset/`, of which 1,000 are genuine (`is_tampered = 0` in `labels.csv`).
- **Investigation**: A read-only task that reads code and files, changes no behaviour, and writes its result to `.kiro/specs/node3-fabrication-rules/investigation.md`.
- **Evaluation_Script**: The read-only report script `fraud_detect/nodes/fabrication_eval.py`.

## Requirements

### Requirement 1: Configurable thresholds

**User Story:** As a developer, I want every fabrication threshold in one named, validated settings object, so that I can tune the rules without editing rule code.

#### Acceptance Criteria

1. THE Fabrication_Module SHALL define Fabrication_Config as an immutable object that holds the invoice patterns, rail prefixes, escalation count, rule weights, instalment limits, sequence limits, fixed-cost and exclusion keywords, jump ratio, stability percentage and minimum-month limits.
2. THE Fabrication_Module SHALL expose `DEFAULT_CONFIG` as the instance used when the caller supplies no Fabrication_Config.
3. WHEN a Fabrication_Config is built with a `credit_jump_ratio` of 1 or less, a `fixed_cost_stability_pct` outside 0 up to but not including 1, a count limit below 1, or a negative weight, THE Fabrication_Config SHALL raise `ValueError`.
4. WHEN a caller passes a Fabrication_Config to `analyze_fabrication` or to `financial_logic.analyze`, THE Fabrication_Module SHALL apply that configuration instead of `DEFAULT_CONFIG`.
5. THE Fabrication_Module SHALL read every threshold from the Fabrication_Config and SHALL contain no threshold written directly inside a rule.

### Requirement 2: Reading invoice references and counterparties

**User Story:** As a reviewer, I want references and customer names read the same way every time, so that the same invoice always matches itself.

#### Acceptance Criteria

1. WHEN a Credit description matches an invoice pattern, THE Fabrication_Module SHALL produce an Invoice_Reference made of the upper-cased prefix and the number with leading zeros removed.
2. THE Fabrication_Module SHALL treat `INV0042`, `inv-042` and `INV #042` as the same Invoice_Reference. References need at least three digits, so short numbers such as `INV12` are not treated as references.
3. WHEN a Credit description has no match, such as `Customer Payment INV`, THE Fabrication_Module SHALL give that Credit no Invoice_Reference.
4. WHEN a description starts with a rail prefix from the Fabrication_Config, THE Fabrication_Module SHALL remove the rail prefix from the Counterparty.
5. IF a Credit has no description, no amount or an amount of zero or less, THEN THE Fabrication_Module SHALL leave that row out of the reference rules without raising an error.
6. THE Fabrication_Module SHALL cut descriptions to 500 characters before matching.

### Requirement 3: Detect a repeated invoice reference

**User Story:** As a credit analyst, I want to be told when the same customer appears to pay the same invoice more than once, so that I can check for copied receipts.

#### Acceptance Criteria

1. WHEN two or more Credits share one Counterparty and one Invoice_Reference, THE Dup_Rule SHALL treat them as a Repeated_Reference_Group.
2. THE Dup_Rule SHALL ignore debits when forming Repeated_Reference_Groups.
3. THE Dup_Rule SHALL keep the same Invoice_Reference under two different Counterparties as two separate keys.
4. WHEN every Invoice_Reference in a file appears on at most one Credit per Counterparty, THE Dup_Rule SHALL report no finding.
5. WHEN the amounts in a Repeated_Reference_Group are all identical, THE Dup_Rule SHALL state "identical amount, possible duplicate posting" in the evidence.

### Requirement 4: Severity, escalation and the instalment caveat

**User Story:** As a credit analyst, I want repeated references rated by how unusual they are, so that one possible instalment does not look like a forged file.

#### Acceptance Criteria

1. WHEN a file has fewer than `dup_escalate_min_refs` Repeated_Reference_Groups, THE Dup_Rule SHALL report one MEDIUM `FL_DUP_INVOICE_REF` finding per group with weight `dup_isolated_weight` (10 by default).
2. WHEN a file has `dup_escalate_min_refs` or more Repeated_Reference_Groups, THE Dup_Rule SHALL report exactly one HIGH `FL_DUP_INVOICE_REF` finding with weight `dup_escalated_weight` (22 by default) that lists every group.
3. WHEN a group has at most `instalment_max_count` Credits and its first and last Credit are at most `instalment_max_days` apart, THE Dup_Rule SHALL state "may be an instalment payment; check the invoice" in the evidence for that group.
4. WHEN a group has more than `instalment_max_count` Credits or spans more than `instalment_max_days`, THE Dup_Rule SHALL leave the instalment sentence out of the evidence for that group.
5. THE Dup_Rule SHALL state in the evidence that it cannot tell which row of a group is the copy.
6. WHERE `evidence_max_groups` is smaller than the number of groups, THE Dup_Rule SHALL list that many groups and end with "and N more".

### Requirement 5: Evidence, page and field convention

**User Story:** As a reviewer, I want each finding to show exactly which rows to look at, so that I can check them against the document.

#### Acceptance Criteria

1. THE Fabrication_Module SHALL give every finding the agent label `Node3:FinancialLogic`, a stable reason code, a severity, a weight and a plain-language evidence string.
2. THE Dup_Rule SHALL list, for every Credit in every group named in the evidence, its date, its amount, its page and its row number.
3. THE Fabrication_Module SHALL set the `field` of a finding that names several rows to `rows ` followed by the row numbers in ascending order separated by commas, and SHALL set the `field` of a finding that names one row to `row N`.
4. THE Fabrication_Module SHALL set the `page` of a finding to the page of the latest row it names.
5. THE Fabrication_Module SHALL leave the `Finding` model and the `Transaction` model unchanged.

### Requirement 6: Detect invoice numbers out of date order

**User Story:** As a credit analyst, I want to know when invoice numbers stop rising with the date, so that I can spot receipts that were inserted from elsewhere.

#### Acceptance Criteria

1. THE Sequence_Rule SHALL sort Credits that have an Invoice_Reference and a date by date, then by reference number.
2. THE Sequence_Rule SHALL find the longest run of non-decreasing reference numbers and treat every Credit outside that run as an Order_Break_Row.
3. WHEN a file has at least `seq_min_refs` such Credits and at least `seq_min_breaks` Order_Break_Rows, THE Sequence_Rule SHALL report exactly one LOW `FL_INVOICE_SEQUENCE_BREAK` finding with weight `seq_weight` (5 by default) that names every Order_Break_Row.
4. WHEN a file has fewer than `seq_min_refs` such Credits or fewer than `seq_min_breaks` Order_Break_Rows, THE Sequence_Rule SHALL report no finding.
5. WHEN Credits fall on the same date, THE Sequence_Rule SHALL treat them as in order whatever their sequence on the page.
6. WHEN the Sequence_Rule is run twice on the same transactions, THE Sequence_Rule SHALL return identical findings.

### Requirement 7: Name the likelier copy

**User Story:** As a credit analyst, I want a hint about which of two matching receipts is the copy, so that I start with the right row.

#### Acceptance Criteria

1. WHEN exactly one row of a Repeated_Reference_Group is an Order_Break_Row, THE Dup_Rule SHALL add "row N breaks the invoice number order, so it is the likelier copy" to the evidence of that group.
2. WHEN both rows or none of the rows of a Repeated_Reference_Group are Order_Break_Rows, THE Dup_Rule SHALL name no likelier copy for that group.
3. THE Fabrication_Module SHALL describe the likelier copy as a hint and SHALL NOT state it as a verdict.

### Requirement 8: Detect turnover jumping while fixed costs stay flat

**User Story:** As a credit analyst, I want a warning when money in jumps with no matching rise in running costs, so that I can ask the customer to explain.

#### Acceptance Criteria

1. THE Turnover_Rule SHALL group transactions by calendar month and use only Full_Months.
2. THE Turnover_Rule SHALL match `fixed_cost_keywords` and `turnover_exclude_keywords` as whole words, case-insensitively.
3. THE Turnover_Rule SHALL judge a file only WHILE at least `turnover_min_months` Full_Months have Fixed_Costs above zero and at least `turnover_min_stable_share` of the Full_Months have Fixed_Costs within `fixed_cost_stability_pct` of the median Fixed_Costs.
4. WHEN a file is judged and has `turnover_min_jump_months` or more Jump_Months, THE Turnover_Rule SHALL report exactly one MEDIUM `FL_TURNOVER_COST_MISMATCH` finding with weight `turnover_weight` (8 by default).
5. WHEN a file is judged and has fewer Jump_Months than `turnover_min_jump_months`, THE Turnover_Rule SHALL report no finding.
6. WHEN Fixed_Costs in a month rise by more than `fixed_cost_stability_pct` above their median, THE Turnover_Rule SHALL NOT count that month as a Jump_Month.
7. THE Turnover_Rule SHALL list in the evidence, for every Jump_Month, its Turnover, the median Turnover it was compared with, the ratio, its Fixed_Costs and the median Fixed_Costs.
8. THE Turnover_Rule SHALL NOT use the ratio of whole-file credits to whole-file debits as a trigger.

### Requirement 9: A faulty rule never stops the pipeline

**User Story:** As an operator, I want one broken check to be visible but harmless, so that a bad statement never takes the screening down.

#### Acceptance Criteria

1. IF the Dup_Rule, the Sequence_Rule or the Turnover_Rule raises an exception, THEN THE Fabrication_Module SHALL catch it, write the exception type to the log and report one LOW `DQ_RULE_SKIPPED` finding with weight 2 whose `field` is the code of the skipped rule.
2. WHEN one fabrication rule is skipped, THE Fabrication_Module SHALL still run and report the findings of the other fabrication rules.
3. THE Fabrication_Module SHALL NOT let any exception from a fabrication rule reach the caller of `financial_logic.analyze`.
4. WHEN a file has no Credit with an Invoice_Reference, THE Fabrication_Module SHALL report no Dup_Rule and no Sequence_Rule finding.
5. THE Fabrication_Module SHALL give the same findings in the same order when called twice with the same input, without changing the input.

### Requirement 10: Wiring into Node 3

**User Story:** As a maintainer, I want the new rules to run inside the existing Node 3 step, so that no other part of the pipeline has to change.

#### Acceptance Criteria

1. WHEN `financial_logic.analyze` is called, THE Node_3 SHALL call `analyze_fabrication` after the existing rules and append its findings to the report.
2. THE Node_3 SHALL accept an optional `fabrication_config` argument so that every existing caller keeps working unchanged.
3. THE Fabrication_Module SHALL leave the weights and thresholds in Risk_Synthesis unchanged.
4. THE Fabrication_Module SHALL leave `fraud_detect/store.py` and the Node 1 package unchanged.
5. THE Fabrication_Module SHALL add no new package dependency.

### Requirement 11: Calibrate the structuring rule (separable group)

**User Story:** As a credit analyst, I want the structuring rule to fire on deposits split over a short time and not on three large payments months apart, so that a genuine business statement is not flagged HIGH by chance.

#### Acceptance Criteria

1. WHEN at least three Near_Threshold_Credits fall within a span of `_STRUCTURING_CLUSTER_DAYS` days (14), THE Structuring_Rule SHALL report one HIGH `FL_STRUCTURING` finding with weight 20 that names the span and the rows.
2. WHEN fewer than three Near_Threshold_Credits fall within any span of 14 days, THE Structuring_Rule SHALL report no finding.
3. WHEN a Near_Threshold_Credit has no date, THE Structuring_Rule SHALL count that row as inside every span.
4. WHEN the generator plants four deposits of 9,797 to 9,800 on consecutive days, THE Structuring_Rule SHALL still report `FL_STRUCTURING`.
5. WHEN `statement_F` is screened, THE Structuring_Rule SHALL report no `FL_STRUCTURING` finding, because its three Near_Threshold_Credits (28 Jul, 1 Aug, 28 Aug) do not fit in 14 days.
6. WHERE the calibration is removed from the change, THE Fabrication_Module SHALL keep working with the three new rules unchanged.
7. WHEN the calibration is kept and `statement_F` no longer produces a HIGH finding, THE test suite SHALL drop the strict `xfail` marker from `test_statement_f_outcome`.

### Requirement 12: Investigate why statement_E has no metadata finding (report only)

**User Story:** As a developer, I want to know why the editing-tool and modified-date evidence in `statement_E` raises nothing, so that the next spec fixes the right thing.

#### Acceptance Criteria

1. WHEN the investigation is run, THE investigation SHALL read the metadata of `statement_E.pdf` and `statement_F.pdf` read-only and record Producer, Creator, CreationDate and ModDate for each.
2. WHEN the investigation is run, THE investigation SHALL record which editing-tool names `fraud_detect/nodes/visual_forensics.py` recognises and whether any check compares ModDate with CreationDate.
3. THE investigation SHALL record the cause of the missing finding as one of: metadata absent from the file, tool name missing from the list, or no ModDate check.
4. THE investigation SHALL record how many of the 64 genuine "re-saved" documents in the Forgery_Dataset (`genuine_resaved = 1`) carry a later ModDate or a print or export Producer, as the false-alarm risk of any future metadata rule.
5. THE investigation SHALL change no behaviour of Node 2 and SHALL write its result to `.kiro/specs/node3-fabrication-rules/investigation.md`.

### Requirement 13: Investigate VF_CURRENCY_MIX on both files (report only)

**User Story:** As a developer, I want to know whether `VF_CURRENCY_MIX` is a real signal or a template quirk, so that the genuine file is not penalised unfairly.

#### Acceptance Criteria

1. WHEN the investigation is run, THE investigation SHALL record the text matches that make `VF_CURRENCY_MIX` fire on `statement_E` and `statement_F`.
2. WHEN the investigation is run, THE investigation SHALL count how many of the 1,000 genuine Forgery_Dataset documents trigger `VF_CURRENCY_MIX`.
3. THE investigation SHALL record a recommendation (keep, narrow or drop the check) for a later Node 2 spec.
4. THE investigation SHALL change no behaviour of Node 2 and SHALL write its result to `.kiro/specs/node3-fabrication-rules/investigation.md`.

### Requirement 14: Acceptance on the labelled files

**User Story:** As the project owner, I want to see the rules catch the known forgery and leave the genuine file alone, so that I can judge whether they work on the evidence we have.

#### Acceptance Criteria

1. WHEN `statement_E` is screened and the Labelled_Files exist, THE Dup_Rule SHALL report `FL_DUP_INVOICE_REF` with evidence that names all 12 `inserted_rows` of the answer key, matched by date and amount.
2. WHEN `statement_E` is screened, THE test SHALL report how many of the 12 inserted rows the Sequence_Rule marks as the likelier copy, and how many rows named by the Dup_Rule are genuine.
3. WHEN `statement_E` is screened with `DEFAULT_CONFIG`, THE Turnover_Rule SHALL report `FL_TURNOVER_COST_MISMATCH`.
4. WHEN `statement_F` is screened with `DEFAULT_CONFIG`, THE Fabrication_Module SHALL report none of `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH` or `DQ_RULE_SKIPPED`.
5. WHEN `statement_E` is screened, THE Fabrication_Module SHALL report no `DQ_RULE_SKIPPED` finding.
6. WHERE any of `statement_E.pdf`, `statement_F.pdf` or `ANSWER_KEY_6M_v2.json` is missing from `~/Downloads`, THE acceptance tests SHALL skip.
7. THE acceptance tests SHALL use path constants defined in the test module and SHALL NOT copy the Labelled_Files into the repository.

### Requirement 15: No regression

**User Story:** As a maintainer, I want the existing behaviour kept, so that this change cannot break what already works.

#### Acceptance Criteria

1. WHEN the existing synthetic clean sample is screened, THE pipeline SHALL route it `AUTO_CLEAR` with a risk score of 5 or less and no fabrication finding.
2. WHEN the existing test suite is run with `AUDIT_SIGNING_KEY_PATH` and `FRAUD_ENABLE_TXN_MODEL` unset, THE suite SHALL pass with the same 571 passing tests and, if the calibration group is kept, with the `xfail` of `test_statement_f_outcome` removed as the only change in outcome.
3. WHEN every planted-issue generator case in `tests/test_detection.py` is screened, THE pipeline SHALL still report the code that each case expects.
4. THE Fabrication_Module SHALL report no fabrication finding for any statement produced by `fraud_detect/synthetic.py`, because those statements carry no invoice references and no fixed-cost pattern.

### Requirement 16: False-alarm report (report only)

**User Story:** As the project owner, I want to see how often the new rules fire on documents that should be clean, so that I know the cost of turning them on.

#### Acceptance Criteria

1. WHEN the Evaluation_Script is run, THE Evaluation_Script SHALL screen the 1,000 genuine Forgery_Dataset documents and a batch of generator statements read-only and print, per rule, how many documents fired.
2. WHERE the Forgery_Dataset folder is missing, THE Evaluation_Script SHALL print that the section was skipped and exit normally.
3. THE Evaluation_Script SHALL report the figures and SHALL NOT assert on the Forgery_Dataset result.
4. THE Evaluation_Script SHALL write nothing to `data/cases.json`, the audit chain or any file other than its own printed output.
5. THE Evaluation_Script SHALL print the share of dataset documents that carry an invoice reference, so that a result of zero firings is read in context.

### Requirement 17: Risk-score effect (report only)

**User Story:** As the project owner, I want to see how the new findings change the risk score of the two files, so that I can check the tampered file now ranks above the genuine one.

#### Acceptance Criteria

1. WHEN the Evaluation_Script is run and the Labelled_Files exist, THE Evaluation_Script SHALL print risk score, risk band, routing and finding codes for `statement_E` and `statement_F`.
2. THE Evaluation_Script SHALL print whether the `statement_E` score is above 25 and above the `statement_F` score, as a reported target and not as an assertion.
3. THE Fabrication_Module SHALL use finding weights chosen from first principles and SHALL NOT fit them to the Labelled_Files.
4. THE Fabrication_Module SHALL leave Risk_Synthesis weights and thresholds unchanged.

### Requirement 18: Documentation and honest limits

**User Story:** As a future maintainer, I want the README to explain the new codes, how to tune them and where they fail, so that nobody over-trusts them.

#### Acceptance Criteria

1. THE documentation in `FRAUD_DETECT_README.md` SHALL list the reason codes `FL_DUP_INVOICE_REF`, `FL_INVOICE_SEQUENCE_BREAK`, `FL_TURNOVER_COST_MISMATCH` and `DQ_RULE_SKIPPED` with their severity and weight.
2. THE documentation SHALL explain how to tune each field of Fabrication_Config and give one worked example.
3. THE documentation SHALL state these limits: two labelled files only; the Rule C jump ratio was informed by both files; a brand-new invoice number evades the Dup_Rule; genuine instalments can look like duplicates; about half of the rows named by the Dup_Rule are genuine; and a result of no finding means no signal, not verified.
4. THE documentation SHALL describe the structuring calibration if it is kept.
5. THE documentation SHALL state that Risk_Synthesis weights were not changed.

### Requirement 19: Repository hygiene

**User Story:** As a maintainer, I want the external files and unrelated modules left alone, so that licensing and review stay clean.

#### Acceptance Criteria

1. THE repository SHALL contain no copy of `statement_E.pdf`, `statement_F.pdf`, `ANSWER_KEY_6M_v2.json` or any Forgery_Dataset PDF as a result of this work.
2. THE change SHALL NOT modify `fraud_detect/store.py`.
3. THE change SHALL NOT modify the folder `.kiro/specs/multi-layout-ingestion-and-fabrication-rules/`.
4. WHEN a new test module needs an external file, THE module SHALL skip when the file is missing.
