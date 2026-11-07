# Requirements Document

## Introduction

This feature is **Stage B** of the synthetic-data work. It retrains and re-evaluates the two machine-learning checks, **Node 6a** (transaction fraud: a gradient-boosted model, GBM, plus a graph-neural-network, GNN, account lookup) and **Node 6b** (statement forgery), on the synthetic corpus built in Stage A (`.kiro/specs/synthetic-corpus-generator/`). The goal is a model that learns what fraud looks like, not how the corpus generator prints a statement.

Three rules drive the work:

1. **Hold out whole groups.** Layouts, companies and Master_Seeds are held out as wholes, never rows. Severity cutoffs are chosen on validation only.
2. **Remove generator tells.** Every feature is audited for whether it could separate genuine from tampered only because of how the generator prints. Such features are dropped or neutralised. Styling is randomised independently of the label. A leak test runs first.
3. **Test on data the model never saw.** `statement_E`, `statement_F`, the answer key and the existing forgery dataset are never trained on. A small hand-built set of awkward cases is kept.

The existing models stay as backups. A new model is switched on only if it is at least as good as the one in use. Node 6a stays off by default unless it clears that bar.

All data is synthetic. Results are an end-to-end demonstration, not real-world accuracy. The risk of circularity (a model trained on the generator's output and judged on the generator's output) is stated wherever results appear.

Nodes 1 to 4, `fraud_detect/registry.py`, `fraud_detect/reference_data.py` and the Node 4 weights and thresholds are not changed. No new runtime dependency is added; `torch` and `torch-geometric` stay offline-only.

## Glossary

- **Retraining_Pipeline**: The new offline tooling under `fraud_detect/ml/` that builds feature tables from a corpus, runs the leak test, trains candidate models, evaluates them and writes a comparison report. It is not used at screening time.
- **Corpus**: A Stage A synthetic corpus (`labels.csv`, `rows.csv`, `companies.json`, PDFs, manifests) built by `python -m fraud_detect.corpus.build`.
- **Layout**: One of the six Node 1 profiles: `generator_v1`, `two_col_dmon`, `two_col_slash`, `two_col_dual_date`, `signed_amount_iso`, `two_col_ddmmyy_cr`.
- **Held_Out_Layout**: A Layout whose documents are all in the `test_unseen_layout` Split and none in any other Split.
- **Master_Seed**: The integer that determines a whole Corpus.
- **Held_Out_Seed_Corpus**: A Corpus built with a Master_Seed that is not used for any training or validation Corpus.
- **Split**: `train`, `validation`, `test` or `test_unseen_layout`.
- **Group_Key**: The identifier that must never straddle Splits: the company id, the Layout, or the Master_Seed.
- **Node_6a**: The transaction-fraud check (`ml_transaction`): GBM score plus GNN account lookup.
- **Node_6b**: The statement-forgery check, which reads PDF file features.
- **PDF_Feature_Set**: The Node_6a features that a PDF statement carries (`--feature-set pdf`).
- **Feature_Audit**: The documented review of every Node_6a and Node_6b feature for generator-specific tells.
- **Generator_Tell**: A feature or style choice that separates genuine from tampered documents in a Corpus only because of how the Corpus_Generator prints, and not because of tampering.
- **Leak_Test**: A check that trains a very simple model, or scores a single feature, on generator data and flags any feature that nearly perfectly separates the classes.
- **Leak_Threshold**: A ROC-AUC of 0.98 or higher for a single feature or for the very simple model, on the validation Split.
- **Hard_Negative**: A genuine document that looks suspicious, such as `genuine_resaved`, an instalment-paying customer, or a seasonal business with real growth.
- **Awkward_Set**: A small hand-built set of hard cases kept in the repository as code, with no PDF committed.
- **Real_Style_Files**: `statement_E.pdf`, `statement_F.pdf`, the answer key and the existing statement forgery dataset. They are read-only and never trained on.
- **Forgery_Dataset**: The existing `statement_forgery_dataset` folder.
- **Current_Model**: The model file now in use for a node (`models/statement_forgery_clf.joblib`, `models/txn_fraud_clf.joblib`, and the current GNN score store).
- **Candidate_Model**: A model trained by the Retraining_Pipeline.
- **Switch_Rule**: The rule that decides whether a Candidate_Model replaces a Current_Model.
- **Size_Guard**: The Node_6b check that makes it abstain on documents whose size features fall outside the central range of its training documents.
- **Comparison_Report**: The file and console output that compares Candidate_Model and Current_Model per source and per Split.
- **Source**: Where an evaluated document comes from: `forgery_dataset`, `generator_corpus`, `real_style` or `awkward_set`.
- **Cutoffs**: The medium and high severity score cutoffs stored in a model file.
- **Backup_Folder**: `models/backup/`, where Current_Model files are kept before any switch.

## Requirements

### Requirement 1: Split by whole groups only

**User Story:** As a project owner, I want training, validation and test data separated by whole layouts, companies and seeds, so that scores are not inflated by near-copies.

#### Acceptance Criteria

1. THE Retraining_Pipeline SHALL assign every document to the Split recorded in the Split column written by the Corpus_Builder and SHALL NOT assign any document to a Split by row, so that the Split of every document after loading equals its Split column value in 100% of documents.
2. WHEN the Retraining_Pipeline loads Corpora and before training starts, THE Retraining_Pipeline SHALL verify that no company id appears in more than one Split across all Corpora read.
3. WHEN the Retraining_Pipeline loads Corpora and before training starts, THE Retraining_Pipeline SHALL verify that every document of the Held_Out_Layout is in `test_unseen_layout` and that no document of any other Layout is in `test_unseen_layout`.
4. WHEN a Held_Out_Seed_Corpus is supplied, THE Retraining_Pipeline SHALL verify before training starts that its Master_Seed differs from the Master_Seed of every Corpus used for training or validation, and SHALL use the Held_Out_Seed_Corpus only for evaluation.
5. WHEN a Held_Out_Seed_Corpus is supplied, THE Retraining_Pipeline SHALL verify before training starts that no company id and no UEN of the Held_Out_Seed_Corpus appears in any training or validation Corpus.
6. WHERE cross-validation is used inside the training Split, THE Retraining_Pipeline SHALL assign all documents with the same company id to the same fold, so that no company id appears in more than one fold.
7. IF any check in criteria 2 to 5 fails, THEN THE Retraining_Pipeline SHALL stop with a non-zero exit status before any training step starts, SHALL print the check name and every offending group key (company id, UEN, Layout or Master_Seed), and SHALL NOT write any trained model artefact.
8. WHEN Node_6a features that use per-account history are built, THE Retraining_Pipeline SHALL compute each feature value only from documents that share the same company id and the same Split as the document being featurised.
9. WHEN a run completes, THE Retraining_Pipeline SHALL record in the Comparison_Report the Master_Seed of each Corpus, the Held_Out_Layout, and the document count per Split and per Layout, where the counts per Split sum to the total document count of all Corpora read.
10. IF a Corpus has a missing Split column, or a document has a Split value that is empty or not one of the Split names defined in the Glossary, THEN THE Retraining_Pipeline SHALL stop before training starts with a non-zero exit status and SHALL print the Corpus name and the offending document identifiers.

### Requirement 2: Choose Cutoffs on validation only

**User Story:** As a project owner, I want severity cutoffs picked without looking at the test data, so that the test numbers are honest.

#### Acceptance Criteria

1. THE Retraining_Pipeline SHALL choose the Cutoffs of a Candidate_Model using only scores computed on the `validation` Split, and SHALL NOT read any score or label from `test`, `test_unseen_layout`, the Held_Out_Seed_Corpus or the Real_Style_Files during Cutoff selection.
2. THE Retraining_Pipeline SHALL record in the Comparison_Report, before any test-Split score is computed, a Cutoff selection rule that names the target false-alarm rate on validation genuine documents (a value between 0.0% and 100.0%) for each Cutoff.
3. WHEN the Cutoffs are fixed, THE Retraining_Pipeline SHALL record each Cutoff value in the Comparison_Report, and only after this record exists SHALL it compute scores on `test`, `test_unseen_layout`, the Held_Out_Seed_Corpus and the Real_Style_Files.
4. WHEN the Retraining_Pipeline reports a score on any Split, THE Retraining_Pipeline SHALL apply the fixed Cutoffs with every Cutoff value identical to the value recorded in the Comparison_Report.
5. IF a Cutoff, a hyperparameter or a feature choice is changed after any test-Split score has been shown, THEN THE Retraining_Pipeline SHALL record the change in the Comparison_Report as a post-test change, SHALL mark every affected test score in the Comparison_Report as "no longer held out", and SHALL keep the original pre-change values in the report.
6. WHEN a training run completes, THE Retraining_Pipeline SHALL print the validation operating curve with one row per Cutoff, each row showing the alert rate, the genuine false-alarm rate and the tampered catch rate, each as a percentage with 1 decimal place.
7. IF the validation Split contains zero genuine documents or zero tampered documents, THEN THE Retraining_Pipeline SHALL stop the run before fixing any Cutoff, SHALL report an error naming the missing document class, and SHALL NOT compute any test-Split score.
8. IF no Cutoff on the validation Split meets the target false-alarm rate, THEN THE Retraining_Pipeline SHALL report an error naming the Cutoff and the closest achieved false-alarm rate, and SHALL NOT compute any test-Split score for that Candidate_Model.

### Requirement 3: Audit every feature for generator tells

**User Story:** As a project owner, I want every feature checked for whether it only reflects how the generator prints, so that the model cannot pass by recognising the generator.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline is run, THE Retraining_Pipeline SHALL produce a Feature_Audit containing exactly one entry for each feature in `statement_features.FEATURE_NAMES` and each feature in the Node_6a PDF_Feature_Set, with no feature omitted and no duplicate entries.
2. THE Feature_Audit SHALL record for each feature these four fields: a description of what it measures, a yes/no value for whether the Corpus_Generator sets it independently of the label, a yes/no value for whether a genuine real-world file could plausibly differ on it, and a decision of exactly one of `keep`, `neutralise` or `drop`.
3. THE Feature_Audit SHALL contain an entry for each of these candidate Generator_Tells: Producer string, Creator string, font names, font count, page width and height, page count, round-amount share, row order, row spacing, creation time stamp, modification time stamp, and footer wording.
4. IF a feature has the decision `drop`, THEN THE Retraining_Pipeline SHALL exclude it from the Candidate_Model input, and the Candidate_Model input SHALL contain zero dropped features.
5. IF a feature has the decision `neutralise`, THEN THE Retraining_Pipeline SHALL replace it with a form whose value is unchanged when only the printing style of the same underlying content changes, and SHALL record that form in the Feature_Audit entry for the feature.
6. WHEN the Comparison_Report is written, THE Retraining_Pipeline SHALL include the full Feature_Audit and the Leak_Test results for the same run, with each feature's Leak_Test result shown in the same table row as its audit entry.
7. WHEN a Candidate_Model is trained, THE Retraining_Pipeline SHALL store in its model file the ordered list of feature names used and the Feature_Audit version identifier.
8. IF a feature appears in the Real_Style_Files only as a creator tool name, THEN THE Retraining_Pipeline SHALL give it the decision `keep` only where the Forgery_Dataset contains at least one forged file and at least one genuine file that differ on that feature, and otherwise SHALL give it the decision `drop`.
9. IF a Feature_Audit entry is missing a required field or has a decision other than `keep`, `neutralise` or `drop`, THEN THE Retraining_Pipeline SHALL stop before training, produce no Candidate_Model, and report an error naming the affected feature.

### Requirement 4: Run a leak test before training

**User Story:** As a project owner, I want a quick check that no single feature gives the answer away, so that a leak is found before it is trained into a model.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline starts, THE Retraining_Pipeline SHALL run the Leak_Test on the `train` and `validation` Splits before it trains any Candidate_Model.
2. THE Leak_Test SHALL compute, for every feature, the validation ROC-AUC of that feature alone, using a decision tree of depth 2 fitted on `train` with a fixed random seed, so that two runs on the same Corpus give the same AUC to 4 decimal places.
3. THE Leak_Test SHALL fit two very simple models on `train`, a logistic regression on standardised features and a decision tree of depth 3, and SHALL report the validation ROC-AUC of each.
4. IF the validation ROC-AUC of a single feature or of either very simple model is 0.98 or higher (the Leak_Threshold), THEN THE Retraining_Pipeline SHALL list the feature (or the model) as a suspected leak, SHALL exit with a non-zero status, and SHALL train no Candidate_Model and write no model file until the project owner has recorded an explicit decision of either "exclude feature" or "accept feature" for each listed feature.
5. WHEN a suspected leak is listed, THE Retraining_Pipeline SHALL print the feature name, its validation AUC, its mean value for genuine documents and its mean value for tampered documents on the validation Split, and the Layout in which the per-Layout AUC is furthest from 0.5.
6. THE Leak_Test SHALL also report the validation AUC of every feature for each of the Layouts present in the `validation` Split, and SHALL report "not computable" for a Layout in which the validation Split has fewer than 2 genuine documents or fewer than 2 tampered documents.
7. WHERE the Forgery_Dataset is available, THE Leak_Test SHALL compute the same single-feature AUC on the Forgery_Dataset features as a read-only comparison, without writing to, training on, or modifying the Forgery_Dataset.
8. WHEN the absolute difference between the generator Corpus AUC and the Forgery_Dataset AUC of a feature is greater than 0.20, THE Retraining_Pipeline SHALL list that feature, with both AUC values, in the Comparison_Report as a feature for investigation.
9. IF the Forgery_Dataset is missing or unreadable, THEN THE Retraining_Pipeline SHALL print a notice that the comparison in criteria 7 and 8 was skipped and SHALL continue with the rest of the Leak_Test.
10. IF the `train` or `validation` Split is empty, or contains documents of only one class (genuine or tampered), THEN THE Retraining_Pipeline SHALL stop before training, exit with a non-zero status, and print an error naming the Split and the missing class.

### Requirement 5: Randomise styling independently of the label

**User Story:** As a project owner, I want producers, fonts and page styling to vary in the same way for genuine and tampered files, so that styling carries no label information.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline builds its training Corpus, THE Retraining_Pipeline SHALL draw each styling choice (Producer string, Creator string, fonts, bank name, address wording, row spacing, creation time of day) from the same distribution for genuine and for tampered documents.
2. WHEN the Retraining_Pipeline finishes building the Corpus, THE Retraining_Pipeline SHALL compute, for each styling value that appears in at least 30 documents, the share of genuine documents and the share of tampered documents that carry that value, and SHALL check that the two shares differ by no more than 10 percentage points.
3. IF a styling value fails criterion 2, THEN THE Retraining_Pipeline SHALL list every failing value with its genuine share and its tampered share, SHALL stop before training starts, and SHALL leave the built Corpus unchanged.
4. WHEN the Retraining_Pipeline builds the genuine documents, THE Retraining_Pipeline SHALL include genuine documents whose Producer string, Creator string and modification time stamp match values that also appear on tampered documents, so that no editing-tool name or time stamp pattern alone determines the label.
5. WHEN the Retraining_Pipeline finishes building the Corpus, THE Retraining_Pipeline SHALL check that `genuine_resaved` documents make up at least 15% of the genuine documents in each of `train`, `validation` and `test`, and IF any split is below 15%, THEN THE Retraining_Pipeline SHALL report the split name and its measured share and SHALL stop before training starts.
6. IF the Corpus_Generator cannot provide the randomisation required by criteria 1 and 4, THEN THE Retraining_Pipeline SHALL stop before training starts, SHALL report the name of each missing styling choice, and SHALL NOT change the Corpus_Generator in this feature.
7. THE Retraining_Pipeline SHALL NOT change the Corpus_Generator, any detector, or any Layout renderer in order to change a measured number.

### Requirement 6: Mix tamper types and difficulty

**User Story:** As a project owner, I want training data that includes subtle tampering and many hard negatives, so that the model does not learn only the obvious cases.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline builds the training Corpus with `--tamper-mix`, THE Retraining_Pipeline SHALL include at least 20 documents for every Tamper_Type present in `train`.
2. IF the build produces fewer than 20 documents for any Tamper_Type present in `train`, THEN THE Retraining_Pipeline SHALL stop with an error that names each Tamper_Type below the minimum and its document count, and SHALL NOT write a training Corpus marked as complete.
3. THE Retraining_Pipeline SHALL include in each of `train`, `validation` and `test` at least 1 document for every Tamper_Type that produces a document-level or row-level label, as reported by the corpus manifest.
4. WHEN evaluation completes, THE Retraining_Pipeline SHALL report the tampered-document catch rate (caught tampered documents divided by total tampered documents, as a percentage with 1 decimal place) per Tamper_Type and per Tamper_Method (`overlay`, `rerender`, `content`, `file`), together with the document count behind each rate.
5. WHERE the corpus manifest records a difficulty or subtlety setting, THE Retraining_Pipeline SHALL report the catch rate per difficulty level, together with the document count behind each rate.
6. THE Retraining_Pipeline SHALL include in each Split that holds tampered documents at least 1 document for each of these subtle tampers: a single changed amount, a single changed date, and an amount change of 5% or less.
7. WHEN evaluation completes, THE Retraining_Pipeline SHALL report the Hard_Negative false-alarm rate (Hard_Negative documents flagged as tampered divided by total Hard_Negative documents) separately from the false-alarm rate on ordinary genuine documents, each with its document count.
8. IF a Tamper_Type is absent from a Split, THEN THE Retraining_Pipeline SHALL list the Tamper_Type and the Split in the Comparison_Report and SHALL NOT report a catch rate for that Tamper_Type in that Split.

### Requirement 7: Never train on the Real_Style_Files

**User Story:** As a project owner, I want the real-style files kept strictly for testing, so that they remain an independent check.

#### Acceptance Criteria

1. THE Retraining_Pipeline SHALL open `statement_E.pdf`, `statement_F.pdf`, the answer key and every Forgery_Dataset file in read-only mode, such that the size, content and modification time of each file are unchanged after a pipeline run.
2. THE Retraining_Pipeline SHALL NOT use any feature value, label, Cutoff or hyperparameter derived from the Real_Style_Files to fit or select a Candidate_Model, a Cutoff or a feature set.
3. WHERE a Candidate_Model is trained on the generator Corpus only, THE Retraining_Pipeline SHALL exclude every Forgery_Dataset document from both the training file list and the validation file list (0 Forgery_Dataset documents in either list).
4. WHERE the Retraining_Pipeline is configured to train on Forgery_Dataset documents, THE Retraining_Pipeline SHALL fit only on the Forgery_Dataset `train` Split, SHALL select Cutoffs only on the `validation` Split, and SHALL report final metrics only on the `test` Split and the out-of-layout Split.
5. WHEN the Retraining_Pipeline reads a Real_Style_File, THE Retraining_Pipeline SHALL record the file path and the SHA-256 hash of the file contents in the Comparison_Report.
6. THE Retraining_Pipeline SHALL create, modify or delete 0 files inside the folders containing the Real_Style_Files, including temporary, cache and output files.
7. IF a Real_Style_File does not exist at its configured path, THEN THE Retraining_Pipeline SHALL skip that file, SHALL print a message naming the skipped file, SHALL mark the Switch_Rule input from that source as missing in the Comparison_Report, and SHALL continue processing the remaining sources without aborting.
8. THE Retraining_Pipeline SHALL include an automated test that fails if the normalized absolute path of any Real_Style_File appears in any training file list or validation file list used by the Retraining_Pipeline.
9. IF a Real_Style_File path is detected in a training or validation file list at run time, THEN THE Retraining_Pipeline SHALL abort before fitting any Candidate_Model, SHALL print an error naming the offending path, and SHALL leave previously saved models and reports unchanged.

### Requirement 8: Keep a hand-built set of awkward cases

**User Story:** As a project owner, I want a small set of hard cases that I can read and reason about, so that I can see how the models behave where it matters.

#### Acceptance Criteria

1. THE Retraining_Pipeline SHALL provide an Awkward_Set built in code from the Corpus_Generator with one fixed Master_Seed, which is the same on every run and is recorded in the Comparison_Report, containing at least these five case types: a seasonal business with real growth (monthly credits at peak at least 2 times those at trough, and year-on-year growth of at least 10%), a genuine file with a legitimate re-save, a genuine file with instalment payments, a genuine file with a long statement of 12 consecutive months, and a genuine file with round-amount payroll and rent (at least 80% of payroll and rent lines are whole multiples of 100).
2. THE Awkward_Set SHALL contain, for each case type in criterion 1, at least 3 genuine documents and at least 3 tampered documents, each tampered document being built from the same company profile and Layout as a genuine document of that case type with one tampering applied, so that the Awkward_Set holds at least 15 genuine and at least 15 tampered documents in total.
3. THE Awkward_Set SHALL be used for evaluation only and SHALL NOT be used for fitting, Cutoff choice or feature choice.
4. IF any Awkward_Set document, company id or Master_Seed is found in a training or validation Corpus, THEN THE Retraining_Pipeline SHALL stop before training and SHALL print the check name and the offending ids.
5. THE Awkward_Set SHALL be built in memory or in a temporary folder outside the repository working tree, THE Retraining_Pipeline SHALL delete the temporary folder when the run ends (whether it succeeds or fails), and no PDF of the Awkward_Set SHALL be committed.
6. WHEN the Retraining_Pipeline evaluates the Awkward_Set, THE Retraining_Pipeline SHALL report it in the Comparison_Report as the Source `awkward_set`, separately from all other Sources, and SHALL report per case type, for the Candidate_Model and the Current_Model, the genuine document count, the tampered document count, the genuine false-alarm rate, the tampered catch rate, and the id of every misclassified document.
7. IF the Awkward_Set cannot be built, or any case type has fewer than 3 genuine or fewer than 3 tampered documents, THEN THE Retraining_Pipeline SHALL stop before the Switch_Rule decision, SHALL print the failing case type and the document counts, and SHALL leave the Current_Model unchanged.
8. THE Awkward_Set definition SHALL be reviewed by the project owner before it is used for a Switch_Rule decision, and THE Retraining_Pipeline SHALL record in the Comparison_Report the reviewer name, the review date and the Awkward_Set definition version that was reviewed.
9. IF the Comparison_Report has no recorded review for the current Awkward_Set definition version, THEN THE Retraining_Pipeline SHALL mark the `awkward_set` results as unreviewed and SHALL NOT use them for a Switch_Rule decision.

### Requirement 9: Report results per source and per seed

**User Story:** As a project owner, I want results reported by source and with spread over several seeds, so that I can see what is stable and what is luck.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline writes a Comparison_Report, THE Comparison_Report SHALL report every metric separately for each Source: `forgery_dataset`, `generator_corpus`, `real_style` and `awkward_set`.
2. THE Comparison_Report SHALL NOT report any metric value computed over documents from more than one Source.
3. WHEN a full retraining run is started, THE Retraining_Pipeline SHALL repeat the build, training and evaluation for at least 5 distinct Master_Seeds.
4. IF fewer than 5 distinct Master_Seeds complete build, training and evaluation, THEN THE Retraining_Pipeline SHALL mark the run "incomplete" in the Comparison_Report and SHALL list the Master_Seeds that failed.
5. THE Comparison_Report SHALL report, for each metric, each Source and each Split, the mean, the minimum, the maximum and the sample standard deviation across the completed Master_Seeds, and SHALL list the Master_Seed values used.
6. THE Comparison_Report SHALL report each metric separately on the `test` Split, on the `test_unseen_layout` Split and on the Held_Out_Seed_Corpus.
7. WHERE the Held_Out_Layout is varied across runs, THE Comparison_Report SHALL report the results for each Held_Out_Layout separately and SHALL name the Layout.
8. FOR Node_6b, THE Comparison_Report SHALL report ROC-AUC, PR-AUC, and the genuine false-alarm rate and the tampered catch rate at the medium Cutoff and at the high Cutoff.
9. FOR Node_6a, THE Comparison_Report SHALL report PR-AUC, ROC-AUC, and precision and recall at the top 1% of scores, where the top 1% is the highest-scoring 1% of documents, rounded up to a whole document.
10. THE Comparison_Report SHALL give, next to each metric, the number of documents of each class (positive and negative) behind it.
11. IF a metric rests on fewer than 30 documents in either class, THEN THE Comparison_Report SHALL mark that metric "indicative only".
12. IF a metric cannot be computed because a class has 0 documents, THEN THE Comparison_Report SHALL show the metric as "not computable" with the document count, and SHALL NOT show a numeric value.
13. THE Comparison_Report SHALL list the Real_Style_Files results per file (`statement_E.pdf` and `statement_F.pdf`), because only two labelled files exist.

### Requirement 10: Node 6b features and training

**User Story:** As a project owner, I want a statement-forgery model trained on the audited features, so that Node 6b reflects tampering and not printing style.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline extracts Node_6b features from a Corpus PDF, THE Retraining_Pipeline SHALL use the existing `statement_features.py`.
2. WHERE the Feature_Audit changes the feature set, THE Retraining_Pipeline SHALL apply the change so that every kept feature has a value identical (exact equality) to the value the existing `statement_features.py` produces for the same PDF before the change.
3. THE Retraining_Pipeline SHALL cache extracted features in a folder ignored by git, keyed by the Master_Seed and the feature-set version.
4. WHEN the Master_Seed or the feature-set version differs from that of an existing cache entry, THE Retraining_Pipeline SHALL re-extract the features and SHALL NOT read the existing entry.
5. WHEN the Retraining_Pipeline trains a Node_6b Candidate_Model, THE Retraining_Pipeline SHALL use the same gradient boosting model family as the Current_Model, so that the comparison measures data and features, not model type.
6. WHEN the Retraining_Pipeline writes a Candidate_Model, THE Retraining_Pipeline SHALL write it to a path under `data/private/ml_retraining/`, which is ignored by git, and SHALL NOT overwrite or modify any Current_Model file.
7. WHEN a Candidate_Model is written, THE Retraining_Pipeline SHALL include in the file the Cutoffs, the feature list, the Feature_Audit version, the Master_Seed, the Held_Out_Layout and the Size_Guard setting.
8. IF a PDF in the Corpus cannot be opened for feature extraction, THEN THE Retraining_Pipeline SHALL count the document as a skipped document, SHALL record its id in the skipped-document list, SHALL exclude it from training and evaluation, and SHALL continue with the remaining documents.
9. WHEN feature extraction finishes, THE Retraining_Pipeline SHALL report the number of skipped documents and the skipped percentage for each Split.
10. IF more than 2% of the documents in any Split are skipped, THEN THE Retraining_Pipeline SHALL stop before training, SHALL indicate the Split and its skipped percentage in an error, and SHALL NOT write a Candidate_Model.

### Requirement 11: Node 6a features and training (PDF-visible only)

**User Story:** As a project owner, I want the transaction model to use only what a PDF statement shows, so that the model can run on a real statement.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline trains the Node_6a GBM Candidate_Model, THE Retraining_Pipeline SHALL use the PDF_Feature_Set only (`--feature-set pdf`) and SHALL record the feature-set name and the ordered list of feature names used in the training report.
2. WHEN the Retraining_Pipeline builds Node_6a feature rows, THE Retraining_Pipeline SHALL build them from `rows.csv` and `labels.csv` using the existing `transaction_features.py`, and SHALL exclude `is_fraud_row`, the Tamper_Type and every other label-bearing column from the feature matrix, so that the feature matrix contains zero columns derived from `labels.csv`.
3. IF the Retraining_Pipeline is asked to use a feature that is not in the PDF_Feature_Set, THEN THE Retraining_Pipeline SHALL refuse before training starts, SHALL name every offending feature in an error message, SHALL exit with a non-zero status, and SHALL NOT write a Candidate_Model artifact.
4. WHEN the Retraining_Pipeline builds Node_6a training labels, THE Retraining_Pipeline SHALL set each label from the row-level label in `rows.csv` (1 for a fraudulent row, 0 for a genuine row), with one label per feature row.
5. IF a `rows.csv` row has a missing or non-binary row-level label, THEN THE Retraining_Pipeline SHALL exclude that row, SHALL report the count of excluded rows, and SHALL abort with a non-zero status when more than 1% of rows are excluded.
6. WHERE the GNN is retrained, THE Retraining_Pipeline SHALL run the GNN offline, SHALL build its graph and score store from the Corpus `train` Split only, and SHALL NOT import `torch` or `torch-geometric` in the runtime imports of any live node.
7. WHEN the GNN score store is evaluated, THE Retraining_Pipeline SHALL score each held-out company through the account-level lookup keyed on the UEN, and SHALL report the share of held-out UENs that the store contains as a percentage with one decimal place.
8. IF a held-out UEN is absent from the GNN score store, THEN THE Retraining_Pipeline SHALL record the lookup as "no score", SHALL NOT substitute a training-company score or a default numeric score, and SHALL report the count of "no score" lookups.
9. WHEN the Retraining_Pipeline evaluates Node_6a, THE Retraining_Pipeline SHALL report the false-alarm rate on genuine documents per statement (the number of genuine statements flagged divided by the number of genuine statements), because the live node reads whole statements.
10. WHEN the Retraining_Pipeline evaluates Node_6a, THE Retraining_Pipeline SHALL report the result separately for the GBM alone, for the GNN alone and for both together, and each of the three results SHALL include the false-alarm rate per statement and the fraud detection rate on the same held-out statements.

### Requirement 12: Decide Node 6b's size guard

**User Story:** As a project owner, I want a clear decision on the size guard, so that Node 6b either handles long statements or says plainly that it does not.

#### Acceptance Criteria

1. WHEN the Comparison_Report is generated, THE Comparison_Report SHALL state, for each Split, the count and the percentage (to one decimal place) of documents that fall outside the Size_Guard range of the Current_Model, and the same count and percentage for the Candidate_Model.
2. WHERE the Candidate_Model is trained on Corpus statements of 6 to 12 months, THE Retraining_Pipeline SHALL compute the Size_Guard ranges from the Candidate_Model training documents using the same quantiles as the Current_Model.
3. WHEN the Comparison_Report is generated, THE Comparison_Report SHALL report Node_6b results per Source with the Size_Guard on and with the Size_Guard off, and each result SHALL include the false-alarm rate and the detection rate.
4. WHEN the Comparison_Report is generated, THE Retraining_Pipeline SHALL recommend exactly one of three options for the Size_Guard (keep the Current_Model guard, use the Candidate_Model guard, or remove the guard) and SHALL state the figures from criteria 1 and 3 that support the recommendation.
5. WHEN the recommendation is to remove the Size_Guard, THE Comparison_Report SHALL show the Candidate_Model's false-alarm rate on genuine 12-month statements and on genuine 6-month statements, and the 12-month rate SHALL NOT exceed the 6-month rate by more than 5 percentage points.
6. IF the condition in criterion 5 is not met, THEN THE Retraining_Pipeline SHALL NOT recommend removing the Size_Guard and SHALL state in the Comparison_Report that the condition failed, with both false-alarm rates.
7. THE Retraining_Pipeline SHALL leave the decision to adopt the recommended Size_Guard to the project owner, and SHALL NOT change the Size_Guard in the Current_Model or the Candidate_Model without the project owner's approval.

### Requirement 13: Compare with the current models

**User Story:** As a project owner, I want every new model compared with the one in use on the same data, so that a switch is justified by numbers.

#### Acceptance Criteria

1. WHEN the Retraining_Pipeline produces a Comparison_Report, THE Retraining_Pipeline SHALL score the Current_Model and the Candidate_Model on the same set of documents, using the same Cutoff rules, for every dataset listed in criterion 2.
2. WHEN the Retraining_Pipeline produces a Comparison_Report, THE Comparison_Report SHALL contain, for each node, one result row per metric for each of these seven datasets: the `test` Split, the Held_Out_Layout, the Held_Out_Seed_Corpus, the Forgery_Dataset test Split, the Forgery_Dataset out-of-layout Split, the Real_Style_Files and the Awkward_Set.
3. WHEN the Comparison_Report is produced, THE Comparison_Report SHALL give, for each metric on each dataset, the Candidate_Model value, the Current_Model value, the difference (Candidate_Model minus Current_Model), and the spread across Master_Seeds as the minimum and maximum of the difference over all Master_Seeds used.
4. THE Comparison_Report SHALL include a statement that the Current_Model may have been fitted on documents that resemble the Forgery_Dataset and that comparisons on Forgery_Dataset data are therefore not symmetric, and the statement SHALL appear next to every Forgery_Dataset result in the report.
5. IF the Current_Model file for a node is missing, THEN THE Comparison_Report SHALL mark every comparison for that node as "no baseline", SHALL leave the Current_Model and difference values for that node empty, and THE Switch_Rule SHALL NOT switch that node on.
6. IF the Current_Model file for a node exists but cannot be loaded or cannot score a dataset, THEN THE Comparison_Report SHALL mark the affected comparisons as "baseline failed" with an error indication naming the node and dataset, SHALL still report the Candidate_Model values, and THE Switch_Rule SHALL NOT switch that node on.

### Requirement 14: Switch rule

**User Story:** As a project owner, I want a fixed rule that decides whether a new model goes live, so that I do not pick the favourable numbers after the fact.

#### Acceptance Criteria

1. WHEN an evaluation run starts, THE Retraining_Pipeline SHALL have already written the Switch_Rule, in the form defined in the code, into the Comparison_Report, and SHALL NOT change the Switch_Rule text or thresholds in that Comparison_Report after the first test-Split score is computed.
2. THE Switch_Rule SHALL recommend switching a node to the Candidate_Model only if the Candidate_Model is at least as good as the Current_Model, using the mean across Master_Seeds, on all of these decisive Sources and Splits: the Held_Out_Layout, the Held_Out_Seed_Corpus, the Forgery_Dataset out-of-layout Split, and the Real_Style_Files.
3. THE Switch_Rule SHALL define "at least as good" per node as follows. For Node_6b, the Candidate_Model's ROC-AUC is not lower than the Current_Model's ROC-AUC by more than the standard deviation of the Candidate_Model's ROC-AUC across Master_Seeds, and the tampered catch rate at the medium Cutoff is not lower than the Current_Model's, and the genuine false-alarm rate at the medium Cutoff is not higher than the Current_Model's. For Node_6a, the PR-AUC is not lower than the Current_Model's and the genuine false-alarm rate per statement is not higher than the Current_Model's. Equal values SHALL count as passing, and no rounding SHALL be applied before comparison.
4. WHERE a Source is marked missing or "indicative only", THE Switch_Rule SHALL show that Source as "not decisive" and SHALL NOT count it as passed.
5. IF any decisive Source fails, THEN THE Switch_Rule SHALL recommend keeping the Current_Model for that node.
6. IF no decisive Source is available for a node, THEN THE Switch_Rule SHALL recommend keeping the Current_Model for that node and SHALL state in the Comparison_Report that no decisive Source was available.
7. IF Node_6a does not clear the Switch_Rule, THEN THE Retraining_Pipeline SHALL recommend that the Node_6a GBM and the GNN lookup stay off in the live app.
8. THE Retraining_Pipeline SHALL NOT copy any Candidate_Model over a Current_Model, SHALL NOT change a live-node default and SHALL NOT edit any model file in `models/` by itself; the switch SHALL be a separate step carried out by the project owner.
9. WHEN the project owner runs the switch command for a node, THE Retraining_Pipeline SHALL copy the Current_Model to the Backup_Folder, SHALL verify that the SHA-256 hash of the backup copy equals the hash of the Current_Model, and SHALL record the Comparison_Report identifier and the SHA-256 hash of both the Current_Model and the Candidate_Model, before replacing the Current_Model.
10. IF the backup copy cannot be written or its hash does not match, THEN THE Retraining_Pipeline SHALL stop the switch command with an error indicating the backup failure and SHALL leave the Current_Model unchanged.
11. WHEN the project owner runs the restore command for a node, THE Retraining_Pipeline SHALL copy the backed-up model from the Backup_Folder over the Current_Model and SHALL verify that the SHA-256 hash of the restored file equals the recorded hash of the backup.
12. IF the Backup_Folder holds no backup for the requested node, THEN THE Retraining_Pipeline SHALL stop the restore command with an error indicating that no backup exists and SHALL leave the Current_Model unchanged.
13. THE Switch_Rule SHALL NOT use the Awkward_Set to pass a node.
14. IF a Candidate_Model's false-alarm rate on Hard_Negatives in the Awkward_Set is higher than the Current_Model's by more than 10 percentage points, THEN THE Switch_Rule SHALL block the switch for that node and SHALL show the Awkward_Set as the reason in the Comparison_Report.

### Requirement 15: Stop and ask when a measurement looks wrong

**User Story:** As a project owner, I want the work to stop when a number looks wrong, so that nobody adjusts the generator or the detectors to make it look better.

#### Acceptance Criteria

1. IF a measurement is implausible, THEN THE Retraining_Pipeline SHALL stop all further measurement runs, SHALL print the measurement name, its value, the Source, the Split, the Master_Seed, and the number of documents of each class, SHALL preserve all results already produced, and SHALL wait for a written decision from the project owner before continuing. A measurement is implausible when any of the following holds: (a) a Candidate_Model ROC-AUC is 0.99 or higher on the `test_unseen_layout` Split; (b) a ROC-AUC is below 0.5 on any Source with at least 30 documents of each class; (c) a Real_Style_File score falls on the opposite side of its Cutoff from the label in its answer key; (d) a document count differs from the count in the corpus manifest by 1 or more.
2. WHILE a measurement is under investigation, THE Retraining_Pipeline SHALL reject any change to the Corpus_Generator, any detector, any Cutoff, or any model feature, and SHALL print a message naming the blocked change.
3. WHEN the project owner decides to change the generator or a feature after an investigation, THE Retraining_Pipeline SHALL record in the Comparison_Report, as a post-test change, the date of the decision, the triggering measurement and its value, the reason given by the project owner, and a description of the change.
4. IF a measurement fails or is implausible, THEN THE Retraining_Pipeline SHALL NOT re-run it with a different Master_Seed in order to obtain a different value.
5. THE Retraining_Pipeline SHALL list in the Comparison_Report every Master_Seed that was run, including seeds of failed or discarded runs, with the outcome of each run.

### Requirement 16: Honest framing in the README and the demo

**User Story:** As a project owner, I want the documents and the demo to say plainly that the data is synthetic, so that the prototype is not oversold to financial institutions.

#### Acceptance Criteria

1. THE README section for Stage B SHALL describe every training, validation and test dataset built from the Corpus as "synthetic data", and SHALL contain no sentence that describes Corpus-derived results without the words "synthetic data" in the same paragraph.
2. THE README section SHALL present the results as an end-to-end demonstration and SHALL state in the same paragraph as the first results table that they are not real-world accuracy.
3. THE README section SHALL state the circularity risk: a model trained on the generator's output and judged on the generator's output can score well by learning the generator's habits, and SHALL name each of these seven mitigations: held-out layouts, held-out Master_Seeds, randomised styling, the Leak_Test, the Real_Style_Files, the Switch_Rule and the Awkward_Set.
4. THE README section SHALL state that only two labelled real-style statements exist (`statement_E`, `statement_F`), that this count is too small to estimate accuracy, and that results on them are indicative only.
5. THE Comparison_Report SHALL carry the words "synthetic data" and the circularity statement in its first 10 lines, before any result table, in both its file and console output.
6. WHERE the demo shows a Node_6a or Node_6b score from a Candidate_Model, THE demo SHALL display next to that score the label "model trained on synthetic data", on every screen or output block that shows such a score.
7. THE README section SHALL list, for each of Node_6a and Node_6b, the Switch_Rule outcome (switched or not switched) with the reason, and SHALL state which model (Current_Model or Candidate_Model) each node uses after Stage B.
8. THE README section SHALL state that Node 4 weights and thresholds are not changed by Stage B.
9. IF the Comparison_Report or the README section reports a score on a Source other than `real_style`, THEN THE report or README section SHALL mark the score as measured on synthetic data and as not real-world accuracy.

### Requirement 17: Output hygiene, scope and compatibility

**User Story:** As a project owner, I want the retraining work to leave the rest of the prototype and the repository unchanged, so that nothing breaks and nothing sensitive is committed.

#### Acceptance Criteria

1. THE Retraining_Pipeline SHALL leave Nodes 1 to 4, `fraud_detect/registry.py` and `fraud_detect/reference_data.py` byte-identical to their content before Stage B, as verified by a file hash comparison.
2. THE Retraining_Pipeline SHALL leave the Node 4 weights and thresholds equal to their values before Stage B, as verified by an exact value comparison.
3. THE Retraining_Pipeline SHALL add zero entries to the runtime dependency list.
4. THE Retraining_Pipeline SHALL keep `torch` and `torch-geometric` out of every module that the live app imports, directly or transitively, as verified by importing the live app entry point in an environment where both packages are not installed and observing zero import errors.
5. THE Retraining_Pipeline SHALL write Corpus files, PDFs, feature caches, Candidate_Model files, experiment snapshots and Comparison_Report data files only into folders matched by a git ignore rule, as verified by `git check-ignore` returning a match for every written path.
6. WHEN any Stage B step finishes, THE Retraining_Pipeline SHALL leave `git status --porcelain` listing zero PDF files, `labels.csv`, `rows.csv`, corpus files, experiment model files, snapshot files or external dataset files.
7. IF a Stage B step is about to write to a path not matched by a git ignore rule, THEN THE Retraining_Pipeline SHALL abort that step with a non-zero exit code, write zero bytes to that path, and report an error indicating the path is not ignored by git.
8. WHEN the existing test suite is run after Stage B, THE test suite SHALL report the same number of passed tests and zero newly failing tests compared with the run before Stage B, where tests skipped because a required file is absent are excluded from the comparison.
9. THE Retraining_Pipeline tests SHALL run on a Mini_Corpus built in memory or in a temporary folder, and SHALL read zero files located outside the repository source tree and that temporary folder.
10. WHERE a test needs a Real_Style_File and the file is absent, THE test SHALL be reported as skipped and SHALL NOT be reported as failed.
11. THE Retraining_Pipeline SHALL reuse the Corpus_Generator build-time UEN collision check.
12. IF the Corpus manifest lacks a passed UEN collision check, THEN THE Retraining_Pipeline SHALL refuse to train, exit with a non-zero code, report an error indicating the missing collision check, and create zero Candidate_Model files.
13. THE Retraining_Pipeline SHALL expose each of the steps `audit`, `leak-test`, `train`, `compare`, `switch` and `restore` as one documented command-line command, and each command SHALL exit with code 0 on success and a non-zero code on failure.
14. WHEN a step is run twice with the same inputs and the same Master_Seed, THE Retraining_Pipeline SHALL produce Cutoffs and metrics that are exactly equal across the two runs.
