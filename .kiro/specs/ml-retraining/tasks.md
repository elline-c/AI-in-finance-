# Implementation Plan: ML Retraining (Stage B)

## Overview

Build the retraining pipeline in small, test-first steps. Each group first writes tests that fail, then writes the code that makes them pass, then (optionally) adds property tests. New code lives in a new package `fraud_detect/ml/retraining/`. New tests live in `tests/ml_retraining/`. Nothing in Nodes 1 to 4, `fraud_detect/registry.py`, `fraud_detect/reference_data.py`, the Node 4 weights, `fraud_detect/corpus/` (the generator), `fraud_detect/ml/statement_features.py`, `fraud_detect/ml/transaction_features.py`, `ml_statement.py`, `ml_transaction.py`, `train_statement.py`, `train_transaction.py`, `fraud_detect/ml/report.py` or any existing test is changed. Everything the pipeline writes goes to git-ignored folders (`data/private/ml_retraining/`, `models/backup/`).

Language: Python (matches the repo; the design uses Python). Test command, run from `/Users/weiyang01/AI-in-finance-`:

```
env -u AUDIT_SIGNING_KEY_PATH -u FRAUD_ENABLE_TXN_MODEL /Users/weiyang01/AI-in-finance-/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore
```

Run only `tests/ml_retraining` while working and the whole suite at each checkpoint. Task 1.1 records the starting counts (passed and skipped) in the task notes; every later "whole suite" check compares against them.

Out of scope (do not do):
- Changing Nodes 1 to 4, `registry.py`, `reference_data.py`, the generator, a layout renderer, a detector or the Node 4 weights. Generator gaps (no Creator or font randomisation, no editor names on genuine files, no amount change of 5% or less) are reported by the pipeline's gates, never fixed here.
- Changing Node 6a or Node 6b code to read cutoffs from the model file or to show a "model trained on synthetic data" label (open decisions 8 and 9). The label is stored in the bundle only.
- Retraining the GNN (the corpus has no transfer graph; finding F8). Only the lookup accounting is built.
- Adding any runtime dependency. `torch` and `torch-geometric` must not be imported by any module in the package.

Hard rules for every task:
- Never read, train on, copy, cache or commit `statement_E`, `statement_F`, the answer key, the forgery dataset (`statement_forgery_dataset`), PDFs, corpus files (`labels.csv`, `rows.csv`, manifests) or model files. Package code reads Real_Style and Forgery_Dataset files only through `sources.py`, read-only, in memory. Tests use only an in-memory or temporary Mini_Corpus; a test that needs a Real_Style_File or the Forgery_Dataset is skipped when the file is absent (env vars `RETRAIN_REAL_STYLE_DIR`, `RETRAIN_FORGERY_DIR`).
- Do not run the full pipeline on real files inside any task. The only runs on real files are the manual steps listed in the Notes, carried out by the user.
- Open decisions from the design are not decided here. Implementation tasks build the design's default behaviour and keep each option configurable in `RunConfig`; the checkpoint tasks ask the user.
- If a measurement looks wrong, stop and ask the user; do not tune the generator, a node or a threshold to change a number.
- Honest framing: all results are synthetic data, not real-world accuracy.

## Tasks

- [x] 1. Foundations and safety net
  - [x] 1.1 Create the package skeleton, test scaffolding and the Mini_Corpus fixture
    - Create `fraud_detect/ml/retraining/__init__.py` (no heavy imports), `__main__.py` and `cli.py` with the ten subcommands (`init`, `build`, `check`, `audit`, `leak-test`, `train`, `compare`, `switch`, `restore`, `decide`) as stubs that parse `--help` and exit 0, and the exit-code constants (0 success, 1 other failure, 2 integrity gate or bad arguments, 3 waivable gate waiting for a decision, 4 implausible measurement)
    - Create `tests/ml_retraining/__init__.py`, `conftest.py` and `strategies.py` (shared generators: label tables with layouts, companies and Splits; score and label arrays with ties and plateaus; style profile tables; corrupted audits; path spellings; model byte strings; metric tables per Source and dataset; event sequences)
    - `conftest.py`: a session-scoped `mini_corpus` fixture that builds, with the existing `CorpusBuilder` in a pytest temporary folder, 6 layouts x (2 genuine + 2 tampered), 6 months, two Master_Seeds, `hold_out_layout="two_col_ddmmyy_cr"`, and a `tamper_mix` containing `genuine_resaved`, `metadata_inconsistency`, `font_mismatch`, `edited_amount`, `fabricated_receipts` and `structuring`; plus fixtures `fake_current_models`, `fake_real_style_dir` (tiny synthetic PDFs written in a temp folder, never the real files), a `skip_without_real_style` marker, and an injectable `is_ignored` predicate and clock
    - Create `tests/ml_retraining/protected_hashes.json` holding SHA-256 hashes of Nodes 1 to 4, `registry.py` and `reference_data.py` taken now, and a pinned copy of the Node 4 weights and thresholds (read from the source, not typed from memory)
    - Run the whole suite once and write the counts in the task notes
    - _Requirements: 17.1, 17.2, 17.3, 17.9, 17.10, 17.13_
  - [x] 1.2 Write failing tests for configuration, safe I/O, gates and the decision log
    - In `tests/ml_retraining/test_config.py`: `RunConfig` is frozen; canonical JSON is key-sorted and stable; `run_id` is the SHA-256 prefix of the canonical JSON; every threshold from the design table is present with its default (`leak_auc` 0.98, `forgery_auc_gap` 0.20, `style_share_points` 10, `style_min_docs` 30, `resaved_min_share` 15, `type_min_docs` 20, `subtle_max_relative_change` 5, `min_seeds` 5, `indicative_min_docs` 30, `skip_max_share` 2, `row_exclusion_max_share` 1, `implausible_auc` 0.99, `guard_condition_points` 5, `hard_negative_block_points` 10, `target_fa_medium` 5.0, `target_fa_high` 1.0); defaults for seeds `20260801` to `20260805`, held-out seed `20260899`, held-out layout `two_col_ddmmyy_cr`; `real_style_role` default `required` and `veto_only` accepted; Node 6a baseline path configurable (default `models/txn_fraud_clf.joblib`); Real_Style and Forgery paths overridable; `AWKWARD_MASTER_SEED` differs from every configured corpus seed
    - In `tests/ml_retraining/test_safe_io.py`: `ensure_ignored` calls `git check-ignore` (through the injected predicate in tests), fails closed when git is unavailable, and names the path; `atomic_write` leaves no partial file on failure; `sha256_file` matches `hashlib`; a write to a non-ignored path writes zero bytes and raises; `ReadOnlySource` opens `rb` only, hashes, records `os.stat` before and after and fails when either changed, and creates no file in the source folder
    - In `tests/ml_retraining/test_gates_decisions.py`: `GateResult` fields; `run_gates` returns offenders; an integrity gate cannot be waived; a waivable gate is waived only by a matching `waive-gate` entry; `DecisionLog` is append-only (no edit or remove), assigns ids `D-0001`, ..., and rejects unknown kinds
    - _Requirements: 2.2, 7.1, 7.5, 7.6, 9.3, 17.5, 17.7, 17.14_
  - [x] 1.3 Implement `config.py`, `safe_io.py`, `gates.py` and `decisions.py`
    - `config.py`: `RunConfig`, `Thresholds`, canonical JSON, `run_id`, default paths under `data/private/ml_retraining/`; all open-decision options (`real_style_role`, target false-alarm rates, baseline file, pooled-minimum switch, capability waiver) are plain fields with the design's default
    - `safe_io.py`: `ensure_ignored` (uses `git check-ignore`, fails closed), `atomic_write`, `sha256_file`, `ReadOnlySource`
    - `gates.py`: `Gate`, `GateResult`, `run_gates`, exit codes; `decisions.py`: `DecisionLog` (`decisions.json`) and the post-test change log (`post_test_changes.json`) writers
    - Imports only the standard library at module level; no `torch`, `sklearn` or `pdfplumber` at import time
    - Make the 1.2 tests pass
    - _Requirements: 2.2, 7.1, 7.5, 7.6, 9.3, 17.5, 17.7, 17.14_
  - [x] 1.4 Write property test for the write guard
    - **Property 39: The write guard**
    - **Validates: Requirements 17.5, 17.7**

- [x] 2. Spike: can the existing simulator knobs reach the Awkward_Set figures
  - [x] 2.1 Write failing tests for the Awkward_Set spike
    - In `tests/ml_retraining/test_awkward_spike.py`: `run_spike(seed)` returns, for each of the five case types, the measured figures: `seasonal_growth` (peak to trough ratio of monthly credits, last-3-months over first-3-months credits ratio), `instalments` (count of invoice references on two receipts), `long_12_months` (month count), `round_payroll_rent` (share of payroll and rent lines that are multiples of 100 after a `dataclasses.replace` rewrite of the frozen rows, with the opening balance raised if the replay would go below zero), `legit_resave` (Producer family and ModDate gap after `CATALOG["genuine_resaved"]`); results are plain numbers, the function is deterministic for a fixed seed, writes nothing to disk, and imports no generator internals beyond the public `corpus` package API
    - A test that no file under `fraud_detect/corpus/` is modified by the spike (hash comparison against `git`-tracked content)
    - _Requirements: 8.1, 8.7, 5.7_
  - [x] 2.2 Implement the spike in `fraud_detect/ml/retraining/awkward_spike.py`
    - Use `make_company`, `simulate_ledger`, `SimulatorKnobs` (raised `seasonality_strength`, positive `growth_range`), `instalment_rate = 0.5`, `n_months = 12` and the row rewrite; try a small grid of knob values and report which settings reach: peak at least 2 times trough, last 3 months at least 1.10 times first 3 months (design item I5), at least 80% of payroll and rent at multiples of 100
    - Add a `python -m fraud_detect.ml.retraining.awkward_spike` entry that prints the table and writes `data/private/ml_retraining/spike/awkward_spike.json` through `safe_io` only
    - Do not change any generator code. Make the 2.1 tests pass
    - _Requirements: 8.1, 8.7, 5.7, 17.5_

- [x] 3. Checkpoint - Awkward_Set spike result
  - Ask the user (reason `general-question`) to look at the spike output and confirm: the existing knobs reach the seasonal and growth figures (proceed with the design recipes in task 17), or they do not (then ask whether to relax the stated figures in a new `awkward-v2` definition, or to stop). Do not change the generator either way.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Corpus loading and group-aware split checks
  - [x] 4.1 Write failing tests for loading and group checks
    - In `tests/ml_retraining/test_corpora.py`: `load_corpus` on the Mini_Corpus returns one `DocRef` per `labels.csv` row with `split` equal to the file value, `company_key = (master_seed, company_id)`, `uen` from `companies.json`, `genuine_resaved`, `tamper_types`, `tamper_method`, `n_months`; counts per Split and per Layout sum to the document count; a missing `split` column, or an empty or unknown Split value, raises `CorpusError` naming the corpus and exactly the offending doc ids; manifest counts are read; the same raw `company_id` in two corpora is two different keys
    - In `tests/ml_retraining/test_group_checks.py`: company-in-two-Splits check lists exactly the offending qualified keys; layout check passes only when the held-out layout's documents are exactly the `test_unseen_layout` documents and lists every violating document; held-out-seed check fails when the seed is in the training or validation list, and lists shared UENs and shared qualified keys; an Awkward_Set id, company or seed found in a training or validation corpus is reported; `grouped_folds` keeps a company in one fold and is deterministic; a registered company name shared between a training and an evaluation corpus is a warning, not a failure; each failure carries the check name and every offender and no model file is written
    - Tests for history locality in the Node 6a table belong to task 9
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9, 1.10, 8.4_
  - [x] 4.2 Implement `fraud_detect/ml/retraining/corpora.py`
    - `DocRef`, `LoadedCorpus`, `load_corpus(path)`, `CorpusError`; read `labels.csv`, `companies.json`, `corpus_manifest.json`; never recompute a Split; per-Split and per-Layout counts
    - Make the 4.1 loading tests pass
    - _Requirements: 1.1, 1.9, 1.10_
  - [x] 4.3 Implement `fraud_detect/ml/retraining/group_checks.py`
    - Company, layout, seed, UEN-sharing and Awkward_Set-overlap checks as `Gate` objects (integrity kind, not waivable), name-sharing warning, `grouped_folds`; the UEN-sharing check reuses `corpus.uen_check.UenChecker` with the other corpora's UENs as the snapshot (design finding F12)
    - Make the 4.1 group-check tests pass
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 8.4, 17.11_
  - [x] 4.4 Write property test for loading
    - **Property 1: Loading preserves the Split column and counts add up**
    - **Validates: Requirements 1.1, 1.9, 1.10**
  - [x] 4.5 Write property tests for the group checks
    - **Property 2: Group-overlap checks report exactly the offenders**
    - **Property 3: Layout and seed separation**
    - **Property 4: Grouped folds keep a company in one fold**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 8.4**

- [x] 5. UEN collision-check requirement on the corpus manifest
  - [x] 5.1 Write failing tests for the manifest UEN gate
    - In `tests/ml_retraining/test_uen_gate.py`: a manifest without a `uen_check` object is refused with the missing check named; `uens_checked` lower than the company count is refused; a good manifest passes; refusal gives exit 2, creates zero Candidate_Model files; a UEN shared by two corpora is reported as taken through `UenChecker`; no network is used (a fixture makes `urllib.request.urlopen` raise)
    - _Requirements: 17.11, 17.12, 1.5_
  - [x] 5.2 Implement the UEN manifest gate in `group_checks.py`
    - Add `uen_manifest_gate(corpus)` (integrity gate); no `--check-live-uen` option is offered by any retraining command (`live=False` only)
    - Make the 5.1 tests pass
    - _Requirements: 17.11, 17.12_
  - [x] 5.3 Write property test for the UEN manifest gate
    - **Property 40: The UEN manifest gate**
    - **Validates: Requirements 17.11, 17.12**

- [x] 6. Style profile and corpus-composition gates
  - [x] 6.1 Write failing tests for the style profile and the composition gates
    - In `tests/ml_retraining/test_style_profile.py`: `style_profile(pdf_bytes)` on Mini_Corpus PDFs returns `producer`, `producer_family` (`base`, `export`, `editor` using `PRODUCERS`, `EXPORT_PRODUCERS`, `EDITOR_PRODUCERS`), `creator`, sorted `fonts`, `bank_name` (from `BANK_NAMES`), `street`, `second_address_line`, `row_step_bucket` (0.1 pt), `creation_hour`, `page_size`, `footer_text`; works from `io.BytesIO`; creates no file
    - In `tests/ml_retraining/test_corpus_gates.py` (gate logic on synthetic tables, thresholds passed as small test values): styling balance lists exactly the values in at least `style_min_docs` documents whose genuine and tampered shares differ by more than `style_share_points`, with both shares, leaving corpus file hashes unchanged; `genuine_resaved` share check per Split (below 15% fails and names the Split and share); tamper-type minimum on the pooled `train` Splits lists exactly the types below 20 with counts and writes no completion marker; coverage check lists the (type, Split) pairs among `train`, `validation`, `test` with zero documents; subtle-tamper classification from `affected_rows` `original` and `new` strings (single changed amount, single changed date, change of 5% or less) and the gate lists categories with no document in a Split that holds tampered documents; capability gate reports a styling choice as missing when one class has fewer than 2 distinct values, reports a Producer family that exists in one class only, never reports a choice with 2 or more values in both classes, and the stop message lists exactly the reported choices; the capability and subtle gates are waivable and the others are not; pooled-minimum behaviour is on by default and switchable to per-corpus through `RunConfig`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.6, 6.8_
  - [x] 6.2 Implement `fraud_detect/ml/retraining/style_profile.py`
    - Read styling values back from the PDF with `pypdf` and `pdfplumber` imported inside the function; no manifest style values are used (finding F6)
    - Make the 6.1 style profile tests pass
    - _Requirements: 5.1, 5.2, 5.4_
  - [x] 6.3 Implement `fraud_detect/ml/retraining/corpus_gates.py`
    - Styling balance, resaved share, tamper-type minimums (pooled by default), coverage, subtle-tamper classification, capability gate with named missing choices; each returns a `GateResult`; never writes to or changes the corpus
    - Make the 6.1 gate tests pass
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.6, 6.8_
  - [x] 6.4 Write property tests for styling and capability
    - **Property 16: Styling balance**
    - **Property 17: Capability gate**
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.6**
  - [x] 6.5 Write property tests for composition and subtle tampers
    - **Property 18: Corpus composition gates**
    - **Property 19: Subtle-tamper classification**
    - **Validates: Requirements 5.5, 6.1, 6.2, 6.3, 6.6**

- [x] 7. Checkpoint - Owner decisions on corpus gates
  - Ask the user (reason `general-question`, one question per decision, with the design default marked as recommended):
    - Decision 3: the capability gate will stop `train` because the generator has no Creator or font randomisation, no editor names on genuine files, and no amount change of 5% or less. Does the user want to record a written waiver (printed in the report), wait for a generator follow-up spec, or keep the stop?
    - Decision 5: apply the tamper-type minimum (20) and the coverage check to the pooled Splits of all training seeds (default), or per corpus?
    - Decision 2: confirm the thresholds in the requirements (leak 0.98, forgery gap 0.20, styling 10 points and 30 documents, 5 seeds, implausible 0.99, 12 vs 6 months 5 points, hard negative 10 points).
  - Apply only non-default answers by editing the `RunConfig` defaults or recording the choice for the `decide` command; do not change the generator.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Feature_Audit
  - [x] 8.1 Write failing tests for the audit and the neutralisation toolbox
    - In `tests/ml_retraining/test_audit.py`: `FeatureAudit` has exactly one `model_feature` entry per name in `statement_features.FEATURE_NAMES` (11) and `transaction_features.PDF_FEATURE_COLUMNS` (13), and the 12 `candidate_tell` entries named in Requirement 3.3 (Producer string, Creator string, font names, font count, page width and height, page count, round-amount share, row order, row spacing, creation time stamp, modification time stamp, footer wording); every field present; decision one of `keep`, `neutralise`, `drop`; a recorded form for every `neutralise` entry; a missing field, unknown decision, duplicate or missing feature stops before training and names the feature; `digest()` is pinned per `version` (a test fails if the literal changes while `version` does not); v1 contents match the design tables (3 kept, 1 neutralised, 7 dropped for Node 6b; `balance_after_sgd` and `day_of_week` dropped for Node 6a); the Requirement 3.8 rule gives `keep` only when both forged and genuine value lists are non-empty and differ, otherwise `drop`
    - In `tests/ml_retraining/test_neutralise.py`: the clip form `clip(0, 0.10)` on `minor_font_share` maps every value at or above 0.10 to one value; the transform pickles as scikit-learn and NumPy objects only and loads in a clean subprocess without importing `fraud_detect.ml.retraining`; a feature with no single-column stable form is not accepted by the toolbox
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.9_
  - [x] 8.2 Implement `fraud_detect/ml/retraining/neutralise.py`
    - The single-column toolbox (`FunctionTransformer(np.clip, ...)`, `Binarizer`) as scikit-learn steps on existing columns only (design D5); imports `sklearn` inside functions
    - Make the 8.1 toolbox tests pass
    - _Requirements: 3.5, 10.2_
  - [x] 8.3 Implement `fraud_detect/ml/retraining/audit.py`
    - `AuditEntry`, `FeatureAudit`, the `audit-v1` literal with the three tables of the design (Node 6b features, Node 6a features, candidate tells), `validate()`, `digest()`, `model_input_columns(audit, node)` (ordered, no dropped feature), `creator_only_rule(forged_values, genuine_values)`; the `has_editing_software` entry is decided by the rule, computed by the `audit` command from the Forgery_Dataset when available and otherwise `drop` with a note
    - Make the 8.1 tests pass
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.9_
  - [x] 8.4 Write property tests for audit validation and the Creator-only rule
    - **Property 11: Audit validation**
    - **Property 13: The Creator-only rule**
    - **Validates: Requirements 3.1, 3.2, 3.8, 3.9**

- [x] 9. Feature tables for Node 6b and Node 6a
  - [x] 9.1 Write failing tests for the feature tables and caches
    - In `tests/ml_retraining/test_features_b.py`: the table has the columns from the design (`corpus_key`, `doc_id`, `split`, `layout`, `company_key`, `is_tampered`, `genuine_resaved`, `tamper_types`, `tamper_method`, `n_months`, `skipped`, the raw `extract_features` outputs, `style_*`); values equal `extract_features` output on a Mini_Corpus PDF exactly (equality test); an unreadable PDF is counted as skipped with its id, excluded, and extraction continues; skipped count and percentage per Split are reported; more than `skip_max_share` skipped in any Split stops with exit 2 and no Candidate_Model; the cache path is `data/private/ml_retraining/cache/features_b/<master_seed>-<labels_sha8>-<extractor_sha8>.csv`, the header repeats the three values, a header that differs from the request is not read, a different seed, labels file or extractor source gives a different key and re-extraction; the cache path is checked with `ensure_ignored`
    - In `tests/ml_retraining/test_features_a.py`: the 6a adapter maps `rows.csv` to the table in the design (`account_id` = qualified company key, `timestamp` = `date`, `direction`, `amount_sgd`, `balance_after_sgd`, `y` = `is_fraud_row`) and matches `ml_transaction._rows_from_ingestion` on a fake `IngestionResult`; PDF feature columns come from `build_pdf_features` and equal it exactly; no column is derived from `is_fraud_row`, `fraud_type`, `kind` or any `labels.csv` column; history features of a company do not change when rows of another company, Split or corpus are added, removed or reordered; rows with a missing or non-binary label are excluded and counted; more than `row_exclusion_max_share` excluded aborts; a request for any column outside `PDF_FEATURE_COLUMNS` is refused with all offending names and writes no artefact; the 6a cache uses the same key scheme with `rows.csv` and the `transaction_features.py` source hash
    - _Requirements: 1.8, 10.1, 10.2, 10.3, 10.4, 10.8, 10.9, 10.10, 11.1, 11.2, 11.3, 11.4, 11.5_
  - [x] 9.2 Implement `fraud_detect/ml/retraining/features_b.py`
    - Call the existing `extract_features` on `io.BytesIO` or path; join the style profile in the same pass; skipped-document accounting; CSV cache with header check; `pdfplumber` imported inside functions
    - Make the 9.1 Node 6b tests pass
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8, 10.9, 10.10_
  - [x] 9.3 Implement `fraud_detect/ml/retraining/features_a.py`
    - `rows.csv` adapter, `build_pdf_features` call, label vector, exclusion accounting, refusal of non-PDF features, history locality by qualified `account_id`, CSV cache
    - Make the 9.1 Node 6a tests pass
    - _Requirements: 1.8, 11.1, 11.2, 11.3, 11.4, 11.5_
  - [x] 9.4 Write property tests for caches and skipped documents
    - **Property 26: Cache keys**
    - **Property 27: Skipped-document accounting**
    - **Validates: Requirements 10.3, 10.4, 10.9, 10.10**
  - [x] 9.5 Write property tests for the Node 6a adapter and history locality
    - **Property 5: Account-history features are local**
    - **Property 28: Node 6a row adapter**
    - **Validates: Requirements 1.8, 11.2, 11.3, 11.4, 11.5**

- [x] 10. Leak_Test
  - [x] 10.1 Write failing tests for the Leak_Test
    - In `tests/ml_retraining/test_leak.py`: single-feature AUC from a depth-2 tree (`random_state=0`) fitted on `train`, scored on `validation`; two runs give equal AUCs to 4 decimal places; logistic regression on standardised features and a depth-3 tree are reported; a feature or simple model at exactly 0.98 is a suspect; each suspect line has the feature name, AUC, genuine and tampered validation means, and the Layout whose per-Layout AUC is furthest from 0.5; per-Layout AUC is "not computable" for a Layout with fewer than 2 genuine or fewer than 2 tampered documents; the run is blocked (exit 3) while a suspect has no recorded `exclude` or `accept` decision, and no model file is written; an empty Split or a Split with one class stops with exit 2 naming the Split and class; the Forgery_Dataset comparison lists a feature when the absolute AUC gap is greater than 0.20 with both values, and prints a skipped notice (and continues) when the dataset is missing, with the Forgery_Dataset comparison skipped when no files are present; the leak table is joined to the audit table by feature name so each report row shows both
    - _Requirements: 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10_
  - [x] 10.2 Implement `fraud_detect/ml/retraining/leak.py`
    - `leak_test(train, valid, audit, thresholds, decisions, forgery=None)` as in the design; fixed seeds; model-free of any test or held-out data; the Forgery_Dataset comparison reads only through `sources.py` objects passed in (read-only), never writes
    - Make the 10.1 tests pass
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10_
  - [x] 10.3 Write property tests for the Leak_Test
    - **Property 14: The Leak_Test is deterministic and flags at the threshold**
    - **Property 15: Leak report rules**
    - **Validates: Requirements 4.2, 4.4, 4.5, 4.6, 4.8**

- [x] 11. Metrics, statistics and cutoff selection
  - [x] 11.1 Write failing tests for metrics and cutoffs
    - In `tests/ml_retraining/test_metrics.py`: ROC-AUC and PR-AUC equal scikit-learn; `top_k_count(n)` is `ceil(0.01 * n)` and at least 1 (pin the difference from the old `round(n * 0.01)` helper in `fraud_detect/ml/report.py`, which stays untouched); precision and recall at top k equal direct counts; `rate(numer, denom)` returns value or `None` plus both counts; a metric with fewer than 30 documents in either class is "indicative only"; a metric with a class of zero documents is "not computable" with no numeric value; percentages print with one decimal; catch rate per Tamper_Type and per Tamper_Method with counts; a (type, Split) pair with no documents is listed and has no rate; Hard_Negative (`genuine_resaved` in the corpus) and ordinary genuine sets are disjoint and together equal the genuine set; `summarise_seeds` gives mean, min, max and sample standard deviation (ddof=1), no standard deviation for one value; a run with fewer than 5 distinct completed seeds is "incomplete" and lists the failed seeds; difference is Candidate minus Current and the spread is min and max of the difference over seeds
    - In `tests/ml_retraining/test_cutoffs.py`: `choose_cutoffs` takes validation arrays only; the medium and high cutoffs each meet their false-alarm target, no lower observed genuine score also meets it, and high is at least medium; when no cutoff meets the target a `CutoffError` names the cutoff and carries the closest achievable rate; a missing class raises before any cutoff is chosen; the operating curve rows show alert rate, genuine false-alarm rate and tampered catch rate each with one decimal, with the fixed grid 0.5, 0.7, 0.8, 0.9, 0.95 printed below; per-statement level (`statement_max_row`) uses the maximum row score; results are identical when other datasets change
    - _Requirements: 2.1, 2.2, 2.6, 2.7, 2.8, 6.4, 6.7, 6.8, 9.4, 9.5, 9.8, 9.9, 9.10, 9.11, 9.12, 11.9, 13.3_
  - [x] 11.2 Implement `fraud_detect/ml/retraining/metrics.py`
    - `top_k_count`, `rate`, `summarise_seeds`, ROC and PR helpers (sklearn imported inside functions), per-type and per-method rates, indicative-only and not-computable marking, seed completeness, difference and spread
    - Make the 11.1 metric tests pass
    - _Requirements: 6.4, 6.7, 6.8, 9.4, 9.5, 9.8, 9.9, 9.10, 9.11, 9.12, 13.3_
  - [x] 11.3 Implement `fraud_detect/ml/retraining/cutoffs.py`
    - `CutoffRule` (percent targets from `RunConfig`, level `document` or `statement_max_row`), `Cutoffs`, `CutoffError`, `choose_cutoffs`, `operating_curve`; the cutoff is the lowest observed genuine validation score whose false-alarm rate is at or under the target, compared with `>=`; works under fixed seeds only
    - Make the 11.1 cutoff tests pass
    - _Requirements: 2.1, 2.2, 2.6, 2.7, 2.8, 11.9_
  - [x] 11.4 Write property tests for cutoffs
    - **Property 6: Cutoffs depend on validation only and meet the target**
    - **Property 7: The operating curve is consistent**
    - **Validates: Requirements 2.1, 2.2, 2.6, 2.8**
  - [x] 11.5 Write property tests for metrics and statistics
    - **Property 20: Rates, counts and flags**
    - **Property 21: Ranking metrics**
    - **Property 22: Seed statistics and run completeness**
    - **Validates: Requirements 6.4, 6.7, 6.8, 9.3, 9.4, 9.5, 9.8, 9.9, 9.10, 9.11, 9.12, 15.5**

- [x] 12. Checkpoint - Owner decisions on cutoffs and the Switch_Rule
  - Ask the user (reason `general-question`, design default marked as recommended) before the freeze, Switch_Rule and evaluation code is built:
    - Decision 1: Real_Style role in the Switch_Rule: `required` (default; no node can ever switch because two files are always "indicative only") or `veto_only`.
    - Decision 7: cutoff targets on validation genuine documents: medium 5.0% and high 1.0% (default), or other values.
    - Decision 4: rule 15.1(c) for `statement_E` on Node 6b: apply as written (default; expected to stop all runs until a `continue` decision), or apply it to `statement_F` false alarms only.
    - Decision 6: Node 6a baseline file: `models/txn_fraud_clf.joblib` (default) or `models/txn_fraud_clf_pdf.joblib`. Also confirm the Node 6b baseline `models/statement_forgery_clf.joblib`.
    - Decisions 8, 9 and 10: confirm the defaults (no Node 6a/6b code change to read cutoffs or show the label; the first audit proposal with the `Candidate_all11` ablation reported alongside).
  - Apply only non-default answers by editing `RunConfig` defaults and the matching tests; do not edit any node.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Freeze, held-out vault and change log
  - [x] 13.1 Write failing tests for the vault, the freeze and the change log
    - In `tests/ml_retraining/test_vault.py`: `HeldOutVault.open()` raises `VaultClosed` unless `frozen.json` exists and its recorded run hash equals the hash of `run.json`; held-out scores use cutoffs equal to `frozen.json`; the Switch_Rule hash and cutoff rule are in `run.json` before any held-out score and the final report hash must equal it (any other value is rejected); the test and held-out datasets are reachable only through the vault; a missing class on `train` or `validation` fails before fitting and the vault is never opened; the code fingerprint (SHA-256 over `fraud_detect/corpus/**`, `fraud_detect/nodes/**`, `statement_features.py`, `transaction_features.py`, `registry.py`, `reference_data.py`) differs if and only if some file differs; a change after the first held-out score is recorded with date and kind in `post_test_changes.json`, every existing held-out result is marked "no longer held out", the original values are kept, and a change before the first score creates no mark; "generator and detector code unchanged since run start: yes or no" is computable
    - _Requirements: 2.3, 2.4, 2.5, 2.7, 5.7, 14.1_
  - [x] 13.2 Implement `fraud_detect/ml/retraining/vault.py`
    - `FreezeRecord`, `HeldOutVault`, `VaultClosed`, `code_fingerprint`, post-test change recording using `decisions.py`
    - Make the 13.1 tests pass
    - _Requirements: 2.3, 2.4, 2.5, 5.7, 14.1_
  - [x] 13.3 Write property tests for the vault and change detection
    - **Property 9: The vault opens only after a valid freeze**
    - **Property 10: Changes are detected and recorded as post-test changes**
    - **Property 8: A missing class stops the run before cutoffs**
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.7, 4.10, 5.7, 14.1**

- [x] 14. Read-only sources and the training-list guard
  - [x] 14.1 Write failing tests for sources and the training-list guard
    - In `tests/ml_retraining/test_sources.py` (on the temp-folder fakes from the fixtures): Real_Style and Forgery_Dataset files are opened `rb`, hashed, and given to feature code as `io.BytesIO`; `os.stat` before and after is equal and a changed file fails the run; no file is created, modified or deleted in the source folders; a missing Real_Style file is skipped with a message naming it and the Switch_Rule input from that source is marked "missing"; a missing or unreadable Forgery_Dataset prints a notice and the datasets are marked "missing"; the report records each Real_Style path and SHA-256, never content; the forgery splits map as `test_in_layout` = `forgery_test` and `test_ood_layout` = `forgery_ood`; the answer-key label reader returns `is_tampered` per file
    - Training-list guard: any spelling of a Real_Style path (dot segments, redundant separators, trailing separator, relative form) in a training or validation list is detected if and only if its normalised absolute form equals a Real_Style path, the run aborts before any fit, and existing model and report files are unchanged; with generator-corpus-only training the lists contain zero Forgery_Dataset documents; training modules do not import `sources` (source scan); a test fails if the normalised path of a Real_Style file appears in any training list built by the pipeline
    - Tests needing the real files (stat and hash equality before and after, no new file in the folder) run only when `RETRAIN_REAL_STYLE_DIR` and `RETRAIN_FORGERY_DIR` are set, and are skipped otherwise
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9, 4.9, 17.10_
  - [x] 14.2 Implement `fraud_detect/ml/retraining/sources.py`
    - Read-only access through `safe_io.ReadOnlySource`, dataset registry (seven dataset ids and their Source), answer-key reader, `assert_not_in_training(paths)` guard, optional Forgery_Dataset training mode that fits on its `train` Split only (selection on `validation`, final metrics on `test` and the out-of-layout Split)
    - Make the 14.1 tests pass
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
  - [x] 14.3 Write property test for the training-list guard
    - **Property 24: Real_Style paths never reach a training or validation list**
    - **Validates: Requirements 7.3, 7.8, 7.9**

- [x] 15. Node 6b candidate training
  - [x] 15.1 Write failing tests for Node 6b training
    - In `tests/ml_retraining/test_train_b.py` (Mini_Corpus, small settings): the candidate uses the same `HistGradientBoostingClassifier` family and settings as `train_statement.py` (read with `ast`); fitted on the `train` Split only; model input columns equal the audited ordered list (no dropped feature); a `neutralise` step is inside the pipeline; the bundle has `model`, `thresholds`, `size_ranges` (the 0.01 and 0.99 quantiles of the training documents' `SIZE_FEATURES`, same `GUARD_QUANTILES`), `feature_names`, `audit_version`, `audit_digest`, `master_seed`, `held_out_layout`, `size_guard`, `cutoff_rule`, `validation_curve`, `model_label` "model trained on synthetic data" and `synthetic_data`; cutoffs come from validation only; the file is written under `data/private/ml_retraining/` through `safe_io` and no file under `models/` is touched or modified; the candidate loads in a clean interpreter without `fraud_detect.ml.retraining` in `sys.modules`, and the unchanged `ml_statement.assess` returns a probability and honours `size_ranges` (empty `size_ranges` means guard off); the informational `Candidate_all11` ablation is trained on the same corpus and flagged "not used by the Switch_Rule"; `train` refuses while any gate (integrity, or waivable and not waived) or Leak_Test suspect is open; a candidate that fails to load or score is written with a failed marker; fitting runs under `threadpoolctl.threadpool_limits(1)`; a release candidate is fitted on the pooled `train` Splits of all completed seeds with cutoffs chosen on the pooled `validation` Splits; the GNN line "GNN: not retrained, corpus has no transfer graph" is printed
    - _Requirements: 2.1, 2.6, 3.4, 3.7, 7.2, 10.5, 10.6, 10.7, 12.2, 17.14_
  - [x] 15.2 Implement `fraud_detect/ml/retraining/train_b.py`
    - Per-seed candidates, release candidate, `Candidate_all11` ablation, bundle writer, validation operating curve printout, Size_Guard ranges from training documents; model-family and settings imported from or equal to `train_statement.py` (equality test)
    - Make the 15.1 tests pass
    - _Requirements: 2.1, 2.6, 3.4, 3.5, 3.7, 10.5, 10.6, 10.7, 12.2_
  - [x] 15.3 Write property test for the audit-decides-input rule
    - **Property 12: The audit decides the model input exactly**
    - **Validates: Requirements 3.4, 3.5, 3.7, 10.2, 10.7, 11.1**

- [x] 16. Node 6a candidate training and GNN lookup accounting
  - [x] 16.1 Write failing tests for Node 6a training and the GNN accounting
    - In `tests/ml_retraining/test_train_a.py`: pipeline parameters equal `train_transaction.build_pipeline("pdf")`; `feature_set` is `"pdf"` and `feature_columns` is the ordered audited subset of `PDF_FEATURE_COLUMNS`; the feature set name and ordered names are recorded; a refused non-PDF feature writes no artefact; the cutoff rule is per statement (the maximum row score; target on validation genuine statements); the bundle holds `thresholds` for the report while noting the live node uses 0.5 and 0.8 (finding F9); statements with zero fraud rows are listed with their count and left out of the Node 6a detection denominator; the candidate is read by the unchanged `ml_transaction` helpers; no `torch` import
    - In `tests/ml_retraining/test_gnn_eval.py`: share of held-out UENs found in a fake score store equals |held-out ∩ store| / |held-out| to one decimal; every absent UEN is recorded as "no score" with a count and no substitute number; combined flag equals GBM OR GNN, equals GBM alone when the GNN has no score; GBM alone, GNN alone ("not computable" without a scored statement) and combined each report the per-statement false-alarm rate and the fraud detection rate on the same statements; no module imports `torch` or `torch_geometric`
    - _Requirements: 11.1, 11.6, 11.7, 11.8, 11.9, 11.10, 17.4_
  - [x] 16.2 Implement `fraud_detect/ml/retraining/train_a.py` and `gnn_eval.py`
    - `train_a.py`: per-seed and release candidates on the PDF_Feature_Set, statement-level cutoffs; `gnn_eval.py`: lookup accounting against the Current score store (store read through the existing `gnn_lookup`, no torch), combination logic
    - Make the 16.1 tests pass
    - _Requirements: 11.1, 11.6, 11.7, 11.8, 11.9, 11.10_
  - [x] 16.3 Write property tests for statement-level flags and GNN accounting
    - **Property 29: Statement-level flags and model combination**
    - **Property 30: GNN lookup accounting**
    - **Validates: Requirements 11.7, 11.8, 11.9, 11.10**

- [x] 17. Awkward_Set
  - [x] 17.1 Write failing tests for the Awkward_Set
    - In `tests/ml_retraining/test_awkward.py`: `AWKWARD_VERSION` and `AWKWARD_MASTER_SEED = 20260701` are recorded; five case types (`seasonal_growth`, `legit_resave`, `instalments`, `long_12_months`, `round_payroll_rent`) each with at least 3 genuine and at least 3 tampered documents, 15 genuine and 15 tampered in total; every tampered document shares company profile and Layout with a genuine document of its case type, with one tamper applied (rotating `edited_amount`, `fabricated_receipts`, `metadata_inconsistency`); each stated figure holds on the ledgers (peak at least 2 times trough and last 3 months at least 1.10 times first 3 months; Producer from the export list and ModDate later than CreationDate; at least one invoice reference on two receipts; exactly 12 consecutive months; at least 80% of payroll and rent at multiples of 100); two builds with the same seed are identical; company ids `AW-<case>-<k>`; UENs drawn through `make_company(..., is_taken=checker.is_taken)` against training and validation corpus UENs so a clash is redrawn; the set is built in memory (`io.BytesIO`), creates no temporary file or folder, and commits no PDF; a case type that cannot meet its figure raises `AwkwardBuildError` naming the type and counts, and the pipeline stops before the Switch_Rule decision with the Current model unchanged; the review record (reviewer name, date, version) is read from the report, and results are marked "unreviewed" without a record for the current version; the Awkward_Set is not importable from training modules (source scan); the builder reproduces `CorpusBuilder._build_document` output for a standard document
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 17.11_
  - [x] 17.2 Implement `fraud_detect/ml/retraining/awkward.py`
    - Recipes as confirmed at checkpoint 3 (design defaults unless the user chose `awkward-v2`); row rewrite with `dataclasses.replace` inside the retraining package only; no generator code is changed; `AwkwardBuildError`; review-record reader
    - Make the 17.1 tests pass
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.7, 8.8, 8.9_
  - [x] 17.3 Write property test for the Awkward_Set structure
    - **Property 25: Awkward_Set structure**
    - Use 10 to 20 examples (PDF rendering is slow); say so in the test docstring
    - **Validates: Requirements 8.1, 8.2, 8.7**

- [x] 18. Baseline scoring and evaluation
  - [x] 18.1 Write failing tests for baseline scoring and evaluation
    - In `tests/ml_retraining/test_baseline.py` (fake Current models in a temp folder): the Current Node 6b model is scored through the live `ml_statement` helpers; guard off is a copy of the bundle without `size_ranges`; the Current Node 6a model is fed the columns it was trained on through `ml_transaction._feature_spec`, with a pinning test that fails if that private function is renamed or changes behaviour; a missing model file gives "no baseline" and a model that cannot load or score a dataset gives "baseline failed" naming node and dataset, with Candidate values still reported and Current and difference cells empty
    - In `tests/ml_retraining/test_evaluation.py`: Candidate and Current are scored on identical lists of document ids per dataset; every result row carries exactly one Source (`forgery_dataset`, `generator_corpus`, `real_style`, `awkward_set`) and one dataset id of the seven; aggregating across Sources is rejected; held-out datasets are read only through the vault; Node 6b flag is probability at or above the medium cutoff, abstention counts as not flagged and is excluded from AUC with the abstained share reported; results are written once and atomically as `results/<node>/<dataset>/<seed>.json` and a re-run resumes at the first unwritten unit; per-Tamper_Type and per-Tamper_Method catch rates with counts; difficulty rows are replaced by the line "difficulty: not recorded in the corpus manifest (generator 1.0.0)"; Hard_Negative and ordinary genuine false-alarm rates separately; the Hard_Negative set in the Awkward_Set is all its genuine documents; Real_Style reported per file; the Forgery_Dataset rows carry the asymmetry statement; the 12-month and 6-month genuine false-alarm rates come from `n_months` over the `test` Splits and the held-out-seed corpus; Node 6a is judged per statement and tampered documents with zero fraud rows are counted and excluded from the detection denominator; each metric is also given "as deployed" (Current at stored cutoffs and guard; Node 6a at fixed 0.5 and 0.8) beside the decisive matched-cutoff basis
    - _Requirements: 6.4, 6.5, 6.7, 8.6, 9.1, 9.2, 9.6, 9.7, 9.8, 9.9, 9.13, 11.9, 12.5, 13.1, 13.2, 13.4, 13.5, 13.6_
  - [x] 18.2 Implement `fraud_detect/ml/retraining/baseline.py`
    - Current-model loading and scoring through live helpers; "no baseline" and "baseline failed" states; guard-off scoring; "as deployed" and matched-cutoff bases
    - Make the 18.1 baseline tests pass
    - _Requirements: 13.1, 13.5, 13.6, 12.3_
  - [x] 18.3 Implement `fraud_detect/ml/retraining/evaluation.py`
    - Per-dataset, per-seed evaluation of per-seed and release candidates and the Current models; metrics through `metrics.py`; Size_Guard on and off; monitor call before every result is written; result files written once and atomically
    - Make the 18.1 evaluation tests pass
    - _Requirements: 6.4, 6.5, 6.7, 8.6, 9.1, 9.2, 9.6, 9.7, 9.8, 9.9, 9.13, 11.9, 12.3, 13.1, 13.2, 13.3, 13.4_
  - [x] 18.4 Write property tests for source separation, comparison and baseline states
    - **Property 23: Sources and datasets never mix**
    - **Property 33: Comparison arithmetic and baseline states**
    - **Validates: Requirements 9.1, 9.2, 9.6, 13.1, 13.3, 13.5, 13.6**

- [x] 19. Size_Guard analysis
  - [x] 19.1 Write failing tests for the Size_Guard analysis
    - In `tests/ml_retraining/test_size_guard.py`: outside-range count and percentage per Split for the Current and the Candidate guard equal direct counts; Candidate ranges equal the 0.01 and 0.99 quantiles of the training documents' size features; with the guard on, the flagged set is a subset of the guard-off flagged set and the abstained share is reported; the recommendation is exactly one of keep Current guard, use Candidate guard, remove guard, with the figures it rests on; "remove" only if the Candidate 12-month genuine false-alarm rate minus its 6-month rate is at most 5 points (and the guard-off rate is not more than 5 points above guard-on on each Source with at least 30 genuine documents); when that condition fails the report states the failure with both rates; the pipeline never changes a guard in any model file, and `switch` for Node 6b refuses without an explicit `--size-guard {candidate,off,current}`
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  - [x] 19.2 Implement `fraud_detect/ml/retraining/size_guard.py`
    - Guard analysis and `recommend_guard` as in the design; uses `GUARD_QUANTILES` from `train_statement.py`
    - Make the 19.1 tests pass
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  - [x] 19.3 Write property tests for the Size_Guard
    - **Property 31: Size_Guard arithmetic**
    - **Property 32: Size_Guard recommendation**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6**

- [x] 20. Implausible-measurement monitor and stop lock
  - [x] 20.1 Write failing tests for the monitor
    - In `tests/ml_retraining/test_monitor.py`: rule (a) fires only for a Candidate ROC-AUC of 0.99 or more on `test_unseen_layout`; rule (b) only for a ROC-AUC below 0.5 with at least 30 documents of each class; rule (c) only for a Real_Style Candidate flag on the opposite side of its cutoff from the answer-key label (and the configured alternative for decision 4); rule (d) only for a document count differing from the manifest count by 1 or more; a firing writes `investigation.json` with measurement name, value, Source, Split, Master_Seed and both class counts, creates `STOP`, prints the same fields, exits 4; results already written stay unchanged; while `STOP` exists `train`, `compare`, a new audit version, and any cutoff or feature change are refused with the blocked change named; `decide investigation --decision continue|abort` records the decision, `continue` removes `STOP` and the run resumes at the next unwritten unit; a measurement of a failed or stopped seed is never re-run under a different seed; the seed ledger (`seeds.json`) lists every seed attempted, including failed and discarded ones, with the outcome; a decision to change the generator or a feature is recorded as a post-test change with the date, the triggering measurement and value, the reason and the description
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_
  - [x] 20.2 Implement `fraud_detect/ml/retraining/monitor.py`
    - `observe(measurement)`, `MeasurementStop`, `STOP` lock, investigation record, seed ledger, blocked-change messages
    - Make the 20.1 tests pass
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_
  - [x] 20.3 Write property test for the monitor
    - **Property 37: The implausible-measurement monitor**
    - **Validates: Requirements 15.1, 15.2, 15.4**

- [x] 21. Switch_Rule, switch and restore
  - [x] 21.1 Write failing tests for the Switch_Rule and model operations
    - In `tests/ml_retraining/test_switch_rule.py`: `SwitchRule` has `version`, decisive sources (`heldout_layout`, `heldout_seed`, `forgery_ood`, `real_style`), `real_style_role` (`required` default, `veto_only`), `hard_negative_block_points`; `text()` and `sha256()` are stable; Node 6b "at least as good" is AUC not lower by more than the Candidate's own across-seed standard deviation, catch rate at the medium cutoff not lower, false-alarm rate at the medium cutoff not higher; Node 6a is PR-AUC not lower and per-statement false-alarm rate not higher; equal values pass with no rounding before comparison; a Source marked missing or indicative-only is "not decisive" and never passes; any failed decisive Source gives keep; no available decisive Source gives keep with that statement; with the default `required`, the two Real_Style files make the result always keep and the report says so in its first lines; with `veto_only` Real_Style can only block; the release candidate must pass the same checks; "no baseline" and "baseline failed" give keep; improving a Candidate metric never turns a pass into a fail; the Awkward_Set never passes a node, blocks only when its definition version has a recorded review and the Candidate Hard_Negative false-alarm rate exceeds the Current by more than 10 points, and unreviewed results have no effect; for Node 6a keep adds the text "keep the GBM and the GNN lookup off in the live app"; the pipeline never copies a model over a Current model by itself
    - In `tests/ml_retraining/test_modelops.py` (temp models folder and temp backup folder): `switch` copies the Current model to the backup folder, verifies the backup SHA-256 equals the Current hash, records the report id and both hashes in `index.json`, verifies the Candidate hash against the report, then replaces; a failed backup write or hash mismatch stops and leaves the Current model unchanged; `restore` copies the backup back and verifies the hash equals the recorded backup hash; `restore` with no backup stops and leaves the Current model unchanged; the backup folder is `models/backup/` and passes `ensure_ignored`; Node 6b `switch` requires `--size-guard`
    - _Requirements: 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.11, 14.12, 14.13, 14.14, 8.9, 13.5, 13.6, 12.7_
  - [x] 21.2 Implement `fraud_detect/ml/retraining/switch_rule.py`
    - `SwitchRule`, `decide(rule, node, evidence)` as a pure function, evidence and node-decision data classes; hash written to `run.json` before evaluation
    - Make the Switch_Rule tests in 21.1 pass
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.13, 14.14, 13.5, 13.6_
  - [x] 21.3 Implement `fraud_detect/ml/retraining/modelops.py`
    - `switch` and `restore` with SHA-256 backup and verification, backup index, `--size-guard` handling that edits only the installed copy, never a file in `data/private/ml_retraining/`
    - Make the model-operation tests in 21.1 pass
    - _Requirements: 14.8, 14.9, 14.10, 14.11, 14.12, 12.7_
  - [x] 21.4 Write property tests for the Switch_Rule
    - **Property 34: The Switch_Rule decision**
    - **Property 35: The Awkward_Set blocks and never passes**
    - **Validates: Requirements 8.9, 14.2, 14.3, 14.4, 14.5, 14.6, 14.13, 14.14**
  - [x] 21.5 Write property test for switch and restore
    - **Property 36: Switch and restore round trip**
    - **Validates: Requirements 14.9, 14.10, 14.11, 14.12**

- [x] 22. Comparison report
  - [x] 22.1 Write failing tests for the report
    - In `tests/ml_retraining/test_comparison_report.py`: the first 10 lines of both the Markdown file and the console output contain the words "synthetic data" and the circularity statement before any result table; the Switch_Rule text and hash, the cutoff rule with targets, the awkward-set seed and version, and the Real_Style limitation ("no switch is possible in this mode" when `real_style_role` is `required`) appear before any result; every result row from a Source other than `real_style` carries the "synthetic data, not real-world accuracy" marker; every Forgery_Dataset row carries the statement that the Current model may have been fitted on similar documents; each audit row is shown with its leak result in the same row; the report lists Master_Seeds of each corpus, the held-out layout, document counts per Split and per Layout (summing to the total), all seeds attempted with outcome, "incomplete" with failed seeds when fewer than 5 completed, skipped documents per Split, Real_Style path and SHA-256 per file, waivers and decisions from `decisions.json`, post-test changes, the code-unchanged yes/no line, the GNN coverage line (expected 0.0%), per-seed mean, min, max and standard deviation, per-metric class counts, "indicative only" and "not computable" marks, the `Candidate_all11` ablation labelled informational, and results per Held_Out_Layout when varied; the Switch_Rule hash in the report equals the one in `run.json`; the console and the Markdown file and `comparison_report.json` agree
    - _Requirements: 1.9, 2.3, 3.6, 7.5, 9.1, 9.3, 9.4, 9.5, 9.7, 9.10, 9.11, 9.12, 10.9, 13.4, 14.1, 15.5, 16.5, 16.9_
  - [x] 22.2 Implement `fraud_detect/ml/retraining/comparison_report.py`
    - Report model, Markdown, JSON and console renderers, banner builder; header (Switch_Rule, cutoff rule, banner) is written before evaluation starts and evaluation results are appended; all writes go through `safe_io`; named to avoid a clash with `fraud_detect/ml/report.py`
    - Make the 22.1 tests pass
    - _Requirements: 2.2, 2.3, 3.6, 9.1, 9.3, 9.4, 9.5, 9.10, 9.11, 9.12, 13.4, 14.1, 15.5, 16.5, 16.9_
  - [x] 22.3 Write property test for the report banner and markers
    - **Property 38: Report banner and markers**
    - **Validates: Requirements 3.6, 13.4, 16.5, 16.9**

- [x] 23. Command-line wiring and orchestration
  - [x] 23.1 Write failing tests for the commands
    - In `tests/ml_retraining/test_cli.py` (Mini_Corpus, fake Current models, injected `is_ignored`, injected clock): each of `audit`, `leak-test`, `train`, `compare`, `switch`, `restore` (and `init`, `build`, `check`, `decide`) exits 0 on `--help` and on success and non-zero on a failing input with the documented exit code; a failing gate prints its offenders and leaves no model file; `init` writes `run.json` with SHA-256, code fingerprint, Switch_Rule text and hash, cutoff rule and the report header before any evaluation; `build` calls `CorpusBuilder` for each seed plus the held-out-seed corpus with the same hold-out layout and writes only under `data/private/ml_retraining/corpora/`; `check --plan` prints the document counts from the sizing formula; `audit` writes `audit/audit-v1.json`; `leak-test` exits 3 on a suspect until `decide leak-feature` records `exclude` or `accept`; `train` refuses (exit 3) with the named list while the capability or subtle gate fails and is not waived, then runs after a `decide waive-gate` entry which the report prints; `train` fits only after all integrity gates, the Leak_Test and freeze checks pass and writes `frozen.json`; `compare` refuses without a valid `frozen.json`; a lock file refuses a second run on the same run folder; an interrupted run resumes at the first unwritten unit; plumbing run `init`, `check`, `audit`, `leak-test`, `train` (6b and 6a), `compare` on the Mini_Corpus with small test thresholds produces a report whose first 10 lines carry the banner; `models/` is byte-identical before and after the plumbing run and `git status --porcelain` lists no PDF, `labels.csv`, `rows.csv`, model or cache file; no network call is made; two runs with the same inputs and seeds give exactly equal cutoffs and metrics
    - _Requirements: 1.7, 2.3, 2.7, 4.4, 5.3, 5.6, 7.9, 14.1, 14.8, 15.4, 17.5, 17.6, 17.7, 17.9, 17.13, 17.14_
  - [x] 23.2 Implement `fraud_detect/ml/retraining/cli.py` fully
    - Replace the stubs with the command implementations that call the modules above in the order of the data-flow diagram; exit codes as documented; `decide` kinds `leak-feature`, `waive-gate`, `investigation`, `awkward-review`; lock file; resume; seed ledger; `train` and `compare` print the validation operating curve, the freeze record and the console banner; no `torch`, `sklearn`, `pdfplumber` import at package import time
    - Make the 23.1 tests pass
    - _Requirements: 1.7, 2.3, 2.6, 2.7, 4.4, 5.3, 5.6, 7.9, 9.3, 14.1, 14.8, 15.4, 17.5, 17.7, 17.13, 17.14_
  - [x] 23.3 Write property test for determinism
    - **Property 41: Determinism**
    - Use 10 to 20 examples for the Mini_Corpus training part; say so in the test docstring
    - **Validates: Requirements 17.14**

- [x] 24. Hygiene tests
  - [x] 24.1 Write and run the hygiene tests
    - In `tests/ml_retraining/test_hygiene.py`: `git check-ignore` matches `data/private/ml_retraining/x`, `models/backup/x.joblib` and every default output path; no PDF, `labels.csv`, `rows.csv`, corpus file, model file, cache file or snapshot file is tracked by git (`git ls-files`); the package source contains none of the strings `statement_E.pdf`, `statement_F.pdf` or `ANSWER_KEY` outside `sources.py` and the config defaults, and no training module imports `sources` or `awkward`; no module in `fraud_detect/ml/retraining/` imports `torch` or `torch_geometric` (source scan) and the live web app entry point imports with both blocked in `sys.modules`; importing `fraud_detect.ml.retraining` imports no `sklearn`, `pdfplumber`, `torch`; no module imports `urllib`, `socket` or `requests` directly and a fixture that makes network opens raise does not break any test; Nodes 1 to 4, `registry.py`, `reference_data.py` and the generator package hashes equal `protected_hashes.json` and Node 4 weights and thresholds equal the pinned values; `requirements.txt` and any runtime dependency list are unchanged; tests read no file outside the repository and the temp folder; a test needing a Real_Style_File or the Forgery_Dataset is skipped (not failed) when it is absent; the `uen_check` module is imported by `group_checks`
    - _Requirements: 3.7, 7.6, 7.8, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.9, 17.10, 17.11_

- [x] 25. README and report framing
  - [x] 25.1 Write failing tests for the README section
    - In `tests/ml_retraining/test_readme.py` (text scan of `FRAUD_DETECT_README.md`): a section "Stage B: ML retraining" exists; every paragraph that describes Corpus-derived results contains the words "synthetic data"; the first results table is in the same paragraph as the statement that results are an end-to-end demonstration and not real-world accuracy; the circularity risk is stated and all seven mitigations are named (held-out layouts, held-out Master_Seeds, randomised styling, the Leak_Test, the Real_Style_Files, the Switch_Rule, the Awkward_Set); it states that only two labelled real-style statements exist (`statement_E`, `statement_F`), that this is too small to estimate accuracy and that results on them are indicative only; it lists the Switch_Rule outcome and the model in use for Node 6a and Node 6b after Stage B; it states that Node 4 weights and thresholds are not changed; it documents each command (`audit`, `leak-test`, `train`, `compare`, `switch`, `restore`) and the exit codes; it states that the "model trained on synthetic data" label is stored in the bundle and is not yet shown in the demo (decision 9); any score on a Source other than `real_style` is marked as measured on synthetic data and not real-world accuracy
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.6, 16.7, 16.8, 16.9, 17.13_
  - [x] 25.2 Update `FRAUD_DETECT_README.md`
    - Add the section "Stage B: ML retraining" in plain language: purpose, the seven mitigations, the commands and exit codes, the stop-and-decide flow (`decide`), the open decisions and what was chosen, the honest limits from the design (including that, with `real_style_role = required`, no node can switch), and a results table with placeholders "not yet run" (the Switch_Rule outcome per node is filled in by the user after the manual run); state that Node 4 is unchanged and which model each node uses (the Current model, until the user runs `switch`)
    - Make the 25.1 tests pass
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.6, 16.7, 16.8, 16.9_

- [x] 26. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run the whole existing suite together with `tests/ml_retraining`: the earlier passed count recorded in task 1.1 is unchanged, no existing test newly fails, skips are unchanged apart from tests that skip when Real_Style_Files or the Forgery_Dataset are absent; the new tests pass on the Mini_Corpus alone.
  - Hygiene: `git status --porcelain` lists no PDF, `labels.csv`, `rows.csv`, corpus, model, cache or snapshot file; `git diff` shows no change to Nodes 1 to 4, `registry.py`, `reference_data.py`, `fraud_detect/corpus/`, the Node 4 weights, `requirements.txt` or any earlier test; delete any temporary folders.
  - Tell the user which manual steps remain (see Notes). Do not run them.
  - _Requirements: 17.1, 17.2, 17.3, 17.6, 17.8, 17.9, 17.10_

## Notes

- Tasks marked with `*` are optional property tests and can be skipped for a faster first version. Unit and example tests are not optional.
- Each group is test first: the x.1 task writes tests that fail until the matching implementation task is done.
- Property tests use `hypothesis`. Pure properties use at least 100 examples; properties that render PDFs or train a model (25 and the training part of 41) use 10 to 20 examples and say so. Tag format: `Feature: ml-retraining, Property N: ...`, with N and the title taken from the design.
- Checkpoints 3, 7 and 12 ask the user. They are where the open decisions are put to the user: the Awkward_Set recipes, the capability-gate waiver (decision 3), pooled tamper-type minimums (5), thresholds (2), `real_style_role` (1), cutoff targets 5% and 1% (7), the statement_E stop rule (4), the Node 6a baseline file (6), and decisions 8, 9 and 10. Implementation tasks build the design's default behaviour and keep every option in `RunConfig`.
- The pipeline can refuse to train on the generator as it stands (capability gate, exit 3). That is intended. Do not change the generator to get past it.
- Group 2 (spike) comes first on purpose: the design has not tried whether the existing simulator knobs reach the seasonal and growth figures.
- Files that more than one task edits are in different waves (`group_checks.py` in tasks 4.3 and 5.2).
- All outputs go to `data/private/ml_retraining/` and `models/backup/` (git-ignored). Never commit a PDF, `labels.csv`, `rows.csv`, a corpus file, a cache file or a model file.
- `statement_E`, `statement_F`, the answer key and the forgery dataset are never read by tests (except skipped integration tests behind `RETRAIN_REAL_STYLE_DIR` and `RETRAIN_FORGERY_DIR`), never trained on, and never committed.
- Manual steps for the user, after the final checkpoint (not tasks, not run by the agent): (1) `python -m fraud_detect.ml.retraining init` with chosen options; (2) `check --plan`, then `build` for the 5 training seeds and the held-out seed (hours; see the design's Performance Considerations); (3) `audit`, `leak-test`, resolving any stop with `decide`; (4) `train`, then `compare` with `--real-style-dir` and `--forgery-dir` set; (5) read the report; (6) `switch` or `restore` only if wanted. If a measurement looks wrong, stop and record a decision; do not re-run under another seed.
- If a measurement looks wrong while working on a task, stop and ask the user; do not tune the generator or any node to change a number.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "4.1", "6.1", "8.1", "11.1"] },
    { "id": 2, "tasks": ["1.3", "4.2", "6.2", "8.2", "11.2"] },
    { "id": 3, "tasks": ["1.4", "2.2", "4.3", "4.4", "5.1", "6.3", "8.3", "11.3", "11.5", "13.1", "14.1", "19.1", "20.1", "21.1"] },
    { "id": 4, "tasks": ["4.5", "5.2", "6.4", "6.5", "8.4", "9.1", "10.1", "11.4", "13.2", "14.2", "19.2", "20.2", "21.2", "21.3"] },
    { "id": 5, "tasks": ["5.3", "9.2", "9.3", "10.2", "13.3", "14.3", "17.1", "19.3", "20.3", "21.4", "21.5", "22.1"] },
    { "id": 6, "tasks": ["9.4", "9.5", "10.3", "15.1", "16.1", "17.2"] },
    { "id": 7, "tasks": ["15.2", "16.2", "17.3"] },
    { "id": 8, "tasks": ["15.3", "16.3", "18.1"] },
    { "id": 9, "tasks": ["18.2"] },
    { "id": 10, "tasks": ["18.3"] },
    { "id": 11, "tasks": ["18.4", "22.2"] },
    { "id": 12, "tasks": ["22.3", "23.1"] },
    { "id": 13, "tasks": ["23.2"] },
    { "id": 14, "tasks": ["23.3", "24.1", "25.1"] },
    { "id": 15, "tasks": ["25.2"] }
  ]
}
```
