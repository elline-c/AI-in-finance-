# Implementation Plan: Generator Styling Variety

## Overview

Add an opt-in Balanced_Styling mode (`style_version` 2) and a subtle amount-change tamper to the corpus generator, keep the default Legacy_Styling byte-for-byte reproducible, then switch the ml-retraining build to the new mode. The work is test-first in small steps. The first task records hashes of the unchanged generator so that every later change can be checked against them.

Language: Python (matches the repository and the design). Test command, run from `/Users/weiyang01/AI-in-finance-`:

```
env -u AUDIT_SIGNING_KEY_PATH -u FRAUD_ENABLE_TXN_MODEL /Users/weiyang01/AI-in-finance-/.venv/bin/python -m pytest -q -p no:cacheprovider -W ignore
```

Work on `tests/corpus` first. Run `tests/ml_retraining` from group 8 on. Run the whole suite at each checkpoint and compare with the counts recorded in task 1.2.

Out of scope (do not do):
- Changing Nodes 1 to 4, `fraud_detect/registry.py`, `fraud_detect/reference_data.py`, `fraud_detect/ml/retraining/style_profile.py`, `corpus_gates.py` or any other file of the Retraining_Pipeline except the three named in task 9.1.
- Editing `tests/ml_retraining/protected_hashes.json`, the pinned Feature_Audit digest `audit-v1`, any recorded `run.json` fingerprint or any Golden_Hashes entry after task 1.1 (these are deliberate re-pins for the project owner; see the design's Re-pin checklist).
- Making Styling_Version 2 the default, restyling the Awkward_Set, adding columns to `labels.csv` or changing `GENERATOR_VERSION`, `TAMPER_TYPES` or `LABELS_COLUMNS`.
- Committing any PDF, corpus file or model file. Running the full-size ml-retraining `check` (a manual step for the project owner).
- Adding any runtime dependency.

Hard rules for every task:
- All strings and data are fictional. No real company, bank, person or street.
- Legacy_Styling output must not change. If the Golden_Hashes test fails, the change is wrong; fix the change, not the hashes.
- If a measurement looks wrong (a layout overlap, a Node 1 mismatch), stop and ask the user. Do not move text or relax a test to make a number pass.
- Honest framing: all results are synthetic data.

## Tasks

- [x] 1. Safety net before any generator change
  - [x] 1.1 Record Golden_Hashes of the unchanged generator
    - Run this task before any other task edits a file under `fraud_detect/corpus/`
    - Create `tests/corpus/golden_tool.py` with `collect(corpus_dir) -> dict` and a command line `--write PATH`. For one corpus folder it records: SHA-256 of `labels.csv`, `rows.csv`, `companies.json`, `corpus_manifest.json`; SHA-256 of each `manifests/<doc_id>.json`; for each PDF the `text_layer_sha256` from `fraud_detect.corpus.labels` and a canonical dump of Title, Subject, Author, Creator, Producer, CreationDate, ModDate and the sorted font names. Raw PDF bytes are not hashed
    - Build two cases with the Mini_Corpus settings of `tests/ml_retraining/conftest.py` (6 Layouts x 2 genuine + 2 tampered, 6 months, hold-out `two_col_ddmmyy_cr`): Master_Seed 20260801 with the default mix, and 20260802 with the six-type `MINI_TAMPER_MIX`. Use `CorpusBuilder` in a pytest temporary folder
    - Run the tool once on the unchanged code and save `tests/corpus/legacy_golden.json` with `recorded_from` set to the current `git rev-parse HEAD` and the date
    - Create `tests/corpus/test_legacy_golden.py` that rebuilds both cases and compares every recorded value; a failure message names the case, the file and the document id
    - _Requirements: 1.2, 1.8, 7.9, 8.5, 8.6_
  - [x] 1.2 Record baseline test counts
    - Run `tests/corpus` and then the whole suite once; write the passed, skipped and failed counts (without the tests added by this feature) in the Notes of this file under "Baseline counts"
    - If a test already fails before any change, list its name there; later checkpoints compare against this list
    - _Requirements: 8.4_

- [x] 2. Schema, builder validation and command line
  - [x] 2.1 Write failing tests for the schema additions and write the schema additions
    - In `tests/corpus/test_style_schema.py`: `STYLE_VERSIONS == (1, 2)`; `EXTRA_TAMPER_TYPES == ("subtle_amount_edit",)`; `ALL_TAMPER_TYPES == TAMPER_TYPES + EXTRA_TAMPER_TYPES`; `StyleKnobs()` equals the legacy default values in every old field and has the defaults `creator=""`, `font_regular="Helvetica"`, `font_bold="Helvetica-Bold"`, `print_address=False`, `address_wording=0`, `producer_family=""`, `style_version=1`; `BuildConfig(...).style_version == 1`; `GENERATOR_VERSION`, `TAMPER_TYPES` (16) and `LABELS_COLUMNS` (25) unchanged
    - In `fraud_detect/corpus/schema.py`: add the three constants, the seven defaulted `StyleKnobs` fields after `second_address_line`, and `BuildConfig.style_version: int = 1` as the last field. Update the module docstring bullets for `StyleKnobs` and `BuildConfig`
    - _Requirements: 1.1, 1.6_
  - [x] 2.2 Builder validation, counts, manifest keys and `--style-version`
    - Write failing tests first in `tests/corpus/test_builder_style_v2.py`: **Property 1** (for any integer other than 1 and 2, `CorpusBuilder` raises `ValueError` naming 1 and 2 and writes nothing; 1 and 2 are accepted), **Property 16** (any string outside `ALL_TAMPER_TYPES` in `tamper_mix` raises `ValueError` naming it; names from `ALL_TAMPER_TYPES` are accepted), the CLI flag (`main(["--style-version","2", ...])` writes `styling_version: 2`; `--style-version 3` returns 2 and prints `Refused:` on stderr)
    - In `fraud_detect/corpus/builder.py`: validate `style_version`; check `tamper_mix` names against `ALL_TAMPER_TYPES`; in `_counts` add each Extra_Tamper_Type that is in the mix to `per_tamper_type` for every Split; in `_warnings` pass `TAMPER_TYPES` plus those types to `find_gaps`; write `styling_version` into `corpus_manifest.json` and a `Styling version : 2 (balanced)` README line only when the version is 2
    - In `fraud_detect/corpus/build.py`: add `--style-version` (type int, default 1) and pass it to `BuildConfig`
    - Do not yet change `_build_document` (task 5.1)
    - _Requirements: 1.3, 1.4, 1.5, 1.7, 7.9, 7.10, 9.4_

- [x] 3. Balanced styling module
  - [x] 3.1 Write failing tests for `styling.py`
    - In `tests/corpus/test_styling_v2.py`: pool sizes and membership (5 distinct Creator strings including the legacy one; 4 base, 4 export, 4 editor Producers); the Creator_Pool keyword scan against `visual_forensics._EDITING_SOFTWARE`, the whole word `editor`, and `statement_features._EDITING_SOFTWARE`, in lower case; the `styling.py` source contains `VF_EDIT_SOFTWARE` and `VF_MODIFIED_AFTER_CREATION` in a comment; distribution shares over 20000 draws (family within 2 points of 65/25/10, regular font within 2 points of 65/35, each creator within 2 points of 20, each wording within 2 points of 33.3); `make_style` output is unchanged after importing `styling`
    - Example tests for the normaliser: `genuine_resaved` and `metadata_inconsistency` notes get the added sentence and keep the old text; `font_mismatch` notes name the drawn font and `font_inconsistent` stays true
    - _Requirements: 2.1, 2.7, 3.1, 3.4, 4.1, 4.6, 6.2, 10.6_
  - [x] 3.2 Implement `fraud_detect/corpus/styling.py`
    - Catalogues and constants as in the design: `FAMILY_WEIGHTS`, `PRODUCER_CATALOGUE`, `CREATOR_POOL`, `DECORATIVE_FONTS`, `FOREIGN_FONT_POOL`, `ADDRESS_TEMPLATES`
    - `make_style_v2(master_seed, *parts)`: the legacy `make_style` result with Producer, family, Creator, regular font, wording, `print_address=True` and `style_version=2` replaced from the stream `("style2", *parts)`; draw order family, producer, creator, font, wording
    - `address_text(style)`, `normalise_tamper_result(result, *, master_seed, doc_id)` (drops the `producer` pair from `set_metadata` and `resave`; replaces the `set_font` font from the stream `("font2", doc_id)`; appends the Producer sentence to the notes of `genuine_resaved` and `metadata_inconsistency`; replaces `Courier` in `font_mismatch` notes by the drawn name; all else unchanged), and `style_block(style)`
    - Put the Node 2 consequences (Requirement 10.6) in the module comment
    - Make the 3.1 tests pass
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.7, 3.1, 3.2, 4.1, 4.5, 4.6, 5.2, 5.3, 6.1, 6.2, 6.3, 9.1_
  - [x]* 3.3 Write property tests for `styling.py`
    - In `tests/corpus/test_prop_style_v2.py`, at least 100 examples each, tagged `Feature: generator-styling-variety, Property N: ...`
    - **Property 2: The style draw is a pure function of seed and document id, and keeps the legacy values**
    - **Property 3: Every drawn value comes from its catalogue**
    - **Property 4: The printed address follows the template and the second-line flag**
    - **Property 5: The normaliser removes only the label-dependent choices**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5, 3.2, 4.1, 4.5, 5.2, 5.3, 6.1, 6.3**

- [x] 4. Renderers and PDF metadata honour the style
  - [x] 4.1 `finish.py`: Creator, footer font and address helper
    - `new_canvas` and `finish_pdf` use `style.creator or CREATOR`
    - `draw_footer(c, page_width, *, margin=40.0, font=FOOTER_FONT)`; the default call draws exactly as before
    - Add `draw_address(c, x, y_baseline, style)`: draws `styling.address_text(style)` at 6.5 pt in `style.font_regular`; does nothing when `style.print_address` is false. Import `styling` lazily inside the function to avoid a cycle
    - Run the Golden_Hashes test and `tests/corpus`; both must still pass
    - _Requirements: 3.3, 3.5, 4.2, 5.1, 5.7_
  - [x] 4.2 `generator_v1.py`: address line and footer font
    - After the title, call `draw_address` at baseline `_PAGE_H - 29*mm`, x = `_X_DATE`; pass `style.font_regular` to `draw_footer`
    - Do not change any other drawing call or constant
    - _Requirements: 4.2, 5.1, 5.4, 5.7_
  - [x] 4.3 `dataset_family.py`: address line, bank-line font and footer font
    - `_page_head` receives `style`; draw the bank line in `style.font_regular`; call `draw_address` at top 44, x = 300; pass the font to `draw_footer`. Update both callers of `_page_head`
    - Do not change any other drawing call or constant
    - _Requirements: 4.2, 5.1, 5.4, 5.7_
  - [x] 4.4 `pdf_ops._apply_metadata` tolerates a missing producer
    - Build the metadata update without `/Producer` when the `producer` param is absent, so the PDF keeps its own; all other behaviour unchanged
    - Add a test in `tests/corpus/test_pdf_ops_metadata.py`: a `set_metadata` op without `producer` keeps the Producer and sets ModDate to creation plus the delta; an op with `producer` behaves as before
    - _Requirements: 2.4_
  - [x] 4.5 Render, read back and compare, for all six Layouts
    - In `tests/corpus/test_layout_style_v2.py`, with fixed examples per Layout and a hypothesis strategy sampling Layout, style and a 1 to 3 month ledger (10 to 20 examples, say so in a comment). Read PDFs with `pdfplumber` and `pypdf` only
    - **Property 6: Producer and Creator survive rendering and a metadata operation**
    - **Property 7: The bank street is the first street-pattern match on page 1**
    - **Property 8: One non-overlapping address line on every page with the bank name**
    - **Property 9: Fonts follow the style and stay inside the built-in set**
    - **Property 10: Table geometry does not depend on the Styling_Version**
    - **Property 11: Node 1 reads the same rows under Balanced_Styling** (use `tests/corpus/helpers.py` `compare_rows` and the Node 1 ingestion that `test_verify_node1.py` uses)
    - These are required tests, not optional. If one fails because an address position collides or Node 1 reads a different row, stop and ask the user before changing a position; record the change in design decision D4
    - **Validates: Requirements 2.4, 3.3, 4.2, 4.3, 4.4, 5.1, 5.4, 5.5, 5.6, 9.5, 10.4, 10.5**

- [x] 5. Builder wiring and manifests
  - [x] 5.1 Version-2 path in `_build_document`, and the Style_Block
    - In `fraud_detect/corpus/builder.py` `_build_document`: when `style_version == 2` use `make_style_v2` and `normalise_tamper_result` after the tamper is planned; otherwise the existing calls
    - In `fraud_detect/corpus/labels.py` `build_manifest`: new keyword `style_block=None`; add the `style` key only when it is not None
    - Run the Golden_Hashes test; it must still pass
    - _Requirements: 1.8, 2.4, 6.4, 8.5, 9.1, 9.2_
  - [x] 5.2 Tests for the version-2 build
    - In `tests/corpus/test_builder_style_v2.py` (continued): tiny version-2 build twice gives equal CSV and JSON bytes and equal PDF text layers and metadata; `labels.csv`, `rows.csv` and `companies.json` are byte-identical to a version-1 build of the same seed and mix; headers equal `LABELS_COLUMNS` and `ROWS_COLUMNS`; each Manifest holds the 11 Style_Block keys and every old key; for documents without `set_metadata` or `resave` the PDF Producer and Creator equal the Style_Block; the renderer receives `make_style_v2` of the document id for genuine and tampered documents alike; non-early documents have a PDF creation time equal to `style.creation_time`
    - **Property 12: The plan does not depend on the Styling_Version** (hypothesis, at least 100 examples; `plan()` only)
    - Counts and warnings: with `subtle_amount_edit` in the mix `per_tamper_type` lists it for each Split; without it the dictionary equals the version-1 dictionary
    - **Validates: Requirements 1.7, 6.4, 6.5, 7.9, 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 9.4, 9.5**

- [x] 6. Subtle amount-change tamper
  - [x] 6.1 Write failing tests for `subtle_amount_edit`
    - In `tests/corpus/test_subtle_amount.py`: registration in `TAMPER_INFO`, `CATALOG` and `EXTRA_TAMPER_TYPES` with group `Document` and methods `overlay`, `rerender`; the `TAMPER_INFO` coverage assertion now accepts `ALL_TAMPER_TYPES`; a ledger whose largest amount is 0.20 raises `ValueError` naming the row; with `subtle_amount_edit` in a mix a tiny build records the type in `tamper_types` and the affected row in the Manifest
    - _Requirements: 7.1, 7.6_
  - [x] 6.2 Implement `tamper/subtle.py` and register it
    - New module as in the design (eligible rows with absolute amount at least 100.00 or the largest row; relative change 1.0% to 4.9% with random sign; bounded 8 redraws then `ValueError`; method and balance handling from the tamper stream; overlay through `pdf_ops.overlay_result`, rerender through `ledger_ops._result`)
    - `tamper/_registry.py`: add the `TAMPER_INFO` entry (group `Document`, methods `overlay`, `rerender`, arithmetic `random`, no `should_be_seen_by`) and change the coverage assertion to `ALL_TAMPER_TYPES`
    - `tamper/catalog.py`: append `"subtle"` to `_IMPLEMENTATIONS`
    - Do not change the planners of the 16 original types
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_
  - [x]* 6.3 Write property tests for the subtle amount edit
    - In `tests/corpus/test_prop_subtle.py`, at least 100 examples each, tagged `Feature: generator-styling-variety, Property N: ...`; ledgers from `simulate_ledger` with a drawn company seed
    - **Property 13: A subtle amount edit changes exactly one eligible amount**
    - **Property 14: A subtle amount edit stays between 1% and 5%** (also assert `corpus_gates.classify_subtle` returns `amount_le_5pct`)
    - **Property 15: A subtle amount edit is deterministic and replayable**
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.7, 7.8**

- [x] 7. Checkpoint - generator complete
  - Ensure all tests pass, ask the user if questions arise.
  - Run `tests/corpus` and compare with the baseline in task 1.2; the Golden_Hashes test must pass unchanged
  - Report any Layout where the address position had to change

- [x] 8. Gate simulation and verify compatibility
  - [x] 8.1 Simulate the ml-retraining gates on Balanced_Styling style tables
    - In `tests/ml_retraining/test_style_v2_gates.py`: build style tables from `make_style_v2` for 480 genuine and 480 tampered documents (columns `style_producer`, `style_producer_family`, `style_creator`, `style_fonts` as the sorted font tuple with the decorative font, `style_bank_name`, `style_street`, `style_row_step_bucket` rounded to 0.1, `style_creation_hour`, `is_tampered`). Over 300 tables from distinct Master_Seeds, `styling_balance` (min 30 documents, 10 points) has at most 6 failing tables; over 20 tables `capability_report` returns an empty list; each Producer family is at least 4% of each class in every table
    - Build a table from `CorpusBuilder.plan()` with the 26-part mix at 80 + 80 slots per Layout and the corpus split fractions and run `resaved_share_gate` with 15%; assert it passes in `train`, `validation` and `test`
    - Use fixed Master_Seeds; if a table fails, report it, do not change a seed, weight or threshold
    - _Requirements: 2.6, 11.1, 11.2, 11.3, 11.5_
  - [x] 8.2 Check `verify.py` and the report with the new tamper type
    - Read `fraud_detect/corpus/verify.py` and the tests in `tests/corpus/test_verify_*.py`. Run `verify` on a tiny corpus that has `subtle_amount_edit` in the mix. If the table, the expected-code columns or the blind-spot list fail for the new type, fix `verify.py` minimally (a type with no `should_be_seen_by` is already handled for `rapid_pass_through`; follow that path). Add a test for the new type
    - If nothing needs fixing, add only the test
    - _Requirements: 7.1, 8.4_

- [x] 9. Switch the ml-retraining build to Balanced_Styling
  - [x] 9.1 Update the build command, the mix and the counts
    - Write failing tests first in `tests/ml_retraining/test_build_mix.py` (edit the existing file): names equal `ALL_TAMPER_TYPES` (17), `genuine_resaved` is 10 parts, every other type 1 part, total 26; **Property 17: Plan counts follow the 26-part mix** (hypothesis, at least 100 examples); `cmd_build` with a fake `corpus_builder` receives a `BuildConfig` with `style_version == 2` for every seed and for the held-out seed
    - In `fraud_detect/ml/retraining/runner.py`: `design_tamper_mix` uses `ALL_TAMPER_TYPES`; rename `RESAVED_PER_25` to `RESAVED_PER_26` (first grep for other users and keep an alias if any exists); `plan_counts` divides by 26; `cmd_build` passes `style_version=2`; update the docstring
    - Leave `tests/ml_retraining/conftest.py` (Mini_Corpus) unchanged
    - Do not edit `protected_hashes.json`, `test_audit.py` or any other pinned value
    - _Requirements: 11.4, 12.1, 12.2, 12.3, 12.4_
  - [x] 9.2 Round-trip test through the Retraining_Pipeline reader
    - In `tests/ml_retraining/test_style_v2_roundtrip.py`: build a tiny version-2 corpus (6 Layouts x 2 genuine + 2 tampered, 6 months), read every PDF with `style_profile` and compare `producer`, `producer_family`, `creator`, `street`, `fonts` and `creation_hour` with the Style_Block of the Manifest (`street` must equal the bank street in all six Layouts); assert the `style_street` column is no longer None in any document; run `capability_report` and `styling_balance_gate` on the tiny corpus table only for the structure of the result (a tiny table cannot pass the share rule)
    - Run all of `tests/ml_retraining` and list every failing test with its reason in the task notes. Expected failures only: `test_hygiene.py::test_protected_paths_not_modified_in_git` while the generator change is uncommitted. Any other failure is a finding to report, not to fix by editing a pin
    - _Requirements: 3.3, 5.4, 9.5, 12.5, 12.7_

- [x] 10. Final checkpoint - ask the user
  - Ensure all tests pass, ask the user if questions arise.
  - Run the whole suite and compare with the baseline of task 1.2
  - Report: which pinned checks changed and why (design, Re-pin checklist), the result of the 300-table simulation (number of failing tables), and the open decisions that remain
  - Ask the user to review and commit the generator change before the manual steps below

## Notes

- Tasks marked with `*` are optional property tests and can be skipped for a faster first version. Unit and example tests are not optional. The render-based properties in task 4.5 are required because they protect Node 1 parity.
- Baseline counts (task 1.2, HEAD c5279b3, unchanged generator, excluding the task 1.1 files `golden_tool.py`, `test_legacy_golden.py`, `legacy_golden.json`). Run as three parallel partitions with the test command above plus `-rfEs`; the totals are the sum:
  - `tests/corpus`: 521 passed, 0 skipped, 1 failed (522 tests).
  - Rest of the suite (`tests` without `tests/corpus` and `tests/ml_retraining`): 1155 passed, 1 skipped (`tests/test_registry.py:273`, needs `RUN_LIVE_TESTS=1`), 0 failed.
  - `tests/ml_retraining`: 1053 passed, 3 skipped (`test_scaffold.py:64` and `test_sources.py:344` need `RETRAIN_REAL_STYLE_DIR`; `test_sources.py:354` needs `RETRAIN_FORGERY_DIR`), 0 failed. Runtime about 12 min.
  - Whole suite total: 2729 passed, 4 skipped, 1 failed (2734 tests).
  - Pre-existing failing test (list for later checkpoints): `tests/corpus/test_prop_tamper_file.py::test_overlay_keeps_both_values_rerender_keeps_only_new` (fails when the shared `.hypothesis` DB replays a saved example).
  - Two further failures appeared only because of the parallel runs and passed when re-run alone, so they are not baseline failures: `tests/corpus/test_end_to_end.py::test_nothing_written_outside_tmp_path` (the other partition appended to `data/audit_chain.jsonl` during the run) and `tests/ml_retraining/test_cli.py::test_models_folder_is_byte_identical_and_nothing_sensitive_is_untracked` (it saw the untracked `tests/corpus/legacy_golden.json` from task 1.1). Run the checkpoints sequentially, and note the second test needs a clean `git status` outside the files under test.
- Task 9.2 result (full `tests/ml_retraining` run, generator change uncommitted): 1063 passed, 3 skipped (same three env-var skips as the baseline), 2 failed, 12 min. New file `test_style_v2_roundtrip.py`: 6 tests, all pass (street equals the bank street in all six Layouts and is never None; producer, family, creator, fonts and creation hour equal the Style_Block).
  - `test_hygiene.py::test_protected_paths_not_modified_in_git` (expected): `git status --porcelain` for `fraud_detect/corpus` lists the uncommitted generator edits and the new `styling.py` and `tamper/subtle.py`.
  - `test_awkward_spike.py::test_spike_modifies_no_corpus_file` (not on the expected list, same root cause): it compares `git hash-object` of every file tracked under `fraud_detect/corpus` with HEAD, and `fraud_detect/corpus/build.py` differs because of the uncommitted `--style-version` change. It is not a pin and not a code bug; it should pass once the generator change is committed. Not edited.
  - `test_cli.py::test_models_folder_is_byte_identical_and_nothing_sensitive_is_untracked` passed in this run. Untracked files (e.g. `tests/ml_retraining/`, `fraud_detect/ml/retraining/`, the new corpus tests) did not make it fail here; the earlier baseline failure came from `tests/corpus/legacy_golden.json` being untracked during a parallel run. Not re-checked after any further `git status` changes.
  - No pin, `protected_hashes.json`, audit digest or fingerprint was edited.
- Each group is test first: the x.1 task writes tests that fail until the matching implementation task is done. Task 2.1 and task 2.2 write their tests and code together because the schema change is small.
- Property tests use `hypothesis`. Pure properties use at least 100 examples; properties that render PDFs (6 to 11) use 10 to 20 examples and say so. Tag format: `Feature: generator-styling-variety, Property N: <text>`.
- Files that more than one task edits are in different waves (`builder.py` in tasks 2.2 and 5.1; `test_builder_style_v2.py` in tasks 2.2 and 5.2). Tasks that run in the same wave write different test files.
- The Golden_Hashes test is the guard for Legacy_Styling. Never regenerate `legacy_golden.json` to make a test pass.
- Steps for the project owner after the final checkpoint (not tasks, not run by the agent). They also answer Requirement 11.6:
  1. Review and commit the generator change. The git-clean test on `fraud_detect/corpus` needs this.
  2. `python -m fraud_detect.ml.retraining init` with the chosen options. Start a new run; old `run.json` files report "generator and detector code unchanged since run start: no".
  3. `python -m fraud_detect.ml.retraining check --plan`, then `build` (5 training seeds and the held-out seed; this now builds Balanced_Styling corpora with the 26-part mix).
  4. `python -m fraud_detect.ml.retraining check` at full size (80 genuine and 80 tampered slots per Layout). Expected: `styling_balance`, `resaved_share`, `capability` and `subtle` pass without a waiver.
  5. If one styling value fails by chance, build that seed again with a different Master_Seed. Do not change a threshold or a weight.
  6. Optional deliberate re-pins, each a separate change: a hash map of `fraud_detect/corpus` instead of the git-clean test; `audit-v2` if the audit text should describe Balanced_Styling; a new Golden_Hashes file if Styling_Version 2 becomes the default.
- Node 2 will flag some genuine Balanced_Styling documents by design (editor-family Producer; re-save with a non-export tool). Do not compare `verify` numbers between version 1 and version 2 corpora as if they measured the same thing.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "6.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.1", "4.4", "6.2"] },
    { "id": 3, "tasks": ["3.3", "4.2", "4.3", "5.1", "6.3", "8.1", "8.2"] },
    { "id": 4, "tasks": ["4.5", "5.2", "9.1"] },
    { "id": 5, "tasks": ["9.2"] }
  ]
}
```
