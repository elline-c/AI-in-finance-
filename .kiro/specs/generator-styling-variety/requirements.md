# Requirements Document

## Introduction

This feature is a **generator follow-up** to the `ml-retraining` spec (Stage B). A full-size `check` run of the Retraining_Pipeline (5 training Master_Seeds, 80 genuine and 80 tampered documents per Layout, 6 Layouts) fails gates that the Corpus_Generator in `fraud_detect/corpus/` cannot satisfy today:

1. **Producer conflict.** `genuine_resaved` documents get an export Producer (`pdf_ops.EXPORT_PRODUCERS`) and tampered documents never do. `metadata_inconsistency` gets an editor Producer (`EDITOR_PRODUCERS`) and no genuine document does. All other documents get one of the four base Producers. The non-waivable `resaved_share` gate needs about 29% of genuine documents to be `genuine_resaved`. At that share the non-waivable `styling_balance` gate fails: export family 28.6% of genuine and 0.0% of tampered, base family 71.4% and 93.4%. With the default equal mix (about 6% re-saved) styling passes and `resaved_share` fails. No mix passes both.
2. **Capability gate (waivable).** Creator is one constant string. Fonts are Helvetica and Helvetica-Bold in every renderer, with Courier only through `font_mismatch`. The export Producer family exists only on genuine documents and the editor family only on tampered documents.
3. **Subtle gate (waivable).** No tamper changes an amount by 5% or less.
4. **Street.** The street that `finish.make_style` draws (`finish.STREETS`) is printed by no Layout, so the Retraining_Pipeline cannot recover it from a PDF.

This feature adds an opt-in, versioned styling mode, **Balanced_Styling**, to the Corpus_Generator. In that mode every styling choice (Producer including the export family, Creator, fonts, bank, address wording, row spacing, creation time of day) is drawn from one distribution that does not depend on the label. Tampered documents can carry an export Producer. Genuine documents can carry an editor-family Producer. A new subtle amount-change tamper (5% or less) is added. The Corpus_Generator stays deterministic per Master_Seed. The default mode, **Legacy_Styling**, stays byte-for-byte reproducible, so existing tests, corpora and the ml-retraining Mini_Corpus fixtures do not change.

After this feature, a full-size ml-retraining `check` run that builds its corpora with Balanced_Styling and the updated tamper mix should pass `styling_balance`, `resaved_share`, `capability` and `subtle` without any waiver. The full-size run is a manual step for the project owner; this document sets measurable generator-level criteria that make it pass, and an automated simulation that checks them.

All data stays synthetic. Nodes 1 to 4, `fraud_detect/registry.py` and `fraud_detect/reference_data.py` are not changed. No runtime dependency is added. This feature does not train, switch or evaluate any model.

## Glossary

- **Corpus_Generator**: The package `fraud_detect/corpus/` and its `build` and `verify` commands.
- **Corpus_Builder**: `CorpusBuilder` and the `build` command, which plan and write a whole corpus.
- **Build_Command**: `python -m fraud_detect.corpus.build`.
- **Styling_Version**: The integer in `BuildConfig.style_version`. `1` selects Legacy_Styling, `2` selects Balanced_Styling. The default is `1`.
- **Legacy_Styling**: The styling behaviour of the Corpus_Generator before this feature (Styling_Version 1).
- **Balanced_Styling**: The new styling behaviour (Styling_Version 2) described in this document.
- **Style_Drawer**: The function that returns the `StyleKnobs` of one document. For Legacy_Styling it is `finish.make_style`. For Balanced_Styling it is the new `make_style_v2` in `fraud_detect/corpus/styling.py`.
- **StyleKnobs**: The frozen data class that carries the styling values of one document into a Layout_Renderer.
- **Layout_Renderer**: One of the six Layout renderers in `fraud_detect/corpus/layouts/`.
- **Layout**: One of the six Node 1 profiles: `generator_v1`, `two_col_dmon`, `two_col_slash`, `two_col_dual_date`, `signed_amount_iso`, `two_col_ddmmyy_cr`.
- **Tamper_Type**: A named tamper in the catalog. The 16 original names are `schema.TAMPER_TYPES`.
- **Extra_Tamper_Type**: A Tamper_Type added by this feature and listed in the new `schema.EXTRA_TAMPER_TYPES`. It is `subtle_amount_edit`.
- **Subtle_Amount_Tamper**: The Tamper_Type `subtle_amount_edit`, which changes one amount by 1.0% to 4.9%.
- **Producer_Family**: One of `base` (the four `finish.PRODUCERS`), `export` (`pdf_ops.EXPORT_PRODUCERS`) or `editor` (`pdf_ops.EDITOR_PRODUCERS`).
- **Creator_Pool**: The tuple of five Creator strings that Balanced_Styling draws from.
- **Decorative_Text**: The specimen footer, the bank-name line of the page head where the bank name is a separate regular-weight line, and the new bank-address line. It excludes every bold title line, every table header word, every table cell, the summary box, the closing-balance line, the page marker and the `SPECIMEN` marker.
- **Foreign_Font**: A built-in font that Node 2 does not list as a body font: any font outside `Helvetica`, `Helvetica-Bold`, `Arial`, `Arial-Bold` and `Times-Roman`.
- **File_Op_Normaliser**: The new function in `styling.py` that rewrites the file operations and the label notes of a tamper result when Balanced_Styling is active.
- **Style_Block**: The new `style` object in a per-document Manifest, written only under Balanced_Styling.
- **Golden_Hashes**: The file `tests/corpus/legacy_golden.json`, which holds SHA-256 values of Legacy_Styling output recorded on the code as it was before this feature.
- **Gate_Suite**: The four ml-retraining corpus gates `styling_balance`, `resaved_share`, `capability` and `subtle` in `fraud_detect/ml/retraining/corpus_gates.py`.
- **Retraining_Pipeline**: The tooling in `fraud_detect/ml/retraining/`.
- **Master_Seed**: The integer that determines a whole corpus.
- **Mini_Corpus**: The small corpus that `tests/ml_retraining/conftest.py` builds with the existing `CorpusBuilder` (6 Layouts, 2 genuine and 2 tampered documents each, 6 months, two Master_Seeds).
- **Protected_Files**: The files under `fraud_detect/nodes/`, `fraud_detect/registry.py` and `fraud_detect/reference_data.py`.

## Requirements

### Requirement 1: Versioned styling with the legacy behaviour preserved

**User Story:** As a project owner, I want the new styling to be opt-in and versioned, so that existing corpora, tests and fixtures stay reproducible.

#### Acceptance Criteria

1. THE Corpus_Builder SHALL accept a Styling_Version of 1 or 2 in `BuildConfig.style_version`, and SHALL use 1 when no value is given.
2. WHERE the Styling_Version is 1, THE Corpus_Generator SHALL produce, for the same Master_Seed and settings, a `labels.csv`, `rows.csv`, `companies.json`, `corpus_manifest.json` and per-document Manifest set whose SHA-256 values equal the Golden_Hashes, and PDFs whose text-layer SHA-256 and metadata equal the Golden_Hashes.
3. IF the Styling_Version is neither 1 nor 2, THEN THE Corpus_Builder SHALL refuse the settings with a message that names the accepted values 1 and 2, and SHALL write no file.
4. WHEN the option `--style-version` is given to the Build_Command, THE Build_Command SHALL pass its value to `BuildConfig.style_version`.
5. IF the value given to `--style-version` is refused by the Corpus_Builder, THEN THE Build_Command SHALL print the refusal on standard error and exit with status 2.
6. THE Corpus_Generator SHALL keep `schema.GENERATOR_VERSION` equal to `"1.0.0"`, `schema.TAMPER_TYPES` equal to its 16 original names in the original order, and `schema.LABELS_COLUMNS` equal to its 25 original columns in the original order.
7. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write the key `styling_version` with the value 2 into `corpus_manifest.json` and one line that names the Styling_Version into `README.txt`.
8. WHERE the Styling_Version is 1, THE Corpus_Builder SHALL write neither the `styling_version` key nor a Style_Block.

### Requirement 2: Draw the Producer independently of the label

**User Story:** As a project owner, I want Producer strings from every family on genuine and on tampered documents, so that the Producer family carries no label information and the re-saved share can be set freely.

#### Acceptance Criteria

1. WHERE the Styling_Version is 2, THE Style_Drawer SHALL draw the Producer_Family with the probabilities base 0.65, export 0.25 and editor 0.10, and SHALL then draw the Producer uniformly from the catalogue of the drawn Producer_Family.
2. WHERE the Styling_Version is 2, THE Style_Drawer SHALL compute the StyleKnobs from the Master_Seed and the document id alone, and SHALL take no argument that carries the label, the Tamper_Type or the `genuine_resaved` flag.
3. WHERE the Styling_Version is 2, WHEN a tamper result contains a `set_metadata` or `resave` file operation with a `producer` parameter, THE File_Op_Normaliser SHALL remove the `producer` parameter and SHALL leave every other parameter of the file operation unchanged.
4. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write for a document with a `set_metadata` or `resave` file operation the Producer that the Style_Drawer drew for that document.
5. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL keep the ModDate gap of `genuine_resaved` between 60 seconds and 3 days and the ModDate gap of `metadata_inconsistency` between 2 and 40 days, as they are under Legacy_Styling.
6. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write, in a corpus of at least 400 genuine and at least 400 tampered documents, each of the three Producer_Families on at least 4% of the genuine documents and on at least 4% of the tampered documents. `genuine_resaved` documents count as genuine.
7. WHERE the Styling_Version is 2, THE File_Op_Normaliser SHALL append to the label notes of a `genuine_resaved` or `metadata_inconsistency` document one sentence that states that the Producer comes from the shared styling draw and is not set by the tamper.

### Requirement 3: Vary the Creator

**User Story:** As a project owner, I want the Creator string to vary independently of the label, so that a constant Creator is not a feature of the corpus.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL define a Creator_Pool of five distinct fictional strings, and the Creator_Pool SHALL contain the legacy string `synthetic-corpus-generator 1.0.0`.
2. WHERE the Styling_Version is 2, THE Style_Drawer SHALL draw the Creator uniformly from the Creator_Pool.
3. WHERE the Styling_Version is 2, THE Layout_Renderer SHALL write the drawn Creator into the PDF `/Creator` field, both when the canvas is created and when `finish_pdf` writes the final metadata.
4. THE Creator_Pool SHALL contain no string that includes a keyword from the editing-software lists of Node 2 (`visual_forensics._EDITING_SOFTWARE` and the whole word `editor`) or of Node 6b (`statement_features._EDITING_SOFTWARE`), compared in lower case.
5. WHERE the Styling_Version is 1, THE Layout_Renderer SHALL write the legacy Creator string `synthetic-corpus-generator 1.0.0`.

### Requirement 4: Vary the fonts without false alarms on genuine documents

**User Story:** As a project owner, I want font variety on genuine and tampered documents, so that the font set is not a tell, without making Node 2 raise font alarms on ordinary genuine statements.

#### Acceptance Criteria

1. WHERE the Styling_Version is 2, THE Style_Drawer SHALL draw the regular font of the Decorative_Text from `Helvetica` with probability 0.65 and `Times-Roman` with probability 0.35, and SHALL set the bold font to `Helvetica-Bold`.
2. WHERE the Styling_Version is 2, THE Layout_Renderer SHALL draw the Decorative_Text in the drawn regular font, and SHALL draw the table header words, the table cells, the summary box and the closing-balance line in the same fonts as under Legacy_Styling.
3. THE Corpus_Generator SHALL use only reportlab built-in fonts in every generated PDF.
4. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write every genuine document with fonts drawn only from `Helvetica`, `Helvetica-Bold` and `Times-Roman`.
5. WHERE the Styling_Version is 2, WHEN the Tamper_Type is `font_mismatch`, THE File_Op_Normaliser SHALL replace the font of the `set_font` file operation by a font drawn uniformly from `Courier`, `Courier-Bold`, `Times-Bold` and `Helvetica-Oblique`, using a random stream that depends on the Master_Seed and the document id only.
6. WHERE the Styling_Version is 2, WHEN the Tamper_Type is `font_mismatch`, THE File_Op_Normaliser SHALL replace the word `Courier` in the label notes by the drawn font name, and SHALL keep `font_inconsistent` true.
7. WHERE the Styling_Version is 1, THE Layout_Renderer SHALL draw all text in `Helvetica` and `Helvetica-Bold` exactly as before this feature.

### Requirement 5: Print the bank address so that the street can be read back

**User Story:** As a project owner, I want the generator's street and address wording printed on the statement, so that the Retraining_Pipeline can recover them from the PDF and check their balance.

#### Acceptance Criteria

1. WHERE the Styling_Version is 2, THE Layout_Renderer SHALL print one bank-address line on every page on which the bank name is printed, in the page-head band, in the drawn regular font at 6.5 points.
2. WHERE the Styling_Version is 2, THE Style_Drawer SHALL draw the address wording uniformly from three templates: `"<street>, Singapore <postcode>"`, `"<street> Singapore <postcode>"` and `"<street>"`.
3. WHERE the Styling_Version is 2 and the drawn `second_address_line` is false, THE Layout_Renderer SHALL print the street only, whatever template was drawn.
4. WHERE the Styling_Version is 2, THE Layout_Renderer SHALL print the bank street so that it is the first match of the pattern `\b\d{1,4} [A-Z][A-Za-z]+ (?:Road|Street|Avenue|Lane|Drive|Way|Crescent|Close|Walk|Link)\b` in the text of page 1 read from top to bottom, ahead of any company address.
5. WHERE the Styling_Version is 2, THE Layout_Renderer SHALL place every text box of the bank-address line so that it overlaps no table cell, no table header word and no other text box on the page.
6. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL keep the number of pages and the position of every table cell of a document equal to those of the same document under Legacy_Styling.
7. WHERE the Styling_Version is 1, THE Layout_Renderer SHALL print no bank-address line.

### Requirement 6: Draw the remaining styling choices from the same distribution for both classes

**User Story:** As a project owner, I want bank, row spacing, address wording and creation time of day to stay label-independent and comparable with the legacy draw, so that results from the two modes can be compared.

#### Acceptance Criteria

1. WHERE the Styling_Version is 2, THE Style_Drawer SHALL return `bank_name`, the street, the postcode, `creation_time`, `row_step` and `second_address_line` equal to the values that `finish.make_style` returns for the same Master_Seed and document id.
2. WHERE the Styling_Version is 2, THE Style_Drawer SHALL draw the new values (Producer_Family, Producer, Creator, regular font, address wording) from a random stream named `style2` that is separate from the legacy `style` stream.
3. THE Style_Drawer SHALL return the same StyleKnobs when called twice with the same Master_Seed and document id.
4. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL pass to the Layout_Renderer the StyleKnobs of the Style_Drawer without change, for genuine documents and for tampered documents alike.
5. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL leave the creation time of day of a document equal to the time of day drawn by the Style_Drawer, except for the documents of `metadata_inconsistency` with an early creation date, which keep their recorded `created` parameter.

### Requirement 7: Provide a subtle amount-change tamper

**User Story:** As a project owner, I want a tamper that changes one amount by 5% or less, so that the subtle gate can pass and the model is tested on hard cases.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL provide the Extra_Tamper_Type `subtle_amount_edit` in `schema.EXTRA_TAMPER_TYPES`, in the tamper catalog and in `TAMPER_INFO`, with the group `Document` and the methods `overlay` and `rerender`.
2. WHEN `subtle_amount_edit` is planned, THE Subtle_Amount_Tamper SHALL change the amount of exactly one ledger row, and SHALL record exactly one `affected_rows` entry with `field` equal to `amount`.
3. WHEN `subtle_amount_edit` is planned, THE Subtle_Amount_Tamper SHALL choose a relative change with an absolute value of at least 1.0% and at most 4.9%, with a random sign, and SHALL compute the new amount rounded to two decimals.
4. THE Subtle_Amount_Tamper SHALL satisfy, for the recorded `original` and `new` strings, `original` not equal to `new` and `abs(new - original) / abs(original)` at least 0.01 and at most 0.05.
5. WHEN `subtle_amount_edit` is planned, THE Subtle_Amount_Tamper SHALL choose the row from the rows whose absolute amount is at least 100.00, and SHALL choose the row with the largest absolute amount when no row reaches 100.00.
6. IF the chosen row cannot be changed by 1.0% to 4.9% after rounding to two decimals, THEN THE Subtle_Amount_Tamper SHALL raise a `ValueError` that names the row, so that the Corpus_Builder records the document under `failed`.
7. WHEN `subtle_amount_edit` is planned, THE Subtle_Amount_Tamper SHALL draw the method (`overlay` or `rerender`) and, for `rerender`, whether the balances are recomputed or left as printed, from the tamper random stream, and SHALL record both in the label.
8. THE Subtle_Amount_Tamper SHALL produce the same result for the same ledger and the same random stream.
9. WHERE the Tamper_Type `subtle_amount_edit` is not in `BuildConfig.tamper_mix`, THE Corpus_Builder SHALL deal the same tamper types and write the same `corpus_manifest.json` counts as before this feature.
10. IF `BuildConfig.tamper_mix` names a type that is neither in `schema.TAMPER_TYPES` nor in `schema.EXTRA_TAMPER_TYPES`, THEN THE Corpus_Builder SHALL refuse the settings with a message that names the unknown type.

### Requirement 8: Deterministic and backward-compatible output

**User Story:** As a project owner, I want the generator to stay deterministic and the old output to stay unchanged, so that earlier results and tests remain valid.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL produce, for the same Master_Seed, settings and Styling_Version, the same `labels.csv`, `rows.csv`, `companies.json`, per-document Manifests and `corpus_manifest.json`, and the same PDF text layers and metadata, in every build.
2. WHEN the same Master_Seed and tamper mix without an Extra_Tamper_Type are built once with Styling_Version 1 and once with Styling_Version 2, THE Corpus_Builder SHALL produce the same document ids, splits, company ids, UENs, document seeds and Tamper_Types in both builds.
3. WHEN the same Master_Seed and tamper mix without an Extra_Tamper_Type are built once with Styling_Version 1 and once with Styling_Version 2, THE Corpus_Builder SHALL produce byte-identical `labels.csv`, `rows.csv` and `companies.json` in both builds.
4. THE Corpus_Generator SHALL pass every test in `tests/corpus/` that passed before this feature, without a change to the assertions of those tests, except the tests that this feature adds.
5. THE Corpus_Generator SHALL leave the random streams of the existing tamper planners unchanged, so that a tamper plan for a given ledger and random stream is identical before and after this feature.
6. THE Corpus_Generator SHALL keep the Mini_Corpus builds of `tests/ml_retraining/conftest.py`, which use Styling_Version 1, identical in their Golden_Hashes-covered output.

### Requirement 9: Additive schema only

**User Story:** As a project owner, I want only additive changes to files and schemas, so that existing readers keep working.

#### Acceptance Criteria

1. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write a Style_Block into each per-document Manifest with the keys `styling_version`, `producer`, `producer_family`, `creator`, `decorative_font`, `bank_name`, `street`, `address_wording`, `second_address_line`, `creation_time` and `row_step`.
2. THE Corpus_Builder SHALL leave all existing keys of the per-document Manifest, of `corpus_manifest.json` and of `labels.csv` in place with their existing meaning.
3. THE Corpus_Builder SHALL leave the columns of `labels.csv` and `rows.csv` unchanged under both Styling_Versions.
4. WHERE an Extra_Tamper_Type is in `BuildConfig.tamper_mix`, THE Corpus_Builder SHALL list that type in `counts.per_tamper_type` of `corpus_manifest.json` for each Split and SHALL include it in the missing-type warnings.
5. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL write for a Style_Block `producer_family` and `producer` values that equal the values read back from the PDF metadata of the same document when the PDF has no `set_metadata` or `resave` file operation.

### Requirement 10: Safety, synthetic data and no change to protected code

**User Story:** As a project owner, I want the new styling to stay synthetic and to leave the detectors and parsers untouched, so that earlier measurements stay valid.

#### Acceptance Criteria

1. THE Corpus_Generator SHALL use only fictional strings in every catalogue added by this feature (Creator_Pool, address wording, Producer additions), with no real company, bank, person or street.
2. THE Corpus_Generator SHALL leave the Protected_Files unchanged, so that every SHA-256 value in `tests/ml_retraining/protected_hashes.json` still matches.
3. THE Corpus_Generator SHALL add no runtime dependency to `requirements.txt` or `requirements-gnn.txt`.
4. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL produce, for every Layout, a PDF from which Node 1 reads the same transaction rows (date, signed amount, balance, order) as from the same document under Styling_Version 1.
5. WHERE the Styling_Version is 2, THE Corpus_Builder SHALL leave the `row_map` cell boxes of every ledger row equal to those of the same document under Styling_Version 1.
6. THE Corpus_Generator SHALL describe in the code comments of `styling.py` that Node 2 raises `VF_EDIT_SOFTWARE` for editor-family Producers and `VF_MODIFIED_AFTER_CREATION` for a later ModDate with a non-export Producer, and that Balanced_Styling therefore creates genuine documents on which Node 2 can raise these findings by design.

### Requirement 11: The ml-retraining gates pass on Balanced_Styling corpora

**User Story:** As a project owner, I want the four failing ml-retraining gates to pass at full size without a waiver, so that the retraining run can start.

#### Acceptance Criteria

1. WHEN a simulation draws the Balanced_Styling styling values of 480 genuine and 480 tampered documents with random Master_Seeds, THE Gate_Suite function `styling_balance` SHALL find no failing styling value in at least 98% of 300 simulated corpora.
2. WHEN a simulation draws the Balanced_Styling styling values of 480 genuine and 480 tampered documents, THE Gate_Suite function `capability_report` SHALL return an empty list for the columns `style_producer`, `style_creator`, `style_fonts`, `style_bank_name`, `style_street`, `style_row_step_bucket` and `style_creation_hour` and for the Producer_Family column.
3. WHERE the Styling_Version is 2, THE Corpus_Generator SHALL produce a Producer_Family column in which each family appears in both classes.
4. WHEN the ml-retraining design mix (updated by Requirement 12) is built for 5 Master_Seeds, THE Gate_Suite function `subtle_gate` SHALL find in `train`, `validation` and `test` at least one document in each of the categories `single_amount`, `single_date` and `amount_le_5pct`.
5. WHERE the share of `genuine_resaved` among the tampered slots is at least 33% of the slots, THE Gate_Suite function `resaved_share_gate` SHALL pass with the threshold of 15% in each of `train`, `validation` and `test` of a corpus with 80 genuine and 80 tampered slots per Layout.
6. THE Corpus_Generator SHALL give the project owner a documented command sequence that runs the ml-retraining `check` at full size against Balanced_Styling corpora and reports the four gates.

### Requirement 12: Integrate with ml-retraining and document the deliberate re-pin

**User Story:** As a project owner, I want the ml-retraining build to use the new styling and the consequences for its pinned checks written down, so that nothing is re-pinned by accident.

#### Acceptance Criteria

1. WHEN the ml-retraining `build` command builds a corpus, THE Retraining_Pipeline SHALL pass `style_version=2` to `BuildConfig`.
2. THE Retraining_Pipeline design mix `design_tamper_mix` SHALL contain every name in `schema.TAMPER_TYPES` and `subtle_amount_edit`, with `genuine_resaved` at 10 parts and every other type at 1 part, so that the total is 26 parts.
3. THE Retraining_Pipeline function `plan_counts` SHALL compute the number of `genuine_resaved` documents and of other tamper types from the 26-part total.
4. THE Retraining_Pipeline SHALL leave the Mini_Corpus fixture on Styling_Version 1.
5. THE design document of this feature SHALL state that `tests/ml_retraining/protected_hashes.json` lists no file under `fraud_detect/corpus/`, and SHALL list the pinned checks that change as a result of this feature: the git-clean assertion on `fraud_detect/corpus` in `tests/ml_retraining/test_hygiene.py`, the code fingerprint of `fraud_detect/corpus` stored in every existing `run.json`, the 16-type and 40% assertions in `tests/ml_retraining/test_build_mix.py`, and the pinned Feature_Audit digest.
6. THE Corpus_Generator tasks SHALL contain no step that edits `protected_hashes.json`, a pinned digest or a recorded fingerprint, and SHALL list each such re-pin as a step for the project owner.
7. IF an existing ml-retraining `run.json` holds a code fingerprint that differs from the current fingerprint after this feature, THEN THE Retraining_Pipeline SHALL report "generator and detector code unchanged since run start: no" through its existing check, and the design document SHALL state that such a run must be started again.
