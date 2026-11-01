# Investigation: Node 2 metadata and currency findings (Requirements 12 and 13)

Report only. No Node 2 or other code was changed. Everything below was measured read-only on the branch `feature/Node3-fabrication-rules` with `fraud_detect.nodes.ingestion.ingest` and `fraud_detect.nodes.visual_forensics.analyze`, plus `pypdf` for the raw PDF metadata. Temporary scripts lived in `/tmp` and were deleted. Nothing was copied from the dataset into the repo.

Any fix named here is input for a LATER Node 2 spec. It is not part of this spec.

## I-1: why `statement_E` has no metadata finding

### What the files carry (read with pypdf)

| Field | statement_E (tampered) | statement_F (genuine) |
|---|---|---|
| Producer | OpenPage Writer 3.8 | Vellichor Paperless 1.12 |
| Creator | OpenPage Writer 3.8 | VCB Statements |
| CreationDate | 2026-10-03 02:36 (+08:00) | 2026-10-01 03:45 (+08:00) |
| ModDate | 2026-10-05 13:36 (+08:00) | 2026-10-01 03:45 (+08:00) |
| ModDate later than CreationDate | yes (2 days 11 hours) | no (identical) |

The metadata is present in the file, and ingestion reads it correctly (`_extract_metadata` keeps producer, creator, creation_date and mod_date). So it is not lost before Node 2 sees it.

### What Node 2 checks (read from `visual_forensics.py`)

- Editing-tool list `_EDITING_SOFTWARE`: `photoshop`, `illustrator`, `gimp`, `inkscape`, `acdsee`, `corel`. The check is a lower-case substring match on Producer plus Creator. `openpage writer 3.8` contains none of these, so `VF_EDIT_SOFTWARE` does not fire.
- Creation date check: `VF_CREATION_BEFORE_PERIOD` compares CreationDate with the statement period start only. For E the creation date (3 Oct 2026) is after the period start, so it does not fire.
- ModDate: `analyze` never reads `mod_date`. A search of the file shows no comparison between ModDate and CreationDate anywhere.

### Cause

Two of the three candidate causes apply. The first is not one of them.

| Candidate cause | Applies? | Evidence |
|---|---|---|
| Metadata absent from the file | No | Producer, Creator, CreationDate and ModDate are all present in E. |
| Tool name missing from the list | Yes | "OpenPage Writer 3.8" is not in `_EDITING_SOFTWARE`. The list holds only image-editor names, so a word-processor style name passes. |
| No ModDate check | Yes | No code compares ModDate with CreationDate. E's ModDate is 2 days later. |

Either gap alone is enough to leave E with no metadata finding. Fixing only one would still leave the other path closed or open depending on the rule chosen (see the discrimination results below).

### False-alarm risk, using the Forgery_Dataset (2,000 PDFs: 1,000 genuine, 1,000 tampered)

The dataset README says 64 genuine documents are "re-saved": a print or export Producer and a later ModDate, no content change, meant as hard negatives for any metadata rule. `labels.csv` has 64 rows with `genuine_resaved = 1`. All 2,000 files have both dates.

Counts of "ModDate later than CreationDate" (full timestamp compared):

| Group | Documents | ModDate later than CreationDate | Share |
|---|---|---|---|
| Genuine, not re-saved | 936 | 0 | 0% |
| Genuine, re-saved (`genuine_resaved = 1`) | 64 | 64 | 100% |
| All genuine | 1,000 | 64 | 6.4% |
| Tampered: metadata_inconsistency | 200 | 200 | 100% |
| Tampered: altered_date | 200 | 101 | 50.5% |
| Tampered: edited_amount | 200 | 98 | 49% |
| Tampered: broken_balance | 200 | 72 | 36% |
| Tampered: font_mismatch | 200 | 70 | 35% |
| All tampered | 1,000 | 541 | 54.1% |

In every tampered document with a later ModDate, `metadata_altered = 1`; none of the 459 other tampered documents has a later ModDate.

Producer values:

| Group | Producer values (count) |
|---|---|
| 64 re-saved genuine | BrowserPrint PDF 118 (18), PrintToPDF Driver 10.0 (18), SaveAsPDF Mobile 3.2 (15), OS Quartz-style PDF Export (13) |
| 936 other genuine | Aldermoor Reports 7.0 (188), Corvane StatementEngine 4.2 (188), HLB DocGen 2.9.1 (188), KMB Mainframe Print Bridge 3.4 (187), Vellichor Paperless 1.12 (185); none has a later ModDate |
| 541 tampered with later ModDate | QuickPDF Editor 7.4, DocuMod Online 2.3, PDFTweak Pro 11.0, OpenPage Writer 3.8, EasyEdit PDF 5.1 (only these five) |

Does the current Node 2 rule fire in the dataset? `VF_EDIT_SOFTWARE` fired on 0 of 2,000 documents and `VF_CREATION_BEFORE_PERIOD` on 0 of 2,000. The five editor names in the tampered files are not in the list, so the existing editing-tool check is blind to the whole dataset, not only to E.

### What a future rule would do on this dataset

| Candidate rule | Tampered flagged (of 1,000) | Genuine flagged (of 1,000) | Reading |
|---|---|---|---|
| ModDate later than CreationDate, alone | 541 | 64 (all re-saved) | Catches over half of the tampered files but falsely flags every re-saved genuine file (6.4% of genuine). Precision about 89%. A weak or MEDIUM signal at most, never HIGH on its own. |
| Producer in a longer editing-tool list (the five editor names above) | 541 | 0 | Separates cleanly here, but only because this synthetic dataset uses disjoint names. |
| Both: later ModDate AND an editing-tool Producer | 541 | 0 | Same as above; the second condition is what removes the 64 false alarms. |
| Both: later ModDate AND a print/export Producer | 0 | 64 | Describes the genuine re-saved group; a possible allow-list to soften the first rule. |

So a later ModDate by itself does discriminate (0% of non-re-saved genuine, 54% of tampered) but it cannot tell a tamper from a re-save. The Producer is what separates them in this data.

### Limits (what was not verified)

- The dataset is synthetic. Its Producer names are invented. Whether real editing tools and real print-to-PDF drivers separate as cleanly is not known.
- Only E and F were read as real labelled files. One tampered example (E) fits the pattern; one genuine example (F) has equal dates.
- ModDate was compared at full timestamp precision. Date-only comparison and a minimum time gap were not tested.
- Not checked: XMP metadata, incremental updates in the PDF body, or any metadata in files that have no ModDate (none in this dataset).

### Recommendation for a later Node 2 spec

1. Do not add "ModDate later than CreationDate" as a stand-alone finding. It would falsely flag all 64 re-saved genuine documents.
2. If a rule is added, combine the later ModDate with a Producer or Creator that is not a known banking or report system or print/export driver, or with a known editing-tool name. Weigh it as MEDIUM unless other evidence agrees.
3. The existing `_EDITING_SOFTWARE` list is image-editor only and misses the PDF editors seen here. Extending it needs real-world tool names, not just the five invented ones.
4. Re-test any change against the 64 `genuine_resaved` documents first.

## I-2: `VF_CURRENCY_MIX` on both files

### The check (read from `visual_forensics.py`)

Node 2 runs the regex `(SGD|S\$|\bUSD\b|US\$|\$)` over the whole `raw_text` of the document. `SGD` counts as one style, `S$` and `$` count as a second style (`$-symbol`), `USD` and `US$` are their own styles. If more than one style is present the finding fires (MEDIUM, weight 10). It does not look at where the text sits: header, table or footer are treated alike.

### Exact matches on the two files

| | statement_E | statement_F |
|---|---|---|
| `S$` matches | 460 | 470 |
| `SGD` matches | 6 | 6 |
| Where `SGD` comes from | the header line `BUSINESS CURRENT ACCOUNT Currency SGD`, repeated once per page (6 pages) | same line, 6 times |
| Where `S$` comes from | every amount, e.g. `S$34,275.95` and `1 Apr GIRO RENT OSTERLY RISE HOLDINGS S$5,800.00 S$25,312.18` (F) | same |
| Styles found | `SGD`, `$-symbol` | `SGD`, `$-symbol` |
| Result | `VF_CURRENCY_MIX` fires | `VF_CURRENCY_MIX` fires |

Both files fire for the same reason. The ISO code is in the account header (it labels the account currency) and the symbol is on every amount. These are not two competing notations for amounts. All amounts use `S$`. No `USD` or `US$` was found. I read the F line text directly; for E I recorded only the counts from the same regex, not the full line text.

### Dataset results (all 2,000 documents)

Counts of documents where `VF_CURRENCY_MIX` fired:

| Layout (bank) | Genuine fired / total | Tampered fired / total |
|---|---|---|
| 1 Corvane Trust | 0 / 200 | 0 / 200 |
| 2 Halcyon Ledger Bank | 0 / 200 | 0 / 200 |
| 3 Aldermoor Savings | 0 / 200 | 0 / 200 |
| 4 Kestrelmark Bank | 0 / 200 | 0 / 200 |
| 5 Vellichor Bank | 200 / 200 | 200 / 200 |
| Total | 200 / 1,000 (20%) | 200 / 1,000 (20%) |

Tampered documents by `tamper_type`: 40 of 200 fire for each of altered_date, broken_balance, edited_amount, font_mismatch and metadata_inconsistency. That is 200 in total, and all of them are layout 5.

The genuine total of 200 splits into 185 non-re-saved and 15 re-saved genuine documents. Layout 5 documents always fire, whether genuine or tampered, and layouts 1 to 4 never do. In one sampled layout 5 document the pattern is the same as F: `Currency SGD` in the header and `S$` on the amounts (70 `S$` and 1 `SGD` in that sample).

Conclusion from the data: the finding is a layout property, not a forgery signal. It fires on 100% of the Vellichor layout (E and F are both Vellichor-style) and on 0% of the other four, with the same rate for genuine and tampered. As a rule it has no discriminating power, and it gives every Vellichor customer a MEDIUM penalty of weight 10.

### Recommendation (for a LATER Node 2 spec): narrow, or drop

- Keep as is: not recommended. 20% of genuine documents get a false finding and it adds no information.
- Narrow (preferred if some check is wanted): compare currency notation only inside the transaction table amounts, for example flag only when amounts in the table use more than one notation (`S$` on some rows, plain numbers or `$` or `SGD` prefix on others). Ignore the account-header currency code, which labels the account and is not an amount. The narrowed rule would not fire on E or F, since all their amounts are `S$`.
- Drop: acceptable if no narrowed version can be shown to catch anything. No dataset tamper type here changes currency notation, so the dataset cannot prove that a narrowed rule has any value.

### Limits (what was not verified)

- No narrowed rule was implemented or run. The expectation that it stays silent on layouts 1 to 5 is from reading the text patterns, not from a run.
- The dataset has no tamper that mixes currency notation inside the amounts, so catch rate for a narrowed rule is unmeasured.
- Layouts 1 to 4 were not inspected for how they print the currency; they simply did not trigger the regex pairing.

## Summary

| Question | Finding | Evidence | Next step |
|---|---|---|---|
| Is metadata absent from E? | No | Producer, Creator, CreationDate, ModDate all present | none |
| Is "OpenPage Writer 3.8" recognised? | No | Not in `_EDITING_SOFTWARE` (image editors only) | extend list in a Node 2 spec |
| Does Node 2 compare ModDate with CreationDate? | No | `mod_date` is never read in `analyze` | add only with a Producer condition |
| Cause for E | Tool name missing from list and no ModDate check | see I-1 table | Node 2 spec |
| Re-saved genuine with later ModDate | 64 of 64 | `genuine_resaved = 1` | test any rule on them first |
| Genuine (not re-saved) with later ModDate | 0 of 936 | dataset run | n/a |
| Tampered with later ModDate | 541 of 1,000 (metadata_inconsistency 200 of 200) | dataset run | n/a |
| Current `VF_EDIT_SOFTWARE` on dataset | 0 of 2,000 | dataset run | list is blind to the dataset |
| `VF_CURRENCY_MIX` cause on E and F | `SGD` in header (6) plus `S$` on amounts (460 and 470) | text counts | narrow or drop |
| `VF_CURRENCY_MIX` on genuine | 200 of 1,000 (all layout 5) | dataset run | narrow or drop |
| `VF_CURRENCY_MIX` on tampered | 200 of 1,000 (40 per tamper type, all layout 5) | dataset run | no discrimination |
| Recommendation | Metadata: combine ModDate with Producer. Currency: narrow to table amounts, or drop | above | LATER Node 2 spec |
