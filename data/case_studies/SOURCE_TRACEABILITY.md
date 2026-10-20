# Record scope and source traceability

The three case-study folders contain observation records, source metadata and reference data used in the manuscript. They are not a collection of every search result or extraction candidate. A record can support source coverage, reporting-date evidence, a comparison reference or a clinical-field assessment without becoming a separate point on the final count curve.

## Hantavirus

| File | Records | Use and source route |
|---|---:|---|
| `case_records.csv` | 13 | Final numbered case records used in the field comparison. Each populated field has its own `*_source_url` and supporting quotation. Follow that field's URL directly; the case-source inventory is not an exhaustive index of every field-level URL. |
| `case_sources.csv` | 375 | The accepted case-to-source associations used for source-coverage reporting. Every row has a source URL. Read `support_scope` before treating an association as direct evidence for a case. |
| `benchmark_cases.csv` | 13 | Global.health benchmark cases numbered 1-13. The source snapshot and upstream attribution are documented in SOURCE_ATTRIBUTION.md; original `Source_I` to `Source_VII` fields retain report links. The other 11 non-case or unnumbered rows in the upstream table are not included. |
| `who_report_records.csv` | 10 | WHO release-date counts used for the cumulative-count comparison. Every report has its source URL. Reports with an unchanged total remain part of the retained reporting sequence. |
| `clinical_reference.csv` | 117 | Nine field slots for each of the 13 cases. All 87 evaluable reference values have source URLs. Of the other 30 slots, one is not applicable and 29 are marked `not_publicly_available`; those 29 have no reference value or source URL and are not additional clinical observations. |

The 375 source associations comprise 146 direct-case associations, 156 event-level corroborative associations, 40 other corroborative associations and 33 citation-only associations. The citation-only rows have no supporting excerpt in this inventory; they are retained as part of the reported source-coverage inventory, not as direct proof of a case's clinical fields. The same report can be associated with more than one case.

## Canadian measles

| File | Records | Use and source route |
|---|---:|---|
| `observations.csv` | 217 | Exactly the source-observation set retained in Supplementary Section S3. Each row has `source_url`. There are 140 original-site URLs and 77 Wayback snapshot URLs; each snapshot address embeds the original-site URL. |
| `reference_records.csv` | 50 | The retained national reference series. Join `count_evidence_ids` and `date_evidence_ids` to `reference_sources.csv:evidence_id` to obtain report and date-evidence URLs. |
| `reference_sources.csv` | 56 | Exactly the sources cited by those count/date evidence IDs: 45 support both count and date, five support count only and six support date only. Every row has `source_url`. |
| `provincial_reference_records.csv` | 107 | The provincial reference records retained for Manitoba (33), Ontario (42) and British Columbia (32). Every row has `source_url`; source identity and reporting dates match the retained provincial reference nodes. |

The 217 observations exclude 26 other records in the wider source ledger: 16 outside the task criteria, two duplicates and eight PHAC-repeated observations. The reference-source file excludes 11 source-ledger entries that were selected for neither count nor date evidence.

Eleven of the 217 retained observations never become an active component of the displayed daily series, but all 217 contribute to the reported source composition. Three national and 12 provincial reference records precede the first compatible workflow observation and do not enter paired-error comparisons; they remain part of the retained reference series. A reference-source link can identify a report explicitly attributing figures to PHAC, rather than a direct PHAC webpage; the publisher and source-role fields preserve this distinction.

## Sierra Leone mpox

| File | Records | Use and source route |
|---|---:|---|
| `national_records.csv` | 75 | Exactly the retained national inventory: 44 workflow and 31 WHO reference observations. Each has `source_url` and a matching `source_id` in national_sources.csv. |
| `national_sources.csv` | 78 | Sources for the 75 observations plus three publisher-index pages explicitly cited as publication-date evidence. Every source has a URL. |
| `district_records.csv` | 51 | Exactly the retained Western Area Urban inventory: 37 workflow and 14 WHO reference observations. Each has `source_url` and a matching `source_id` in district_sources.csv. |
| `district_sources.csv` | 51 | Sources corresponding to the retained district observations; every source has a URL. |

The wider ledgers contain another 78 national and 24 district observations that fail the retained eligibility criteria; they are not in this deposit. All 126 deposited observations resolve to their source records, with matching report addresses. The three additional national source pages support the dates of the NYSDOH relay reports of 4 September, 6 November and 18 December 2025.

Retained observations are not all distinct final count nodes. Ten national observations are not selected as count-changing nodes: three workflow reports and seven WHO reports. Some unchanged reports contribute to evidence freshness; older statistical cutoffs and unresolved downward reports remain visible in the reference inventory but do not update the retained count. The district inventory has 37 workflow observations, while the final displayed curve selects 33 workflow and 14 WHO observations. The other four workflow reports remain part of the source inventory described in Supplementary Section S4.

## What the source links establish

Direct source fields or the documented identifier joins allow the retained observed values and reference reports to be traced to their recorded report locations. Empty clinical-reference slots are explicitly identified above. This is a check of dataset membership and recorded provenance, not a new validation of every quoted claim or a guarantee that every external website remains online. Archived links are retained when the dataset uses a historical page version. Missing reference values are not replaced by guessed URLs.
