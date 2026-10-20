# Field guide

## Reading the records

`column_inventory.csv` lists every CSV column, record count and non-empty count. `categorical_values.csv` preserves the actual categorical labels and their frequencies. [SOURCE_TRACEABILITY.md](SOURCE_TRACEABILITY.md) explains file-level uses and source-link routes.

## Identifiers and links

| Fields | Interpretation |
|---|---|
| `case_label` | Case identifier linking hantavirus records, source associations and clinical-reference entries. |
| `Gh_ID`, `WHO_case number` | Identifiers retained from the Global.health benchmark; not additional cases. |
| `observation_id`, `source_id` | Observation and source identifiers linking the corresponding tables within a case study. |
| `evidence_id`, `count_evidence_ids`, `date_evidence_ids` | Canadian reference-source identifiers supporting counts and dates; count and date evidence can differ. |
| `source_url`, `url`, `final_url`, `landing_url`, `canonical_source_url` | Source addresses retained in the corresponding record. Canonical or final addresses may differ from the original URL. |
| `source_domain` | Publisher hostname recorded for a source association. |
| `raw_sha256`, `source_sha256`, `sha256`, `text_sha256`, `archived_sha256` | Hashes identifying the corresponding source representation. The source documents themselves are not bundled. |
| `evidence_quote`, `supporting_quote`, `supporting_quotes`, `evidence_text`, `evidence_locator` | Retained support passages or locators; inspect them together with source identity and evidence scope. |

## Counts and missing values

Counts retain their source definitions. `confirmed`, `probable`, `suspected`, `reported_total`, `combined_total` and `reported_count` are not interchangeable. Read `count_semantics`, `count_scope`, `case_definition`, geographic scope and period scope before comparing values. Cumulative observations must not be summed across reporting dates. Canadian observations include overlapping reports and disjoint components; the national series applies the source-selection and aggregation rules described in the manuscript rather than summing every report.

Blank values mean missing or unspecified, not zero. In hantavirus clinical_reference.csv, `status=confirmed` describes an available reference value, `not_publicly_available` marks an unavailable reference, and `not_applicable` marks an inapplicable field. These status labels describe the reference cell, not the patient's disease classification. Only 87 of the 117 clinical-reference cells are evaluable in the reported paired assessment.

## Date meanings

| Fields | Interpretation |
|---|---|
| `publication_date`, `public_date`, `public_release_date`, `report_date`, `report_issue_date` | Dates of reports, issues, releases or documented public availability, according to the accompanying basis fields. They do not establish the workflow's real-time acquisition date. |
| `as_of_date`, `data_as_of_date`, `week_end`, `period_end` | Statistical cutoff or reporting-period end. |
| `retrieval_date`, `retrieved_at`, `original_retrieved_at` | Recorded retrieval timestamps; distinct from a report's publication date. |
| `*_lower_bound`, `*_upper_bound` | Bounds on the named date or quantity. Date bounds are not confidence intervals. |
| `date_quality`, `publication_date_basis`, `publication_date_provenance`, `publication_date_status` | Recorded evidence or interpretation underlying the date. Preserve the original labels. |
| `effective_public_date` | Date selected for a comparison series under the documented date rules; consult bounds and provenance before interpreting it. |
| `onset_date`, `confirmation_date`, `Date_onset`, `Date_confirmation` | Clinical-event dates retained in the case or benchmark records; not workflow discovery timestamps. |

The Western Area Urban ledger includes a district cutoff assigned from an adjacent national update; it is not separately established by the district report. Read the record's date-quality and notes fields before using that cutoff.

## Source roles and evidence scope

In the mpox files, `role=workflow` identifies the collection series and `role=who_reference` identifies the reference series. In the other case studies, the collection and reference roles are largely separated by file. Publisher labels and source categories describe reporting channels, not independent surveillance systems.

The 375 hantavirus case-to-source associations contain multiple evidence scopes. `support_scope=direct_case_evidence` denotes direct support for a specific case; other labels retain corroborative or benchmark-linked associations. An association must not be treated as direct case evidence solely because it has a URL. The published direct-evidence subset contains 146 associations. Refer to `corroboration_basis`, `anchor_fields` and the retained support passages for each association's basis.

The Global.health case list is the comparison benchmark. Clinical reference values used to assess both datasets are separately provided in clinical_reference.csv, together with field-level source links. Full citations for the case-table publication, its addendum and the WHO risk assessment are listed in [SOURCE_ATTRIBUTION.md](SOURCE_ATTRIBUTION.md).
