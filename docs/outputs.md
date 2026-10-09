# Reading the output files

Start with **`final_report.html`**. It brings together the task and conclusions, all discovered sources, unresolved questions, data downloads, and recorded run settings. **`session_report.zip`** contains that report and its supporting `data/` and `evidence/` files; extract it before opening the report.

Paths below are relative to the session directory. The report's `data/` exports and the detailed `collection/` exports are views of the same run, so do not concatenate them as independent data.

## Files to open

| File | Contents |
| --- | --- |
| `data/final_dataset.csv` / `.json` | Observations retained in the final dataset. In evidence mode, these passed evidence qualification. Standard-mode collected observations are not automatically evidence-qualified. |
| `data/final_case_dataset.csv` / `.json` | The individual-case subset. An empty file does not establish that no cases occurred. |
| `collection/aggregate_dataset.csv` / `.json` | Aggregate observations, such as a district's case count for a reporting period. |
| `data/candidate_records.csv` / `.json` | Records still requiring evidence checks. For standard-mode runs, this export may also contain pending-review or excluded records, with their decisions. |
| `data/context_records.csv` / `.json` | Background observations kept separately from the main quantitative results. |
| `data/source_catalog.csv` / `.json` | All report sources, including unread, failed, deferred, and excluded sources; publisher information, URLs, dates, processing status, and data contributions. |
| `data/task_result.json` | Task answers and their scope: full-period, completed-outbreak, latest-available, or specified-period results, with supporting sources. |
| `data/evidence_chunks.json` | Saved passages with document and location metadata. |
| `data/record_inclusion_decisions.json` | Record-level qualification or inclusion decisions and their reasons. |
| `data/run_settings.csv` / `.json` | Configurable settings, recorded values, and where those values came from. |
| `data/report_snapshot.json` | The report's task, answers, result status, counts, and run information. Its generation time can be later than the actual run. |
| `evidence/` | Saved source material referenced by the report, when available. |

## Main observation fields

CSV and JSON contain the same observation values. JSON keeps nested objects and arrays; CSV places those objects and arrays in cells as JSON text. The exact columns depend on the run and dataset. The [record models](../src/data_collection_workflow/models.py) define the shared fields; evidence mode adds the qualification object described below.

| Field or field group | How to read it |
| --- | --- |
| `record_id` | The observation's identifier, used to join its evidence and decisions. It is not a patient identifier. |
| `disease`, `disease_standard_name` | The reported disease and its standardized name when available. |
| `country`, `subnational_location`, `locality` | Geographic scope. A blank local area does not imply national coverage; also inspect `geographic_scope` and `aggregation_level`. |
| `cases_confirmed`, `cases_probable`, `cases_suspected`, `cases_unspecified` | Counts by the source's case definition. These categories may overlap; do not sum them without supporting definitions. |
| `deaths`, `hospitalizations`, `icu_admissions` | Counts for the stated scope and period. Admission counts and numbers of people may differ; retain the source's unit and definition. |
| `metric_name`, `metric_value`, `metric_unit`, `metric_denominator` | Other measures, including rates and percentages. Keep the value, unit, and denominator together; a rate is not a case count. |
| `count_unit`, `count_semantics`, `statistical_count_type` | What is counted and whether a value is cumulative, newly reported, annual, a subset, or unspecified. These are descriptive strings, not a guarantee that every record has a known count basis. |
| `reporting_period` | The source's statistical interval label. It can be text such as a month, week, or date range. |
| `metric_period_start`, `metric_period_end` | The start/end of the metric's statistical interval, when supported. |
| `as_of_date` | The statistical cutoff: the value is reported as of this date. |
| `date_reported`, `report_date` | The observation's reporting date. This is not necessarily its statistical cutoff or the webpage's publication date. |
| `publication_date` | The source's publication date, when recorded. It does not itself establish the observation period. |
| `event_start_date`, `event_end_date`, `date_onset`, `date_confirmation`, `date_death` | Event or person-level dates, when the source supports them. Do not substitute these for each other. |
| `source_id`, `source_url`, `source_title`, `publisher` | The source reference for the observation. Use the source catalogue for publisher-verification and republication details. |
| `evidence_quote`, `supporting_chunk_id`, `document_id` | The quoted passage and references used to locate supporting material. A quote alone is not proof that every extracted field passed checking. |
| `evidence_qualification` | Evidence-mode decision object: `status`, `product_kind`, `reasons`, and `field_evidence`. Each field-evidence entry identifies the field/value, document hash, locator, quote, support decision, and reason. |
| `product_kind` | Evidence product type: `case_level`, `aggregate`, `context`, or `unresolved`. A row with an aggregate count is not one patient. |
| `requires_human_review`, `review_status`, `record_final_inclusion_status`, `quality_gate_reasons` | Review and inclusion information, where the run records it. Review status is separate from field-evidence support. |
| `duplicate_of_record_id`, `representative_record_id`, `linked_event_id` | Links between overlapping or related records. Use them before aggregating across sources. |
| Fields ending in `_raw`, plus warning/reason arrays | Original values and explanations retained alongside normalized values. |

An evidence qualification `status` of `qualified` means the record passed the workflow's evidence checks for its scope. `candidate` means it did not; read `reasons` and individual `field_evidence` entries. Being present in a source catalogue, being downloaded successfully, and being supported evidence are different states.

Dates retain the available precision: a day may be `2025-01-31`, while a month-level value may remain `2025-01`. A task's requested start/end dates are search requirements, not evidence that an observation covers those dates. A completed-outbreak total also remains distinct from a calendar-year total. The report and `data/task_result.json` make these scope distinctions explicit.

`country` can identify a province's parent country. For example, an Ontario count belongs in a Canada collection task while its statistical scope remains Ontario; it cannot supply Canada's national total. A combined confirmed-and-probable count retains that definition in `case_definition` and is not relabeled as confirmed cases alone.

An observation crossing a task boundary can be retained with `task_temporal_relation: "overlaps_task_boundary"` in its qualification. Its original count and interval remain intact; the workflow does not split or prorate the count. A month-precision reporting date also does not establish coverage of every day in that month.

A year label alone can locate a record within the task without proving a full-year total. Such records carry `temporal_extent: "year_label_only"`; they do not complete annual coverage or supply the full-year headline. Explicit annual reporting and supported statistical intervals retain their own meaning.

If an unsupported optional descriptor can be set aside while all remaining facts independently pass checks, the workflow retains the original candidate and emits a rechecked record linked by `recovered_from_record_id`. Its `evidence_normalization_actions` preserve the original value, citation, and reason. This does not remove failures in counts, geographic scope, statistical meaning, required dates, or evidence integrity.

Source-processing details distinguish recorded empty extraction results, failed attempts, skipped passages, pending work, and attempts whose outcomes were not recorded. An empty extraction result does not by itself mean retrieval or model execution failed.

Missing numeric values are **`null` in JSON and blank cells in CSV**. They mean the value is unavailable, not zero. An absent JSON key means that field was not included in that record; an empty string can be retained source text. CSV cannot preserve all these distinctions, so use JSON when they matter. Boolean values appear as `true`/`false` in JSON and normally `True`/`False` in CSV. Empty arrays and objects are `[]` and `{}`. Read CSV as UTF-8; the report's `data/` CSV files include a UTF-8 byte-order mark for spreadsheet compatibility.

Do not add cumulative snapshots from different dates together. Do not add a national total to its constituent regional totals or count duplicated reports as independent observations.

## Small format illustration

**The two records below are fictional illustrations, not session results or qualified evidence.** They show only selected fields. The first is a weekly newly reported count; the second is cumulative through a cutoff. They must not be added together.

```csv
record_id,disease,country,cases_confirmed,deaths,statistical_count_type,reporting_period,as_of_date,source_url
example_1,Example disease,Example country,12,,newly_reported,2025-01-01 to 2025-01-07,,https://example.org/weekly
example_2,Example disease,Example country,40,1,cumulative,,2025-01-31,https://example.org/cumulative
```

The same selected fields in JSON:

```json
[
  {
    "record_id": "example_1",
    "disease": "Example disease",
    "country": "Example country",
    "cases_confirmed": 12,
    "deaths": null,
    "statistical_count_type": "newly_reported",
    "reporting_period": "2025-01-01 to 2025-01-07",
    "as_of_date": null,
    "source_url": "https://example.org/weekly"
  },
  {
    "record_id": "example_2",
    "disease": "Example disease",
    "country": "Example country",
    "cases_confirmed": 40,
    "deaths": 1,
    "statistical_count_type": "cumulative",
    "reporting_period": null,
    "as_of_date": "2025-01-31",
    "source_url": "https://example.org/cumulative"
  }
]
```

## Source catalogue and run settings

`source_catalog.json` is an object containing `metadata` and a `sources` array; the CSV contains those source rows. `report_source_id` matches report citations such as `S001`. A row can retain several original `source_ids` and explicit URL aliases. Key fields include `url`, `title`, `publisher`, `publication_date`, `statistics_period`, `processing`, `record_counts`, and `records_by_role`. Publisher, dates, and processing information can be nested objects, so JSON is easier to inspect in code. Source contributions are not additive record totals: a record may reference more than one source.

`run_settings.json` groups setting rows under `groups`; its CSV flattens those rows and adds `group_id` and `group_label`. Each setting has a `key`, `configured_value`, `effective_value`, `value_status`, and `basis`. `recorded_configuration` means the value came from the saved configuration; `recorded_effective` means runtime evidence recorded it. `not_recorded` means the historical value is unknown, even if a current default is displayed. `credential_redacted` marks a setting whose secret value is not exported. Explicit `false`, `0`, and `null` are not interchangeable with an unrecorded setting.

See [Configuration: reporting periods and task answers](configuration.md#reporting-periods-and-task-answers) for the machine-readable answer fields and [the recorded runtime example](runtime_example.md) for one session's budgets and elapsed time.
