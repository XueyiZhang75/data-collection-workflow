# Source data for three public-health case studies

Data accompanying *Reimagining Public Health Data Collection with Agentic AI*.

Package version: 2026-10-20. The observation, source and reference records retain the dataset versions documented in their source metadata.

## Start here

- [Observation records and source information](records/README.txt): the compact CSV datasets for the MV Hondius hantavirus outbreak, Canadian measles in 2025, and Sierra Leone mpox in 2025.
- [Field guide](FIELD_GUIDE.md): identifiers, counts, missing values, date meanings and evidence categories.
- [Record scope and source traceability](SOURCE_TRACEABILITY.md): how the retained records were used and how to follow their source links.
- [Column inventory](column_inventory.csv): every CSV column and its non-empty value count.
- [Categorical values](categorical_values.csv): recorded role, date-quality, count-semantics and evidence-category values in the compact CSVs.
- [Source attribution](SOURCE_ATTRIBUTION.md): comparison datasets, clinical references and source-specific licence statements.
- [File manifest](file_manifest.csv): file sizes, SHA-256 checksums and CSV record counts. The manifest does not include itself.

## Compact data

| Case study | Folder | Principal records |
|---|---|---|
| Hantavirus | [records/hantavirus](records/hantavirus) | 13 case records; 375 case-to-source associations; 10 WHO report records; 13 Global.health benchmark cases; 117 clinical-reference field entries |
| Canadian measles | [records/canada](records/canada) | 217 observations; 50 national reference records; 56 reference-source records; 107 provincial reference records |
| Sierra Leone mpox | [records/mpox](records/mpox) | 75 national records (44 workflow and 31 WHO reference); 78 national source records; 51 district records (37 workflow and 14 WHO reference); 51 district source records |

These files contain retained observations, source information and comparison or assessment references for the three case studies. An observation or source association is not necessarily one patient, a new case, or an independent source. The retained inventories and reference series are broader than the subset of observations selected for individual curves or paired comparisons. Their roles and exclusions are explained in SOURCE_TRACEABILITY.md.

CSV files use UTF-8 encoding. Blank cells are missing or unspecified values, not zero. Report publication dates, statistical cutoffs, event dates and date bounds have distinct meanings. Original publisher PDF and HTML files are not included; source URLs, document hashes and retained evidence passages identify the supporting reports. A source address records the original report location, not a guarantee that a third-party website will remain accessible.

## Attribution and reuse

Reference datasets and quoted source material retain their original attribution and terms. See [SOURCE_ATTRIBUTION.md](SOURCE_ATTRIBUTION.md), including the unchanged upstream Global.health README and licence file. No package-wide reuse licence is specified.
