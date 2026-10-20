CASE-STUDY DATA

The three folders contain the observation records, source information and reference data used for the hantavirus, Canadian measles and Sierra Leone mpox case studies.

hantavirus/
  case_records.csv: 13 case records with field-level source evidence.
  case_sources.csv: case-to-source associations.
  who_report_records.csv: WHO report counts, dates and sources.
  benchmark_cases.csv: the 13 numbered cases in the Global.health benchmark.
  clinical_reference.csv: reference field values and supporting sources.

canada/
  observations.csv: 217 source observations, reported counts and evidence.
  reference_records.csv: 50 national reference records.
  reference_sources.csv: sources supporting the national reference counts and dates.
  provincial_reference_records.csv: reference records for Manitoba, Ontario and British Columbia.

mpox/
  national_records.csv: national observations and WHO reference records.
  national_sources.csv: the corresponding source information.
  district_records.csv: Western Area Urban observations and WHO reference records.
  district_sources.csv: the corresponding source information.

CSV files use UTF-8 encoding. Blank values are missing or unspecified, not zero. Source dates, statistical cutoffs and date bounds retain their stated meanings. Source URLs, quotations, identifiers and document hashes identify the supporting reports. The role field distinguishes collection records from reference records.

The Global.health benchmark version is dated 30 June 2026, commit 16dd94e127223a3abc5295cf72c649c132b53509:
https://github.com/kraemer-lab/Hondius_hantavirus_h2026/tree/16dd94e127223a3abc5295cf72c649c132b53509
The source repository provides its attribution and licensing information.
