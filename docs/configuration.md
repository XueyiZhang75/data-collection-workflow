# Configuration

Use `scripts/collect.py` for interactive task entry, or copy `configs/workflow.jsonc` into the Git-ignored `configs/local/` directory, edit the copy, and run `scripts/run_workflow.py --config <path>`. JSONC permits comments; use valid JSON syntax without trailing commas. Configuration values merge with built-in defaults, so explicitly set optional switches that matter to your task.

## Task and modes

Fill in `structured_task.disease`, `location`, `start_date`, and `end_date`. Also provide a nonempty `structured_task.target_fields` list when using `validate-config`; the [configuration-file guide below](#run-with-a-configuration-file-optional) contains a complete task example. Target fields and source preferences are configurable. `user_request` can add context to these explicit fields; it does not replace them. Missing task fields stop collection before provider calls. The blank template does not select a case study for you.

`pipeline_mode` selects execution behavior:

- `standard`: collection and validation chain.
- `evidence`: field evidence qualification, local acquisition readiness, persisted budgets, recovery, and checkpoint resume.

`workflow.collection_mode` is a separate setting controlling source and validation handling (`standard`, `masked_validation`, or `direct_collection`). A pipeline mode is not a source-holdout policy.

The interactive entry accepts `--pipeline-mode`; the configured runner and `collect` CLI subcommand read the mode from the configuration file. Their flags are not interchangeable. Use `--help` on the entry you are running.

## Run with a configuration file (optional)

Use this alternative to interactive task entry when you want to edit target fields, search/fetch limits, model settings, or allowed/blocked source domains. It is not an additional step after the README tutorial. Complete the installation, credentials, and browser/OCR setup in the [README](../README.md) first.

Start from the blank template and keep your task file in the Git-ignored `configs/local/` directory.

**PowerShell:**

```powershell
New-Item -ItemType Directory -Force configs/local
Copy-Item configs/workflow.jsonc configs/local/my_task.jsonc
```

**macOS or Linux:**

```bash
mkdir -p configs/local
cp configs/workflow.jsonc configs/local/my_task.jsonc
```

Edit `configs/local/my_task.jsonc`:

1. Set the top-level `pipeline_mode` to `"evidence"`.
2. Replace the empty `structured_task` object with your task, for example:

   ```json
   "structured_task": {
     "disease": "Measles",
     "location": "Canada",
     "start_date": "2025-01-01",
     "end_date": "2025-01-31",
     "target_fields": [
       "disease", "country", "subnational_location", "date_reported",
       "cases_confirmed", "deaths", "source_url", "source_type", "evidence_quote"
     ]
   }
   ```

3. In the existing `output` object, set `session_id` to `"my_configured_collection"`. Leave `run_output_root` as `"outputs"`.
4. Leave `llm.provider` and `llm.model` blank to inherit `.env`, or fill both explicitly. Adjust the [budgets](#budgets) if needed. The file-driven runner uses the file/default budgets; it does not add the interactive quick-run preset.

Preserve the surrounding commas when editing the JSONC template. Validate and preview it:

```bash
data-collection-workflow validate-config --config configs/local/my_task.jsonc
data-collection-workflow collect --config configs/local/my_task.jsonc --dry-run
```

Resolve any validation issues and confirm `valid: true`. Then start the configured run:

```bash
python scripts/run_workflow.py --config configs/local/my_task.jsonc
```

To resume it later, keep the same configuration file:

```bash
python scripts/run_workflow.py --config configs/local/my_task.jsonc --resume-session my_configured_collection
```

`pipeline_mode` belongs in the configuration for this route: `scripts/run_workflow.py` and the installed `collect` subcommand do not accept `--pipeline-mode`. The separate `workflow.collection_mode` setting controls source/validation roles. See [Task and modes](#task-and-modes) for the distinction.

## Credentials and providers

Search supports Tavily and offline fixtures. Model adapters support Anthropic and OpenAI. Set keys in your environment or local `.env`; select the provider/model with `llm.provider` and `llm.model` or supported command-line flags. Model availability depends on your account.

Use `ANTHROPIC_API_KEY` with `anthropic` or `OPENAI_API_KEY` with `openai`; use `TAVILY_API_KEY` for live search. Copy `.env.example` to `.env` in the source checkout. Collection entry points load that file without replacing existing shell variables. Credentials stay in the environment and must not be added to shared configuration files.

Provider and model selection follows this precedence:

1. Explicit `--provider` and `--model` arguments, where supported.
2. Nonempty `llm.provider` and `llm.model` in the task configuration.
3. `LLM_PROVIDER` and `LLM_MODEL` in the environment (shell values take precedence over `.env`).

If the provider is still unspecified, it defaults to `anthropic`. There is no default model. The task template leaves both fields blank so users can configure their own account. Model IDs are free-form; the workflow does not limit users to the models used during development. The selected model must support the features used by the workflow, and live preflight checks report account or model compatibility errors.

Changing the provider without supplying a model discards a model inherited from the previous provider. An environment model is inherited only when its provider matches the selected provider, or when `LLM_PROVIDER` is unset. Supply both values when switching accounts or providers. The workflow does not silently substitute another model.

The interactive entry prompts for a missing model in a terminal. Noninteractive model-assisted runs stop with a configuration error before collection if no model is selected. Configuration previews, the blank visual editor, and offline runs can be used without one. Langflow exposes provider and model fields; its blank fields inherit the environment. Model and provider choices are saved in the effective session configuration, so changing them requires a new session.

## Budgets

Evidence mode persists operation usage in the session. `universal.budget_policy` selects its budget policy and `universal.budget_limits` sets operation limits. Retried calls count; cached successful responses are reused. Usage limits are operation counts, not a dollar spending cap. Token usage is recorded when the provider returns it.

### Historical source versions

Live discovery in evidence mode also checks the Internet Archive CDX index for historical versions of task-matching pages found by search. It prioritizes registered official sites and also admits monitoring or report pages on unregistered domains, without assigning them official status. It reserves at most one fifth of the existing search-query capacity (up to four original URLs) and one quarter of the result capacity. Shared session limits still apply. Standard mode and fixture searches do not contact the archive.

Optional settings under `universal.historical_discovery` are `enabled` (default `true`), `max_origins` (default and maximum `4`), `max_records_per_origin` (default `64`, maximum `256`), `max_results` (default `64`, maximum `256`), and `timeout_seconds` (default `10`, maximum `30`). Set `enabled` to `false` to use all discovery capacity for ordinary search.

The lookup uses exact original URLs and capture dates within the task window, so separate monthly or quarterly tasks discover versions from their own periods. This is a retrieval window, not evidence that the captured page reports observations in that period. It does not retrieve archived page bodies. Each candidate retains its original URL, parent source, replay URL, capture timestamp, and index record in `historical_snapshot` in the source registry. Capture time is not a publication date or a reporting period; candidates remain `index_only` and do not qualify as data evidence. The discovery summary distinguishes empty indexes, errors, exhausted budgets, and truncated results. A truncated index is incomplete coverage, not evidence that no other versions exist.

Evidence-mode coverage exports include a `reporting_timeline` diagnostic inside `source_coverage_audit.json`. It keeps candidate publication metadata separate from source-supported observation cutoffs (`as_of_date`) and reporting dates (`report_date` or `date_reported`). A supported reporting date describes the observation's reporting event, not verified document publication. Archive captures retain their own provenance and never fill either timeline. Excluded, validation-only, unrelated, and unconfirmed collection sources do not supply timeline points.

Each timeline lists known dates, separate counts of unknown dates and sources dated only outside the window, and up to 24 gaps between known points, including the window boundaries. Gaps carry stable identifiers, their date basis, inclusive endpoints, and day counts; larger gaps appear first, and omitted gaps are counted. These are discovery opportunities, not missing reports, zero cases, or a completeness measure. An annual aggregate can satisfy observation-period coverage while leaving the report-date inventory empty. The diagnostic uses exact task endpoints for daily through multi-year windows, handles leap days, and does not create calendar bins or assume weekly publication. Invalid or reversed windows are marked `not_evaluable`. Building this diagnostic performs no searches and changes neither qualification nor budgets.

The extraction scheduler's `soft_checkpoint_calls` and `safety_max_calls` govern its checkpoint and hard-call behavior. `llm.max_chunks` is a legacy setting and does not by itself express the current hard limit. To preview settings without collecting data, use `scripts/collect.py --print-config-only` with your task inputs, or `data-collection-workflow collect --config <path> --dry-run` for a configuration file.

In evidence-mode iterative search, follow-up selection balances attempts across official reports, databases, literature, and media before preferring less-tried retrieval directions within each family: named months, historical archives, data downloads, and general searches. Official, database, and literature queries win ties in family effort; existing query order breaks remaining ties. While those queries remain available, a continuation batch uses at most one media slot. This prevents repeated broad paraphrases from always preceding available month, archive, or media leads. Attempts measure search effort, not verified source coverage; task-fit checks, channel restrictions, query limits, and stopping rules still apply. Adaptive settings preserve the capacity already reserved for historical indexes.

Generic news-reporting leads supplement both iterative and one-shot evidence discovery. One-shot selection gives each available official/database/literature family its first slot, then gives media an opportunity before selecting additional queries in those families; its existing limit of three media queries remains in force. After the available family floor has executed, an untried media search can share the next temporal probe slot. That probe counts toward both the media batch limit and the existing four-probe session allowance. These opportunities do not impose a target media proportion, change provider search modes, or grant the returned pages official status; fetched evidence still passes the existing identity, screening, and qualification checks.

Temporal discovery uses at most three representative months across any valid task window, without creating separate monthly runs. Initial follow-up batches can use one existing query slot for a bounded date-gap probe. Across initial discovery and recovery, at most four such probes may be dispatched per session; completed and failed attempts retain their accounting in the existing operation ledger, including interruptions before a graph checkpoint. A budget denial spends no probe allowance and can reopen after an explicit budget amendment. Recovery gives a novel temporal probe priority among searches, after local parsing, extraction, and acquisition work. Provider stops, shared budgets, historical-index reservations, disabled channels, and the strict recovery round limit remain in force.

Accepted search candidates can suggest probe dates from publication metadata before collection relevance has been confirmed. Explicit exclusions and prior registry routing still apply, and these raw hints do not add points to the stricter exported timeline. Source-supported observation dates and candidate publication dates retain separate meanings. Date terms are retrieval hints, not provider publication-date filters or claims that a report exists for every interval. Once ordinary discovery stalls, only the remaining bounded probes may extend it; unsuccessful probes do not create an unlimited retry loop.

## Sources and user-provided fixtures

Evidence-mode resource discovery follows explicit next-page links on fetched, task-matching report indexes. Both anchor links and HTML `link rel="next"` declarations are supported. The next URL must preserve the origin, report-series path, and non-pagination query filters; the workflow does not invent page URLs. Pagination retains the series depth, while existing resource and acquisition limits bound the number of pages.

Observed previous/next report versions also retain the series depth. A version link must name a report or epidemiological summary, or occur in an explicit previous/next-report attribution sentence, with disease scope supported by its fetched label, sentence, or report heading. Date-only anchors can therefore lead to an explicitly referenced earlier summary. HTML `link rel="prev"` and `link rel="next"` declarations retain their actual titles and provenance, including when their scope is insufficient for fetching. Version URLs must preserve the origin and non-pagination query filters; no sibling URLs, report dates, or reporting cadence are inferred. Seen URLs, shared acquisition budgets, resource limits (including zero), and source exclusions still apply. These links identify retrieval candidates and grant neither official trust nor factual or reporting-period evidence.

Report links found beyond the acquisition depth remain in the source registry as deferred candidates with `resource_depth_limit`, their parent source, and link provenance. They are not fetched or treated as evidence. Explicit resource-expansion limits, including zero, still apply; links outside those limits remain in the parsed page's outbound-link metadata.

The default template has no study-specific source overlay or allowlist. Supply optional source policies explicitly.

Offline fixture support accepts your own search results, content map, and review decisions. `init-config` does not select fixture files from the disease name. Supply paths explicitly:

| `init-config` argument | Configuration field |
| --- | --- |
| `--search-fixture-path` | `source_search.fixture_path` |
| `--content-fixture-map-path` | `content_fetch.content_fixture_map_path` |
| `--review-decisions-path` | `human_review.decisions_path` |

Use `--mode fixture-search` with your search fixture and content map to generate a configuration for local fixture collection. The review-decision file is optional. Replace the task, date, and file placeholders before running:

```bash
data-collection-workflow init-config --disease "YOUR_DISEASE" --location "YOUR_LOCATION" --start-date "YYYY-MM-DD" --end-date "YYYY-MM-DD" --mode fixture-search --search-fixture-path "PATH_TO_SEARCH_FIXTURE.json" --content-fixture-map-path "PATH_TO_CONTENT_MAP.json" --output configs/my_task.jsonc
```

Review the generated configuration, then use `validate-config` and `collect` with `--config configs/my_task.jsonc`. Live search and model-assisted stages require their corresponding credentials when enabled.

## Reporting periods and task answers

`final_report.html` separates evidence support from the period covered by a figure. A report does not need to be published on the query's end date: a later annual report can support the earlier year's total. Qualified counts from irregular reporting intervals also appear as supported results, with their own dates and source references. The report distinguishes a full-query-period total, a completed-outbreak total, the latest available cumulative count among processed qualified evidence, and a value for a particular reporting interval. A completed-outbreak result requires an explicit health-authority closure declaration linked to the count in verified local source text; a title, a predicted or denied closure, or the absence of later reports is insufficient.

The report's final section lists configurable task, model, search, acquisition, extraction, validation, recovery, review, budget, output, and execution settings. Each row separates the configured value from the recorded effective value and documents its source. Current defaults are reference information, not proof that an older run used them. Historical runs without a saved value show `Not recorded`; explicit `false`, `0`, and `null` remain distinct. The settings snapshot is captured during execution and is preserved when the report is rebuilt. Credentials are redacted, and report rebuilding does not read the current `.env` file to fill gaps in an older session.

An outbreak ending before the query end can therefore provide its final outbreak count without being relabeled as a complete calendar-year total. Month-level evidence retains month precision rather than acquiring an invented day. Dates outside the supported interval remain unknown, not zero. Overlapping cumulative snapshots and separate outbreaks are not added together. Candidates remain unverified leads, and conflicting qualified figures for the same period remain visible for review.

In `task_result.json`, each measure retains `status`, `value`, and `sources` for its full-query-period answer. The additive `supported_results` list records each usable scoped result with `result_kind` (`full_period_total`, `completed_outbreak_total`, `latest_available`, or `reported_period`), dates, numeric qualifier, source locators, and a boundary note. `selected_result` supplies the headline and table when a unique scoped result is available; `scoped_result_conflict` flags incompatible values for the same selected scope and interval. A legacy full-period `status: "unconfirmed"` can coexist with a supported partial-period result. The readable report displays that scoped result directly.

A verified outbreak closure event can place a completed-outbreak total within the query even when the statistical period is unresolved. Such results use `temporal_basis: "outbreak_closure_event"` and a separate `closure_date`; missing `period_start`, `period_end`, and `as_of_date` remain null. A relative closure date derived from a source's publication date retains its rule and quoted anchors in `closure_date_source.derivation`. The report labels this as a source-specific event-date inference, not a statistical cutoff. Sources may disagree about the declaration date. An unknown outbreak start year is not filled from the publication year or task window, and the total may include cases before the query. This event anchor does not establish full-period coverage or an annual total.

When an explicitly linked recovered record qualifies the same source claim, its original candidate remains in the raw data. Its JSON lead is marked `superseded_by_qualified_record_ids`; the readable report omits that duplicate unverified lead only when the record lineage, source URL, metric value, and numeric qualifier match the supported result.

## Recovery and review

Resume a session with the same code, resources, task, and configuration. Sessions created by the interactive entry have a saved generated configuration. For configured runs, retain the original configuration file. Set `output.session_id` before the first run; `--resume-session` verifies that ID but does not choose the output directory. If the original run used a `--session-id` override instead, repeat the same override together with `--resume-session`.

`--budget-amendment <file>` supplies an amendment with an identifier, reason, and increased cumulative operation limits. `--provider-resume <file>` supplies the provider, stopped-event identifier, and reason after an account limitation is resolved. Resume does not reset past usage or automatically clear provider stops.

`review-summary` displays pending records. A decision file and the apply-review option are inputs to a run. Changing an existing session's configuration to add decisions changes its fingerprint; there is currently no separate command to apply decisions directly to a completed evidence-mode session. Evidence support is still required for the final qualified data.
