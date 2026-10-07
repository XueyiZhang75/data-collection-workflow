# Configuration

Use `scripts/collect.py` for interactive task entry, or edit `configs/workflow.jsonc` and run `scripts/run_workflow.py --config <path>`. JSONC permits comments; use valid JSON syntax without trailing commas. Configuration values merge with built-in defaults, so explicitly set optional switches that matter to your task.

## Task and modes

Fill in `structured_task.disease`, `location`, `start_date`, and `end_date`. The target fields and source preferences are configurable. `user_request` can add context to these explicit fields; it does not replace them. Missing task fields stop collection before provider calls. The blank template does not select a case study for you.

`pipeline_mode` selects execution behavior:

- `standard`: collection and validation chain.
- `evidence`: field evidence qualification, local acquisition readiness, persisted budgets, recovery, and checkpoint resume.

`workflow.collection_mode` is a separate setting controlling source and validation handling (`standard`, `masked_validation`, or `direct_collection`). A pipeline mode is not a source-holdout policy.

The interactive entry accepts `--pipeline-mode`; the configured runner and `collect` CLI subcommand read the mode from the configuration file. Their flags are not interchangeable. Use `--help` on the entry you are running.

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

Each timeline lists known dates, separate counts of unknown dates and sources dated only outside the window, and up to 24 gaps between known points, including the window boundaries. Gaps carry stable identifiers, their date basis, inclusive endpoints, and day counts; larger gaps appear first, and omitted gaps are counted. These are discovery opportunities, not missing reports, zero cases, or a completeness measure. An annual aggregate can satisfy observation-period coverage while leaving the report-date inventory empty. The diagnostic uses exact task endpoints for daily through multi-year windows, handles leap days, and does not create calendar bins or assume weekly publication. Invalid or reversed windows are marked `not_evaluable`. It performs no additional searches and changes neither qualification nor budgets.

The extraction scheduler's `soft_checkpoint_calls` and `safety_max_calls` govern its checkpoint and hard-call behavior. `llm.max_chunks` is a legacy setting and does not by itself express the current hard limit. Review the resolved configuration with `--print-config-only` before a live run.

In evidence-mode iterative search, follow-up selection first balances the existing source families, then prefers less-tried retrieval directions within each family: named months, historical archives, data downloads, and general searches. Existing query order breaks remaining ties. This prevents repeated broad paraphrases from always preceding available month or archive leads. Attempts measure search effort, not verified source coverage; task-fit checks, channel restrictions, query limits, and stopping rules still apply. Adaptive settings preserve the capacity already reserved for historical indexes.

## Sources and examples

Evidence-mode resource discovery follows explicit next-page links on fetched, task-matching report indexes. Both anchor links and HTML `link rel="next"` declarations are supported. The next URL must preserve the origin, report-series path, and non-pagination query filters; the workflow does not invent page URLs. Pagination retains the series depth, while existing resource and acquisition limits bound the number of pages.

Observed previous/next report versions also retain the series depth. A version link must name a report or epidemiological summary, or occur in an explicit previous/next-report attribution sentence, with disease scope supported by its fetched label, sentence, or report heading. Date-only anchors can therefore lead to an explicitly referenced earlier summary. HTML `link rel="prev"` and `link rel="next"` declarations retain their actual titles and provenance, including when their scope is insufficient for fetching. Version URLs must preserve the origin and non-pagination query filters; no sibling URLs, report dates, or reporting cadence are inferred. Seen URLs, shared acquisition budgets, resource limits (including zero), and source exclusions still apply. These links identify retrieval candidates and grant neither official trust nor factual or reporting-period evidence.

Report links found beyond the acquisition depth remain in the source registry as deferred candidates with `resource_depth_limit`, their parent source, and link provenance. They are not fetched or treated as evidence. Explicit resource-expansion limits, including zero, still apply; links outside those limits remain in the parsed page's outbound-link metadata.

The default template has no study-specific source overlay or allowlist. Supply optional source policies explicitly. Example configurations under `configs/examples/covid19/` and `configs/examples/dengue/` describe their actual fixture tasks. Files prefixed `offline_` use synthetic local resources. Other example configurations may enable live search, fetch, or inherited model stages and require corresponding credentials.

## Recovery and review

Resume a session with the same code, resources, task, and configuration. Sessions created by the interactive entry have a saved generated configuration. For configured runs, retain the original configuration file.

`--budget-amendment <file>` supplies an amendment with an identifier, reason, and increased cumulative operation limits. `--provider-resume <file>` supplies the provider, stopped-event identifier, and reason after an account limitation is resolved. Resume does not reset past usage or automatically clear provider stops.

`review-summary` displays pending records. A decision file and the apply-review option are inputs to a run. Changing an existing session's configuration to add decisions changes its fingerprint; there is currently no separate command to apply decisions directly to a completed evidence-mode session. Evidence support is still required for the final qualified data.
