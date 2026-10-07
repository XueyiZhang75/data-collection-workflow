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

The extraction scheduler's `soft_checkpoint_calls` and `safety_max_calls` govern its checkpoint and hard-call behavior. `llm.max_chunks` is a legacy setting and does not by itself express the current hard limit. Review the resolved configuration with `--print-config-only` before a live run.

## Sources and user-provided fixtures

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

## Recovery and review

Resume a session with the same code, resources, task, and configuration. Sessions created by the interactive entry have a saved generated configuration. For configured runs, retain the original configuration file.

`--budget-amendment <file>` supplies an amendment with an identifier, reason, and increased cumulative operation limits. `--provider-resume <file>` supplies the provider, stopped-event identifier, and reason after an account limitation is resolved. Resume does not reset past usage or automatically clear provider stops.

`review-summary` displays pending records. A decision file and the apply-review option are inputs to a run. Changing an existing session's configuration to add decisions changes its fingerprint; there is currently no separate command to apply decisions directly to a completed evidence-mode session. Evidence support is still required for the final qualified data.
