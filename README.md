# Data Collection Workflow

Collect public-health information for a disease, location, and time range. The workflow discovers sources, retrieves documents, extracts structured observations, checks their supporting evidence, and exports data with provenance.

Two pipeline modes are available: `standard` runs the collection and validation chain; `evidence` adds field-level evidence qualification, browser and OCR acquisition, a budgeted recovery loop, and session checkpoints. Choose `evidence` for that complete collection path.

## Install

Use Python 3.11 or newer.

On Windows, use a short ASCII checkout path such as `C:/Projects/data-collection-workflow` to avoid path and encoding limits in some packaging and OCR tools.

Create a virtual environment from this repository's root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in bash, then install:

```bash
python -m pip install -e .
```

The command-line runner currently uses the repository's `scripts/` and `configs/` directories, so keep the source checkout available.

For the evidence pipeline, also prepare Chromium and Tesseract as described in [Runtime setup](docs/runtime.md). The workflow checks these local tools before sending model or search requests.

## Configure credentials

Copy `.env.example` to `.env` in this checkout. Add your own `TAVILY_API_KEY` and either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Choose a model available to that account. Live searches and model calls use those accounts and may incur provider charges.

Collection entry points read the checkout's `.env`; existing shell variables take precedence. Keep credentials out of task configuration files.

| Provider | Credential | Model selection |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | Set `LLM_MODEL` to an Anthropic model ID available to your account |
| `openai` | `OPENAI_API_KEY` | Set `LLM_MODEL` to an OpenAI model ID available to your account |

For example, fill these values in `.env` for OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=YOUR_OPENAI_MODEL_ID
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

For Anthropic, use `LLM_PROVIDER=anthropic`, your Anthropic model ID, and `ANTHROPIC_API_KEY`. Tavily is the separate search service. There is no fixed model: choose one in `.env`, the task configuration, or `--model`. When model-assisted stages are enabled, collection stops if no model is selected.

## Collect your own task

Start an interactive task and enter the requested task fields and any missing model selection:

```bash
python scripts/collect.py --pipeline-mode evidence --no-dashboard
```

Or supply the task explicitly. Replace the disease, location, date, and model placeholders with your task inputs:

```bash
python scripts/collect.py --pipeline-mode evidence --disease "YOUR_DISEASE" --location "YOUR_LOCATION" --start-date "YYYY-MM-DD" --end-date "YYYY-MM-DD" --provider anthropic --model "YOUR_MODEL_ID" --session-id my_collection --no-dashboard
```

Add `--print-config-only` to inspect the resolved settings without starting collection. `--quick-test-mode` reduces the live-run budget; it is not an offline mode.

For a configuration-driven run, fill in the task in `configs/workflow.jsonc`, set `pipeline_mode` to `evidence`, and use:

```bash
data-collection-workflow validate-config --config configs/workflow.jsonc
python scripts/run_workflow.py --config configs/workflow.jsonc
```

The template has no default disease, location, or study-specific source list. See [Configuration](docs/configuration.md) for modes, budgets, and optional source policies.

## Read the results

Each session is saved under `outputs/sessions/<session_id>/`.

Generated report text, status messages, and interface labels use English. Source quotations, original source names, and localized search terms retain their original language so the evidence and retrieval process remain verifiable.

| File | Contents |
| --- | --- |
| `task_result.md` | Task answers in the evidence pipeline |
| `result_manifest.json` | Evidence-pipeline data availability, coverage, budget, and stopping reason |
| `collection/final_dataset.csv` | Qualified case and aggregate observations |
| `collection/final_case_dataset.csv` | Qualified individual-case observations |
| `collection/aggregate_dataset.csv` | Qualified aggregate observations |
| `collection/candidate_records.csv` | Records awaiting sufficient evidence or review |
| `collection/context_records.csv` | Contextual information |
| `collection/source_processing_status.csv` | Source acquisition and evidence-contribution status |

JSON counterparts are provided. Standard mode also writes a workflow summary, collection outputs, and reports; the evidence-specific manifest and product split are produced in evidence mode.

An observation row is not necessarily one patient. A completed execution can still have incomplete coverage or no qualified observations. Use the manifest's separate status fields when assessing a run.

```bash
data-collection-workflow review-summary --session-dir outputs/sessions/my_collection
data-collection-workflow export --session-dir outputs/sessions/my_collection --output-dir outputs/exports/my_collection --format both
```

`review-summary` displays the review queue. Review decisions can be supplied to a configured run; it does not itself apply decisions to a completed session.

## Resume a session

For a session created by the interactive entry:

```bash
python scripts/collect.py --resume-session my_collection --no-dashboard
```

For a session created with a configuration file:

```bash
python scripts/run_workflow.py --config configs/my_task.jsonc --resume-session my_collection
```

Resume requires the original task, configuration, code, and resources. It preserves used budgets and cached responses. Budget increases and provider-account recovery use explicit amendment files; see [Configuration](docs/configuration.md). Use a new session after changing source code or task settings.

## Project layout

```text
src/data_collection_workflow/  Workflow nodes, agents, evidence, runtime, and reports
  agents/                     Source planning, identity, discovery, and critique
  nodes/                      Collection and validation steps
  resources/                  Policies, prompts, disease profiles, and fixtures
scripts/                      Collection, configured execution, and optional interfaces
configs/                      Blank task template
integrations/langflow/        Optional visual interface components
tests/                        Software behavior and integration tests
docs/                         Configuration and runtime instructions
```

See [Code map](docs/code_map.md) for the execution chain and prompt locations. Langflow, Studio, and the dashboard are optional. Collection can run entirely from the terminal.

## Tests

```bash
python -m pytest
```

The suite includes local browser and OCR integration tests; prepare the tools in [Runtime setup](docs/runtime.md) before running the complete suite. Provider responses are mocked in software tests. Synthetic fixtures and passing tests do not establish factual accuracy on a live collection task.
