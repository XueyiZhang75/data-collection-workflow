# Data Collection Workflow

[![Figure 1. Public-health data collection approaches and the agentic workflow architecture.](docs/assets/figure-1.png)](docs/assets/figure-1.pdf)

Collect public-health information for a disease, location, and date range. The workflow discovers sources, retrieves documents, extracts structured observations, checks their supporting evidence, and exports data with source references.

The collection process is: **define the task → discover sources → retrieve documents → extract and check evidence → export data and reports**.

This repository contains the application, configuration templates, documentation, optional interfaces, and software tests. Sessions, downloaded documents, logs, generated reports, and research results are created locally when you run it. They belong under `outputs/`, which Git ignores. Files in `tests/` are test code and fixed test inputs, not saved test-run reports or completed collection sessions.

## Source discovery in evidence mode

For a task spanning a date range, the workflow uses representative months and bounded date-gap queries, balances search opportunities across official reports, databases, literature, and media, and follows explicitly linked report pages and previous/next versions. A separate report-date timeline helps identify further retrieval opportunities without treating gaps as missing reports or zero cases.

Live evidence discovery also looks up historical page-version metadata in the Internet Archive CDX index within the shared budget. These index records are discovery candidates, not downloaded historical page bodies or qualified observations. Set `universal.historical_discovery.enabled` to `false` in a task configuration to disable those lookups. See [Configuration](docs/configuration.md#historical-source-versions) for limits and provenance rules.

## Step 1. Download the project and install Python dependencies

Install **Git** and **Python 3.11 or newer** first. On Windows, use a short ASCII checkout path such as `C:/Projects/data-collection-workflow`; some packaging and OCR tools have trouble with long or non-ASCII paths.

Open a terminal in the folder where you want the project, then follow the commands for your shell.

**Windows PowerShell:**

```powershell
git clone https://github.com/XueyiZhang75/data-collection-workflow.git
cd data-collection-workflow
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

**macOS or Linux:**

```bash
git clone https://github.com/XueyiZhang75/data-collection-workflow.git
cd data-collection-workflow
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Keep this terminal open and run the remaining commands from the project root. If you open a new terminal, return to this directory and activate `.venv` again. Keep the source checkout: the command-line runner uses its `scripts/` and `configs/` directories.

Confirm the command is available:

```bash
data-collection-workflow --help
```

If PowerShell blocks activation, install and check the command using the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\data-collection-workflow.exe --help
```

Then use `.\.venv\Scripts\python.exe` instead of `python`, and `.\.venv\Scripts\data-collection-workflow.exe` instead of `data-collection-workflow`, throughout this tutorial.

## Step 2. Add your search and model credentials

Open the `.env` file you just copied. Fill in the key for **Tavily** and the key and model ID for **one** supported model provider.

For OpenAI:

```dotenv
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
LLM_PROVIDER=openai
LLM_MODEL=YOUR_OPENAI_MODEL_ID
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

For Anthropic:

```dotenv
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
LLM_PROVIDER=anthropic
LLM_MODEL=YOUR_ANTHROPIC_MODEL_ID
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
```

Replace the `YOUR_...` values with your own account settings. Tavily supplies live search; the selected model assists with task understanding, source planning, and extraction. There is no preselected model: use an exact model ID available to your account. Live search and model requests use those accounts and may incur provider charges.

Collection commands load `.env` automatically. Existing shell variables take precedence over `.env`; command-line provider/model arguments override task configuration, which overrides environment values. Keep API keys in `.env` or your environment, not in task JSON files. `.env` is ignored by Git.

## Step 3. Prepare the browser and OCR tools

This tutorial uses **`evidence` mode**, which provides field-level evidence checks, browser/OCR acquisition, persisted budgets, a recovery loop, and checkpoint resume. It requires Chromium and Tesseract even if a particular task eventually uses only text pages. The Python installation in Step 1 already includes PDF parsing libraries.

Install Chromium for the active Python environment:

```bash
python -m playwright install chromium
```

On a supported Linux system that needs browser system libraries, use `python -m playwright install --with-deps chromium`. See the [Playwright browser installation instructions](https://playwright.dev/python/docs/browsers).

Install **Tesseract 5** using the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html). On Windows, that guide links to the UB Mannheim installer. Install all six language/data files required by this workflow: **`eng`, `osd`, `fra`, `spa`, `por`, `chi_sim`**.

Check the installation:

```bash
tesseract --version
tesseract --list-langs
```

The version must start with 5, and the language list must contain all six names. If `tesseract` is not on your PATH, run these checks with the full executable path. For example, in PowerShell:

```powershell
& "C:/Program Files/Tesseract-OCR/tesseract.exe" --list-langs
```

If the workflow cannot find Tesseract or its language files, set these values in `.env`, using your actual installation paths:

```dotenv
TESSERACT_EXECUTABLE="C:/Program Files/Tesseract-OCR/tesseract.exe"
TESSDATA_DIR="C:/Program Files/Tesseract-OCR/tessdata"
```

See [Runtime setup](docs/runtime.md) for custom Chromium paths, OCR self-check fonts, and Windows OCR path settings. The workflow checks browser, PDF, and OCR readiness locally before sending provider requests in evidence mode.

## Step 4. Define your task and start collection

You need four task inputs and a name for this run:

| Input | Meaning | Example used below |
| --- | --- | --- |
| `--disease` | Disease or virus to collect information about | `Measles` |
| `--location` | Geographic scope | `Canada` |
| `--start-date` | Start of the requested period | `2025-01-01` |
| `--end-date` | End of the requested period | `2025-01-31` |
| `--session-id` | Local name for this run; use letters, digits, `_`, or `-` | `my_first_collection` |

Replace the example disease, location, and dates with your own task, keeping quotation marks around names that contain spaces. This command **starts collection**:

```bash
python scripts/collect.py --pipeline-mode evidence --disease "Measles" --location "Canada" --start-date "2025-01-01" --end-date "2025-01-31" --session-id my_first_collection --quick-test-mode --no-dashboard
```

The terminal prints the chosen provider and model, generated configuration path, session ID, and node progress. The workflow checks local acquisition tools and model compatibility, then proceeds through discovery, retrieval, extraction, validation, and export. Keep the terminal running until it reports completion or a stopping reason.

Alternatively, let the entry point ask for the task inputs:

```bash
python scripts/collect.py --pipeline-mode evidence --quick-test-mode --no-dashboard
```

Answer the disease, location, start/end dates, and session-name prompts. Press Enter to accept the generated session name or task description. A missing model selection is also requested in an interactive terminal.

`--quick-test-mode` uses smaller search and extraction budgets for a first run. It still makes real search and model requests. The budgets count operations; they are not a dollar spending cap. `--no-dashboard` shows progress in the terminal without opening an additional monitoring interface.

**Optional preview:** add `--print-config-only` to either command to display the task, model, and budgets without starting collection or creating a session. Remove that flag when you are ready to run. Previewing is optional; it does not validate the account or install missing browser/OCR tools.

## Step 5. Open and assess the results

For the named example, open `outputs/sessions/my_first_collection/`. For an automatically named run, use the session path printed in the terminal.

Open **`final_report.html`** in a browser. This is the single English report for the session:

1. Read **Task and conclusions** for the requested task, supported answers, their statistical periods, and linked sources. Missing evidence is identified explicitly; it is not reported as zero.
2. Browse the **Source catalogue** for all discovered sources, including sources that failed retrieval or were excluded. Search, filter, or change pages; expand a source to inspect its publisher, original URL, publication date, processing status, contribution, evidence, and unresolved issues. Publication dates and the periods covered by figures are shown separately.
3. Check **Unresolved questions**, when present, for gaps that still affect the answer.
4. Use **Data and evidence** to open the underlying CSV/JSON files and saved evidence passages.
5. Open **Run information and settings** for timing, model, and all configurable settings. The report distinguishes configured values, recorded effective values, current defaults, inactive controls, and values not recorded for this run. Credentials are redacted.

| Path inside the session | What to read it for |
| --- | --- |
| `final_report.html` | Task, conclusions, complete source catalogue, unresolved questions, data links, and run settings |
| `session_report.zip` | Portable copy of the report with its `data/` and `evidence/` files |
| `data/report_snapshot.json` | The task answers, result status, counts, and run information used in the report |
| `data/source_catalog.csv` / `.json` | Complete report source catalogue |
| `data/run_settings.csv` / `.json` | Settings inventory with recorded values and their provenance |
| `run_config.json` | Sanitized configuration snapshot for this run; resume uses the original saved configuration |
| `task_result.json` | Machine-readable task answers in evidence mode |
| `result_manifest.json` | Data availability, coverage, budget usage, and stopping reason |
| `collection/final_dataset.csv` | Retained case and aggregate observations; evidence-mode rows have passed evidence qualification |
| `collection/final_case_dataset.csv` | Qualified individual-case observations |
| `collection/aggregate_dataset.csv` | Qualified aggregate observations |
| `data/candidate_records.csv` | Evidence candidates, or standard-mode records pending review or excluded by run checks |
| `collection/context_records.csv` | Background or contextual observations |
| `collection/source_processing_status.csv` | Which sources were retrieved, processed, or contributed evidence |

The report's `data/` folder contains the CSV/JSON datasets for sharing; `collection/` retains the detailed machine outputs, compatibility views, and review exports. These views are not independent datasets. Stage diagnostics remain available locally. Normal session execution no longer generates separate `task_result.md`, `final_report.md`, `workflow_run_report*.md`, or `workflow_interpretive_report*.md` reading reports.

Standard-mode accepted observations are identified as collected observations unless they carry explicit evidence qualification. Its pending-review and excluded records remain available with their original decisions; their presence does not confirm a task total.

To share a result, send **`session_report.zip`**. Extract it before opening `final_report.html`, keeping `data/` and `evidence/` beside the HTML file. No server or network request is needed to read the report; original website links require internet access. The bundle excludes execution logs and build history. The workflow does not automatically produce a final PDF report or an epidemic-curve figure.

A run that stops during setup may not have these final files; use its terminal error and saved session status to diagnose the problem.

You can inspect the session and pending review items from the terminal:

```bash
data-collection-workflow inspect-run --session-dir outputs/sessions/my_first_collection
data-collection-workflow review-summary --session-dir outputs/sessions/my_first_collection
```

An observation row is not necessarily one patient. A completed execution can have incomplete coverage or no qualified observations. Check the manifest and the cited evidence before using the data. `review-summary` displays the review queue; it does not itself apply decisions to an existing session.

Generated reports and interface labels use English. Quotations, original source names, and localized search terms retain their source language.

To export the report, its supporting materials, collection data, and machine-readable run summary to another local directory:

```bash
data-collection-workflow export --session-dir outputs/sessions/my_first_collection --output-dir outputs/exports/my_first_collection --format both
```

The export includes `final_report.html`, `session_report.zip`, and the report's supporting `data/` and `evidence/` folders. `--format` selects the collection data export format; the portable report retains both CSV and JSON. The same report is used by command-line collection and export.

## Step 6. Run again or resume a session (optional)

For a run with the regular interactive budgets, omit `--quick-test-mode`. This command asks for your task inputs; choose a **new session ID**, such as `my_full_collection`:

```bash
python scripts/collect.py --pipeline-mode evidence --no-dashboard
```

Use a new ID whenever you change the task, configuration, model, code, or resources. Changing the model provider also requires choosing a model for that provider, for example `--provider openai --model "YOUR_OPENAI_MODEL_ID"`.

To continue an existing evidence-mode session created by `scripts/collect.py`:

```bash
python scripts/collect.py --resume-session my_first_collection --no-dashboard
```

Resume loads its saved task and configuration. It preserves used budgets, cached successful responses, and checkpoints. Keep the original code/resources and session files. Resuming does not reset exhausted budgets or clear a provider-account stop; explicit budget amendments and provider recovery are explained in [Configuration](docs/configuration.md#recovery-and-review).

For custom extraction fields, detailed budgets, or reusable task files, see the optional [configuration-file guide](docs/configuration.md#run-with-a-configuration-file-optional).

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `data-collection-workflow` is not found | Activate the project's virtual environment and rerun `python -m pip install -e .`. |
| Missing API key or missing model | Check `.env`, the matching provider key, and `LLM_MODEL`. Check whether an existing shell variable overrides the file. |
| Model preflight or account error | Use a model supported by your account and check the provider's error. When switching providers, set both `--provider` and `--model`. |
| Browser executable is missing | Run `python -m playwright install chromium` in the same virtual environment. |
| Missing OCR languages or failed OCR readiness | Check Tesseract 5, all six language files, and the paths in [Runtime setup](docs/runtime.md). |
| An existing session has a different configuration or fingerprint | Use a new session ID for the changed task, or restore the original settings and explicitly resume. |
| Run stops at its budget | Inspect `result_manifest.json`; resume with an explicit amendment if appropriate, or create a new run with reviewed limits. |
| No qualified rows | Read the task result, manifest, candidate records, and source-processing status to see whether the cause was unavailable sources, scope mismatch, missing evidence, or budget limits. |

## Optional interfaces

Terminal collection works without Langflow, Studio, or Streamlit. To add the live dashboard:

```bash
python -m pip install -e ".[visualization]"
python -m streamlit run scripts/dashboard.py -- --session-dir outputs/sessions/my_first_collection
```

Langflow and Studio have separate optional dependencies and launchers. See [Code map](docs/code_map.md) for their entry points.

## Project layout and software tests

```text
src/data_collection_workflow/  Workflow nodes, model adapters, evidence checks, runtime, and reports
  agents/                     Model-assisted source planning and assessment
  nodes/                      Collection and validation steps
  resources/                  Policies, prompts, disease knowledge, and compatibility test inputs
scripts/                      Terminal entry points and optional interfaces
configs/workflow.jsonc        Blank task configuration template
integrations/langflow/        Optional visual interface components
tests/                        Test code and fixed inputs used to check software behavior
docs/                         Configuration, runtime setup, and code map
```

`session_runtime.py` implements session storage and resumption; it is application code. The actual session state, raw documents, caches, and checkpoints are generated in each local output directory. Local task files in `configs/local/`, machine settings in `.runtime/`, `.env`, `outputs/`, and test-report/cache directories are ignored by Git.

Developers can run the software tests with:

```bash
python -m pytest
```

Prepare the browser and OCR tools before running the complete suite. Tests use temporary directories, fixed inputs, and mocked provider responses. Passing software tests does not establish factual accuracy for a live collection task.
