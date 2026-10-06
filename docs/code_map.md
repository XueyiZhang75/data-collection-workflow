# Code map

The Python package is `data_collection_workflow`. `scripts/collect.py` gathers task inputs and creates a configuration; `scripts/run_workflow.py` executes it. The installed `data-collection-workflow` command provides collection, configuration validation, result inspection, review summaries, and export.

## Execution

`graph.py` builds the LangGraph state machine around `state.py` and the data models in `models.py`:

1. Task intake, disease intelligence, schema, and source/query planning.
2. Source discovery, deduplication, identity checks, screening, and critique.
3. Document acquisition, parsing, quality checks, and evidence chunking.
4. Structured extraction, schema repair, normalization, linking, and consistency checks.
5. Quality gates, optional human review, and final data/report export.

Evidence mode adds runtime readiness and a recovery controller after the quality gates. A recovery action returns through schema repair, normalization, linking, consistency checks, and quality gates. Recovery stops when no useful permitted action remains, progress stops, or a budget or provider limit prevents continuation.

| Module | Responsibility |
| --- | --- |
| `runtime_profile.py` | Configuration defaults, overrides, and environment mapping |
| `session_runtime.py` | Session identity, fingerprints, operation ledger, caches, and checkpoints |
| `document_acquisition.py` | HTTP, browser, PDF, OCR, and source artifacts |
| `evidence_chunking.py` | Chunks and locators for acquired evidence |
| `evidence_qualification.py` | Field-level support and task compatibility |
| `workflow_recovery.py` | Gap identification, action planning, and bounded recovery |
| `evidence_products.py` | Case, aggregate, context, and candidate products |
| `result_manifest.py` | Consistent output counts, coverage, and run status |
| `provider_failures.py` | Provider stop classification and explicit recovery |
| `collection_readiness.py` | Run-output completeness and acquisition status |
| `collection_diagnostics.py` | Collection diagnostics |
| `case_field_recovery.py` | Evidence-backed case-field recovery |

## Agents and prompts

Source-planning and source-critic prompt resources are in `resources/source_planning_agent_prompt.json` and `resources/source_critic_agent_prompt.json`. Their agents read those resources. Source identity and iterative discovery prompts are defined with their agents in `agents/`. Disease-intelligence and credibility prompts live with their corresponding task/source code. Extraction instructions combine `resources/llm_structured_extraction_policy.json` with task, schema, and source evidence in `llm_clients.py`.

Disease profiles and schemas keep their disease names because their content is disease-specific. Synthetic fixtures retain their task identity and synthetic-data notices. The live task and its evidence determine which observations qualify.

## Optional interfaces

Langflow components call the workflow's local API. Studio exposes the LangGraph graph, and the dashboard displays run events. These interfaces are separate entry paths; the terminal runner is the complete evidence-mode orchestration path. Static visualizations currently describe the standard graph, so use `graph.py` and the result manifest to inspect evidence-mode recovery.
