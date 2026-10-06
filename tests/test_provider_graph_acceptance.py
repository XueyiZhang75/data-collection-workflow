"""Real LangGraph acquisition-to-export acceptance with new synthetic originals.

Only task/discovery planning, HTTP and model endpoints are replaced. Acquisition,
parsing, chunking, extraction, schema, normalization, consistency, quality,
recovery, ledger/checkpoint persistence and final exports execute real nodes.
"""
from collections import Counter
from copy import deepcopy
import json
import re
import socket

import pytest
import requests

from data_collection_workflow import llm_clients
from data_collection_workflow import graph as graph_module
from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.models import LLMExtractionOutput
from data_collection_workflow.session_runtime import RunContext, checkpoint_graph
from test_evidence_resource_discovery import _env, _source
from test_provider_account_guard import ProviderError


@pytest.mark.parametrize("disease,country,regions", [
    ("measles", "Canada", ("Quebec", "Ontario")),
    ("dengue", "Brazil", ("Para", "Pernambuco")),
])
def test_provider_account_halt_completes_real_graph_and_exports_partial_evidence(tmp_path, monkeypatch, disease, country, regions):
    _env(monkeypatch)
    for key, value in {
        "ENABLE_LLM_EXTRACTION": "true", "LLM_FALLBACK_TO_RULE_BASED": "true",
        "LLM_PROVIDER": "anthropic", "LLM_MODEL": "offline-synthetic-reader",
        "ENABLE_LIVE_SEARCH": "false", "ENABLE_LLM_DISEASE_INTELLIGENCE": "false",
        "ENABLE_HUMAN_REVIEW": "false", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
        "LLM_EXTRACTION_SCHEDULER_MODE": "quality_adaptive",
        "LLM_EXTRACTION_MAX_CONCURRENCY": "1", "LLM_EXTRACTION_SAFETY_MAX_CALLS": "100",
    }.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("External network is forbidden"))

    north, south = regions
    source_a = "https://synthetic-a.invalid/report"
    source_b = "https://synthetic-b.invalid/report"
    deferred_url = "https://synthetic-c.invalid/report"
    paragraphs = [
        f"During 2025, {north}, {country} reported 26 confirmed {disease} cases.",
        f"During 2025, {south}, {country} reported 9 confirmed {disease} cases.",
        f"During 2025, {north}, {country} reported at least 31 confirmed {disease} cases.",
        f"During 2025 in {north}, {country}, {disease} surveillance recorded symptoms: fever.",
    ]
    conflicting = f"During 2025, {north}, {country} reported 42 confirmed {disease} cases."
    pages = {
        source_a: "<main><p>" + paragraphs[0] + " " + paragraphs[2] + "</p>" + "".join(f"<h2>Surveillance note {index}</h2><p>{sentence}</p>" for index, sentence in enumerate(paragraphs[1:])) + "</main>",
        source_b: f"<main><p>{conflicting}</p></main>",
        deferred_url: f"<main><p>During 2025, {country} reported 77 confirmed {disease} cases.</p></main>",
    }
    http_calls = []

    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        def __init__(self, url): self.url = url
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield pages[self.url].encode()

    def get(url, **kwargs):
        assert url in pages
        assert kwargs.get("allow_redirects") is False
        http_calls.append(url)
        return Response(url)
    monkeypatch.setattr(requests, "get", get)

    model_calls = []
    class SourceReader:
        model_name = "offline-synthetic-reader"
        def with_structured_output(self, schema):
            assert schema is LLMExtractionOutput
            return self
        def invoke(self, messages, config=None):
            text = messages[-1]["content"].split("Evidence chunk text:", 1)[1].strip()
            model_calls.append(text)
            if len(model_calls)>1:
                raise ProviderError()
            records = []
            # Deterministic endpoint fixture reads only the actual supplied source;
            # it does not consult task history, prior sessions or expected products.
            for sentence in [*paragraphs, conflicting]:
                if sentence not in text: continue
                location = next(region for region in regions if region in sentence)
                record = {"disease": disease, "country": country, "subnational_location": location,
                    "reporting_period": "2025", "evidence_quote": sentence,
                    "field_provenance_json": {key: {"quote": sentence} for key in
                        ("disease", "country", "subnational_location", "reporting_period")}}
                match = re.search(r"(?:at least )?(\d+) confirmed", sentence)
                if match:
                    record["cases_confirmed"] = int(match.group(1))
                    record["field_provenance_json"]["cases_confirmed"] = {"quote": sentence}
                else:
                    record["symptoms"] = "fever"
                    record["field_provenance_json"]["symptoms"] = {"quote": sentence}
                records.append(record)
            return {"chunk_is_relevant": bool(records), "records": records}
    monkeypatch.setattr(llm_clients, "build_chat_model", lambda *a, **k: SourceReader())
    from data_collection_workflow.nodes import extraction
    real_builder = extraction._build_record_from_llm_output
    old_source_ordinals = []
    def observe_builder(row, chunk, index, *args, **kwargs):
        old_source_ordinals.append((chunk.get("source_id"), index))
        return real_builder(row, chunk, index, *args, **kwargs)
    monkeypatch.setattr(extraction, "_build_record_from_llm_output", observe_builder)

    # A fixed, fresh intake/discovery fixture avoids unrelated planner models.
    for name in ("task_intake_and_scope_planning", "disease_intelligence_builder", "profile_and_schema_setup",
                 "executable_source_planning", "query_strategy_builder", "source_discovery",
                 "source_dedup_and_registry", "source_screening", "source_critic_and_uncertainty_routing"):
        monkeypatch.setattr(graph_module, name, lambda state: {})

    sources = []
    for index, url in enumerate(pages, 1):
        source = _source(url, index)
        source["title"] = f"{disease} surveillance in {country} during 2025"
        sources.append(source)
    initial = {"structured_task": {"disease": disease, "location": country, "start_date": "2025-01-01", "end_date": "2025-12-31"},
        "collection_spec": {"disease": disease, "geography": country, "time_window": "2025", "start_date": "2025-01-01", "end_date": "2025-12-31", "target_population": "humans"},
        "source_registry": sources, "human_review_queue": [], "collection_trace": []}
    runtime = RunContext(tmp_path / "session", {"llm":{"provider":"anthropic"}, "pipeline_mode": "evidence", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 1},
        "budget_limits": {"source_targets": 3, "http_requests": 3, "search": 0, "search_results": 0,
                          "extraction": 100, "browser": 0, "ocr": 0}, "extraction_reserve": 0}})
    runtime.readiness = {"ready": True, "fixture": "No browser/OCR transports used"}
    updates = []
    with runtime.activate(), checkpoint_graph(runtime) as graph:
        config = {"configurable": {"thread_id": "fresh-synthetic-graph"}, "recursion_limit": 150}
        for update in graph.stream(initial, config, stream_mode="updates"):
            updates.append(update)
        saved = graph.get_state(config)
        state = dict(saved.values)
        assert not saved.next
        package = state["final_data_package"]
        export_final_data_package(package, tmp_path / "export")

    assert len(model_calls)==2
    assert len(http_calls)==3
    assert not saved.next
    ledger=runtime.ledger.snapshot()
    assert ledger['used']['extraction']==2
    assert ledger['remaining']['extraction']==98
    assert ledger['providers']['anthropic']['status']=='halted'
    assert state['recovery_stop_reason']=='provider_account_limit'
    assert not (state.get('recovery_plan') or {}).get('actions')
    assert state['extraction_attempted_chunk_ids']
    assert set(state['extraction_attempted_chunk_ids']) < {c['chunk_id'] for c in state['evidence_chunks']}
    assert package['aggregate_dataset']
    assert any(r.get('cases_confirmed')==26 for r in package['aggregate_dataset'])
    assert any(r.get('cases_confirmed')==31 for r in package['candidate_records'])
    for name in ('final_dataset','aggregate_dataset','candidate_records'):
        assert json.loads((tmp_path/'export'/(name+'.json')).read_text(encoding='utf-8'))==package[name]
    manifest=package['result_manifest']
    assert manifest['recovery_stop_reason']=='provider_account_limit'
    assert manifest['counts']['qualified_observations']==len(package['final_dataset'])
    assert manifest['counts']['candidate_records']==len(package['candidate_records'])
    assert manifest['budget']['used']==ledger['used']
    assert all(a['status'] not in {'completed','completed_empty'} for a in state.get('recovery_action_history') or [] if a['kind']=='extract' and a['target_id'] not in state['extraction_attempted_chunk_ids'])
