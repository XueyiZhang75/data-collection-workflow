"""Offline contracts for malformed live tool envelopes and safe local repair."""

import json

import pytest

from data_collection_workflow import llm_clients
from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.models import LLMStructuredExtractionPolicy, SearchRefinementDecision
from data_collection_workflow.session_runtime import RunContext


@pytest.fixture
def model_transport(monkeypatch):
    import anthropic
    import httpx
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "auto")
    monkeypatch.setenv("LLM_THINKING", "auto")
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    clients = []

    def setup(content, stop_reason="tool_use", method="function_calling"):
        monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", method)
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "msg_contract_fixture", "type": "message", "role": "assistant",
                "model": "claude-sonnet-5", "content": content,
                "stop_reason": stop_reason, "stop_sequence": None,
                "usage": {"input_tokens": 19, "output_tokens": 7,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            })

        def build(settings=None):
            model = ChatAnthropic(
                **llm_clients.chat_model_kwargs(settings or llm_clients.get_llm_settings()),
                api_key="offline-placeholder", base_url="https://offline.invalid",
            )
            http_client = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(http_client)
            monkeypatch.setitem(model.__dict__, "_client", anthropic.Anthropic(
                api_key="offline-placeholder", base_url="https://offline.invalid",
                max_retries=0, http_client=http_client,
            ))
            return model

        monkeypatch.setattr(llm_clients, "build_chat_model", build)
        return requests

    yield setup
    for client in clients:
        client.close()


def tool(name, args, ident="fixture_tool"):
    return {"type": "tool_use", "id": ident, "name": name, "input": args}


def extract():
    return llm_clients.extract_chunk_with_llm(
        {"chunk_id": "fixture-chunk", "text": "Source evidence is supplied in the response fixture."},
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),
    )


@pytest.mark.parametrize("disease,country,count", [("Dengue", "Brazil", 12), ("Measles", "Canada", 7)])
def test_complete_extraction_envelope_serialized_in_records_is_repaired_once(model_transport, tmp_path, disease, country, count):
    quote = f"{country} reported {count} {disease} cases."
    inner = {"chunk_is_relevant": True, "extraction_notes": "Source-bound observation.",
             "records": [{"disease": disease, "country": country,
                          "cases_confirmed": count, "evidence_quote": quote}]}
    requests = model_transport([tool("LLMExtractionOutput", {"records": json.dumps(inner)})])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})

    with context.activate():
        first = extract()
        cached = extract()

    assert first.records[0].disease == cached.records[0].disease == disease
    assert first.records[0].country == country
    assert first.records[0].cases_confirmed == count
    assert first.records[0].evidence_quote == quote
    assert len(requests) == 1
    assert context.ledger.snapshot()["used"]["extraction"] == 1
    assert context.ledger.snapshot()["token_usage"]["total_tokens"] == 26


def test_sonnet5_complex_schema_uses_prompt_json_before_first_dispatch(model_transport, tmp_path):
    requests = model_transport(
        [{"type": "text", "text": '{"records":[],"chunk_is_relevant":false}'}],
        stop_reason="end_turn", method="auto",
    )
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate():
        assert extract().records == []
    assert len(requests) == 1
    assert "tool_choice" not in requests[0]
    assert "tools" not in requests[0]
    assert "records" in requests[0]["system"]
    assert requests[0]["max_tokens"] == 4096


@pytest.mark.parametrize("args", [
    {"records": json.dumps({"records": [], "chunk_is_relevant": True}), "chunk_is_relevant": False},
    {"records": '{"records":[]}{"records":[]}'},
    {"records": '{"records":['},
    {"records": json.dumps({"records": "[]"})},
])
def test_ambiguous_partial_conflicting_or_recursive_repairs_fail_once_with_raw_response(model_transport, tmp_path, args):
    requests = model_transport([tool("LLMExtractionOutput", args)])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate(), pytest.raises(ValueError) as error:
        extract()
    assert len(requests) == 1
    assert context.ledger.snapshot()["operations"] == {"failed": 1}
    assert context.ledger.snapshot()["used"]["extraction"] == 1
    assert context.ledger.snapshot()["token_usage"]["total_tokens"] == 26
    raw = error.value.llm_raw_response
    assert raw["tool_calls"][0]["args"] == args
    json.dumps(raw)
    assert len(str(error.value)) < 500


@pytest.mark.parametrize("args", [
    {"iteration_index": 2, "coverage_assessment": 'More regional evidence needed.",\\n "decision": "continue_search",\\n "next_query_batch": []\\n}\\n'},
    {"iteration_index": 2, "coverage_assessment": "More regional evidence needed."},
    {"iteration_index": 2, "decision": "stop_sufficient",
     "coverage_assessment": 'Text.", "decision": "continue_search", "next_query_batch": []}'},
])
def test_missing_or_embedded_control_decision_cannot_inflate_stop_default(model_transport, tmp_path, args):
    requests = model_transport([tool("SearchRefinementDecision", args)])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate(), pytest.raises(ValueError) as error:
        llm_clients.run_pydantic_structured_llm("Assess coverage.", "Continue if evidence is missing.", SearchRefinementDecision)
    assert len(requests) == 1
    assert context.ledger.snapshot()["operations"] == {"failed": 1}
    assert error.value.llm_raw_response["tool_calls"][0]["args"] == args


def test_multiple_matching_tool_envelopes_are_not_silently_reduced_to_first(model_transport, tmp_path):
    requests = model_transport([
        tool("LLMExtractionOutput", {"records": [], "chunk_is_relevant": False}, "first"),
        tool("LLMExtractionOutput", {"records": [], "chunk_is_relevant": True}, "second"),
    ])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate(), pytest.raises(ValueError):
        extract()
    assert len(requests) == 1
    assert context.ledger.snapshot()["operations"] == {"failed": 1}


@pytest.mark.parametrize("answer", [
    '{"records":[],"chunk_is_relevant":false} {"records":[],"chunk_is_relevant":true}',
    '{"records":[],"chunk_is_relevant":false,"chunk_is_relevant":true}',
])
def test_ambiguous_prompt_json_objects_or_duplicate_keys_fail(model_transport, tmp_path, answer):
    requests = model_transport([{"type": "text", "text": answer}], stop_reason="end_turn", method="auto")
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate(), pytest.raises(ValueError):
        extract()
    assert len(requests) == 1
    assert context.ledger.snapshot()["operations"] == {"failed": 1}


def test_valid_text_field_containing_json_is_not_rewritten_as_an_envelope(model_transport, tmp_path):
    from typing import Literal
    from pydantic import BaseModel

    class AnnotatedDecision(BaseModel):
        decision: Literal["continue", "stop"]
        notes: str | None = None

    notes = json.dumps({"notes": "A quoted JSON example.", "decision": "stop"})
    requests = model_transport([tool("AnnotatedDecision", {"decision": "continue", "notes": notes})])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate():
        result = llm_clients.run_pydantic_structured_llm("Keep source text verbatim.", "Assess.", AnnotatedDecision)
    assert result["decision"] == "continue"
    assert result["notes"] == notes
    assert len(requests) == 1


def test_batched_extraction_maps_failure_to_actual_source_row_ids(model_transport, tmp_path):
    requests = model_transport([tool("LLMExtractionOutput", {"records": "malformed"})])
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    chunk = {"chunk_id": "batch-fixture", "text": "Two source rows.", "chunk_kind": "metric_row_batch",
             "metric_rows": [{"chunk_id": "row-one", "text": "Dengue cases in Brazil."},
                             {"chunk_id": "row-two", "text": "Measles cases in Canada."}],
             "row_chunk_id_by_id": {"first": "row-one", "second": "row-two"}}
    with context.activate(), pytest.raises(ValueError):
        llm_clients.extract_chunk_with_llm(chunk, LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()))
    assert {"row-one", "row-two"} <= set(context.ledger.failed_chunk_ids(retryable_only=True))
    assert len(requests) == 1
    assert context.ledger.snapshot()["used"]["extraction"] == 1


def test_sonnet5_refinement_schema_uses_json_prompt_and_explicit_decision(model_transport, tmp_path):
    answer = {"iteration_index": 2, "decision": "continue_search", "coverage_assessment": "A source family remains missing.", "next_query_batch": []}
    requests = model_transport([{"type": "text", "text": json.dumps(answer)}], stop_reason="end_turn", method="auto")
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    with context.activate():
        result = llm_clients.run_pydantic_structured_llm("Assess coverage.", "Use explicit evidence gaps.", SearchRefinementDecision)
    assert result["decision"] == "continue_search"
    assert result["_structured_output_mode"] == "json_prompt"
    assert "tools" not in requests[0]
    assert "tool_choice" not in requests[0]
    assert len(requests) == 1
