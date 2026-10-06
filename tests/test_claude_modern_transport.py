"""Exercise modern Claude routing through real adapters with offline HTTP responses."""

import json

import pytest
from pydantic import BaseModel, ConfigDict

from data_collection_workflow import llm_clients
from data_collection_workflow.models import LLMStructuredExtractionPolicy
from data_collection_workflow.config import load_llm_structured_extraction_policy


class AvailabilityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    available: bool


@pytest.fixture
def native_transport(monkeypatch):
    import anthropic
    import httpx
    from langchain_anthropic import ChatAnthropic

    monkeypatch.delenv("PIPELINE_MODE", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MAX_TOKENS", "257")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    clients = []

    def setup(model_name, content, stop_reason="end_turn"):
        monkeypatch.setenv("LLM_MODEL", model_name)
        requests = []

        def respond(request):
            requests.append({"path": request.url.path, "body": json.loads(request.content)})
            return httpx.Response(200, json={
                "id": "msg_offline_fixture", "type": "message", "role": "assistant",
                "model": model_name, "content": content, "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 3,
                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            })

        def build(settings=None):
            settings = settings or llm_clients.get_llm_settings()
            model = ChatAnthropic(
                **llm_clients.chat_model_kwargs(settings), api_key="offline-placeholder",
                base_url="https://offline.invalid",
            )
            http_client = httpx.Client(transport=httpx.MockTransport(respond))
            clients.append(http_client)
            sdk = anthropic.Anthropic(
                api_key="offline-placeholder", base_url="https://offline.invalid",
                max_retries=0, http_client=http_client,
            )
            monkeypatch.setitem(model.__dict__, "_client", sdk)
            return model

        monkeypatch.setattr(llm_clients, "build_chat_model", build)
        return requests

    yield setup
    for client in clients:
        client.close()


def test_sonnet5_tool_request_omits_temperature_and_preserves_output_cap(native_transport):
    requests = native_transport("claude-sonnet-5", [{
        "type": "tool_use", "id": "tool_fixture", "name": "AvailabilityCheck",
        "input": {"available": True},
    }], stop_reason="tool_use")

    result = llm_clients.run_pydantic_structured_llm(
        "Check availability.", "Return available true.", AvailabilityCheck,
    )

    assert result["available"] is True
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/messages"
    body = requests[0]["body"]
    assert "temperature" not in body
    assert body["max_tokens"] == 257
    assert body["tool_choice"] == {"type": "tool", "name": "AvailabilityCheck"}


@pytest.mark.parametrize("model_name", ["claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1"])
def test_newer_claude_small_schema_uses_native_json_without_forced_tools(native_transport, model_name):
    requests = native_transport(model_name, [{"type": "text", "text": '{"available":true}'}])

    result = llm_clients.run_pydantic_structured_llm(
        "Check availability.", "Return available true.", AvailabilityCheck,
    )

    assert result["available"] is True
    assert len(requests) == 1
    body = requests[0]["body"]
    assert "temperature" not in body
    assert body["max_tokens"] == 257
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body.get("tool_choice", {}).get("type") not in {"tool", "any"}


def test_opus55_large_extraction_preserves_records_and_ignores_thinking(native_transport):
    requests = native_transport("claude-opus-5-5", [
        {"type": "thinking", "thinking": "Internal reasoning; not extraction JSON.",
         "signature": "offline-signature"},
        {"type": "text", "text": json.dumps({
            "records": [{"disease": "Measles", "country": "Canada",
                         "cases_unspecified": 7, "evidence_quote": "Canada reported 7 measles cases."}],
            "chunk_is_relevant": True,
        })},
    ])
    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())

    result = llm_clients.extract_chunk_with_llm(
        {"chunk_id": "offline-chunk", "text": "Canada reported 7 measles cases."}, policy,
    )

    assert len(requests) == 1
    assert len(result.records) == 1
    assert result.records[0].cases_unspecified == 7
    assert result.records[0].evidence_quote == "Canada reported 7 measles cases."
    body = requests[0]["body"]
    assert "temperature" not in body
    assert body["max_tokens"] == 257
    assert "output_config" not in body
    assert body.get("tool_choice", {}).get("type") not in {"tool", "any"}
    assert "field_provenance_json" in json.dumps(body)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_provider_refusal_or_truncation_never_becomes_success_or_second_call(native_transport, stop_reason):
    requests = native_transport(
        "claude-opus-5-5", [{"type": "text", "text": '{"available":true}'}],
        stop_reason=stop_reason,
    )

    with pytest.raises(ValueError):
        llm_clients.run_pydantic_structured_llm(
            "Check availability.", "Return available true.", AvailabilityCheck,
        )

    assert len(requests) == 1


@pytest.mark.parametrize("text", ['not a JSON response', '{"records":"not a list"}', '{}', '{"unrelated":true}'])
def test_invalid_prompt_json_is_failed_in_ledger(native_transport, monkeypatch, tmp_path, text):
    from data_collection_workflow.session_runtime import RunContext

    requests = native_transport("claude-opus-5-5", [{"type": "text", "text": text}])
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())

    with context.activate(), pytest.raises(ValueError):
        llm_clients.extract_chunk_with_llm(
            {"chunk_id": "invalid-response", "text": "Canada reported 7 measles cases."}, policy,
        )

    assert len(requests) == 1
    snapshot = context.ledger.snapshot()
    assert snapshot["operations"] == {"failed": 1}
    assert snapshot["used"]["extraction"] == 1
    assert snapshot["token_usage"]["total_tokens"] == 13
