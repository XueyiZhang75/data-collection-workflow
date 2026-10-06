"""Offline regressions for provider token totals and persisted Claude usage."""

import json

import pytest

from data_collection_workflow.session_runtime import RunContext, provider_token_usage


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            {"input_tokens": 10, "output_tokens": 3,
             "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
            {"input_tokens": 120, "output_tokens": 3, "total_tokens": 123},
        ),
        (
            {"input_tokens": 10, "output_tokens": 3,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        ),
        (
            {"output_tokens": 3,
             "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
            {"input_tokens": None, "output_tokens": 3, "total_tokens": None},
        ),
        (
            {"input_tokens": 10,
             "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
            {"input_tokens": 120, "output_tokens": None, "total_tokens": None},
        ),
        (
            {"input_tokens": 10, "output_tokens": 3,
             "cache_read_input_tokens": None, "cache_creation_input_tokens": 20},
            {"input_tokens": None, "output_tokens": 3, "total_tokens": None},
        ),
    ],
)
def test_raw_anthropic_usage_includes_cached_input_without_inventing_counts(raw, expected):
    assert provider_token_usage({"usage": raw}) == expected


@pytest.mark.parametrize("bad_count", [-1, True, "90"])
def test_invalid_cache_counts_leave_input_and_total_unknown(bad_count):
    assert provider_token_usage({"usage": {
        "input_tokens": 10, "output_tokens": 3,
        "cache_read_input_tokens": bad_count, "cache_creation_input_tokens": 20,
    }}) == {"input_tokens": None, "output_tokens": 3, "total_tokens": None}


def test_normalized_message_usage_does_not_add_cached_input_twice():
    from langchain_core.messages import AIMessage

    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 120, "output_tokens": 3, "total_tokens": 123,
            "input_token_details": {"cache_read": 90, "cache_creation": 20},
        },
        response_metadata={"usage": {
            "input_tokens": 10, "output_tokens": 3,
            "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20,
        }},
    )
    expected = {"input_tokens": 120, "output_tokens": 3, "total_tokens": 123}
    assert provider_token_usage(message) == expected
    assert provider_token_usage(message.model_dump()) == expected


def test_absent_usage_is_unknown_not_zero():
    assert provider_token_usage({}) is None
    assert provider_token_usage({"usage": {"input_tokens": None}}) is None


def test_incomplete_cached_usage_remains_partial_in_ledger(tmp_path):
    context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
    context.call("model:fixture", {}, lambda: {"usage": {
        "input_tokens": 10, "cache_read_input_tokens": 90,
        "cache_creation_input_tokens": 20,
    }})
    usage = context.ledger.snapshot()["token_usage"]
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["availability"] == "partial"


def test_anthropic_sdk_cached_usage_reaches_collector_and_ledger_once(tmp_path, monkeypatch):
    import anthropic
    import httpx
    from langchain_anthropic import ChatAnthropic

    from data_collection_workflow.llm_clients import _invoke_with_config
    from data_collection_workflow.models import LLMExtractionOutput

    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_offline_fixture", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-6", "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [{"type": "tool_use", "id": "tool_fixture",
                         "name": "LLMExtractionOutput",
                         "input": {"records": [], "chunk_is_relevant": False}}],
            "usage": {"input_tokens": 10, "output_tokens": 3,
                      "cache_read_input_tokens": 90, "cache_creation_input_tokens": 20},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        sdk = anthropic.Anthropic(
            api_key="offline-placeholder", base_url="https://offline.invalid",
            max_retries=0, http_client=http_client,
        )
        model = ChatAnthropic(
            model="claude-sonnet-4-6", api_key="offline-placeholder",
            base_url="https://offline.invalid", max_tokens=128, max_retries=0,
        )
        monkeypatch.setitem(model.__dict__, "_client", sdk)
        structured = model.with_structured_output(LLMExtractionOutput)
        context = RunContext(tmp_path, {"pipeline_mode": "evidence"})
        config = {"metadata": {"hdc_stage": "structured_extraction"}}
        with context.activate():
            first = _invoke_with_config(structured, [("user", "Return no records")], config)
            cached = _invoke_with_config(structured, [("user", "Return no records")], config)

    assert first["records"] == cached["records"] == []
    assert len(requests) == 1
    snapshot = context.ledger.snapshot()
    assert snapshot["used"]["extraction"] == 1
    assert snapshot["token_usage"] == {
        "input_tokens": 120, "output_tokens": 3, "total_tokens": 123,
        "availability": "complete", "known_operations": 1, "model_operations": 1,
    }
