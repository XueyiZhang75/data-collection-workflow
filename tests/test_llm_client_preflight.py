from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_preflight_llm_model_returns_configured_model(monkeypatch):
    from data_collection_workflow import llm_clients

    class FakeModel:
        def with_structured_output(self, schema):
            from data_collection_workflow.models import LLMExtractionOutput
            assert schema is LLMExtractionOutput
            return self
        def invoke(self, messages, config=None):
            assert messages
            return {'records': [{'disease': 'measles', 'cases_confirmed': 3,
                                 'evidence_quote': 'There were 3 confirmed measles cases.'}],
                    'chunk_is_relevant': True}

    monkeypatch.setattr(llm_clients, "build_chat_model", lambda settings: FakeModel())

    result = llm_clients.preflight_llm_model(
        {"provider": "anthropic", "model": "claude-fable-5"}
    )

    assert result == {
        "status": "ok",
        "provider": "anthropic",
        "model": "claude-fable-5",
        "structured_output_method": "json_prompt",
        "schema": "LLMExtractionOutput",
    }


def test_preflight_llm_model_fails_fast_with_model_name(monkeypatch):
    from data_collection_workflow import llm_clients

    class FakeModel:
        def with_structured_output(self, schema):
            return self
        def invoke(self, messages, config=None):
            raise RuntimeError("model not found")

    monkeypatch.setattr(llm_clients, "build_chat_model", lambda settings: FakeModel())

    try:
        llm_clients.preflight_llm_model(
            {"provider": "anthropic", "model": "claude-fable-5"}
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive; the test expects failure
        raise AssertionError("preflight should fail for an unavailable model")

    assert "claude-fable-5" in message
    assert "model not found" in message


def test_anthropic_fable_model_kwargs_omit_deprecated_temperature():
    from data_collection_workflow import llm_clients

    kwargs = llm_clients.chat_model_kwargs(
        {
            "provider": "anthropic",
            "model": "claude-fable-5",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 2,
        }
    )

    assert kwargs["model"] == "claude-fable-5"
    assert "temperature" not in kwargs


def test_anthropic_sonnet_5_model_kwargs_omit_deprecated_temperature():
    from data_collection_workflow import llm_clients

    kwargs = llm_clients.chat_model_kwargs(
        {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 2,
        }
    )

    assert kwargs["model"] == "claude-sonnet-5"
    assert "temperature" not in kwargs


def test_anthropic_non_fable_model_kwargs_keep_temperature():
    from data_collection_workflow import llm_clients

    kwargs = llm_clients.chat_model_kwargs(
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "temperature": 0.0,
            "max_tokens": 4096,
            "max_retries": 2,
        }
    )

    assert kwargs["temperature"] == 0.0


def test_preflight_ok_text_is_not_structured_compatibility(monkeypatch):
    import pytest
    from data_collection_workflow import llm_clients
    class Model:
        def invoke(self, messages, config=None): return 'OK'
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    monkeypatch.setattr(llm_clients, 'build_chat_model', lambda *args: Model())
    with pytest.raises(ValueError, match='preflight failed'):
        llm_clients.preflight_llm_model({'provider':'anthropic', 'model':'claude-opus-5-5'})


def test_structured_preflight_is_budgeted_and_reused_on_resume(tmp_path, monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.session_runtime import RunContext
    from langchain_core.messages import AIMessage
    import json
    class Model:
        calls = 0
        def invoke(self, messages, config=None):
            self.calls += 1
            return AIMessage(content=json.dumps({'records': [{'disease':'measles', 'cases_confirmed':3,
                'evidence_quote':'There were 3 confirmed measles cases.'}], 'chunk_is_relevant':True}))
    model = Model()
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setattr(llm_clients, 'build_chat_model', lambda *args: model)
    settings = {'provider':'anthropic', 'model':'claude-opus-5-5'}
    config = {'llm':settings}
    original = RunContext(tmp_path, config)
    with original.activate():
        first = llm_clients.preflight_llm_model(settings)
    resumed = RunContext(tmp_path, config, resume=True)
    with resumed.activate():
        second = llm_clients.preflight_llm_model(settings)
    assert first == second
    assert model.calls == 1
    assert resumed.ledger.snapshot()['used'] == {'model:preflight':1}


def test_preflight_failure_has_dedicated_value_error_subclass(monkeypatch):
    from data_collection_workflow import llm_clients
    def unavailable(settings):
        raise RuntimeError("offline model not found")
    monkeypatch.setattr(llm_clients, "build_chat_model", unavailable)
    try:
        llm_clients.preflight_llm_model({"provider":"anthropic", "model":"claude-selected"})
    except ValueError as exc:
        assert type(exc).__name__ == "LLMModelPreflightError"
        assert "claude-selected" in str(exc)
        assert isinstance(exc.__cause__, RuntimeError)
    else:
        raise AssertionError("preflight failure must propagate")
