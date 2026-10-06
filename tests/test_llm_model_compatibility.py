"""Model upgrades must preserve one-call validation and explicit token caps."""
import pytest
from pydantic import BaseModel
from data_collection_workflow import llm_clients as clients
from data_collection_workflow.models import LLMExtractionOutput


def settings(model, **extra):
    return dict(provider='anthropic', model=model, temperature=0.0, max_tokens=4096, max_retries=0, **extra)


@pytest.mark.parametrize('model', ['claude-opus-4-7', 'claude-opus-4-8', 'claude-opus-5', 'claude-opus-5-5', 'claude-mythos-preview'])
def test_modern_claude_does_not_send_rejected_temperature(model):
    kwargs = clients.chat_model_kwargs(settings(model))
    assert 'temperature' not in kwargs
    assert kwargs['max_tokens'] == 4096


def test_adaptive_thinking_has_one_explicit_cap_and_no_temperature():
    kwargs = clients.chat_model_kwargs(settings('claude-sonnet-4-6', thinking='adaptive', effort='medium'))
    assert kwargs['thinking'] == {'type': 'adaptive'}
    assert kwargs['output_config'] == {'effort': 'medium'}
    assert 'temperature' not in kwargs
    assert kwargs['max_tokens'] == 4096


@pytest.mark.parametrize('model', ['claude-opus-5-5', 'claude-fable-5-1', 'claude-mythos-5-1'])
def test_always_thinking_models_reject_disable_locally(model):
    with pytest.raises(ValueError, match='thinking'):
        clients.chat_model_kwargs(settings(model, thinking='disabled'))


def test_gateway_keeps_token_cap_without_native_claude_parameters():
    kwargs = clients.chat_model_kwargs({**settings('claude-sonnet-5'), 'provider': 'openai'})
    assert kwargs['max_tokens'] == 4096
    assert 'thinking' not in kwargs
    with pytest.raises(ValueError, match='anthropic'):
        clients.chat_model_kwargs({**settings('claude-sonnet-5', thinking='adaptive'), 'provider': 'openai'})


class SmallOutput(BaseModel):
    value: int


@pytest.mark.parametrize('model,schema,expected', [
    ('claude-sonnet-4-6', LLMExtractionOutput, 'function_calling'),
    ('claude-sonnet-5', LLMExtractionOutput, 'json_prompt'),
    ('claude-opus-5-5', LLMExtractionOutput, 'json_prompt'),
    ('claude-opus-5-5', SmallOutput, 'json_schema'),
    ('claude-fable-5-1', LLMExtractionOutput, 'json_prompt'),
])
def test_output_routing_respects_model_and_actual_schema(model, schema, expected):
    from data_collection_workflow.llm_compatibility import structured_output_method
    assert structured_output_method(settings(model), schema) == expected


@pytest.mark.parametrize('method', ['json_schema', 'function_calling'])
def test_incompatible_explicit_extraction_mode_fails_before_dispatch(method):
    from data_collection_workflow.llm_compatibility import structured_output_method
    with pytest.raises(ValueError):
        structured_output_method(settings('claude-opus-5-5', structured_output_method=method), LLMExtractionOutput)


def test_thinking_content_is_not_parsed_as_answer():
    from types import SimpleNamespace
    message = SimpleNamespace(content=[{'type':'thinking', 'text':'{"value":999}'}, {'type':'text','text':'{"value":3}'}])
    assert clients._message_content(message) == '{"value":3}'


def test_provider_failure_does_not_trigger_hidden_second_request(monkeypatch):
    class Model:
        calls = 0
        def with_structured_output(self, schema): return self
        def invoke(self, messages, config=None):
            self.calls += 1
            raise RuntimeError('model is unavailable')
    model = Model()
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    monkeypatch.setattr(clients, 'build_chat_model', lambda *args, **kwargs: model)
    with pytest.raises(RuntimeError, match='unavailable'):
        clients.run_pydantic_structured_llm('system', 'user', SmallOutput, model='claude-sonnet-5')
    assert model.calls == 1


@pytest.mark.parametrize('reason', ['max_tokens', 'length', 'refusal', 'content_filter'])
def test_stopped_response_cannot_look_like_successful_empty_extraction(monkeypatch, reason):
    from types import SimpleNamespace
    class Model:
        def invoke(self, messages, config=None):
            return SimpleNamespace(content='{"records":[]}', response_metadata={'stop_reason':reason})
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    with pytest.raises(ValueError, match=reason):
        clients._invoke_with_config(Model(), [], {'metadata':{}})


def test_advisory_content_field_is_not_a_provider_message(tmp_path, monkeypatch):
    from data_collection_workflow.session_runtime import RunContext
    class Article(BaseModel):
        content: str
    class Model:
        calls = 0
        def with_structured_output(self, schema): return self
        def invoke(self, messages, config=None):
            self.calls += 1
            return Article(content='A short validated summary.')
    model = Model()
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setattr(clients, 'build_chat_model', lambda *args: model)
    ctx = RunContext(tmp_path, {})
    with ctx.activate():
        first = clients.run_pydantic_structured_llm('system', 'user', Article, model='claude-sonnet-5')
        second = clients.run_pydantic_structured_llm('system', 'user', Article, model='claude-sonnet-5')
    assert first == second
    assert first['content'] == 'A short validated summary.'
    assert model.calls == 1


def test_older_opus_effort_support_is_not_confused_with_adaptive_support():
    kwargs = clients.chat_model_kwargs(settings('claude-opus-4-5', effort='high'))
    assert kwargs['output_config'] == {'effort':'high'}
    assert 'thinking' not in kwargs
    with pytest.raises(ValueError, match='effort'):
        clients.chat_model_kwargs(settings('claude-opus-4-5', effort='max'))


def test_mythos_preview_does_not_support_xhigh_effort():
    with pytest.raises(ValueError, match='xhigh'):
        clients.chat_model_kwargs(settings('claude-mythos-preview', effort='xhigh'))
