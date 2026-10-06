"""Public entry points present the English task result without a second locale."""
import pytest
from pydantic import BaseModel


def test_interactive_summary_prints_english_result_and_report(capsys):
    from scripts.collect import _print_task_result_summary
    _print_task_result_summary({
        'artifact_paths': {'task_result_english': 'outputs/sessions/example/task_result.md'},
        'task_result_summary': {'headline': 'Confirmed: 12 cases.'},
    })
    output = capsys.readouterr().out
    assert 'task_result_report: outputs/sessions/example/task_result.md' in output
    assert 'task_result: Confirmed: 12 cases.' in output
    assert 'chinese' not in output.lower()


@pytest.mark.parametrize('entry', ['json', 'pydantic', 'extraction'])
def test_model_message_requests_english_prose_without_rewriting_evidence(monkeypatch, entry):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.config import load_llm_structured_extraction_policy
    from data_collection_workflow.models import LLMStructuredExtractionPolicy

    class Reply(BaseModel):
        note: str

    captured = []
    evidence = '报告了12例。'
    monkeypatch.setattr(llm_clients, 'get_llm_settings', lambda: {'provider': 'openai', 'model': 'test-model'})
    monkeypatch.setattr(llm_clients, 'build_chat_model', lambda *args: object())
    monkeypatch.setattr(llm_clients, '_langsmith_runnable_config', lambda *args, **kwargs: {})

    def invoke(model, messages, *args, **kwargs):
        captured.extend(messages)
        return {'note': 'English explanation.'}

    def invoke_structured(model, messages, *args, **kwargs):
        captured.extend(messages)
        return {'note': 'English explanation.'}, 'function_calling'

    monkeypatch.setattr(llm_clients, '_invoke_with_config', invoke)
    monkeypatch.setattr(llm_clients, '_invoke_structured_schema', invoke_structured)
    if entry == 'json':
        llm_clients.run_structured_llm_json('Assess the evidence.', evidence, 'Reply')
    elif entry == 'pydantic':
        llm_clients.run_pydantic_structured_llm('Assess the evidence.', evidence, Reply)
    else:
        policy = LLMStructuredExtractionPolicy.model_validate(load_llm_structured_extraction_policy())
        captured = llm_clients._build_llm_messages({'text': evidence}, policy)
    system = captured[0]['content']
    assert 'Write generated explanations, summaries, and rationale in English.' in system
    assert 'Preserve verbatim source quotations' in system
    assert evidence in captured[1]['content']
