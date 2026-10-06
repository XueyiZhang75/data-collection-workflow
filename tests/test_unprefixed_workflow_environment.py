"""Public environment settings use unprefixed names only."""
from __future__ import annotations

import warnings

from data_collection_workflow.llm_clients import get_llm_settings


def test_old_prefixed_model_and_temperature_cannot_select_a_new_run(monkeypatch):
    for name in ('LLM_MODEL', 'LLM_TEMPERATURE', 'HDC_LLM_MODEL', 'HDC_LLM_TEMPERATURE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HDC_LLM_MODEL', 'obsolete-model')
    monkeypatch.setenv('HDC_LLM_TEMPERATURE', '0.7')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        settings = get_llm_settings()
    assert settings['model'] == ''
    assert settings['temperature'] == 0.0
    assert not any('HDC_' in str(item.message) for item in caught)


def test_unprefixed_model_and_temperature_are_read_as_configured(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'chosen-model')
    monkeypatch.setenv('LLM_TEMPERATURE', '0.2')
    monkeypatch.setenv('HDC_LLM_MODEL', 'obsolete-model')
    monkeypatch.setenv('HDC_LLM_TEMPERATURE', '0.7')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        settings = get_llm_settings()
    assert settings['model'] == 'chosen-model'
    assert settings['temperature'] == 0.2
    assert not any('HDC_' in str(item.message) for item in caught)
