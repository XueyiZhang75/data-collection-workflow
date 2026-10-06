"""Environment migration contract; all calls are offline."""
from __future__ import annotations
import os
import warnings
import pytest
from data_collection_workflow import llm_clients
from data_collection_workflow.runtime_profile import temporary_workflow_env, workflow_run_env_from_config

@pytest.fixture(autouse=True)
def clean_model_env(monkeypatch):
    for name in ('LLM_MODEL','LLM_PROVIDER','LLM_EFFORT','LLM_THINKING','LLM_STRUCTURED_OUTPUT_METHOD'):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv('HDC_' + name, raising=False)


def test_neutral_model_and_provider_take_priority(monkeypatch):
    monkeypatch.setenv('HDC_LLM_MODEL', 'legacy-model')
    monkeypatch.setenv('LLM_MODEL', 'arbitrary-account-model')
    monkeypatch.setenv('HDC_LLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        settings = llm_clients.get_llm_settings()
    assert settings['model'] == 'arbitrary-account-model'
    assert settings['provider'] == 'openai'
    assert not [w for w in captured if 'HDC_LLM_MODEL' in str(w.message)]


def test_explicit_empty_neutral_value_does_not_revive_legacy(monkeypatch):
    monkeypatch.setenv('HDC_LLM_MODEL', 'legacy-model')
    monkeypatch.setenv('LLM_MODEL', '')
    monkeypatch.setenv('HDC_LLM_EFFORT', 'high')
    monkeypatch.setenv('LLM_EFFORT', '')
    settings = llm_clients.get_llm_settings()
    assert settings['model'] == ''
    assert settings['effort'] is None


def test_prefixed_model_is_ignored_without_warning(monkeypatch):
    monkeypatch.setenv('HDC_LLM_MODEL', 'private-account-model-value')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        assert llm_clients.get_llm_settings()['model'] == ''
    assert not [w for w in captured if 'HDC_' in str(w.message)]


@pytest.mark.parametrize('fail', [False, True])
def test_scope_shadows_legacy_and_restores_both_on_failure(monkeypatch, fail):
    monkeypatch.setenv('HDC_LLM_MODEL', 'legacy-parent')
    monkeypatch.setenv('HDC_LLM_EFFORT', 'high')
    monkeypatch.setenv('LLM_MODEL', 'parent')
    before = {name: os.environ.get(name) for name in ('LLM_MODEL','HDC_LLM_MODEL','LLM_EFFORT','HDC_LLM_EFFORT')}
    try:
        with temporary_workflow_env({'LLM_MODEL': 'run-model', 'LLM_EFFORT': ''}):
            assert llm_clients.get_llm_settings()['model'] == 'run-model'
            assert llm_clients.get_llm_settings()['effort'] is None
            os.environ['LLM_MODEL'] = 'mutated-child'
            if fail:
                raise RuntimeError('interrupted')
    except RuntimeError:
        pass
    assert {name: os.environ.get(name) for name in before} == before


def test_legacy_update_mapping_normalizes_without_emitting_old_keys(monkeypatch):
    with temporary_workflow_env({'HDC_LLM_MODEL': 'old-config-model'}):
        assert os.environ['LLM_MODEL'] == 'old-config-model'
        assert llm_clients.get_llm_settings()['model'] == 'old-config-model'
    assert 'LLM_MODEL' not in os.environ


def test_scope_canonical_update_wins_independent_of_order():
    for updates in ({'LLM_MODEL': '', 'HDC_LLM_MODEL': 'legacy'}, {'HDC_LLM_MODEL': 'legacy', 'LLM_MODEL': ''}):
        with temporary_workflow_env(updates):
            assert llm_clients.get_llm_settings()['model'] == ''


def test_runtime_profile_emits_only_neutral_environment_keys():
    updates = workflow_run_env_from_config({'llm': {'provider': 'anthropic', 'model': 'account-model'}})
    assert updates['LLM_MODEL'] == 'account-model'
    assert not any(name.startswith('HDC_') for name in updates)


def test_legacy_policy_name_resolves_canonical_environment(monkeypatch):
    from data_collection_workflow.environment import get_env
    monkeypatch.setenv('HDC_COLLECTION_MODE', 'masked_validation')
    monkeypatch.setenv('COLLECTION_MODE', 'direct_collection')
    assert get_env('HDC_COLLECTION_MODE') == 'direct_collection'


def test_unrelated_provider_keys_do_not_inherit_legacy_prefix(monkeypatch):
    from data_collection_workflow.environment import get_env
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('HDC_ANTHROPIC_API_KEY', 'unrelated-secret')
    assert get_env('ANTHROPIC_API_KEY') is None


@pytest.mark.parametrize('origin', ['legacy_shell', 'legacy_dotenv', 'neutral_dotenv_over_legacy_shell', 'explicit'])
def test_interactive_reports_actual_model_source(monkeypatch, tmp_path, capsys, origin):
    import scripts.collect as interactive
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    options = []
    expected_model = ''
    expected_source = 'not selected'
    if origin != 'legacy_dotenv':
        monkeypatch.setenv('HDC_LLM_MODEL', 'legacy-model')
    if origin == 'legacy_dotenv':
        (tmp_path / '.env').write_text('HDC_LLM_MODEL=legacy-model\n', encoding='utf-8')
        expected_source = 'project default'
    if origin == 'neutral_dotenv_over_legacy_shell':
        (tmp_path / '.env').write_text('LLM_MODEL=neutral-model\n', encoding='utf-8')
        expected_model = 'neutral-model'
        expected_source = f"project .env ({tmp_path / '.env'}): LLM_MODEL"
    if origin == 'explicit':
        options = ['--model', 'selected-model']
        expected_model, expected_source = 'selected-model', '--model'
    monkeypatch.setattr(interactive, '_require_keys', lambda **kwargs: [])
    monkeypatch.setattr(interactive, '_write_generated_config', lambda *args: tmp_path / 'unused.json')
    def run(args):
        out = capsys.readouterr().out
        assert f'llm_model: {expected_model} (source: {expected_source})' in out
        return {}
    monkeypatch.setattr(interactive, 'run_workflow', run)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = interactive.main([
            '--pipeline-mode', 'evidence', '--disease', 'mpox',
            '--location', 'Sierra Leone', '--start-date', '2025', '--end-date', '2025',
            '--session-id', 'model_source_test', '--no-dashboard', *options,
        ])
    if origin in {'legacy_shell', 'legacy_dotenv'}:
        assert result == 2
        assert '--model' in capsys.readouterr().err
    else:
        assert result == 0
    warnings_used = [w for w in captured if 'HDC_LLM_MODEL' in str(w.message)]
    assert not warnings_used
    # python-dotenv writes process variables directly; restore our fixture's keys.
    for name in ('LLM_MODEL', 'HDC_LLM_MODEL'):
        monkeypatch.delenv(name, raising=False)


def test_old_policy_property_accepts_new_collection_mode(monkeypatch):
    from data_collection_workflow.config import get_collection_mode
    monkeypatch.setenv('HDC_COLLECTION_MODE', 'masked_validation')
    monkeypatch.setenv('COLLECTION_MODE', 'direct_collection')
    assert get_collection_mode({
        'enabled_env_var': 'HDC_COLLECTION_MODE',
        'default_collection_mode': 'standard',
        'supported_collection_modes': ['standard', 'masked_validation', 'direct_collection'],
    }) == 'direct_collection'


def test_nested_scopes_restore_outer_and_original_aliases(monkeypatch):
    monkeypatch.setenv('HDC_LLM_MODEL', 'legacy-original')
    with temporary_workflow_env({'LLM_MODEL': 'outer'}):
        with temporary_workflow_env({'HDC_LLM_MODEL': 'inner'}):
            assert llm_clients.get_llm_settings()['model'] == 'inner'
        assert llm_clients.get_llm_settings()['model'] == 'outer'
    assert 'LLM_MODEL' not in os.environ
    assert os.environ['HDC_LLM_MODEL'] == 'legacy-original'


def test_scoped_session_and_trace_do_not_promote_legacy_values(monkeypatch):
    import scripts.run_workflow as configured
    monkeypatch.delenv('TRACE_ID', raising=False)
    monkeypatch.setenv('HDC_TRACE_ID', 'old-parent')
    previous = configured._install_trace_env({'metadata': {'trace_id': 'new-run'}})
    try:
        assert os.environ['TRACE_ID'] == 'new-run'
    finally:
        configured._restore_env(previous)
    assert 'TRACE_ID' not in os.environ
    assert os.environ['HDC_TRACE_ID'] == 'old-parent'
