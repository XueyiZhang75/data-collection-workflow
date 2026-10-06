"""Interactive users select their account's model before any external work."""
import json
import pytest
import scripts.collect as interactive

TASK = ['--disease', 'measles', '--location', 'Canada', '--start-date', '2024', '--end-date', '2024', '--session-id', 'model_choice', '--no-dashboard']


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    for name in ('LLM_PROVIDER', 'LLM_MODEL', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'TAVILY_API_KEY'):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv('HDC_' + name, raising=False)
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(interactive.sys.stdin, 'isatty', lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Model selection must not invoke external work.')

    monkeypatch.setattr(interactive, 'run_workflow', forbidden)
    monkeypatch.setattr(interactive, '_preflight_llm_with_trace_policy', forbidden)
    monkeypatch.setattr(interactive, '_write_generated_config', forbidden)


def preview(capsys, *options):
    assert interactive.main([*TASK, '--print-config-only', *options]) == 0
    return json.loads(capsys.readouterr().out.split('sanitized_config_json:', 1)[1])['config']


def test_provider_switch_does_not_reuse_another_providers_environment_model(monkeypatch, capsys):
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('LLM_MODEL', 'claude-user-selected')
    config = preview(capsys, '--provider', 'openai')
    assert config['llm']['provider'] == 'openai'
    assert config['llm']['model'] == ''


def test_no_model_noninteractive_stops_before_key_checks(monkeypatch, capsys):
    def forbidden(**kwargs):
        raise AssertionError('Choose a model before checking account credentials.')
    monkeypatch.setattr(interactive, '_require_keys', forbidden)
    assert interactive.main([*TASK, '--provider', 'openai']) == 2
    error = capsys.readouterr().err
    assert '--model' in error and 'LLM_MODEL' in error


def test_interactive_user_selects_provider_and_model(monkeypatch, capsys):
    monkeypatch.setattr(interactive.sys.stdin, 'isatty', lambda: True)
    answers = iter(['openai', 'user-selected-model'])
    prompts = []
    def prompt(label, current=None):
        prompts.append(label)
        return next(answers)
    monkeypatch.setattr(interactive, '_prompt', prompt)
    observed = []
    def keys(**kwargs):
        observed.append(kwargs)
        return ['OPENAI_API_KEY']
    monkeypatch.setattr(interactive, '_require_keys', keys)
    assert interactive.main(TASK) == 2
    output = capsys.readouterr().out
    assert any('provider' in label.lower() for label in prompts)
    assert any('model' in label.lower() for label in prompts)
    assert 'user-selected-model' in output
    assert observed == [{'provider': 'openai', 'llm_enabled': True}]


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_arbitrary_account_model_id_is_preserved(capsys, provider):
    config = preview(capsys, '--provider', provider, '--model', 'my-account-model-2029')
    assert config['llm']['provider'] == provider
    assert config['llm']['model'] == 'my-account-model-2029'
