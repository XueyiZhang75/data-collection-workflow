"""Neutral runnable metadata must retain accounting and cache identities."""
import pytest
from data_collection_workflow import llm_clients
from data_collection_workflow.session_runtime import RunContext, BudgetExceeded


def _context(tmp_path):
    return RunContext(tmp_path, {
        'pipeline_mode': 'evidence',
        'structured_task': {'disease': 'measles'},
        'universal': {'budget_limits': {'extraction': 3}, 'extraction_reserve': 1},
    })


@pytest.mark.parametrize('legacy_first', [False, True])
def test_metadata_aliases_share_cache_and_recovery_accounting(tmp_path, monkeypatch, legacy_first):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    ctx = _context(tmp_path)
    observed = []
    class Model:
        def invoke(self, messages, config=None):
            observed.append(config['metadata'])
            return {'records': []}
    def invoke(chunk, recovery=False, legacy=False):
        metadata = {'workflow_stage': 'structured_extraction', 'llm_settings': {'model': 'account-model'},
                    'chunk_id': chunk, 'chunk_ids': [chunk], 'recovery': recovery}
        if legacy:
            names = {'workflow_stage': 'hdc_stage', 'llm_settings': 'hdc_llm_settings',
                     'chunk_id': 'hdc_chunk_id', 'chunk_ids': 'hdc_chunk_ids', 'recovery': 'hdc_recovery'}
            metadata = {names[key]: value for key, value in metadata.items()}
        return llm_clients._invoke_with_config(Model(), [{'content': chunk}], {'metadata': metadata})
    with ctx.activate():
        invoke('a', legacy=legacy_first)
        invoke('a', legacy=not legacy_first)
        invoke('b', legacy=legacy_first)
        with pytest.raises(BudgetExceeded):
            invoke('blocked', legacy=legacy_first)
        invoke('recovery', recovery=True, legacy=legacy_first)
    assert len(observed) == 3
    assert ctx.ledger.snapshot()['used']['extraction'] == 3
    assert set(ctx.ledger.attempted_chunk_ids()) == {'a', 'b', 'recovery'}
    assert all(row['workflow_stage'] == 'structured_extraction' for row in observed)
    assert all(not any(key.startswith('hdc_') for key in row) for row in observed)


def test_canonical_metadata_presence_wins_over_conflicting_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    ctx = _context(tmp_path)
    observed = []
    class Model:
        def invoke(self, messages, config=None):
            observed.append(config['metadata'])
            return {'records': []}
    metadata = {'workflow_stage': 'structured_extraction', 'hdc_stage': 'planner',
                'llm_settings': {}, 'hdc_llm_settings': {'model': 'wrong-model'},
                'chunk_id': '', 'hdc_chunk_id': 'wrong-chunk',
                'chunk_ids': [], 'hdc_chunk_ids': ['wrong-span'],
                'recovery': False, 'hdc_recovery': True}
    with ctx.activate():
        for content in ('first', 'second'):
            llm_clients._invoke_with_config(Model(), [{'content': content}], {'metadata': metadata})
        with pytest.raises(BudgetExceeded):
            llm_clients._invoke_with_config(Model(), [{'content': 'third'}], {'metadata': metadata})
    assert ctx.ledger.snapshot()['used']['extraction'] == 2
    assert ctx.ledger.attempted_chunk_ids() == []
    assert observed[0]['llm_settings'] == {}
    assert observed[0]['recovery'] is False
    assert metadata['hdc_stage'] == 'planner'  # Caller-owned metadata remains untouched.


def test_new_runnable_configs_emit_only_neutral_metadata():
    config = llm_clients._langsmith_runnable_config('test', stage='planner', settings={'model': 'available-model'})
    assert config['metadata']['workflow_stage'] == 'planner'
    assert config['metadata']['llm_settings']['model'] == 'available-model'
    assert not any(key.startswith('hdc_') for key in config['metadata'])
