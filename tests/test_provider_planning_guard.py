import pytest
from data_collection_workflow import llm_clients
from data_collection_workflow.nodes.task_scope import disease_intelligence_builder
from data_collection_workflow.provider_failures import ProviderAccountLimit


def test_planning_account_halt_preserves_generic_local_fallback(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_DISEASE_INTELLIGENCE','true')
    monkeypatch.setenv('DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED','false')
    def stopped(**kwargs):raise ProviderAccountLimit('anthropic',{'reason':'account credits exhausted'})
    monkeypatch.setattr(llm_clients,'run_pydantic_structured_llm',stopped)
    result=disease_intelligence_builder({'collection_spec':{'disease':'measles','geography':'Canada','time_window':'2025'},'collection_trace':[]})
    assert result['disease_intelligence']['generation_method']=='generic_deterministic_fallback'
    assert any('ProviderAccountLimit' in str(item) for item in result['collection_trace'])


def test_nonaccount_planning_failure_still_obeys_required_llm_policy(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_DISEASE_INTELLIGENCE','true')
    monkeypatch.setenv('DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED','false')
    def broken(**kwargs):raise ValueError('invalid request schema')
    monkeypatch.setattr(llm_clients,'run_pydantic_structured_llm',broken)
    with pytest.raises(RuntimeError,match='LLM required'):
        disease_intelligence_builder({'collection_spec':{'disease':'measles','geography':'Canada','time_window':'2025'}})


def test_required_identity_account_stop_is_deferred_without_asserting_authority(monkeypatch):
    from data_collection_workflow import source_identity
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def stopped(**kwargs):raise ProviderAccountLimit('anthropic',{'reason':'account credits exhausted'})
    monkeypatch.setattr(source_identity,'assess_source_identity_with_llm',stopped)
    sources,assessments,summary=source_identity.apply_source_identity_to_registry(
        [{'source_id':'s','url':'https://unknown.invalid/report','title':'measles in Canada in 2025'}],
        collection_spec={'disease':'measles','geography':'Canada','time_window':'2025'},
        llm_enabled=True,require_llm=True,allow_deterministic_fallback=False)
    assert summary['blocked_llm_required_count']==0
    assert sources[0]['source_identity_status']=='provider_deferred'
    assert sources[0]['source_identity_unverified'] is True
    assert 'provider_account_limit' in assessments[0]['source_identity_llm_skipped_reason']
    assert assessments[0]['llm_used'] is False


def test_required_identity_other_error_retains_strict_policy(monkeypatch):
    from data_collection_workflow import source_identity
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def broken(**kwargs):raise ValueError('invalid source schema')
    monkeypatch.setattr(source_identity,'assess_source_identity_with_llm',broken)
    sources,assessments,summary=source_identity.apply_source_identity_to_registry(
        [{'source_id':'s','url':'https://unknown.invalid/report','title':'measles in Canada in 2025'}],
        collection_spec={'disease':'measles','geography':'Canada','time_window':'2025'},
        llm_enabled=True,require_llm=True,allow_deterministic_fallback=False)
    assert summary['blocked_llm_required_count']==1
    assert sources[0]['source_identity_status']=='blocked_llm_required'
