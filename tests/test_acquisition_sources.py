"""Offline contracts for source discovery and acquisition eligibility."""
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest

from data_collection_workflow.models import SourceScreeningPolicy
from data_collection_workflow.config import load_source_screening_policy
from data_collection_workflow.query_policy import assess_query_task_fit

discovery = importlib.import_module('data_collection_workflow.nodes.source_discovery')
screening = importlib.import_module('data_collection_workflow.nodes.source_screening')


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setenv('ENABLE_LIVE_SEARCH', 'false')
    monkeypatch.setenv('ENABLE_LIVE_FETCH', 'false')
    import socket
    def prohibited(*args, **kwargs):
        raise AssertionError('Network is forbidden in acquisition repair tests')
    monkeypatch.setattr(socket.socket, 'connect', prohibited)


def state(disease='Example fever', location='Example Region'):
    return {'structured_task': {'disease': disease, 'location': location,
            'start_date': '2025-01-01', 'end_date': '2025-12-31',
            'collection_mode': 'direct_collection'},
            'collection_spec': {'collection_mode': 'direct_collection'},
            'collection_trace': []}


def source(**changes):
    return {'source_id': 'report', 'canonical_url': 'https://health.invalid/reports/42',
            'title': 'Example fever: External Situation Report 42',
            'source_type': 'international_organization_report',
            'discovery_method': 'live_search_result', **changes}


def test_search_snippet_survives_registry_and_supports_source_screening():
    candidate = source(snippet='Example fever in Example Region in 2025: 412 confirmed cases and 3 deaths.')
    row = discovery._registry_entry_from_candidate(candidate, candidate['canonical_url'], None).model_dump()
    assert row['snippet'] == candidate['snippet']
    result = screening.source_screening({**state(), 'source_registry': [row]})
    retained = result['source_registry'][0]
    assert retained['screening_decision'] == 'include'
    assert retained['source_role'] == 'data_source'
    assert retained['geography_fit'] == 'match'


def test_unknown_source_function_is_a_candidate_not_a_relevance_rejection():
    result = screening.source_screening({**state(), 'source_registry': [source()]})
    retained = result['source_registry'][0]
    assert retained['screening_decision'] != 'exclude'
    assert retained['source_role'] != 'irrelevant_source'
    assert retained['geography_fit'] == 'candidate'


def test_verified_task_metadata_replaces_only_provisional_legacy_exclusion():
    entry = source(title='Example fever in Example Region, June 2025',
                   source_role='irrelevant_source', screening_decision='exclude',
                   screening_flags=['ambiguous_source_role'])
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    preliminary = screening._screen_entry(entry, policy, state())
    revised, _ = screening._apply_direct_triage_verification(entry, preliminary, state(), collection_mode='direct_collection')
    assert revised['screening_decision'] == 'include'
    assert revised['source_role'] == 'data_source'


@pytest.mark.parametrize('negative', [
    {'source_disease_relevance_status': 'unrelated_disease'},
    {'geography_fit': 'wrong_geography'},
    {'blocked_from_fetch': True, 'blocked_from_fetch_reason': 'user_excluded'},
])
def test_positive_text_does_not_erase_explicit_contradiction_or_user_boundary(negative):
    entry = source(title='Example fever in Example Region 2025', **negative)
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    preliminary = screening._screen_entry(entry, policy, state())
    revised, _ = screening._apply_direct_triage_verification(entry, preliminary, state(), collection_mode='direct_collection')
    assert revised.get('geography_fit') == 'wrong_geography' or revised.get('disease_fit') == 'mismatch' or revised.get('blocked_from_fetch')


@pytest.mark.parametrize('intent', [
    'Ministry of Health and Sanitation surveillance reports',
    'IDSR epidemiological bulletins', 'WHO AFRO surveillance',
    'Africa CDC epidemiology reports', 'health ministry prevention service notices',
])
def test_institution_and_retrieval_vocabulary_do_not_need_a_word_allowlist(intent):
    task = state('mpox', 'Sierra Leone')
    fit = assess_query_task_fit({'query': f'mpox Sierra Leone 2025 {intent}',
                                'provider_channel': 'official_site_search'}, task)
    assert fit['accepted'], fit


@pytest.mark.parametrize('query', [
    {'query': 'mpox Canada surveillance'},
    {'query': 'mpox Sierra Leone measles reports'},
    {'query': 'mpox Sierra Leone vessel Orion reports'},
    {'query': 'mpox Sierra Leone reports', 'event_terms': ['vessel Orion']},
])
def test_broader_retrieval_vocabulary_keeps_wrong_task_and_event_guards(query):
    assert not assess_query_task_fit(query, state('mpox', 'Sierra Leone'))['accepted']


def test_unknown_journal_article_is_not_classified_as_a_fact_sheet():
    from data_collection_workflow.source_product_profile import profile_source_product
    profile = profile_source_product(source(
        title='Genomic epidemiology of Example fever in Example Region, 2025',
        source_type='peer_reviewed_literature'), state())
    assert profile['data_product_type'] != 'background_fact_sheet'
    assert profile['machine_readability'] != 'not_extractable'


def test_direct_critic_payload_does_not_inject_global_reserved_domains(monkeypatch):
    from data_collection_workflow.agents import source_critic_agent
    seen = {}
    def assess(**kwargs):
        seen.update(json.loads(kwargs['user_prompt']))
        return {'critic_decision': 'uncertain'}
    monkeypatch.setattr(source_critic_agent.llm_clients, 'run_structured_llm_json', assess)
    source_critic_agent.assess_source_with_llm(source(), state()['collection_spec'], {},
        {'validation_reserved_domains': ['who.int'], 'validation_reserved_source_ids': ['reserved']})
    assert seen['source_role_policy_summary']['validation_reserved_domains'] == []
    assert seen['source_role_policy_summary']['validation_reserved_source_ids'] == []


def test_post_fetch_identity_is_not_globally_disabled_by_coverage_ids(monkeypatch):
    from data_collection_workflow import source_identity
    calls = []
    def assess(**kwargs):
        calls.append(kwargs['source_entry']['source_id'])
        return {}
    monkeypatch.setattr(source_identity, 'assess_source_identity_with_llm', assess)
    row = source(coverage_requirement_ids=['task_metric'])
    documents = [{'source_id': 'report', 'title': row['title'], 'clean_text': 'Example fever surveillance.',
                  'fetch_status': 'success', 'parse_status': 'parsed', 'content_readable': True}]
    source_identity.enrich_source_identity_registry_post_fetch([row], [], documents,
        collection_spec=state()['collection_spec'], llm_enabled=True)
    assert calls == ['report']


def test_metadata_and_fetched_content_share_task_fit_without_cross_source_inheritance():
    assess = getattr(screening, 'assess_source_task_fit', None)
    assert callable(assess)
    row = source()
    text = 'Example fever in Example Region in 2025: 412 confirmed cases.'
    fit = assess(row, state(), document={'source_id': 'report', 'clean_text': text, 'content_readable': True})
    assert fit['target_verification_status'] == 'verified_target'
    wrong = assess(row, state(), document={'source_id': 'other', 'clean_text': text, 'content_readable': True})
    assert wrong['target_verification_status'] != 'verified_target'


def test_adaptive_discovery_uses_shared_hard_limits_and_strict_keeps_small_limits(monkeypatch):
    from data_collection_workflow import session_runtime
    settings = discovery.SourceSearchSettings(iterative_max_iterations=3,
        iterative_max_total_queries=8, iterative_max_total_results=88, max_total_results=88)
    runtime = SimpleNamespace(config={'pipeline_mode': 'evidence', 'universal': {'budget_policy': {'version': 2, 'mode': 'adaptive'}}},
                              ledger=SimpleNamespace(limits={'search': 40, 'search_results': 240}))
    monkeypatch.setattr(session_runtime, 'get_runtime', lambda: runtime)
    adjust = getattr(discovery, '_effective_discovery_settings', None)
    assert callable(adjust)
    expanded = adjust(settings)
    assert expanded.iterative_max_total_queries == 40
    assert expanded.iterative_max_total_results == expanded.max_total_results == 240
    assert expanded.iterative_max_iterations >= 40
    runtime.config['universal']['budget_policy']['mode'] = 'strict'
    assert adjust(settings) == settings


def test_generic_supplementary_discovery_does_not_require_an_outbreak_label():
    gate = getattr(discovery, '_supplementary_discovery_allowed', None)
    assert callable(gate)
    assert gate(state(), []) is True


@pytest.mark.parametrize('document_changes', [
    {'source_id': None}, {'content_readable': False}, {'http_status_code': 503},
    {'quality_issues': ['error_page']}, {'acquisition_status': 'blocked'},
])
def test_unbound_or_failed_page_cannot_verify_metadata(document_changes):
    document = {'source_id': 'report', 'content_readable': True,
                'clean_text': 'Example fever in Example Region during 2025: 412 confirmed cases.',
                **document_changes}
    fit = screening.assess_source_task_fit(source(), state(), document=document)
    assert fit['target_verification_status'] != 'verified_target'


def test_publication_line_does_not_erase_later_observation_on_same_line():
    row = source(title='Clinical findings', published_date='2026-08-01',
        snippet='Published 2026. Example fever in Example Region during 2025: 412 confirmed cases.')
    fit = screening.assess_source_task_fit(row, state())
    assert fit['target_verification_status'] == 'verified_target'


def test_query_hints_and_publication_date_cannot_verify_an_observation():
    row = source(title='Situation report', published_date='2025-08-01',
                 query_used='Example fever Example Region 2025',
                 expected_fields=['cases_confirmed'], source_type='official_public_health_agency')
    fit = screening.assess_source_task_fit(row, state())
    assert fit['date_fit'] == fit['geography_fit'] == fit['disease_fit'] == 'candidate'


def test_source_user_boundary_survives_a_provisional_classification_flag():
    row = source(title='Example fever Example Region 2025', source_role='irrelevant_source',
                 screening_decision='exclude', screening_flags=['ambiguous_source_role'],
                 blocked_from_fetch=True, blocked_from_fetch_reason='user_excluded')
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    preliminary = screening._screen_entry(row, policy, state())
    revised, _ = screening._apply_direct_triage_verification(row, preliminary, state(), collection_mode='direct_collection')
    assert revised['screening_decision'] == 'exclude'
    assert revised['blocked_from_fetch_reason'] == 'user_excluded'


@pytest.mark.parametrize('term', ['Orion reports', 'Mayflower cluster reports', 'USA cases', 'U.S. cases'])
def test_institution_channel_cannot_license_new_event_or_foreign_country(term):
    fit = assess_query_task_fit({'query': 'mpox Sierra Leone 2025 ' + term,
                                'provider_channel': 'official_site_search'}, state('mpox', 'Sierra Leone'))
    assert not fit['accepted'], fit


def test_reference_year_is_not_an_observation_period():
    fit = screening.assess_source_task_fit(source(), state(), document={
        'source_id': 'report', 'content_readable': True,
        'clean_text': 'Example fever in Example Region. References: 2024.'})
    assert fit['date_fit'] == 'candidate'
    assert fit['target_verification_status'] != 'temporal_mismatch'


def test_abbreviated_publication_month_does_not_supply_observation_year():
    fit = screening.assess_source_task_fit(source(title='Published Sep. 2025.'), state())
    assert fit['date_fit'] == 'candidate'



def test_legacy_critic_payload_preserves_reserved_policy(monkeypatch):
    from data_collection_workflow.agents import source_critic_agent
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    seen = {}
    def assess(**kwargs):
        seen.update(json.loads(kwargs['user_prompt']))
        return {'critic_decision': 'uncertain'}
    monkeypatch.setattr(source_critic_agent.llm_clients, 'run_structured_llm_json', assess)
    source_critic_agent.assess_source_with_llm(source(), state()['collection_spec'], {},
        {'validation_reserved_domains': ['who.int'], 'validation_reserved_source_ids': ['reserved']})
    assert seen['source_role_policy_summary']['validation_reserved_domains'] == ['who.int']


def test_research_product_does_not_claim_individual_case_report_before_reading():
    from data_collection_workflow.source_product_profile import profile_source_product
    profile = profile_source_product(source(title='Genomic epidemiology of Example fever in Example Region 2025',
                                            source_type='peer_reviewed_literature'), state())
    assert profile['data_product_type'] == 'research_article'
    assert profile['expected_evidence_role'] != 'individual_case_evidence'
    assert profile['source_product_profile_reason'] != 'peer_reviewed_case_report'


def test_unrelated_task_does_not_inherit_legacy_event_product_priority():
    from data_collection_workflow.source_product_profile import profile_source_product
    profile = profile_source_product(source(title='Arrival and cleaning of cruise ship Hondius',
                                            source_type='official_public_health_agency'), state('mpox', 'Canada'))
    assert profile['data_product_type'] != 'event_outbreak_report'
    assert profile['task_specificity'] != 'event_specific'


@pytest.mark.parametrize("content", [
    "Published online. Example fever in Example Region during 2025: 412 confirmed cases.",
    "Updated case counts for Example fever in Example Region during 2025: 412 confirmed cases.",
])
def test_publication_words_without_metadata_date_preserve_observation_period(content):
    fit = screening.assess_source_task_fit(source(), state(), document={
        'source_id': 'report', 'content_readable': True, 'clean_text': content})
    assert fit['date_fit'] == 'match'
    assert fit['target_verification_status'] == 'verified_target'
