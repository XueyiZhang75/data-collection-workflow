"""Advisory planning cannot override explicit acquisition or held-out boundaries."""
import pytest
from data_collection_workflow.models import ContentFetchPolicy
from data_collection_workflow.nodes.content_processing import _classify_skip_reason, load_content_fetch_policy
from data_collection_workflow.source_credibility import _role_recommendation


def source(**changes):
    return {'source_id': 's', 'url': 'https://health.example/reports/2025.pdf',
            'canonical_url': 'https://health.example/reports/2025.pdf',
            'discovery_method': 'live_search_result', 'query_source': 'authority_gap_retry',
            'source_type': 'official_public_health_agency', 'source_type_final': 'national_public_health_agency',
            'source_role': 'data_source', 'source_role_final': 'collection',
            'title': 'Pertussis surveillance report', 'role_hint': 'collection',
            'data_product_type': 'event_outbreak_report', 'task_specificity': 'event_specific',
            'machine_readability': 'pdf_extractable', 'expected_evidence_role': 'aggregate_event_evidence',
            'credibility_level': 'high', 'credibility_score': .9,
            'ready_for_content_fetch': True, 'final_screening_decision': 'include_for_content_fetch', **changes}


def reason(entry, **config):
    cfg = {'fetch_search_derived_sources': True, 'allow_needs_review': True,
           'allowed_final_roles': ['collection', 'collection_support', 'validation', 'context'],
           'min_credibility_score': .55, **config}
    return _classify_skip_reason(entry, ContentFetchPolicy(**load_content_fetch_policy()),
                                 collection_mode='direct_collection', fetch_config=cfg)


@pytest.mark.parametrize('boundary', [
    {'source_role_final': 'excluded', 'final_screening_decision': 'exclude'},
    {'blocked_from_fetch': True, 'blocked_from_fetch_reason': 'explicit_exclusion'},
    {'source_role': 'irrelevant_source', 'source_role_final': 'excluded'},
])
@pytest.mark.parametrize('must_fetch', [False, True])
def test_evidence_authority_rescue_respects_explicit_exclusion(monkeypatch, boundary, must_fetch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert reason(source(**boundary, must_fetch=must_fetch)) is not None


def test_evidence_must_fetch_respects_explicit_domain_block(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert reason(source(must_fetch=True), domain_blocklist=['health.example']) == 'domain_blocklisted'


def test_evidence_uncertain_but_fetchable_source_is_retained(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert reason(source(source_role_final='needs_human_review', requires_human_review=True)) is None


def test_evidence_planned_validation_hint_does_not_reserve_collected_source(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    role, why = _role_recommendation(source(role_hint='validation'),
        {'disease_relevance_score': .9, 'data_granularity_score': .9}, [])
    assert role == 'collection'


@pytest.mark.parametrize('role', ['validation_reserved', 'validation_source'])
def test_evidence_explicit_validation_reservation_is_preserved(monkeypatch, role):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert _role_recommendation(source(role_hint='validation', source_role=role),
        {'disease_relevance_score': .9, 'data_granularity_score': .9}, [])[0] == 'validation'


def test_legacy_planned_validation_hint_is_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'standard')
    assert _role_recommendation(source(role_hint='validation'),
        {'disease_relevance_score': .9, 'data_granularity_score': .9}, [])[0] == 'validation'

@pytest.mark.parametrize('config,expected', [
    ({'domain_allowlist': ['other.example']}, 'domain_not_allowlisted'),
    ({'fetch_search_derived_sources': False}, 'search_derived_fetch_disabled'),
    ({'allowed_final_roles': ['context']}, 'final_role_not_allowed'),
])
def test_evidence_priority_does_not_override_user_fetch_constraints(monkeypatch, config, expected):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert reason(source(must_fetch=True), **config) == expected


@pytest.mark.parametrize('must_fetch', [False, True])
def test_evidence_review_prohibition_wins_even_when_role_is_listed(monkeypatch, must_fetch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    assert reason(source(must_fetch=must_fetch, source_role_final='needs_human_review'),
                  allow_needs_review=False, allowed_final_roles=['needs_human_review']) == 'needs_review_not_allowed'


@pytest.mark.parametrize('direct', [False, True])
def test_zero_fetch_selection_limit_disables_normal_sources(monkeypatch, tmp_path, direct):
    from test_evidence_resource_discovery import _node
    pages = {'https://example.invalid/index': ('text/html',
        b'<h1>Example fever Canada 2025</h1><a href="/cases.csv">Download case counts</a>'),
        'https://example.invalid/cases.csv': ('text/csv', b'year,cases\n2025,12\n')}
    result, visits, ledger = _node(monkeypatch, tmp_path, pages, budget=3, total_limit=0, direct=direct)
    assert visits == []
    assert ledger['used'].get('fetch', 0) == 0


def test_zero_search_derived_selection_limit_is_respected(monkeypatch):
    from data_collection_workflow.nodes.content_processing import _build_fetch_requests
    from test_evidence_resource_discovery import _env, _source
    _env(monkeypatch)
    state = {'structured_task': {'disease': 'Example fever', 'location': 'Canada',
             'start_date': '2025-01-01', 'end_date': '2025-12-31'},
             'source_registry': [{**_source('https://example.invalid/index'),
                                  'discovery_method': 'live_search_result'}]}
    requests, skipped, manifest = _build_fetch_requests(state,
        ContentFetchPolicy(**load_content_fetch_policy()), True,
        {'fetch_search_derived_sources': True, 'max_total_sources': 8,
         'max_search_derived_sources': 0, 'allowed_final_roles': ['collection'],
         'min_credibility_score': 0})
    assert requests == []
    assert sum(skipped.values()) == 1
    assert not manifest[0]['selected_for_fetch']


@pytest.mark.parametrize('must_fetch', [False, True])
def test_priority_cannot_fetch_held_out_source_in_masked_mode(monkeypatch, must_fetch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    entry = source(must_fetch=must_fetch, source_role='validation_reserved', source_role_final='validation')
    assert _classify_skip_reason(entry, ContentFetchPolicy(**load_content_fetch_policy()),
        collection_mode='masked_validation', role_policy={'validation_reserved_source_ids': ['s']},
        fetch_config={'fetch_search_derived_sources': True, 'allowed_final_roles': ['validation']}) == 'validation_reserved'
