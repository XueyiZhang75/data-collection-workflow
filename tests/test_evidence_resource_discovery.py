"""Current-session resource discovery through real parsing, routing and budgets."""
import os

import pytest

from data_collection_workflow.models import Document
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.session_runtime import RunContext


def test_html_links_preserve_redirect_base_anchor_and_origin(tmp_path):
    parsed = parse_response(
        b'<html><nav><a href="/privacy">Privacy</a></nav><main><h1>Example fever data</h1>'
        b'<p>Annual 2025 data: <a href="../files/cases.csv#data">Download CSV</a></p></main></html>',
        url='https://example.invalid/old', final_url='https://example.invalid/reports/index.html',
        source_id='index', session_dir=tmp_path, content_type='text/html')
    doc = Document(**parsed)
    links = doc.metadata.get('outbound_links', [])
    link = next((link for link in links if link.get('text') == 'Download CSV'), None)
    assert link is not None, 'evidence discarded the actual href before resource discovery'
    assert link['href'] == 'https://example.invalid/files/cases.csv'
    assert link['source_url'] == 'https://example.invalid/reports/index.html'
    assert link['source_content_hash'] == doc.content_hash
    assert 'Annual 2025 data' in link['context']
    assert link['locator']['anchor_index'] == 1


def _env(monkeypatch, limit=8):
    from data_collection_workflow.environment import WORKFLOW_ENV_NAMES
    for key in list(os.environ):
        if key in WORKFLOW_ENV_NAMES or key.startswith('HDC_'):
            monkeypatch.delenv(key, raising=False)
    for key, value in {
        'PIPELINE_MODE': 'evidence', 'ENABLE_LIVE_FETCH': 'true',
        'FETCH_SEARCH_DERIVED_SOURCES': 'true', 'FETCH_MAX_SEARCH_DERIVED_SOURCES': '8',
        'FETCH_MAX_TOTAL_SOURCES': '8', 'USE_FIXTURE_DOCUMENTS': 'false',
        'ENABLE_LLM_SOURCE_IDENTITY': 'false', 'LLM_SOURCE_IDENTITY_POST_FETCH': 'false',
        'EVENT_LINK_EXPANSION_LIMIT': str(limit),
    }.items():
        monkeypatch.setenv(key, value)


def _source(url, number=1):
    return dict(source_id=f's{number}', url=url, canonical_url=url,
                title='Example fever surveillance data, Canada 2025',
                source_type='official_public_health_agency', publisher='Fixture authority',
                status='ready_for_content_fetch', ready_for_content_fetch=True,
                final_screening_decision='include_for_content_fetch', discovery_method='web_search',
                source_role='data_source', source_role_final='collection', search_rank=number,
                credibility_score=0.88, credibility_level='high', requires_human_review=False)


def _node(monkeypatch, tmp_path, pages, *, sources=None, budget=8, limit=8, total_limit=None, direct=False):
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    import requests
    _env(monkeypatch, limit)
    if direct:
        monkeypatch.setenv("COLLECTION_MODE", "direct_collection")
    if total_limit is not None:
        monkeypatch.setenv("FETCH_MAX_TOTAL_SOURCES", str(total_limit))
    visits = []

    class Response:
        def __init__(self, url):
            self.url = url
            self.status_code = 200
            kind, self.body = pages[url]
            self.headers = {'content-type': kind}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter_content(self, size):
            yield self.body

    def get(url, **kwargs):
        visits.append(url)
        return Response(url)

    monkeypatch.setattr(requests, 'get', get)
    state = {'structured_task': {'disease': 'Example fever', 'location': 'Canada',
                                'start_date': '2025-01-01', 'end_date': '2025-12-31'},
             'source_registry': sources or [_source('https://example.invalid/index')],
             'collection_trace': []}
    runtime = RunContext(tmp_path / 'session', {
        'pipeline_mode': 'evidence',
        'universal': {'budget_limits': {'fetch_ordinary': budget, 'fetch': budget}},
    })
    with runtime.activate():
        result = content_fetch_and_parse(state)
    return result, visits, runtime.ledger.snapshot()


def test_data_resource_fetch_preempts_remaining_search_hits_without_extra_budget(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance data</h1>'
            b'<p>Canada annual 2025 <a href="/cases.csv">Download case counts CSV</a></p>'),
        'https://example.invalid/cases.csv': ('text/csv', b'disease,location,year,cases\nExample fever,Canada,2025,12\n'),
        'https://example.invalid/other-index': ('text/plain', b'Example fever background'),
    }
    result, visits, budget = _node(monkeypatch, tmp_path, pages, budget=2,
        sources=[_source('https://example.invalid/index'), _source('https://example.invalid/other-index', 2)])
    assert visits == ['https://example.invalid/index', 'https://example.invalid/cases.csv']
    assert budget['used'] == {'fetch': 2, 'fetch_ordinary': 2}
    assert any(doc['document_type'] == 'csv' and doc['tables'][0]['rows'][0][-1] == '12'
               for doc in result['documents'])


def test_archive_frontier_is_two_hops_deduplicated_and_ignores_navigation(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance</h1>'
            b'<nav><a href="/privacy">Privacy data policy</a></nav>'
            b'<a href="/archive/2025">2025 annual report archive</a>'),
        'https://example.invalid/archive/2025': ('text/html', b'<h1>Example fever Canada report series 2025</h1>'
            b'<a href="/cases.json">Download case counts JSON</a>'
            b'<a href="/cases.json#duplicate">Download same JSON</a>'
            b'<a href="/index">Report archive index</a>'),
        'https://example.invalid/cases.json': ('application/json', b'{"disease":"Example fever","cases":12}'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages)
    assert visits == list(pages)
    children = [s for s in result['source_registry'] if s.get('discovery_method') == 'task_resource_link']
    assert sorted(s['resource_link_depth'] for s in children) == [1, 2]


def test_child_resource_does_not_inherit_parent_trust_or_reporting_period(monkeypatch, tmp_path):
    parent = _source('https://example.invalid/index')
    parent.update(must_fetch=True, source_identity_unverified=False, metadata_only_identity=False,
                  date_fit='match', reporting_period_start='2025-01-01', reporting_period_end='2025-12-31')
    pages = {
        parent['url']: ('text/html', b'<h1>Example fever surveillance data Canada 2025</h1>'
                       b'<a href="https://download.invalid/cases.csv">Example fever Canada 2025 Download CSV</a>'),
        'https://download.invalid/cases.csv': ('text/csv', b'year,cases\n2024,12\n'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages, sources=[parent])
    assert len(visits) == 2
    child = next(s for s in result['source_registry'] if s.get('parent_source_id'))
    assert child['source_identity_unverified'] is True
    assert not child.get('must_fetch')
    assert child.get('date_fit') != 'match'
    assert not child.get('reporting_period_start')
    assert child.get('publisher') != parent['publisher']


def test_user_resource_limit_and_unsupported_spreadsheet_are_honest(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance data Canada 2025</h1>'
            b'<a href="/data.xlsx">Download spreadsheet annual case counts</a>'
            b'<a href="/later.csv">Download case counts CSV</a>'),
        'https://example.invalid/data.xlsx': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', b'PK\x03\x04binary workbook'),
        'https://example.invalid/later.csv': ('text/csv', b'cases\n12\n'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages, limit=1)
    assert len(visits) == 2
    spreadsheet = next(doc for doc in result['documents'] if doc.get('url', '').endswith('.xlsx'))
    assert not spreadsheet['content_readable']
    assert spreadsheet['parse_status'] == 'unsupported_format'
    assert spreadsheet['clean_text'] == ''


def test_resource_depth_limit_stops_at_second_linked_html_page(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance Canada 2025</h1><a href="/archive/2025">Report archive 2025</a>'),
        'https://example.invalid/archive/2025': ('text/html', b'<h1>Example fever report archive 2025</h1><a href="/reports/table">Annual report table</a>'),
        'https://example.invalid/reports/table': ('text/html', b'<h1>Example fever report 2025</h1><a href="/third-report">Report archive</a>'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages)
    assert visits == list(pages)
    assert not any(source['url'].endswith('/third-report') for source in result['source_registry'])


def test_explicit_download_on_task_section_can_follow_unverified_data_host(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance data Canada 2025</h1><a href="https://files.invalid/data.csv">Download CSV data</a>'),
        'https://files.invalid/data.csv': ('text/csv', b'year,cases\n2025,12\n'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages)
    assert visits == list(pages)
    child = next(source for source in result['source_registry'] if source.get('parent_source_id'))
    assert child['publisher'] != 'Fixture authority'


def test_promoted_already_discovered_resource_is_not_fetched_twice(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever data Canada 2025</h1><a href="/cases.csv">Download CSV data</a>'),
        'https://example.invalid/cases.csv': ('text/csv', b'year,cases\n2025,12\n'),
    }
    parent = _source('https://example.invalid/index')
    parent['target_fit_status'] = 'verified_target_collection'
    result, visits, _ = _node(monkeypatch, tmp_path, pages, sources=[parent, _source('https://example.invalid/cases.csv', 2)])
    assert visits == list(pages)
    assert len(result['source_registry']) == 2


def test_zero_expansion_limit_is_respected(monkeypatch, tmp_path):
    pages = {'https://example.invalid/index': ('text/html', b'<h1>Example fever data Canada 2025</h1><a href="/cases.csv">Download CSV data</a>')}
    pages['https://example.invalid/cases.csv'] = ('text/csv', b'cases\n12\n')
    result, visits, _ = _node(monkeypatch, tmp_path, pages, limit=0)
    assert visits == ['https://example.invalid/index']
    assert len(result['source_registry']) == 1


def test_search_title_alone_cannot_turn_unrelated_page_links_into_task_resources(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>General ministry procurement archive</h1><a href="/report.csv">Download annual data report</a>'),
        'https://example.invalid/report.csv': ('text/csv', b'contracts\n12\n'),
    }
    result, visits, _ = _node(monkeypatch, tmp_path, pages)
    assert visits == ['https://example.invalid/index']
    assert len(result['source_registry']) == 1


def test_heading_spans_retain_original_text_locations(tmp_path):
    parsed = parse_response(b'<h1>Example fever</h1><h2>Canada <em>2025</em></h2><p>12 cases</p>',
                            url='https://example.invalid/report', source_id='s', session_dir=tmp_path,
                            content_type='text/html')
    headings = [span for span in parsed['locator_spans'] if span.get('role') == 'heading']
    assert [span['heading_level'] for span in headings] == [1, 2, 2]
    assert [parsed['clean_text'][span['char_start']:span['char_end']] for span in headings] == ['Example fever', 'Canada', '2025']
    assert headings[1]['section_id'] == headings[2]['section_id']


def test_resource_queue_respects_node_total_even_when_ledger_is_larger(monkeypatch, tmp_path):
    pages = {
        'https://example.invalid/index': ('text/html', b'<h1>Example fever surveillance data Canada 2025</h1><a href="/cases.csv">Download CSV data</a>'),
        'https://example.invalid/cases.csv': ('text/csv', b'year,cases\n2025,12\n'),
    }
    result, visits, budget = _node(monkeypatch, tmp_path, pages, total_limit=1, budget=8)
    assert visits == ['https://example.invalid/index']
    assert budget['used'] == {'fetch': 1, 'fetch_ordinary': 1}
    assert any(row.get('skip_reason') == 'max_total_sources_reached_after_resource_priority'
               for row in result['fetch_manifest'])


def test_registered_child_uses_maintained_publisher_name():
    from data_collection_workflow.resource_discovery import resource_source_entry
    from data_collection_workflow.source_identity import lookup_source_identity_registry
    url = 'https://www.canada.ca/example.csv'
    identity = lookup_source_identity_registry(url)
    child = resource_source_entry({'url':url, 'link_text':'Data', 'resource_type':'csv',
                                   'score':1, 'selection_reasons':[], 'provenance':{}},
                                  parent={'source_id':'parent'}, depth=1)
    assert child['publisher'] == identity['publisher_name']


def test_direct_collection_keeps_candidates_after_search_verified_primary_fails(monkeypatch, tmp_path):
    primary = _source('https://example.invalid/index')
    fallback = _source('https://example.invalid/fallback-index', 2)
    for entry in (primary, fallback):
        entry['discovery_method'] = 'live_search_result'
    primary.update(target_fit_status='verified_target_collection', target_verification_status='search_verified')
    fallback.update(target_fit_status='task_record_collection_candidate')
    pages = {
        primary['url']: ('text/html', b'<h1>Page not found</h1>'),
        fallback['url']: ('text/plain', b'Example fever in Canada during 2025: 12 confirmed cases reported. This surveillance report provides the annual observation.'),
    }
    result, visits, budget = _node(monkeypatch, tmp_path, pages, sources=[primary,fallback], direct=True)
    assert visits == list(pages)
    assert budget['used'] == {'fetch': 2, 'fetch_ordinary': 2}
    assert any(doc.get('source_id') == fallback['source_id'] and doc['content_readable'] for doc in result['documents'])


def test_final_fetch_permission_can_resolve_metadata_uncertainty_without_clearing_review(monkeypatch, tmp_path):
    source = _source('https://example.invalid/report')
    source.update(discovery_method='live_search_result', source_role_final='collection_support',
                  credibility_score=0.7012, credibility_level='medium',
                  requires_human_review=True, human_review_recommended=True,
                  human_review_reason='disease_relevant_but_time_window_unclear')
    pages = {source['url']: ('text/html', b'<p>Example fever in Canada: 12 cases in 2025.</p>')}
    result, visits, budget = _node(monkeypatch, tmp_path, pages, sources=[source])
    assert visits == [source['url']]
    assert budget['used'] == {'fetch': 1, 'fetch_ordinary': 1}
    retained = result['source_registry'][0]
    assert retained['requires_human_review'] and retained['human_review_recommended']
    assert retained['credibility_score'] == source['credibility_score']
    assert not retained.get('must_fetch')
    assert not retained.get('reporting_period_start')


@pytest.mark.parametrize('revision,changes', [
    ('legacy', {}),
    ('evidence', {'ready_for_content_fetch': False}),
    ('evidence', {'final_screening_decision': 'needs_human_review'}),
    ('evidence', {'final_screening_decision': 'exclude'}),
    ('evidence', {'source_role_final': 'needs_human_review'}),
    ('evidence', {'source_role_final': 'excluded'}),
    ('evidence', {'credibility_score': 0.4}),
    ('evidence', {'credibility_level': 'needs_review'}),
    ('evidence', {'source_disease_relevance_status': 'unrelated_disease'}),
])
def test_metadata_review_fetch_exception_retains_explicit_routing_and_trust_gates(monkeypatch, revision, changes):
    from data_collection_workflow.models import ContentFetchPolicy
    from data_collection_workflow.nodes.content_processing import _classify_skip_reason, load_content_fetch_policy
    _env(monkeypatch)
    monkeypatch.setenv('PIPELINE_MODE', revision)
    source = _source('https://example.invalid/report')
    source.update(discovery_method='live_search_result', requires_human_review=True,
                  human_review_recommended=True)
    source.update(changes)
    reason = _classify_skip_reason(source, ContentFetchPolicy(**load_content_fetch_policy()),
        fetch_config={'fetch_search_derived_sources': True, 'min_credibility_score': 0.55})
    assert reason is not None
