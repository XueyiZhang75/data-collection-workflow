"""Source and child-link scope cannot be borrowed from unrelated page furniture."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.source_screening import assess_source_task_fit
from data_collection_workflow.resource_discovery import task_resource_candidates, resource_source_entry

STATE = {'structured_task': {'disease': 'measles', 'location': 'Canada',
    'start_date': '2025-01-01', 'end_date': '2025-12-31'}}


def fit(text, **entry):
    return assess_source_task_fit({'source_id': 'source', **entry}, STATE,
        document={'source_id': 'source', 'content_readable': True, 'clean_text': text})


def links(tmp_path, html, state=STATE):
    doc = parse_response(html.encode(), url='https://publisher.example/story',
        source_id='parent', session_dir=tmp_path, content_type='text/html')
    return task_resource_candidates(doc, {}, state)


@pytest.mark.parametrize('href,label', [
    ('https://globeandmail.my.site.com/helpcentre/s/contactsupport', 'Report a technical issue'),
    ('https://www.who.int/data/gho/info/contact-us', 'here'),
    ('https://news.example/p/canada-measles-2025-year-to-date/comment/115903831', 'Measles Canada 2025 comments'),
])
def test_help_contact_and_comment_links_are_not_reports(tmp_path, href, label):
    html = '<h1>Measles Canada 2025 report</h1><h2>Alberta measles case counts</h2>'
    html += f'<p>Measles surveillance report: <a href="{href}">{label}</a>.</p>'
    assert links(tmp_path, html) == []


def test_ambiguous_disease_child_remains_unverified_without_parent_heading(tmp_path):
    rows = links(tmp_path, '<h1>Measles Canada 2025</h1>'
        '<a href="/health/measles-symptoms">Measles symptoms and transmission</a>')
    assert len(rows) == 1  # A genuine unresolved article stays in the acquisition queue.
    child = resource_source_entry(rows[0], parent={'source_id': 'parent'}, depth=1, state=STATE)
    assessed = assess_source_task_fit(child, STATE)
    assert child['ready_for_content_fetch']
    assert assessed['target_verification_status'] != 'verified_target'
    assert assessed['geography_fit'] == 'candidate'
    assert assessed['date_fit'] == 'candidate'


def test_child_article_cannot_borrow_sibling_anchor_observation(tmp_path):
    rows = links(tmp_path, '<h1>Related coverage</h1><ul><li>'
        '<a href="/canada-report">Measles Canada 2025 report</a>'
        '<a href="/symptoms">Measles symptoms and transmission</a></li></ul>')
    candidate = next(row for row in rows if row['url'].endswith('/symptoms'))
    child = resource_source_entry(candidate, parent={'source_id': 'parent'}, depth=1, state=STATE)
    assessed = assess_source_task_fit(child, STATE)
    assert assessed['target_verification_status'] != 'verified_target'
    assert assessed['geography_fit'] == 'candidate'


def test_navigation_language_selector_is_not_canadian_geographic_evidence():
    result = fit('French (Canada)\nMeasles in Philadelphia during 2025: 9 cases.')
    assert result['geography_fit'] != 'match'
    assert result['target_verification_status'] != 'verified_target'


def test_link_destinations_cannot_supply_observation_disease_place_or_year():
    result = fit('Kansas has reported 37 measles cases during 2025.\n'
        'Read [related coverage](https://news.example/2025/measles-canada).')
    assert result['geography_fit'] == 'mismatch'
    assert result['triage_role'] == 'excluded'
    assert result['screening_decision'] == 'exclude'
    assert result['blocked_from_fetch']


def test_other_country_only_observations_are_excluded_even_on_canadian_publisher():
    result = fit('Measles in the United States during 2025: 800 cases.',
        canonical_url='https://canadian-news.example/canada-measles', publisher='Canada News')
    assert result['geography_fit'] == 'mismatch'
    assert result['triage_role'] == 'excluded'
    assert result['screening_decision'] == 'exclude'
    assert result['source_role_final'] == 'excluded'


def test_incidental_link_to_canada_does_not_verify_united_states_counts():
    result = fit('Measles outbreak: Missouri reports its first case of 2025.\n'
        '[Kansas](https://www.kdhe.ks.gov/2314/Measles-Data) has reported 37 cases.\n'
        'There are outbreaks in [Canada and Mexico]'
        '(https://news.example/2025/04/17/health/measles-texas-mexico-canada.html), too.')
    assert result['target_verification_status'] != 'verified_target'
    assert result['date_fit'] == 'match'


@pytest.mark.parametrize('province', ['Alberta', 'Ontario', 'British Columbia', 'Quebec', 'Québec'])
def test_provincial_reports_are_in_scope_for_canada(province):
    result = fit(f'Measles in {province} during 2025: 12 cases.')
    assert result['target_verification_status'] == 'verified_target'
    assert result['geography_fit'] == 'match'


def test_foreign_publisher_with_actual_canada_section_stays_usable():
    result = fit('Measles in United States during 2025: 800 cases.\n'
        'Measles in Canada during 2025: 600 cases.',
        canonical_url='https://www.who.int/global-report', publisher='WHO')
    assert result['target_verification_status'] == 'verified_target'
    assert result['geography_fit'] == 'match'


def test_unknown_geography_genuine_report_stays_unresolved():
    result = fit('Measles surveillance 2025: 12 cases were confirmed.')
    assert result['geography_fit'] == 'candidate'
    assert result['triage_role'] == 'task_record_collection_candidate'


@pytest.mark.parametrize('text', [
    'Measles outbreaks during 2025: 12 cases in California and 8 cases in France.',
    'Measles in California during 2025: 12 cases. Measles in France during 2025: 8 cases.',
    'Measles in the state of Georgia during 2025: 12 cases.',
])
def test_us_state_observations_prevent_foreign_only_exclusion(text):
    state = {'structured_task': {**STATE['structured_task'], 'location': 'United States'}}
    result = assess_source_task_fit({'source_id': 's'}, state,
        document={'source_id': 's', 'content_readable': True, 'clean_text': text})
    assert result['geography_fit'] == 'match'
    assert result['target_verification_status'] == 'verified_target'
    assert not result.get('blocked_from_fetch')


@pytest.mark.parametrize('text,expected', [
    ('Measles in Georgia during 2025: 12 cases.', 'candidate'),
    ('Measles in the country of Georgia during 2025: 12 cases.', 'mismatch'),
    ('University of California reported measles in France during 2025: 12 cases.', 'mismatch'),
])
def test_state_homonyms_and_institutions_do_not_invent_domestic_observations(text, expected):
    state = {'structured_task': {**STATE['structured_task'], 'location': 'United States'}}
    result = assess_source_task_fit({'source_id': 's'}, state,
        document={'source_id': 's', 'content_readable': True, 'clean_text': text})
    assert result['geography_fit'] == expected
    assert result['target_verification_status'] != 'verified_target'
    assert bool(result.get('blocked_from_fetch')) == (expected == 'mismatch')


def test_foreign_only_source_is_retained_for_audit_but_never_scheduled(tmp_path, monkeypatch):
    from test_acquisition_queue import setup_node
    from test_evidence_resource_discovery import _source
    from data_collection_workflow.nodes.source_screening import source_screening, source_critic_and_uncertainty_routing
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    runtime, state, visits = setup_node(monkeypatch, tmp_path, targets=3)
    monkeypatch.setenv('ENABLE_LLM_SOURCE_CRITIC', 'false')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_IDENTITY', 'false')
    sources = [
        ('https://canada.example/us-report', 'Example fever in the United States during 2025: 99 cases.'),
        ('https://foreign-publisher.example/canada-report', 'Example fever in Canada during 2025: 12 cases.'),
        ('https://province.example/alberta-report', 'Example fever in Alberta during 2025: 12 cases.'),
        ('https://who.example/global-report', 'Example fever in United States during 2025: 99 cases. Example fever in Canada during 2025: 12 cases.'),
    ]
    state['source_registry'] = [{**_source(url, number), 'title': title}
                                for number, (url, title) in enumerate(sources, 1)]
    state['source_registry'][0]['must_fetch'] = True
    for _ in range(2):  # Repeated screening must not turn exclusion into background.
        state.update(source_screening(state))
        state.update(source_critic_and_uncertainty_routing(state))
    excluded = state['source_registry'][0]
    assert excluded['source_role_final'] == 'excluded'
    assert excluded['final_screening_decision'] == 'exclude'
    assert excluded['blocked_from_fetch'] and not excluded['ready_for_content_fetch']
    with runtime.activate():
        result = content_fetch_and_parse(state)
    assert set(visits) == {url for url, _ in sources[1:]}
    assert len(result['documents']) == 3
    assert any(row['source_id'] == excluded['source_id'] for row in result['source_registry'])
    assert all(row.get('attempts', 0) == 0 for row in result['acquisition_frontier']['items']
               if row['source_id'] == excluded['source_id'])
