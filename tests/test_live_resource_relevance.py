"""Regression boundaries for task resources versus page utility links."""
from html import escape

import pytest

from data_collection_workflow.resource_discovery import task_resource_candidates
from data_collection_workflow.document_acquisition import parse_response


def _candidates(tmp_path, href, label, *, attributes=""):
    body = (
        '<main><h1>Example fever surveillance data</h1>'
        '<p>Example fever annual case data 2025: '
        f'<a href="{escape(href, quote=True)}" {attributes}>{escape(label)}</a>'
        '</p></main>'
    ).encode()
    document = parse_response(
        body, url='https://agency.example/reports', final_url='https://agency.example/reports',
        source_id='agency-report', session_dir=tmp_path, content_type='text/html',
    )
    state = {'structured_task': {'disease': 'Example fever', 'location': 'Canada',
                                'start_date': '2025-01-01', 'end_date': '2025-12-31'}}
    return task_resource_candidates(document, {'source_id': 'agency-report'}, state)


@pytest.mark.parametrize(('href', 'label'), [
    ('/feedback?report=annual-data', 'Give feedback on this page'),
    ('/survey?report=annual-data', 'Tell us what you think'),
    ('/survey/data', 'Complete our survey'),
    ('/contact?report=annual-data', 'Contact us'),
    ('/privacy?report=annual-data', 'Privacy policy'),
    ('/account/sign-in?return=/reports/data', 'Access your account'),
    ('/share?url=https%3A%2F%2Fagency.example%2Freports', 'Share this report'),
    ('https://search.example/lookup?title=Example%20fever%20surveillance%20data',
     'Find this reference'),
])
def test_utility_link_does_not_inherit_case_data_purpose(tmp_path, href, label):
    assert _candidates(tmp_path, href, label) == []


@pytest.mark.parametrize(('href', 'label', 'attributes'), [
    ('https://files.example/get?download=1&id=annual-report&format=csv',
     'Download surveillance data', ''),
    ('https://files.example/api/export?year=2025&topic=cases',
     'Download case counts', 'type="text/csv" download'),
    ('/data/surveillance-survey.csv?year=2025', 'Download surveillance survey data', ''),
    ('/reports/feedback-study.pdf?year=2025', 'Example fever clinical study report', ''),
])
def test_real_downloads_keep_query_parameters_and_evidence_purpose(tmp_path, href, label, attributes):
    candidates = _candidates(tmp_path, href, label, attributes=attributes)
    assert len(candidates) == 1
    assert '?' in candidates[0]['url']
    assert candidates[0]['resource_type'] in {'csv', 'pdf'}


@pytest.mark.parametrize(('href', 'label', 'kind'), [
    ('/data/contact-tracing.csv', 'Download contact tracing data', 'csv'),
    ('/survey/data', 'Example fever surveillance survey data', 'html'),
])
def test_subject_words_do_not_turn_data_products_into_utility_actions(tmp_path, href, label, kind):
    candidates = _candidates(tmp_path, href, label)
    assert len(candidates) == 1
    assert candidates[0]['resource_type'] == kind
