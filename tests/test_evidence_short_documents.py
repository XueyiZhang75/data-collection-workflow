"""Document readability must not discard concise source evidence by length."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.content_processing import document_quality_check


@pytest.mark.parametrize('body,content_type', [
    ('<h1>Pertussis in Canada 2025</h1><table><tr><th>Confirmed cases</th></tr><tr><td>12</td></tr></table>','text/html'),
    ('Pertussis in Canada: 12 confirmed cases in 2025.','text/plain'),
    ('disease,country,year,cases_confirmed\nPertussis,Canada,2025,12','text/csv'),
])
def test_concise_readable_evidence_is_kept_for_field_assessment(tmp_path,monkeypatch,body,content_type):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    doc=parse_response(body.encode(),url='https://example.invalid/report',source_id='s',session_dir=tmp_path,content_type=content_type)
    assert len(doc['clean_text'])<80
    state={'structured_task':{'disease':'Pertussis','location':'Canada'},'documents':[doc]}
    result=document_quality_check(state)['documents'][0]
    assert result['quality_status']=='partial'
    assert 'insufficient_clean_text' not in result['quality_issues']


@pytest.mark.parametrize('body', [
    '<h1>Access denied</h1>',
    '<script>load()</script><h1>Pertussis</h1><p>Loading...</p>',
    '',
])
def test_short_error_shell_and_empty_responses_still_unusable(tmp_path,monkeypatch,body):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    doc=parse_response(body.encode(),url='https://example.invalid/report',source_id='s',session_dir=tmp_path,content_type='text/html')
    result=document_quality_check({'documents':[doc]})['documents'][0]
    assert result['quality_status']=='unusable'


def test_legacy_short_document_threshold_is_unchanged(tmp_path,monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    doc=parse_response(b'Pertussis in Canada: 12 confirmed cases in 2025.',url='https://example.invalid/report',source_id='s',session_dir=tmp_path)
    assert document_quality_check({'documents':[doc]})['documents'][0]['quality_status']=='unusable'
