"""Received content supplies extraction readiness; discovery metadata does not."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.content_processing import document_quality_check


@pytest.mark.parametrize('disease,country,body',[
    ('pertussis','Canada','Canada reported 18 000 suspected pertussis cases during 2025.'),
    ('chikungunya','France','France a signalé 32 456 cas confirmés de chikungunya durant 2025.'),
])
def test_received_metric_assertion_is_extractable_without_publisher_or_search_title(tmp_path,monkeypatch,disease,country,body):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    doc=parse_response(body.encode(),url='https://unknown.example/report',source_id='s',session_dir=tmp_path)
    result=document_quality_check({'structured_task':{'disease':disease,'location':country},'documents':[doc]})
    row=result['documents'][0]
    assert row['extraction_readiness']=='ready'
    assert row['task_relevance_status']=='target_disease_match'


def test_inherited_search_title_cannot_make_wrong_received_disease_ready(tmp_path,monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    body='Brazil reported 123 confirmed dengue cases during 2025.'
    doc=parse_response(body.encode(),url='https://unknown.example/report',source_id='s',session_dir=tmp_path)
    doc['title']='Pertussis in Canada during 2025'
    result=document_quality_check({'structured_task':{'disease':'pertussis','location':'Canada'},'documents':[doc]})
    row=result['documents'][0]
    assert row['extraction_readiness']=='not_ready'
    assert row['not_extractable_for_task_disease'] is True
