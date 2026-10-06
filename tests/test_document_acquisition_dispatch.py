from types import SimpleNamespace
import pytest
from data_collection_workflow.models import ContentFetchRequest
from data_collection_workflow.nodes import content_processing as cp
from data_collection_workflow import document_acquisition, session_runtime


def entry():
    return dict(source_id='s',url='https://www.cdc.gov/report',must_fetch=True,
                source_identity_unverified=False,metadata_only_identity=False,
                discovery_method='live_search_result',target_verification_status='verified',
                target_verification_reason='Report matches requested disease, country and reporting period.',
                disease_fit='match',geography_fit='match',date_fit='match')


def dispatch(monkeypatch,tmp_path,source,fetch_config=None):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    runtime=SimpleNamespace(session_dir=tmp_path,config={'timeout_seconds':31,'universal':{'acquisition':{'ocr_languages':'spa','max_bytes':1234}}})
    monkeypatch.setattr(session_runtime,'get_runtime',lambda:runtime)
    captured={}
    def acquire(url,**kwargs):
        captured.update(kwargs)
        return document_acquisition.parse_response(b'Fixture report',url=url,source_id='s',session_dir=tmp_path)
    monkeypatch.setattr(document_acquisition,'acquire_document',acquire)
    request=ContentFetchRequest(source_id='s',url=source.get('_request_url','https://www.cdc.gov/report'),canonical_url='https://www.cdc.gov/report',final_screening_decision='include',fetch_purpose='collection')
    doc=cp._fetch_live_document_with_providers(request,source,None,fetch_config or {})
    assert doc.clean_text=='Fixture report'
    return captured


def test_dispatcher_passes_nested_acquisition_and_verified_priority(monkeypatch,tmp_path):
    result=dispatch(monkeypatch,tmp_path,entry(),{'max_bytes':5678})
    assert result['config']['ocr_languages']=='spa'
    assert result['config']['max_bytes']==5678
    assert result['config']['timeout_seconds']==31
    assert result['priority_context']==dict(source_id='s',verified_url='https://www.cdc.gov/report',must_fetch=True,source_identity_verified=True,task_fit_verified=True)


@pytest.mark.parametrize('changes',[
    {'must_fetch':False}, {'source_identity_unverified':True}, {'metadata_only_identity':True},
    {'source_id':'other'}, {'url':'https://www.cdc.gov/different'},
    {'url':'https://cdc.gov.evil.invalid/report','_request_url':'https://cdc.gov.evil.invalid/report','domain':'cdc.gov','publisher':'CDC','source_type':'national_public_health_agency'},
    {'date_fit':'unknown'}, {'geography_fit':'broader_than_task'}, {'disease_fit':None},
    {'target_verification_reason':''}, {'target_verification_status':'unknown'},
    {'disease_fit':'match','source_disease_relevance_status':'unrelated_disease'},
])
def test_dispatcher_rejects_unverified_metadata_host_or_task_fit(monkeypatch,tmp_path,changes):
    source=entry();source.update(changes)
    result=dispatch(monkeypatch,tmp_path,source,{'priority':True,'task_fit_verified':True})
    assert result.get('priority_context') is None
