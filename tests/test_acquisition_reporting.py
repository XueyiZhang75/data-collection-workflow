"""All views must distinguish unattempted, deferred and real failed sources."""
import pytest
from data_collection_workflow.source_progress import build_source_progress


def test_budget_refusal_is_pending_not_fetch_failure():
    state={'source_registry':[{'source_id':'s','url':'https://data.example/report'}],
        'documents':[{'source_id':'s','url':'https://data.example/report',
          'acquisition_status':'budget_exhausted','budget_exhausted_kind':'http_requests',
          'request_success':False,'is_live_fetched':False,'clean_text':'',
          'parse_status':'not_started'}]}
    progress=build_source_progress({},state)
    row=progress['sources'][0]
    assert row['processing_status']=='budget_deferred'
    assert row['processing_reason']=='http_requests'
    assert progress['fetch_failed_sources']==0
    assert progress['budget_deferred_sources']==1


def test_unattempted_excluded_failed_and_awaiting_extraction_remain_distinct():
    sources=[{'source_id':s,'url':f'https://data.example/{s}'} for s in ['new','excluded','failed','readable']]
    sources[1].update(source_role_final='excluded',target_fit_status='unrelated_disease',
                      final_screening_decision='exclude')
    state={'source_registry':sources,'documents':[
        {'source_id':'failed','acquisition_status':'request_error','fetch_status':'failed',
         'is_live_fetched':False,'fetch_error':'Timeout after two attempts','clean_text':''},
        {'source_id':'readable','acquisition_status':'readable','content_readable':True,
         'request_success':True,'fetch_status':'success','parse_status':'parsed','clean_text':'Dengue 12 cases.'}],
        'evidence_chunks':[{'chunk_id':'c','source_id':'readable','extraction_status':'pending'}]}
    rows={row['source_ids'][0]:row for row in build_source_progress({},state)['sources']}
    assert rows['new']['processing_status']=='not_attempted'
    assert rows['excluded']['processing_status']=='screening_excluded'
    assert rows['failed']['processing_status']=='acquisition_failed'
    assert rows['readable']['processing_status']=='awaiting_extraction'


def test_persisted_frontier_reason_is_used_without_fake_document():
    state={'source_registry':[{'source_id':'s','url':'https://data.example/report'}],
      'acquisition_frontier':{'items':[{'source_id':'s','target_id':'https://data.example/report',
        'status':'budget_deferred','reason':'source_targets','attempts':0}]}}
    result=build_source_progress({},state)
    assert result['sources'][0]['processing_status']=='budget_deferred'
    assert result['sources'][0]['processing_reason']=='source_targets'
    assert result['fetch_failed_sources']==0


def test_manifest_does_not_describe_browser_failure_as_budget_exhaustion():
    from data_collection_workflow.result_manifest import build_result_manifest
    state={'source_registry':[{'source_id':'s','url':'https://data.example/page'}],
        'documents':[{'source_id':'s','acquisition_incomplete':True,
            'acquisition_status':'readable','content_readable':True,'clean_text':'Legitimate partial content.',
            'quality_issues':['browser_resources_incomplete']}]}
    result=build_result_manifest({'final_dataset':[]},state)
    assert result['acquisition']['budget_deferred_document_count']==0
    assert result['acquisition']['incomplete_document_count']==1
    assert result['source_progress']['sources'][0]['processing_status']=='acquisition_incomplete'


def test_manifest_includes_deferred_frontier_without_fake_documents():
    from data_collection_workflow.result_manifest import build_result_manifest
    state={'source_registry':[{'source_id':'s','url':'https://data.example/page'}],
        'acquisition_frontier':{'items':[{'source_id':'s','target_id':'https://data.example/page',
            'status':'budget_deferred','reason':'http_requests','attempts':0}]}}
    result=build_result_manifest({'final_dataset':[]},state)
    assert result['acquisition']['budget_deferred_source_count']==1
    assert result['acquisition']['budget_deferred_document_count']==0
    assert result['acquisition']['unresolved_source_ids']==['s']


def test_current_success_supersedes_prior_source_failure_without_losing_audit():
    state={'source_registry':[{'source_id':'s','url':'https://data.example/page'}],
        'documents':[{'source_id':'s','acquisition_status':'request_error','fetch_status':'failed',
            'fetch_error':'First transport timed out','content_readable':False},
            {'source_id':'s','acquisition_status':'readable','fetch_status':'success',
             'content_readable':True,'clean_text':'Dengue 12 cases.'}],
        'evidence_chunks':[{'source_id':'s','chunk_id':'c','extraction_status':'pending'}]}
    result=build_source_progress({},state)
    assert result['fetch_failed_sources']==0
    assert result['sources'][0]['processing_status']=='awaiting_extraction'
    assert result['sources'][0]['prior_acquisition_failures']==1


def test_source_status_exports_and_reports_read_one_manifest(tmp_path):
    import csv
    import json
    from data_collection_workflow.result_manifest import build_result_manifest,write_universal_outputs
    state={'source_registry':[{'source_id':'s','url':'https://data.example/page'}],
        'acquisition_frontier':{'items':[{'source_id':'s','target_id':'https://data.example/page',
            'status':'budget_deferred','reason':'http_requests','attempts':0}]}}
    package={'final_dataset':[], 'source_registry':state['source_registry']}
    package['result_manifest']=build_result_manifest(package,state)
    write_universal_outputs(package,tmp_path)
    expected=package['result_manifest']['source_progress']['sources']
    assert json.loads((tmp_path/'source_processing_status.json').read_text(encoding='utf-8'))==expected
    with (tmp_path/'source_processing_status.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert rows[0]['processing_status']=='budget_deferred'
    assert rows[0]['processing_reason']=='http_requests'
    summary=json.loads((tmp_path/'workflow_console_summary.json').read_text(encoding='utf-8'))
    assert summary['source_progress']['sources']==expected
    registry=json.loads((tmp_path/'source_registry.json').read_text(encoding='utf-8'))
    assert registry[0]['source_id']=='s'
    assert registry[0]['processing_status']=='budget_deferred'
    assert registry[0]['processing_reason']=='http_requests'
    assert 'budget-deferred sources: 1' in (tmp_path/'workflow_console.html').read_text(encoding='utf-8')


def test_finalization_snapshots_live_frontier_when_graph_projection_is_stale(tmp_path,monkeypatch):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_HUMAN_REVIEW','false')
    runtime=RunContext(tmp_path,{'pipeline_mode':'evidence',
        'universal':{'budget_policy':{'version':2,'mode':'adaptive'},
                     'budget_limits':{'http_requests':0}}})
    url='https://data.example/report'
    runtime.frontier.enqueue(target_id=url,url=url,source_id='s',payload={})
    runtime.frontier.claim_next()
    runtime.frontier.finish(url,status='budget_deferred',reason='http_requests')
    state={'structured_task':{'disease':'dengue','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},
        'source_registry':[{'source_id':'s','url':url,'canonical_url':url,'status':'ready_for_content_fetch'}],
        'collection_trace':[], 'normalized_records':[], 'acquisition_frontier':{'items':[]}}
    with runtime.activate():
        result=final_data_package_builder(state)
    progress=result['final_data_package']['result_manifest']['source_progress']
    assert progress['budget_deferred_sources']==1
    assert result['acquisition_frontier']['counts']=={'budget_deferred':1}
    assert result['source_registry'][0]['processing_status']=='budget_deferred'


def test_source_processing_fields_survive_existing_registry_model():
    from data_collection_workflow.models import SourceRegistryEntry
    source={'source_id':'s','url':'https://data.example/report','canonical_url':'https://data.example/report',
        'status':'ready_for_content_fetch','processing_status':'budget_deferred','processing_reason':'http_requests',
        'acquisition_status':'not_fetched','evidence_contribution_status':'none'}
    restored=SourceRegistryEntry(**source).model_dump()
    for name in ('processing_status','processing_reason','acquisition_status','evidence_contribution_status'):
        assert restored.get(name)==source[name]


@pytest.mark.parametrize('status,partial,expected',[
    ('request_error',False,'error'),('budget_exhausted',False,'deferred'),('readable',True,'partial')])
def test_fetch_progress_event_does_not_claim_failed_or_partial_attempt_complete(tmp_path,monkeypatch,status,partial,expected):
    from test_acquisition_queue import setup_node
    from data_collection_workflow.nodes import content_processing as content
    from data_collection_workflow.models import Document
    runtime,state,_=setup_node(monkeypatch,tmp_path)
    state['source_registry']=state['source_registry'][:1]
    emitted=[]
    def transport(request,*args):
        return Document(source_id=request.source_id,url=request.url,canonical_url=request.url,
            acquisition_status=status,fetch_status='failed' if status=='request_error' else 'fetched',
            content_readable=partial,clean_text='Some acquired content.' if partial else '',
            acquisition_incomplete=partial,budget_exhausted_kind='http_requests' if status=='budget_exhausted' else None)
    def progress(node,message,payload=None,**kwargs):
        if payload and payload.get('source_id')=='s1' and 'fetch_status' in payload:
            emitted.append((payload,kwargs))
    monkeypatch.setattr(content,'_fetch_live_document_with_providers',transport)
    monkeypatch.setattr(content,'emit_workflow_progress',progress)
    with runtime.activate():
        content.content_fetch_and_parse(state)
    assert len(emitted)==1
    payload,options=emitted[0]
    assert options['status']==expected
    assert payload['acquisition_status']==status
