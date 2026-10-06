"""Adaptive queue recovery uses current-session artifacts, never historical input."""
import requests
from test_evidence_resource_discovery import _env,_source
from test_acquisition_transport import _adaptive_runtime
from data_collection_workflow.nodes.content_processing import content_fetch_and_parse


def setup_node(monkeypatch,tmp_path,*,targets=3):
    _env(monkeypatch)
    visits=[]
    class Response:
        status_code=200;headers={'content-type':'text/plain'}
        def __init__(self,url):self.url=url
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield b'Example fever Canada during 2025: 12 confirmed cases.'
    def get(url,**kwargs):visits.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    runtime=_adaptive_runtime(tmp_path,targets=targets,requests=10)
    state={'structured_task':{'disease':'Example fever','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},
        'source_registry':[_source('https://data.example/one',1),_source('https://other.example/two',2)],'collection_trace':[]}
    return runtime,state,visits


def test_node_restart_recovers_completed_documents_without_network_or_duplicates(tmp_path,monkeypatch):
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    with runtime.activate():first=content_fetch_and_parse(state)
    assert len(visits)==2 and len(first['documents'])==2
    # Simulate the node returning no checkpoint after acquisition was persisted.
    with runtime.activate():second=content_fetch_and_parse(state)
    assert len(visits)==2 and len(second['documents'])==2
    assert {doc['content_hash'] for doc in second['documents']}=={doc['content_hash'] for doc in first['documents']}
    assert runtime.ledger.snapshot()['used']['source_targets']==2


def test_denied_target_retained_in_frontier_without_fabricated_document(tmp_path,monkeypatch):
    runtime,state,visits=setup_node(monkeypatch,tmp_path,targets=1)
    with runtime.activate():result=content_fetch_and_parse(state)
    assert len(visits)==1 and len(result['documents'])==1
    jobs=result['acquisition_frontier']['items']
    deferred=[job for job in jobs if job['status']=='budget_deferred']
    assert len(deferred)==1 and deferred[0]['attempts']==0
    assert deferred[0]['reason']=='source_targets'


def test_one_source_transport_error_does_not_abort_remaining_queue(tmp_path,monkeypatch):
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    original=requests.get
    def get(url,**kwargs):
        if url.endswith('/one'):raise requests.ConnectionError('simulated unreachable source')
        return original(url,**kwargs)
    monkeypatch.setattr(requests,'get',get)
    with runtime.activate():result=content_fetch_and_parse(state)
    assert len([doc for doc in result['documents'] if doc['content_readable']])==1
    assert result['acquisition_frontier']['counts']['completed']==1
    assert result['acquisition_frontier']['counts']['failed']==1


def test_restart_expands_saved_page_after_crash_before_link_discovery(tmp_path,monkeypatch):
    import pytest
    from data_collection_workflow import resource_discovery
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    state['source_registry']=state['source_registry'][:1]
    parent=state['source_registry'][0]['url']
    child='https://data.example/counts.csv'
    class Response:
        status_code=200
        def __init__(self,url):self.url=url;self.headers={'content-type':'text/csv' if url==child else 'text/html'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):
            yield (b'year,cases\n2025,12\n' if self.url==child else
                b'<h1>Example fever Canada 2025</h1><p>Data availability <a href="/counts.csv">Download case counts</a></p>')
    def get(url,**kwargs):visits.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    discover=resource_discovery.task_resource_candidates
    def crash(*args,**kwargs):raise SystemExit('between content persistence and link expansion')
    monkeypatch.setattr(resource_discovery,'task_resource_candidates',crash)
    with pytest.raises(SystemExit),runtime.activate():content_fetch_and_parse(state)
    monkeypatch.setattr(resource_discovery,'task_resource_candidates',discover)
    with runtime.activate():result=content_fetch_and_parse(state)
    assert visits==[parent,child]
    assert len(result['documents'])==2
    assert any(row.get('url')==child for row in result['source_registry'])


def test_pending_job_rechecks_current_source_exclusion_before_dispatch(tmp_path,monkeypatch):
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    entry=state['source_registry'][0]
    runtime.frontier.enqueue(target_id=entry['url'],url=entry['url'],source_id=entry['source_id'],priority=100,
        payload={'entry':dict(entry),'request':{'source_id':entry['source_id'],'url':entry['url'],
            'canonical_url':entry['url'],'final_screening_decision':'include_for_content_fetch','fetch_purpose':'data_extraction'}})
    entry.update(blocked_from_fetch=True,blocked_from_fetch_reason='user_excluded',
        source_role_final='excluded',final_screening_decision='exclude')
    with runtime.activate():result=content_fetch_and_parse(state)
    assert visits==[state['source_registry'][1]['url']]
    blocked=next(row for row in result['acquisition_frontier']['items'] if row['source_id']==entry['source_id'])
    assert blocked['status']=='blocked' and blocked['attempts']==0


def test_refetch_consumer_preserves_existing_document_locator_identity(tmp_path,monkeypatch):
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    with runtime.activate():
        first=content_fetch_and_parse(state)
    first['documents'][0]['document_id']='stable-original-document'
    first_hash=first['documents'][0]['content_hash']
    first_source=first['documents'][0]['source_id']
    with runtime.activate():
        second=content_fetch_and_parse({**state,**first})
    retained=[doc for doc in second['documents'] if doc['source_id']==first_source
              and doc['content_hash']==first_hash]
    assert len(retained)==1
    assert retained[0].get('document_id')=='stable-original-document'
    assert len(visits)==2, 'Keeping a locator identity must not cause another network call.'
