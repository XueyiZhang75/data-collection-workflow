"""Recovery acquisition must preserve pending work and its chosen transport."""
import pytest
from test_acquisition_queue import setup_node
from data_collection_workflow.nodes import content_processing as content


def test_adaptive_node_walltime_yields_pending_sources_without_budget_amendment(tmp_path,monkeypatch):
    import requests
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    monkeypatch.setenv('FETCH_MAX_NODE_SECONDS','1')
    elapsed=[0.0]
    monkeypatch.setattr(content,'monotonic',lambda:elapsed[0])
    original=requests.get
    def get(*args,**kwargs):
        response=original(*args,**kwargs)
        elapsed[0]+=2.0
        return response
    monkeypatch.setattr(requests,'get',get)
    with runtime.activate():
        first=content.content_fetch_and_parse(state)
    assert len(visits)==1
    assert first['acquisition_frontier']['counts']=={'completed':1,'pending':1}
    pending=next(row for row in first['acquisition_frontier']['items'] if row['status']=='pending')
    assert pending['attempts']==0
    assert not runtime.ledger.snapshot()['budget_amendments']
    with runtime.activate():
        second=content.content_fetch_and_parse({**state,**first})
    assert len(visits)==2
    assert len(second['documents'])==2
    assert second['acquisition_frontier']['counts']=={'completed':2}
    assert runtime.ledger.snapshot()['used']['http_requests']==2
    assert runtime.ledger.snapshot()['used']['source_targets']==2


@pytest.mark.parametrize('configured_strategy',['native','browser'])
def test_recovery_source_transport_choice_reaches_acquisition_consumer(tmp_path,monkeypatch,configured_strategy):
    from data_collection_workflow import document_acquisition as acquisition
    runtime,state,visits=setup_node(monkeypatch,tmp_path)
    runtime.config['universal']['acquisition']={'acquisition_strategy':configured_strategy}
    state['source_registry'][0]['acquisition_strategy']='browser'
    directions={}
    def acquire(url,*,source_id,session_dir,config,priority_context=None):
        directions[source_id]=config.get('acquisition_strategy')
        return acquisition.parse_response(b'Example fever Canada 2025: 12 confirmed cases.',
            url=url,source_id=source_id,session_dir=session_dir,content_type='text/plain')
    monkeypatch.setattr(acquisition,'acquire_document',acquire)
    with runtime.activate():
        content.content_fetch_and_parse(state)
    assert directions=={'s1':'browser','s2':configured_strategy}
    assert {row['source_id']:row['active_strategy'] for row in runtime.frontier.snapshot()['items']}==directions
