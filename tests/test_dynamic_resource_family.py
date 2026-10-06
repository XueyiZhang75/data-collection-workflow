"""Session-local completed content must update the whole pending family order."""
import copy
import io
import socket
import pytest
import requests
from test_acquisition_queue import setup_node
from test_evidence_resource_discovery import _source
from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
from data_collection_workflow.acquisition_scheduling import acquisition_priority, rerank_frontier

@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Only fixture HTTP responses are permitted')
    monkeypatch.setattr(socket.socket,'connect',forbidden)


def run_pages(monkeypatch,tmp_path,*,disease='Example fever',country='Canada',targets=5,links=None,pages=None,extra_sources=None,soft=50,parent_group=None):
    runtime,state,visits=setup_node(monkeypatch,tmp_path,targets=targets)
    runtime.config['universal']['budget_policy']['soft_source_target']=soft
    state['structured_task'].update(disease=disease,location=country)
    parent,independent=[r['url'] for r in state['source_registry']]
    for index,entry in enumerate(state['source_registry']):
        entry.update(title=f'{disease} surveillance data, {country} 2025',
                     target_verification_status='verified_target' if index==0 else 'unverified_candidate',
                     data_product_type='surveillance_report')
    if parent_group:state['source_registry'][0]['source_independence_group']=parent_group
    if extra_sources:state['source_registry'].extend(extra_sources)
    links=links if links is not None else [(f'/series-{i}.csv','Download case counts') for i in range(3)]
    html=f'<h1>{disease} data in {country}, 2025</h1><h2>Data availability</h2><p>{disease} case counts in {country} during 2025 are available below.</p>'
    html+=''.join(f'<p><a href="{href}">{label}</a></p>' for href,label in links)
    default_csv=f'disease,country,year,cases\n{disease},{country},2025,12\n'.encode()
    supplied={parent:('text/html',html.encode()),independent:('text/plain',f'{disease} in {country} during 2025: 19 confirmed cases.'.encode()),**(pages or {})}
    class Response:
        status_code=200
        def __init__(self,url):
            self.url=url;kind,self.body=supplied.get(url,('text/csv',default_csv));self.headers={'content-type':kind}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield self.body
    def get(url,**kwargs):visits.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    claims=[];claim=runtime.frontier.claim_next
    def record_claim(**kwargs):
        row=claim(**kwargs)
        if row:claims.append(copy.deepcopy(row))
        return row
    monkeypatch.setattr(runtime.frontier,'claim_next',record_claim)
    with runtime.activate():result=content_fetch_and_parse(state)
    return runtime,state,visits,claims,result


@pytest.mark.parametrize('disease,country',[('Example fever','Canada'),('Measles','Brazil'),('Dengue','India')])
def test_new_same_family_resources_do_not_receive_a_new_family_bonus(monkeypatch,tmp_path,disease,country):
    runtime,state,visits,claims,result=run_pages(monkeypatch,tmp_path,disease=disease,country=country,targets=3)
    parent,independent=[r['url'] for r in state['source_registry'][:2]]
    assert visits[0]==parent
    assert independent in visits
    assert sum('/series-' in url for url in visits)==1
    assert runtime.ledger.snapshot()['used']['source_targets']==3
    assert sum(r['status']=='budget_deferred' for r in result['acquisition_frontier']['items'])==2


def test_queued_siblings_are_reassessed_after_each_completed_target(monkeypatch,tmp_path):
    _,_,visits,claims,_=run_pages(monkeypatch,tmp_path)
    children=[r for r in claims if '/series-' in r['url']]
    assert len(children)==3
    assert children[0]['priority']>children[1]['priority']>children[2]['priority']
    assert len(visits)==5  # all distinct child data remain retrievable


def test_new_unknown_repository_family_remains_available_before_repeated_family(monkeypatch,tmp_path):
    independent_repo='https://unknown-repository.example/data.csv'
    _,_,visits,_,result=run_pages(monkeypatch,tmp_path,targets=4,links=[('/series-0.csv','Download case counts'),(independent_repo,'Download case counts')])
    assert visits.index(independent_repo)<visits.index('https://data.example/series-0.csv')
    assert len(visits)==4 and len(result['documents'])==4


def test_navigation_and_distinct_pdf_table_are_not_removed_or_merged(monkeypatch,tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    pdf=io.BytesIO();canvas=Canvas(pdf);canvas.drawString(30,760,'Example fever Canada 2025: 31 confirmed cases.');canvas.save()
    nav='https://data.example/series';pdf_url='https://data.example/unique-table.pdf';csv_url='https://data.example/counts.csv'
    pages={nav:('text/html',b'<h1>Example fever Canada 2025 surveillance data</h1><p>Data availability <a href="/counts.csv">Download case counts</a></p>'),
           pdf_url:('application/pdf',pdf.getvalue())}
    _,_,visits,_,result=run_pages(monkeypatch,tmp_path,targets=5,links=[('/series','Example fever surveillance data'),('/unique-table.pdf','Download supplementary case table')],pages=pages)
    assert {nav,pdf_url,csv_url}.issubset(visits)
    docs={r['url']:r for r in result['documents']}
    assert '31 confirmed cases' in docs[pdf_url]['clean_text']
    assert docs[pdf_url]['content_hash']!=docs[csv_url]['content_hash']
    assert len(result['documents'])==5


def test_dynamic_family_order_is_retained_after_node_restart_without_new_charges(monkeypatch,tmp_path):
    runtime,state,visits,_,first=run_pages(monkeypatch,tmp_path)
    before=list(visits);used=runtime.ledger.snapshot()['used']
    with runtime.activate():second=content_fetch_and_parse(state)
    assert visits==before and runtime.ledger.snapshot()['used']==used
    assert {d['content_hash'] for d in second['documents']}=={d['content_hash'] for d in first['documents']}
    assert len(second['documents'])==len(first['documents'])


def test_repeated_rerank_recomputes_counts_without_cumulative_alias_inflation(monkeypatch,tmp_path):
    runtime,state,_=setup_node(monkeypatch,tmp_path)
    source=state['source_registry'][0];target='https://data.example/pending'
    child={**_source(target,7),'data_product_type':'surveillance_report'}
    registry=[source,child]
    runtime.frontier.enqueue(target_id=target,url=target,source_id=child['source_id'],payload={'entry':child})
    docs=[{'source_id':source['source_id'],'content_hash':'same-version','content_readable':True}]
    scores=[]
    for _ in range(6):
        rerank_frontier(runtime,registry,docs,state=state)
        scores.append(runtime.frontier.snapshot()['items'][0]['priority'])
    assert len(set(scores))==1
    # Existing policy counts readable document rows, not independent works.
    # Repeated reranking itself must not accumulate this duplicated input.
    duplicate_scores=[]
    for _ in range(6):
        rerank_frontier(runtime,registry,docs+docs,state=state)
        duplicate_scores.append(runtime.frontier.snapshot()['items'][0]['priority'])
    assert len(set(duplicate_scores))==1
    assert duplicate_scores[0]==scores[0]-5


def test_failed_content_does_not_claim_a_readable_family(monkeypatch,tmp_path):
    runtime,state,_=setup_node(monkeypatch,tmp_path)
    source=state['source_registry'][0];target='https://data.example/pending'
    child={**_source(target,7),'data_product_type':'surveillance_report'}
    runtime.frontier.enqueue(target_id=target,url=target,source_id=child['source_id'],payload={'entry':child})
    rerank_frontier(runtime,[source,child],[{'source_id':source['source_id'],'content_readable':False}],state=state)
    assert runtime.frontier.snapshot()['items'][0]['priority']==acquisition_priority(child,state=state)


def test_soft_checkpoint_reuses_the_single_post_document_rerank(monkeypatch,tmp_path):
    from data_collection_workflow import acquisition_scheduling
    actual=acquisition_scheduling.rerank_frontier;calls=[]
    def tracked(runtime,registry,documents,**kwargs):
        calls.append(len(documents))
        return actual(runtime,registry,documents,**kwargs)
    monkeypatch.setattr(acquisition_scheduling,'rerank_frontier',tracked)
    _,_,visits,_,result=run_pages(monkeypatch,tmp_path,soft=1)
    assert len(visits)==5 and len(result['documents'])==5
    assert calls==[0,1,2,3,4,5]


def test_known_publisher_parent_and_unknown_children_update_after_first_child(monkeypatch,tmp_path):
    _,state,visits,claims,_=run_pages(monkeypatch,tmp_path,targets=4,parent_group='publisher:fixture')
    independent=state['source_registry'][1]['url']
    assert independent in visits
    assert sum('/series-' in url for url in visits)==2
    children=[r for r in claims if '/series-' in r['url']]
    assert children[0]['priority']>children[1]['priority']>children[2]['priority']


def test_post_fetch_group_enrichment_and_amended_resume_use_current_registry(monkeypatch,tmp_path):
    from data_collection_workflow.acquisition_scheduling import source_family
    runtime,state,visits,claims,first=run_pages(monkeypatch,tmp_path,targets=4,parent_group='publisher:fixture')
    children=[r for r in first['source_registry'] if r.get('discovery_method')=='task_resource_link']
    assert len(children)==3 and all(r.get('source_independence_group') for r in children)
    assert len({source_family(r) for r in children})==1
    # Real post-fetch enrichment has populated both completed and pending rows;
    # old queued discovery payloads still had no publisher identity.
    child_claims=[r for r in claims if '/series-' in r['url']]
    assert all(not r['payload']['entry'].get('source_independence_group') for r in child_claims)
    before=list(visits)
    runtime.amend_budget(amendment_id='fixture-more-data',increases={'source_targets':5},reason='offline continuation control')
    with runtime.activate():second=content_fetch_and_parse({**state,**first})
    assert len(visits)==len(before)+1 and visits[:len(before)]==before
    assert len(second['documents'])==5
    assert runtime.ledger.snapshot()['used']['source_targets']==5
    resumed=claims[-1]
    assert resumed['priority']<acquisition_priority(resumed['payload']['entry'],state=state)
