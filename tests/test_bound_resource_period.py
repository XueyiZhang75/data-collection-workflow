"""Bound data-resource periods must survive queue construction."""
import copy
import inspect
import socket
import pytest
from data_collection_workflow.resource_discovery import task_resource_candidates, resource_source_entry
from data_collection_workflow.acquisition_scheduling import acquisition_priority
from data_collection_workflow.nodes.source_screening import assess_source_task_fit

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def forbidden(*args,**kwargs):raise AssertionError('External call prohibited')
    monkeypatch.setattr(socket.socket,'connect',forbidden)


def task(disease='Example fever',location='Canada',year=2025):
    return {'structured_task':{'disease':disease,'location':location,'start_date':f'{year}-01-01','end_date':f'{year}-12-31'}}


def child(link,state=None):
    state=state or task();d=state['structured_task']['disease'];loc=state['structured_task']['location']
    link={'href':'https://repository.example/opaque.csv','text':'Download','heading':f'Statistics on {d}, 2025',
          'heading_context':[f'{d} in {loc}, 2025'],'scope_version':2,'anchor_context':'','locator':{'anchor_index':9},**link}
    doc={'source_id':'parent','url':'https://repository.example/catalog','content_readable':True,
         'clean_text':f'{d} in {loc}, 2025','metadata':{'outbound_links':[link]}}
    before=copy.deepcopy(doc);rows=task_resource_candidates(doc,{},state)
    assert len(rows)==1
    kwargs={'parent':{'source_id':'parent'},'depth':1}
    # Exercise the old behavior before introducing the compatible state input.
    if 'state' in inspect.signature(resource_source_entry).parameters:kwargs['state']=state
    entry=resource_source_entry(rows[0],**kwargs)
    assert doc==before and entry['resource_link_provenance']==link
    return entry


@pytest.mark.parametrize('year,language,code',[
    (year,language,code)
    for year in range(2009,2025)
    for language,code in [('English','en'),('Traditional Chinese','tc')]
    if year<2024 or code=='en'
])
def test_annual_statistics_row_period_survives_entry_creation(year,language,code):
    state=task(location='Example Region')
    heading=f'Statistics on Example fever, {year} ({language})'
    context=f'Select Item {heading} CSV Details Download'
    entry=child({
        'href':f'https://repository.example/files/statistics_{year}_{code}.csv',
        'heading':heading,
        'heading_context':['Statistics on Example fever','Data Resources',heading],
        'context':context,
        'anchor_context':context,
        'page_topic':'Statistics on Example fever | Example Data Catalog',
        'anchor_present':True,
        'navigation':False,
    },state)
    assert entry['date_fit']=='mismatch'
    assert entry['target_verification_status']=='temporal_mismatch'
    assert 'Statistics on Example fever,' in entry['snippet']
    assert entry['source_role_final']!='excluded'
    assert not entry.get('blocked_from_fetch')
    assert acquisition_priority(entry,state=state)<0


@pytest.mark.parametrize('disease,location',[('Measles','Canada'),('Dengue','Brazil'),('Example fever','Example Region')])
@pytest.mark.parametrize('language',['English','Traditional Chinese'])
def test_old_statistics_are_low_priority_but_retained(disease,location,language):
    state=task(disease,location)
    entry=child({'heading':f'Statistics on {disease}, 2009 ({language})','anchor_context':f'Statistics on {disease}, 2009 ({language}) CSV Details Download'},state)
    assert entry['date_fit']=='mismatch'
    assert entry['ready_for_content_fetch'] and not entry.get('blocked_from_fetch')
    assert acquisition_priority(entry,state=state)<0


@pytest.mark.parametrize('text,expected',[
    ('Statistics on Example fever, 2024–2025','match'),
    ('Statistics on Example fever, 2024 and 2025','match'),
    ('2025 report comparing Example fever case counts in 2024 and 2025','match'),
    ('Example fever data paper published in 2024','candidate'),
    ('Example fever data paper was published in 2024','candidate'),
    ('Example fever observations in 2024, published in 2025','mismatch'),
    ('Example fever observations in 2025, published in 2026','match'),
    ('Statistics on Example fever','candidate'),
])
def test_bound_period_contract_distinguishes_observation_and_publication(text,expected):
    entry=child({'heading':text,'anchor_context':text})
    assert entry['date_fit']==expected
    if expected!='mismatch':assert entry['target_verification_status']!='temporal_mismatch'


def test_task_period_swap_makes_same_annual_resource_relevant():
    link={'heading':'Statistics on Example fever, 2009','anchor_context':'Statistics on Example fever, 2009 CSV Download'}
    old=child(link,task(year=2025));matching=child(link,task(year=2009))
    assert old['date_fit']=='mismatch' and matching['date_fit']=='match'
    assert old['source_id']==matching['source_id']
    assert acquisition_priority(old,state=task(year=2025))<acquisition_priority(matching,state=task(year=2009))


def test_filename_year_alone_is_not_a_resource_observation_period():
    entry=child({'href':'https://repository.example/2024.csv','heading':'Underlying data','anchor_context':'','text':''})
    assert entry['date_fit']=='candidate'
    assert entry['target_verification_status']!='temporal_mismatch'


def test_child_current_statistics_override_older_catalog_heading():
    entry=child({'heading_context':['Example fever 2009 archive','Statistics on Example fever, 2025'],
                 'heading':'Statistics on Example fever, 2025','anchor_context':'Download'})
    assert entry['date_fit']=='match'


def test_multicountry_current_comparison_remains_available():
    entry=child({'heading':'Example fever statistics in Canada and Brazil, 2025','anchor_context':'Canada and Brazil: Example fever observations in 2024 and 2025'})
    assert entry['date_fit']=='match' and not entry.get('blocked_from_fetch')


def test_language_forms_remain_separate_linked_resources():
    en=child({'href':'https://repository.example/data-en.csv','heading':'Statistics on Example fever, 2025 (English)'})
    fr=child({'href':'https://repository.example/data-fr.csv','heading':'Statistics on Example fever, 2025 (French)'})
    assert en['source_id']!=fr['source_id']
    assert en['parent_source_id']==fr['parent_source_id']=='parent'
    assert en['date_fit']==fr['date_fit']=='match'


def test_current_source_is_fetched_before_explicit_old_link(tmp_path,monkeypatch):
    import requests
    from test_acquisition_queue import setup_node
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    runtime,state,visits=setup_node(monkeypatch,tmp_path,targets=2)
    parent,other=[r['url'] for r in state['source_registry']]
    state['source_registry'][0].update(target_verification_status='verified_target',data_product_type='surveillance_report')
    state['source_registry'][1].update(target_verification_status='unverified_candidate',data_product_type='surveillance_report')
    html='<h1>Example fever Canada, 2025</h1><h2>Statistics on Example fever, 2009 (English)</h2><p><a href="/archive.csv">Download</a></p>'
    class Response:
        status_code=200
        def __init__(self,url):self.url=url;self.headers={'content-type':'text/html' if url==parent else 'text/plain'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,size):yield (html if self.url==parent else 'Example fever Canada during 2025: 12 confirmed cases.').encode()
    def get(url,**kwargs):visits.append(url);return Response(url)
    monkeypatch.setattr(requests,'get',get)
    with runtime.activate():result=content_fetch_and_parse(state)
    assert visits==[parent,other]
    historical=next(x for x in result['source_registry'] if x['url'].endswith('/archive.csv'))
    assert historical['date_fit']=='mismatch'
    pending=next(x for x in result['acquisition_frontier']['items'] if x['url'].endswith('/archive.csv'))
    assert pending['status']=='budget_deferred' and pending['attempts']==0
