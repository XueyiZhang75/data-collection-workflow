"""Independent offline controls for resource observation-period metadata."""
import copy
import inspect
import socket
import pytest
from data_collection_workflow.resource_discovery import resource_source_entry
from data_collection_workflow.acquisition_scheduling import acquisition_priority

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def deny(*args,**kwargs):raise AssertionError('No external calls')
    monkeypatch.setattr(socket.socket,'connect',deny)

def make(heading='',context='',label='Download',disease='measles',region='Canada',year=2025,**extra):
    state={'structured_task':{'disease':disease,'location':region,'start_date':f'{year}-01-01','end_date':f'{year}-12-31'}}
    candidate={'url':'https://unverified.example/data.csv','resource_type':'csv','link_text':label,'score':60,
               'selection_reasons':['scoped_data_availability_link'],
               'provenance':{'heading':heading,'anchor_context':context,'text':label,**extra}}
    parent={'source_id':'p','title':f'{disease} {region} 2025','target_verification_status':'verified_target'}
    before=copy.deepcopy((candidate,parent,state))
    kwargs={'parent':parent,'depth':1}
    if 'state' in inspect.signature(resource_source_entry).parameters:kwargs['state']=state
    entry=resource_source_entry(candidate,**kwargs)
    assert (candidate,parent,state)==before
    return entry,state

@pytest.mark.parametrize('disease,region',[('measles','Canada'),('dengue','Brazil'),('Example syndrome','Unknown region')])
@pytest.mark.parametrize('text,expected',[
    ('Statistics on DISEASE, 2018','mismatch'),
    ('DISEASE observations, 2018-2026','match'),
    ('DISEASE cases in 2018 and 2025','match'),
    ('DISEASE publication published in 2018','candidate'),
    ('DISEASE cases in 2018. Published in 2025','mismatch'),
    ('DISEASE cases in 2025. Published in 2026','match'),
])
def test_periods_are_observational_and_task_generic(disease,region,text,expected):
    e,s=make(text.replace('DISEASE',disease),disease=disease,region=region)
    assert e['date_fit']==expected
    assert e['source_role_final']!='excluded' and e['ready_for_content_fetch']
    assert not e.get('blocked_from_fetch')
    assert e['disease_fit']=='candidate' and e['geography_fit']=='candidate'
    if expected=='mismatch':
        unknown,_=make('Underlying data',disease=disease,region=region)
        assert acquisition_priority(e,state=s)<acquisition_priority(unknown,state=s)

@pytest.mark.parametrize('metadata',['Published in 2018','Updated in 2018','Reviewed in 2018','Published online in 2018','Copyright 2018'])
def test_publication_metadata_never_creates_observation_year(metadata):
    e,_=make('Measles data paper: '+metadata)
    assert e['date_fit']=='candidate'

@pytest.mark.parametrize('heading,context',[
    ('Underlying data',''),
    ('',''),
    ('未知資料，版本 2018',''),
    ('Measles figures','Agency founded in 2018'),
])
def test_unknown_content_remains_available(heading,context):
    e,_=make(heading,context,label='Open dataset')
    assert e['date_fit']=='candidate'
    assert e['ready_for_content_fetch'] and not e.get('blocked_from_fetch')


def test_parent_year_and_location_cannot_be_inherited():
    e,_=make('', '', '',heading_context=['Measles Canada 2018'])
    assert e['date_fit']==e['disease_fit']==e['geography_fit']=='candidate'
    assert e['snippet']==''


def test_local_current_heading_can_survive_archival_ancestor():
    e,_=make('Measles cases, 2025','',heading_context=['Measles 2018 archive','Measles cases, 2025'])
    assert e['date_fit']=='match'


def test_open_current_counts_do_not_prove_a_closed_historical_period():
    e,_=make('Measles outbreak began in 2018 and currently has 12 cases')
    assert e['date_fit']=='candidate'


def test_task_year_change_preserves_identity_and_changes_ranking():
    e,s=make('Measles cases, 2018');matched,ms=make('Measles cases, 2018',year=2018)
    assert e['source_id']==matched['source_id']
    assert e['date_fit']=='mismatch' and matched['date_fit']=='match'
    assert acquisition_priority(e,state=s)<acquisition_priority(matched,state=ms)
