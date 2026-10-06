"""Independent observation-role controls; no network or production changes."""
import hashlib
import socket
import pytest
from data_collection_workflow.source_assertions import observation_dates,typed_date_support
from data_collection_workflow.nodes.extraction import _official_geography
from data_collection_workflow.evidence_qualification import assess_record_evidence

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def deny(*args,**kwargs):raise AssertionError('No external calls')
    monkeypatch.setattr(socket.socket,'connect',deny)


def qualification(body,heading='',country='Canada',disease='measles',bound=True,**fields):
    full=heading+'\n'+body if heading else body
    start=len(heading)+1 if heading else 0
    sha=hashlib.sha256(full.encode()).hexdigest()
    spans=[dict(role='heading',quote=heading,char_start=0,char_end=len(heading),heading_level=1,section_id='h')] if heading else []
    doc=dict(source_id='s',document_id='d',clean_text=full,content_hash=sha,text_hash=sha,parser_version='independent/1',locator_spans=spans)
    chunk=dict(source_id='s',chunk_id='c',text=body,document_hash=sha,char_start=start,char_end=len(full),bound_context_spans=spans if bound else [])
    r=dict(record_id='r',source_id='s',supporting_chunk_id='c',disease=disease,country=country,cases_unspecified=76,**fields)
    return assess_record_evidence(r,contract={},evidence_index={'documents':[doc],'evidence_chunks':{'c':chunk}})

@pytest.mark.parametrize('prefix',['Depuis le','Since the','From the'])
def test_explicit_start_article_never_asof(prefix):
    text=prefix+' 1 January 2025, Canada recorded 76 measles cases.'
    got=observation_dates(text)
    assert got.get('metric_period_start')=='2025-01-01'
    assert not got.get('as_of_date')

@pytest.mark.parametrize('prefix',['Until the','Through the',"Jusqu’au"])
def test_explicit_end_article_never_asof(prefix):
    got=observation_dates(prefix+' 1 January 2025, Canada recorded 76 measles cases.')
    assert got.get('metric_period_end')=='2025-01-01'
    assert not got.get('as_of_date')

@pytest.mark.parametrize('prefix',['As of','Au','Le'])
def test_snapshot_date_remains_asof(prefix):
    assert observation_dates(prefix+' 1 January 2025, Canada recorded 76 measles cases.')['as_of_date']=='2025-01-01'

@pytest.mark.parametrize('text',[
    'Rapport du 1 janvier 2025 : 76 cas de rougeole au Canada.',
    'Bulletin du 1 janvier 2025 : 76 cas de rougeole au Canada.',
    'Report from January 1, 2025: Canada has 76 measles cases.',
])
def test_report_date_is_not_observation_period_start(text):
    assert not observation_dates(text).get('metric_period_start')

@pytest.mark.parametrize('body,expected',[
    ('On board MV Aurora, 76 measles cases were reported during 2025.','vessel'),
    ('On the cruise ship, 76 measles cases were reported during 2025.','travel_associated'),
    ('The Region of the Americas reported 76 measles cases during 2025.','region'),
])
def test_explicit_nonnational_geography_still_parses(body,expected):
    got=_official_geography(body,'',{'structured_task':{'location':'global'}})
    assert got.get('geographic_scope_type')==expected,got
    assert not got.get('country')

@pytest.mark.parametrize('origin,observed,disease',[('Brazil','Canada','measles'),('La Réunion','France','chikungunya'),('India','Peru','dengue')])
@pytest.mark.parametrize('role',['originating in','venaient de'])
def test_origin_cannot_claim_count_but_observation_heading_can(origin,observed,disease,role):
    body=f'During 2025, 76 {disease} cases were identified, 97% {role} {origin}.'
    bad=qualification(body,f'{disease} cases in {origin}',country=origin,disease=disease,reporting_period='2025')
    assert bad.status=='candidate'
    assert any(f.field=='cases_unspecified' and not f.supported for f in bad.field_evidence)
    good=qualification(body,f'{disease} cases in {observed}',country=observed,disease=disease,reporting_period='2025')
    assert good.status=='qualified',good.reasons


def test_unbound_heading_cannot_supply_observation_country():
    got=qualification('During 2025, 76 measles cases were reported.','Measles Canada',bound=False,reporting_period='2025')
    assert got.status=='candidate'


def test_start_role_count_supports_observation_place_only():
    body='Depuis le 1 janvier 2025, 76 measles cases were identified, 97% venaient de Brazil.'
    good=qualification(body,'Imported measles cases in Canada',metric_period_start='2025-01-01')
    assert good.status=='qualified',good.reasons
    bad=qualification(body,'Imported measles cases in Canada',country='Brazil',as_of_date='2025-01-01')
    assert bad.status=='candidate'
    assert any(f.field=='as_of_date' and not f.supported for f in bad.field_evidence)


def test_legacy_geography_branch_preserved(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','legacy_v2')
    got=_official_geography('76 cases, 97% venaient de Canada.','')
    assert got.get('country')=='Canada'
