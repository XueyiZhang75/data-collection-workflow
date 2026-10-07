"""Isolated evaluation regressions for explicitly labelled resource icons."""
import copy
import socket
from urllib.parse import urlencode
import pytest
from data_collection_workflow.resource_discovery import task_resource_candidates

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    def forbidden(*args, **kwargs):
        raise AssertionError('No network in resource-label regression')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def select(link, disease='Example fever', location='Example Region'):
    state={'structured_task':{'disease':disease,'location':location,'start_date':'2025-01-01','end_date':'2025-12-31'}}
    supplied={'href':'https://repository.example/record/123', 'text':'', 'title':'', 'aria_label':'',
              'heading':'Data Availability','heading_context':[f'{disease} in {location}, 2025','Data Availability'],
              'scope_version':2,'navigation':False,'locator':{'anchor_index':7,'source_line':20},**link}
    doc={'source_id':'parent','url':'https://report.example/article','content_readable':True,
         'clean_text':f'{disease} in {location}, 2025','metadata':{'outbound_links':[supplied]}}
    before=copy.deepcopy(doc)
    result=task_resource_candidates(doc,{},state)
    assert doc==before
    for row in result:
        assert row['provenance']==supplied
        assert row['url']==supplied['href']
    return result


@pytest.mark.parametrize('version,event',[(2,'epidemic'),(1,'outbreak')])
def test_empty_share_icons_cannot_borrow_data_heading(version,event):
    parent=f'https://report.example/article/v{version}'
    title=f'The 2024-2025 Example fever {event} in Example Region'
    share_query=urlencode({'mini':'true','url':parent,'title':title,'summary':'','source':'Example Reports'})
    link={
        'href':f'https://www.linkedin.com/shareArticle?{share_query}',
        'text':'',
        'title':'Publish this post to LinkedIn',
        'aria_label':'',
        'heading':'Data Availability',
        'heading_context':[title,'Data Availability'],
        'anchor_context':'',
        'context':'',
        'anchor_present':True,
        'download':False,
        'navigation':False,
        'rel':['nofollow'],
        'scope_version':2,
        'locator':{'anchor_index':7,'source_line':20},
    }
    original=copy.deepcopy(link)
    state={'structured_task':{'disease':'Example fever','location':'Example Region',
                             'start_date':'2025-01-01','end_date':'2025-12-31'}}
    doc={'source_id':'parent','url':parent,'content_readable':True,
         'metadata':{'outbound_links':[link]}}
    assert task_resource_candidates(doc,{},state)==[]
    assert link==original


@pytest.mark.parametrize('disease,location',[('Measles','Canada'),('Dengue','Brazil'),('Example fever','Example Region')])
@pytest.mark.parametrize('field',['title','aria_label'])
@pytest.mark.parametrize('visible',['','here'])
def test_explicit_control_label_is_not_an_unlabelled_deictic(disease,location,field,visible):
    assert select({'text':visible,field:'Publish this post to Example Network'},disease,location)==[]


@pytest.mark.parametrize('link',[
    {'title':'Download underlying data'},
    {'aria_label':'Download underlying data'},
    {'title':'Partager les données','download':True},
    {'title':'Download','download':True},
    {'title':'Supplemental observations','type':'application/pdf'},
    {'title':'Results','type':'text/csv'},
    {'aria_label':'Underlying data','href':'https://repository.example/share/access-token'},
    {'title':'Underlying data','href':'https://repository.example/sharing/access-token'},
    {'text':'We share data with researchers','href':'https://repository.example/share/access-token'},
    {'text':'here'},
    {'text':'here','title':'here','aria_label':'here'},
    {},
])
def test_scoped_data_icons_and_real_repository_links_remain_candidates(link):
    assert len(select(link))==1


@pytest.mark.parametrize('field',['title','aria_label'])
def test_unknown_label_keeps_scoped_candidate_for_verification(field):
    assert len(select({field:'Unknown purpose'}))==1


@pytest.mark.parametrize('label',['External link','Opens in a new tab','Download location is provided here'])
@pytest.mark.parametrize('field',['title','aria_label'])
def test_neutral_access_label_keeps_scoped_data_relationship(label,field):
    assert len(select({field:label}))==1


@pytest.mark.parametrize('link',[
    {'title':'We publish this article to describe disease surveillance data'},
    {'title':'Publish dataset to Repository Service','download':True},
    {'aria_label':'Repository for data shared with researchers'},
])
def test_publication_prose_or_dataset_download_is_not_a_sharing_widget(link):
    assert len(select(link))==1
