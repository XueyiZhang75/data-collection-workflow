"""Independent HTML-to-resource-candidate controls; no external calls."""
import hashlib
import pytest
from bs4 import BeautifulSoup
from data_collection_workflow.resource_discovery import html_resource_links,task_resource_candidates


def candidates(anchors):
 html='<title>Dengue in Brazil, 2025</title><h1>Dengue in Brazil, 2025</h1><h2>Data Availability</h2>'+anchors
 links=html_resource_links(BeautifulSoup(html,'html.parser'),source_url='https://report.example/article',content_hash=hashlib.sha256(html.encode()).hexdigest())
 doc={'source_id':'parent','url':'https://report.example/article','content_readable':True,'metadata':{'outbound_links':links}}
 state={'structured_task':{'disease':'Dengue','location':'Brazil','start_date':'2025-01-01','end_date':'2025-12-31'}}
 return links,task_resource_candidates(doc,{},state)


@pytest.mark.parametrize('attrs,text',[
 ('title="Publish this post to Network"',''),
 ('aria-label="Publish this post to Network"',''),
])
def test_parsed_explicit_labels_cannot_inherit_empty_deictic_scope(attrs,text):
 links,rows=candidates('<p><a href="https://repo.example/record/123" '+attrs+'>'+text+'<svg></svg></a></p>')
 assert len(links)==1
 assert rows==[]


@pytest.mark.parametrize('attrs,text',[
 ('aria-label="Underlying data"',''),
 ('title="open" aria-label="Underlying data"','here'),
 ('title="Unknown representation" type="text/csv"',''),
 ('aria-label="Download observations"',''),
 ('title="here" aria-label="here"','here'),
 ('title="External file" download',''),
])
def test_parsed_data_and_explicit_attachment_routes_remain(attrs,text):
 links,rows=candidates('<p><a href="https://repo.example/record/123" '+attrs+'>'+text+'<svg></svg></a></p>')
 assert len(rows)==1
 assert rows[0]['provenance']==links[0]
 assert rows[0]['provenance']['locator']['anchor_index']==0


def test_sibling_data_link_does_not_supply_share_icon_purpose():
 links,rows=candidates('<p>The data are available <a href="https://repo.example/data" aria-label="Underlying data">here</a>. '
   '<a href="https://network.example/publish?title=Dengue%20Brazil%202025" title="Publish this post to Network"><svg></svg></a></p>')
 assert len(links)==2
 assert [r['url'] for r in rows]==['https://repo.example/data']


def test_aria_foreign_country_data_does_not_inherit_parent_task_country():
 _,rows=candidates('<p><a href="https://repo.example/record/123" aria-label="Dengue in Peru data"><svg></svg></a></p>')
 assert rows==[]


@pytest.mark.parametrize('attrs,text',[
 ('title="External link"','here'),
 ('aria-label="External link"',''),
 ('title="Opens in a new tab"','here'),
 ('aria-label="Opens in a new tab"',''),
 ('title="Unknown destination purpose"','here'),
 ('aria-label="Unknown destination purpose"','here'),
 ('title="here" aria-label="Unknown destination purpose"',''),
 ('title="Unknown destination purpose" aria-label="here"',''),
])
def test_neutral_accessibility_description_is_not_unrelated_destination(attrs,text):
 _,rows=candidates('<p><a href="https://repo.example/record/123" '+attrs+'>'+text+'<svg></svg></a></p>')
 assert len(rows)==1
