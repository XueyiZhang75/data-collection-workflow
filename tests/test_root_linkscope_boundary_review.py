from bs4 import BeautifulSoup
import pytest
from data_collection_workflow.resource_discovery import html_resource_links, task_resource_candidates


def select(html):
    url='https://journal.example/article/123'
    soup=BeautifulSoup(html,'html.parser')
    doc={'source_id':'parent','url':url,'content_readable':True,'clean_text':soup.get_text(' ',strip=True),'metadata':{'outbound_links':html_resource_links(soup,source_url=url,content_hash='test')}}
    state={'structured_task':{'disease':'dengue','location':'Brazil','start_date':'2025-01-01','end_date':'2025-12-31'}}
    return task_resource_candidates(doc,{},state)

@pytest.mark.parametrize('utility',[
 '<nav><a href="/policies/data">Data Availability</a></nav>',
 '<footer><a href="/publication-policies/data">Data access policy</a></footer>',
])
def test_article_data_heading_does_not_authorize_site_policy_navigation(utility):
    rows=select('<article><h1>Dengue Brazil 2025</h1><h2>Data availability</h2>'+utility+'</article>')
    assert rows==[]

@pytest.mark.parametrize('binding',[
 '<meta name="citation_pdf_url" content="https://journal.example/article/123.pdf">',
 '<link rel="alternate" type="application/pdf" href="https://journal.example/article/123.pdf">',
])
def test_declared_article_representation_is_preserved_outside_article(binding):
    html='<html><head><title>Dengue Brazil 2025</title>'+binding+'</head><body><nav><a href="/article/123.pdf">PDF</a></nav><article><h1>Dengue Brazil 2025</h1></article></body></html>'
    assert [x['url'] for x in select(html)]==['https://journal.example/article/123.pdf']


def test_declared_representation_does_not_authorize_neighboring_navigation_pdf():
    html='<head><title>Dengue Brazil 2025</title><meta name="citation_pdf_url" content="https://journal.example/article/123.pdf"></head><nav><a href="/agency-law.pdf">Agency law</a></nav><article><h1>Dengue Brazil 2025</h1></article>'
    assert [x['url'] for x in select(html)]==['https://journal.example/article/123.pdf']


def test_undeclared_neighboring_navigation_pdf_is_not_discovered():
    html='<head><title>Dengue Brazil 2025</title></head><nav><a href="/agency-law.pdf">Agency law</a></nav><article><h1>Dengue Brazil 2025</h1></article>'
    assert select(html)==[]


def test_task_named_reference_is_still_discoverable():
    html='<article><h1>Dengue Brazil 2025</h1><h2>References</h2><p><a href="https://other.example/article/77">Dengue surveillance Brazil 2025</a></p></article>'
    assert len(select(html))==1
