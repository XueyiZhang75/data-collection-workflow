"""Independent navigation admission must survive duplicate menu anchors."""
import pytest
from bs4 import BeautifulSoup
from data_collection_workflow.resource_discovery import html_resource_links, task_resource_candidates


@pytest.mark.parametrize('later_navigation',[False,True])
def test_rejected_generic_menu_anchor_does_not_hide_later_task_product(later_navigation):
    parent='https://health.example/report'
    tag='nav' if later_navigation else 'p'
    soup=BeautifulSoup('<nav><a href="/reports">Reports</a></nav>'
        '<h1>Example fever surveillance</h1>'
        f'<{tag}><a href="/reports">Example fever report archive</a></{tag}>','html.parser')
    doc={'source_id':'s','url':parent,'content_readable':True,
        'clean_text':soup.get_text(' ',strip=True),
        'metadata':{'outbound_links':html_resource_links(soup,source_url=parent,content_hash='current')}}
    state={'structured_task':{'disease':'Example fever','location':'Example Region',
        'start_date':'2025-01-01','end_date':'2025-12-31'}}
    rows=task_resource_candidates(doc,{},state)
    assert len(rows)==1
    assert rows[0]['url']=='https://health.example/reports'
    assert rows[0]['link_text']=='Example fever report archive'
